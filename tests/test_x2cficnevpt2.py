# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent Fock-space checks of complex-spinor FIC-NEVPT2."""

import re
import sys
from types import SimpleNamespace

import numpy as np
from pyscf import ao2mo, gto, mcscf, scf
from pyblock2.icmr.icnevpt2_full import WickICNEVPT2

from socutils.mrpt import spinor_helper, x2cficnevpt2

from test_x2cscnevpt2_wick import (
    _active_ground_state,
    _apply_string,
    _embed_active_state,
    _hamiltonian_action,
    _random_physical_integrals,
    _raw_active_rdms,
    _spatial_to_spinor_integrals,
)


def _fic_basis_vectors(reference, eris, key, free_indices):
    """Build the implementation's IC basis directly in Fock space."""

    orbital_map = {
        label: (
            int(index) if label in "ij" else eris.nocc + int(index)
        )
        for label, index in zip(key, free_indices, strict=True)
    }
    columns = []
    for component in x2cficnevpt2._IC_COMPONENTS[key]:
        _selection, active_tuples = x2cficnevpt2._active_selection(
            eris.ncas,
            len(component.ket_active),
            component.active_pairs,
        )
        operators = tuple(
            re.findall(r"([CD])\[([a-z])\]", component.expression)
        )
        for active_tuple in active_tuples:
            component_map = dict(orbital_map)
            component_map.update(
                {
                    label: eris.ncore + int(index)
                    for label, index in zip(
                        component.ket_active, active_tuple, strict=True
                    )
                }
            )
            columns.append(
                _apply_string(
                    reference,
                    tuple(
                        (kind, component_map[label])
                        for kind, label in operators
                    ),
                )
            )
    return np.asarray(columns).T


def _active_hamiltonian_matrix(h1e, w):
    dimension = 1 << h1e.shape[0]
    result = np.empty((dimension, dimension), dtype=complex)
    for column in range(dimension):
        vector = np.zeros(dimension, dtype=complex)
        vector[column] = 1.0
        result[:, column] = _hamiltonian_action(vector, h1e, w)
    return result


def _apply_active_hamiltonian(vectors, matrix, ncore, ncas, nvirt):
    """Apply an active-only Hamiltonian while preserving spectators."""

    vectors = np.asarray(vectors)
    was_vector = vectors.ndim == 1
    if was_vector:
        vectors = vectors[:, None]
    result = np.zeros_like(vectors)
    active_bits = np.arange(1 << ncas) << ncore
    for core_bits in range(1 << ncore):
        for virtual_bits in range(1 << nvirt):
            indices = (
                core_bits
                | active_bits
                | (virtual_bits << (ncore + ncas))
            )
            result[indices] = matrix @ vectors[indices]
    return result[:, 0] if was_vector else result


def _fic_fixture(seed=849123, *, ncas=6, nelec=4):
    ncore, nvirt = 2, 2
    h1e, eri = _random_physical_integrals(
        ncore + ncas + nvirt, seed=seed
    )
    eris = spinor_helper.init_eris(h1e, eri, ncore, ncas)
    active = slice(eris.ncore, eris.nocc)
    w = eris.get_phys("PPPP")
    active_h = eris.h1eff[active, active]
    active_w = w[active, active, active, active]
    active_energy, active_state = _active_ground_state(
        active_h, active_w, nelec
    )
    reference = _embed_active_state(active_state, ncore, ncas, nvirt)
    return {
        "ncore": ncore,
        "ncas": ncas,
        "nvirt": nvirt,
        "h1e": h1e,
        "w": w,
        "eris": eris,
        "active_h": active_h,
        "active_w": active_w,
        "active_energy": active_energy,
        "active_state": active_state,
        "reference": reference,
        "h_reference": _hamiltonian_action(reference, h1e, w),
        "pdms": _raw_active_rdms(active_state, ncas),
    }


def test_fic_wick_matrices_and_energies_match_direct_complex_fock_space():
    """Audit every matrix element, all eight spaces, and the final solve."""

    data = _fic_fixture()
    assert np.max(np.abs(data["pdms"][3])) > 1.0e-8
    eris = data["eris"]
    core_energy = np.array([-10.0, -8.0])
    virtual_energy = np.array([7.0, 9.0])
    sub_eners, diagnostics = x2cficnevpt2._evaluate_fic_subspaces(
        eris,
        data["pdms"],
        core_energy,
        virtual_energy,
        contraction_backend="numpy",
        matrix_atol=3.0e-11,
        matrix_rtol=3.0e-11,
        return_diagnostics=True,
    )

    active_matrix = _active_hamiltonian_matrix(
        data["active_h"], data["active_w"]
    )
    equations = x2cficnevpt2._compile_fic_equations()
    context = x2cficnevpt2._utils._execution_context(eris, data["pdms"])
    context.update(
        ident1=np.ones((1,), dtype=complex),
        ident2=np.ones((1, 1), dtype=complex),
        ident3=np.ones((1, 1, 1), dtype=complex),
    )

    for key in x2cficnevpt2.SUBSPACE_ORDER:
        components = x2cficnevpt2._IC_COMPONENTS[key]
        free_labels = tuple(key)
        rhs_tensors = {}
        matrix_tensors = {name: {} for name in ("metric", "right", "left")}
        matrix_sources = {
            "metric": equations.metric_code,
            "right": equations.right_code,
            "left": equations.left_code,
        }
        for bra_component in components:
            rhs_tensors[bra_component.name] = x2cficnevpt2._execute_tensor(
                equations.rhs_code[(key, bra_component.name)],
                "rhs",
                free_labels + bra_component.bra_active,
                context,
                {"np": np},
                np.complex128,
                eris,
            )
            for ket_component in components:
                pair = (key, bra_component.name, ket_component.name)
                short_pair = (bra_component.name, ket_component.name)
                labels = (
                    free_labels
                    + bra_component.bra_active
                    + ket_component.ket_active
                )
                for target, source in matrix_sources.items():
                    matrix_tensors[target][short_pair] = (
                        x2cficnevpt2._execute_tensor(
                            source[pair],
                            target,
                            labels,
                            context,
                            {"np": np},
                            np.complex128,
                            eris,
                        )
                    )

        direct_subspace_energy = 0.0
        direct_ranks = []
        for free_indices in x2cficnevpt2._iter_free_tuples(key, eris):
            vectors = _fic_basis_vectors(
                data["reference"], eris, key, free_indices
            )
            h_vectors = _apply_active_hamiltonian(
                vectors,
                active_matrix,
                data["ncore"],
                data["ncas"],
                data["nvirt"],
            )
            exact_rhs = vectors.conj().T @ data["h_reference"]
            exact_metric = vectors.conj().T @ vectors
            exact_active = (
                vectors.conj().T @ h_vectors
                - data["active_energy"] * exact_metric
            )
            actual_rhs = x2cficnevpt2._assemble_vector(
                rhs_tensors, free_indices, components, data["ncas"]
            )
            np.testing.assert_allclose(
                actual_rhs, exact_rhs, atol=8.0e-13, rtol=8.0e-13
            )
            for target, exact in (
                ("metric", exact_metric),
                ("right", exact_active),
                ("left", exact_active),
            ):
                actual = x2cficnevpt2._assemble_matrix(
                    matrix_tensors[target],
                    free_indices,
                    components,
                    data["ncas"],
                )
                np.testing.assert_allclose(
                    actual, exact, atol=8.0e-13, rtol=8.0e-13
                )

            q, singular_values, _vh = np.linalg.svd(
                vectors, full_matrices=False
            )
            retained = singular_values > 1.0e-10
            q = q[:, retained]
            direct_ranks.append(q.shape[1])
            hq = _apply_active_hamiltonian(
                q,
                active_matrix,
                data["ncore"],
                data["ncas"],
                data["nvirt"],
            )
            orbital_gap = x2cficnevpt2._orbital_gap_at(
                key, free_indices, core_energy, virtual_energy
            )
            dyall = q.conj().T @ hq
            dyall += (
                orbital_gap - data["active_energy"]
            ) * np.eye(q.shape[1])
            source = q.conj().T @ data["h_reference"]
            direct_subspace_energy -= np.vdot(
                source, np.linalg.solve(dyall, source)
            ).real

        np.testing.assert_allclose(
            sub_eners[key], direct_subspace_energy, atol=8.0e-13, rtol=8.0e-13
        )
        assert diagnostics[key]["minimum_metric_rank"] == min(direct_ranks)
        assert diagnostics[key]["maximum_metric_rank"] == max(direct_ranks)


def test_fic_numpy_and_pytblis_backends_agree():
    data = _fic_fixture(seed=7811, ncas=4, nelec=2)
    arguments = (
        data["eris"],
        data["pdms"],
        np.array([-9.0, -7.0]),
        np.array([6.0, 8.0]),
    )
    numpy_result = x2cficnevpt2._evaluate_fic_subspaces(
        *arguments, contraction_backend="numpy"
    )
    tblis_result = x2cficnevpt2._evaluate_fic_subspaces(
        *arguments, contraction_backend="pytblis"
    )
    np.testing.assert_allclose(
        [numpy_result[key] for key in x2cficnevpt2.SUBSPACE_ORDER],
        [tblis_result[key] for key in x2cficnevpt2.SUBSPACE_ORDER],
        atol=2.0e-13,
        rtol=2.0e-13,
    )


def test_public_fic_kernel_commits_a_complete_semicanonical_result():
    data = _fic_fixture(seed=4471, ncas=2, nelec=1)
    nmo = data["eris"].nmo
    molecule = SimpleNamespace(verbose=0, stdout=sys.stdout)
    solver = SimpleNamespace(nelecas=1, nroots=1)
    mc = SimpleNamespace(
        _scf=SimpleNamespace(mol=molecule),
        verbose=0,
        stdout=sys.stdout,
        mo_coeff=np.eye(nmo, dtype=complex),
        mo_energy=np.array([-9.0, -7.0, -1.0, 1.0, 6.0, 8.0]),
        e_tot=-3.0,
        ncore=2,
        ncas=2,
        nelecas=1,
        fcisolver=solver,
        frozen=0,
    )
    pt = x2cficnevpt2.WickX2CFICNEVPT2(mc)
    pt.canonicalized = True
    correction = pt.kernel(
        pdms=data["pdms"],
        eris=data["eris"],
        eris_basis="semicanonical",
        contraction_backend="numpy",
    )

    assert correction == pt.e_corr
    assert set(pt.sub_eners) == set(x2cficnevpt2.SUBSPACE_ORDER)
    assert set(pt.sub_diagnostics) == set(x2cficnevpt2.SUBSPACE_ORDER)
    assert pt.eris_basis == "semicanonical"
    assert pt.contraction_backend == "numpy"
    np.testing.assert_allclose(pt.e_tot, mc.e_tot + correction, atol=1.0e-14)


def test_spinor_fic_matches_independent_pyblock2_spatial_implementation():
    """A real alpha/beta embedding agrees with Block2 IC class by class."""

    molecule = gto.M(
        atom="Li 0 0 0; H 0 0 1.6",
        basis="sto-3g",
        spin=0,
        verbose=0,
    )
    mean_field = scf.RHF(molecule).run(conv_tol=1.0e-13)
    mc = mcscf.CASCI(mean_field, 2, 2)
    mc.fcisolver.conv_tol = 1.0e-14
    mc.kernel()
    assert mc.converged

    reference = WickICNEVPT2(mc)
    reference.canonicalized = True
    reference.kernel()

    mo = mc.mo_coeff
    nmo = mo.shape[1]
    spatial_h1e = mo.T @ mean_field.get_hcore() @ mo
    spatial_eri = ao2mo.restore(1, ao2mo.kernel(molecule, mo), nmo)
    h1e, eri = _spatial_to_spinor_integrals(spatial_h1e, spatial_eri)
    eris = spinor_helper.init_eris(
        h1e,
        eri,
        ncore=2 * mc.ncore,
        ncas=2 * mc.ncas,
    )
    active = slice(eris.ncore, eris.nocc)
    w = eris.get_phys("PPPP")
    _energy, active_state = _active_ground_state(
        eris.h1eff[active, active],
        w[active, active, active, active],
        nelec=2,
    )
    pdms = _raw_active_rdms(active_state, eris.ncas)
    actual = x2cficnevpt2._evaluate_fic_subspaces(
        eris,
        pdms,
        np.repeat(mc.mo_energy[: mc.ncore], 2),
        np.repeat(mc.mo_energy[mc.ncore + mc.ncas :], 2),
        contraction_backend="numpy",
    )

    np.testing.assert_allclose(
        sum(actual.values()), reference.e_corr, atol=2.0e-11, rtol=2.0e-11
    )
    for key in x2cficnevpt2.SUBSPACE_ORDER:
        np.testing.assert_allclose(
            actual[key], reference.sub_eners[key], atol=2.0e-11, rtol=2.0e-11
        )
