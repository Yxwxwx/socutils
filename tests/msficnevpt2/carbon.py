# SPDX-License-Identifier: GPL-3.0-or-later
"""C ANO-RCC VTZP / X2CAMF / SA15 DMRG, five identified 1D2 roots.

Sequential benchmark. Restarts use the isolated, completed reference and
contracted matrices; the formal densities always come from DMRG, not FCI.
"""

import itertools
import json
import pickle
import resource
import time

import numpy as np
from pyscf.fci import cistring, fci_dhf_slow
from socutils.mrpt import nevpt2_utils as u
from socutils.mrpt import prepare_msfic, solve_msfic

from .carbon_reference import DIRECTORY, canonicalize_npdm_reference, reference


def one_body(matrix, strings):
    positions = {int(b): i for i, b in enumerate(strings)}
    result = np.zeros((len(strings), len(strings)), complex)
    for column, bits in enumerate(strings):
        for p, q in itertools.product(range(len(matrix)), repeat=2):
            state = int(bits)
            if not (state >> q) & 1:
                continue
            sign = (-1) ** ((state & ((1 << q) - 1)).bit_count())
            state ^= 1 << q
            if (state >> p) & 1:
                continue
            sign *= (-1) ** ((state & ((1 << p) - 1)).bit_count())
            state ^= 1 << p
            result[positions[state], column] += sign * matrix[p, q]
    return result


def annihilated_coefficients(vector, ncas, nelec, rank):
    """Independent determinant operator actions; never used as PT densities."""
    incoming = cistring.make_strings(range(ncas), nelec)
    outgoing = cistring.make_strings(range(ncas), nelec - rank)
    positions = {int(b): i for i, b in enumerate(outgoing)}
    tuples = list(itertools.product(range(ncas), repeat=rank))
    result = np.zeros((len(tuples), len(outgoing)), complex)
    for row, indices in enumerate(tuples):
        for amplitude, bits in zip(vector, incoming):
            state, sign = int(bits), 1
            for p in reversed(indices):
                if not (state >> p) & 1:
                    break
                sign *= (-1) ** ((state & ((1 << p) - 1)).bit_count())
                state ^= 1 << p
            else:
                result[row, positions[state]] += sign * amplitude
    return result


def density_from_annihilated(bra, ket, ncas, rank):
    reverse = (
        np.arange(ncas**rank)
        .reshape((ncas,) * rank)
        .transpose(tuple(reversed(range(rank))))
        .reshape(-1)
    )
    return (bra[reverse].conj() @ ket.T).reshape((ncas,) * (2 * rank))


def audit_reference(mc, solver):
    """Actual MPS coefficients vs independent 70-D Hamiltonian and J/L/S."""
    strings = cistring.make_strings(range(8), 4)
    dets = np.array(
        [[(int(b) >> p) & 1 for p in range(8)] for b in strings], dtype=np.uint8
    )
    vectors = []
    for root in range(15):
        driver, ket = u._root_ket(solver, root)
        assert driver.reorder_idx is None or np.array_equal(
            driver.reorder_idx, np.arange(8)
        )
        working = driver.copy_mps(ket, tag=f"C_MS_AUDIT_{root}")
        driver.align_mps_center(working, ref=0)
        vectors.append(
            np.asarray(
                driver.get_csf_coefficients(
                    working,
                    cutoff=0.0,
                    given_dets=dets,
                    max_print=0,
                    fci_conv=True,
                    iprint=0,
                )[1]
            )
        )
    vectors = np.column_stack(vectors)
    snap = solver.checkpoint_hamiltonian
    h, g = snap["h1e"], snap["eri"]
    absorbed = fci_dhf_slow.absorb_h1e(h, g, 8, 4, 0.5)
    ha = np.column_stack(
        [fci_dhf_slow.contract_2e(absorbed, v, 8, 4) for v in np.eye(70, dtype=complex)]
    )
    np.testing.assert_allclose(ha, ha.conj().T, atol=1e-11, rtol=0)
    exact_e, exact_v = np.linalg.eigh(ha)
    active_e = np.asarray(mc.e_states) - snap["ecore"]
    residuals = np.linalg.norm(ha @ vectors - vectors * active_e, axis=0)
    gram = np.linalg.norm(vectors.conj().T @ vectors - np.eye(15))
    overlaps = np.linalg.svd(exact_v[:, :15].conj().T @ vectors, compute_uv=False)
    # Physical atomic rotations: orbital angular momentum plus spin,
    # transformed through the *actual* AO -> final active MO coefficients.
    mol = mc.mol
    ca, cb = mol.sph2spinor_coeff()
    overlap = mol.intor("int1e_ovlp_sph")
    orbital = -1j * mol.intor("int1e_cg_irxp_sph", comp=3)
    spin = np.array(
        [
            0.5 * (ca.conj().T @ overlap @ cb + cb.conj().T @ overlap @ ca),
            0.5j * (cb.conj().T @ overlap @ ca - ca.conj().T @ overlap @ cb),
            0.5 * (ca.conj().T @ overlap @ ca - cb.conj().T @ overlap @ cb),
        ]
    )
    orbital = np.array([ca.conj().T @ l @ ca + cb.conj().T @ l @ cb for l in orbital])
    mo = mc.mo_coeff[:, 2:10]
    observables = {}
    for name, ao in (("J2", orbital + spin), ("L2", orbital), ("S2", spin)):
        operators = [one_body(mo.conj().T @ a @ mo, strings) for a in ao]
        observables[name] = sum(
            np.sum(np.abs(op @ vectors) ** 2, axis=0) for op in operators
        )
    roots = tuple(
        int(r)
        for r in range(15)
        if abs(observables["J2"][r] - 6.0) < 1e-5
        and observables["S2"][r] < 0.1
        and observables["L2"][r] > 5.9
    )
    if len(roots) != 5:
        raise AssertionError(
            f"J/L/S identification did not find 1D2 quintet: {observables}"
        )
    target_overlap = np.linalg.svd(
        exact_v[:, roots].conj().T @ vectors[:, roots], compute_uv=False
    )
    print(
        "REFERENCE AUDIT",
        "Gram",
        gram,
        "residuals",
        residuals,
        "energies vs exact",
        active_e - exact_e[:15],
        "JLS",
        observables,
        "subspace singular values",
        overlaps,
        flush=True,
    )
    assert gram < 1e-9 and np.max(residuals) < 1e-9
    assert min(overlaps) > 1 - 1e-10 and min(target_overlap) > 1 - 1e-10
    return (
        roots,
        vectors,
        {
            "roots": roots,
            "all_energies": mc.e_states,
            "residuals": residuals,
            "gram_defect": gram,
            "sa_subspace_singular_values": overlaps,
            "model_subspace_singular_values": target_overlap,
            "angular_momenta": observables,
            "exact_active_energies": exact_e,
            "reference_spread": float(np.ptp(np.asarray(mc.e_states)[list(roots)])),
        },
    )


def mps_bonds(solver):
    dimensions = []
    for ket in solver.kets:
        ket.info.load_mutable()
        dimensions.append(int(ket.info.get_max_bond_dimension()))
        ket.info.deallocate_mutable()
    return dimensions


def json_value(value):
    if isinstance(value, complex):
        return [value.real, value.imag]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value))


def run(
    directory=DIRECTORY, bond_dim=64, *, npdm_cutoff=1e-24, reference_directory=None
):
    start = time.perf_counter()
    directory.mkdir(parents=True, exist_ok=True)
    cache = directory / "prepared.pkl"
    if cache.exists():
        with cache.open("rb") as handle:
            prepared, reference_audit = pickle.load(handle)
    else:
        mc, solver = reference(
            directory=directory if reference_directory is None else reference_directory,
            bond_dim=bond_dim,
        )
        solver.npdm_cutoff = npdm_cutoff
        try:
            roots, vectors, reference_audit = audit_reference(mc, solver)
            print("IDENTIFIED MODEL ROOTS", roots, reference_audit, flush=True)
            reference_audit["requested_bond_dimension"] = max(solver.bond_dims)
            reference_audit["original_actual_bond_dimensions"] = mps_bonds(solver)
            canonicalize_npdm_reference(mc)
            qr_roots, qr_vectors, qr_audit = audit_reference(mc, solver)
            assert qr_roots == roots
            qr_defect = float(np.max(np.abs(vectors - qr_vectors)))
            np.testing.assert_allclose(qr_vectors, vectors, atol=1e-12, rtol=0)
            reference_audit["npdm_qr_state_and_phase_defect"] = qr_defect
            reference_audit["npdm_qr_reference_residuals"] = qr_audit["residuals"]
            reference_audit["npdm_cutoff"] = npdm_cutoff
            reference_audit["qr_actual_bond_dimensions"] = mps_bonds(solver)
            print("QR STATE/PHASE PRESERVATION", qr_defect, flush=True)
            annihilated = {
                (root, rank): annihilated_coefficients(qr_vectors[:, root], 8, 4, rank)
                for root in roots
                for rank in range(1, 5)
            }
            oracle_audit = {}

            def native_densities(bra, ket):
                pdms = u.make_transition_dm1234(solver, bra, ket)
                errors = []
                for rank, dm in enumerate(pdms, 1):
                    expected = density_from_annihilated(
                        annihilated[bra, rank], annihilated[ket, rank], 8, rank
                    )
                    error = u._maximum_abs_relation(
                        dm, expected, sign=-1.0, work_memory=64 * 2**20
                    )
                    if error > 1e-11:
                        raise AssertionError(
                            f"native NPDM oracle {bra},{ket}/{rank}: {error}"
                        )
                    errors.append(error)
                    del expected
                oracle_audit[f"{bra},{ket}"] = errors
                print("NATIVE TRANSITION RDM ORACLE", bra, ket, errors, flush=True)
                return pdms

            prepared = prepare_msfic(
                mc,
                sa_roots=tuple(range(15)),
                sa_weights=np.ones(15) / 15,
                model_roots=roots,
                transition_pdms=native_densities,
            )
            reference_audit["transition_rdm_oracle"] = oracle_audit
            assert len(oracle_audit) == 25
            with cache.open("wb") as handle:
                pickle.dump((prepared, reference_audit), handle)
        finally:
            solver.close()
    results = {}
    for shift in (0.2, 0.0):
        for ansatz in ("ss_sr", "ms_mr"):
            result = solve_msfic(prepared, ansatz=ansatz, shift=shift)
            results[f"{ansatz}_eta_{shift}"] = result.__dict__
            print(
                ansatz,
                "eta",
                shift,
                "energies",
                repr(result.energies),
                "spread / cm-1",
                result.spread * 219474.63137,
                flush=True,
            )
    data = {
        "reference_audit": reference_audit,
        "preparation": {str(k): v for k, v in prepared.diagnostics.items()},
        "results": results,
        "wall_seconds": time.perf_counter() - start,
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
    }
    with (directory / "results.json").open("w") as handle:
        json.dump(data, handle, default=json_value, indent=2)
    return data


if __name__ == "__main__":
    run()
