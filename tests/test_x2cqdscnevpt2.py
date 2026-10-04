"""Independent fixed-N Fock-space checks for QD-SC-NEVPT2."""

import itertools
import json
from types import SimpleNamespace

import numpy as np
import pytest

import socutils.mrpt as mrpt_api
from socutils.mrpt import spinor_helper
from socutils.mrpt import x2cqdscnevpt2 as qdmod
from socutils.mrpt import x2cscnevpt2 as ssmod
from socutils.mcscf import zmcscf


def _apply_operator(vector, kind, orbital):
    result = np.zeros_like(vector)
    lower_mask = (1 << orbital) - 1
    orbital_mask = 1 << orbital
    for state, amplitude in enumerate(vector):
        if amplitude == 0:
            continue
        occupied = bool(state & orbital_mask)
        if (kind == "D" and not occupied) or (kind == "C" and occupied):
            continue
        if kind not in ("C", "D"):
            raise ValueError(kind)
        sign = -1 if (state & lower_mask).bit_count() % 2 else 1
        result[state ^ orbital_mask] += sign * amplitude
    return result


def _apply_string(vector, operators):
    result = vector
    for kind, orbital in reversed(operators):
        result = _apply_operator(result, kind, orbital)
    return result


def _hamiltonian_action(vector, h1e, w):
    result = np.zeros_like(vector)
    nmo = h1e.shape[0]
    for p, q in itertools.product(range(nmo), repeat=2):
        coefficient = h1e[p, q]
        if coefficient:
            result += coefficient * _apply_string(
                vector, (("C", p), ("D", q))
            )
    for p, q, r, s in itertools.product(range(nmo), repeat=4):
        coefficient = 0.5 * w[p, q, r, s]
        if coefficient:
            result += coefficient * _apply_string(
                vector,
                (("C", p), ("C", q), ("D", s), ("D", r)),
            )
    return result


def _fixed_particle_basis(norb, nelec):
    return [state for state in range(1 << norb) if state.bit_count() == nelec]


def _active_eigenstates(h1e, w, nelec):
    norb = h1e.shape[0]
    basis = _fixed_particle_basis(norb, nelec)
    matrix = np.empty((len(basis), len(basis)), dtype=complex)
    for column, state in enumerate(basis):
        ket = np.zeros(1 << norb, dtype=complex)
        ket[state] = 1.0
        matrix[:, column] = _hamiltonian_action(ket, h1e, w)[basis]
    np.testing.assert_allclose(matrix, matrix.T.conj(), atol=2.0e-12)
    energies, vectors = np.linalg.eigh(matrix)
    states = []
    for column in range(vectors.shape[1]):
        state = np.zeros(1 << norb, dtype=complex)
        state[basis] = vectors[:, column]
        states.append(state)
    return energies, tuple(states)


def _embed_active_state(active_state, ncore, ncas, nvirt):
    result = np.zeros(1 << (ncore + ncas + nvirt), dtype=complex)
    core_bits = (1 << ncore) - 1
    for active_bits, amplitude in enumerate(active_state):
        result[core_bits | (active_bits << ncore)] = amplitude
    return result


def _raw_transition_rdm(bra, ket, ncas, rank):
    tuples = tuple(itertools.product(range(ncas), repeat=rank))
    bra_annihilated = np.asarray(
        [
            _apply_string(
                bra, tuple(("D", orbital) for orbital in indices)
            )
            for indices in tuples
        ]
    )
    ket_annihilated = np.asarray(
        [
            _apply_string(
                ket, tuple(("D", orbital) for orbital in indices)
            )
            for indices in tuples
        ]
    )
    rows = np.arange(ncas**rank).reshape((ncas,) * rank)
    reversed_rows = rows.transpose(tuple(reversed(range(rank)))).reshape(-1)
    density = np.einsum(
        "xi,yi->xy",
        bra_annihilated[reversed_rows].conj(),
        ket_annihilated,
        optimize=True,
    )
    return density.reshape((ncas,) * (2 * rank))


def _raw_transition_rdms(bra, ket, ncas):
    return tuple(
        _raw_transition_rdm(bra, ket, ncas, rank) for rank in range(1, 5)
    )


def _random_physical_integrals(nmo, seed=7319):
    rng = np.random.default_rng(seed)
    trial = rng.normal(size=(nmo, nmo)) + 1j * rng.normal(size=(nmo, nmo))
    h1e = 0.025 * (trial + trial.T.conj())
    factors = rng.normal(size=(5, nmo, nmo))
    factors = factors + 1j * rng.normal(size=factors.shape)
    factors = 0.018 * (factors + factors.swapaxes(1, 2).conj())
    eri = np.einsum("Ppq,Prs->pqrs", factors, factors, optimize=True)
    spinor_helper.check_eri_symmetry(eri, atol=2.0e-13, rtol=2.0e-13)
    return h1e, eri


def _state_specific_generalized_fock(eris, active_dm1):
    """Independent spinor generalized Fock in the input MO basis."""

    gamma = np.zeros((eris.nmo, eris.nmo), dtype=complex)
    gamma[np.arange(eris.ncore), np.arange(eris.ncore)] = 1.0
    gamma[eris.ncore : eris.nocc, eris.ncore : eris.nocc] = active_dm1
    fock = np.array(eris.h1e, copy=True)
    fock += np.einsum("pqrs,rs->pq", eris.pppp, gamma, optimize=True)
    fock -= np.einsum("psrq,rs->pq", eris.pppp, gamma, optimize=True)
    np.testing.assert_allclose(fock, fock.T.conj(), atol=2.0e-12, rtol=0.0)
    return 0.5 * (fock + fock.T.conj())


def _independent_canonstep(fock, ncore, ncas):
    nmo = fock.shape[0]
    nocc = ncore + ncas
    rotation = np.eye(nmo, dtype=complex)
    energies = np.diag(fock).real.copy()
    for space in (slice(0, ncore), slice(nocc, nmo)):
        values, vectors = np.linalg.eigh(fock[space, space])
        rotation[space, space] = vectors
        energies[space] = values
    return rotation, energies


def _add_term(target, coefficient, reference, operators):
    if coefficient:
        target += coefficient * _apply_string(reference, operators)


_PAIR_RESTRICTIONS = {
    "ijrs": ((0, 1), (2, 3)),
    "rsi": ((0, 1),),
    "ijr": ((0, 1),),
    "rs": ((0, 1),),
    "ij": ((0, 1),),
    "ir": (),
    "r": (),
    "i": (),
}


def _direct_free_shape(key, ncore, nvirt):
    return tuple(ncore if char in "ij" else nvirt for char in key)


def _direct_pair_mask(key, shape):
    mask = np.ones(shape, dtype=bool)
    grid = np.indices(shape, sparse=True)
    for left, right in _PAIR_RESTRICTIONS[key]:
        mask &= grid[left] < grid[right]
    return mask


def _direct_orbital_gap(key, core_energy, virtual_energy):
    shape = tuple(
        len(core_energy) if char in "ij" else len(virtual_energy)
        for char in key
    )
    result = np.zeros(shape)
    for axis, char in enumerate(key):
        values = -np.asarray(core_energy) if char in "ij" else np.asarray(
            virtual_energy
        )
        reshape = [1] * len(key)
        reshape[axis] = len(values)
        result += values.reshape(reshape)
    return result


def _occupation_projected_h_reference(reference, eris):
    """Independently form every P_alpha H|Psi> from the full Hamiltonian."""

    h_reference = _hamiltonian_action(
        reference, eris.h1e, eris.get_phys("PPPP")
    )
    ncore, nocc = eris.ncore, eris.nocc
    projected = {
        key: np.zeros(
            _direct_free_shape(key, eris.ncore, eris.nvirt)
            + (reference.size,),
            dtype=complex,
        )
        for key in ssmod.SUBSPACE_ORDER
    }
    for state, amplitude in enumerate(h_reference):
        if amplitude == 0:
            continue
        holes = tuple(i for i in range(ncore) if not state & (1 << i))
        particles = tuple(
            r - nocc for r in range(nocc, eris.nmo) if state & (1 << r)
        )
        signature = (len(holes), len(particles))
        if signature == (2, 2):
            key, free = "ijrs", holes + particles
        elif signature == (1, 2):
            key, free = "rsi", particles + holes
        elif signature == (2, 1):
            key, free = "ijr", holes + particles
        elif signature == (0, 2):
            key, free = "rs", particles
        elif signature == (2, 0):
            key, free = "ij", holes
        elif signature == (1, 1):
            key, free = "ir", holes + particles
        elif signature == (0, 1):
            key, free = "r", particles
        elif signature == (1, 0):
            key, free = "i", holes
        else:
            continue
        projected[key][free + (state,)] = amplitude
    return projected


def _active_dyall_action(vector, eris):
    ncore, nocc = eris.ncore, eris.nocc
    h = eris.h1eff
    w = eris.pppp.transpose(0, 2, 1, 3)
    result = np.zeros_like(vector)
    for a, b in itertools.product(range(ncore, nocc), repeat=2):
        _add_term(result, h[a, b], vector, (("C", a), ("D", b)))
    for a, b, c, d in itertools.product(range(ncore, nocc), repeat=4):
        _add_term(
            result,
            0.5 * w[a, b, c, d],
            vector,
            (("C", a), ("C", b), ("D", d), ("D", c)),
        )
    return result


def _direct_row_data(reference, eris, active_energy, core_energy, virtual_energy):
    perturbers = _occupation_projected_h_reference(reference, eris)
    norms = {}
    denominators = {}
    masks = {}
    for key, vectors in perturbers.items():
        norm = np.empty(vectors.shape[:-1], dtype=complex)
        commutator = np.empty_like(norm)
        for free in np.ndindex(vectors.shape[:-1]):
            vector = vectors[free]
            norm[free] = np.vdot(vector, vector)
            commutator[free] = (
                np.vdot(vector, _active_dyall_action(vector, eris))
                - active_energy * norm[free]
            )
        ordered = _direct_pair_mask(key, norm.shape)
        nonzero = ordered & (norm.real > 1.0e-14)
        denominator = _direct_orbital_gap(
            key, core_energy, virtual_energy
        ).astype(complex)
        denominator[nonzero] += commutator[nonzero] / norm[nonzero]
        norms[key] = norm
        denominators[key] = denominator
        masks[key] = nonzero
    return perturbers, norms, denominators, masks


def _cross_perturber_overlaps(left, right):
    result = {}
    for key in ssmod.SUBSPACE_ORDER:
        result[key] = np.einsum(
            "...x,...x->...", left[key].conj(), right[key], optimize=True
        )
    return result


def _make_fake_mc(ncore, ncas, nelec, reference_energies):
    mol = SimpleNamespace(verbose=0, stdout=None)
    scf = SimpleNamespace(mol=mol, get_ovlp=lambda: np.eye(ncore + ncas + 2))
    solver = SimpleNamespace(
        nroots=len(reference_energies),
        nelecas=nelec,
        convergence_info={"local_residual_bound": 0.0},
        driver=None,
        kets=None,
    )
    return SimpleNamespace(
        _scf=scf,
        mol=mol,
        stdout=None,
        verbose=0,
        frozen=0,
        ncore=ncore,
        ncas=ncas,
        nelecas=nelec,
        fcisolver=solver,
        mo_coeff=np.eye(ncore + ncas + 2, dtype=complex),
        e_states=np.asarray(reference_energies),
    )


def _complex_fixture():
    ncore, ncas, nvirt, nelec = 2, 6, 2, 4
    nmo = ncore + ncas + nvirt
    h1e, eri = _random_physical_integrals(nmo)
    h1e = np.array(h1e, copy=True)
    h1e[:ncore, :ncore] += np.diag([-2.1, -1.7])
    h1e[-nvirt:, -nvirt:] += np.diag([1.4, 1.9])
    eris_0 = spinor_helper.init_eris(h1e, eri, ncore, ncas)
    active = slice(ncore, ncore + ncas)
    w = eris_0.get_phys("PPPP")
    active_energies, active_states = _active_eigenstates(
        eris_0.h1eff[active, active],
        w[active, active, active, active],
        nelec,
    )
    state_dm1 = tuple(
        _raw_transition_rdm(state, state, ncas, 1)
        for state in active_states[:2]
    )
    state_fock = tuple(
        _state_specific_generalized_fock(eris_0, dm1) for dm1 in state_dm1
    )
    canonstep = tuple(
        _independent_canonstep(fock, ncore, ncas) for fock in state_fock
    )
    rotations = tuple(item[0] for item in canonstep)
    orbital_energies = tuple(item[1] for item in canonstep)
    row_eris = tuple(ssmod._rotate_eris(eris_0, rotation) for rotation in rotations)
    references = tuple(
        _embed_active_state(state, ncore, ncas, nvirt)
        for state in active_states[:2]
    )
    reference_energies = -75.0 + active_energies[:2]
    return {
        "ncore": ncore,
        "ncas": ncas,
        "nvirt": nvirt,
        "nelec": nelec,
        "eris": row_eris,
        "input_eris": eris_0,
        "state_dm1": state_dm1,
        "state_fock": state_fock,
        "row_rotation": rotations,
        "active_energies": active_energies[:2],
        "active_states": active_states[:2],
        "references": references,
        "core_energy": tuple(energy[:ncore] for energy in orbital_energies),
        "virtual_energy": tuple(
            energy[ncore + ncas :] for energy in orbital_energies
        ),
        "orbital_energy": orbital_energies,
        "reference_energies": reference_energies,
    }


def test_qd_type_public_api_defaults_normalization_and_legacy_guard(
    monkeypatch,
):
    mc = _make_fake_mc(1, 2, 1, (-1.0,))

    assert {
        "QDSCNEVPT2Result",
        "QDBlochSCNEVPT2Result",
        "WickX2CQDSCNEVPT2",
        "X2CQDSCNEVPT2",
        "WickX2CQDBlochSCNEVPT2",
        "X2CQDBlochSCNEVPT2",
    }.issubset(qdmod.__all__)
    assert mrpt_api.QDSCNEVPT2Result is qdmod.QDSCNEVPT2Result
    assert mrpt_api.QDBlochSCNEVPT2Result is qdmod.QDSCNEVPT2Result
    assert mrpt_api.WickX2CQDSCNEVPT2 is qdmod.WickX2CQDSCNEVPT2
    assert mrpt_api.X2CQDSCNEVPT2 is qdmod.WickX2CQDSCNEVPT2
    assert mrpt_api.MRPTNumericalWarning is ssmod.MRPTNumericalWarning
    assert qdmod.X2CQDSCNEVPT2 is qdmod.WickX2CQDSCNEVPT2
    assert qdmod.X2CQDBlochSCNEVPT2 is qdmod.WickX2CQDBlochSCNEVPT2
    assert qdmod.QDBlochSCNEVPT2Result is qdmod.QDSCNEVPT2Result
    assert qdmod.WickX2CQDSCNEVPT2(mc).qd_type == "van_vleck"
    for value in ("van_vleck", "VAN_VLECK", "Van-Vleck", "vanvleck"):
        assert qdmod._normalize_qd_type(value) == "van_vleck"
    for value in ("bloch", "BLOCH", " Bloch "):
        assert qdmod._normalize_qd_type(value) == "bloch"
    for value in (None, "hermitian", "", 7):
        with pytest.raises(ValueError, match="qd_type"):
            qdmod._normalize_qd_type(value)

    generic = qdmod.WickX2CQDSCNEVPT2(mc)
    generic.h_eff = np.eye(1)
    with pytest.raises(ValueError, match="qd_type"):
        generic.kernel(qd_type="not-a-representation")
    assert generic.result is None
    assert generic.h_eff is None

    forwarded = {}
    fluent = qdmod.WickX2CQDSCNEVPT2(mc)

    def fake_kernel(*args, **kwargs):
        forwarded["args"] = args
        forwarded["kwargs"] = kwargs

    monkeypatch.setattr(fluent, "kernel", fake_kernel)
    assert (
        fluent.run(
            "positional",
            roots=(0,),
            qd_type="BLOCH",
            state_pdms={0: "sentinel"},
            verbose=7,
        )
        is fluent
    )
    assert forwarded == {
        "args": ("positional",),
        "kwargs": {
            "roots": (0,),
            "qd_type": "BLOCH",
            "state_pdms": {0: "sentinel"},
        },
    }
    assert fluent.verbose == 7
    assert fluent.qd_type == "BLOCH"

    legacy = qdmod.WickX2CQDBlochSCNEVPT2(mc)
    assert legacy.qd_type == "bloch"
    with pytest.raises(ValueError, match="only supports qd_type='bloch'"):
        qdmod.WickX2CQDBlochSCNEVPT2(mc, qd_type="van_vleck")
    with pytest.raises(ValueError, match="only supports qd_type='bloch'"):
        legacy.kernel(qd_type="van_vleck")
    assert legacy.result is None
    assert legacy.h_eff_bloch is None
    assert legacy.h_eff_van_vleck is None

    run_legacy = qdmod.WickX2CQDBlochSCNEVPT2(mc)
    with pytest.raises(ValueError, match="only supports qd_type='bloch'"):
        run_legacy.run(qd_type="van_vleck")
    assert run_legacy.qd_type == "bloch"


def test_van_vleck_formula_and_classwise_decomposition_use_adjoint():
    rng = np.random.default_rng(11409)
    synthetic_bloch = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    expected = 0.5 * (synthetic_bloch + synthetic_bloch.conj().T)
    wrong_transpose_only = 0.5 * (synthetic_bloch + synthetic_bloch.T)
    actual = qdmod._van_vleck_hermitian_part(synthetic_bloch)
    np.testing.assert_array_equal(actual, expected)
    assert np.max(np.abs(actual - wrong_transpose_only)) > 1.0e-2
    assert np.max(np.abs(actual - actual.conj().T)) == 0.0

    nmodel = 3
    blocks = {}
    for key in ssmod.SUBSPACE_ORDER:
        block = 0.03 * (
            rng.normal(size=(nmodel, nmodel))
            + 1j * rng.normal(size=(nmodel, nmodel))
        )
        block[np.diag_indices(nmodel)] = block.diagonal().real
        blocks[key] = block
    reference = np.array([-75.1, -74.9, -74.7])
    h2_bloch = sum(blocks.values())
    h_eff_bloch = np.diag(reference.astype(complex)) + h2_bloch
    vv_blocks, h2_vv, h_eff_vv, diagnostics = (
        qdmod._build_van_vleck_matrices(
            blocks,
            h_eff_bloch,
            reference,
            audit_tol=1.0e-13,
        )
    )
    for key in ssmod.SUBSPACE_ORDER:
        expected_key = 0.5 * (blocks[key] + blocks[key].conj().T)
        np.testing.assert_array_equal(vv_blocks[key], expected_key)
        np.testing.assert_array_equal(vv_blocks[key], vv_blocks[key].conj().T)
        np.testing.assert_array_equal(
            np.diag(vv_blocks[key]), np.diag(blocks[key])
        )
    np.testing.assert_allclose(
        sum(vv_blocks.values()), h2_vv, atol=0.0, rtol=0.0
    )
    np.testing.assert_allclose(
        h_eff_vv,
        0.5 * (h_eff_bloch + h_eff_bloch.conj().T),
        atol=2.0e-16,
        rtol=0.0,
    )
    np.testing.assert_array_equal(np.diag(h_eff_vv), np.diag(h_eff_bloch))
    assert {
        "maximum_bloch_nonhermiticity",
        "frobenius_bloch_nonhermiticity",
        "maximum_van_vleck_nonhermiticity",
        "frobenius_van_vleck_nonhermiticity",
        "maximum_hermitization_change",
        "frobenius_hermitization_change",
        "van_vleck_global_formula_error",
        "van_vleck_subspace_sum_error",
        "maximum_van_vleck_diagonal_change",
        "maximum_subspace_diagonal_change",
    }.issubset(diagnostics)
    assert diagnostics["maximum_bloch_nonhermiticity"] > 1.0e-3
    assert diagnostics["maximum_van_vleck_nonhermiticity"] == 0.0
    assert diagnostics["frobenius_van_vleck_nonhermiticity"] == 0.0
    assert diagnostics["van_vleck_global_formula_error"] < 1.0e-14
    assert diagnostics["van_vleck_subspace_sum_error"] < 1.0e-14
    assert diagnostics["maximum_van_vleck_diagonal_change"] == 0.0
    assert diagnostics["maximum_subspace_diagonal_change"] == 0.0

    invalid = {key: np.array(value, copy=True) for key, value in blocks.items()}
    invalid["ijrs"][0, 0] += 1.0e-6j
    invalid_h_eff = np.diag(reference.astype(complex)) + sum(invalid.values())
    with pytest.warns(ssmod.MRPTNumericalWarning, match="diagonal"):
        _by_subspace, _h2, _h_eff, diagnostics = (
            qdmod._build_van_vleck_matrices(
                invalid,
                invalid_h_eff,
                reference,
                audit_tol=1.0e-13,
            )
        )
    assert not diagnostics["bloch_diagonal_reality_gate_passed"]


def test_van_vleck_solver_is_hermitian_real_orthonormal_and_phase_gauged(
    monkeypatch,
):
    rng = np.random.default_rng(20200145)
    trial = rng.normal(size=(5, 5)) + 1j * rng.normal(size=(5, 5))
    matrix = 0.5 * (trial + trial.conj().T)

    def forbidden_general_solver(*_args, **_kwargs):
        raise AssertionError("Van Vleck path called the general eigensolver")

    monkeypatch.setattr(qdmod, "eig", forbidden_general_solver)
    eigenvalues, vectors, diagnostics = (
        qdmod._solve_van_vleck_eigensystem(
            matrix,
            hermiticity_tol=1.0e-13,
            residual_tol=1.0e-12,
        )
    )
    assert np.isrealobj(eigenvalues)
    np.testing.assert_allclose(
        eigenvalues, np.linalg.eigvalsh(matrix), atol=2.0e-14, rtol=0.0
    )
    np.testing.assert_allclose(
        vectors.conj().T @ vectors, np.eye(5), atol=2.0e-14, rtol=0.0
    )
    np.testing.assert_allclose(
        matrix @ vectors,
        vectors * eigenvalues[np.newaxis, :],
        atol=2.0e-14,
        rtol=0.0,
    )
    for column in range(vectors.shape[1]):
        pivot = int(np.argmax(np.abs(vectors[:, column])))
        assert vectors[pivot, column].imag == 0.0
        assert vectors[pivot, column].real >= 0.0
    assert diagnostics["maximum_matrix_nonhermiticity"] == 0.0
    assert diagnostics["residual_norm"] < 3.0e-14
    assert diagnostics["relative_residual_norm"] < 3.0e-14
    assert diagnostics["orthonormality_error"] < 3.0e-14


def test_van_vleck_solver_warns_on_nonhermitian_matrix():
    matrix = np.array([[0.0, 1.0j], [0.25j, 1.0]], dtype=complex)
    with pytest.warns(
        ssmod.MRPTNumericalWarning, match="pre-eigh Hermiticity"
    ):
        eigenvalues, vectors, diagnostics = (
            qdmod._solve_van_vleck_eigensystem(
                matrix,
                hermiticity_tol=1.0e-13,
                residual_tol=1.0e-12,
            )
        )
    assert np.all(np.isfinite(eigenvalues))
    assert np.all(np.isfinite(vectors))
    assert not diagnostics["hermiticity_gate_passed"]


def test_bloch_solver_retains_general_complex_left_right_eigenproblem():
    matrix = np.array([[0.0, 1.0], [-2.0, 0.0]], dtype=complex)
    eigenvalues, left, right, diagnostics = qdmod._solve_bloch_eigensystem(
        matrix
    )
    assert np.max(np.abs(eigenvalues.imag)) > 1.0
    np.testing.assert_allclose(
        matrix @ right,
        right * eigenvalues[np.newaxis, :],
        atol=2.0e-14,
        rtol=0.0,
    )
    left_rows = left.conj().T
    np.testing.assert_allclose(
        left_rows @ matrix,
        eigenvalues[:, np.newaxis] * left_rows,
        atol=2.0e-14,
        rtol=0.0,
    )
    assert diagnostics["maximum_nonhermiticity"] > 1.0
    assert diagnostics["maximum_eigenvalue_imaginary_part"] > 1.0


def test_transition_pdms_validate_all_ranks_and_adjoint_relation():
    data = _complex_fixture()
    bra, ket = data["active_states"]
    pdms_ij = _raw_transition_rdms(bra, ket, data["ncas"])
    pdms_ji = _raw_transition_rdms(ket, bra, data["ncas"])
    overlap = np.vdot(bra, ket)
    checked, diagnostics = qdmod.validate_transition_pdms(
        pdms_ij,
        data["ncas"],
        data["nelec"],
        overlap_ij=overlap,
        pdms_ji=pdms_ji,
        atol=2.0e-12,
        rtol=2.0e-12,
    )
    assert abs(np.trace(checked[0]) - data["nelec"] * overlap) < 2.0e-12
    assert np.max(abs(checked[2])) > 1.0e-7
    assert np.max(abs(checked[3])) > 1.0e-7
    for rank, (forward, reverse) in enumerate(
        zip(checked, pdms_ji, strict=True), start=1
    ):
        np.testing.assert_allclose(
            qdmod.adjoint_transition_pdm(forward),
            reverse,
            atol=2.0e-12,
            rtol=2.0e-12,
            err_msg=f"rank {rank}",
        )
        assert diagnostics[f"dm{rank}"]["reverse_adjoint_error"] < 2.0e-12


def test_injected_overlap_warns_on_inconsistent_reverse_direction():
    inconsistent = np.eye(2, dtype=complex)
    inconsistent[0, 1] = 2.0e-9 + 3.0e-9j
    inconsistent[1, 0] = 2.0e-9 + 3.0e-9j
    with pytest.warns(ssmod.MRPTNumericalWarning, match="not Hermitian"):
        matrix = qdmod._injected_model_overlap(
            inconsistent, (0, 1), atol=1.0e-12, rtol=0.0
        )
    np.testing.assert_array_equal(matrix, inconsistent)
    with pytest.warns(ssmod.MRPTNumericalWarning, match="not adjoints"):
        matrix = qdmod._injected_model_overlap(
            {(0, 1): 2.0e-9 + 3.0e-9j, (1, 0): 2.0e-9 + 3.0e-9j},
            (0, 1),
            atol=1.0e-12,
            rtol=0.0,
        )
    assert matrix.shape == (2, 2)


def test_production_transition_wick_needs_no_rank_four_density():
    equations = ssmod._compile_wick_equations()
    assert all(
        "dm4" not in equations.transition_norm_code[key]
        for key in ssmod.SUBSPACE_ORDER
    )
    assert any(
        "dm3" in equations.transition_norm_code[key]
        for key in ssmod.SUBSPACE_ORDER
    )


def test_discarded_near_null_coupling_uses_elementwise_cauchy_bound():
    row_norm = np.array([2.0e-15, 1.0])
    partner_norm = np.array([2.0e-2, 1.0])
    bound = np.sqrt(row_norm[0] * partner_norm[0])
    coupling = np.array([0.75 * bound + 0.5e-12j, 99.0 + 3.0j])
    discarded = np.array([True, False])

    diagnostics = qdmod._validate_discarded_couplings_cauchy(
        coupling,
        discarded,
        row_norm,
        partner_norm,
        row_root=0,
        column_root=1,
        subspace="ijr",
        norm_tol=1.0e-14,
        atol=1.0e-12,
        rtol=0.0,
    )
    assert diagnostics["zero_norm_count"] == 1
    assert diagnostics["zero_norm_maximum_absolute_index"] == [0]
    assert diagnostics["zero_norm_cauchy_gate_passed"] is True
    assert diagnostics["zero_norm_maximum_acceptance_excess"] < 0.0
    assert np.isfinite(diagnostics["zero_norm_maximum_cauchy_ratio"])
    magnitude = abs(coupling[0])
    assert diagnostics["zero_norm_maximum_absolute_value"] == pytest.approx(
        magnitude
    )
    assert diagnostics[
        "zero_norm_numerical_allowance_at_maximum"
    ] == pytest.approx(1.0e-12)
    assert diagnostics[
        "zero_norm_acceptance_limit_at_maximum"
    ] == pytest.approx(bound + 1.0e-12)
    assert diagnostics[
        "zero_norm_physical_excess_at_maximum"
    ] == pytest.approx(magnitude - bound)
    assert diagnostics[
        "zero_norm_maximum_acceptance_excess"
    ] == pytest.approx(magnitude - bound - 1.0e-12)
    assert diagnostics["zero_norm_maximum_acceptance_excess_index"] == [0]
    assert diagnostics["zero_norm_maximum_cauchy_ratio"] == pytest.approx(
        magnitude / bound
    )
    np.testing.assert_allclose(
        diagnostics["zero_norm_cauchy_bound_at_maximum"], bound
    )
    np.testing.assert_allclose(
        diagnostics["zero_norm_row_norm_at_maximum"], row_norm[0]
    )
    np.testing.assert_allclose(
        diagnostics["zero_norm_partner_norm_at_maximum"], partner_norm[0]
    )

    roundoff_only = qdmod._validate_discarded_couplings_cauchy(
        np.array([0.5e-12]),
        np.array([True]),
        np.array([0.0]),
        np.array([1.0]),
        row_root=0,
        column_root=1,
        subspace="ijr",
        norm_tol=1.0e-14,
        atol=1.0e-12,
        rtol=0.0,
    )
    assert roundoff_only["zero_norm_maximum_cauchy_ratio"] is None
    assert roundoff_only["zero_norm_cauchy_ratio_unbounded_count"] == 1
    assert roundoff_only["zero_norm_cauchy_bound_at_maximum"] == 0.0
    assert roundoff_only[
        "zero_norm_numerical_allowance_at_maximum"
    ] == pytest.approx(1.0e-12)
    assert roundoff_only[
        "zero_norm_acceptance_limit_at_maximum"
    ] == pytest.approx(1.0e-12)
    assert np.isfinite(
        roundoff_only["zero_norm_maximum_finite_cauchy_ratio"]
    )
    json.dumps(roundoff_only, allow_nan=False)

    exact_boundary = qdmod._validate_discarded_couplings_cauchy(
        np.array([bound + 1.0e-12]),
        np.array([True]),
        np.array([row_norm[0]]),
        np.array([partner_norm[0]]),
        row_root=0,
        column_root=1,
        subspace="ijr",
        norm_tol=1.0e-14,
        atol=1.0e-12,
        rtol=0.0,
    )
    assert exact_boundary["zero_norm_cauchy_gate_passed"] is True
    assert exact_boundary["zero_norm_maximum_acceptance_excess"] <= 0.0

    with pytest.warns(
        ssmod.MRPTNumericalWarning,
        match=(
            r"row root 0, column root 1, subspace ijr:.*at \(0,\): "
            r"\|B\|=.*N_row=.*N_partner=.*sqrt\(N_row\*N_partner\)="
        ),
    ):
        failed = qdmod._validate_discarded_couplings_cauchy(
            np.array([1.5 * bound, 0.0]),
            discarded,
            row_norm,
            partner_norm,
            row_root=0,
            column_root=1,
            subspace="ijr",
            norm_tol=1.0e-14,
            atol=1.0e-12,
            rtol=0.0,
        )
    assert failed["zero_norm_cauchy_gate_passed"] is False


def test_blockwise_wick_eri_rotation_matches_full_dense_rotation():
    data = _complex_fixture()
    base = data["input_eris"]
    rotation = data["row_rotation"][1]
    compact = ssmod._rotate_wick_eris(base, rotation)
    full = ssmod._rotate_eris(base, rotation)
    for key in ssmod._H1_KEYS:
        np.testing.assert_allclose(
            compact.get_h1eff(key),
            full.get_h1eff(key),
            atol=5.0e-13,
            rtol=5.0e-13,
            err_msg=f"h1eff {key}",
        )
        assert not np.shares_memory(compact.get_h1eff(key), base.h1eff)
    for key in ssmod._W_KEYS:
        np.testing.assert_allclose(
            compact.get_phys(key),
            full.get_phys(key),
            atol=5.0e-13,
            rtol=5.0e-13,
            err_msg=f"w {key}",
        )
        assert not np.shares_memory(compact.get_phys(key), base.pppp)

    invalid = rotation.copy()
    invalid[0, -1] = 1.0e-5
    with pytest.raises(RuntimeError, match="not unitary"):
        ssmod._rotate_wick_eris(base, invalid)


def test_single_precision_transition_is_promoted_before_tblis():
    data = _complex_fixture()
    bra, ket = data["active_states"]
    pdms = tuple(
        density.astype(np.complex64)
        for density in _raw_transition_rdms(bra, ket, data["ncas"])[:3]
    )
    checked, _diagnostics = qdmod.validate_transition_pdms(
        pdms,
        data["ncas"],
        data["nelec"],
        overlap_ij=np.vdot(bra, ket),
        atol=2.0e-6,
        rtol=2.0e-6,
    )
    numpy_result = qdmod._evaluate_transition_perturber_couplings(
        data["eris"][0],
        checked,
        np.vdot(bra, ket),
        row_root=0,
        column_root=1,
        contraction_backend="numpy",
    )
    tblis_result = qdmod._evaluate_transition_perturber_couplings(
        data["eris"][0],
        checked,
        np.vdot(bra, ket),
        row_root=0,
        column_root=1,
        contraction_backend="pytblis",
    )
    for key in ssmod.SUBSPACE_ORDER:
        assert tblis_result[key].dtype == np.complex128
        np.testing.assert_allclose(
            tblis_result[key],
            numpy_result[key],
            atol=2.0e-11,
            rtol=2.0e-11,
        )


def test_tblis_rejects_single_precision_state_or_integral_operands():
    data = _complex_fixture()
    state = data["active_states"][0]
    state_pdms = tuple(
        density.astype(np.complex64)
        for density in _raw_transition_rdms(state, state, data["ncas"])
    )
    with pytest.raises(TypeError, match="unsupported base precision"):
        ssmod._evaluate_wick_subspaces(
            data["eris"][0],
            state_pdms,
            data["core_energy"][0],
            data["virtual_energy"][0],
            contraction_backend="pytblis",
        )

    compact = ssmod._compact_wick_eris(data["eris"][0])
    low_precision = ssmod._WickERIBlocks(
        ncore=compact.ncore,
        ncas=compact.ncas,
        nvirt=compact.nvirt,
        h1eff_blocks={
            key: value.astype(np.complex64)
            for key, value in compact.h1eff_blocks.items()
        },
        phys_blocks={
            key: value.astype(np.complex64)
            for key, value in compact.phys_blocks.items()
        },
    )
    transition = _raw_transition_rdms(
        data["active_states"][0], data["active_states"][1], data["ncas"]
    )[:3]
    with pytest.raises(TypeError, match="unsupported base precision"):
        qdmod._evaluate_transition_perturber_couplings(
            low_precision,
            transition,
            0.0,
            row_root=0,
            column_root=1,
            contraction_backend="pytblis",
        )


def test_qd_rejects_nonfinite_tolerances_before_preparation():
    mc = _make_fake_mc(1, 2, 1, (-1.0,))
    qd = qdmod.WickX2CQDBlochSCNEVPT2(mc)
    qd.model_overlap_atol = np.nan
    with pytest.raises(ValueError, match="model_overlap_atol"):
        qd.kernel(roots=(0,))
    assert qd.result is None

    qd = qdmod.WickX2CQDBlochSCNEVPT2(mc)
    with pytest.raises(ValueError, match="eigenvalue_imag_warn"):
        qd.kernel(roots=(0,), eigenvalue_imag_warn=np.inf)
    assert qd.result is None


def test_reverse_injected_rank_four_is_not_materialized(monkeypatch):
    calls = []

    def adjoint(density):
        if density is rank_four_sentinel:
            raise AssertionError("production touched injected transition dm4")
        calls.append(density)
        return np.asarray(density).conj()

    rank_four_sentinel = object()
    ranks = tuple(np.asarray(rank) for rank in (1.0, 2.0, 3.0))
    monkeypatch.setattr(qdmod, "adjoint_transition_pdm", adjoint)
    actual, source = qdmod._transition_mapping_item(
        {(1, 0): ranks + (rank_four_sentinel,)},
        0,
        1,
        max_rank=3,
    )
    assert source == "injected_adjoint"
    assert len(actual) == 3
    assert calls == list(ranks)


def test_transition_scalar_dm0_controls_active_free_ijrs_coupling():
    data = _complex_fixture()
    bra, orthogonal = data["active_states"]
    eris = data["eris"][0]
    pdms = _raw_transition_rdms(bra, orthogonal, data["ncas"])
    couplings = qdmod._evaluate_transition_perturber_couplings(
        eris,
        pdms,
        np.vdot(bra, orthogonal),
        row_root=0,
        column_root=1,
        contraction_backend="numpy",
    )
    np.testing.assert_allclose(couplings["ijrs"], 0.0, atol=2.0e-12)

    ket = 0.6 * bra + 0.8j * orthogonal
    overlap = np.vdot(bra, ket)
    pdms = _raw_transition_rdms(bra, ket, data["ncas"])
    couplings = qdmod._evaluate_transition_perturber_couplings(
        eris,
        pdms,
        overlap,
        row_root=0,
        column_root=1,
        contraction_backend="numpy",
    )
    tblis_couplings = qdmod._evaluate_transition_perturber_couplings(
        eris,
        pdms,
        overlap,
        row_root=0,
        column_root=1,
        contraction_backend="pytblis",
    )
    for key in ssmod.SUBSPACE_ORDER:
        np.testing.assert_allclose(
            tblis_couplings[key], couplings[key], atol=2.0e-12, rtol=2.0e-12
        )
    direct_bra = _occupation_projected_h_reference(
        data["references"][0], eris
    )
    direct_ket_state = _embed_active_state(
        ket, data["ncore"], data["ncas"], data["nvirt"]
    )
    direct_ket = _occupation_projected_h_reference(direct_ket_state, eris)
    expected = _cross_perturber_overlaps(direct_bra, direct_ket)["ijrs"]
    ordered = _direct_pair_mask("ijrs", expected.shape)
    np.testing.assert_allclose(
        couplings["ijrs"][ordered], expected[ordered], atol=2.0e-12
    )
    diagonal = _cross_perturber_overlaps(direct_bra, direct_bra)["ijrs"]
    np.testing.assert_allclose(
        couplings["ijrs"][ordered],
        overlap * diagonal[ordered],
        atol=2.0e-12,
    )


def test_qd_bloch_driver_against_direct_fock_space_and_covariances(monkeypatch):
    data = _complex_fixture()
    nroots = 2
    state_pdms = {
        root: _raw_transition_rdms(
            data["active_states"][root],
            data["active_states"][root],
            data["ncas"],
        )
        for root in range(nroots)
    }
    transition_pdms = {
        (0, 1): _raw_transition_rdms(
            data["active_states"][0],
            data["active_states"][1],
            data["ncas"],
        )
    }
    direct_rows = []
    for root in range(nroots):
        direct_rows.append(
            _direct_row_data(
                data["references"][root],
                data["eris"][root],
                data["active_energies"][root],
                data["core_energy"][root],
                data["virtual_energy"][root],
            )
        )
    direct_h2_by_subspace = {
        key: np.zeros((nroots, nroots), dtype=complex)
        for key in ssmod.SUBSPACE_ORDER
    }
    for irow in range(nroots):
        perturbers_i, norms_i, denominators_i, masks_i = direct_rows[irow]
        for jcol in range(nroots):
            perturbers_j_in_i_basis = _occupation_projected_h_reference(
                data["references"][jcol], data["eris"][irow]
            )
            cross = _cross_perturber_overlaps(
                perturbers_i, perturbers_j_in_i_basis
            )
            for key in ssmod.SUBSPACE_ORDER:
                mask = masks_i[key]
                direct_h2_by_subspace[key][irow, jcol] = -np.sum(
                    cross[key][mask] / denominators_i[key][mask]
                )
                if irow == jcol:
                    np.testing.assert_allclose(
                        cross[key], norms_i[key], atol=2.0e-12, rtol=2.0e-12
                    )
    assert all(
        np.max(abs(matrix)) > 1.0e-10
        for matrix in direct_h2_by_subspace.values()
    )

    mc = _make_fake_mc(
        data["ncore"],
        data["ncas"],
        data["nelec"],
        data["reference_energies"],
    )
    canonicalize_calls = []
    mc.ci = (0, 1)
    mc.fcisolver.root_overlap = np.eye(2, dtype=complex)
    dense_eris_calls = []

    def shared_dense_eris(_mc, mo_coeff, *, roundoff_factor):
        dense_eris_calls.append(
            (np.array(mo_coeff, copy=True), roundoff_factor)
        )
        return data["input_eris"]

    monkeypatch.setattr(qdmod._utils, "_dense_eris_from_mc", shared_dense_eris)

    def get_fock(mo_coeff, ci, eris, casdm1, verbose):
        del ci, eris, verbose
        np.testing.assert_allclose(mo_coeff, np.eye(mo_coeff.shape[1]))
        return _state_specific_generalized_fock(
            data["input_eris"], casdm1
        )

    mc.get_fock = get_fock

    def canonicalize(mo, *, ci, cas_natorb, casdm1, verbose):
        root = int(ci)
        canonicalize_calls.append((root, np.array(casdm1, copy=True)))
        result = zmcscf.canonicalize(
            mc,
            mo_coeff=mo,
            ci=ci,
            cas_natorb=cas_natorb,
            casdm1=casdm1,
            verbose=verbose,
        )
        np.testing.assert_allclose(
            result[2], data["orbital_energy"][root], atol=2.0e-12
        )
        return result

    mc.canonicalize = canonicalize
    transition_phases = {"values": np.ones(nroots, dtype=complex)}
    transition_rdm_calls = []

    def make_transition_dm123(_solver, bra_root, ket_root):
        bra_root = int(bra_root)
        ket_root = int(ket_root)
        transition_rdm_calls.append((bra_root, ket_root))
        direct = transition_pdms[(0, 1)][:3]
        if (bra_root, ket_root) == (0, 1):
            base = direct
        elif (bra_root, ket_root) == (1, 0):
            base = tuple(
                qdmod.adjoint_transition_pdm(density) for density in direct
            )
        else:
            raise AssertionError("unexpected transition root pair")
        phases = transition_phases["values"]
        scale = phases[bra_root].conjugate() * phases[ket_root]
        return tuple(scale * density for density in base)

    monkeypatch.setattr(qdmod, "_make_transition_dm123", make_transition_dm123)

    qd = qdmod.WickX2CQDSCNEVPT2(mc, qd_type="bloch")
    qd.integral_roundoff_factor = 2.5
    qd.kernel(
        roots=(0, 1),
        state_pdms=state_pdms,
        contraction_backend="numpy",
    )
    generic_bloch_transition_call_count = len(transition_rdm_calls)
    assert transition_rdm_calls == [(0, 1)]
    assert len(dense_eris_calls) == 1
    np.testing.assert_array_equal(dense_eris_calls[0][0], mc.mo_coeff)
    assert dense_eris_calls[0][1] == qd.integral_roundoff_factor
    assert (
        qd._new_ss_adapter(mc, 0).integral_roundoff_factor
        == qd.integral_roundoff_factor
    )
    assert qd.diagnostics["shared_input_eris_generated"] is True
    assert all(qd.row_data[root].pdms123 is None for root in range(nroots))

    # The historical wrapper delegates to the exact same Bloch kernel and
    # neither representation requests an additional transition RDM.
    transition_rdm_calls.clear()
    legacy = qdmod.WickX2CQDBlochSCNEVPT2(mc)
    assert (
        legacy.run(
            roots=(0, 1),
            state_pdms=state_pdms,
            eris=data["input_eris"],
            eris_basis="input_mo",
            contraction_backend="numpy",
        )
        is legacy
    )
    legacy_transition_call_count = len(transition_rdm_calls)
    assert legacy_transition_call_count == generic_bloch_transition_call_count
    assert transition_rdm_calls == [(0, 1)]
    assert legacy.qd_type == qd.qd_type == "bloch"
    assert legacy.result.qd_type == "bloch"
    for key in ssmod.SUBSPACE_ORDER:
        np.testing.assert_allclose(
            legacy.h2_by_subspace[key],
            qd.h2_by_subspace[key],
            atol=2.0e-13,
            rtol=0.0,
        )
    for name in (
        "h2_bloch",
        "h_eff_bloch",
        "h2_van_vleck",
        "h_eff_van_vleck",
        "e_qd",
        "left_eigenvectors",
        "right_eigenvectors",
    ):
        np.testing.assert_allclose(
            getattr(legacy, name), getattr(qd, name), atol=2.0e-13, rtol=0.0
        )
    del legacy

    maximum_norm_error = 0.0
    maximum_partner_norm_error = 0.0
    maximum_wrong_row_partner_difference = 0.0
    maximum_denominator_error = 0.0
    maximum_cross_error = 0.0
    for irow in range(nroots):
        _perturbers, direct_norms, direct_denominators, direct_masks = (
            direct_rows[irow]
        )
        for key in ssmod.SUBSPACE_ORDER:
            ordered = _direct_pair_mask(key, direct_norms[key].shape)
            row_arrays = qd.row_data[irow].subspace_arrays[key]
            maximum_norm_error = max(
                maximum_norm_error,
                float(
                    np.max(
                        np.abs(
                            row_arrays["norm"][ordered]
                            - direct_norms[key][ordered].real
                        ),
                        initial=0.0,
                    )
                ),
            )
            np.testing.assert_array_equal(
                row_arrays["nonzero"], direct_masks[key]
            )
            maximum_denominator_error = max(
                maximum_denominator_error,
                float(
                    np.max(
                        np.abs(
                            row_arrays["denominator"][direct_masks[key]]
                            - direct_denominators[key][direct_masks[key]].real
                        ),
                        initial=0.0,
                    )
                ),
            )

        jcol = 1 - irow
        pdms_ij = (
            transition_pdms[(0, 1)][:3]
            if irow == 0
            else tuple(
                qdmod.adjoint_transition_pdm(density)
                for density in transition_pdms[(0, 1)][:3]
            )
        )
        direct_ket = _occupation_projected_h_reference(
            data["references"][jcol], data["eris"][irow]
        )
        direct_partner_norms = _cross_perturber_overlaps(
            direct_ket, direct_ket
        )
        wick_partner_norms, partner_norm_diagnostics = (
            qdmod._evaluate_row_basis_partner_norms(
                qd.row_data[irow].eris,
                state_pdms[jcol][:3],
                row_root=irow,
                partner_root=jcol,
                contraction_backend="numpy",
                return_diagnostics=True,
            )
        )
        for key in ssmod.SUBSPACE_ORDER:
            ordered = _direct_pair_mask(
                key, direct_partner_norms[key].shape
            )
            maximum_partner_norm_error = max(
                maximum_partner_norm_error,
                float(
                    np.max(
                        np.abs(
                            wick_partner_norms[key][ordered]
                            - direct_partner_norms[key][ordered].real
                        ),
                        initial=0.0,
                    )
                ),
            )
            maximum_wrong_row_partner_difference = max(
                maximum_wrong_row_partner_difference,
                float(
                    np.max(
                        np.abs(
                            wick_partner_norms[key][ordered]
                            - direct_rows[jcol][1][key][ordered].real
                        ),
                        initial=0.0,
                    )
                ),
            )
            expected_selected = direct_partner_norms[key][ordered].real
            expected_minimum = (
                float(np.min(expected_selected))
                if expected_selected.size
                else 0.0
            )
            expected_maximum = (
                float(np.max(expected_selected))
                if expected_selected.size
                else 0.0
            )
            np.testing.assert_allclose(
                partner_norm_diagnostics[key]["minimum_ordered_norm"],
                expected_minimum,
                atol=3.0e-11,
            )
            np.testing.assert_allclose(
                partner_norm_diagnostics[key]["maximum_ordered_norm"],
                expected_maximum,
                atol=3.0e-11,
            )
        couplings = qdmod._evaluate_transition_perturber_couplings(
            qd.row_data[irow].eris,
            pdms_ij,
            0.0,
            row_root=irow,
            column_root=jcol,
            row_nonzero={
                key: qd.row_data[irow].subspace_arrays[key]["nonzero"]
                for key in ssmod.SUBSPACE_ORDER
            },
            row_norm={
                key: qd.row_data[irow].subspace_arrays[key]["norm"]
                for key in ssmod.SUBSPACE_ORDER
            },
            partner_norm=wick_partner_norms,
            contraction_backend="numpy",
        )
        direct_cross = _cross_perturber_overlaps(
            direct_rows[irow][0], direct_ket
        )
        for key in ssmod.SUBSPACE_ORDER:
            ordered = _direct_pair_mask(key, direct_cross[key].shape)
            maximum_cross_error = max(
                maximum_cross_error,
                float(
                    np.max(
                        np.abs(
                            couplings[key][ordered]
                            - direct_cross[key][ordered]
                        ),
                        initial=0.0,
                    )
                ),
            )
    assert maximum_norm_error < 3.0e-11
    assert maximum_partner_norm_error < 3.0e-11
    assert maximum_wrong_row_partner_difference > 1.0e-5
    assert maximum_denominator_error < 3.0e-11
    assert maximum_cross_error < 3.0e-11
    pair_diagnostics = qd.diagnostics["pair_diagnostics"]["0,1"]
    for direction, irow in (("row_i_couplings", 0), ("row_j_couplings", 1)):
        for key in ssmod.SUBSPACE_ORDER:
            shape = qd.row_data[irow].subspace_arrays[key]["norm"].shape
            ordered = _direct_pair_mask(key, shape)
            nonzero = qd.row_data[irow].subspace_arrays[key]["nonzero"]
            diagnostics = pair_diagnostics[direction][key]
            assert diagnostics["zero_norm_count"] == int(
                np.count_nonzero(ordered & ~nonzero)
            )
            assert diagnostics["zero_norm_cauchy_gate_passed"] is True
    for key in ssmod.SUBSPACE_ORDER:
        np.testing.assert_allclose(
            qd.h2_by_subspace[key],
            direct_h2_by_subspace[key],
            atol=3.0e-11,
            rtol=3.0e-11,
            err_msg=key,
        )
        assert qd.h2_by_subspace[key] is qd.h2_bloch_by_subspace[key]
        np.testing.assert_allclose(
            qd.h2_van_vleck_by_subspace[key],
            0.5
            * (
                qd.h2_bloch_by_subspace[key]
                + qd.h2_bloch_by_subspace[key].conj().T
            ),
            atol=2.0e-13,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            qd.h2_van_vleck_by_subspace[key],
            qd.h2_van_vleck_by_subspace[key].conj().T,
            atol=2.0e-13,
            rtol=0.0,
        )
        for root in range(nroots):
            assert (
                abs(
                    qd.h2_by_subspace[key][root, root]
                    - qd.row_data[root].subspace_energies[key]
                )
                    < 2.0e-13
                )
            np.testing.assert_allclose(
                qd.h2_van_vleck_by_subspace[key][root, root],
                qd.row_data[root].subspace_energies[key],
                atol=2.0e-13,
                rtol=0.0,
            )
    state_specific_totals = {}
    for root in range(nroots):
        state_specific = ssmod.WickX2CSCNEVPT2(mc)
        state_specific.kernel(
            root=root,
            pdms=state_pdms[root],
            eris=data["input_eris"],
            eris_basis="input_mo",
            denominator_mode="strict_si",
            contraction_backend="numpy",
        )
        state_specific_totals[root] = float(state_specific.e_tot)
        for key in ssmod.SUBSPACE_ORDER:
            np.testing.assert_allclose(
                qd.h2_by_subspace[key][root, root],
                state_specific.sub_eners[key],
                atol=2.0e-13,
                rtol=0.0,
            )
        np.testing.assert_allclose(
            qd.h_eff_bloch[root, root],
            state_specific.e_tot,
            atol=2.0e-13,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            qd.h_eff_van_vleck[root, root],
            state_specific.e_tot,
            atol=2.0e-13,
            rtol=0.0,
        )

    transition_rdm_calls.clear()
    single = qdmod.WickX2CQDSCNEVPT2(mc)
    assert (
        single.run(
            roots=(0,),
            state_pdms=state_pdms,
            eris=data["input_eris"],
            eris_basis="input_mo",
            contraction_backend="numpy",
        )
        is single
    )
    assert transition_rdm_calls == []
    assert single.qd_type == "van_vleck"
    assert np.isrealobj(single.e_qd)
    np.testing.assert_allclose(
        single.e_qd[0], state_specific_totals[0], atol=2.0e-13, rtol=0.0
    )
    np.testing.assert_allclose(
        single.h_eff_bloch[0, 0],
        state_specific_totals[0],
        atol=2.0e-13,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        single.h_eff_van_vleck[0, 0],
        state_specific_totals[0],
        atol=2.0e-13,
        rtol=0.0,
    )
    bloch_single_eigenvalue = qdmod._solve_bloch_eigensystem(
        single.h_eff_bloch
    )[0][0]
    np.testing.assert_allclose(
        bloch_single_eigenvalue,
        single.e_qd[0],
        atol=2.0e-13,
        rtol=0.0,
    )
    del single
    direct_h2 = sum(direct_h2_by_subspace.values())
    direct_h_eff = np.diag(data["reference_energies"]) + direct_h2
    direct_h2_error = float(np.max(np.abs(qd.h2_bloch - direct_h2)))
    direct_h_eff_error = float(
        np.max(np.abs(qd.h_eff_bloch - direct_h_eff))
    )
    np.testing.assert_allclose(qd.h2_bloch, direct_h2, atol=3.0e-11, rtol=3.0e-11)
    np.testing.assert_allclose(
        qd.h_eff_bloch, direct_h_eff, atol=3.0e-11, rtol=3.0e-11
    )
    np.testing.assert_allclose(
        qd.h2_van_vleck,
        sum(qd.h2_van_vleck_by_subspace.values()),
        atol=2.0e-13,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        qd.h_eff_van_vleck,
        0.5 * (qd.h_eff_bloch + qd.h_eff_bloch.conj().T),
        atol=2.0e-13,
        rtol=0.0,
    )
    assert qd.h2_effective is qd.h2_bloch
    assert qd.h_eff is qd.h_eff_bloch
    assert qd.eigenvectors is qd.right_eigenvectors
    direct_eigenvalues = np.linalg.eigvals(direct_h_eff)
    direct_eigenvalues = direct_eigenvalues[
        np.lexsort((direct_eigenvalues.imag, direct_eigenvalues.real))
    ]
    direct_eigenvalue_error = float(
        np.max(np.abs(qd.e_qd - direct_eigenvalues))
    )
    np.testing.assert_allclose(qd.e_qd, direct_eigenvalues, atol=3.0e-11)
    assert np.max(abs(qd.h_eff_bloch - qd.h_eff_bloch.T.conj())) > 1.0e-8
    assert qd.diagnostics["diagonal_reduction_maximum_error"] < 2.0e-13
    assert qd.diagnostics["eigensystem"]["right_residual_norm"] < 1.0e-10
    assert qd.diagnostics["eigensystem"]["left_residual_norm"] < 1.0e-10
    assert qd.diagnostics["qd_type"] == "bloch"
    assert "QD_Bloch" in qd.diagnostics["method"]
    assert "row-I ERIs/perturbers/strict-SI gaps" in qd.diagnostics[
        "matrix_convention"
    ]
    assert qd.diagnostics["maximum_bloch_nonhermiticity"] > 1.0e-8
    assert qd.diagnostics["maximum_van_vleck_nonhermiticity"] < 2.0e-13
    assert qd.diagnostics["van_vleck_global_formula_error"] < 2.0e-13
    assert qd.diagnostics["van_vleck_subspace_sum_error"] < 2.0e-13
    assert qd.diagnostics["maximum_van_vleck_diagonal_change"] < 2.0e-13
    assert qd.diagnostics["maximum_subspace_diagonal_change"] < 2.0e-13
    assert "six_root_splittings" not in qd.diagnostics
    assert "h2_assembly_error" not in qd.diagnostics
    assert "h_eff_assembly_error" not in qd.diagnostics
    assert (
        "root_assignment_reporting_only"
        not in qd.diagnostics["eigensystem"]
    )
    overlap_sources = qd.diagnostics["model_overlap_sources"]
    assert overlap_sources["0,0"] == "solver_cached"
    assert overlap_sources["1,1"] == "solver_cached"
    assert overlap_sources["0,1"] == "solver_cached"
    assert overlap_sources["1,0"] == "adjoint"
    assert [root for root, _dm1 in canonicalize_calls[:2]] == [0, 1]
    for root, dm1 in canonicalize_calls[:2]:
        np.testing.assert_allclose(dm1, state_pdms[root][0], atol=2.0e-13)

    theta = np.array([0.37, -0.81])
    phases = np.exp(1j * theta)
    transition_phases["values"] = phases
    transition_rdm_calls.clear()
    phased = qdmod.WickX2CQDSCNEVPT2(mc)
    assert phased.qd_type == "van_vleck"
    phased.kernel(
        roots=(0, 1),
        state_pdms=state_pdms,
        model_overlap=np.eye(2),
        eris=data["input_eris"],
        eris_basis="input_mo",
        contraction_backend="numpy",
    )
    assert len(transition_rdm_calls) == generic_bloch_transition_call_count
    phase_matrix = np.diag(phases)
    np.testing.assert_allclose(
        phased.h_eff_bloch,
        phase_matrix.conj().T @ qd.h_eff_bloch @ phase_matrix,
        atol=3.0e-11,
        rtol=3.0e-11,
    )
    np.testing.assert_allclose(
        phased.h_eff_van_vleck,
        phase_matrix.conj().T @ qd.h_eff_van_vleck @ phase_matrix,
        atol=3.0e-11,
        rtol=3.0e-11,
    )
    for key in ssmod.SUBSPACE_ORDER:
        np.testing.assert_allclose(
            phased.h2_bloch_by_subspace[key],
            phase_matrix.conj().T
            @ qd.h2_bloch_by_subspace[key]
            @ phase_matrix,
            atol=3.0e-11,
            rtol=3.0e-11,
        )
        np.testing.assert_allclose(
            phased.h2_van_vleck_by_subspace[key],
            phase_matrix.conj().T
            @ qd.h2_van_vleck_by_subspace[key]
            @ phase_matrix,
            atol=3.0e-11,
            rtol=3.0e-11,
        )
    phased_bloch_eigenvalues = qdmod._solve_bloch_eigensystem(
        phased.h_eff_bloch
    )[0]
    np.testing.assert_allclose(
        phased_bloch_eigenvalues, qd.e_qd, atol=3.0e-11
    )
    expected_van_vleck_eigenvalues = np.linalg.eigvalsh(
        qd.h_eff_van_vleck
    )
    np.testing.assert_allclose(
        phased.e_qd, expected_van_vleck_eigenvalues, atol=3.0e-11
    )
    assert np.isrealobj(phased.e_qd)
    assert phased.h2_by_subspace is phased.h2_van_vleck_by_subspace
    assert phased.h2_effective is phased.h2_van_vleck
    assert phased.h_eff is phased.h_eff_van_vleck
    np.testing.assert_allclose(
        phased.left_eigenvectors.conj().T @ phased.right_eigenvectors,
        np.eye(nroots),
        atol=2.0e-13,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        phased.h_eff @ phased.eigenvectors,
        phased.eigenvectors * phased.e_qd[np.newaxis, :],
        atol=2.0e-12,
        rtol=0.0,
    )
    assert phased.diagnostics["qd_type"] == "van_vleck"
    assert "canonical_Van_Vleck" in phased.diagnostics["method"]
    assert (
        "maximum_eigenvalue_imaginary_part"
        not in phased.diagnostics["eigensystem"]
    )

    transition_phases["values"] = np.ones(nroots, dtype=complex)
    transition_rdm_calls.clear()
    permuted = qdmod.WickX2CQDSCNEVPT2(mc)
    permuted.kernel(
        roots=(1, 0),
        qd_type="bloch",
        state_pdms=state_pdms,
        model_overlap=np.eye(2),
        eris=data["input_eris"],
        eris_basis="input_mo",
        contraction_backend="numpy",
    )
    assert len(transition_rdm_calls) == generic_bloch_transition_call_count
    assert permuted.qd_type == "bloch"
    permutation = np.array([[0.0, 1.0], [1.0, 0.0]])
    np.testing.assert_allclose(
        permuted.h_eff_bloch,
        permutation.T @ qd.h_eff_bloch @ permutation,
        atol=3.0e-11,
        rtol=3.0e-11,
    )
    np.testing.assert_allclose(
        permuted.h_eff_van_vleck,
        permutation.T @ qd.h_eff_van_vleck @ permutation,
        atol=3.0e-11,
        rtol=3.0e-11,
    )
    for key in ssmod.SUBSPACE_ORDER:
        np.testing.assert_allclose(
            permuted.h2_bloch_by_subspace[key],
            permutation.T @ qd.h2_bloch_by_subspace[key] @ permutation,
            atol=3.0e-11,
            rtol=3.0e-11,
        )
        np.testing.assert_allclose(
            permuted.h2_van_vleck_by_subspace[key],
            permutation.T @ qd.h2_van_vleck_by_subspace[key] @ permutation,
            atol=3.0e-11,
            rtol=3.0e-11,
        )
    np.testing.assert_allclose(permuted.e_qd, qd.e_qd, atol=3.0e-11)
    np.testing.assert_allclose(
        np.linalg.eigvalsh(permuted.h_eff_van_vleck),
        expected_van_vleck_eigenvalues,
        atol=3.0e-11,
    )

    print(
        "DIRECT_FOCK_MAX_ERRORS "
        f"norm={maximum_norm_error:.16e} "
        f"partner_norm={maximum_partner_norm_error:.16e} "
        f"denominator={maximum_denominator_error:.16e} "
        f"cross={maximum_cross_error:.16e} "
        f"h2={direct_h2_error:.16e} "
        f"h_eff={direct_h_eff_error:.16e} "
        f"eigenvalue={direct_eigenvalue_error:.16e}"
    )

    with pytest.raises(ValueError, match="strict_si"):
        qd.kernel(
            roots=(0, 1),
            denominator_mode="hermitianized",
        )
    assert qd.roots == ()
    assert qd.result is None
    assert qd.row_data == {}
    assert qd.h_eff_bloch is None


def test_block2_transition_rdm1234_against_fixed_n_oracle(tmp_path):
    from pyblock2.driver.core import DMRGDriver, SymmetryTypes

    data = _complex_fixture()
    ncas, nelec = data["ncas"], data["nelec"]
    basis = _fixed_particle_basis(ncas, nelec)
    occupations = np.zeros((len(basis), ncas), dtype=np.uint8)
    for row, state in enumerate(basis):
        occupations[row] = [(state >> site) & 1 for site in range(ncas)]
    driver = DMRGDriver(
        stack_mem=300_000_000,
        scratch=str(tmp_path / "qd-npdm"),
        clean_scratch=True,
        symm_type=SymmetryTypes.SGFCPX,
        n_threads=1,
    )
    driver.initialize_system(n_sites=ncas, n_elec=nelec, orb_sym=[0] * ncas)
    kets = tuple(
        driver.get_mps_from_csf_coefficients(
            occupations,
            data["active_states"][root][basis],
            f"QD-ROOT-{root}",
            dot=1,
            iprint=0,
        )
        for root in range(2)
    )
    solver = SimpleNamespace(
        driver=driver,
        kets=kets,
        ncas=ncas,
        nelecas=nelec,
        npdm_site_type=2,
        npdm_cutoff=1.0e-24,
    )
    try:
        actual = qdmod.make_transition_dm1234(solver, 0, 1)
        reverse = qdmod.make_transition_dm1234(solver, 1, 0)
        expected = _raw_transition_rdms(
            data["active_states"][0], data["active_states"][1], ncas
        )
        phase = np.vdot(expected[0].reshape(-1), actual[0].reshape(-1))
        phase /= abs(phase)
        for rank, (block2_dm, direct_dm) in enumerate(
            zip(actual, expected, strict=True), start=1
        ):
            np.testing.assert_allclose(
                block2_dm,
                phase * direct_dm,
                atol=2.0e-11,
                rtol=2.0e-11,
                err_msg=f"transition dm{rank}",
            )
        overlap = qdmod.make_transition_overlap(solver, 0, 1)
        qdmod.validate_transition_pdms(
            actual,
            ncas,
            nelec,
            overlap_ij=overlap,
            pdms_ji=reverse,
            atol=2.0e-11,
            rtol=2.0e-11,
        )
    finally:
        driver.finalize()
