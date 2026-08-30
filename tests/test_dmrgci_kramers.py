from pathlib import Path

import numpy as np
import pytest
import scipy.linalg
from pyscf import gto

from socutils.dmrg.dmrgci import DMRGCI, energy_from_rdms
from socutils.dmrg.kramers import (
    KramersResultAdapter,
    canonicalize_root_space_rdm1,
    identify_kramers_orbitals,
    kramers_residual,
    time_reverse_integrals,
    time_reverse_one_body,
)
from socutils.fci import zfci
from socutils.mcscf import zmc_superci, zmcscf
from socutils.scf import spinor_hf


def _nonadjacent_time_reversal():
    """Complex phases and nonadjacent pairs prevent ordering assumptions."""
    matrix = np.zeros((4, 4), dtype=complex)
    for (p, q), phase in zip(((0, 2), (1, 3)), (0.37, -0.81)):
        value = np.exp(1j * phase)
        matrix[q, p] = value
        matrix[p, q] = -value
    return matrix


def _kramers_hamiltonian():
    rng = np.random.default_rng(193)
    time_reversal = _nonadjacent_time_reversal()
    one_body = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    one_body = (one_body + one_body.T.conj()) * 0.5

    raw = rng.normal(size=(4, 4, 4, 4))
    permutations = (
        (0, 1, 2, 3), (1, 0, 2, 3),
        (0, 1, 3, 2), (1, 0, 3, 2),
        (2, 3, 0, 1), (3, 2, 0, 1),
        (2, 3, 1, 0), (3, 2, 1, 0),
    )
    two_body = sum(raw.transpose(axes) for axes in permutations) / 8
    one_body_tr, two_body_tr = time_reverse_integrals(
        time_reversal, one_body, two_body
    )
    one_body = (one_body + one_body_tr) * 0.5
    two_body = (two_body + two_body_tr) * 0.5
    return time_reversal, one_body, two_body


def test_sparse_time_reversal_one_body_matches_dense_definition():
    time_reversal = _nonadjacent_time_reversal()
    rng = np.random.default_rng(661)
    matrix = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    expected = np.einsum(
        "ap,bq,pq->ab",
        time_reversal,
        time_reversal.conj(),
        matrix.conj(),
        optimize=True,
    )
    assert np.max(
        abs(time_reverse_one_body(time_reversal, matrix) - expected)
    ) <= 1e-14


def _exact_root_space(solver, states, dm1s, norb, nelec):
    root_space = np.empty((2, 2, norb, norb), dtype=complex)
    for bra in range(2):
        for ket in range(2):
            if bra == ket:
                root_space[bra, ket] = dm1s[bra]
            else:
                root_space[bra, ket] = solver.trans_rdm1(
                    states[bra], states[ket], norb, nelec
                )
    return root_space


def test_fourfold_degenerate_manifold_is_basis_invariant():
    """Arbitrary roots in two degenerate doublets need manifold validation."""
    time_reversal = _nonadjacent_time_reversal()
    rng = np.random.default_rng(719)
    trial = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    root_vectors = np.linalg.qr(trial)[0]
    dm1s = [
        np.outer(root_vectors[:, root].conj(), root_vectors[:, root])
        for root in range(4)
    ]
    dm2s = [np.zeros((4,) * 4, dtype=complex) for _ in range(4)]

    adapter = KramersResultAdapter(time_reversal)
    pairs = adapter.analyze(
        np.zeros(4),
        dm1s,
        dm2s,
        weights=np.ones(4) / 4,
        overlap=np.eye(4),
        projected_hamiltonian=np.zeros((4, 4)),
    )

    assert pairs == ()
    assert adapter.root_pairs == ()
    assert adapter.root_manifolds == ((0, 1, 2, 3),)
    assert adapter.diagnostics["unresolved_manifolds"] == ((0, 1, 2, 3),)
    assert adapter.diagnostics["raw_ensemble_residual"] <= 1e-12
    assert len(adapter.manifold_results) == 1
    result = adapter.manifold_results[0]
    assert result.diagnostics["basis_invariant_validation"]
    assert np.max(abs(result.dm1 - np.eye(4) / 4)) <= 1e-12
    assert np.max(abs(result.dm2)) == 0.0


def test_numerical_kramers_residual_warns_and_is_recorded():
    time_reversal = np.array(
        [[0.0, -1.0], [1.0, 0.0]], dtype=complex
    )
    dm1s = [
        np.diag([1.0, 0.0]),
        np.diag([0.0, 0.9]),
    ]
    dm2s = [np.zeros((2,) * 4), np.zeros((2,) * 4)]
    adapter = KramersResultAdapter(time_reversal)

    with pytest.warns(RuntimeWarning) as caught:
        pairs = adapter.analyze(np.zeros(2), dm1s, dm2s)

    assert len(pairs) == 1
    assert not adapter.diagnostics["validation_passed"]
    emitted = [str(item.message) for item in caught]
    assert any("raw Kramers manifold residual" in item for item in emitted)
    assert any("raw Kramers partner residual" in item for item in emitted)
    messages = adapter.diagnostics["validation_warnings"]
    assert any("raw Kramers manifold residual" in item for item in messages)
    assert any("raw Kramers partner residual" in item for item in messages)


def test_incomplete_kramers_energy_manifold_warns_and_is_recorded():
    time_reversal = np.array(
        [[0.0, -1.0], [1.0, 0.0]], dtype=complex
    )
    dm1s = [np.diag([1.0, 0.0]), np.diag([0.0, 1.0])]
    dm2s = [np.zeros((2,) * 4), np.zeros((2,) * 4)]
    adapter = KramersResultAdapter(time_reversal, energy_tolerance=1e-8)

    with pytest.warns(RuntimeWarning) as caught:
        pairs = adapter.analyze(np.array([0.0, 2e-8]), dm1s, dm2s)

    assert pairs == ()
    assert adapter.root_manifolds == ((0,), (1,))
    assert not adapter.diagnostics["validation_passed"]
    emitted = [str(item.message) for item in caught]
    assert sum("odd dimension" in item for item in emitted) == 2
    assert any(
        "odd dimension" in item
        for item in adapter.diagnostics["validation_warnings"]
    )


def test_one_body_time_reversal_projection_is_idempotent():
    time_reversal = _nonadjacent_time_reversal()
    rng = np.random.default_rng(991)
    generator = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    generator = generator - generator.T.conj()
    projected = (
        generator + time_reverse_one_body(time_reversal, generator)
    ) * 0.5

    assert np.max(abs(projected + projected.T.conj())) <= 1e-12
    assert np.max(
        abs(projected - time_reverse_one_body(time_reversal, projected))
    ) <= 1e-12


def test_projected_hamiltonian_check_cancels_large_constant_energy():
    time_reversal = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=complex)
    dm1s = [np.diag([1.0, 0.0]), np.diag([0.0, 1.0])]
    dm2s = [np.zeros((2,) * 4), np.zeros((2,) * 4)]
    energies = np.array([-30000.0, -30000.0])
    overlap = np.array(
        [[1.0, 5e-13j], [-5e-13j, 1.0]], dtype=complex
    )
    projected_hamiltonian = overlap * energies[np.newaxis, :]

    adapter = KramersResultAdapter(time_reversal)
    adapter.analyze(
        energies,
        dm1s,
        dm2s,
        overlap=overlap,
        projected_hamiltonian=projected_hamiltonian,
    )
    assert adapter.diagnostics["root_orthogonality_error"] == 5e-13
    assert adapter.diagnostics["projected_hamiltonian_error"] == 0.0


def _dmrg_solver(tmp_path, norb, nelec, nroots, mol=None):
    return DMRGCI(mol).init(
        ncas=norb,
        nelecas=nelec,
        nroots=nroots,
        bond_dims=[32] * 8,
        noises=[0.0] * 8,
        thrds=[1e-20] * 8,
        n_sweeps=8,
        tol=1e-12,
        scratch=tmp_path,
        n_threads=1,
        stack_memory=256,
        dav_max_iter=1000,
        random_seed=2468,
        npdm_site_type=2,
    )


def test_repository_kramers_orbital_order_and_phase():
    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        spin=0,
        verbose=0,
    )
    mf = spinor_hf.KRHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    assert mf.converged

    mapping = identify_kramers_orbitals(
        mol, mf.mo_coeff, mf.get_ovlp(), tolerance=1e-10
    )
    assert mapping.pairs == ((0, 1), (2, 3))
    # The signed forward AO map gives Theta C_0 = +C_1 and
    # Theta C_1 = -C_0 for zquatev's interleaved output.
    assert np.max(abs(np.asarray(mapping.phases) - 1.0)) <= 1e-12
    assert mapping.diagnostics["subspace_closure_error"] <= 1e-12
    assert mapping.diagnostics["partner_orbital_error"] <= 1e-12
    assert mapping.diagnostics["partner_phase_error"] <= 1e-12
    assert mapping.diagnostics["time_reversal_square_error"] <= 1e-12


def test_kramers_subspace_eigh_accepts_nonadjacent_pairs():
    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        spin=0,
        verbose=0,
    )
    mf = spinor_hf.KRHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    assert mf.converged
    orbitals = mf.mo_coeff[:, [0, 2, 1, 3]]
    mapping = identify_kramers_orbitals(
        mol,
        orbitals,
        mf.get_ovlp(),
    )
    rng = np.random.default_rng(478)
    raw = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    matrix = 0.5 * (raw + raw.T.conj())
    matrix = 0.5 * (
        matrix + time_reverse_one_body(mapping.time_reversal, matrix)
    )
    matrix = 0.5 * (matrix + matrix.T.conj())
    mc = zmcscf.CASSCF(mf, ncas=2, nelecas=2, ncore=0)

    eigenvalues, eigenvectors = zmc_superci._kramers_subspace_eigh(
        mc,
        matrix,
        orbitals,
    )
    diagonal = eigenvectors.T.conj().dot(matrix).dot(eigenvectors)
    final_mapping = identify_kramers_orbitals(
        mol,
        orbitals.dot(eigenvectors),
        mf.get_ovlp(),
        tolerance=1e-8,
    )

    assert np.linalg.norm(diagonal - np.diag(eigenvalues)) <= 1e-10
    assert np.max(abs(eigenvalues[::2] - eigenvalues[1::2])) <= 1e-12
    assert final_mapping.diagnostics["partner_orbital_error"] <= 1e-8


@pytest.mark.integration
def test_odd_electron_kramers_pair_exact_dmrg_and_transition(tmp_path):
    norb, nelec, nroots = 4, 3, 2
    time_reversal, h1e, eri = _kramers_hamiltonian()
    h1e_tr, eri_tr = time_reverse_integrals(time_reversal, h1e, eri)
    assert np.max(abs(h1e - h1e_tr)) <= 1e-12
    assert np.max(abs(eri - eri_tr)) <= 1e-12
    assert np.max(abs(h1e.imag)) > 1e-2
    assert np.max(abs(eri.imag)) > 1e-2

    exact = zfci.FCISolver()
    exact.nroots = nroots
    exact_energy, exact_states = exact.kernel(
        h1e, eri, norb, nelec, nroots=nroots, verbose=0
    )
    exact_rdms = [
        exact.make_rdm12(state, norb, nelec) for state in exact_states
    ]
    exact_adapter = KramersResultAdapter(time_reversal)
    exact_pairs = exact_adapter.analyze(
        exact_energy,
        [rdm[0] for rdm in exact_rdms],
        [rdm[1] for rdm in exact_rdms],
    )
    assert exact_adapter.root_pairs == ((0, 1),)

    projected_adapter = KramersResultAdapter(time_reversal, project=True)
    projected_pair = projected_adapter.analyze(
        exact_energy,
        [rdm[0] for rdm in exact_rdms],
        [rdm[1] for rdm in exact_rdms],
    )[0]
    assert projected_pair.diagnostics["projection_applied"]
    assert projected_pair.diagnostics["raw_ensemble_dm1_residual"] <= 1e-12
    assert projected_pair.diagnostics["raw_ensemble_dm2_residual"] <= 1e-12
    assert max(
        kramers_residual(
            time_reversal, projected_pair.dm1, projected_pair.dm2
        ).values()
    ) <= 1e-12

    # Projection has a stricter raw-residual gate than ordinary pairing.  A
    # deliberately damaged root can pass a loose pairing tolerance but must
    # not be "repaired" and presented as a valid Kramers ensemble.
    damaged_dm1 = [np.array(rdm[0], copy=True) for rdm in exact_rdms]
    damaged_dm1[0][0, 0] += 1e-4
    guarded_adapter = KramersResultAdapter(
        time_reversal,
        residual_tolerance=1e-2,
        project=True,
        projection_tolerance=1e-8,
    )
    with pytest.raises(RuntimeError, match="refusing Kramers projection"):
        guarded_adapter.analyze(
            exact_energy,
            damaged_dm1,
            [rdm[1] for rdm in exact_rdms],
        )

    solver = _dmrg_solver(tmp_path, norb, nelec, nroots)
    solver.kramers_restricted(time_reversal)
    dmrg_energy, _ = solver.kernel(
        h1e, eri, norb, nelec, nroots=nroots, verbose=0
    )
    dmrg_dm1, dmrg_dm2 = solver.make_kramers_pair_rdm12()
    dmrg_rdms = [solver.make_rdm12(root, norb, nelec) for root in range(2)]

    energy_error = np.max(abs(dmrg_energy - exact_energy))
    dm1_error = np.max(abs(dmrg_dm1 - exact_pairs[0].dm1))
    dm2_error = np.max(abs(dmrg_dm2 - exact_pairs[0].dm2))
    rdm_energy_errors = [
        abs(energy_from_rdms(h1e, eri, *dmrg_rdms[root]) - dmrg_energy[root])
        for root in range(2)
    ]
    assert energy_error <= 1e-9
    assert dm1_error <= 1e-7
    assert dm2_error <= 1e-7
    assert max(rdm_energy_errors) <= 1e-9
    assert solver.converged
    assert solver.convergence_info["root_strategy"] == "state-averaged-multimps"
    assert solver.convergence_info["effective_dav_type"] == "Normal"
    assert solver.convergence_info["effective_twosite_to_onesite"] == 2
    assert np.max(
        abs(np.asarray(solver.convergence_info["state_average_weights"]) - 0.5)
    ) <= 1e-15
    assert np.max(abs(solver.root_overlap - np.eye(2))) <= 1e-8
    assert np.max(
        abs(solver.projected_hamiltonian - np.diag(dmrg_energy))
    ) <= 1e-8

    diagnostics = solver.kramers_diagnostics
    assert diagnostics["root_pairs"] == ((0, 1),)
    assert diagnostics["root_order"] == (0, 1)
    assert diagnostics["raw_ensemble_residual"] <= 1e-8
    assert diagnostics["projection_change"] == 0.0
    assert not diagnostics["projection_applied"]
    assert diagnostics["root_orthogonality_error"] <= 1e-8
    assert diagnostics["projected_hamiltonian_error"] <= 1e-8
    raw_residual = kramers_residual(
        time_reversal, dmrg_dm1, dmrg_dm2
    )
    assert max(raw_residual.values()) <= 1e-8

    exact_root_space = _exact_root_space(
        exact,
        exact_states,
        [rdm[0] for rdm in exact_rdms],
        norb,
        nelec,
    )
    exact_canonical, _, exact_transition_info = \
        canonicalize_root_space_rdm1(exact_root_space)
    dmrg_canonical, _, dmrg_transition_info = \
        solver.canonical_kramers_root_space_rdm1()
    transition_error = np.max(abs(dmrg_canonical - exact_canonical))
    assert transition_error <= 1e-7
    assert exact_transition_info["root_hermiticity_error"] <= 1e-10
    assert dmrg_transition_info["root_hermiticity_error"] <= 1e-10

    print(
        "kramers-active-space",
        "dE=%.3e" % energy_error,
        "dm1=%.3e" % dm1_error,
        "dm2=%.3e" % dm2_error,
        "transition=%.3e" % transition_error,
        "TR=%.3e" % diagnostics["raw_ensemble_residual"],
        "S=%.3e" % diagnostics["root_orthogonality_error"],
        "PHP=%.3e" % diagnostics["projected_hamiltonian_error"],
    )
    run_scratch = Path(solver._scratch)
    solver.close()
    assert not run_scratch.exists()


@pytest.mark.integration
def test_general_complex_multiroot_path_remains_unrestricted(tmp_path):
    h1e = np.array(
        [
            [-1.3, 0.17 + 0.21j, -0.08j],
            [0.17 - 0.21j, -0.2, 0.13 + 0.04j],
            [0.08j, 0.13 - 0.04j, 0.9],
        ]
    )
    eri = np.zeros((3, 3, 3, 3), dtype=complex)
    exact = zfci.FCISolver()
    exact_energy, exact_states = exact.kernel(
        h1e, eri, 3, 1, nroots=2, verbose=0
    )
    solver = _dmrg_solver(tmp_path, 3, 1, 2)
    solver.weights = np.array([0.7, 0.3])
    dmrg_energy, _ = solver.kernel(h1e, eri, 3, 1, nroots=2, verbose=0)

    assert solver.kramers_adapter is None
    assert solver._multi_mps is not None
    assert solver.convergence_info["root_strategy"] == "state-averaged-multimps"
    assert solver.convergence_info["effective_dav_type"] == "Normal"
    assert solver.convergence_info["effective_twosite_to_onesite"] == 2
    assert np.max(
        abs(
            np.asarray(solver.convergence_info["state_average_weights"])
            - [0.7, 0.3]
        )
    ) <= 1e-15
    assert np.max(abs(np.asarray(solver._multi_mps.weights) - [0.7, 0.3])) <= 1e-15
    assert np.max(abs(dmrg_energy - exact_energy)) <= 1e-9
    for root in range(2):
        exact_dm1, exact_dm2 = exact.make_rdm12(
            exact_states[root], 3, 1
        )
        dmrg_dm1, dmrg_dm2 = solver.make_rdm12(root, 3, 1)
        assert np.max(abs(dmrg_dm1 - exact_dm1)) <= 1e-7
        assert np.max(abs(dmrg_dm2 - exact_dm2)) <= 1e-7
    solver.close()


def test_complex_multimps_supports_explicit_local_eigensolver(tmp_path):
    solver = _dmrg_solver(tmp_path, 3, 1, 2)
    solver.dav_type = "Exact"
    energies, _ = solver.kernel(
        np.diag([-1.0, 0.0, 1.0]),
        np.zeros((3,) * 4),
        3,
        1,
        nroots=2,
        verbose=0,
    )
    assert np.max(abs(energies - [-1.0, 0.0])) <= 1e-10
    assert solver.convergence_info["effective_dav_type"] == "Exact"
    assert solver.convergence_info["root_orthogonality_error"] <= 1e-10
    assert solver.convergence_info["root_eigen_equation_error"] <= 1e-10


def _subspace_projector(mo, overlap, columns):
    eigenvalues, eigenvectors = scipy.linalg.eigh(overlap)
    overlap_half = (eigenvectors * np.sqrt(eigenvalues)).dot(
        eigenvectors.T.conj()
    )
    vectors = overlap_half.dot(mo[:, columns])
    return vectors.dot(vectors.T.conj())


def _tilted_h_kramers_reference():
    mol = gto.M(
        atom="H 0 0 0",
        basis="6-31g",
        spin=1,
        charge=0,
        verbose=0,
        max_memory=1000,
    )
    mf = spinor_hf.KRHF(mol).x2camf(
        with_gaunt=False, with_breit=False
    ).cholesky(tau=1e-10)
    mf.init_guess = "1e"
    mf.conv_tol = 1e-12
    mf.max_cycle = 100
    mf.kernel()
    assert mf.converged

    rotation = np.eye(4, dtype=complex)
    cosine, sine = np.cos(0.18), np.sin(0.18)
    for occupied, virtual in ((0, 2), (1, 3)):
        rotation[occupied, occupied] = cosine
        rotation[virtual, virtual] = cosine
        rotation[occupied, virtual] = sine
        rotation[virtual, occupied] = -sine
    return mol, mf, mf.mo_coeff.dot(rotation)


def _kramers_casscf(mf, initial_mo, fcisolver=None):
    mc = zmcscf.CASSCF(mf, ncas=2, nelecas=1)
    mc.mo_coeff = initial_mo.copy()
    if fcisolver is not None:
        mc.fcisolver = fcisolver
    mc.state_average_([0.5, 0.5])
    # Exercise the repository's Kramers eigensolver for active natural
    # orbitals.  Core/virtual canonicalization is immaterial for this
    # zero-core one-electron gate and is disabled independently.
    mc.natorb = True
    mc.canonicalize_ = False
    mc.max_cycle_macro = 30
    mc.conv_tol = 1e-10
    mc.conv_tol_grad = 1e-7
    mc.max_stepsize = 0.1
    mc.superci_davidson_tol = 1e-11
    mc.superci_davidson_max_space = 20
    mc.superci_davidson_strict = True
    mc.verbose = 0
    return mc


@pytest.mark.integration
def test_kramers_superci_orbital_diis_preserves_pairs():
    mol, mf, initial_mo = _tilted_h_kramers_reference()
    mc = _kramers_casscf(mf, initial_mo)
    mc.natorb = False
    mc.superci(symm="kramers", use_diis=True)

    mapping = identify_kramers_orbitals(
        mol,
        mc.mo_coeff,
        mf.get_ovlp(),
        tolerance=1e-7,
    )
    assert mc.converged
    assert mc.superci_diagnostics["kramers_restricted"]
    assert mc.superci_diagnostics["diis"]
    assert mc.canonicalization_diagnostics["enabled"]
    assert mc.canonicalization_diagnostics["active_orbital_change"] == 0.0
    assert mc.canonicalization_diagnostics["virtual_offdiagonal_after"] <= 1e-10
    assert np.max(abs(mc.mo_energy[0::2] - mc.mo_energy[1::2])) <= 1e-10
    assert mapping.diagnostics["subspace_closure_error"] <= 1e-7
    assert mapping.diagnostics["partner_orbital_error"] <= 1e-7
    assert any(
        row.get("diis", {}).get("extrapolated", False)
        for row in mc.macro_history
    )


@pytest.mark.integration
def test_kramers_supercipt_diis_dmrg_matches_exact(tmp_path):
    mol, mf, initial_mo = _tilted_h_kramers_reference()
    exact = _kramers_casscf(mf, initial_mo)
    exact.natorb = False
    exact.supercipt(symm="kramers", use_diis=True, use_cderi=True)

    base_solver = _dmrg_solver(tmp_path, 2, 1, 2, mol=mol)
    base_solver.kramers_restricted()
    dmrg = _kramers_casscf(mf, initial_mo, base_solver)
    dmrg.natorb = False
    dmrg.callback = dmrg.fcisolver.restart_scheduler_()
    dmrg.supercipt(symm="kramers", use_diis=True, use_cderi=True)

    assert exact.converged and dmrg.converged and dmrg.fcisolver.converged
    assert abs(dmrg.e_tot - exact.e_tot) <= 1e-7
    assert abs(dmrg.e_cas - exact.e_cas) <= 1e-7
    assert abs(
        dmrg.final_orbital_gradient_norm
        - exact.final_orbital_gradient_norm
    ) <= 1e-7
    assert all(
        row["kramers_rotation"]["output_generator_residual"] <= 1e-12
        for row in dmrg.supercipt_history[:-1]
    )
    assert dmrg.fcisolver.kramers_diagnostics[
        "raw_ensemble_residual"
    ] <= 1e-8
    assert (
        dmrg.fcisolver.convergence_info["block2_sweep_tolerance"]
        == dmrg.fcisolver.tol
    )
    assert dmrg.fcisolver.convergence_info[
        "minimal_multiroot_restart_fallback"
    ]
    dmrg.fcisolver.close()


@pytest.mark.integration
def test_kramers_restricted_x2c_dmrg_scf_matches_exact(tmp_path):
    mol, mf, initial_mo = _tilted_h_kramers_reference()
    exact = _kramers_casscf(mf, initial_mo)
    exact.kernel()

    base_solver = _dmrg_solver(tmp_path, 2, 1, 2, mol=mol)
    base_solver.kramers_restricted()
    dmrg = _kramers_casscf(mf, initial_mo, base_solver)
    dmrg.kernel()
    solver = dmrg.fcisolver

    exact_dm1, exact_dm2 = exact.fcisolver.make_rdm12(
        exact.ci, exact.ncas, exact.nelecas
    )
    dmrg_dm1, dmrg_dm2 = solver.make_rdm12(
        dmrg.ci, dmrg.ncas, dmrg.nelecas
    )
    energy_error = abs(dmrg.e_tot - exact.e_tot)
    active_energy_error = abs(dmrg.e_cas - exact.e_cas)
    dm1_error = np.max(abs(dmrg_dm1 - exact_dm1))
    dm2_error = np.max(abs(dmrg_dm2 - exact_dm2))
    gradient_error = abs(
        dmrg.final_orbital_gradient_norm
        - exact.final_orbital_gradient_norm
    )
    trajectory_error = max(
        abs(dmrg_row["total_energy"] - exact_row["total_energy"])
        for exact_row, dmrg_row in zip(exact.macro_history, dmrg.macro_history)
    )

    overlap = mf.get_ovlp()
    active = slice(0, 2)
    virtual = slice(2, 4)
    projector_error = max(
        np.max(
            abs(
                _subspace_projector(dmrg.mo_coeff, overlap, columns)
                - _subspace_projector(exact.mo_coeff, overlap, columns)
            )
        )
        for columns in (active, virtual)
    )

    assert exact.converged and dmrg.converged and solver.converged
    assert exact.final_orbital_gradient_norm <= exact.conv_tol_grad
    assert dmrg.final_orbital_gradient_norm <= dmrg.conv_tol_grad
    assert len(exact.macro_history) == len(dmrg.macro_history)
    assert energy_error <= 1e-7
    assert active_energy_error <= 1e-7
    assert dm1_error <= 1e-7
    assert dm2_error <= 1e-7
    assert gradient_error <= 1e-7
    assert trajectory_error <= 1e-7
    assert projector_error <= 1e-6
    assert np.max(abs(np.asarray(solver.e_states) - exact.fcisolver.e_states)) <= 1e-8
    assert solver.convergence_info["root_strategy"] == "state-averaged-multimps"
    assert solver.kramers_diagnostics["raw_ensemble_residual"] <= 1e-8
    assert solver.kramers_diagnostics["projection_change"] == 0.0
    assert not solver.kramers_diagnostics["projection_applied"]
    assert solver.kramers_diagnostics["root_orthogonality_error"] <= 1e-8
    assert solver.kramers_diagnostics["projected_hamiltonian_error"] <= 1e-8
    assert solver.kramers_adapter.orbital_diagnostics[
        "subspace_closure_error"
    ] <= 1e-8
    assert solver.kramers_adapter.orbital_diagnostics[
        "partner_phase_error"
    ] <= 1e-8
    assert all(
        row["subspace_closure_error"] <= 1e-8
        for row in solver.kramers_adapter.orbital_history
    )
    for row in exact.macro_history[:-1] + dmrg.macro_history[:-1]:
        assert row["kramers_rotation"]["output_generator_residual"] <= 1e-12

    print(
        "kr-x2c-dmrg-scf",
        "Eref=%.15f" % exact.e_tot,
        "Edmrg=%.15f" % dmrg.e_tot,
        "dE=%.3e" % energy_error,
        "dEcas=%.3e" % active_energy_error,
        "dm1=%.3e" % dm1_error,
        "dm2=%.3e" % dm2_error,
        "dgrad=%.3e" % gradient_error,
        "dprojector=%.3e" % projector_error,
        "trajectory=%.3e" % trajectory_error,
        "TR=%.3e" % solver.kramers_diagnostics["raw_ensemble_residual"],
    )
    run_scratch = Path(solver._scratch)
    solver.close()
    assert not run_scratch.exists()
