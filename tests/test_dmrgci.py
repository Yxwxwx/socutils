import json
import math
from pathlib import Path

import numpy as np
from pyscf.fci import fci_dhf_slow

from socutils.dmrg.dmrgci import (
    DMRGCI,
    energy_from_rdms,
    pyscf_dmrg_schedule,
)
from socutils.fci import zfci


ENERGY_TOL = 1e-9
RDM_TOL = 1e-8


def _solver(tmp_path, norb, nelec, nroots=1, bond_dim=32):
    return DMRGCI().init(
        ncas=norb,
        nelecas=nelec,
        nroots=nroots,
        bond_dims=[bond_dim] * 8,
        noises=[0.0] * 8,
        thrds=[1e-14] * 8,
        n_sweeps=8,
        tol=1e-12,
        scratch=tmp_path,
        n_threads=1,
        stack_memory=256,
        dav_max_iter=1000,
        random_seed=2468,
        npdm_site_type=2,
    )


def _assert_rdm_invariants(dm1, dm2, nelec):
    assert abs(np.trace(dm1) - nelec) <= RDM_TOL
    assert np.max(abs(np.einsum("pqrr->pq", dm2) - (nelec - 1) * dm1)) <= RDM_TOL
    assert abs(np.einsum("pprr->", dm2) - nelec * (nelec - 1)) <= RDM_TOL
    assert np.max(abs(dm1 - dm1.T.conj())) <= RDM_TOL
    assert np.max(abs(dm2.conj() - dm2.transpose(1, 0, 3, 2))) <= RDM_TOL
    assert np.max(abs(dm2 + dm2.transpose(2, 1, 0, 3))) <= RDM_TOL
    assert np.max(abs(dm2 + dm2.transpose(0, 3, 2, 1))) <= RDM_TOL


def test_pyscf_schedule_is_expanded_for_direct_pyblock2():
    schedule = pyscf_dmrg_schedule(
        max_bond_dimension=1000,
        start_bond_dimension=200,
        tol=1e-7,
    )

    # These are the anchor rows produced by the official PySCF loop.  The
    # extra 1e-7 row reflects its literal repeated floating-point division.
    assert schedule.anchor_sweeps == (0, 4, 8, 12, 14, 16, 18, 20)
    assert schedule.anchor_bond_dims == (
        200,
        400,
        800,
        1000,
        1000,
        1000,
        1000,
        1000,
    )
    assert np.allclose(
        schedule.anchor_thrds,
        [1e-4, 1e-4, 1e-4, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
        rtol=1e-14,
        atol=0.0,
    )
    assert schedule.n_sweeps == 32
    assert schedule.twosite_to_onesite == 24
    assert schedule.bond_dims[:4] == (200,) * 4
    assert schedule.bond_dims[4:8] == (400,) * 4
    assert schedule.noises[20:] == (0.0,) * 12

    restart = pyscf_dmrg_schedule(
        max_bond_dimension=1000, tol=1e-7, restart=True
    )
    assert restart.n_sweeps == 8
    assert restart.twosite_to_onesite is None
    assert restart.bond_dims == (1000,) * 8
    assert restart.noises == (0.0,) * 8
    assert restart.thrds == (1e-8,) * 8

    tuned = pyscf_dmrg_schedule(
        max_bond_dimension=32,
        start_bond_dimension=16,
        tol=1e-10,
        noise_scale=0.5,
        max_davidson_threshold=1e-12,
    )
    assert np.isclose(max(tuned.thrds), 1e-8)
    assert np.isclose(min(tuned.thrds), 1e-12)
    assert tuned.anchor_noises[0] == 5e-5
    assert tuned.anchor_noises[-1] == 0.0


def test_tight_davidson_threshold_is_staged_independently_of_noise():
    schedule = pyscf_dmrg_schedule(
        max_bond_dimension=1000,
        start_bond_dimension=200,
        tol=1e-7,
        max_davidson_threshold=1e-16,
    )

    assert np.allclose(
        schedule.anchor_thrds,
        [1e-8] * 4
        + [1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15, 1e-16],
        rtol=1e-14,
        atol=0.0,
    )
    assert schedule.anchor_noises[-1] == 0.0
    assert schedule.noises[-8:] == (0.0,) * 8
    assert len(set(schedule.thrds)) > 1


def test_relativistic_solver_defaults_to_tight_davidson_schedule():
    solver = DMRGCI()

    assert solver.schedule_mode == "pyscf"
    assert solver.max_bond_dimension == 1000
    assert solver.tol == 1e-8
    assert solver.schedule_thrd_max == 1e-16
    assert np.isclose(max(solver.schedule_thrds), 1e-8)
    assert np.isclose(min(solver.schedule_thrds), 1e-16)
    assert len(set(solver.thrds)) > 1
    assert np.allclose(
        solver._schedule_snapshot(restart=True).thrds,
        (1e-16,) * 8,
        rtol=1e-14,
        atol=0.0,
    )


def test_restart_scheduler_accepts_both_callback_vocabularies():
    solver = DMRGCI()
    callback = solver.restart_scheduler_()

    assert not callback({"orbital_gradient_norm": 2e-3})
    assert callback({"orbital_gradient_norm": 5e-4})
    assert solver.restart_diagnostics["reasons"] == ["orbital_gradient"]
    assert callback({"norm_gorb": 2e-3, "norm_ddm": 5e-3})
    assert solver.restart_diagnostics["reasons"] == ["density_change"]
    assert not callback({"norm_gorb": 2e-3, "norm_ddm": 2e-2})


def test_analytic_one_electron_and_determinant_rdms(tmp_path):
    h1 = np.array(
        [
            [-1.0, 0.2 + 0.3j, 0.0],
            [0.2 - 0.3j, 0.7, -0.1j],
            [0.0, 0.1j, 1.4],
        ]
    )
    eri1 = np.zeros((3,) * 4, dtype=complex)
    eig, coeff = np.linalg.eigh(h1)
    expected_dm1 = np.outer(coeff[:, 0].conj(), coeff[:, 0])
    solver = _solver(tmp_path, 3, 1, bond_dim=8)
    energy, state = solver.kernel(h1, eri1, 3, 1, ecore=0.31, verbose=0)
    dm1, dm2 = solver.make_rdm12(state, 3, 1)
    run_scratch = Path(solver._scratch)

    assert abs(energy - (eig[0] + 0.31)) <= ENERGY_TOL
    assert np.max(abs(dm1 - expected_dm1)) <= RDM_TOL
    assert np.max(abs(dm2)) <= RDM_TOL
    _assert_rdm_invariants(dm1, dm2, 1)
    assert abs(energy_from_rdms(h1, eri1, dm1, dm2, 0.31) - energy) <= ENERGY_TOL
    assert solver.converged
    solver.close()
    assert not run_scratch.exists()

    h2 = np.diag([-2.0, -1.0, 1.5]).astype(complex)
    eri2 = np.zeros((3,) * 4, dtype=complex)
    expected_dm1 = np.diag([1.0, 1.0, 0.0]).astype(complex)
    expected_dm2 = (
        np.einsum("pq,rs->pqrs", expected_dm1, expected_dm1)
        - np.einsum("ps,rq->pqrs", expected_dm1, expected_dm1)
    )
    solver = _solver(tmp_path, 3, 2, bond_dim=8)
    energy, state = solver.kernel(h2, eri2, 3, 2, verbose=0)
    dm1, dm2 = solver.make_rdm12(state, 3, 2)

    assert abs(energy + 3.0) <= ENERGY_TOL
    assert np.max(abs(dm1 - expected_dm1)) <= RDM_TOL
    assert np.max(abs(dm2 - expected_dm2)) <= RDM_TOL
    assert abs(dm2[0, 0, 1, 1] - 1.0) <= RDM_TOL
    assert abs(dm2[0, 1, 1, 0] + 1.0) <= RDM_TOL
    _assert_rdm_invariants(dm1, dm2, 2)
    solver.close()


def test_large_ecore_is_excluded_from_sweep_convergence(tmp_path):
    """A heavy-atom core shift must not consume active-energy precision."""
    h1 = np.diag([-1.25, 0.4, 1.1]).astype(complex)
    eri = np.zeros((3,) * 4, dtype=complex)
    ecore = -1.0e8
    solver = _solver(tmp_path, 3, 1, bond_dim=8)
    energy, state = solver.kernel(h1, eri, 3, 1, ecore=ecore, verbose=0)
    dm1, dm2 = solver.make_rdm12(state, 3, 1)

    assert abs(energy - (ecore - 1.25)) <= ENERGY_TOL
    assert abs(energy_from_rdms(h1, eri, dm1, dm2, ecore) - energy) <= ENERGY_TOL
    assert solver.converged
    assert solver.convergence_info["constant_energy_shift"] == ecore
    assert solver.convergence_info["sweep_energy_origin"] == (
        "active-space Hamiltonian without ecore"
    )
    assert np.max(abs(np.asarray(solver.convergence_info["sweep_energies"]))) < 2
    solver.close()


def test_fiedler_reordering_restores_original_rdm_indices(tmp_path, monkeypatch):
    """A nontrivial Block2 site permutation must be invisible to PySCF."""
    from pyblock2.driver.core import DMRGDriver

    h1, eri, h1_rot, eri_rot, _ = _complex_hamiltonians()
    norb, nelec = h1.shape[0], 2
    reorder_idx = np.array([2, 0, 3, 1])
    calls = []

    def fixed_reordering(driver, h1e, g2e, method="fiedler", **kwargs):
        calls.append((np.array(h1e, copy=True), np.array(g2e, copy=True)))
        assert method == "fiedler"
        assert np.min(h1e) >= 0.0
        assert np.min(g2e) >= 0.0
        return reorder_idx.copy()

    monkeypatch.setattr(
        DMRGDriver, "orbital_reordering", fixed_reordering
    )
    reference = zfci.FCISolver()
    energy_ref, ci_ref = reference.kernel(
        h1_rot, eri_rot, norb, nelec
    )
    dm1_ref, dm2_ref = reference.make_rdm12(ci_ref, norb, nelec)

    solver = _solver(tmp_path, norb, nelec)
    energy, state = solver.kernel(
        h1_rot, eri_rot, norb, nelec, verbose=0
    )
    dm1, dm2 = solver.make_rdm12(state, norb, nelec)

    assert len(calls) == 1
    assert np.array_equal(solver.driver.reorder_idx, reorder_idx)
    assert solver.convergence_info["orbital_reordering"] == reorder_idx.tolist()
    assert abs(energy - energy_ref) <= ENERGY_TOL
    assert np.max(abs(dm1 - dm1_ref)) <= RDM_TOL
    assert np.max(abs(dm2 - dm2_ref)) <= RDM_TOL
    assert abs(energy_from_rdms(h1_rot, eri_rot, dm1, dm2) - energy) <= ENERGY_TOL
    solver.close()


def test_casscf_restart_reuses_only_compatible_internal_mps(
    tmp_path, monkeypatch
):
    from pyblock2.driver.core import DMRGDriver

    proposed_reorderings = iter(
        (np.array([2, 0, 1]), np.array([1, 2, 0]))
    )

    def changing_reordering(driver, h1e, g2e, method="fiedler", **kwargs):
        return next(proposed_reorderings).copy()

    monkeypatch.setattr(
        DMRGDriver, "orbital_reordering", changing_reordering
    )
    h1 = np.array(
        [
            [-1.1, 0.08j, 0.02],
            [-0.08j, -0.3, -0.04j],
            [0.02, 0.04j, 0.7],
        ],
        dtype=complex,
    )
    eri = np.zeros((3,) * 4, dtype=complex)
    solver = _solver(tmp_path, 3, 1, bond_dim=8)
    energy0, state0 = solver.kernel(h1, eri, 3, 1, verbose=0)
    solver.make_rdm12(state0, 3, 1)
    assert solver.convergence_info["block2_sweep_tolerance"] == solver.tol
    reorder0 = np.array(solver.driver.reorder_idx, copy=True)
    driver_id = id(solver.driver)
    scratch = solver._scratch

    h1_next = h1.copy()
    h1_next[1, 1] -= 0.03
    solver.restart_scheduler_step({"orbital_gradient_norm": 5e-4})
    energy1, _ = solver.kernel(
        h1_next, eri, 3, 1, ci0=state0, verbose=0
    )
    reference, _ = zfci.FCISolver().kernel(h1_next, eri, 3, 1)

    assert abs(energy0 - np.linalg.eigvalsh(h1)[0]) <= ENERGY_TOL
    assert abs(energy1 - reference) <= ENERGY_TOL
    assert id(solver.driver) != driver_id
    assert solver._scratch != scratch
    assert solver.convergence_info["run_mode"] == "casscf-warm-start"
    assert solver.convergence_info["restart_transport"] == (
        "fresh-driver-mps-reload"
    )
    assert solver.convergence_info["schedule"]["restart"]
    assert solver.convergence_info["schedule"]["n_sweeps"] == 8
    assert solver.convergence_info["block2_sweep_tolerance"] == 0.0
    assert solver.convergence_info["sweeps"] == 8
    assert solver._multi_mps.dot == 1
    assert np.array_equal(reorder0, [2, 0, 1])
    assert np.array_equal(solver.driver.reorder_idx, reorder0)
    assert solver.convergence_info["orbital_reordering"] == reorder0.tolist()
    solver.close()


def test_multiroot_checkpoint_resume_and_fingerprint_gate(tmp_path):
    h1 = np.diag([-1.3, -0.4, 0.8]).astype(complex)
    eri = np.zeros((3,) * 4, dtype=complex)
    checkpoint = tmp_path / "checkpoint"
    first = DMRGCI().init(
        ncas=3,
        nelecas=1,
        nroots=2,
        bond_dims=[8] * 8,
        noises=[0.0] * 8,
        thrds=[1e-14] * 8,
        n_sweeps=8,
        tol=1e-12,
        scratch=tmp_path / "scratch-first",
        checkpoint_dir=checkpoint,
        n_threads=1,
        stack_memory=256,
        random_seed=2468,
    )
    energy0, _ = first.kernel(h1, eri, 3, 1, verbose=0)
    manifest_path = checkpoint / "dmrgci-checkpoint.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "complete"
    assert manifest["orbital_reordering"] == first.convergence_info[
        "orbital_reordering"
    ]
    assert (checkpoint / "mps" / "GS-mps_info.bin").is_file()
    # A killed process leaves status=running; the last completed sweep is
    # nevertheless a valid pyblock2 restart image.
    manifest["status"] = "running"
    manifest_path.write_text(json.dumps(manifest))
    first.close()

    resumed = DMRGCI().init(
        ncas=3,
        nelecas=1,
        nroots=2,
        bond_dims=[8] * 8,
        noises=[0.0] * 8,
        thrds=[1e-14] * 8,
        n_sweeps=8,
        tol=1e-12,
        scratch=tmp_path / "scratch-resumed",
        checkpoint_dir=checkpoint,
        resume=True,
        n_threads=1,
        stack_memory=256,
        random_seed=999,
    )
    energy1, _ = resumed.kernel(h1, eri, 3, 1, verbose=0)
    assert np.max(abs(energy1 - energy0)) <= ENERGY_TOL
    assert resumed.convergence_info["run_mode"] == "checkpoint-resume"
    assert resumed.convergence_info["schedule"]["restart"]
    assert resumed.convergence_info["block2_sweep_tolerance"] == 0.0
    assert resumed.convergence_info["sweeps"] == 8
    assert np.array_equal(
        resumed.driver.reorder_idx, manifest["orbital_reordering"]
    )
    assert not resumed.resume
    resumed.close()

    mismatched = DMRGCI().init(
        ncas=3,
        nelecas=1,
        nroots=2,
        bond_dims=[8] * 8,
        noises=[0.0] * 8,
        thrds=[1e-14] * 8,
        n_sweeps=8,
        tol=1e-12,
        scratch=tmp_path / "scratch-mismatch",
        checkpoint_dir=checkpoint,
        resume=True,
        n_threads=1,
        stack_memory=256,
    )
    h1_mismatch = h1.copy()
    h1_mismatch[0, 0] += 1e-6
    with np.testing.assert_raises_regex(ValueError, "fingerprint"):
        mismatched.kernel(h1_mismatch, eri, 3, 1, verbose=0)
    mismatched.close()


def _complex_hamiltonians():
    rng = np.random.default_rng(8128)
    norb = 4
    x = rng.normal(size=(norb, norb))
    h1 = x + x.T
    pair = rng.normal(size=(norb * norb, norb * norb))
    pair = (pair + pair.T).reshape((norb,) * 4)
    eri = 0.25 * (
        pair
        + pair.transpose(1, 0, 3, 2)
        + pair.transpose(2, 3, 0, 1)
        + pair.transpose(3, 2, 1, 0)
    )
    z = rng.normal(size=(norb, norb)) + 1j * rng.normal(size=(norb, norb))
    unitary = np.linalg.qr(z)[0]
    h1_rot = np.einsum("ap,ab,bq->pq", unitary.conj(), h1, unitary)
    eri_rot = np.einsum(
        "ap,bq,cr,ds,abcd->pqrs",
        unitary.conj(),
        unitary,
        unitary.conj(),
        unitary,
        eri,
        optimize=True,
    )
    return h1, eri, h1_rot, eri_rot, unitary


def test_complex_unitary_covariance_multiroot_and_transition_rdm(tmp_path):
    norb, nelec, nroots = 4, 2, 2
    h1, eri, h1_rot, eri_rot, unitary = _complex_hamiltonians()
    ecore = 0.137

    reference = zfci.FCISolver()
    energy_ref0, ci_ref0 = reference.kernel(h1, eri, norb, nelec, ecore=ecore)
    dm1_ref0, dm2_ref0 = reference.make_rdm12(ci_ref0, norb, nelec)
    reference.nroots = nroots
    energy_ref, ci_ref = reference.kernel(
        h1_rot, eri_rot, norb, nelec, ecore=ecore
    )
    energy_dhf, ci_dhf = fci_dhf_slow.kernel(
        h1_rot, eri_rot, norb, nelec, ecore=ecore,
        nroots=nroots, verbose=0,
    )
    assert np.max(abs(np.asarray(energy_dhf) - energy_ref)) <= ENERGY_TOL
    for root in range(nroots):
        dhf1, dhf2 = fci_dhf_slow.make_rdm12(
            ci_dhf[root], norb, nelec
        )
        ref1, ref2 = reference.make_rdm12(ci_ref[root], norb, nelec)
        assert np.max(abs(dhf1 - ref1)) <= RDM_TOL
        assert np.max(abs(dhf2 - ref2)) <= RDM_TOL

    predicted_dm1 = np.einsum(
        "ap,bq,ab->pq", unitary, unitary.conj(), dm1_ref0
    )
    predicted_dm2 = np.einsum(
        "ap,bq,cr,ds,abcd->pqrs",
        unitary,
        unitary.conj(),
        unitary,
        unitary.conj(),
        dm2_ref0,
        optimize=True,
    )
    dm1_ref, dm2_ref = reference.make_rdm12(ci_ref[0], norb, nelec)
    assert abs(energy_ref[0] - energy_ref0) <= ENERGY_TOL
    assert np.max(abs(dm1_ref - predicted_dm1)) <= RDM_TOL
    assert np.max(abs(dm2_ref - predicted_dm2)) <= RDM_TOL
    assert np.max(abs(h1_rot.imag)) > 1e-3
    assert np.max(abs(eri_rot.imag)) > 1e-3

    solver = _solver(tmp_path, norb, nelec, nroots=nroots)
    energy_dmrg, states = solver.kernel(
        h1_rot, eri_rot, norb, nelec, ecore=ecore, verbose=0
    )
    assert isinstance(energy_dmrg, np.ndarray)
    assert energy_dmrg.shape == (nroots,)
    assert isinstance(states, list) and len(states) == nroots
    assert np.max(abs(energy_dmrg - energy_ref)) <= ENERGY_TOL
    assert np.max(abs(solver.e_cas - (energy_ref - ecore))) <= ENERGY_TOL

    for root in range(nroots):
        ref1, ref2 = reference.make_rdm12(ci_ref[root], norb, nelec)
        dm1, dm2 = solver.make_rdm12(root, norb, nelec)
        dm1_mps, dm2_mps = solver.make_rdm12(states[root], norb, nelec)
        assert np.iscomplexobj(dm1) and np.iscomplexobj(dm2)
        assert np.max(abs(dm1 - ref1)) <= RDM_TOL
        assert np.max(abs(dm2 - ref2)) <= RDM_TOL
        assert np.max(abs(dm1_mps - dm1)) == 0.0
        assert np.max(abs(dm2_mps - dm2)) == 0.0
        assert (
            abs(energy_from_rdms(h1_rot, eri_rot, dm1, dm2, ecore) - energy_dmrg[root])
            <= ENERGY_TOL
        )
        _assert_rdm_invariants(dm1, dm2, nelec)
        diagnostics = solver.rdm_diagnostics[id(states[root])]
        assert diagnostics["projection_change"] == 0.0
        assert diagnostics["contraction_error"] <= RDM_TOL

    transition_ref = reference.trans_rdm1(
        ci_ref[0], ci_ref[1], norb, nelec
    )
    transition_dmrg = solver.trans_rdm1(0, 1, norb, nelec)
    reverse_dmrg = solver.trans_rdm1(1, 0, norb, nelec)
    phase = np.vdot(transition_dmrg, transition_ref)
    phase /= abs(phase)
    assert np.max(abs(phase * transition_dmrg - transition_ref)) <= RDM_TOL
    assert np.max(abs(reverse_dmrg - transition_dmrg.T.conj())) <= RDM_TOL

    print(
        "complex-contract",
        "dE=%.3e" % np.max(abs(energy_dmrg - energy_ref)),
        "dm1=%.3e" % max(
            np.max(abs(solver.make_rdm1(root, norb, nelec)
                       - reference.make_rdm1(ci_ref[root], norb, nelec)))
            for root in range(nroots)
        ),
        "dm2=%.3e" % max(
            np.max(abs(solver.make_rdm2(root, norb, nelec)
                       - reference.make_rdm2(ci_ref[root], norb, nelec)))
            for root in range(nroots)
        ),
        "tdm1=%.3e" % np.max(abs(phase * transition_dmrg - transition_ref)),
    )

    info = solver.convergence_info
    assert info["converged"]
    assert info["energy_change"] <= solver.tol
    assert info["discarded_weight"] <= RDM_TOL
    assert info["bond_dimension"] == 32
    assert info["npdm_site_type"] == 2
    assert math.isclose(info["local_residual_bound"], 1e-7)
    run_scratch = Path(solver._scratch)
    solver.close()
    assert not run_scratch.exists()
