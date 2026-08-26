import math
from pathlib import Path

import numpy as np
from pyscf.fci import fci_dhf_slow

from socutils.dmrg.dmrgci import DMRGCI, energy_from_rdms
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
