from pathlib import Path

import numpy as np
import pytest
import scipy.linalg
from pyscf import gto

from socutils.dmrg.dmrgci import DMRGCI
from socutils.mcscf import zmcscf
from socutils.scf import spinor_hf


def test_dmrg_casscf_production_defaults():
    mol = gto.M(
        atom='H 0 0 0',
        basis='sto-3g',
        spin=1,
        verbose=0,
    )
    mc = zmcscf.CASSCF(mol, ncas=2, nelecas=1)

    assert mc.max_cycle_macro == 50
    assert mc.max_stepsize == 0.2
    assert mc.conv_tol == 1e-8
    assert mc.conv_tol_grad == 1e-4
    assert not mc.natorb
    assert not mc.canonicalize_
    assert mc.canonicalization
    assert mc.canonicalization_diagnostics is None
    assert mc.superci_davidson_tol == 1e-8
    assert mc.superci_davidson_max_space == 200
    assert mc.superci_davidson_strict


@pytest.fixture(scope='module')
def be_x2c_reference():
    mol = gto.M(
        atom='Be 0 0 0',
        basis='sto-3g',
        spin=0,
        charge=0,
        verbose=0,
        max_memory=1000,
    )
    mf = spinor_hf.SCF(mol).x2camf(
        with_gaunt=False, with_breit=False).cholesky(tau=1e-10)
    mf.init_guess = '1e'
    mf.conv_tol = 1e-11
    mf.max_cycle = 100
    mf.kernel()
    assert mf.converged
    return mol, mf, mf.mo_coeff.copy()


def _make_casscf(mf, initial_mo, natorb, fcisolver=None):
    mc = zmcscf.CASSCF(mf, ncas=4, nelecas=2)
    mc.mo_coeff = initial_mo.copy()
    if fcisolver is not None:
        mc.fcisolver = fcisolver
    mc.natorb = natorb
    mc.canonicalize_ = natorb
    mc.max_cycle_macro = 20
    mc.conv_tol = 1e-9
    mc.conv_tol_grad = 1e-5
    mc.max_stepsize = 0.1
    mc.superci_davidson_tol = 1e-10
    mc.superci_davidson_max_space = 20
    mc.superci_davidson_strict = True
    mc.verbose = 0
    return mc


def _subspace_projector(mo, overlap, columns):
    eigenvalues, eigenvectors = scipy.linalg.eigh(overlap)
    overlap_half = (eigenvectors * np.sqrt(eigenvalues)).dot(
        eigenvectors.T.conj())
    vectors = overlap_half.dot(mo[:, columns])
    return vectors.dot(vectors.T.conj())


@pytest.mark.integration
def test_superci_orbital_diis_reaches_unaccelerated_solution(be_x2c_reference):
    """Exercise orbital DIIS on the ordinary complex-spinor Super-CI path."""
    _, mf, initial_mo = be_x2c_reference
    reference = _make_casscf(mf, initial_mo, natorb=False)
    reference.max_cycle_macro = 30
    reference.kernel()

    accelerated = _make_casscf(mf, initial_mo, natorb=False)
    accelerated.max_cycle_macro = 30
    accelerated.superci(use_diis=True)

    overlap = mf.get_ovlp()
    assert reference.converged and accelerated.converged
    assert accelerated.superci_diagnostics['diis']
    assert any(
        row.get('diis', {}).get('extrapolated', False)
        for row in accelerated.macro_history
    )
    assert accelerated.final_orbital_gradient_norm <= accelerated.conv_tol_grad
    assert abs(accelerated.e_tot - reference.e_tot) <= 1e-8
    assert np.max(
        abs(
            accelerated.mo_coeff.T.conj()
            .dot(overlap)
            .dot(accelerated.mo_coeff)
            - np.eye(accelerated.mo_coeff.shape[1])
        )
    ) <= 1e-9


@pytest.mark.integration
@pytest.mark.parametrize('natorb', [False, True], ids=['fixed-active', 'natorb'])
def test_cholesky_x2c_dmrg_scf_matches_exact(
        be_x2c_reference, tmp_path, natorb):
    mol, mf, initial_mo = be_x2c_reference
    exact = _make_casscf(mf, initial_mo, natorb)
    exact.kernel()
    exact_dm1 = exact.fcisolver.make_rdm1(
        exact.ci, exact.ncas, exact.nelecas)

    solver = DMRGCI(mol).init(
        ncas=4,
        nelecas=2,
        nroots=1,
        bond_dims=[32] * 8,
        noises=[0.0] * 8,
        thrds=[1e-20] * 8,
        n_sweeps=8,
        tol=1e-12,
        scratch=tmp_path / ('natorb' if natorb else 'fixed-active'),
        n_threads=1,
        stack_memory=256,
        dav_max_iter=1000,
        random_seed=2468,
        npdm_site_type=2,
    )
    dmrg = _make_casscf(mf, initial_mo, natorb, solver)
    dmrg.kernel()
    dmrg_dm1 = solver.make_rdm1(dmrg.ci, dmrg.ncas, dmrg.nelecas)

    exact_occ = np.linalg.eigvalsh(exact_dm1)[::-1]
    dmrg_occ = np.linalg.eigvalsh(dmrg_dm1)[::-1]
    energy_error = abs(dmrg.e_tot - exact.e_tot)
    active_energy_error = abs(dmrg.e_cas - exact.e_cas)
    occupation_error = np.max(abs(dmrg_occ - exact_occ))
    gradient_error = abs(
        dmrg.final_orbital_gradient_norm -
        exact.final_orbital_gradient_norm)

    overlap = mf.get_ovlp()
    ncore = exact.ncore
    nocc = ncore + exact.ncas
    spaces = {
        'core': slice(0, ncore),
        'active': slice(ncore, nocc),
        'virtual': slice(nocc, initial_mo.shape[1]),
    }
    projector_errors = {}
    for name, columns in spaces.items():
        exact_projector = _subspace_projector(
            exact.mo_coeff, overlap, columns)
        dmrg_projector = _subspace_projector(
            dmrg.mo_coeff, overlap, columns)
        projector_errors[name] = np.max(
            abs(dmrg_projector - exact_projector))

    assert exact.converged and dmrg.converged and solver.converged
    assert exact.superci_diagnostics['converged']
    assert dmrg.superci_diagnostics['converged']
    assert exact.final_orbital_gradient_norm <= exact.conv_tol_grad
    assert dmrg.final_orbital_gradient_norm <= dmrg.conv_tol_grad
    assert energy_error <= 1e-7
    assert active_energy_error <= 1e-7
    assert occupation_error <= 1e-6
    assert gradient_error <= 1e-6
    assert max(projector_errors.values()) <= 1e-6
    assert exact.cholesky_diagnostics['active']
    assert dmrg.cholesky_diagnostics['active']
    assert exact.cholesky_diagnostics['threshold'] == 1e-10
    assert dmrg.cholesky_diagnostics['threshold'] == 1e-10
    for calculation in (exact, dmrg):
        canonicalization = calculation.canonicalization_diagnostics
        assert canonicalization['enabled']
        assert canonicalization['ci_object_preserved']
        assert canonicalization['active_orbital_change'] == 0.0
        assert canonicalization['core_density_change'] <= 1e-9
        assert canonicalization['virtual_projector_change'] <= 1e-9
        assert canonicalization['orthonormality_error'] <= 1e-9
        assert canonicalization['energy_diagonal_error'] <= 1e-9
        assert canonicalization['core_offdiagonal_after'] <= 1e-9
        assert canonicalization['virtual_offdiagonal_after'] <= 1e-9
        assert calculation.mo_energy.shape == (initial_mo.shape[1],)
        assert np.all(np.isfinite(calculation.mo_energy))
        assert (
            calculation.superci_diagnostics['canonicalization']
            == canonicalization
        )

    assert len(exact.macro_history) == len(dmrg.macro_history)
    trajectory_error = max(
        abs(drow['total_energy'] - erow['total_energy'])
        for erow, drow in zip(exact.macro_history, dmrg.macro_history))
    trajectory_gradient_error = max(
        abs(drow['orbital_gradient_norm'] - erow['orbital_gradient_norm'])
        for erow, drow in zip(exact.macro_history, dmrg.macro_history))
    assert trajectory_error <= 1e-7
    assert trajectory_gradient_error <= 1e-6
    assert exact.macro_history[-1]['converged']
    assert dmrg.macro_history[-1]['converged']
    assert all(row['ci_solver_converged'] for row in dmrg.macro_history)
    for row in exact.macro_history[:-1] + dmrg.macro_history[:-1]:
        linear = row['linear_solver']
        assert linear['solver'] == 'davidson'
        assert linear['converged']
        assert linear['residual_norm'] <= 1e-10

    print(
        'x2c-dmrg-scf',
        'natorb=%s' % natorb,
        'naux=%d' % dmrg.cholesky_diagnostics['naux'],
        'Eref=%.15f' % exact.e_tot,
        'Edmrg=%.15f' % dmrg.e_tot,
        'dE=%.3e' % energy_error,
        'dEcas=%.3e' % active_energy_error,
        'dgrad=%.3e' % gradient_error,
        'docc=%.3e' % occupation_error,
        'dprojector=%.3e' % max(projector_errors.values()),
        'trajectory=%.3e' % trajectory_error,
    )
    run_scratch = Path(solver._scratch)
    solver.close()
    assert not run_scratch.exists()
