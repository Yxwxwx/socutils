from pathlib import Path

import numpy as np
import pytest
import scipy.linalg
from pyscf import gto, scf

from socutils.dmrg.dmrgci import DMRGCI, energy_from_rdms
from socutils.mcscf import zmc_ao2mo, zmc_supercipt, zmcscf
from socutils.scf import spinor_hf


@pytest.fixture(scope="module")
def tilted_hf_supercipt():
    """A physical complex-X2C case with all three orbital blocks active."""
    mol = gto.M(
        atom="H 0 0 0; F 0.35 0.27 0.8035",
        basis="sto-3g",
        spin=0,
        charge=0,
        verbose=0,
        max_memory=1000,
    )
    mf = spinor_hf.SCF(mol).x2camf(
        with_gaunt=False, with_breit=False
    )
    mf.init_guess = "1e"
    mf.conv_tol = 1e-11
    mf.max_cycle = 100
    mf.kernel()
    assert mf.converged

    mo = mf.mo_coeff.copy()
    kappa = np.zeros((mo.shape[1], mo.shape[1]), dtype=complex)
    kappa[8, 7] = 0.04 + 0.02j       # core/active
    kappa[11, 10] = -0.035 + 0.015j  # active/virtual
    kappa[11, 7] = 0.02 - 0.01j      # core/virtual
    kappa -= kappa.T.conj()
    mo = mo.dot(scipy.linalg.expm(kappa))
    mf_cd = mf.cholesky(tau=1e-10)
    mf_cd.mo_coeff = mf.mo_coeff.copy()
    return mol, mf, mf_cd, mo


def _casscf(mf, mo, solver=None):
    mc = zmcscf.CASSCF(mf, ncas=3, nelecas=2)
    mc.mo_coeff = mo.copy()
    if solver is not None:
        mc.fcisolver = solver
    mc.natorb = False
    mc.canonicalize_ = True
    mc.max_cycle_macro = 24
    mc.conv_tol = 1e-10
    mc.conv_tol_grad = 1e-5
    mc.max_stepsize = 0.2
    mc.verbose = 0
    return mc


def _dmrg(mol, scratch):
    return DMRGCI(mol).init(
        ncas=3,
        nelecas=2,
        nroots=1,
        bond_dims=[16] * 8,
        noises=[0.0] * 8,
        thrds=[1e-20] * 8,
        n_sweeps=8,
        tol=1e-12,
        scratch=scratch,
        n_threads=1,
        stack_memory=256,
        dav_max_iter=1000,
        random_seed=2468,
        npdm_site_type=2,
    )


def _projector(mo, overlap, columns):
    eigenvalues, eigenvectors = scipy.linalg.eigh(overlap)
    overlap_half = (eigenvectors * np.sqrt(eigenvalues)).dot(
        eigenvectors.T.conj()
    )
    vectors = overlap_half.dot(mo[:, columns])
    return vectors.dot(vectors.T.conj())


def test_supercipt_metric_eigenproblem_truncates_null_space():
    matrix = np.diag([2.0, 1.5, 7.0]).astype(complex)
    metric = np.diag([1.0, 0.5, 1e-9]).astype(complex)
    values, vectors, diagnostics = (
        zmc_supercipt.solve_metric_eigenproblem(
            matrix, metric, metric_tol=1e-6
        )
    )

    assert diagnostics["rank"] == 2
    assert diagnostics["dimension"] == 3
    assert np.max(abs(values - [2.0, 3.0])) <= 1e-12
    assert np.max(
        abs(vectors.T.conj().dot(metric).dot(vectors) - np.eye(2))
    ) <= 1e-12
    assert diagnostics["orthonormality_error"] <= 1e-12

    indefinite = metric.copy()
    indefinite[2, 2] = -1e-3
    with pytest.raises(ValueError, match="not positive semidefinite"):
        zmc_supercipt.solve_metric_eigenproblem(matrix, indefinite)


@pytest.mark.integration
def test_supercipt_fixed_orbital_block2_matches_exact(
    tilted_hf_supercipt, tmp_path
):
    mol, mf, mf_cd, mo = tilted_hf_supercipt
    mc = _casscf(mf, mo)
    eris = zmc_ao2mo._ERIS(mc, mo.copy(), level=2)
    mci = zmcscf._fake_h_for_fast_casci(mc, mo.copy(), eris)
    h1eff, ecore = mci.get_h1eff(mo)
    exact_energy, _, exact_ci = mci.kernel(mo, verbose=0)
    exact_dm1, exact_dm2 = mc.fcisolver.make_rdm12(
        exact_ci, mc.ncas, mc.nelecas
    )

    solver = _dmrg(mol, tmp_path / "fixed")
    dmrg_energy, dmrg_ci = solver.kernel(
        h1eff,
        eris.aaaa,
        mc.ncas,
        mc.nelecas,
        ecore=ecore,
        verbose=0,
    )
    dmrg_dm1, dmrg_dm2 = solver.make_rdm12(
        dmrg_ci, mc.ncas, mc.nelecas
    )
    exact_quantities = zmc_supercipt.build_orbital_quantities(
        mc, mo, exact_dm1, exact_dm2, eris
    )
    dmrg_quantities = zmc_supercipt.build_orbital_quantities(
        mc, mo, dmrg_dm1, dmrg_dm2, eris
    )
    exact_step = zmc_supercipt.supercipt_step(
        mc, mo, exact_dm1, exact_dm2, eris, canonicalize=False
    )
    dmrg_step = zmc_supercipt.supercipt_step(
        mc, mo, dmrg_dm1, dmrg_dm2, eris, canonicalize=False
    )
    mc_cd = _casscf(mf_cd, mo)
    eris_cd = zmc_ao2mo._CDERIS(mc_cd, mo.copy(), level=2)
    cd_quantities = zmc_supercipt.build_orbital_quantities(
        mc_cd, mo, exact_dm1, exact_dm2, eris_cd
    )
    cd_step = zmc_supercipt.supercipt_step(
        mc_cd, mo, exact_dm1, exact_dm2, eris_cd, canonicalize=False
    )

    assert abs(dmrg_energy - exact_energy) <= 1e-9
    assert np.max(abs(dmrg_dm1 - exact_dm1)) <= 1e-8
    assert np.max(abs(dmrg_dm2 - exact_dm2)) <= 1e-8
    assert abs(
        energy_from_rdms(
            h1eff, eris.aaaa, exact_dm1, exact_dm2, ecore
        )
        - exact_energy
    ) <= 1e-9
    assert abs(
        energy_from_rdms(
            h1eff, eris.aaaa, dmrg_dm1, dmrg_dm2, ecore
        )
        - dmrg_energy
    ) <= 1e-9
    assert np.max(
        abs(
            dmrg_quantities.screened_gradient
            - exact_quantities.screened_gradient
        )
    ) <= 1e-7
    assert np.max(abs(dmrg_step.kappa - exact_step.kappa)) <= 1e-7
    assert np.max(
        abs(
            cd_quantities.screened_gradient
            - exact_quantities.screened_gradient
        )
    ) <= 1e-7
    assert np.max(abs(cd_step.kappa - exact_step.kappa)) <= 1e-7

    ncore = mc.ncore
    nocc = ncore + mc.ncas
    assert np.linalg.norm(exact_step.kappa[ncore:nocc, :ncore]) > 1e-8
    assert np.linalg.norm(exact_step.kappa[nocc:, :ncore]) > 1e-8
    assert np.linalg.norm(exact_step.kappa[nocc:, ncore:nocc]) > 1e-8
    assert solver.converged
    run_scratch = Path(solver._scratch)
    solver.close()
    assert not run_scratch.exists()


@pytest.mark.integration
def test_supercipt_block2_macroiterations_match_exact(
    tilted_hf_supercipt, tmp_path
):
    mol, mf, _, mo = tilted_hf_supercipt
    exact = _casscf(mf, mo)
    exact.supercipt()

    solver = _dmrg(mol, tmp_path / "macro")
    dmrg = _casscf(mf, mo, solver)
    dmrg.supercipt()

    exact_dm1, exact_dm2 = exact.fcisolver.make_rdm12(
        exact.ci, exact.ncas, exact.nelecas
    )
    dmrg_dm1, dmrg_dm2 = solver.make_rdm12(
        dmrg.ci, dmrg.ncas, dmrg.nelecas
    )
    energy_trajectory_error = max(
        abs(drow["total_energy"] - erow["total_energy"])
        for erow, drow in zip(
            exact.supercipt_history, dmrg.supercipt_history
        )
    )
    gradient_trajectory_error = max(
        abs(
            drow["orbital_gradient_norm"]
            - erow["orbital_gradient_norm"]
        )
        for erow, drow in zip(
            exact.supercipt_history, dmrg.supercipt_history
        )
    )
    legacy_exact_energy = -98.63650918755290
    legacy_exact_gradient = 6.185480215550907e-6

    assert exact.converged and dmrg.converged and solver.converged
    assert len(exact.supercipt_history) == len(dmrg.supercipt_history)
    assert abs(dmrg.e_tot - exact.e_tot) <= 1e-7
    assert abs(dmrg.e_cas - exact.e_cas) <= 1e-7
    assert abs(
        dmrg.final_orbital_gradient_norm
        - exact.final_orbital_gradient_norm
    ) <= 1e-7
    assert exact.final_orbital_gradient_norm <= exact.conv_tol_grad
    assert dmrg.final_orbital_gradient_norm <= dmrg.conv_tol_grad
    assert abs(exact.e_tot - legacy_exact_energy) <= 1e-7
    assert abs(
        exact.final_orbital_gradient_norm - legacy_exact_gradient
    ) <= 1e-7
    assert energy_trajectory_error <= 1e-7
    assert gradient_trajectory_error <= 1e-7
    assert np.max(abs(dmrg_dm1 - exact_dm1)) <= 1e-8
    assert np.max(abs(dmrg_dm2 - exact_dm2)) <= 1e-8
    assert np.max(
        abs(
            np.linalg.eigvalsh(dmrg_dm1)
            - np.linalg.eigvalsh(exact_dm1)
        )
    ) <= 1e-8
    assert all(
        row["rdm_energy_error"] <= 1e-9
        for row in exact.supercipt_history + dmrg.supercipt_history
    )

    overlap = mf.get_ovlp()
    ncore = exact.ncore
    nocc = ncore + exact.ncas
    for columns in (
        slice(0, ncore),
        slice(ncore, nocc),
        slice(nocc, mo.shape[1]),
    ):
        assert np.max(
            abs(
                _projector(exact.mo_coeff, overlap, columns)
                - _projector(dmrg.mo_coeff, overlap, columns)
            )
        ) <= 1e-7

    run_scratch = Path(solver._scratch)
    solver.close()
    assert not run_scratch.exists()


@pytest.mark.integration
@pytest.mark.expensive
def test_supercipt_state_average_reaches_legacy_stationary_energy():
    """Reproduce the six-root F/cc-pVTZ legacy Super-CIPT endpoint."""
    mol = gto.M(
        atom="F 0 0 0",
        basis="cc-pvtz",
        spin=0,
        charge=-1,
        verbose=0,
        symmetry=True,
    )
    mf = scf.X2C(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    assert mf.converged

    # This is the original workflow: converge F- orbitals, then optimize the
    # neutral F seven-electron/six-root active-space ensemble.
    mol.charge = 0
    mol.spin = 1
    mc = zmcscf.CASSCF(mf, ncas=8, nelecas=7)
    mc.state_average_(np.ones(6) / 6)
    mc.natorb = False
    mc.canonicalize_ = True
    mc.max_stepsize = 0.2
    mc.max_cycle_macro = 20
    mc.conv_tol = 1e-8
    mc.conv_tol_grad = 1e-4
    mc.verbose = 0
    mc.supercipt()

    legacy_pykylin_energy = -99.4807969507779
    assert mc.converged
    assert len(mc.supercipt_history[0]["root_energies"]) == 6
    assert abs(mc.e_tot - legacy_pykylin_energy) <= 1e-7
    assert mc.final_orbital_gradient_norm <= mc.conv_tol_grad
    assert all(
        row["rdm_energy_error"] <= 1e-9
        for row in mc.supercipt_history
    )
    assert all(
        later["total_energy"] <= earlier["total_energy"] + 1e-10
        for earlier, later in zip(
            mc.supercipt_history, mc.supercipt_history[1:]
        )
    )


def test_supercipt_rejects_kramers_restricted_orbital_equations():
    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        spin=0,
        charge=0,
        verbose=0,
    )
    mf = spinor_hf.KRHF(mol)
    mc = zmcscf.CASSCF(mf, ncas=2, nelecas=2, ncore=0)
    with pytest.raises(
        NotImplementedError,
        match="Kramers-restricted Super-CIPT orbital equations",
    ):
        zmc_supercipt.mcscf_supercipt(
            mc, np.eye(2 * mol.nao_nr(), dtype=complex)
        )
