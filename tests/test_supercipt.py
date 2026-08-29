from functools import reduce
from pathlib import Path

import numpy as np
import pytest
import scipy.linalg
from pyscf import gto, scf
from pyscf.fci import cistring

from socutils.dmrg.dmrgci import DMRGCI, energy_from_rdms
from socutils.dmrg.kramers import identify_kramers_orbitals
from socutils.mcscf import (
    zmc_ao2mo,
    zmc_supercipt,
    zmc_supercipt_new,
    zmcscf,
)
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


@pytest.fixture(scope="module")
def complex_correlated_supercipt():
    """Complex CAS(2,4) point with nonsymmetric active density matrices."""
    mol = gto.M(
        atom="H 0 0 0; F 0.35 0.27 0.8035",
        basis="6-31g",
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
    kappa[10, 7] = 0.04 + 0.02j
    kappa[12, 8] = -0.035 + 0.015j
    kappa[12, 7] = 0.02 - 0.01j
    kappa -= kappa.T.conj()
    mo = mo.dot(scipy.linalg.expm(kappa))

    mc = zmcscf.CASSCF(mf, ncas=4, nelecas=2)
    mc.mo_coeff = mo.copy()
    mc.natorb = False
    mc.canonicalize_ = False
    mc.verbose = 0
    eris = zmc_ao2mo._ERIS(mc, mo.copy(), level=2)
    mci = zmcscf._fake_h_for_fast_casci(mc, mo.copy(), eris)
    _, _, ci = mci.kernel(mo, verbose=0)
    dm1, dm2 = mc.fcisolver.make_rdm12(ci, mc.ncas, mc.nelecas)
    quantities = zmc_supercipt.build_orbital_quantities(
        mc, mo, dm1, dm2, eris
    )
    return mc, mo, ci, dm1, dm2, eris, quantities


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


def _casci_energy(mc, orbitals):
    eris = zmc_ao2mo._ERIS(mc, orbitals, level=2)
    mci = zmcscf._fake_h_for_fast_casci(mc, orbitals, eris)
    energy, _, _ = mci.kernel(orbitals, verbose=0)
    return float(np.real(energy))


def _fock_space_operators(norb):
    dimension = 1 << norb
    annihilation = []
    for orbital in range(norb):
        operator = np.zeros((dimension, dimension), dtype=complex)
        for ket_string in range(dimension):
            if ket_string & (1 << orbital):
                bra_string = ket_string ^ (1 << orbital)
                parity = (
                    ket_string & ((1 << orbital) - 1)
                ).bit_count()
                operator[bra_string, ket_string] = (-1) ** parity
        annihilation.append(operator)
    creation = [operator.T.conj() for operator in annihilation]
    return annihilation, creation


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


def test_supercipt_use_cderi_explicitly_selects_integral_route(
    tilted_hf_supercipt,
):
    _, _, mf_cd, mo = tilted_hf_supercipt
    mc = _casscf(mf_cd, mo)
    full, full_info = zmc_supercipt._build_eris(
        mc,
        mo,
        use_cderi=False,
    )
    factorized, factorized_info = zmc_supercipt._build_eris(
        mc,
        mo,
        use_cderi=True,
    )

    assert isinstance(full, zmc_ao2mo._ERIS)
    assert isinstance(factorized, zmc_ao2mo._CDERIS)
    assert not full_info["factorized"]
    assert factorized_info["factorized"]
    assert np.max(abs(full.aaaa - factorized.aaaa)) <= 1e-9


@pytest.mark.integration
def test_supercipt_complex_gradient_matches_casci_finite_difference(
    complex_correlated_supercipt,
):
    mc, mo, _, _, _, _, quantities = complex_correlated_supercipt
    epsilon = 2e-4

    # The selected elements are the largest old complex-density errors in
    # the core-active, active-virtual, and core-virtual blocks, respectively.
    for row, column in ((8, 4), (12, 8), (17, 4)):
        derivatives = []
        for value in (1.0, 1.0j):
            generator = np.zeros((mo.shape[1], mo.shape[1]), dtype=complex)
            generator[row, column] = value
            generator[column, row] = -value.conjugate()
            plus = _casci_energy(
                mc, mo.dot(scipy.linalg.expm(epsilon * generator))
            )
            minus = _casci_energy(
                mc, mo.dot(scipy.linalg.expm(-epsilon * generator))
            )
            derivatives.append((plus - minus) / (2.0 * epsilon))

        finite_difference = 0.5 * (
            derivatives[0] + 1.0j * derivatives[1]
        )
        assert abs(
            finite_difference - quantities.gradient[row, column]
        ) <= 1e-7


@pytest.mark.integration
def test_supercipt_koopmans_sectors_and_pt_resolvents_are_exact(
    complex_correlated_supercipt,
):
    mc, mo, ci, dm1, dm2, eris, quantities = (
        complex_correlated_supercipt
    )
    norb = mc.ncas
    ncore = mc.ncore
    nocc = ncore + norb
    active = slice(ncore, nocc)
    annihilation, creation = _fock_space_operators(norb)
    dimension = 1 << norb
    hamiltonian = np.zeros((dimension, dimension), dtype=complex)
    h1_active = quantities.fock_core[active, active]
    for p in range(norb):
        for q in range(norb):
            hamiltonian += (
                h1_active[p, q] * creation[p].dot(annihilation[q])
            )
            for r in range(norb):
                for s in range(norb):
                    hamiltonian += (
                        0.5
                        * eris.aaaa[p, q, r, s]
                        * creation[p]
                        .dot(creation[r])
                        .dot(annihilation[s])
                        .dot(annihilation[q])
                    )

    psi = np.zeros(dimension, dtype=complex)
    for coefficient, occupied in zip(
        ci, cistring.gen_occslst(range(norb), mc.nelecas)
    ):
        address = sum(1 << int(index) for index in occupied)
        psi[address] = coefficient
    energy_n = float(np.real(np.vdot(psi, hamiltonian.dot(psi))))

    removal_direct = np.empty((norb, norb), dtype=complex)
    addition_direct = np.empty_like(removal_direct)
    for t in range(norb):
        for u in range(norb):
            removal_direct[t, u] = np.vdot(
                psi,
                creation[u]
                .dot(
                    hamiltonian.dot(annihilation[t])
                    - annihilation[t].dot(hamiltonian)
                )
                .dot(psi),
            )
            addition_direct[u, t] = np.vdot(
                psi,
                annihilation[u]
                .dot(
                    hamiltonian.dot(creation[t])
                    - creation[t].dot(hamiltonian)
                )
                .dot(psi),
            )

    step = zmc_supercipt.supercipt_step(
        mc,
        mo,
        dm1,
        dm2,
        eris,
        metric_tol=1e-8,
        canonicalize=False,
    )
    assert np.max(abs(step.koopmans_removal - removal_direct)) <= 1e-10
    assert np.max(abs(step.koopmans_addition - addition_direct)) <= 1e-10

    minus_sector = [
        index
        for index in range(dimension)
        if index.bit_count() == mc.nelecas - 1
    ]
    plus_sector = [
        index
        for index in range(dimension)
        if index.bit_count() == mc.nelecas + 1
    ]
    exact_removal = energy_n - np.linalg.eigvalsh(
        hamiltonian[np.ix_(minus_sector, minus_sector)]
    )
    exact_addition = np.linalg.eigvalsh(
        hamiltonian[np.ix_(plus_sector, plus_sector)]
    ) - energy_n
    assert np.max(
        abs(np.sort(step.removal_energies) - np.sort(exact_removal))
    ) <= 1e-10
    assert np.max(
        abs(np.sort(step.addition_energies) - np.sort(exact_addition))
    ) <= 1e-10

    # Check eqs. 24--26 independently as direct generalized resolvents.
    diagonal = np.diag(quantities.fock_effective).real
    expected_lower = np.zeros_like(step.kappa_unscaled)
    expected_lower[nocc:, :ncore] = (
        quantities.gradient[nocc:, :ncore]
        / (diagonal[:ncore][None, :] - diagonal[nocc:, None])
    )
    removal_metric = dm1.T
    addition_metric = (np.eye(norb) - dm1).T
    for core in range(ncore):
        expected_lower[active, core] = -addition_metric.dot(
            np.linalg.solve(
                step.koopmans_addition
                - diagonal[core] * addition_metric,
                quantities.gradient[active, core],
            )
        )
    for virtual in range(nocc, mo.shape[1]):
        expected_lower[virtual, active] = removal_metric.dot(
            np.linalg.solve(
                -step.koopmans_removal
                - diagonal[virtual] * removal_metric,
                quantities.gradient[virtual, active],
            )
        )
    expected_kappa = expected_lower - expected_lower.T.conj()
    assert np.max(abs(step.kappa_unscaled - expected_kappa)) <= 1e-10


@pytest.mark.integration
def test_supercipt_always_canonicalizes_pt_denominator_spaces(
    complex_correlated_supercipt,
):
    mc, mo, _, dm1, dm2, eris, _ = complex_correlated_supercipt
    assert not mc.canonicalize_
    step = zmc_supercipt.supercipt_step(
        mc, mo, dm1, dm2, eris, canonicalize=True
    )
    assert len(step.canonical_energies["core"]) == mc.ncore
    assert len(step.canonical_energies["virtual"]) == (
        mo.shape[1] - mc.ncore - mc.ncas
    )

    nocc = mc.ncore + mc.ncas
    dm_core = step.mo_coeff[:, :mc.ncore].dot(
        step.mo_coeff[:, :mc.ncore].T.conj()
    )
    mo_active = step.mo_coeff[:, mc.ncore:nocc]
    dm_active = reduce(
        np.dot, (mo_active, dm1.T, mo_active.T.conj())
    )
    vj_core, vk_core = mc.get_jk(mc.mol, dm_core)
    vj_active, vk_active = mc.get_jk(mc.mol, dm_active)
    fock_ao = (
        mc.get_hcore()
        + vj_core
        - vk_core
        + vj_active
        - vk_active
    )
    fock_mo = reduce(
        np.dot,
        (step.mo_coeff.T.conj(), fock_ao, step.mo_coeff),
    )
    for block in (
        fock_mo[:mc.ncore, :mc.ncore],
        fock_mo[nocc:, nocc:],
    ):
        assert np.linalg.norm(block - np.diag(np.diag(block))) <= 1e-10


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


def test_supercipt_kramers_diis_requires_explicit_symmetry():
    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        spin=0,
        charge=0,
        verbose=0,
    )
    mf = spinor_hf.KRHF(mol)
    mc = zmcscf.CASSCF(mf, ncas=2, nelecas=2, ncore=0)
    with pytest.raises(ValueError, match="symm='kramers' is required"):
        zmc_supercipt.mcscf_supercipt(
            mc,
            np.eye(2 * mol.nao_nr(), dtype=complex),
            use_diis=True,
        )


@pytest.mark.integration
def test_contributed_supercipt_interface_generates_cderi(
    tilted_hf_supercipt,
):
    _, mf, _, mo = tilted_hf_supercipt
    mc = _casscf(mf, mo)
    converged, energy, final_mo = zmc_supercipt_new.mcscf_superci_pt(
        mc,
        mf,
        max_cycle=1,
        use_cderi=True,
        use_diis=True,
    )

    assert not converged
    assert energy == pytest.approx(mc.e_tot)
    assert np.max(abs(final_mo - mc.mo_coeff)) == 0.0
    assert mc.supercipt_diagnostics["integrals"]["factorized"]
    assert mc.supercipt_diagnostics["integrals"]["source"] == "legacy-cderi"
    assert mc.supercipt_diagnostics["diis"]


@pytest.mark.integration
def test_supercipt_diis_rejects_an_uphill_extrapolation(
    tilted_hf_supercipt,
    monkeypatch,
):
    """An extrapolated step may not move the variational energy uphill."""
    _, mf, _, mo = tilted_hf_supercipt
    mc = _casscf(mf, mo)
    mc.max_cycle_macro = 6
    injected = {"done": False}
    original_update = zmc_supercipt.IncrementalOrbitalDIIS.update

    def uphill_update(self, current_mo, proposed_mo, gradient, **kwargs):
        result = original_update(
            self,
            current_mo,
            proposed_mo,
            gradient,
            **kwargs,
        )
        if (
            result.diagnostics["extrapolated"]
            and kwargs["cycle"] == mc.max_cycle_macro - 2
            and not injected["done"]
        ):
            injected["done"] = True
            generator = np.zeros(
                (current_mo.shape[1],) * 2,
                dtype=np.complex128,
            )
            generator[-1, 0] = 0.2
            generator[0, -1] = -0.2
            diagnostics = dict(result.diagnostics)
            diagnostics["test_uphill_injection"] = True
            return result.__class__(
                current_mo.dot(scipy.linalg.expm(generator)),
                generator,
                diagnostics,
            )
        return result

    monkeypatch.setattr(
        zmc_supercipt.IncrementalOrbitalDIIS,
        "update",
        uphill_update,
    )
    mc.supercipt(
        use_diis=True,
        diis_start_cycle=0,
    )

    rejected = [row for row in mc.supercipt_history if not row["accepted"]]
    assert injected["done"]
    assert len(rejected) == 1
    assert rejected[0]["diis_rejection"]["reason"] == "energy-increase"
    assert mc.supercipt_diagnostics["diis_energy_safeguard"]
    assert mc.supercipt_diagnostics["diis_rejected_steps"] == 1
    assert mc.supercipt_diagnostics["last_step_scale"] is None
    rejected_index = mc.supercipt_history.index(rejected[0])
    assert rejected_index == len(mc.supercipt_history) - 2
    assert len(mc.supercipt_history) == mc.max_cycle_macro + 1
    assert mc.supercipt_history[rejected_index + 1]["accepted"]
    assert (
        mc.supercipt_history[rejected_index + 1]["total_energy"]
        < rejected[0]["total_energy"]
    )


@pytest.mark.integration
def test_supercipt_kramers_diis_preserves_time_reversal():
    mol = gto.M(
        atom="H 0 0 0",
        basis="6-31g",
        spin=1,
        charge=0,
        verbose=0,
        max_memory=1000,
    )
    mf = spinor_hf.KRHF(mol).x2camf(
        with_gaunt=False,
        with_breit=False,
    ).cholesky(tau=1e-10)
    mf.init_guess = "1e"
    mf.conv_tol = 1e-12
    mf.kernel()
    assert mf.converged

    rotation = np.eye(4, dtype=complex)
    cosine, sine = np.cos(0.18), np.sin(0.18)
    for occupied, virtual in ((0, 2), (1, 3)):
        rotation[occupied, occupied] = cosine
        rotation[virtual, virtual] = cosine
        rotation[occupied, virtual] = sine
        rotation[virtual, occupied] = -sine
    initial_mo = mf.mo_coeff.dot(rotation)

    mc = zmcscf.CASSCF(mf, ncas=2, nelecas=1)
    mc.mo_coeff = initial_mo
    mc.state_average_([0.5, 0.5])
    mc.natorb = False
    mc.canonicalize_ = False
    mc.max_cycle_macro = 30
    mc.conv_tol = 1e-10
    mc.conv_tol_grad = 1e-7
    mc.max_stepsize = 0.1
    mc.verbose = 0
    converged, energy, final_mo = zmc_supercipt_new.mcscf_superci_pt(
        mc,
        mf,
        symm="kramers",
        max_cycle=mc.max_cycle_macro,
        conv_etol=mc.conv_tol,
        conv_gtol=mc.conv_tol_grad,
        max_step=mc.max_stepsize,
        use_diis=True,
        use_cderi=True,
    )

    mapping = identify_kramers_orbitals(
        mol,
        mc.mo_coeff,
        mf.get_ovlp(),
        tolerance=1e-7,
    )
    assert converged and mc.converged
    assert energy == pytest.approx(mc.e_tot)
    assert np.max(abs(final_mo - mc.mo_coeff)) == 0.0
    assert mc.supercipt_diagnostics["kramers_restricted"]
    assert mc.supercipt_diagnostics["diis"]
    assert mc.supercipt_diagnostics["integrals"]["factorized"]
    assert mapping.diagnostics["subspace_closure_error"] <= 1e-7
    assert mapping.diagnostics["partner_orbital_error"] <= 1e-7
    assert any(
        row.get("diis", {}).get("extrapolated", False)
        for row in mc.supercipt_history
    )


@pytest.mark.integration
def test_cl_kramers_supercipt_incremental_diis_accelerates_convergence():
    """Cl exposes gauge contamination in fixed-reference Super-CIPT DIIS."""
    mol = gto.M(
        atom="Cl 0 0 0",
        basis="dyallv2z",
        charge=-1,
        spin=0,
        verbose=0,
        max_memory=4000,
    )
    mf = spinor_hf.KRHF(mol).x2camf(
        with_gaunt=False,
        with_breit=False,
    ).cholesky(tau=1e-10)
    mf.init_guess = "1e"
    mf.conv_tol = 1e-12
    mf.kernel()
    assert mf.converged

    initial_mo = mf.mo_coeff.copy()
    mol.charge = 0
    mol.spin = 1

    def make_casscf():
        mc = zmcscf.CASSCF(mf, ncas=6, nelecas=5)
        mc.mo_coeff = initial_mo.copy()
        mc.state_average_(np.ones(6) / 6)
        mc.natorb = False
        mc.canonicalize_ = False
        mc.max_cycle_macro = 15
        mc.max_stepsize = 0.1
        mc.conv_tol = 1e-9
        mc.conv_tol_grad = 1e-6
        mc.verbose = 0
        return mc

    plain = make_casscf()
    plain.supercipt(
        symm="kramers",
        use_cderi=True,
        use_diis=False,
    )

    accelerated = make_casscf()
    converged, energy, final_mo = zmc_supercipt_new.mcscf_superci_pt(
        accelerated,
        mf,
        symm="kramers",
        max_cycle=15,
        conv_etol=accelerated.conv_tol,
        conv_gtol=accelerated.conv_tol_grad,
        max_step=accelerated.max_stepsize,
        use_cderi=True,
        use_diis=True,
    )
    mapping = identify_kramers_orbitals(
        mol,
        final_mo,
        mf.get_ovlp(),
        tolerance=1e-7,
    )

    full_superci = make_casscf()
    full_superci.superci(symm="kramers", use_diis=True)

    assert not plain.converged
    assert converged and accelerated.converged
    assert full_superci.converged
    assert accelerated.final_orbital_gradient_norm <= accelerated.conv_tol_grad
    assert accelerated.final_orbital_gradient_norm < plain.final_orbital_gradient_norm
    assert abs(energy - (-460.8793608192712)) <= 1e-8
    assert abs(energy - full_superci.e_tot) <= 1e-8
    assert accelerated.supercipt_diagnostics[
        "pt_core_virtual_canonicalization"
    ] is False
    assert any(
        row.get("diis", {}).get("coordinate_system")
        == "accumulated-incremental"
        and row["diis"]["extrapolated"]
        for row in accelerated.supercipt_history
    )
    assert any(
        row.get("diis", {}).get("extrapolated", False)
        for row in full_superci.macro_history
    )
    assert mapping.diagnostics["partner_orbital_error"] <= 1e-7
