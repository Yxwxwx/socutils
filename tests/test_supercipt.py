import inspect
from functools import partial, reduce
from pathlib import Path

import numpy as np
import pytest
import scipy.linalg
from pyscf import gto, scf
from pyscf.fci import cistring

from socutils.dmrg.dmrgci import DMRGCI, energy_from_rdms
from socutils.dmrg.kramers import identify_kramers_orbitals
from socutils.mcscf.orbital_cg import SpectralOrbitalCG
from socutils.mcscf import (
    zmc_ao2mo,
    zmc_supercipt,
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


def _run_experimental_supercipt(mc, acceleration, **kwargs):
    """Exercise an internal optimizer experiment without extending the API."""
    kernel = partial(
        zmc_supercipt.mcscf_supercipt,
        acceleration=acceleration,
    )
    return mc.supercipt(_kern=kernel, **kwargs)


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


def _random_unitary(rng, dimension):
    if dimension == 0:
        return np.empty((0, 0), dtype=complex)
    raw = rng.standard_normal((dimension, dimension)) + 1.0j * rng.standard_normal(
        (dimension, dimension)
    )
    unitary, _ = np.linalg.qr(raw)
    return unitary


def test_line_search_trust_is_committed_transactionally():
    zoom = zmc_supercipt._resolve_line_search_trust(
        0.1,
        0.08,
        accepted_on_boundary=False,
        linear_ratio=0.7,
        boundary_failed=True,
        max_stepsize=0.2,
    )
    assert zoom["trust_radius_after"] == pytest.approx(0.08)
    assert zoom["trust_radius_after"] >= zoom["accepted_step_norm"]
    assert zoom["trust_action"] == (
        "boundary-failure-raised-to-accepted-step"
    )

    restored = zmc_supercipt._resolve_line_search_trust(
        0.1,
        0.0,
        accepted_on_boundary=False,
        linear_ratio=None,
        boundary_failed=True,
        max_stepsize=0.2,
    )
    assert restored["trust_radius_after"] == pytest.approx(0.05)
    assert restored["trust_action"] == "halved-base-restore"

    expanded = zmc_supercipt._resolve_line_search_trust(
        0.1,
        0.1,
        accepted_on_boundary=True,
        linear_ratio=0.8,
        boundary_failed=False,
        max_stepsize=0.2,
    )
    assert expanded["trust_radius_after"] == pytest.approx(0.2)


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


def test_supercipt_gradient_norm_uses_independent_orbital_variables(
    complex_correlated_supercipt,
):
    mc, _, _, _, _, _, quantities = complex_correlated_supercipt
    packed = mc.pack_uniq_var(quantities.screened_gradient)

    assert quantities.gradient_norm == pytest.approx(np.linalg.norm(packed))
    assert quantities.gradient_frobenius_norm == pytest.approx(
        np.linalg.norm(quantities.screened_gradient)
    )
    assert quantities.gradient_frobenius_norm == pytest.approx(
        np.sqrt(2.0) * quantities.gradient_norm
    )
    assert quantities.raw_gradient_norm == pytest.approx(
        quantities.gradient_norm
    )
    assert quantities.raw_gradient_frobenius_norm == pytest.approx(
        quantities.gradient_frobenius_norm
    )
    assert quantities.kramers_gradient_diagnostics is None


def test_supercipt_anderson_uses_actual_fixed_reference_coordinates():
    reference = np.eye(2, dtype=complex)
    generator = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=complex)
    accelerator = zmc_supercipt.IncrementalOrbitalDIIS(
        reference,
        np.eye(2),
        space=4,
        start_cycle=0,
        start_gradient=None,
    )
    gradient = -generator

    first = accelerator.update(
        reference,
        scipy.linalg.expm(0.2 * generator),
        gradient,
        cycle=0,
    )
    assert not first.diagnostics["extrapolated"]

    plain_second = first.mo_coeff.dot(scipy.linalg.expm(0.1 * generator))
    second = accelerator.update(
        first.mo_coeff,
        plain_second,
        gradient,
        cycle=1,
    )
    # Anderson solves theta = 0.5 theta + 0.2 exactly from the first two
    # fixed-point residuals: the extrapolated coordinate is theta=0.4.
    assert second.diagnostics["extrapolated"]
    assert np.max(
        abs(second.mo_coeff - scipy.linalg.expm(0.4 * generator))
    ) <= 1e-10
    assert second.diagnostics["coordinate_system"] == (
        "fixed-reference-unitary-log"
    )
    assert second.diagnostics["pulay"]["coefficient_l1_norm"] == (
        pytest.approx(3.0)
    )
    assert second.diagnostics["proposed_directional_derivative"] < 0.0


def test_supercipt_anderson_constrained_svd_resolves_slow_collinear_mode():
    reference = np.eye(2, dtype=complex)
    generator = np.array(
        [[0.0, -0.6 + 0.8j], [0.6 + 0.8j, 0.0]],
        dtype=complex,
    )
    accelerator = zmc_supercipt.AndersonOrbitalDIIS(
        reference,
        np.eye(2),
        space=4,
        start_cycle=0,
        start_gradient=None,
        coefficient_l1_max=1000.0,
    )
    gradient = -generator

    first = accelerator.update(
        reference,
        scipy.linalg.expm(0.001 * generator),
        gradient,
        cycle=0,
    )
    # r(theta) = 0.001 - 0.01 theta has contraction factor 0.99.
    # Its first two residuals are nearly collinear, but the constrained SVD
    # still recovers theta=0.1 without forming a squared-condition Gram KKT.
    plain_second = first.mo_coeff.dot(scipy.linalg.expm(0.00099 * generator))
    second = accelerator.update(
        first.mo_coeff,
        plain_second,
        gradient,
        cycle=1,
    )

    pulay = second.diagnostics["pulay"]
    assert second.diagnostics["extrapolated"]
    assert pulay["solver"] == "constrained-residual-svd"
    assert pulay["residual_svd_rank"] == 1
    assert pulay["coefficient_sum"] == pytest.approx(1.0, abs=1e-12)
    assert pulay["coefficients"] == pytest.approx([-99.0, 100.0], abs=1e-8)
    assert pulay["coefficient_l1_norm"] == pytest.approx(199.0, abs=1e-8)
    assert pulay["predicted_residual_norm"] <= 1e-12
    assert pulay["predicted_residual_norm"] <= pulay["best_residual_norm"]
    assert np.max(
        abs(second.mo_coeff - scipy.linalg.expm(0.1 * generator))
    ) <= 1e-10


def test_supercipt_anderson_preserves_candidate_when_coefficient_limit_rejects():
    reference = np.eye(2, dtype=complex)
    generator = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=complex)
    accelerator = zmc_supercipt.AndersonOrbitalDIIS(
        reference,
        np.eye(2),
        space=4,
        start_cycle=0,
        start_gradient=None,
        coefficient_l1_max=10.0,
    )
    gradient = -generator

    first = accelerator.update(
        reference,
        scipy.linalg.expm(0.001 * generator),
        gradient,
        cycle=0,
    )
    plain_second = first.mo_coeff.dot(scipy.linalg.expm(0.00099 * generator))
    second = accelerator.update(
        first.mo_coeff,
        plain_second,
        gradient,
        cycle=1,
    )

    pulay = second.diagnostics["pulay"]
    assert not second.diagnostics["extrapolated"]
    assert second.diagnostics["pulay_reset"]
    assert second.diagnostics["extrapolation_rejection"] == (
        "coefficient-l1-limit"
    )
    assert pulay["coefficients"] == pytest.approx([-99.0, 100.0], abs=1e-8)
    assert pulay["coefficient_sum"] == pytest.approx(1.0, abs=1e-12)
    assert pulay["coefficient_l1_norm"] == pytest.approx(199.0, abs=1e-8)
    assert pulay["coefficient_l1_norm"] > second.diagnostics[
        "coefficient_l1_limit"
    ]
    assert second.diagnostics["vectors"] == 1
    assert np.max(abs(second.mo_coeff - plain_second)) <= 1e-12


def test_supercipt_automatically_selects_integral_route(
    tilted_hf_supercipt,
):
    _, mf, mf_cd, mo = tilted_hf_supercipt
    mc = _casscf(mf, mo)
    full, full_info = zmc_supercipt._build_eris(
        mc,
        mo,
    )
    mc_cd = _casscf(mf_cd, mo)
    factorized, factorized_info = zmc_supercipt._build_eris(
        mc_cd,
        mo,
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

    # Check eqs. 24--26 independently as direct generalized resolvents in
    # the temporary semicanonical frame, then transform the physical step
    # back to the incoming MO gauge.
    (
        transform,
        gradient_canonical,
        core_energies,
        virtual_energies,
        _,
        _,
    ) = zmc_supercipt._pt_semicanonical_frame(mc, mo, quantities)
    expected_lower_canonical = np.zeros_like(step.kappa_unscaled)
    expected_lower_canonical[nocc:, :ncore] = (
        gradient_canonical[nocc:, :ncore]
        / (core_energies[None, :] - virtual_energies[:, None])
    )
    removal_metric = dm1.T
    addition_metric = (np.eye(norb) - dm1).T
    for core in range(ncore):
        expected_lower_canonical[active, core] = -addition_metric.dot(
            np.linalg.solve(
                step.koopmans_addition
                - core_energies[core] * addition_metric,
                gradient_canonical[active, core],
            )
        )
    for virtual in range(nocc, mo.shape[1]):
        expected_lower_canonical[virtual, active] = removal_metric.dot(
            np.linalg.solve(
                -step.koopmans_removal
                - virtual_energies[virtual - nocc] * removal_metric,
                gradient_canonical[virtual, active],
            )
        )
    expected_kappa_canonical = (
        expected_lower_canonical - expected_lower_canonical.T.conj()
    )
    expected_kappa = reduce(
        np.dot,
        (
            transform,
            expected_kappa_canonical,
            transform.T.conj(),
        ),
    )
    expected_kappa = mc.unpack_uniq_var(mc.pack_uniq_var(expected_kappa))
    assert np.max(abs(step.kappa_unscaled - expected_kappa)) <= 1e-10


@pytest.mark.integration
def test_supercipt_uses_temporary_semicanonical_pt_frame(
    complex_correlated_supercipt,
):
    mc, mo, _, dm1, dm2, eris, quantities = complex_correlated_supercipt
    assert not mc.canonicalize_
    step = zmc_supercipt.supercipt_step(mc, mo, dm1, dm2, eris)
    assert len(step.canonical_energies["core"]) == mc.ncore
    assert len(step.canonical_energies["virtual"]) == (
        mo.shape[1] - mc.ncore - mc.ncas
    )
    assert (
        step.semicanonical_diagnostics["core_fock_offdiagonal_after"]
        <= 1e-10
    )
    assert (
        step.semicanonical_diagnostics[
            "virtual_fock_offdiagonal_after"
        ]
        <= 1e-10
    )
    # The redundant transformation is internal: the returned orbitals contain
    # only the interspace PT rotation in the original MO gauge.
    assert np.max(abs(step.mo_coeff - mo.dot(step.rotation))) <= 1e-12
    assert step.direction_diagnostics["total"]["directional_derivative"] < 0.0
    assert all(
        block["directional_derivative"] <= 1e-14
        for block in step.direction_diagnostics.values()
    )


@pytest.mark.integration
def test_supercipt_step_is_covariant_to_redundant_orbital_gauge(
    complex_correlated_supercipt,
):
    mc, mo, _, dm1, dm2, eris, _ = complex_correlated_supercipt
    ncore = mc.ncore
    nocc = ncore + mc.ncas
    nmo = mo.shape[1]
    rng = np.random.default_rng(9127)
    gauge = np.eye(nmo, dtype=complex)
    gauge[:ncore, :ncore] = _random_unitary(rng, ncore)
    gauge[nocc:, nocc:] = _random_unitary(rng, nmo - nocc)
    rotated_mo = mo.dot(gauge)
    rotated_eris = zmc_ao2mo._ERIS(mc, rotated_mo, level=2)

    # A small radius deliberately activates trust scaling.  The physical
    # scale must remain invariant even though max(abs(kappa)) is gauge
    # dependent.
    step = zmc_supercipt.supercipt_step(
        mc, mo, dm1, dm2, eris, max_stepsize=0.01
    )
    rotated_step = zmc_supercipt.supercipt_step(
        mc, rotated_mo, dm1, dm2, rotated_eris, max_stepsize=0.01
    )
    expected_unscaled = reduce(
        np.dot,
        (gauge.T.conj(), step.kappa_unscaled, gauge),
    )
    expected_applied = reduce(
        np.dot,
        (gauge.T.conj(), step.kappa, gauge),
    )
    assert np.linalg.norm(
        rotated_step.kappa_unscaled - expected_unscaled
    ) <= 1e-10
    assert np.linalg.norm(rotated_step.kappa - expected_applied) <= 1e-10
    assert rotated_step.unscaled_step_norm == pytest.approx(
        step.unscaled_step_norm, abs=1e-12
    )
    assert rotated_step.scale == pytest.approx(step.scale, abs=1e-12)
    assert rotated_step.minimum_denominator == pytest.approx(
        step.minimum_denominator, abs=1e-10
    )

    overlap = mc._scf.get_ovlp()
    for columns in (
        slice(0, ncore),
        slice(ncore, nocc),
        slice(nocc, nmo),
    ):
        assert np.linalg.norm(
            _projector(step.mo_coeff, overlap, columns)
            - _projector(rotated_step.mo_coeff, overlap, columns)
        ) <= 1e-10


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
        mc, mo, exact_dm1, exact_dm2, eris
    )
    dmrg_step = zmc_supercipt.supercipt_step(
        mc, mo, dmrg_dm1, dmrg_dm2, eris
    )
    mc_cd = _casscf(mf_cd, mo)
    eris_cd = zmc_ao2mo._CDERIS(mc_cd, mo.copy(), level=2)
    cd_quantities = zmc_supercipt.build_orbital_quantities(
        mc_cd, mo, exact_dm1, exact_dm2, eris_cd
    )
    cd_step = zmc_supercipt.supercipt_step(
        mc_cd, mo, exact_dm1, exact_dm2, eris_cd
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
    density_trajectory_error = max(
        abs(drow["norm_ddm"] - erow["norm_ddm"])
        for erow, drow in zip(
            exact.supercipt_history[1:], dmrg.supercipt_history[1:]
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
    assert exact.supercipt_history[0]["norm_ddm"] is None
    assert dmrg.supercipt_history[0]["norm_ddm"] is None
    assert all(
        np.isfinite(row["norm_ddm"])
        for row in exact.supercipt_history[1:]
        + dmrg.supercipt_history[1:]
    )
    assert density_trajectory_error <= 1e-7
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
    mc.supercipt(use_diis=False)

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


def test_supercipt_infers_kramers_symmetry_from_reference():
    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        spin=0,
        charge=0,
        verbose=0,
    )
    mf = spinor_hf.KRHF(mol)
    mc = zmcscf.CASSCF(mf, ncas=2, nelecas=2, ncore=0)
    assert zmc_supercipt._resolve_kramers_mode(mc)


@pytest.mark.integration
def test_supercipt_public_interface_uses_automatic_defaults(
    tilted_hf_supercipt,
):
    parameters = inspect.signature(zmcscf.CASSCF.supercipt).parameters
    assert "symm" not in parameters
    assert "use_cderi" not in parameters
    assert "acceleration" not in parameters

    _, _, mf_cd, mo = tilted_hf_supercipt
    mc = _casscf(mf_cd, mo)
    assert not hasattr(mc, "supercipt_acceleration")
    assert mc.supercipt_diis is False
    mc.max_cycle_macro = 1
    mc.supercipt()

    assert not mc.converged
    assert np.isfinite(mc.e_tot)
    assert mc.supercipt_diagnostics["integrals"]["factorized"]
    assert mc.supercipt_diagnostics["integrals"]["source"] != "full-integral"
    assert not mc.supercipt_diagnostics["diis"]


@pytest.mark.parametrize(
    "acceleration", ["spectral-cg", "pt-trust", "lbfgs"]
)
def test_supercipt_diis_and_pt_acceleration_are_mutually_exclusive(
    tilted_hf_supercipt,
    acceleration,
):
    _, mf, _, mo = tilted_hf_supercipt
    mc = _casscf(mf, mo)
    with pytest.raises(ValueError, match="mutually exclusive"):
        _run_experimental_supercipt(
            mc,
            acceleration,
            use_diis=True,
        )


@pytest.mark.integration
def test_supercipt_pt_trust_physical_trajectory_is_variational(
    tilted_hf_supercipt,
):
    _, mf, _, mo = tilted_hf_supercipt
    mc = _casscf(mf, mo)
    mc.max_cycle_macro = 8
    callback_rows = []
    mc.callback = lambda row: callback_rows.append(dict(row))
    _run_experimental_supercipt(mc, "pt-trust")

    history = mc.supercipt_history
    accepted = [row for row in history if row["accepted"]]
    nonfinal_trials = [row for row in history if not row["accepted"]]
    assert len(history) <= mc.max_cycle_macro
    assert history[-1]["accepted"]
    assert all(
        later["total_energy"] <= earlier["total_energy"] + 1e-10
        for earlier, later in zip(accepted, accepted[1:])
    )
    assert all(row["rdm_energy_error"] <= 1e-9 for row in accepted)
    assert all("pt_trust_trial" in row for row in history[1:])
    assert all(
        "applied_orbital_step_norm" not in row
        for row in nonfinal_trials
    )
    assert any("pt_trust" in row for row in accepted)
    assert all(
        row["applied_orbital_step_norm"] <= mc.max_stepsize + 1e-12
        for row in accepted
        if "applied_orbital_step_norm" in row
    )
    assert all(
        row["pt_trust_acceptance"]["accepted_step_norm"]
        <= row["pt_trust_acceptance"]["trust_radius_after"] + 1e-14
        for row in accepted
        if "pt_trust_acceptance" in row
    )
    diagnostics = mc.supercipt_diagnostics
    assert diagnostics["pt_trust"]
    assert diagnostics["hard_evaluation_budget"] == mc.max_cycle_macro
    assert diagnostics["energy_evaluations"] == len(history)
    assert diagnostics["pt_trust_trial_evaluations"] == len(history) - 1
    assert diagnostics["pt_trust_nonfinal_trials"] == len(nonfinal_trials)
    assert mc.e_tot == pytest.approx(history[-1]["total_energy"], abs=1e-12)
    assert [row["accepted"] for row in callback_rows] == [
        row["accepted"] for row in history
    ]


@pytest.mark.integration
def test_supercipt_pt_trust_rolls_back_uphill_with_hard_budget(
    tilted_hf_supercipt,
    monkeypatch,
):
    """An uphill trial is rejected and the base is replayed within max_cycle."""
    _, mf, _, mo = tilted_hf_supercipt
    mc = _casscf(mf, mo)
    mc.max_cycle_macro = 3
    callback_rows = []
    evaluation_count = {"value": 0}
    original_fake_casci = zmcscf._fake_h_for_fast_casci

    def fake_casci_with_one_uphill_evaluation(*args, **kwargs):
        mci = original_fake_casci(*args, **kwargs)
        original_kernel = mci.kernel

        def kernel(*kernel_args, **kernel_kwargs):
            result = original_kernel(*kernel_args, **kernel_kwargs)
            evaluation_count["value"] += 1
            if evaluation_count["value"] == 2:
                total, cas, ci = result
                return total + 1.0, cas + 1.0, ci
            return result

        mci.kernel = kernel
        return mci

    monkeypatch.setattr(
        zmcscf,
        "_fake_h_for_fast_casci",
        fake_casci_with_one_uphill_evaluation,
    )
    mc.callback = lambda row: callback_rows.append(dict(row))
    _run_experimental_supercipt(mc, "pt-trust")

    history = mc.supercipt_history
    assert evaluation_count["value"] == mc.max_cycle_macro
    assert len(history) == mc.max_cycle_macro
    assert [row["accepted"] for row in history] == [True, False, True]
    rejected = history[1]
    restored = history[2]
    assert rejected["pt_trust_trial"]["action"] == "restore-base"
    assert rejected["pt_trust_trial"]["reason"] == (
        "budget-base-reevaluation"
    )
    trust_update = rejected["pt_trust_trial"]["boundary_trust_update"]
    assert trust_update["deferred"]
    assert trust_update["trust_radius_after"] == pytest.approx(
        0.5 * trust_update["trust_radius_before"]
    )
    assert trust_update["global_trust_radius_unchanged"] == pytest.approx(
        trust_update["trust_radius_before"]
    )
    assert "applied_orbital_step_norm" not in rejected
    assert restored["pt_trust_trial"]["reason"] == "budget-restored-base"
    assert restored["pt_trust_acceptance"]["accepted_step_norm"] == 0.0
    acceptance = restored["pt_trust_acceptance"]
    assert acceptance["boundary_failed_during_search"]
    assert acceptance["trust_radius_after"] == pytest.approx(
        0.5 * acceptance["trust_radius_before"]
    )
    assert restored["total_energy"] == pytest.approx(
        history[0]["total_energy"], abs=1e-12
    )
    assert restored["rdm_energy_error"] <= 1e-9
    assert np.max(abs(mc.mo_coeff - mo)) <= 1e-10
    assert mc.e_tot == pytest.approx(restored["total_energy"], abs=1e-12)
    assert mc.supercipt_diagnostics["energy_evaluations"] == 3
    assert mc.supercipt_diagnostics["accepted_evaluations"] == 2
    assert mc.supercipt_diagnostics["pt_trust_nonfinal_trials"] == 1
    assert mc.supercipt_diagnostics["hard_evaluation_budget"] == 3
    assert [row["accepted"] for row in callback_rows] == [True, False, True]


@pytest.mark.integration
def test_supercipt_lbfgs_physical_trajectory_is_variational_and_transactional(
    tilted_hf_supercipt,
):
    _, mf, _, mo = tilted_hf_supercipt
    mc = _casscf(mf, mo)
    mc.max_cycle_macro = 10
    callback_rows = []
    mc.callback = lambda row: callback_rows.append(dict(row))
    _run_experimental_supercipt(mc, "lbfgs")

    history = mc.supercipt_history
    accepted = [row for row in history if row["accepted"]]
    provisional = [row for row in history if not row["accepted"]]
    assert len(history) <= mc.max_cycle_macro
    assert history[-1]["accepted"]
    assert all(
        later["total_energy"] <= earlier["total_energy"] + 1e-10
        for earlier, later in zip(accepted, accepted[1:])
    )
    assert all(row["rdm_energy_error"] <= 1e-9 for row in accepted)
    assert all("applied_orbital_step_norm" not in row for row in provisional)
    assert all("lbfgs_trial" in row for row in history[1:])

    first = history[0]
    first_direction = first["lbfgs"]["lbfgs"]
    assert first_direction["history_size_before"] == 0
    assert first_direction["first_plain_pt_equivalence_error"] <= 1e-12
    assert first["step_scale"] == pytest.approx(1.0, abs=1e-12)
    assert first["applied_orbital_step_norm"] == pytest.approx(
        first["proposed_orbital_step_norm"], abs=1e-12
    )

    natural_lbfgs_steps = [
        row
        for row in accepted
        if row.get("lbfgs", {}).get("phase") == "proposal"
        and row["lbfgs"]["lbfgs"]["history_size_before"] > 0
        and row["lbfgs"]["direction_norm"]
        < row["lbfgs"]["trust_radius"]
    ]
    assert natural_lbfgs_steps
    assert all(
        row["lbfgs"]["alpha"] == pytest.approx(1.0, abs=1e-12)
        and row["lbfgs"]["initial_alpha_policy"]
        == "natural-step-trust-clipped"
        and row["applied_orbital_step_norm"]
        == pytest.approx(row["lbfgs"]["direction_norm"], abs=1e-12)
        for row in natural_lbfgs_steps
    )

    pair_actions = [
        row["lbfgs_acceptance"]["lbfgs_pair_action"]
        for row in accepted
        if row.get("lbfgs_acceptance", {}).get("accepted_step_norm", 0.0)
        > 0.0
    ]
    assert pair_actions
    assert all(
        row["lbfgs_acceptance"]["accepted_step_norm"]
        <= row["lbfgs_acceptance"]["trust_radius_after"] + 1e-14
        for row in accepted
        if "lbfgs_acceptance" in row
    )
    assert all(
        action["history_size"] > action["history_size_before"]
        for action in pair_actions
    )
    assert all(
        row["lbfgs_trial"]["lbfgs_pending"]
        and "lbfgs_acceptance" not in row
        for row in provisional
    )
    assert all(
        row["lbfgs_trial"]["lbfgs_history_size"]
        <= mc.supercipt_diagnostics["lbfgs_history_size"]
        for row in provisional
    )

    diagnostics = mc.supercipt_diagnostics
    assert diagnostics["acceleration"] == "lbfgs"
    assert diagnostics["line_search_mode"] == "lbfgs"
    assert diagnostics["line_search_c2"] == pytest.approx(0.9)
    assert diagnostics["lbfgs"]
    assert diagnostics["lbfgs_history_size"] == pair_actions[-1][
        "history_size"
    ]
    assert diagnostics["lbfgs_secant_updates"] == len(pair_actions)
    assert diagnostics["energy_evaluations"] == len(history)
    assert diagnostics["hard_evaluation_budget"] == mc.max_cycle_macro
    assert mc.e_tot == pytest.approx(history[-1]["total_energy"], abs=1e-12)
    assert [row["accepted"] for row in callback_rows] == [
        row["accepted"] for row in history
    ]


@pytest.mark.integration
def test_supercipt_lbfgs_uphill_rollback_preserves_memory_and_hard_budget(
    tilted_hf_supercipt,
    monkeypatch,
):
    _, mf, _, mo = tilted_hf_supercipt
    mc = _casscf(mf, mo)
    mc.max_cycle_macro = 3
    callback_rows = []
    evaluation_count = {"value": 0}
    original_fake_casci = zmcscf._fake_h_for_fast_casci

    def fake_casci_with_one_uphill_evaluation(*args, **kwargs):
        mci = original_fake_casci(*args, **kwargs)
        original_kernel = mci.kernel

        def kernel(*kernel_args, **kernel_kwargs):
            result = original_kernel(*kernel_args, **kernel_kwargs)
            evaluation_count["value"] += 1
            if evaluation_count["value"] == 2:
                total, cas, ci = result
                return total + 1.0, cas + 1.0, ci
            return result

        mci.kernel = kernel
        return mci

    monkeypatch.setattr(
        zmcscf,
        "_fake_h_for_fast_casci",
        fake_casci_with_one_uphill_evaluation,
    )
    mc.callback = lambda row: callback_rows.append(dict(row))
    _run_experimental_supercipt(mc, "lbfgs")

    history = mc.supercipt_history
    assert evaluation_count["value"] == mc.max_cycle_macro
    assert len(history) == mc.max_cycle_macro
    assert [row["accepted"] for row in history] == [True, False, True]
    rejected = history[1]
    restored = history[2]
    assert rejected["lbfgs_trial"]["action"] == "restore-base"
    assert rejected["lbfgs_trial"]["lbfgs_history_size"] == 0
    assert rejected["lbfgs_trial"]["lbfgs_pending"]
    deferred = rejected["lbfgs_trial"]["boundary_trust_update"]
    assert deferred["deferred"]
    assert deferred["global_trust_radius_unchanged"] == pytest.approx(
        deferred["base_trust_radius"]
    )
    assert "applied_orbital_step_norm" not in rejected
    assert restored["lbfgs_trial"]["reason"] == "budget-restored-base"
    pair_action = restored["lbfgs_acceptance"]["lbfgs_pair_action"]
    assert not pair_action["accepted"]
    assert pair_action["history_size"] == 0
    assert restored["lbfgs_acceptance"]["accepted_step_norm"] == 0.0
    assert restored["lbfgs_acceptance"]["trust_radius_after"] == pytest.approx(
        0.5 * restored["lbfgs_acceptance"]["trust_radius_before"]
    )
    assert restored["total_energy"] == pytest.approx(
        history[0]["total_energy"], abs=1e-12
    )
    assert restored["rdm_energy_error"] <= 1e-9
    assert np.max(abs(mc.mo_coeff - mo)) <= 1e-10
    diagnostics = mc.supercipt_diagnostics
    assert diagnostics["lbfgs_history_size"] == 0
    assert diagnostics["lbfgs_secant_updates"] == 0
    assert diagnostics["lbfgs_rejected_directions"] == 1
    assert diagnostics["energy_evaluations"] == 3
    assert diagnostics["accepted_evaluations"] == 2
    assert diagnostics["hard_evaluation_budget"] == 3
    assert mc.e_tot == pytest.approx(restored["total_energy"], abs=1e-12)
    assert [row["accepted"] for row in callback_rows] == [True, False, True]


@pytest.mark.integration
def test_supercipt_spectral_cg_physical_trajectory_is_variational(
    tilted_hf_supercipt,
):
    _, mf, _, mo = tilted_hf_supercipt
    mc = _casscf(mf, mo)
    mc.max_cycle_macro = 6
    _run_experimental_supercipt(mc, "spectral-cg")

    history = mc.supercipt_history
    accepted = [row for row in history if row["accepted"]]
    assert len(history) <= mc.max_cycle_macro
    assert history[-1]["accepted"]
    assert all(
        later["total_energy"] <= earlier["total_energy"] + 1e-10
        for earlier, later in zip(accepted, accepted[1:])
    )
    assert all(row["rdm_energy_error"] <= 1e-9 for row in accepted)
    outgoing = [
        row for row in accepted if "applied_orbital_step_norm" in row
    ]
    assert outgoing
    assert all(
        row["applied_orbital_step_norm"] <= mc.max_stepsize + 1e-12
        for row in outgoing
    )
    assert all(
        row["applied_direction"]["total"]["directional_derivative"] < 0.0
        for row in outgoing
    )
    assert mc.e_tot == pytest.approx(history[-1]["total_energy"], abs=1e-12)


@pytest.mark.integration
def test_supercipt_spectral_cg_rejects_uphill_with_hard_budget(
    tilted_hf_supercipt,
    monkeypatch,
):
    """A rejected CG point cannot consume the required fallback budget."""
    _, mf, _, mo = tilted_hf_supercipt
    mc = _casscf(mf, mo)
    mc.max_cycle_macro = 3
    injected = {"done": False}
    callback_rows = []
    original_propose = SpectralOrbitalCG.propose

    def uphill_propose(self, current_mo, gradient, direction, **kwargs):
        result = original_propose(
            self,
            current_mo,
            gradient,
            direction,
            **kwargs,
        )
        if not injected["done"]:
            injected["done"] = True
            generator = np.zeros(
                (current_mo.shape[1],) * 2,
                dtype=np.complex128,
            )
            generator[-1, 0] = 0.6
            generator[0, -1] = -0.6
            diagnostics = dict(result.diagnostics)
            diagnostics.update(
                {
                    "plain_equivalent": False,
                    "guarded": True,
                    "slope": -1.0,
                    "test_uphill_injection": True,
                }
            )
            return result.__class__(
                current_mo.dot(scipy.linalg.expm(generator)),
                generator,
                diagnostics,
            )
        return result

    monkeypatch.setattr(
        SpectralOrbitalCG,
        "propose",
        uphill_propose,
    )
    mc.callback = lambda row: callback_rows.append(dict(row))
    _run_experimental_supercipt(mc, "spectral-cg")

    history = mc.supercipt_history
    rejected = [row for row in history if not row["accepted"]]
    assert injected["done"]
    assert len(history) == mc.max_cycle_macro
    assert [row["accepted"] for row in history] == [True, False, True]
    assert len(rejected) == 1
    assert rejected[0]["spectral_cg_rejection"]["reason"] == "energy-increase"
    assert "applied_orbital_step_norm" not in rejected[0]
    assert history[-1]["total_energy"] < rejected[0]["total_energy"]
    assert history[-1]["rdm_energy_error"] <= 1e-9
    assert mc.e_tot == pytest.approx(history[-1]["total_energy"], abs=1e-12)
    assert mc.supercipt_diagnostics["energy_evaluations"] == 3
    assert mc.supercipt_diagnostics["accepted_evaluations"] == 2
    assert mc.supercipt_diagnostics["spectral_cg_rejected_steps"] == 1
    assert mc.supercipt_diagnostics["hard_evaluation_budget"] == 3
    assert [row["accepted"] for row in callback_rows] == [True, False, True]


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
def test_plain_kramers_supercipt_projects_gradient_before_optimization(
    monkeypatch,
):
    """Project forbidden components at the orbital-gradient boundary."""
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

    def make_casscf(callback=None):
        mc = zmcscf.CASSCF(mf, ncas=2, nelecas=1)
        mc.mo_coeff = initial_mo.copy()
        mc.state_average_([0.5, 0.5])
        mc.natorb = False
        mc.canonicalize_ = False
        mc.max_cycle_macro = 30
        mc.conv_tol = 1e-10
        mc.conv_tol_grad = 1e-7
        mc.max_stepsize = 0.1
        mc.verbose = 0
        mc.callback = callback
        return mc

    clean = make_casscf()
    clean.supercipt(use_diis=False)
    for row in clean.supercipt_history:
        diagnostics = row["kramers_gradient"]
        assert row["raw_orbital_gradient_norm"] == pytest.approx(
            diagnostics["raw_gradient_norm"]
        )
        assert row["orbital_gradient_norm"] == pytest.approx(
            diagnostics["constrained_gradient_norm"]
        )
        assert row[
            "raw_orbital_gradient_frobenius_norm"
        ] == pytest.approx(diagnostics["raw_gradient_frobenius_norm"])
        assert row["orbital_gradient_frobenius_norm"] == pytest.approx(
            diagnostics["constrained_gradient_frobenius_norm"]
        )

    original_screen = zmc_supercipt._screen_orbital_gradient
    injected_components = []

    def screen_with_forbidden_component(
        mc,
        mo,
        gradient,
        *,
        kramers=False,
        kramers_mapping=None,
    ):
        if not kramers:
            return original_screen(
                mc,
                mo,
                gradient,
                kramers=False,
                kramers_mapping=kramers_mapping,
            )
        allowed = mc.uniq_var_indices(
            mo.shape[1], mc.ncore, mc.ncas, mc.frozen
        )
        row, column = np.argwhere(allowed)[0]
        probe = np.zeros_like(gradient)
        probe[row, column] = 1.0 + 0.25j
        probe[column, row] = -probe[row, column].conjugate()
        kr_probe, _ = zmc_supercipt._project_kramers_rotation(
            mc,
            mo,
            probe,
            force=True,
            mapping=kramers_mapping,
        )
        forbidden = probe - kr_probe
        forbidden *= 10.0 / np.linalg.norm(forbidden)
        injected_components.append(np.array(forbidden, copy=True))
        result = original_screen(
            mc,
            mo,
            gradient + forbidden,
            kramers=True,
            kramers_mapping=kramers_mapping,
        )
        result = dict(result)
        diagnostics = dict(result["kramers_gradient_diagnostics"])
        diagnostics["injected_non_kramers_frobenius_norm"] = float(
            np.linalg.norm(forbidden)
        )
        result["kramers_gradient_diagnostics"] = diagnostics
        return result

    monkeypatch.setattr(
        zmc_supercipt,
        "_screen_orbital_gradient",
        screen_with_forbidden_component,
    )
    callback_gradients = []
    restart_solver = DMRGCI(mol)
    restart_solver.dmrg_switch_tol = 1.0
    restart_boundary = []

    def record_callback(environment):
        callback_gradients.append(
            {
                "constrained": environment["orbital_gradient_norm"],
                "raw": environment["raw_orbital_gradient_norm"],
                "nested_raw": environment["kramers_gradient"][
                    "raw_gradient_norm"
                ],
            }
        )
        enabled = restart_solver.restart_scheduler_step(environment)
        restart_boundary.append(
            {
                "enabled": enabled,
                "constrained": environment["orbital_gradient_norm"],
                "raw": environment["raw_orbital_gradient_norm"],
            }
        )

    injected = make_casscf(record_callback)
    injected.supercipt(use_diis=False)
    restart_solver.close()

    assert injected_components
    assert injected.converged is clean.converged
    assert len(clean.supercipt_history) == len(injected.supercipt_history)
    for callback_row, history_row in zip(
        callback_gradients,
        injected.supercipt_history,
    ):
        assert callback_row["constrained"] == pytest.approx(
            history_row["orbital_gradient_norm"], abs=1e-12
        )
        assert callback_row["raw"] == pytest.approx(
            history_row["raw_orbital_gradient_norm"], abs=1e-12
        )
        assert callback_row["nested_raw"] == pytest.approx(
            callback_row["raw"], abs=1e-12
        )
    assert any(
        row["enabled"]
        and row["constrained"] < restart_solver.dmrg_switch_tol
        and row["raw"] > restart_solver.dmrg_switch_tol
        for row in restart_boundary
    )
    assert injected.final_orbital_gradient_norm == pytest.approx(
        clean.final_orbital_gradient_norm,
        abs=1e-11,
    )
    assert injected.e_tot == pytest.approx(clean.e_tot, abs=1e-11)
    assert np.linalg.norm(injected.mo_coeff - clean.mo_coeff) <= 1e-10

    clean_steps = [
        row for row in clean.supercipt_history
        if "plain_pt_direction" in row
    ]
    injected_steps = [
        row for row in injected.supercipt_history
        if "plain_pt_direction" in row
    ]
    assert len(clean_steps) == len(injected_steps)
    for clean_row, injected_row in zip(clean_steps, injected_steps):
        gradient_diagnostics = injected_row["kramers_gradient"]
        assert injected_row[
            "raw_orbital_gradient_frobenius_norm"
        ] > 9.0
        assert injected_row["raw_orbital_gradient_norm"] == pytest.approx(
            gradient_diagnostics["raw_gradient_norm"], abs=1e-12
        )
        assert injected_row[
            "raw_orbital_gradient_frobenius_norm"
        ] == pytest.approx(
            gradient_diagnostics["raw_gradient_frobenius_norm"],
            abs=1e-12,
        )
        assert injected_row["orbital_gradient_norm"] == pytest.approx(
            clean_row["orbital_gradient_norm"], abs=1e-11
        )
        assert injected_row["applied_orbital_step_norm"] == pytest.approx(
            clean_row["applied_orbital_step_norm"], abs=1e-11
        )
        injected_direction = injected_row["plain_pt_direction"]["total"]
        clean_direction = clean_row["plain_pt_direction"]["total"]
        assert injected_direction["directional_derivative"] == pytest.approx(
            clean_direction["directional_derivative"], abs=1e-11
        )
        assert gradient_diagnostics[
            "injected_non_kramers_frobenius_norm"
        ] == pytest.approx(10.0)
        assert gradient_diagnostics["output_gradient_residual"] <= 1e-12


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
    mc.supercipt(use_diis=True)
    converged, energy, final_mo = mc.converged, mc.e_tot, mc.mo_coeff

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
def test_cl_kramers_supercipt_fixed_reference_diis_accelerates_convergence():
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
    plain.supercipt(use_diis=False)

    accelerated = make_casscf()
    accelerated.supercipt(use_diis=True)
    converged = accelerated.converged
    energy = accelerated.e_tot
    final_mo = accelerated.mo_coeff
    mapping = identify_kramers_orbitals(
        mol,
        final_mo,
        mf.get_ovlp(),
        tolerance=1e-7,
    )

    full_superci = make_casscf()
    full_superci.superci(use_diis=True)

    assert not plain.converged
    assert converged and accelerated.converged
    assert full_superci.converged
    assert accelerated.final_orbital_gradient_norm <= accelerated.conv_tol_grad
    assert accelerated.final_orbital_gradient_norm < plain.final_orbital_gradient_norm
    assert abs(energy - (-460.8793608192712)) <= 1e-8
    assert abs(energy - full_superci.e_tot) <= 1e-8
    assert accelerated.supercipt_diagnostics[
        "pt_core_virtual_canonicalization"
    ] is True
    assert accelerated.supercipt_diagnostics[
        "pt_semicanonical_frame_is_temporary"
    ] is True
    assert any(
        row.get("diis", {}).get("coordinate_system")
        == "fixed-reference-unitary-log"
        and row["diis"]["extrapolated"]
        for row in accelerated.supercipt_history
    )
    assert any(
        row.get("diis", {}).get("extrapolated", False)
        for row in full_superci.macro_history
    )
    assert mapping.diagnostics["partner_orbital_error"] <= 1e-7
