"""Perturbative Super-CI orbital optimization for complex spinor CASSCF.

This module implements the two-component Super-CIPT equations of

    Y. Guo and A. K. Dutta, J. Chem. Theory Comput. 22, 7154 (2026),
    https://doi.org/10.1021/acs.jctc.6c00400

which extend the perturbation-based Super-CI construction of Kollmar et al.
to complex spinors.  The implementation follows the RDM convention used by
``socutils.fci.zfci`` and ``socutils.dmrg.DMRGCI``:

``dm1[p,q] = <p^+ q>`` and ``dm2[p,q,r,s] = <p^+ r^+ s q>``.

Super-CIPT is an additional optimizer.  It does not replace the full
Super-CI/Davidson path in :mod:`socutils.mcscf.zmc_superci`.
"""

from dataclasses import dataclass
from functools import reduce

import numpy as np
import scipy.linalg
from pyscf import lib
from pyscf.lib import logger

from socutils.mcscf import zmc_ao2mo
from socutils.mcscf.zmc_superci import (
    _ci_convergence_snapshot,
    _contract_dm2_gradient,
    _subspace_eigh,
)


def _hermitize(matrix):
    matrix = np.asarray(matrix, dtype=np.complex128)
    return 0.5 * (matrix + matrix.T.conj())


def solve_metric_eigenproblem(matrix, metric, metric_tol=1e-6):
    """Solve ``matrix C = metric C e`` in the supported metric range.

    The active 1-RDM and hole-density metrics in the Super-CIPT equations are
    positive semidefinite rather than positive definite.  The caller supplies
    them in the index orientation of the corresponding Koopmans matrix;
    :func:`supercipt_step` therefore passes ``D.T`` and ``(I-D).T`` for this
    repository's ``D[p,q] = <p^+ q>`` convention.  Canonical
    orthogonalization removes eigenvectors whose metric eigenvalue is not
    larger than ``metric_tol``.  Returned columns obey ``C^H metric C = I``.

    Returns ``(eigenvalues, coefficients, diagnostics)``.  An empty supported
    range is represented by arrays with zero columns rather than by an
    ill-conditioned generalized eigensolve.
    """
    matrix = _hermitize(matrix)
    metric = _hermitize(metric)
    if matrix.shape != metric.shape or matrix.ndim != 2:
        raise ValueError("matrix and metric must be square arrays of equal shape")
    if metric_tol <= 0:
        raise ValueError("metric_tol must be positive")

    metric_eigenvalues, metric_vectors = scipy.linalg.eigh(metric)
    if metric_eigenvalues.size and metric_eigenvalues[0] < -metric_tol:
        raise ValueError(
            "Super-CIPT metric is not positive semidefinite: minimum "
            "eigenvalue %.6g" % metric_eigenvalues[0]
        )
    keep = metric_eigenvalues > metric_tol
    rank = int(np.count_nonzero(keep))
    diagnostics = {
        "rank": rank,
        "dimension": int(metric.shape[0]),
        "metric_eigenvalues": metric_eigenvalues.real.tolist(),
        "metric_tolerance": float(metric_tol),
    }
    if rank == 0:
        return (
            np.empty(0, dtype=float),
            np.empty((metric.shape[0], 0), dtype=np.complex128),
            diagnostics,
        )

    orthogonalizer = metric_vectors[:, keep] / np.sqrt(
        metric_eigenvalues[keep]
    )
    reduced = _hermitize(orthogonalizer.T.conj().dot(matrix).dot(orthogonalizer))
    eigenvalues, reduced_vectors = scipy.linalg.eigh(reduced)
    coefficients = orthogonalizer.dot(reduced_vectors)
    diagnostics["orthonormality_error"] = float(
        np.max(abs(coefficients.T.conj().dot(metric).dot(coefficients) - np.eye(rank)))
    )
    return eigenvalues.real, coefficients, diagnostics


@dataclass
class OrbitalQuantities:
    """Intermediates shared by the gradient and Dyall denominators."""

    h1e: np.ndarray
    fock_core: np.ndarray
    fock_effective: np.ndarray
    two_rdm_contraction: np.ndarray
    lagrangian: np.ndarray
    gradient: np.ndarray
    screened_gradient: np.ndarray
    gradient_norm: float


@dataclass
class SuperCIPTStep:
    """One fully diagnosed Super-CIPT orbital update."""

    mo_coeff: np.ndarray
    quantities: OrbitalQuantities
    koopmans_removal: np.ndarray
    koopmans_addition: np.ndarray
    removal_energies: np.ndarray
    addition_energies: np.ndarray
    kappa_unscaled: np.ndarray
    kappa: np.ndarray
    rotation: np.ndarray
    scale: float
    maximum_amplitude: float
    minimum_denominator: float
    metric_diagnostics: dict
    canonical_energies: dict


def build_orbital_quantities(mc, mo, casdm1, casdm2, eris):
    """Build the state-averaged orbital gradient and Dyall Fock matrices.

    ``fock_core`` is :math:`h + f^{occ}` (the one-electron plus inactive
    Coulomb/exchange contribution).  ``fock_effective`` additionally contains
    :math:`f^{act}`.  Contracting ``eris.paaa`` with the 2-RDM produces the
    final active-column contribution to the generalized Fock/Lagrangian
    matrix.  This is the literal tensor structure used by the validated
    historical implementation, but it accepts both full and Cholesky ERIs.
    """
    mo = np.asarray(mo)
    casdm1 = np.asarray(casdm1)
    casdm2 = np.asarray(casdm2)
    ncore, ncas = mc.ncore, mc.ncas
    nocc = ncore + ncas
    nmo = mo.shape[1]

    h1e = reduce(np.dot, (mo.T.conj(), mc.get_hcore(), mo))
    dm_core_mo = np.zeros((nmo, nmo), dtype=np.complex128)
    dm_core_mo[np.arange(ncore), np.arange(ncore)] = 1.0
    dm_core_ao = reduce(np.dot, (mo, dm_core_mo, mo.T.conj()))
    core_occ = np.zeros(nmo)
    core_occ[:ncore] = 1.0
    vj_core, vk_core = eris.get_jk(
        dm_core_ao, mo_coeff=mo, mo_occ=core_occ
    )
    vhf_core = reduce(np.dot, (mo.T.conj(), vj_core - vk_core, mo))
    # ``casdm1[p,q] = <p^+ q>`` whereas PySCF's JK builders consume the
    # covariant density with the annihilation index first.  The transpose is
    # immaterial for real orbitals but essential for complex spinors.
    vj_active, vk_active = eris.get_jk_active_mo(casdm1.T)
    vhf_active = vj_active - vk_active

    fock_core = _hermitize(h1e + vhf_core)
    fock_effective = _hermitize(fock_core + vhf_active)
    two_rdm = _contract_dm2_gradient(eris, casdm2)

    lagrangian = np.zeros((nmo, nmo), dtype=np.complex128)
    lagrangian[:, :ncore] = fock_effective[:, :ncore]
    lagrangian[:, ncore:nocc] = (
        fock_core[:, ncore:nocc].dot(casdm1.T) + two_rdm
    )
    gradient = lagrangian - lagrangian.T.conj()

    allowed = mc.uniq_var_indices(nmo, ncore, ncas, mc.frozen)
    lower = np.zeros_like(gradient)
    lower[allowed] = gradient[allowed]
    screened = lower - lower.T.conj()
    return OrbitalQuantities(
        h1e=h1e,
        fock_core=fock_core,
        fock_effective=fock_effective,
        two_rdm_contraction=two_rdm,
        lagrangian=lagrangian,
        gradient=gradient,
        screened_gradient=screened,
        gradient_norm=float(np.linalg.norm(screened)),
    )


def build_koopmans_matrices(mc, quantities, casdm1):
    """Return the removal/addition Koopmans matrices (paper eqs. 22--23)."""
    ncore, ncas = mc.ncore, mc.ncas
    active = slice(ncore, ncore + ncas)
    fock_active = quantities.fock_core[active, active]
    effective_active = quantities.fock_effective[active, active]
    q_active = quantities.two_rdm_contraction[active, :]

    # This is the complex-spinor form used in the source implementation:
    # K = -1/2[(f D^T) + (f D^T)^H] - 1/2[Q + Q^H].
    product = -fock_active.dot(np.asarray(casdm1).T)
    removal = _hermitize(product) - _hermitize(q_active)
    addition = _hermitize(removal + effective_active)
    return removal, addition


def _shift_denominator(denominator, level_shift):
    """Move a real denominator away from zero without changing its sign."""
    denominator = np.asarray(denominator, dtype=float)
    if level_shift == 0:
        return denominator
    sign = np.where(denominator < 0, -1.0, 1.0)
    return denominator + sign * float(level_shift)


def _check_denominators(denominator, tolerance, label):
    if denominator.size == 0:
        return np.inf
    minimum = float(np.min(abs(denominator)))
    if minimum <= tolerance:
        position = np.unravel_index(np.argmin(abs(denominator)), denominator.shape)
        raise RuntimeError(
            "%s Super-CIPT denominator is singular at %s: %.6g"
            % (label, position, denominator[position])
        )
    return minimum


def _canonicalize_core_virtual(mc, mo, casdm1):
    """Canonicalize the redundant core/virtual blocks required by PT.

    This is an internal condition of the Dyall denominators, not the optional
    final-orbital presentation controlled by ``mc.canonicalize_``.
    """
    ncore, ncas = mc.ncore, mc.ncas
    nocc = ncore + ncas
    nmo = mo.shape[1]
    dm_core = mo[:, :ncore].dot(mo[:, :ncore].T.conj())
    mo_active = mo[:, ncore:nocc]
    dm_active = reduce(
        np.dot, (mo_active, np.asarray(casdm1).T, mo_active.T.conj())
    )
    vj_core, vk_core = mc.get_jk(mc.mol, dm_core)
    vj_active, vk_active = mc.get_jk(mc.mol, dm_active)
    fock_ao = mc.get_hcore() + vj_core - vk_core + vj_active - vk_active
    fock_mo = _hermitize(reduce(np.dot, (mo.T.conj(), fock_ao, mo)))

    result = np.array(mo, copy=True)
    energies = {"core": [], "virtual": []}
    if ncore:
        values, vectors = _subspace_eigh(
            mc, fock_mo[:ncore, :ncore], mo[:, :ncore]
        )
        result[:, :ncore] = mo[:, :ncore].dot(vectors)
        energies["core"] = np.asarray(values).real.tolist()
    if nocc < nmo:
        values, vectors = _subspace_eigh(
            mc, fock_mo[nocc:, nocc:], mo[:, nocc:]
        )
        result[:, nocc:] = mo[:, nocc:].dot(vectors)
        energies["virtual"] = np.asarray(values).real.tolist()
    return result, energies


def supercipt_step(
    mc,
    mo,
    casdm1,
    casdm2,
    eris,
    *,
    max_stepsize=0.2,
    level_shift=0.0,
    metric_tol=1e-6,
    denominator_tol=1e-10,
    canonicalize=True,
):
    """Form and apply one perturbative Super-CI orbital step.

    The three blocks correspond directly to paper eqs. 24--26.  In the stored
    Koopmans-matrix orientation, the active solves use ``D.T`` and
    ``(I-D).T`` as the particle and hole metrics.  Only interspace rotations
    selected by ``mc.uniq_var_indices`` survive.  The historical
    maximum-element trust bound is retained, followed by the unitary update
    ``C <- C exp(kappa)``.
    """
    if max_stepsize <= 0:
        raise ValueError("max_stepsize must be positive")
    if level_shift < 0 or denominator_tol < 0:
        raise ValueError("level_shift and denominator_tol must be nonnegative")

    quantities = build_orbital_quantities(mc, mo, casdm1, casdm2, eris)
    removal, addition = build_koopmans_matrices(mc, quantities, casdm1)
    ncore, ncas = mc.ncore, mc.ncas
    nocc = ncore + ncas
    nmo = mo.shape[1]
    active = slice(ncore, nocc)
    virtual = slice(nocc, nmo)
    density = _hermitize(casdm1)
    # K[t,u] is stored with the reverse orientation of the projected
    # (N-1)-electron Hamiltonian, while dm1[p,q] = <p^+ q>.  Consequently
    # the generalized removal problem is -K C = D^T C e.  Using D instead
    # gives the right scalar equations but incorrect ionization energies and
    # rotations for complex spinors.
    removal_metric = density.T
    holes = _hermitize(np.eye(ncas) - density)
    addition_metric = holes.T

    removal_e, removal_c, removal_info = solve_metric_eigenproblem(
        -removal, removal_metric, metric_tol
    )
    addition_e, addition_c, addition_info = solve_metric_eigenproblem(
        addition, addition_metric, metric_tol
    )

    lower = np.zeros((nmo, nmo), dtype=np.complex128)
    minimum_denominator = np.inf
    diagonal = np.diag(quantities.fock_effective).real

    # Core -> virtual, eq. 24.
    if ncore and nocc < nmo:
        denominator = _shift_denominator(
            diagonal[:ncore][None, :] - diagonal[nocc:, None], level_shift
        )
        minimum_denominator = min(
            minimum_denominator,
            _check_denominators(denominator, denominator_tol, "core-virtual"),
        )
        lower[virtual, :ncore] = (
            quantities.gradient[virtual, :ncore] / denominator
        )

    # Core -> active, eq. 25 (electron-addition/hole metric).
    if ncore and addition_e.size:
        block = quantities.gradient[active, :ncore]
        transformed = addition_c.T.conj().dot(block)
        overlap = addition_metric.dot(addition_c)
        denominator = _shift_denominator(
            diagonal[:ncore][None, :] - addition_e[:, None], level_shift
        )
        minimum_denominator = min(
            minimum_denominator,
            _check_denominators(denominator, denominator_tol, "core-active"),
        )
        lower[active, :ncore] = overlap.dot(transformed / denominator)
    elif ncore and np.linalg.norm(quantities.gradient[active, :ncore]) > metric_tol:
        raise RuntimeError(
            "core-active gradient is nonzero but the hole-density metric has rank zero"
        )

    # Active -> virtual, eq. 26 (ionization/particle metric).
    if nocc < nmo and removal_e.size:
        block = quantities.gradient[virtual, active]
        transformed = removal_c.T.conj().dot(block.T)
        overlap = removal_metric.dot(removal_c)
        denominator = _shift_denominator(
            removal_e[:, None] - diagonal[nocc:][None, :], level_shift
        )
        minimum_denominator = min(
            minimum_denominator,
            _check_denominators(denominator, denominator_tol, "active-virtual"),
        )
        lower[virtual, active] = (
            overlap.dot(transformed / denominator)
        ).T
    elif nocc < nmo and np.linalg.norm(quantities.gradient[virtual, active]) > metric_tol:
        raise RuntimeError(
            "active-virtual gradient is nonzero but the density metric has rank zero"
        )

    allowed = mc.uniq_var_indices(nmo, ncore, ncas, mc.frozen)
    lower[~allowed] = 0.0
    kappa_unscaled = lower - lower.T.conj()
    maximum = float(np.max(abs(kappa_unscaled))) if kappa_unscaled.size else 0.0
    scale = min(1.0, float(max_stepsize) / maximum) if maximum else 1.0
    kappa = scale * kappa_unscaled
    rotation = scipy.linalg.expm(kappa)
    mo_new = np.asarray(mo).dot(rotation)
    canonical_energies = {"core": [], "virtual": []}
    if canonicalize:
        mo_new, canonical_energies = _canonicalize_core_virtual(
            mc, mo_new, casdm1
        )

    return SuperCIPTStep(
        mo_coeff=mo_new,
        quantities=quantities,
        koopmans_removal=removal,
        koopmans_addition=addition,
        removal_energies=removal_e,
        addition_energies=addition_e,
        kappa_unscaled=kappa_unscaled,
        kappa=kappa,
        rotation=rotation,
        scale=scale,
        maximum_amplitude=maximum,
        minimum_denominator=float(minimum_denominator),
        metric_diagnostics={"removal": removal_info, "addition": addition_info},
        canonical_energies=canonical_energies,
    )


def _build_eris(mc, mo, cderi=None):
    with_df = getattr(mc._scf, "with_df", None)
    if with_df is not None or cderi is not None:
        eris = zmc_ao2mo._CDERIS(mc, mo, cderi=cderi, level=2)
        return eris, {
            "factorized": True,
            "source": type(with_df).__name__ if with_df is not None else "legacy-cderi",
            "naux": int(eris.cd_pa.shape[0]),
            "threshold": getattr(with_df, "tau", None),
        }
    eris = zmc_ao2mo._ERIS(mc, mo, level=2)
    return eris, {
        "factorized": False,
        "source": "full-integral",
        "naux": None,
        "threshold": None,
    }


def _scalar_energy(value, label):
    array = np.asarray(value)
    if array.ndim:
        raise ValueError(
            "%s is multi-root; call state_average_(weights) before Super-CIPT"
            % label
        )
    if abs(np.imag(array)) > 1e-10:
        raise RuntimeError("%s has a non-negligible imaginary part" % label)
    return float(np.real(array))


def _root_energies(solver):
    values = getattr(solver, "e_states", None)
    if values is None:
        values = getattr(solver, "e_tot", None)
    if values is None or np.asarray(values).ndim == 0:
        return None
    return np.asarray(values).real.tolist()


def _reject_kramers_restricted(mc):
    from socutils.scf import spinor_hf

    if isinstance(mc._scf, spinor_hf.KRHF) or getattr(
        mc.fcisolver, "kramers_adapter", None
    ) is not None:
        raise NotImplementedError(
            "Kramers-restricted Super-CIPT orbital equations are not implemented; "
            "use the general complex solver/reference or the validated Super-CI path"
        )


def mcscf_supercipt(
    mc,
    mo_coeff,
    *,
    max_stepsize=0.2,
    conv_tol=None,
    conv_tol_grad=None,
    max_cycle=None,
    level_shift=0.0,
    metric_tol=1e-6,
    denominator_tol=1e-10,
    verbose=None,
    cderi=None,
    callback=None,
):
    """Drive state-specific or state-averaged 2C-CASSCF with Super-CIPT."""
    _reject_kramers_restricted(mc)
    log = logger.new_logger(mc, verbose)
    if conv_tol is None:
        conv_tol = mc.conv_tol
    if conv_tol_grad is None:
        conv_tol_grad = np.sqrt(conv_tol)
    if max_cycle is None:
        max_cycle = mc.max_cycle_macro
    if max_cycle <= 0:
        raise ValueError("max_cycle must be positive")

    mo = np.array(mo_coeff, dtype=np.complex128, copy=True)
    previous_energy = None
    history = []
    mc.macro_history = history
    mc.supercipt_history = history
    converged = False
    last_step = None
    last_integral_info = None
    e_tot = e_cas = ci = casdm1 = casdm2 = quantities = None

    log.info("Super-CIPT optimizer (Guo-Dutta 2026), max cycles = %d", max_cycle)
    log.info(
        "Super-CIPT metric/denominator tolerance = %.3g / %.3g, level shift = %.3g",
        metric_tol,
        denominator_tol,
        level_shift,
    )
    log.info(
        "Super-CIPT core/virtual PT canonicalization is enabled "
        "independently of canonicalize_"
    )

    for macro in range(max_cycle):
        eris, integral_info = _build_eris(mc, mo, cderi=cderi)
        last_integral_info = integral_info
        from socutils.mcscf.zmcscf import _fake_h_for_fast_casci

        mci = _fake_h_for_fast_casci(mc, mo, eris)
        e_tot_raw, e_cas_raw, ci = mci.kernel(mo, ci0=None, verbose=verbose)
        e_tot = _scalar_energy(e_tot_raw, "total energy")
        e_cas = _scalar_energy(e_cas_raw, "CAS energy")
        ci_converged = bool(np.all(getattr(mc.fcisolver, "converged", True)))
        if not ci_converged:
            raise RuntimeError("The active-space CI solver did not converge")
        casdm1, casdm2 = mc.fcisolver.make_rdm12(ci, mc.ncas, mc.nelecas)
        quantities = build_orbital_quantities(mc, mo, casdm1, casdm2, eris)
        energy_change = (
            None if previous_energy is None else float(e_tot - previous_energy)
        )
        occupations = np.linalg.eigvalsh(_hermitize(casdm1)).real[::-1]
        h1eff, ecore = mci.get_h1eff(mo)
        rdm_energy = (
            np.einsum("pq,pq->", h1eff, casdm1)
            + 0.5 * np.einsum("pqrs,pqrs->", eris.aaaa, casdm2)
            + ecore
        )
        rdm_energy_error = float(abs(rdm_energy - e_tot))
        entry = {
            "macro_iteration": int(macro),
            "total_energy": e_tot,
            "cas_energy": e_cas,
            "root_energies": _root_energies(mc.fcisolver),
            "energy_change": energy_change,
            "orbital_gradient_norm": quantities.gradient_norm,
            "natural_occupations": occupations.tolist(),
            "rdm_energy_error": rdm_energy_error,
            "ci_solver_converged": ci_converged,
            "ci_solver_diagnostics": _ci_convergence_snapshot(mc.fcisolver),
            "integrals": dict(integral_info),
            "converged": False,
        }
        history.append(entry)
        log.info(
            "Super-CIPT macro %d  E = %.15g  dE = %s  |g| = %.6g",
            macro,
            e_tot,
            "---" if energy_change is None else "%.6g" % energy_change,
            quantities.gradient_norm,
        )

        if (
            energy_change is not None
            and abs(energy_change) < conv_tol
            and quantities.gradient_norm < conv_tol_grad
        ):
            converged = True
            entry["converged"] = True
            if callback is not None:
                callback(dict(entry))
            break
        if macro + 1 == max_cycle:
            if callback is not None:
                callback(dict(entry))
            break

        last_step = supercipt_step(
            mc,
            mo,
            casdm1,
            casdm2,
            eris,
            max_stepsize=max_stepsize,
            level_shift=level_shift,
            metric_tol=metric_tol,
            denominator_tol=denominator_tol,
            canonicalize=True,
        )
        entry.update(
            {
                "proposed_maximum_amplitude": last_step.maximum_amplitude,
                "step_scale": last_step.scale,
                "applied_orbital_step_norm": float(np.linalg.norm(last_step.kappa)),
                "minimum_denominator": last_step.minimum_denominator,
                "metric": last_step.metric_diagnostics,
                "removal_energies": last_step.removal_energies.tolist(),
                "addition_energies": last_step.addition_energies.tolist(),
                "canonical_energies": last_step.canonical_energies,
            }
        )
        log.info(
            "Super-CIPT step  %d  max(raw) = %.6g  scale = %.6g  "
            "|kappa| = %.6g  min|denom| = %.6g  metric ranks = %d/%d",
            macro,
            last_step.maximum_amplitude,
            last_step.scale,
            np.linalg.norm(last_step.kappa),
            last_step.minimum_denominator,
            last_step.metric_diagnostics["removal"]["rank"],
            last_step.metric_diagnostics["addition"]["rank"],
        )
        if callback is not None:
            callback(dict(entry))
        previous_energy = e_tot
        mo = last_step.mo_coeff

    mc.converged = converged
    mc.e_tot = e_tot
    mc.e_cas = e_cas
    mc.ci = ci
    mc.mo_coeff = mo
    mc.mo_energy = np.diag(quantities.fock_effective).real
    mc.final_orbital_gradient_norm = quantities.gradient_norm
    mc.cholesky_diagnostics = last_integral_info
    mc.supercipt_diagnostics = {
        "converged": bool(converged),
        "macro_iterations": len(history),
        "final_gradient_norm": float(quantities.gradient_norm),
        "energy_tolerance": float(conv_tol),
        "gradient_tolerance": float(conv_tol_grad),
        "metric_tolerance": float(metric_tol),
        "denominator_tolerance": float(denominator_tol),
        "level_shift": float(level_shift),
        "maximum_step": float(max_stepsize),
        "pt_core_virtual_canonicalization": True,
        "integrals": dict(last_integral_info),
        "kramers_restricted": False,
        "last_step_scale": None if last_step is None else last_step.scale,
    }
    return converged, e_tot, e_cas, ci, mo, mc.mo_energy
