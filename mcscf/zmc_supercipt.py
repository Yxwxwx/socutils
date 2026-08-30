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
from socutils.mcscf.orbital_diis import (
    AndersonOrbitalDIIS,
    IncrementalOrbitalDIIS,
)
from socutils.mcscf.zmc_superci import (
    _ci_convergence_snapshot,
    _contract_dm2_gradient,
    _identify_kramers_mapping,
    _kramers_subspace_eigh,
    _project_kramers_rotation,
    _resolve_kramers_mode,
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
    raw_screened_gradient: np.ndarray
    screened_gradient: np.ndarray
    raw_gradient_norm: float
    raw_gradient_frobenius_norm: float
    gradient_norm: float
    gradient_frobenius_norm: float
    kramers_gradient_diagnostics: dict | None


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
    unscaled_step_norm: float
    minimum_denominator: float
    metric_diagnostics: dict
    canonical_energies: dict
    semicanonical_diagnostics: dict
    direction_diagnostics: dict
    kramers_diagnostics: dict | None


def _project_kramers_gradient(mc, mo, raw_gradient, mapping):
    """Project one evaluated screened gradient into the KR tangent space.

    ``mapping`` belongs to this exact MO point.  The shared Super-CI
    projector constructs the phase-resolved time-reversal matrix from that
    mapping; keeping this wrapper gradient-specific avoids a second mapping
    implementation while giving diagnostics unambiguous gradient names.
    """
    constrained, projection = _project_kramers_rotation(
        mc,
        mo,
        raw_gradient,
        force=True,
        mapping=mapping,
    )
    raw_packed = mc.pack_uniq_var(raw_gradient)
    constrained_packed = mc.pack_uniq_var(constrained)
    diagnostics = {
        "input_gradient_residual": projection[
            "input_generator_residual"
        ],
        "output_gradient_residual": projection[
            "output_generator_residual"
        ],
        "projection_change_norm": projection[
            "projection_change_norm"
        ],
        "raw_gradient_norm": float(np.linalg.norm(raw_packed)),
        "constrained_gradient_norm": float(
            np.linalg.norm(constrained_packed)
        ),
        "raw_gradient_frobenius_norm": float(
            np.linalg.norm(raw_gradient)
        ),
        "constrained_gradient_frobenius_norm": float(
            np.linalg.norm(constrained)
        ),
        "orbital_closure_before_projection": projection[
            "orbital_closure_before_step"
        ],
        "orbital_partner_error_before_projection": projection[
            "orbital_partner_error_before_step"
        ],
        "pairs": projection["pairs"],
    }
    return constrained, diagnostics


def _screen_orbital_gradient(
    mc,
    mo,
    gradient,
    *,
    kramers=False,
    kramers_mapping=None,
):
    """Screen one raw orbital gradient and apply the optional KR constraint.

    This is the single boundary at which the Lagrangian gradient becomes an
    optimizer tangent.  Both raw diagnostics and constrained optimizer norms
    are consequently derived from exactly the arrays returned here.
    """
    nmo = mo.shape[1]
    allowed = mc.uniq_var_indices(
        nmo,
        mc.ncore,
        mc.ncas,
        mc.frozen,
    )
    lower = np.zeros_like(gradient)
    lower[allowed] = gradient[allowed]
    raw_screened = lower - lower.T.conj()
    screened = raw_screened
    kramers_gradient_diagnostics = None
    if kramers:
        screened, kramers_gradient_diagnostics = _project_kramers_gradient(
            mc,
            mo,
            raw_screened,
            kramers_mapping,
        )

    return {
        "raw_screened_gradient": raw_screened,
        "screened_gradient": screened,
        "raw_gradient_norm": float(
            np.linalg.norm(mc.pack_uniq_var(raw_screened))
        ),
        "raw_gradient_frobenius_norm": float(
            np.linalg.norm(raw_screened)
        ),
        "gradient_norm": float(
            np.linalg.norm(mc.pack_uniq_var(screened))
        ),
        "gradient_frobenius_norm": float(np.linalg.norm(screened)),
        "kramers_gradient_diagnostics": kramers_gradient_diagnostics,
    }


def build_orbital_quantities(
    mc,
    mo,
    casdm1,
    casdm2,
    eris,
    *,
    kramers=False,
    kramers_mapping=None,
):
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

    gradient_screening = _screen_orbital_gradient(
        mc,
        mo,
        gradient,
        kramers=kramers,
        kramers_mapping=kramers_mapping,
    )
    # Optimizer thresholds in PySCF and in the full Super-CI driver refer to
    # the norm of the independent orbital variables.  The Frobenius norm of
    # the expanded anti-Hermitian matrix counts every variable and its
    # conjugate partner and is therefore sqrt(2) larger.  Keep the expanded
    # norm for diagnostics, but do not use it for convergence, DIIS start, or
    # the DMRG restart callback.
    return OrbitalQuantities(
        h1e=h1e,
        fock_core=fock_core,
        fock_effective=fock_effective,
        two_rdm_contraction=two_rdm,
        lagrangian=lagrangian,
        gradient=gradient,
        raw_screened_gradient=gradient_screening[
            "raw_screened_gradient"
        ],
        screened_gradient=gradient_screening["screened_gradient"],
        raw_gradient_norm=gradient_screening["raw_gradient_norm"],
        raw_gradient_frobenius_norm=gradient_screening[
            "raw_gradient_frobenius_norm"
        ],
        gradient_norm=gradient_screening["gradient_norm"],
        gradient_frobenius_norm=gradient_screening[
            "gradient_frobenius_norm"
        ],
        kramers_gradient_diagnostics=gradient_screening[
            "kramers_gradient_diagnostics"
        ],
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


def _pt_semicanonical_frame(mc, mo, quantities, *, kramers=False):
    """Return the temporary core/virtual frame required by Dyall PT.

    Core--core and virtual--virtual rotations are redundant CASSCF gauge
    transformations.  The Super-CIPT denominators nevertheless require the
    effective Fock matrix to be diagonal in those two spaces.  Construct that
    frame only for the perturbative solve and later transform the physical
    interspace generator back to the caller's MO gauge.  Applying the
    redundant rotation to ``mo`` itself would contaminate fixed-reference
    acceleration coordinates and make the step depend on the incoming gauge.
    """
    ncore, ncas = mc.ncore, mc.ncas
    nocc = ncore + ncas
    nmo = mo.shape[1]
    transform = np.eye(nmo, dtype=np.complex128)
    energies = {"core": [], "virtual": []}
    diagonalizer = _kramers_subspace_eigh if kramers else _subspace_eigh
    if ncore:
        values, vectors = diagonalizer(
            mc,
            quantities.fock_effective[:ncore, :ncore],
            mo[:, :ncore],
        )
        transform[:ncore, :ncore] = vectors
        energies["core"] = np.asarray(values).real.tolist()
    if nocc < nmo:
        values, vectors = diagonalizer(
            mc,
            quantities.fock_effective[nocc:, nocc:],
            mo[:, nocc:],
        )
        transform[nocc:, nocc:] = vectors
        energies["virtual"] = np.asarray(values).real.tolist()

    fock = quantities.fock_effective
    fock_canonical = reduce(
        np.dot, (transform.T.conj(), fock, transform)
    )
    # The screened gradient is the optimizer's authoritative tangent.  In KR
    # mode it has already been projected, at this evaluated MO point, with
    # the phase-resolved AO time-reversal mapping.  Using the unscreened raw
    # Lagrangian gradient here would reintroduce precisely the forbidden
    # component before the perturbative solve.
    gradient_canonical = reduce(
        np.dot,
        (transform.T.conj(), quantities.screened_gradient, transform),
    )

    def offdiagonal_norm(matrix):
        return float(np.linalg.norm(matrix - np.diag(np.diag(matrix))))

    diagnostics = {
        "core_fock_offdiagonal_before": offdiagonal_norm(
            fock[:ncore, :ncore]
        ),
        "core_fock_offdiagonal_after": offdiagonal_norm(
            fock_canonical[:ncore, :ncore]
        ),
        "virtual_fock_offdiagonal_before": offdiagonal_norm(
            fock[nocc:, nocc:]
        ),
        "virtual_fock_offdiagonal_after": offdiagonal_norm(
            fock_canonical[nocc:, nocc:]
        ),
    }
    return (
        transform,
        gradient_canonical,
        np.asarray(energies["core"], dtype=float),
        np.asarray(energies["virtual"], dtype=float),
        energies,
        diagnostics,
    )


def _direction_diagnostics(mc, gradient, generator):
    """Resolve a proposed direction into the three independent CASSCF blocks."""
    ncore, ncas = mc.ncore, mc.ncas
    nocc = ncore + ncas
    nmo = generator.shape[0]
    blocks = {
        "core_active": (slice(ncore, nocc), slice(0, ncore)),
        "core_virtual": (slice(nocc, nmo), slice(0, ncore)),
        "active_virtual": (slice(nocc, nmo), slice(ncore, nocc)),
    }
    result = {}
    for label, indices in blocks.items():
        gradient_block = np.asarray(gradient[indices]).ravel()
        generator_block = np.asarray(generator[indices]).ravel()
        gradient_norm = float(np.linalg.norm(gradient_block))
        generator_norm = float(np.linalg.norm(generator_block))
        directional_derivative = float(
            np.vdot(gradient_block, generator_block).real
        )
        denominator = gradient_norm * generator_norm
        result[label] = {
            "gradient_norm": gradient_norm,
            "step_norm": generator_norm,
            "directional_derivative": directional_derivative,
            "cosine": (
                None if denominator == 0.0 else directional_derivative / denominator
            ),
        }
    packed_gradient = mc.pack_uniq_var(gradient)
    packed_generator = mc.pack_uniq_var(generator)
    total_gradient_norm = float(np.linalg.norm(packed_gradient))
    total_generator_norm = float(np.linalg.norm(packed_generator))
    total_derivative = float(
        np.vdot(packed_gradient, packed_generator).real
    )
    total_denominator = total_gradient_norm * total_generator_norm
    result["total"] = {
        "gradient_norm": total_gradient_norm,
        "step_norm": total_generator_norm,
        "directional_derivative": total_derivative,
        "cosine": (
            None
            if total_denominator == 0.0
            else total_derivative / total_denominator
        ),
    }
    return result


def _resolve_line_search_trust(
    base_trust_radius,
    accepted_step_norm,
    *,
    accepted_on_boundary,
    linear_ratio,
    boundary_failed,
    max_stepsize,
):
    """Commit one transactional trust update after a line search resolves."""
    base = float(base_trust_radius)
    step = float(accepted_step_norm)
    maximum = float(max_stepsize)
    if not np.isfinite(base) or base <= 0.0:
        raise ValueError("base trust radius must be positive and finite")
    if not np.isfinite(step) or step < 0.0:
        raise ValueError("accepted step norm must be nonnegative and finite")
    if not np.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("maximum step must be positive and finite")
    if step > maximum * (1.0 + 1e-10):
        raise ValueError("accepted step exceeds the maximum step")
    if linear_ratio is not None:
        linear_ratio = float(linear_ratio)
        if not np.isfinite(linear_ratio):
            raise ValueError("linear reduction ratio must be finite")

    half = max(np.finfo(float).eps, 0.5 * base)
    restored_base = step <= np.finfo(float).tiny
    if restored_base:
        candidate = half
        action = "halved-base-restore"
    elif boundary_failed:
        candidate = max(half, step)
        action = (
            "halved-after-boundary-failure"
            if candidate <= half
            else "boundary-failure-raised-to-accepted-step"
        )
    elif linear_ratio is not None and linear_ratio < 0.1:
        candidate = max(half, step)
        action = (
            "halved-poor-ratio"
            if candidate <= half
            else "poor-ratio-raised-to-accepted-step"
        )
    elif (
        accepted_on_boundary
        and linear_ratio is not None
        and linear_ratio > 0.5
    ):
        candidate = min(maximum, 2.0 * base)
        action = (
            "doubled-good-boundary"
            if candidate > base
            else "unchanged-at-maximum"
        )
    else:
        candidate = base
        action = "unchanged"

    next_trust = min(maximum, max(candidate, step))
    if next_trust + 1e-14 < step:
        raise RuntimeError("resolved trust radius is smaller than accepted step")
    return {
        "base_trust_radius": base,
        "accepted_step_norm": step,
        "boundary_failed": bool(boundary_failed),
        "restored_base": restored_base,
        "linear_ratio": linear_ratio,
        "trust_radius_after": float(next_trust),
        "trust_action": action,
    }


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
    kramers=False,
    kramers_mapping=None,
):
    """Form and apply one perturbative Super-CI orbital step.

    The three blocks correspond directly to paper eqs. 24--26.  In the stored
    Koopmans-matrix orientation, the active solves use ``D.T`` and
    ``(I-D).T`` as the particle and hole metrics.  Only interspace rotations
    selected by ``mc.uniq_var_indices`` survive.  The core and virtual
    denominators are formed in a temporary semicanonical
    frame and the resulting generator is transformed back to the input MO
    gauge.  A Frobenius trust radius, matching full Super-CI's step norm,
    precedes the unitary update ``C <- C exp(kappa)``.
    """
    if max_stepsize <= 0:
        raise ValueError("max_stepsize must be positive")
    if level_shift < 0 or denominator_tol < 0:
        raise ValueError("level_shift and denominator_tol must be nonnegative")

    quantities = build_orbital_quantities(
        mc,
        mo,
        casdm1,
        casdm2,
        eris,
        kramers=kramers,
        kramers_mapping=kramers_mapping,
    )
    removal, addition = build_koopmans_matrices(mc, quantities, casdm1)
    ncore, ncas = mc.ncore, mc.ncas
    nocc = ncore + ncas
    nmo = mo.shape[1]
    active = slice(ncore, nocc)
    virtual = slice(nocc, nmo)
    (
        pt_transform,
        gradient_canonical,
        core_energies,
        virtual_energies,
        canonical_energies,
        semicanonical_diagnostics,
    ) = _pt_semicanonical_frame(
        mc, mo, quantities, kramers=kramers
    )
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

    # Core -> virtual, eq. 24.
    if ncore and nocc < nmo:
        denominator = _shift_denominator(
            core_energies[None, :] - virtual_energies[:, None], level_shift
        )
        minimum_denominator = min(
            minimum_denominator,
            _check_denominators(denominator, denominator_tol, "core-virtual"),
        )
        lower[virtual, :ncore] = (
            gradient_canonical[virtual, :ncore] / denominator
        )

    # Core -> active, eq. 25 (electron-addition/hole metric).
    if ncore and addition_e.size:
        block = gradient_canonical[active, :ncore]
        transformed = addition_c.T.conj().dot(block)
        overlap = addition_metric.dot(addition_c)
        denominator = _shift_denominator(
            core_energies[None, :] - addition_e[:, None], level_shift
        )
        minimum_denominator = min(
            minimum_denominator,
            _check_denominators(denominator, denominator_tol, "core-active"),
        )
        lower[active, :ncore] = overlap.dot(transformed / denominator)
    elif ncore and np.linalg.norm(gradient_canonical[active, :ncore]) > metric_tol:
        raise RuntimeError(
            "core-active gradient is nonzero but the hole-density metric has rank zero"
        )

    # Active -> virtual, eq. 26 (ionization/particle metric).
    if nocc < nmo and removal_e.size:
        block = gradient_canonical[virtual, active]
        transformed = removal_c.T.conj().dot(block.T)
        overlap = removal_metric.dot(removal_c)
        denominator = _shift_denominator(
            removal_e[:, None] - virtual_energies[None, :], level_shift
        )
        minimum_denominator = min(
            minimum_denominator,
            _check_denominators(denominator, denominator_tol, "active-virtual"),
        )
        lower[virtual, active] = (
            overlap.dot(transformed / denominator)
        ).T
    elif nocc < nmo and np.linalg.norm(gradient_canonical[virtual, active]) > metric_tol:
        raise RuntimeError(
            "active-virtual gradient is nonzero but the density metric has rank zero"
        )

    allowed = mc.uniq_var_indices(nmo, ncore, ncas, mc.frozen)
    lower[~allowed] = 0.0
    kappa_canonical = lower - lower.T.conj()
    kappa_unscaled = reduce(
        np.dot,
        (pt_transform, kappa_canonical, pt_transform.T.conj()),
    )
    # Remove roundoff-level redundant components after returning to the input
    # gauge.  This is also the authoritative frozen-orbital screening.
    kappa_unscaled = mc.unpack_uniq_var(
        mc.pack_uniq_var(kappa_unscaled)
    )
    kramers_diagnostics = None
    if kramers:
        kappa_unscaled, kramers_diagnostics = _project_kramers_rotation(
            mc,
            mo,
            kappa_unscaled,
            force=True,
            mapping=kramers_mapping,
        )
    direction_diagnostics = _direction_diagnostics(
        mc, quantities.screened_gradient, kappa_unscaled
    )
    maximum = float(np.max(abs(kappa_unscaled))) if kappa_unscaled.size else 0.0
    unscaled_norm = float(np.linalg.norm(kappa_unscaled))
    scale = (
        min(1.0, float(max_stepsize) / unscaled_norm)
        if unscaled_norm
        else 1.0
    )
    kappa = scale * kappa_unscaled
    rotation = scipy.linalg.expm(kappa)
    mo_new = np.asarray(mo).dot(rotation)

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
        unscaled_step_norm=unscaled_norm,
        minimum_denominator=float(minimum_denominator),
        metric_diagnostics={"removal": removal_info, "addition": addition_info},
        canonical_energies=canonical_energies,
        semicanonical_diagnostics=semicanonical_diagnostics,
        direction_diagnostics=direction_diagnostics,
        kramers_diagnostics=kramers_diagnostics,
    )


def _build_eris(mc, mo, cderi=None):
    """Select full or factorized integrals from the SCF object."""
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
    use_diis=False,
    diis_space=15,
    diis_start_cycle=3,
    diis_start_gradient=0.02,
    acceleration=None,
    callback=None,
):
    """Drive state-specific or state-averaged 2C-CASSCF with Super-CIPT.

    ``acceleration`` is an internal research hook used by the convergence
    diagnostics under ``tests/supercipt_debug``.  It is deliberately absent
    from the stable :meth:`CASSCF.supercipt` interface.
    """
    kramers = _resolve_kramers_mode(mc)
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
    with_df = getattr(mc._scf, "with_df", None)
    if acceleration not in (None, "spectral-cg", "pt-trust", "lbfgs"):
        raise ValueError(
            "Super-CIPT acceleration must be None, 'spectral-cg', "
            "'pt-trust', or 'lbfgs'"
        )
    if use_diis and acceleration is not None:
        raise ValueError(
            "Super-CIPT DIIS and PT acceleration are mutually exclusive"
        )
    orbital_diis = None
    if use_diis:
        orbital_diis = AndersonOrbitalDIIS(
            mo,
            mc._scf.get_ovlp(),
            space=diis_space,
            start_cycle=diis_start_cycle,
            start_gradient=diis_start_gradient,
        )
    orbital_cg = None
    if acceleration == "spectral-cg":
        from socutils.mcscf.orbital_cg import SpectralOrbitalCG

        orbital_cg = SpectralOrbitalCG(mo, mc._scf.get_ovlp())
    orbital_lbfgs = None
    if acceleration == "lbfgs":
        from socutils.mcscf.orbital_lbfgs import PTSeededOrbitalLBFGS

        orbital_lbfgs = PTSeededOrbitalLBFGS(
            mo, mc._scf.get_ovlp()
        )
    orbital_line_search = None
    line_search_mode = (
        acceleration if acceleration in ("pt-trust", "lbfgs") else None
    )
    line_search_history_key = (
        None
        if line_search_mode is None
        else line_search_mode.replace("-", "_")
    )
    if line_search_mode is not None:
        from socutils.mcscf.orbital_linesearch import (
            FixedReferenceStrongWolfeLineSearch,
        )

        orbital_line_search = FixedReferenceStrongWolfeLineSearch(
            mo,
            mc._scf.get_ovlp(),
            c2=0.9 if orbital_lbfgs is not None else 0.5,
            energy_tolerance=max(float(conv_tol), 1e-10),
            boundary_acceptance_ratio=0.1,
        )
    previous_energy = None
    history = []
    mc.macro_history = history
    mc.supercipt_history = history
    converged = False
    last_step = None
    last_integral_info = None
    pending_diis_fallback = None
    rejected_diis_steps = 0
    pending_cg_trial = None
    rejected_cg_steps = 0
    last_cg_diagnostics = None
    pending_pt_trial = None
    pt_trust_radius = None
    pt_trial_evaluations = 0
    pt_nonfinal_trials = 0
    last_pt_trust_diagnostics = None
    last_lbfgs_diagnostics = None
    lbfgs_secant_updates = 0
    lbfgs_rejected_directions = 0
    last_applied_step_norm = None
    last_applied_step_scale = None
    terminal_fallback = False
    previous_accepted_casdm1 = None
    e_tot = e_cas = ci = casdm1 = casdm2 = quantities = None

    log.info("Super-CIPT optimizer (Guo-Dutta 2026), max cycles = %d", max_cycle)
    if acceleration is not None:
        log.warn(
            "Super-CIPT acceleration '%s' is an internal experiment without "
            "production convergence validation",
            acceleration,
        )
    log.info(
        "Super-CIPT metric/denominator tolerance = %.3g / %.3g, level shift = %.3g",
        metric_tol,
        denominator_tol,
        level_shift,
    )
    log.info(
        "Super-CIPT core/virtual PT semicanonicalization = temporary",
    )
    log.info(
        "Super-CIPT Kramers = %s, DIIS = %s, acceleration = %s, "
        "integral route = %s (automatic)",
        kramers,
        bool(use_diis),
        "plain-PT" if acceleration is None else acceleration,
        "factorized" if with_df is not None or cderi is not None else "full",
    )

    # DIIS may use one extra evaluation only to restore its accepted plain-PT
    # fallback.  The transactional accelerators instead treat ``max_cycle``
    # as a hard CI/DMRG energy-evaluation budget and reserve restoration points
    # before issuing a guarded trial.
    hard_evaluation_budget = (
        orbital_cg is not None or orbital_line_search is not None
    )
    evaluation_limit = max_cycle if hard_evaluation_budget else max_cycle + 1
    for macro in range(evaluation_limit):
        if (
            not hard_evaluation_budget
            and macro == max_cycle
            and not terminal_fallback
        ):
            break
        eris, integral_info = _build_eris(
            mc,
            mo,
            cderi=cderi,
        )
        last_integral_info = integral_info
        from socutils.mcscf.zmcscf import _fake_h_for_fast_casci

        mci = _fake_h_for_fast_casci(mc, mo, eris)
        # Reuse the previous determinant-space vectors as Davidson guesses.
        # DMRGCI deliberately ignores external ``ci0`` and relies on its
        # fingerprinted MPS restart path, so the same call is safe for both
        # exact-CI debugging and production Block2 calculations.
        e_tot_raw, e_cas_raw, ci = mci.kernel(
            mo, ci0=ci, verbose=verbose
        )
        e_tot = _scalar_energy(e_tot_raw, "total energy")
        e_cas = _scalar_energy(e_cas_raw, "CAS energy")
        ci_converged = bool(np.all(getattr(mc.fcisolver, "converged", True)))
        if not ci_converged:
            raise RuntimeError("The active-space CI solver did not converge")
        casdm1, casdm2 = mc.fcisolver.make_rdm12(ci, mc.ncas, mc.nelecas)
        density_change = (
            None
            if previous_accepted_casdm1 is None
            else float(np.linalg.norm(casdm1 - previous_accepted_casdm1))
        )
        kramers_mapping = (
            _identify_kramers_mapping(mc, mo) if kramers else None
        )
        quantities = build_orbital_quantities(
            mc,
            mo,
            casdm1,
            casdm2,
            eris,
            kramers=kramers,
            kramers_mapping=kramers_mapping,
        )
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
            "orbital_gradient_frobenius_norm": (
                quantities.gradient_frobenius_norm
            ),
            "raw_orbital_gradient_norm": quantities.raw_gradient_norm,
            "raw_orbital_gradient_frobenius_norm": (
                quantities.raw_gradient_frobenius_norm
            ),
            "kramers_gradient": quantities.kramers_gradient_diagnostics,
            "norm_ddm": density_change,
            "natural_occupations": occupations.tolist(),
            "rdm_energy_error": rdm_energy_error,
            "ci_solver_converged": ci_converged,
            "ci_solver_diagnostics": _ci_convergence_snapshot(mc.fcisolver),
            "integrals": dict(integral_info),
            "accepted": True,
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

        energy_increase_tolerance = max(float(conv_tol), 1e-10)
        if (
            pending_diis_fallback is not None
            and energy_change is not None
            and energy_change > energy_increase_tolerance
        ):
            entry["accepted"] = False
            entry["diis_rejection"] = {
                "reason": "energy-increase",
                "source_macro_iteration": pending_diis_fallback[
                    "source_macro_iteration"
                ],
                "energy_increase": energy_change,
                "tolerance": energy_increase_tolerance,
            }
            rejected_diis_steps += 1
            log.warn(
                "Rejecting extrapolated Super-CIPT DIIS step from macro %d: "
                "energy increased by %.6g Eh; retrying its plain PT step",
                pending_diis_fallback["source_macro_iteration"],
                energy_change,
            )
            if callback is not None:
                callback(dict(entry))
            mo = pending_diis_fallback["plain_pt_mo"]
            pending_diis_fallback = None
            orbital_diis.reset()
            last_step = None
            last_applied_step_norm = None
            last_applied_step_scale = None
            terminal_fallback = macro + 1 == max_cycle
            continue
        pending_diis_fallback = None
        terminal_fallback = False

        if pending_cg_trial is not None:
            trial_change = float(e_tot - pending_cg_trial["base_energy"])
            guarded = bool(pending_cg_trial["guarded"])
            if guarded and trial_change > energy_increase_tolerance:
                entry["accepted"] = False
                rejection = orbital_cg.reject()
                entry["spectral_cg_rejection"] = {
                    "reason": "energy-increase",
                    "source_macro_iteration": pending_cg_trial[
                        "source_macro_iteration"
                    ],
                    "energy_increase": trial_change,
                    "tolerance": energy_increase_tolerance,
                    **rejection,
                }
                rejected_cg_steps += 1
                log.warn(
                    "Rejecting spectral-CG Super-CIPT step from macro %d: "
                    "energy increased by %.6g Eh; evaluating its plain PT "
                    "fallback",
                    pending_cg_trial["source_macro_iteration"],
                    trial_change,
                )
                if callback is not None:
                    callback(dict(entry))
                mo = pending_cg_trial["plain_pt_mo"]
                pending_cg_trial = None
                last_step = None
                last_cg_diagnostics = None
                last_applied_step_norm = None
                last_applied_step_scale = None
                continue

            predicted_linear = pending_cg_trial["predicted_linear"]
            linear_ratio = (
                trial_change / predicted_linear
                if predicted_linear < 0.0
                else None
            )
            acceptance = orbital_cg.accept(
                energy=e_tot,
                linear_ratio=linear_ratio,
            )
            entry["spectral_cg_acceptance"] = {
                "source_macro_iteration": pending_cg_trial[
                    "source_macro_iteration"
                ],
                "energy_change": trial_change,
                "predicted_linear": predicted_linear,
                **acceptance,
            }
            pending_cg_trial = None

        if pending_pt_trial is not None:
            current_trial = pending_pt_trial["trial"]
            base_energy = pending_pt_trial["base_energy"]
            trial_change = float(e_tot - base_energy)
            predicted_linear = float(
                current_trial.alpha * orbital_line_search.dphi0
            )
            linear_ratio = (
                trial_change / predicted_linear
                if predicted_linear < 0.0
                else None
            )
            remaining_evaluations = max_cycle - (macro + 1)
            decision = orbital_line_search.evaluate(
                mo,
                e_tot,
                quantities.screened_gradient,
                evaluations_remaining=remaining_evaluations,
            )
            pt_trial_evaluations += 1
            if decision.reason.startswith("zoom-"):
                pending_pt_trial["line_search_had_zoom"] = True
            boundary_trust_update = None
            if (
                current_trial.on_boundary
                and not decision.accepted
                and not pending_pt_trial["boundary_failed"]
            ):
                base_trust = float(
                    pending_pt_trial["base_trust_radius"]
                )
                deferred_trust = max(
                    np.finfo(float).eps,
                    0.5 * base_trust,
                )
                pending_pt_trial["boundary_failed"] = True
                boundary_trust_update = {
                    "reason": "rejected-boundary-trial",
                    "deferred": True,
                    "base_trust_radius": base_trust,
                    "trust_radius_before": base_trust,
                    "provisional_next_trust_radius": deferred_trust,
                    "trust_radius_after": deferred_trust,
                    "global_trust_radius_unchanged": float(
                        pt_trust_radius
                    ),
                }
            trial_diagnostics = {
                "source_macro_iteration": pending_pt_trial[
                    "source_macro_iteration"
                ],
                "alpha": float(current_trial.alpha),
                "step_norm": float(current_trial.step_norm),
                "on_boundary": bool(current_trial.on_boundary),
                "purpose": current_trial.purpose,
                "energy_change": trial_change,
                "predicted_full_linear_change": predicted_linear,
                "actual_to_full_linear_ratio": linear_ratio,
                "action": decision.action,
                "reason": decision.reason,
                "remaining_evaluations": int(remaining_evaluations),
                "decision": dict(decision.diagnostics),
                "line_search_had_zoom": bool(
                    pending_pt_trial["line_search_had_zoom"]
                ),
                "boundary_failed": bool(
                    pending_pt_trial["boundary_failed"]
                ),
                "lbfgs_history_size": (
                    int(orbital_lbfgs.pair_count)
                    if orbital_lbfgs is not None
                    else None
                ),
                "lbfgs_pending": (
                    bool(orbital_lbfgs.has_pending)
                    if orbital_lbfgs is not None
                    else None
                ),
            }
            if boundary_trust_update is not None:
                trial_diagnostics["boundary_trust_update"] = (
                    boundary_trust_update
                )
            if decision.trial is not None:
                trial_diagnostics["next_alpha"] = float(
                    decision.trial.alpha
                )
                trial_diagnostics["next_step_norm"] = float(
                    decision.trial.step_norm
                )
                trial_diagnostics["next_purpose"] = decision.trial.purpose
            entry[f"{line_search_history_key}_trial"] = trial_diagnostics
            log.info(
                "Super-CIPT %s trial %d  purpose = %s  alpha = %.6g  "
                "|step| = %.6g  rho = %s  action = %s (%s)  "
                "trust = %.6g",
                line_search_mode,
                macro,
                current_trial.purpose,
                current_trial.alpha,
                current_trial.step_norm,
                "---" if linear_ratio is None else "%.6g" % linear_ratio,
                decision.action,
                decision.reason,
                pt_trust_radius,
            )

            if not decision.accepted:
                entry["accepted"] = False
                pt_nonfinal_trials += 1
                if callback is not None:
                    callback(dict(entry))
                if decision.trial is None:
                    # The state machine normally prevents this by reserving a
                    # base/best replay before every continuation.  If a
                    # nondeterministic CI evaluation invalidates that replay,
                    # returning the rejected orbitals with stale accepted
                    # CI/RDM data would be worse than failing explicitly.
                    if orbital_lbfgs is not None:
                        last_lbfgs_diagnostics = {
                            **pending_pt_trial["lbfgs_proposal"],
                            "pair_action": orbital_lbfgs.reject(),
                            "resolution": "budget-exhausted-reject",
                        }
                        lbfgs_rejected_directions += 1
                    raise RuntimeError(
                        f"{line_search_mode} line search exhausted its CI "
                        "evaluation "
                        "budget before it could restore an accepted point"
                    )
                pending_pt_trial["trial"] = decision.trial
                mo = np.array(decision.trial.mo_coeff, copy=True)
                last_applied_step_norm = None
                last_applied_step_scale = None
                continue

            accepted_point = orbital_line_search.accepted
            if accepted_point is None:
                raise RuntimeError(
                    f"{line_search_mode} line search accepted without an "
                    "endpoint"
                )
            accepted_diagnostics = dict(accepted_point.diagnostics)
            accepted_ratio = accepted_diagnostics.get("linear_ratio")
            lbfgs_pair_action = None
            if orbital_lbfgs is not None:
                if (
                    accepted_point.alpha > 0.0
                    and accepted_diagnostics["step_norm"]
                    > np.finfo(float).tiny
                ):
                    lbfgs_pair_action = orbital_lbfgs.accept_secant(
                        **orbital_line_search.accepted_secant
                    )
                    if lbfgs_pair_action["accepted"]:
                        lbfgs_secant_updates += 1
                    resolution = "accepted-secants-resolved"
                else:
                    lbfgs_pair_action = orbital_lbfgs.reject()
                    lbfgs_rejected_directions += 1
                    resolution = "zero-step-direction-rejected"
                last_lbfgs_diagnostics = {
                    **pending_pt_trial["lbfgs_proposal"],
                    "pair_action": lbfgs_pair_action,
                    "resolution": resolution,
                    "history_size": orbital_lbfgs.pair_count,
                }
            trust_resolution = _resolve_line_search_trust(
                pending_pt_trial["base_trust_radius"],
                accepted_diagnostics["step_norm"],
                accepted_on_boundary=accepted_diagnostics[
                    "on_boundary"
                ],
                linear_ratio=accepted_ratio,
                boundary_failed=pending_pt_trial["boundary_failed"],
                max_stepsize=max_stepsize,
            )
            trust_before = trust_resolution["base_trust_radius"]
            pt_trust_radius = trust_resolution[
                "trust_radius_after"
            ]
            trust_action = trust_resolution["trust_action"]
            last_pt_trust_diagnostics = {
                "source_macro_iteration": pending_pt_trial[
                    "source_macro_iteration"
                ],
                "accepted_macro_iteration": int(macro),
                "accepted_alpha": float(accepted_point.alpha),
                "accepted_step_norm": float(
                    accepted_diagnostics["step_norm"]
                ),
                "energy_change": float(
                    accepted_point.energy - base_energy
                ),
                "actual_to_full_linear_ratio": accepted_ratio,
                "trust_radius_before": trust_before,
                "trust_radius_after": float(pt_trust_radius),
                "trust_action": trust_action,
                "trust_resolution": trust_resolution,
                "line_search_had_zoom": bool(
                    pending_pt_trial["line_search_had_zoom"]
                ),
                "boundary_failed_during_search": bool(
                    pending_pt_trial["boundary_failed"]
                ),
                "mode": line_search_mode,
                "lbfgs_pair_action": lbfgs_pair_action,
                "line_search": accepted_diagnostics,
            }
            entry[f"{line_search_history_key}_acceptance"] = dict(
                last_pt_trust_diagnostics
            )
            log.info(
                "Super-CIPT %s accept %d  |step| = %.6g  "
                "rho = %s  trust = %.6g -> %.6g (%s)",
                line_search_mode,
                macro,
                accepted_diagnostics["step_norm"],
                (
                    "---"
                    if accepted_ratio is None
                    else "%.6g" % accepted_ratio
                ),
                trust_before,
                pt_trust_radius,
                trust_action,
            )
            if lbfgs_pair_action is not None:
                log.info(
                    "Super-CIPT L-BFGS pair %d  action = %s  "
                    "history = %d/%d  curvature = %s",
                    macro,
                    lbfgs_pair_action.get("action", "rejected"),
                    orbital_lbfgs.pair_count,
                    orbital_lbfgs.memory,
                    (
                        "---"
                        if lbfgs_pair_action.get("stored_curvature") is None
                        else "%.3e"
                        % lbfgs_pair_action["stored_curvature"]
                    ),
                )
            pending_pt_trial = None

        previous_accepted_casdm1 = np.array(casdm1, copy=True)
        previous_energy = e_tot

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
        if macro + 1 >= max_cycle:
            if callback is not None:
                callback(dict(entry))
            break

        if orbital_line_search is not None:
            remaining_evaluations = max_cycle - (macro + 1)
            # The first trial needs one evaluation and a rejected trial needs
            # one further evaluation to replay the accepted base.  Stop at
            # the current accepted state rather than start a transaction that
            # cannot be rolled back inside the hard budget.
            if remaining_evaluations < 2:
                last_pt_trust_diagnostics = {
                    "enabled": True,
                    "mode": line_search_mode,
                    "budget_stop": True,
                    "remaining_evaluations": int(remaining_evaluations),
                    "trust_radius": (
                        None
                        if pt_trust_radius is None
                        else float(pt_trust_radius)
                    ),
                }
                entry[line_search_history_key] = dict(
                    last_pt_trust_diagnostics
                )
                log.info(
                    "Super-CIPT %s stops at accepted macro %d: "
                    "%d evaluation remains, but a guarded trial and "
                    "rollback require two",
                    line_search_mode,
                    macro,
                    remaining_evaluations,
                )
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
            kramers=kramers,
            kramers_mapping=kramers_mapping,
        )

        plain_pt_mo = np.array(last_step.mo_coeff, copy=True)
        applied_mo = last_step.mo_coeff
        applied_kappa = last_step.kappa
        applied_scale = last_step.scale

        def project_generator(current_mo, generator):
            screened = mc.unpack_uniq_var(mc.pack_uniq_var(generator))
            if kramers:
                return _project_kramers_rotation(
                    mc,
                    current_mo,
                    screened,
                    force=True,
                    mapping=kramers_mapping,
                )
            return screened, None

        if use_diis:
            diis_result = orbital_diis.update(
                mo,
                last_step.mo_coeff,
                quantities.screened_gradient,
                cycle=macro,
                gradient_norm=quantities.gradient_norm,
                max_stepsize=max_stepsize,
                step_metric="frobenius",
                projector=project_generator,
            )
            applied_mo = diis_result.mo_coeff
            applied_kappa = diis_result.generator
            applied_scale = diis_result.diagnostics["step_scale"]
            entry["diis"] = diis_result.diagnostics
            pulay = diis_result.diagnostics["pulay"]
            log.info(
                "Super-CIPT Anderson  %d  vectors = %d  extrapolated = %s  "
                "reject = %s  cond = %s  |c|1 = %s  g.kappa = %.3e",
                macro,
                diis_result.diagnostics["vectors"],
                diis_result.diagnostics["extrapolated"],
                diis_result.diagnostics["extrapolation_rejection"],
                (
                    "---"
                    if pulay["gram_condition"] is None
                    else "%.3e" % pulay["gram_condition"]
                ),
                (
                    "---"
                    if pulay["coefficient_l1_norm"] is None
                    else "%.3e" % pulay["coefficient_l1_norm"]
                ),
                diis_result.diagnostics["proposed_directional_derivative"],
            )
            if diis_result.diagnostics["extrapolated"]:
                pending_diis_fallback = {
                    "source_macro_iteration": int(macro),
                    "plain_pt_mo": plain_pt_mo,
                }

        if orbital_cg is not None:
            remaining_evaluations = max_cycle - (macro + 1)
            # A guarded trial can consume one evaluation and its mandatory
            # plain-PT restoration another.  Do not issue such a trial unless
            # both evaluations remain inside the hard max_cycle budget.
            if remaining_evaluations >= 2:
                cg_result = orbital_cg.propose(
                    mo,
                    quantities.screened_gradient,
                    last_step.kappa_unscaled,
                    energy=e_tot,
                    projector=project_generator,
                    max_stepsize=max_stepsize,
                )
                applied_mo = cg_result.mo_coeff
                applied_kappa = cg_result.generator
                raw_norm = last_step.unscaled_step_norm
                applied_scale = (
                    float(np.linalg.norm(applied_kappa)) / raw_norm
                    if raw_norm
                    else 1.0
                )
                last_cg_diagnostics = cg_result.diagnostics
                entry["spectral_cg"] = cg_result.diagnostics
                pending_cg_trial = {
                    "source_macro_iteration": int(macro),
                    "base_energy": float(e_tot),
                    "predicted_linear": float(
                        cg_result.diagnostics["slope"]
                    ),
                    "plain_pt_mo": plain_pt_mo,
                    "guarded": not bool(
                        cg_result.diagnostics["plain_equivalent"]
                    ),
                }
                log.info(
                    "Super-CIPT spectral-CG %d  gamma = %.6g  beta = %.6g  "
                    "trust = %.6g  |raw| = %.6g  |step| = %.6g  "
                    "slope = %.3e  restart = %s%s",
                    macro,
                    cg_result.diagnostics["gamma"],
                    cg_result.diagnostics["beta"],
                    cg_result.diagnostics["trust_radius"],
                    cg_result.diagnostics["raw_norm"],
                    cg_result.diagnostics["applied_norm"],
                    cg_result.diagnostics["slope"],
                    cg_result.diagnostics["restart"],
                    (
                        " (" + ",".join(
                            cg_result.diagnostics["restart_reasons"]
                        ) + ")"
                        if cg_result.diagnostics["restart_reasons"]
                        else ""
                    ),
                )
            else:
                last_cg_diagnostics = {
                    "enabled": True,
                    "method": "spectral-flexible-hybrid-PRP+",
                    "budget_plain_pt": True,
                    "remaining_evaluations": int(remaining_evaluations),
                }
                entry["spectral_cg"] = last_cg_diagnostics

        if orbital_line_search is not None:
            raw_pt_direction = np.asarray(last_step.kappa_unscaled)
            raw_pt_norm = float(np.linalg.norm(raw_pt_direction))
            search_direction = raw_pt_direction
            lbfgs_proposal_diagnostics = None
            if orbital_lbfgs is not None and raw_pt_norm > 0.0:
                lbfgs_proposal = orbital_lbfgs.propose(
                    mo,
                    quantities.screened_gradient,
                    raw_pt_direction,
                    projector=project_generator,
                )
                search_direction = lbfgs_proposal.generator
                lbfgs_proposal_diagnostics = dict(
                    lbfgs_proposal.diagnostics
                )
                if lbfgs_proposal_diagnostics["history_size_before"] == 0:
                    lbfgs_proposal_diagnostics[
                        "first_plain_pt_equivalence_error"
                    ] = float(
                        np.linalg.norm(
                            search_direction - raw_pt_direction
                        )
                    )
                last_lbfgs_diagnostics = dict(
                    lbfgs_proposal_diagnostics
                )
            direction_norm = float(np.linalg.norm(search_direction))
            full_linear_slope = float(
                np.vdot(
                    quantities.screened_gradient, search_direction
                ).real
            )
            if direction_norm == 0.0 or full_linear_slope >= 0.0:
                lbfgs_pair_action = None
                if orbital_lbfgs is not None and orbital_lbfgs.has_pending:
                    lbfgs_pair_action = orbital_lbfgs.reject()
                    lbfgs_rejected_directions += 1
                    last_lbfgs_diagnostics = {
                        **lbfgs_proposal_diagnostics,
                        "pair_action": lbfgs_pair_action,
                        "resolution": "abandoned-non-descent-direction",
                    }
                last_pt_trust_diagnostics = {
                    "enabled": True,
                    "mode": line_search_mode,
                    "proposal_rejected": True,
                    "reason": (
                        "zero-search-direction"
                        if direction_norm == 0.0
                        else "non-descent-search-direction"
                    ),
                    "raw_pt_step_norm": raw_pt_norm,
                    "direction_norm": direction_norm,
                    "full_linear_slope": full_linear_slope,
                    "lbfgs": lbfgs_proposal_diagnostics,
                    "lbfgs_pair_action": lbfgs_pair_action,
                    "trust_radius": (
                        None
                        if pt_trust_radius is None
                        else float(pt_trust_radius)
                    ),
                }
                entry[line_search_history_key] = dict(
                    last_pt_trust_diagnostics
                )
                log.warn(
                    "Super-CIPT %s stops at macro %d: %s "
                    "(|raw PT| = %.6g, |direction| = %.6g, "
                    "g.kappa = %.3e)",
                    line_search_mode,
                    macro,
                    last_pt_trust_diagnostics["reason"],
                    raw_pt_norm,
                    direction_norm,
                    full_linear_slope,
                )
                if callback is not None:
                    callback(dict(entry))
                break

            if pt_trust_radius is None:
                # The first raw PT displacement defines the physical trust
                # scale.  ``max_stepsize`` remains the global hard cap.
                pt_trust_radius = min(raw_pt_norm, float(max_stepsize))
            if orbital_lbfgs is not None:
                # The quasi-Newton generator already carries its inverse-
                # Hessian scale.  Trust is therefore only an upper bound on
                # the natural alpha=1 proposal; the line search may expand a
                # successful interior trial later.
                initial_alpha = min(
                    1.0, pt_trust_radius / direction_norm
                )
                initial_alpha_policy = "natural-step-trust-clipped"
            else:
                # Plain PT supplies a direction but no quasi-Newton step
                # length, so its first trial deliberately reaches trust.
                initial_alpha = pt_trust_radius / direction_norm
                initial_alpha_policy = "trust-boundary"
            pt_trial = orbital_line_search.begin(
                mo,
                search_direction,
                e_tot,
                quantities.screened_gradient,
                alpha=initial_alpha,
                trust_radius=pt_trust_radius,
                max_stepsize=max_stepsize,
            )
            applied_mo = pt_trial.mo_coeff
            applied_kappa = pt_trial.alpha * search_direction
            applied_scale = pt_trial.alpha
            pending_pt_trial = {
                "source_macro_iteration": int(macro),
                "base_energy": float(e_tot),
                "base_trust_radius": float(pt_trust_radius),
                "trial": pt_trial,
                "line_search_had_zoom": False,
                "boundary_failed": False,
                "lbfgs_proposal": lbfgs_proposal_diagnostics,
            }
            last_pt_trust_diagnostics = {
                "enabled": True,
                "mode": line_search_mode,
                "phase": "proposal",
                "source_macro_iteration": int(macro),
                "alpha": float(pt_trial.alpha),
                "initial_alpha_policy": initial_alpha_policy,
                "step_norm": float(pt_trial.step_norm),
                "raw_pt_step_norm": raw_pt_norm,
                "direction_norm": direction_norm,
                "trust_radius": float(pt_trust_radius),
                "on_boundary": bool(pt_trial.on_boundary),
                "full_linear_slope": full_linear_slope,
                "predicted_full_linear_change": float(
                    pt_trial.alpha * full_linear_slope
                ),
                "lbfgs": lbfgs_proposal_diagnostics,
            }
            entry[line_search_history_key] = dict(
                last_pt_trust_diagnostics
            )
            log.info(
                "Super-CIPT %s proposal %d  alpha = %.6g  "
                "|raw PT| = %.6g  |direction| = %.6g  "
                "|step| = %.6g  trust = %.6g  g.kappa = %.3e",
                line_search_mode,
                macro,
                pt_trial.alpha,
                raw_pt_norm,
                direction_norm,
                pt_trial.step_norm,
                pt_trust_radius,
                full_linear_slope,
            )
            if lbfgs_proposal_diagnostics is not None:
                log.info(
                    "Super-CIPT L-BFGS direction %d  history = %d/%d  "
                    "used = %s  fallback = %s  |direction| = %.6g  "
                    "slope = %.3e",
                    macro,
                    lbfgs_proposal_diagnostics["history_size_before"],
                    orbital_lbfgs.memory,
                    lbfgs_proposal_diagnostics["history_used"],
                    lbfgs_proposal_diagnostics["fallback"],
                    direction_norm,
                    full_linear_slope,
                )

        applied_direction = _direction_diagnostics(
            mc, quantities.screened_gradient, applied_kappa
        )
        last_applied_step_norm = float(np.linalg.norm(applied_kappa))
        last_applied_step_scale = float(applied_scale)
        entry.update(
            {
                "proposed_maximum_amplitude": last_step.maximum_amplitude,
                "proposed_orbital_step_norm": last_step.unscaled_step_norm,
                "step_scale": float(applied_scale),
                "applied_orbital_step_norm": last_applied_step_norm,
                "minimum_denominator": last_step.minimum_denominator,
                "metric": last_step.metric_diagnostics,
                "removal_energies": last_step.removal_energies.tolist(),
                "addition_energies": last_step.addition_energies.tolist(),
                "canonical_energies": last_step.canonical_energies,
                "semicanonical": last_step.semicanonical_diagnostics,
                "plain_pt_direction": last_step.direction_diagnostics,
                "applied_direction": applied_direction,
                "kramers_rotation": last_step.kramers_diagnostics,
            }
        )
        log.info(
            "Super-CIPT step  %d  |kappa(raw)| = %.6g  max(raw) = %.6g  "
            "scale = %.6g  |kappa| = %.6g  min|denom| = %.6g  "
            "g.kappa = %.3e  cos = %.3f  metric ranks = %d/%d",
            macro,
            last_step.unscaled_step_norm,
            last_step.maximum_amplitude,
            applied_scale,
            last_applied_step_norm,
            last_step.minimum_denominator,
            applied_direction["total"][
                "directional_derivative"
            ],
            (
                applied_direction["total"]["cosine"]
                if applied_direction["total"]["cosine"]
                is not None
                else 0.0
            ),
            last_step.metric_diagnostics["removal"]["rank"],
            last_step.metric_diagnostics["addition"]["rank"],
        )
        if callback is not None:
            callback(dict(entry))
        mo = applied_mo

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
        "final_gradient_frobenius_norm": float(
            quantities.gradient_frobenius_norm
        ),
        "final_raw_gradient_norm": float(quantities.raw_gradient_norm),
        "final_raw_gradient_frobenius_norm": float(
            quantities.raw_gradient_frobenius_norm
        ),
        "last_kramers_gradient": quantities.kramers_gradient_diagnostics,
        "energy_tolerance": float(conv_tol),
        "gradient_tolerance": float(conv_tol_grad),
        "metric_tolerance": float(metric_tol),
        "denominator_tolerance": float(denominator_tol),
        "level_shift": float(level_shift),
        "maximum_step": float(max_stepsize),
        "pt_core_virtual_canonicalization": True,
        "pt_semicanonical_frame_is_temporary": True,
        "step_metric": "frobenius",
        "integrals": dict(last_integral_info),
        "kramers_restricted": bool(kramers),
        "diis": bool(use_diis),
        "diis_space": int(diis_space) if use_diis else None,
        "diis_energy_safeguard": bool(use_diis),
        "diis_energy_increase_tolerance": max(float(conv_tol), 1e-10),
        "diis_rejected_steps": int(rejected_diis_steps),
        "acceleration": acceleration,
        "spectral_cg": orbital_cg is not None,
        "spectral_cg_energy_safeguard": orbital_cg is not None,
        "spectral_cg_rejected_steps": int(rejected_cg_steps),
        "line_search": orbital_line_search is not None,
        "line_search_mode": line_search_mode,
        "line_search_c2": (
            float(orbital_line_search.c2)
            if orbital_line_search is not None
            else None
        ),
        "line_search_energy_safeguard": orbital_line_search is not None,
        "line_search_trust_radius": (
            None if pt_trust_radius is None else float(pt_trust_radius)
        ),
        "line_search_trial_evaluations": int(pt_trial_evaluations),
        "line_search_nonfinal_trials": int(pt_nonfinal_trials),
        "last_line_search": last_pt_trust_diagnostics,
        "pt_trust": line_search_mode == "pt-trust",
        "pt_trust_energy_safeguard": line_search_mode == "pt-trust",
        "pt_trust_radius": (
            float(pt_trust_radius)
            if line_search_mode == "pt-trust"
            and pt_trust_radius is not None
            else None
        ),
        "pt_trust_trial_evaluations": (
            int(pt_trial_evaluations)
            if line_search_mode == "pt-trust"
            else 0
        ),
        "pt_trust_nonfinal_trials": (
            int(pt_nonfinal_trials)
            if line_search_mode == "pt-trust"
            else 0
        ),
        "last_pt_trust": (
            last_pt_trust_diagnostics
            if line_search_mode == "pt-trust"
            else None
        ),
        "lbfgs": orbital_lbfgs is not None,
        "lbfgs_memory": (
            int(orbital_lbfgs.memory)
            if orbital_lbfgs is not None
            else None
        ),
        "lbfgs_history_size": (
            int(orbital_lbfgs.pair_count)
            if orbital_lbfgs is not None
            else 0
        ),
        "lbfgs_secant_updates": int(lbfgs_secant_updates),
        "lbfgs_rejected_directions": int(lbfgs_rejected_directions),
        "lbfgs_trust_radius": (
            float(pt_trust_radius)
            if orbital_lbfgs is not None and pt_trust_radius is not None
            else None
        ),
        "lbfgs_trial_evaluations": (
            int(pt_trial_evaluations)
            if orbital_lbfgs is not None
            else 0
        ),
        "lbfgs_nonfinal_trials": (
            int(pt_nonfinal_trials)
            if orbital_lbfgs is not None
            else 0
        ),
        "last_lbfgs": (
            last_lbfgs_diagnostics
            if orbital_lbfgs is not None
            else None
        ),
        "energy_evaluations": len(history),
        "accepted_evaluations": int(
            sum(bool(row.get("accepted", True)) for row in history)
        ),
        "hard_evaluation_budget": (
            int(max_cycle) if hard_evaluation_budget else None
        ),
        "last_spectral_cg": last_cg_diagnostics,
        "last_applied_step_norm": last_applied_step_norm,
        "last_step_scale": last_applied_step_scale,
    }
    return converged, e_tot, e_cas, ci, mo, mc.mo_energy
