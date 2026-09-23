"""Orbital/CI helpers shared by peer Super-CI and second-order AH optimizers."""

import numpy as np
import scipy
from dataclasses import dataclass
from functools import reduce
from numpy.linalg import norm
from pyscf import lib

from socutils.mcscf import zmc_ao2mo


class OrbitalSolveNumericalError(RuntimeError):
    """An inner orbital solve that can recover at a smaller radius."""

def _physical_active_density(casdm1, tolerance=1e-7):
    """Return a Hermitian, roundoff-bounded spinor 1-RDM for the metric."""
    casdm1 = np.asarray(casdm1, dtype=complex)
    hermiticity_error = float(np.max(abs(casdm1 - casdm1.T.conj())))
    hermitian_dm1 = (casdm1 + casdm1.T.conj()) * 0.5
    occupations, orbitals = scipy.linalg.eigh(hermitian_dm1)
    lower_violation = max(0.0, -float(occupations[0]))
    upper_violation = max(0.0, float(occupations[-1]) - 1.0)
    representability_error = max(hermiticity_error, lower_violation, upper_violation)
    if representability_error > tolerance:
        raise RuntimeError(
            "MCSCF active 1-RDM violates spinor N-representability by "
            "%.6e (tolerance %.6e)" % (representability_error, tolerance)
        )
    bounded_occupations = np.clip(occupations, 0.0, 1.0)
    bounded_dm1 = (orbitals * bounded_occupations).dot(orbitals.T.conj())
    diagnostics = {
        "dm1_hermiticity_error": hermiticity_error,
        "minimum_natural_occupation": float(occupations[0]),
        "maximum_natural_occupation": float(occupations[-1]),
        "occupation_bound_correction": float(
            np.max(abs(bounded_occupations - occupations), initial=0.0)
        ),
    }
    return bounded_dm1, diagnostics

def _contract_dm2_gradient(eris, casdm2):
    """Return the ``p,t`` two-particle contribution to the orbital gradient.

    ``casdm2[t,u,v,w]`` follows the socutils/PySCF spinor convention
    ``<t† v† w u>``.  For full integrals this contracts
    ``(p u|v w) casdm2[t,u,v,w]``.  The Cholesky form uses the bilinear
    (unconjugated) reconstruction
    ``(p u|v w) = sum_P cd_pa[P,p,u] cd_aa[P,v,w]``.
    """
    if isinstance(eris, zmc_ao2mo._ERIS):
        return lib.einsum("puvw,tuvw->pt", eris.paaa, casdm2)
    if isinstance(eris, zmc_ao2mo._CDERIS):
        tmp = lib.einsum("Pvw,tuvw->Ptu", eris.cd_aa, casdm2)
        return lib.einsum("Ppu,Ptu->pt", eris.cd_pa, tmp)
    raise TypeError("Unsupported ERI container %s" % type(eris).__name__)

def _build_eris(mc, mo, cderi=None):
    """Build the MCSCF integral container selected by the SCF source.

    An attached ``with_df`` object (or the legacy explicit ``cderi`` argument)
    selects the existing factorized route.  Otherwise the direct four-index
    spinor transformation is used.  Keeping this decision in one helper is
    important because MCSCF rebuilds the transformed integrals after every
    accepted orbital rotation and, optionally, after active natural-orbital
    rotations.

    Returns
    -------
    eris
        A :class:`~socutils.mcscf.zmc_ao2mo._CDERIS` or
        :class:`~socutils.mcscf.zmc_ao2mo._ERIS` instance.
    diagnostics : dict
        Stable provenance fields shared by the macroiteration history and the
        final orbital-optimizer diagnostics.
    """
    with_df = getattr(mc._scf, "with_df", None)
    if with_df is not None or cderi is not None:
        eris = zmc_ao2mo._CDERIS(mc, mo, cderi=cderi, level=2)
        from socutils.cd.cd import CD

        is_cholesky = isinstance(with_df, CD)
        return eris, {
            "representation": "factorized",
            "factorized": True,
            "active": is_cholesky,
            "container": type(eris).__name__,
            "source": (
                type(with_df).__name__
                if with_df is not None
                else "legacy-cderi"
            ),
            "naux": int(eris.cd_pa.shape[0]),
            "threshold": getattr(with_df, "tau", None),
        }

    eris = zmc_ao2mo._ERIS(mc, mo, level=2)
    return eris, {
        "representation": "full",
        "factorized": False,
        "active": False,
        "container": type(eris).__name__,
        "source": "full-integral",
        "naux": None,
        "threshold": None,
    }

def _resolve_kramers_mode(casscf, symm=None):
    """Infer Kramers mode from the reference or active-space solver."""
    from socutils.scf import spinor_hf

    if symm is not None:
        symm = str(symm).lower()
        if symm != "kramers":
            raise ValueError("symm must be None or 'kramers'")
    native = isinstance(casscf._scf, spinor_hf.KRHF) or getattr(
        casscf.fcisolver, "kramers_adapter", None
    ) is not None
    return bool(native or symm == "kramers")

def _identify_kramers_mapping(casscf, mo):
    from socutils.dmrg.kramers import (
        identify_kramers_orbitals,
    )

    return identify_kramers_orbitals(
        casscf.mol,
        mo,
        casscf._scf.get_ovlp(),
        tolerance=1e-8,
    )

def _project_kramers_rotation(
    casscf,
    mo,
    generator,
    *,
    force=False,
    mapping=None,
):
    """Project a generator onto the allowed Kramers-invariant tangent space.

    The ordinary CASSCF mask can contain constraints beyond the
    core/active/virtual partition (``frozen``, ``freeze_pair`` and ``irrep``).
    Such a mask need not itself be closed under time reversal.  The admissible
    KR tangent is therefore the intersection of the full anti-Hermitian support
    with its time-reversed image.  This freezes the partner direction as well
    when only one member of a Kramers-related rotation was excluded.
    """
    from socutils.dmrg.kramers import time_reverse_one_body
    from socutils.scf import spinor_hf

    if not force and not isinstance(casscf._scf, spinor_hf.KRHF):
        return generator, None

    if mapping is None:
        mapping = _identify_kramers_mapping(casscf, mo)
    # Use the phase-resolved, exactly sparse representation rather than the
    # measured matrix's roundoff-level off-pair entries.  This makes the
    # symmetry projection idempotent and prevents a sequence of orbital steps
    # from accumulating Kramers-closure drift.
    time_reversal = np.zeros_like(mapping.time_reversal)
    ncore = casscf.ncore
    nocc = ncore + casscf.ncas

    def orbital_space(index):
        if index < ncore:
            return "core"
        if index < nocc:
            return "active"
        return "virtual"

    for (first, second), phase in zip(mapping.pairs, mapping.phases):
        if orbital_space(first) != orbital_space(second):
            raise RuntimeError(
                "a Kramers orbital pair crosses a core/active/virtual boundary"
            )
        phase /= abs(phase)
        time_reversal[second, first] = phase
        time_reversal[first, second] = -phase

    nonzero = abs(time_reversal) > 0.0
    if not (
        np.all(np.count_nonzero(nonzero, axis=0) == 1)
        and np.all(np.count_nonzero(nonzero, axis=1) == 1)
    ):
        raise RuntimeError(
            "the Kramers orbital mapping is not a signed permutation"
        )
    partners = np.argmax(nonzero, axis=1)

    # Build the complete support explicitly instead of calling pack_uniq_var:
    # the latter invokes screen_irrep(), which mutates its matrix argument.
    # ``uniq_var_indices`` is the authoritative lower-triangle mask and already
    # includes frozen, freeze_pair and irrep restrictions.
    nmo = generator.shape[0]
    lower_allowed = np.asarray(
        casscf.uniq_var_indices(nmo, ncore, casscf.ncas, casscf.frozen),
        dtype=bool,
    )
    if lower_allowed.shape != generator.shape:
        raise ValueError("orbital generator and allowed-variable mask disagree")
    allowed_support = lower_allowed | lower_allowed.T
    time_reversed_support = allowed_support[np.ix_(partners, partners)]
    intersection_support = allowed_support & time_reversed_support

    input_residual = float(
        np.max(abs(generator - time_reverse_one_body(time_reversal, generator)))
    )
    # The intersection support is symmetric and invariant under time reversal,
    # so support screening, anti-Hermitization and KR symmetrization commute.
    # A final explicit screen removes only roundoff and cannot break KR.
    screened = np.zeros_like(generator, dtype=np.complex128)
    screened[intersection_support] = generator[intersection_support]
    screened = (screened - screened.T.conj()) * 0.5
    projected = (
        screened + time_reverse_one_body(time_reversal, screened)
    ) * 0.5
    projected = (projected - projected.T.conj()) * 0.5
    projected[~intersection_support] = 0.0
    output_residual = float(
        np.max(abs(projected - time_reverse_one_body(time_reversal, projected)))
    )
    forbidden_residual = float(
        np.max(abs(projected[~intersection_support]), initial=0.0)
    )
    return projected, {
        "input_generator_residual": input_residual,
        "output_generator_residual": output_residual,
        "projection_change_norm": float(norm(projected - generator)),
        "allowed_support_size": int(np.count_nonzero(allowed_support)),
        "intersection_support_size": int(
            np.count_nonzero(intersection_support)
        ),
        "support_directions_removed_by_kramers": int(
            np.count_nonzero(allowed_support & ~intersection_support)
        ),
        "forbidden_support_residual": forbidden_residual,
        "orbital_closure_before_step": mapping.diagnostics["subspace_closure_error"],
        "orbital_partner_error_before_step": mapping.diagnostics[
            "partner_orbital_error"
        ],
        "pairs": mapping.pairs,
    }

def _ci_convergence_snapshot(solver):
    info = getattr(solver, "convergence_info", None) or {}
    keys = (
        "sweeps",
        "energy_change",
        "discarded_weight",
        "local_residual_bound",
        "bond_dimension",
        "npdm_site_type",
        "npdm_cutoff",
        "run_mode",
        "restart_transport",
        "restart_requested",
        "restart_fallback",
        "schedule_mode",
        "effective_twosite_to_onesite",
    )
    return {key: info[key] for key in keys if key in info}

def _schedule_orbital_trial(mc, gradient, step, *, accepted=True, ci_converged=True):
    """Gate the MPS immediately before the Hamiltonian it will initialize."""
    schedule = getattr(mc.fcisolver, 'restart_scheduler_step', None)
    if schedule is not None:
        # Disk resume/manual restart is a one-shot initial condition. Trials
        # use only the current validated MPS and the actual proposed step.
        mc.fcisolver.restart = False
        mc.fcisolver.resume = False
        schedule(dict(accepted=accepted, ci_solver_converged=ci_converged,
                      orbital_gradient_norm=float(gradient),
                      applied_orbital_step_norm=step))
        return dict(mc.fcisolver.restart_diagnostics)
    return None


def _bounded_orbital_update(mc, mo, base_energy, g, hd, hop, sop, solve,
                            radius, max_radius, tol, mmax, conv_tol,
                            conv_tol_grad, second_order, verbose, log, cderi=None):
    """Try bounded steps from one accepted point; rejected MPS are never reused.

    A cold replay of the base restores live CI/MPS, its RDMs and checkpoint
    on exhaustion/error without requiring an in-memory copy of a native Block2 driver. Numerical
    inner failures shrink the radius before touching any CI/MPS state.
    """
    from scipy.linalg import expm as expmat
    from socutils.mcscf import zmcscf

    gradient = float(np.linalg.norm(g))
    trials = []
    touched = False
    micro_tol = (getattr(mc, 'second_order_micro_step_tol', 1e-4)
                 if gradient > 10*conv_tol_grad else 0.)

    def evaluate(coeff):
        eri, provenance = _build_eris(mc, coeff, cderi=cderi)
        cas = zmcscf._fake_h_for_fast_casci(mc, coeff, eri)
        energy, ecas, ci = cas.kernel(coeff, ci0=None, verbose=verbose)
        if not np.all(getattr(mc.fcisolver, 'converged', True)) or not np.isfinite(energy):
            raise RuntimeError('Active-space solver failed at the trial orbitals')
        return energy, ecas, ci, eri, provenance

    try:
        for attempt in range(6):
            options = dict(micro_step_tol=micro_tol) if second_order else dict(step_radius=radius)
            try:
                # BAGEL accepts a step within 1% of its requested bound.
                # Reserve that margin so the outer trust radius stays strict.
                solve_radius = radius/1.01 if second_order else radius
                x, _, info = solve(hop, g, hd, sop=sop, max_stepsize=solve_radius,
                                   tol=tol, mmax=mmax, log=log, **options)
            except (np.linalg.LinAlgError, OrbitalSolveNumericalError) as error:
                if not second_order:
                    raise
                trials.append(dict(attempt=attempt, radius=radius, accepted=False,
                                   stage='orbital_solve', error=str(error)))
                log.info('AH solve retry: attempt=%d radius=%.5g error=%s; shrink radius',
                         attempt, radius, error)
                radius *= .5
                continue
            info = dict(info, solver='davidson')
            if not info['converged']:
                trials.append(dict(attempt=attempt, radius=radius, accepted=False,
                                   stage='orbital_solve', linear_solver=info))
                if (info.get('reason') not in ('maximum_space', 'linear_dependence')
                        or not np.isfinite(info.get('residual_norm', np.nan))):
                    raise RuntimeError('Orbital solve did not converge: %s' % info)
                log.info('Orbital solve retry: attempt=%d radius=%.5g reason=%s residual=%.3e; shrink radius',
                         attempt, radius, info['reason'], info['residual_norm'])
                radius *= .5
                continue
            dr = mc.unpack_uniq_var(x)
            size = float(np.linalg.norm(dr))
            if size > radius*(1+1e-10):
                trials.append(dict(attempt=attempt, radius=radius, accepted=False,
                                   stage='orbital_solve', error='AH step exceeded trust radius'))
                radius *= .5
                continue
            predicted = float(2*np.vdot(x, g).real)
            if second_order:
                predicted += info['quadratic_form']
            proposed = mo @ expmat(dr)
            restart = _schedule_orbital_trial(mc, gradient, size, accepted=not touched)
            record = dict(attempt=attempt, radius=radius, step_norm=size,
                          predicted_energy_change=predicted, accepted=False, stage='casci',
                          linear_solver=info, restart=restart)
            trials.append(record)
            touched = True
            try:
                energy, ecas, ci, eri, provenance = evaluate(proposed)
            except RuntimeError as error:
                record['error'] = str(error)
            else:
                change = float(energy-base_energy)
                ratio = change/predicted if predicted < -1e-16 else None
                # Admit only noise-scale rises at already-small gradients;
                # the unchanged outer energy and gradient tests still apply.
                slack = conv_tol if gradient < conv_tol_grad else 0.
                accepted = change <= slack and (abs(change) < conv_tol or
                                                (ratio is not None and ratio >= .1))
                record.update(accepted=bool(accepted), energy=float(energy),
                              energy_change=change, ratio=ratio,
                              ci_solver_diagnostics=_ci_convergence_snapshot(mc.fcisolver))
                if accepted:
                    next_radius = radius
                    if ratio is not None and ratio < .25:
                        next_radius *= .5
                    elif ratio is not None and ratio > .75 and size >= .9*radius and change < 0:
                        next_radius = min(max_radius, radius*1.5)
                    return (proposed, energy, ecas, ci, eri, provenance, dr, x,
                            info, predicted, change, ratio, next_radius, trials)
            log.info('Orbital trial rejected: attempt=%d radius=%.5g dE=%s; retry from accepted orbitals',
                     attempt, radius, record.get('energy_change', record.get('error')))
            radius *= .5
        raise RuntimeError('Six orbital trials failed (inner solve or energy acceptance)')
    except Exception:
        mc.converged = False
        mc.orbital_trial_history = trials
        if touched:
            _schedule_orbital_trial(mc, gradient, None, accepted=False)
            energy, ecas, ci, _, _ = evaluate(mo)
            mc.mo_coeff, mc.e_tot, mc.e_cas, mc.ci = mo.copy(), energy, ecas, ci
            mc.fcisolver.make_rdm12(ci, mc.ncas, mc.nelecas)
            log.info('Restored accepted orbitals and recomputed matching CI/RDM/checkpoint')
        raise


def _kramers_subspace_eigh(casscf, matrix, mo_subspace):
    """Diagonalize a Kramers-invariant MO subspace without pair-order assumptions.

    ``zquatev`` expects a canonical Kramers basis.  The actual partners and
    their phases are therefore identified in the AO metric first, and the
    returned eigenvectors are transformed back to the caller's MO ordering.
    """
    from socutils.dmrg.kramers import (
        identify_kramers_orbitals,
        time_reverse_one_body,
    )
    from socutils.lib import zquatev

    matrix = np.asarray(matrix, dtype=np.complex128)
    mo_subspace = np.asarray(mo_subspace, dtype=np.complex128)
    nmo = matrix.shape[0]
    if matrix.shape != (nmo, nmo) or mo_subspace.shape[1] != nmo:
        raise ValueError("Kramers subspace matrix and orbitals disagree")
    if nmo == 0:
        return np.empty(0), np.empty((0, 0), dtype=np.complex128)
    if nmo % 2:
        raise ValueError("a Kramers orbital subspace must have even dimension")

    mapping = identify_kramers_orbitals(
        casscf.mol,
        mo_subspace,
        casscf._scf.get_ovlp(),
        tolerance=1e-8,
    )
    pair_basis = np.zeros((nmo, nmo), dtype=np.complex128)
    for pair_index, ((first, second), phase) in enumerate(
        zip(mapping.pairs, mapping.phases)
    ):
        phase = phase / abs(phase)
        pair_basis[first, 2 * pair_index] = 1.0
        pair_basis[second, 2 * pair_index + 1] = phase

    paired_matrix = reduce(
        np.dot,
        (pair_basis.T.conj(), matrix, pair_basis),
    )
    canonical_time_reversal = np.zeros_like(paired_matrix)
    canonical_time_reversal[1::2, 0::2] = np.eye(nmo // 2)
    canonical_time_reversal[0::2, 1::2] = -np.eye(nmo // 2)
    paired_matrix = 0.5 * (
        paired_matrix
        + time_reverse_one_body(canonical_time_reversal, paired_matrix)
    )
    paired_matrix = 0.5 * (paired_matrix + paired_matrix.T.conj())

    block_order = np.r_[np.arange(0, nmo, 2), np.arange(1, nmo, 2)]
    block_matrix = paired_matrix[np.ix_(block_order, block_order)]
    eigenvalues, block_vectors = zquatev.eigh(block_matrix, iop=1)
    paired_vectors = np.zeros_like(block_vectors)
    paired_vectors[block_order] = block_vectors
    return eigenvalues, pair_basis.dot(paired_vectors)


@dataclass
class OrbitalQuantities:
    fock_core: np.ndarray
    fock_effective: np.ndarray
    two_rdm_contraction: np.ndarray
    lagrangian: np.ndarray
    gradient: np.ndarray
    screened_gradient: np.ndarray


def build_orbital_quantities(mc, mo, casdm1, casdm2, eris):
    """Build the fixed-RDM generalized Fock and orbital gradient."""
    mo = np.asarray(mo)
    casdm1 = np.asarray(casdm1)
    ncore, ncas = mc.ncore, mc.ncas
    nocc = ncore + ncas
    nmo = mo.shape[1]

    h1e = reduce(np.dot, (mo.T.conj(), mc.get_hcore(), mo))
    mo_core = mo[:, :ncore]
    dm_core_ao = mo_core @ mo_core.conj().T
    core_occ = np.zeros(nmo)
    core_occ[:ncore] = 1.0
    vj_core, vk_core = eris.get_jk(dm_core_ao, mo_coeff=mo, mo_occ=core_occ)
    vhf_core = reduce(np.dot, (mo.T.conj(), vj_core - vk_core, mo))
    # RDM[p,q] = <p^+ q>; JK needs the covariant density.
    vj_active, vk_active = eris.get_jk_active_mo(casdm1.T)
    fock_core = h1e + vhf_core
    fock_core = .5 * (fock_core + fock_core.T.conj())
    fock_effective = fock_core + vj_active - vk_active
    fock_effective = .5 * (fock_effective + fock_effective.T.conj())
    two_rdm = _contract_dm2_gradient(eris, casdm2)

    lagrangian = np.zeros((nmo, nmo), dtype=np.complex128)
    lagrangian[:, :ncore] = fock_effective[:, :ncore]
    lagrangian[:, ncore:nocc] = (
        fock_core[:, ncore:nocc] @ casdm1.T + two_rdm
    )
    gradient = lagrangian - lagrangian.T.conj()
    allowed = mc.uniq_var_indices(nmo, ncore, ncas, mc.frozen)
    lower = np.zeros_like(gradient)
    lower[allowed] = gradient[allowed]
    screened_gradient = lower - lower.T.conj()
    return OrbitalQuantities(
        fock_core, fock_effective, two_rdm, lagrangian, gradient,
        screened_gradient,
    )
