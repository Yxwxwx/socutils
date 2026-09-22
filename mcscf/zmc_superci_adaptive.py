"""Opt-in radius-bound Super-CI (full ERI, unrestricted spinor rotations).

The complex operator and metric conditioning are shared with ordinary
Super-CI. This policy additionally solves for an orbital shift when the
unshifted solution exceeds the radius. It is not an exact orbital Hessian.
"""
import numpy as np

from socutils.mcscf import zmc_superci as sci
from socutils.mcscf.zmc_superci import (
    _full_eri_operators as _operators,
    _gen_g_hop_full as gen_g_hop,
)


def validate(mc, mo, *, solver="davidson", cderi=None, kramers=False):
    """Reject unsupported routes before expensive integral/CI work."""
    if solver != "davidson":
        raise ValueError("Adaptive Super-CI requires solver='davidson'")
    if cderi is not None or getattr(mc._scf, "with_df", None) is not None:
        raise ValueError("Adaptive Super-CI requires full ERI (no CD/DF)")
    if kramers or sci._resolve_kramers_mode(mc):
        raise ValueError("Adaptive Super-CI does not support Kramers restriction")
    if mc.canonicalize_ or mc.natorb:
        raise ValueError("Adaptive Super-CI requires canonicalize_=False and natorb=False")
    if not np.isfinite(mc.max_stepsize) or mc.max_stepsize <= 0:
        raise ValueError("Adaptive Super-CI requires a positive finite max_stepsize")
    nc, na, nm = mc.ncore, mc.ncas, mo.shape[1]
    mask = mc.uniq_var_indices(nm, nc, na, mc.frozen)
    if np.count_nonzero(mask) != nc * (nm - nc) + (nm - nc - na) * na:
        raise ValueError("Adaptive Super-CI requires unscreened, unfrozen spinor rotations")


def davidson(hop, g, hdiag, sop=None, max_stepsize=1., tol=1e-9,
             neig=1, mmax=500, lindep=1e-12, log=None, step_radius=None):
    """Reuse one complex Krylov space across all orbital-shift trials.

    The whitened shift is mu*B^H B, not mu*I. Its projection also measures
    the physical step norm. Every accepted solve checks both full and
    retained-coordinate residuals, including discarded occupation directions.
    """
    from scipy.linalg import eigh
    expand, restrict, diagonal, shift_diag, cap = hop._superci_frame
    if step_radius is not None:
        cap = step_radius / np.sqrt(2)
    if neig != 1 or mmax < 1 or not np.isfinite(cap) or cap <= 0:
        raise ValueError('Adaptive Super-CI requires one root and positive space/radius')
    g = np.asarray(g, complex)
    wg = restrict(g)
    if np.linalg.norm(g) <= tol and np.linalg.norm(wg) <= tol:
        return np.zeros_like(g), 0., dict(converged=True, iterations=0,
            residual_norm=float(max(np.linalg.norm(g), np.linalg.norm(wg))),
            orbital_shift=0., step_norm=0., total_davidson_iterations=0,
            reason='zero_gradient')
    limit = min(mmax, len(wg))
    vectors = np.empty((len(wg), limit), complex)
    products = np.empty((len(g), limit), complex)
    white_products = np.empty_like(vectors)
    matrix = np.zeros((limit, limit), complex)
    metric = np.zeros_like(matrix)
    source = np.zeros(limit, complex)
    trial = -wg / np.maximum(abs(diagonal), 1e-3)
    history, shift_trials = [], []
    projected_solves = 0
    for iteration in range(1, limit+1):
        k = iteration-1
        trial /= np.linalg.norm(trial)
        vectors[:, k] = trial
        products[:, k] = hop(expand(trial))
        white_products[:, k] = restrict(products[:, k])
        basis = vectors[:, :iteration]
        column = .5*(basis.conj().T @ white_products[:, k]
                      + white_products[:, :iteration].conj().T @ trial)
        matrix[:iteration, k] = column
        matrix[k, :iteration] = column.conj()
        column = basis.conj().T @ (shift_diag*trial)
        metric[:iteration, k] = column
        metric[k, :iteration] = column.conj()
        source[k] = np.vdot(trial, wg)
        ah = np.zeros((iteration+1, iteration+1), complex)
        ah[1:, 0], ah[0, 1:] = source[:iteration], source[:iteration].conj()
        projected_metric = metric[:iteration, :iteration]

        def solve(shift):
            nonlocal projected_solves
            projected_solves += 1
            ah[1:, 1:] = matrix[:iteration, :iteration] + shift*projected_metric
            values, eigenvectors = eigh(ah)
            roots = np.flatnonzero(abs(eigenvectors[0]) > .1)
            if not len(roots):
                raise RuntimeError('Shifted Super-CI has no usable reference component')
            root = roots[0]
            coeff = eigenvectors[1:, root] / eigenvectors[0, root]
            size = np.sqrt(max(0., np.vdot(coeff, projected_metric @ coeff).real))
            return coeff, float(values[root]), float(size)

        shift = 0.
        coeff, energy, size = solve(shift)
        if size > cap:
            lower, upper = 0., max(1e-8, np.linalg.norm(g)/cap)
            for _ in range(60):
                coeff, energy, size = solve(upper)
                if size <= cap:
                    shift = upper
                    break
                lower, upper = upper, 2*upper
            else:
                raise RuntimeError('No shifted Super-CI step satisfies the radius')
            for _ in range(32):
                if size >= .995*cap:
                    break
                middle = .5*(lower+upper)
                candidate, value, length = solve(middle)
                if length <= cap:
                    upper = shift = middle
                    coeff, energy, size = candidate, value, length
                else:
                    lower = middle
        step = expand(basis @ coeff)
        raw = products[:, :iteration] @ coeff + shift*step + g - energy*sop(step)
        reference = abs(np.vdot(g, step)-energy)
        retained = restrict(raw)
        raw_norm = float(np.hypot(np.linalg.norm(raw), reference))
        white_norm = float(np.hypot(np.linalg.norm(retained), reference))
        residual = max(raw_norm, white_norm)
        converged = residual <= tol and np.linalg.norm(step) <= cap*(1+1e-12)
        history.append(residual)
        shift_trials.append(dict(shift=shift, step_norm=size, full_residual=raw_norm,
                                 retained_residual=white_norm, iterations=1, converged=converged))
        info = dict(converged=converged, iterations=iteration, residual_norm=residual,
                    residual_history=history.copy(), raw_full_residual=raw_norm,
                    retained_residual=white_norm, orbital_shift=shift, step_norm=size,
                    retained_dimension=len(wg), full_dimension=len(g),
                    total_davidson_iterations=iteration, projected_shift_solves=projected_solves,
                    shift_trials=shift_trials.copy(), shift_trial_count=len(shift_trials),
                    step_cap_fraction=size/cap,
                    reason=('converged_shifted' if shift else 'converged') if converged else 'maximum_space')
        if log is not None:
            log.info('Adaptive Super-CI %d: shift=%.6g step=%.6g residual=%.3e projected_solves=%d',
                     iteration, shift, size, residual, projected_solves)
        if converged:
            return step, energy, info
        denom = diagonal + shift*shift_diag-energy
        denom = np.where(abs(denom)>1e-8, denom, np.where(denom<0, -1e-8, 1e-8))
        trial = -retained/denom
        for _ in range(2):
            trial -= basis @ (basis.conj().T @ trial)
        if np.linalg.norm(trial) <= lindep:
            info['reason'] = 'linear_dependence'
            return step, energy, info
    return step, energy, info
