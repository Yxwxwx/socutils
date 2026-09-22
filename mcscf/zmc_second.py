"""Fixed-RDM complex orbital Hessian and scaled augmented-Hessian steps.

The organization follows BAGEL's ZCASSecond::compute_hess_trial and AugHess
(Reynolds, Yanai, Shiozaki, JCP 149, 014106, 2018). Integrals and derivatives
here refer to our fixed X2C Hamiltonian, using full ERIs and no Kramers
projection. The Hessian is real-linear in the complex rotation parameters.
"""
import numpy as np
from pyscf import lib
from scipy.linalg import eigh

from socutils.mcscf import zmc_superci as sci
from socutils.mcscf.zmc_supercipt import build_orbital_quantities


def gen_g_hop(mc, mo, dm1, dm2, eris):
    from socutils.mcscf.zmc_superci_adaptive import validate
    validate(mc, mo)
    if not isinstance(eris, sci.zmc_ao2mo._ERIS):
        raise ValueError('Second-order orbital optimization requires full ERIs')
    nc, na = mc.ncore, mc.ncas
    no, nm = nc + na, mo.shape[1]
    active = slice(nc, no)
    a = mo[:, active]
    q = build_orbital_quantities(mc, mo, dm1, dm2, eris)
    density, metric_info = sci._physical_active_density(dm1.T)
    mc.superci_metric_diagnostics = metric_info
    # ponytail: O(nmo^2*ncas^2) in-core blocks; use streamed contractions if
    # these three blocks exceed the available memory on a larger system.
    block_mb = 3 * nm**2 * na**2 * 16 / 1e6
    if block_mb > max(0., mc.max_memory - lib.current_memory()[0]) * .6:
        raise MemoryError('Second-order MO blocks need %.0f MB; increase max_memory' % block_mb)
    blocks = []
    core = np.zeros((nm, nm), complex)
    core[:nc, :nc] = np.eye(nc)
    total = core.copy()
    total[active, active] = dm1.T
    gradient = q.gradient

    def hop(v):
        if not blocks:
            lib.logger.info(mc, 'Second-order shared integral blocks: aapp, papa, paap')
            blocks.extend(eris.second_order_blocks(reserved_mb=block_mb))
        ppaa, papa, paap = blocks
        x = mc.unpack_uniq_var(v)
        responses = np.array([x @ core - core @ x, x @ total - total @ x])
        j, k = eris.get_jk(mo @ responses @ mo.conj().T)
        dc, dt = mo.conj().T @ (j-k) @ mo
        fc = q.fock_core @ x - x @ q.fock_core + dc
        ft = q.fock_effective @ x - x @ q.fock_effective + dt
        # Differentiate all four orbital indices of (p u|v w). This includes
        # the Q' and Q'' terms absent from a mixed-one-body Super-CI operator.
        dv = lib.einsum('pqvw,qu->puvw', ppaa, x[:, active])
        dv -= lib.einsum('puqw,vq->puvw', papa, x[active, :])
        dv += lib.einsum('puvq,qw->puvw', paap, x[:, active])
        dq = -x @ q.two_rdm_contraction + lib.einsum('puvw,tuvw->pt', dv, dm2)
        dl = np.zeros_like(x)
        dl[:, :nc] = ft[:, :nc]
        dl[:, active] = fc[:, active] @ dm1.T + dq
        # Convert the moving-body gradient derivative to the exponential
        # chart at the current orbitals. Omitting this term breaks symmetry.
        hg = dl - dl.conj().T + .5 * (x @ gradient - gradient @ x)
        return mc.pack_uniq_var(hg)

    # Natural/semicanonical coordinates are used only by the preconditioner:
    # the physical active orbitals, RDMs and finite-M MPS never rotate.
    parts = [slice(0, nc), active, slice(no, nm)]
    units = [eigh(q.fock_effective[parts[0], parts[0]])[1],
             eigh(density)[1], eigh(q.fock_effective[parts[2], parts[2]])[1]]
    fd = np.concatenate([np.diag(u.conj().T @ q.fock_effective[p, p] @ u).real
                         for p, u in zip(parts, units)])
    ld = np.diag(units[1].conj().T @ q.lagrangian[active, active] @ units[1]).real
    occ = np.diag(units[1].conj().T @ density @ units[1]).real
    diagonal = np.zeros((nm, nm))
    diagonal[no:, :nc] = fd[no:, None] - fd[None, :nc]
    diagonal[active, :nc] = (fd[active]-ld)[:, None] - (1-occ)[:, None]*fd[None, :nc]
    diagonal[no:, active] = fd[no:, None]*occ[None, :] - ld[None, :]
    hd = mc.pack_uniq_var(diagonal)

    def precondition(vector, shift, floor):
        mat = mc.unpack_uniq_var(vector)
        out = np.zeros_like(mat)
        for i, j in ((1, 0), (2, 0), (2, 1)):
            pi, pj = parts[i], parts[j]
            ui, uj = units[i], units[j]
            block = ui.conj().T @ mat[pi, pj] @ uj
            block /= np.maximum(abs(diagonal[pi, pj]-shift), floor)
            out[pi, pj] = ui @ block @ uj.conj().T
        return mc.pack_uniq_var(out)
    hop.precondition = precondition
    hop.orbital_hessian = True
    return mc.pack_uniq_var(gradient), hd, hop, lambda v: v, None, mo


def davidson(hop, g, hdiag, sop=None, max_stepsize=.2, tol=1e-8,
             neig=1, mmax=40, lindep=1e-12, log=None, micro_step_tol=0.):
    """Scaled AH in a real Krylov space, including the complex conjugate part.

    For each projected problem solve [[0,g^T],[g,H/lambda]], then set
    x = v_orb/(lambda*v_ref). Adjust lambda in that small problem to bound
    ||x|| before taking another Hessian product, as in BAGEL AugHess.
    """
    if neig != 1 or mmax < 1 or not np.isfinite(max_stepsize) or max_stepsize <= 0:
        raise ValueError('Scaled AH requires one root, positive space and step radius')
    g = np.asarray(g, complex)
    hd = np.asarray(hdiag, float)
    if g.ndim != 1 or hd.shape != g.shape or not np.all(np.isfinite(g)):
        raise ValueError('Invalid scaled-AH gradient/diagonal')
    cap = max_stepsize / np.sqrt(2)
    if np.linalg.norm(g) <= tol:
        return np.zeros_like(g), 0., dict(converged=True, iterations=0,
            residual_norm=float(np.linalg.norm(g)), reason='zero_gradient', step_norm=0.,
            quadratic_form=0., strict_converged=True, effective_tolerance=tol,
            scaled_residual_norm=float(np.linalg.norm(g)), micro_step_tolerance=0.)
    if not np.isfinite(micro_step_tol) or micro_step_tol < 0:
        raise ValueError('micro_step_tol must be nonnegative and finite')
    precondition = getattr(hop, 'precondition',
                           lambda v, shift, floor: v / np.maximum(abs(hd-shift), floor))
    trial = -precondition(g, -.001, 1e-3)
    limit = min(mmax, 2*len(g))
    vectors = np.empty((len(g), limit), complex)
    products = np.empty_like(vectors)
    matrix = np.zeros((limit, limit))
    projected_g = np.zeros(limit)
    history = []
    for iteration in range(1, limit + 1):
        trial /= np.linalg.norm(trial)
        k = iteration-1
        vectors[:, k] = trial
        products[:, k] = hop(trial)
        basis, sigma = vectors[:, :iteration], products[:, :iteration]
        column = .5*((basis.conj().T @ products[:, k]).real
                      + (sigma.conj().T @ trial).real)
        matrix[:iteration, k] = matrix[k, :iteration] = column
        projected_g[k] = np.vdot(trial, g).real
        projected, source = matrix[:iteration, :iteration], projected_g[:iteration]

        def solve(scale):
            ah = np.zeros((iteration+1, iteration+1))
            ah[0, 1:] = ah[1:, 0] = source
            ah[1:, 1:] = projected / scale
            energies, coeffs = eigh(ah)
            roots = np.flatnonzero(abs(coeffs[0]) > .1)
            if not len(roots):
                raise RuntimeError('Scaled AH has no root with a usable reference component')
            root = roots[0]
            return coeffs[1:, root] / (scale*coeffs[0, root]), float(energies[root])

        scale = 1.
        coeff, energy = solve(scale)
        if np.linalg.norm(coeff) > cap:
            lower, upper = 1., 2.
            while True:
                candidate, eig = solve(upper)
                if np.linalg.norm(candidate) <= cap:
                    coeff, energy, scale = candidate, eig, upper
                    break
                lower, upper = upper, upper*2
                if upper > 1e12:
                    raise RuntimeError('Scaled AH could not bound the orbital step')
            for _ in range(32):
                if np.linalg.norm(coeff) >= .995*cap:
                    break
                middle = (lower+upper)*.5
                candidate, eig = solve(middle)
                if np.linalg.norm(candidate) <= cap:
                    upper, scale, coeff, energy = middle, middle, candidate, eig
                else:
                    lower = middle
        step = basis @ coeff
        hstep = sigma @ coeff
        residual = g + hstep - scale*energy*step
        reference = abs(scale*np.vdot(g, step).real - energy)
        residual_norm = float(np.hypot(np.linalg.norm(residual), reference))
        history.append(residual_norm)
        scaled_residual = residual_norm / scale
        effective_tol = max(tol, float(np.linalg.norm(step))*micro_step_tol)
        # A zero relative tolerance retains the original strict raw test.
        converged = (scaled_residual <= effective_tol if micro_step_tol else residual_norm <= tol)
        info = dict(converged=converged, iterations=iteration,
            residual_norm=residual_norm, residual_history=history.copy(),
            step_norm=float(np.linalg.norm(step)), ah_scale=scale,
            orbital_shift=-scale*energy, total_davidson_iterations=iteration,
            reason=('converged' if residual_norm <= tol else 'converged_inexact') if converged else 'maximum_space',
            scaled_residual_norm=scaled_residual, effective_tolerance=effective_tol,
            strict_converged=residual_norm <= tol, micro_step_tolerance=micro_step_tol,
            quadratic_form=float(np.vdot(step, hstep).real),
            preconditioner='natural-semicanonical' if hasattr(hop, 'precondition') else 'diagonal')
        if log is not None:
            log.info('Second-order AH %d: residual=%.3e lambda=%.5g step=%.5g',
                     iteration, residual_norm, scale, np.linalg.norm(step))
        if converged:
            return step, energy, info
        trial = -precondition(residual, scale*energy, 1e-8)
        # Real coefficients are essential: hop(i*x) need not equal i*hop(x).
        for _ in range(2):
            trial -= basis @ (basis.conj().T @ trial).real
        if np.linalg.norm(trial) <= lindep:
            info['reason'] = 'linear_dependence'
            return step, energy, info
    return step, energy, info
