"""Fixed-RDM complex orbital Hessian and scaled augmented-Hessian steps.

The organization follows BAGEL's ZCASSecond::compute_hess_trial and AugHess
(Reynolds, Yanai, Shiozaki, JCP 149, 014106, 2018). Integrals and derivatives
here refer to our fixed X2C Hamiltonian. The Hessian is real-linear in the
complex rotation parameters.
"""
import numpy as np
from pyscf import lib
from pyscf.lib import logger
from scipy.linalg import eigh

from socutils.mcscf import zmc_ao2mo, zmc_utils
from socutils.mcscf.zmc_utils import build_orbital_quantities


AHNumericalError = zmc_utils.OrbitalSolveNumericalError


def kernel(mc, mo_coeff, *, max_stepsize=.2, conv_tol=None,
           conv_tol_grad=None, verbose=5, cderi=None, bfgs=False,
           solver='davidson', davidson_maxiter=40, davidson_tol=1e-8,
           davidson_strict=True, symm=None, callback=None):
    """Optimize orbitals and CI with a fixed-RDM AH step at each macro point."""
    from socutils.mcscf import zmcscf

    mo = np.array(mo_coeff, dtype=np.complex128, copy=True)
    kramers = zmc_utils._resolve_kramers_mode(mc, symm)
    validate(mc, mo, solver=solver, kramers=kramers)
    if bfgs:
        raise ValueError('Second-order AH requires BFGS disabled')
    conv_tol = mc.conv_tol if conv_tol is None else conv_tol
    conv_tol_grad = np.sqrt(conv_tol) if conv_tol_grad is None else conv_tol_grad
    cap = float(mc.second_order_max_rotation)
    if not np.isfinite(cap) or cap <= 0:
        raise ValueError('second_order_max_rotation must be finite and positive')
    max_radius = np.sqrt(2.) * cap
    radius = max_radius
    log = logger.new_logger(mc, verbose)
    log.info('Orbital optimizer = second-order Hessian / scaled AH (BAGEL scheme)')
    log.info('Second-order independent rotation-vector cap = %.6g', cap)
    log.info('MCSCF Kramers = %s', kramers)
    log.info('AH Davidson tolerance = %.3g, maximum space = %d, strict = %s',
             davidson_tol, davidson_maxiter, davidson_strict)
    eris, provenance = zmc_utils._build_eris(mc, mo, cderi=cderi)
    mc.cholesky_diagnostics = dict(provenance)
    log.info('MCSCF ERI route: %s, source = %s, naux = %s',
             provenance['representation'], provenance['source'], provenance['naux'])
    cas = zmcscf._fake_h_for_fast_casci(mc, mo, eris)
    energy, ecas, ci = cas.kernel(mo, verbose=verbose)
    if not np.all(getattr(mc.fcisolver, 'converged', True)):
        raise RuntimeError('The active-space CI solver did not converge')
    dm1, dm2 = mc.fcisolver.make_rdm12(ci, mc.ncas, mc.nelecas)
    mc.macro_history = []
    mc.canonicalization_diagnostics = None
    mc.e_tot, mc.e_cas, mc.ci = energy, ecas, ci
    previous_change = np.inf
    previous_step = 0.
    last_linear_info = None
    converged = False
    for macro in range(mc.max_cycle_macro):
        start = logger.perf_counter()
        g, hd, hop, sop, _, _ = gen_g_hop(
            mc, mo, dm1, dm2, eris, kramers=kramers)
        gradient_norm = float(np.linalg.norm(g))
        log.info('MCSCF macro = %4d | E = %22.15f | dE = %11.3e | '
                 'Grad norm = %9.3e | Step norm = %9.3e',
                 macro, energy, previous_change, gradient_norm, previous_step)
        row = dict(macro_iteration=macro, total_energy=float(energy),
                   energy_change=None if not np.isfinite(previous_change) else float(previous_change),
                   cas_energy=float(ecas), orbital_gradient_norm=gradient_norm,
                   orbital_step_norm=previous_step, accepted=True, converged=False,
                   ci_solver_converged=True,
                   ci_solver_diagnostics=zmc_utils._ci_convergence_snapshot(mc.fcisolver),
                   integral_representation=provenance['representation'],
                   integral_factorized=provenance['factorized'],
                   integral_source=provenance['source'],
                   cholesky_active=provenance['active'],
                   cholesky_naux=provenance['naux'],
                   orbital_method='second_order', adaptive=False,
                   superci_metric=dict(mc.superci_metric_diagnostics))
        mc.macro_history.append(row)
        if abs(previous_change) < conv_tol and gradient_norm < conv_tol_grad:
            converged = row['converged'] = True
            zmc_utils._schedule_orbital_trial(mc, gradient_norm, None)
            log.info('MCSCF converged | Macro = %4d | E = %22.15f | Grad norm = %.3e',
                     macro, energy, gradient_norm)
            if callback is not None:
                callback(dict(row))
            break
        (next_mo, next_energy, next_ecas, next_ci, next_eris, next_provenance,
         rotation, step, last_linear_info, predicted, change, ratio, radius,
         trials) = zmc_utils._bounded_orbital_update(
            mc, mo, energy, g, hd, hop, sop, davidson, radius, max_radius,
            davidson_tol, davidson_maxiter, conv_tol, conv_tol_grad,
            True, verbose, log, cderi=cderi)
        previous_step = float(np.linalg.norm(rotation))
        row.update(linear_solver=last_linear_info, orbital_trials=trials,
                   proposed_orbital_step_norm=previous_step,
                   applied_orbital_step_norm=previous_step, step_rescaled=False,
                   next_total_energy=float(next_energy), accepted_energy_change=float(change),
                   predicted_energy_change=float(predicted),
                   prediction_model='quadratic_orbital_hessian',
                   next_ci_solver_diagnostics=zmc_utils._ci_convergence_snapshot(mc.fcisolver),
                   restart_before_trial=trials[-1]['restart'],
                   rejected_trials=sum(t['stage'] == 'casci' and not t['accepted'] for t in trials),
                   inner_solver_failures=sum(t['stage'] == 'orbital_solve' for t in trials),
                   trial_radius=trials[-1]['radius'], trust_radius=float(radius),
                   trust_action='accepted / bounded AH',
                   macro_wall_time=float(logger.perf_counter()-start))
        log.info('MCSCF update = %3d | E = %22.15f | dE = %.3e | Pred = %.3e | '
                 'Ratio = %s | Step = %.3e | Trust = %.3e',
                 macro, next_energy, change, predicted, ratio, previous_step, radius)
        mo, energy, ecas, ci, eris, provenance = (
            next_mo, next_energy, next_ecas, next_ci, next_eris, next_provenance)
        previous_change = float(change)
        dm1, dm2 = mc.fcisolver.make_rdm12(ci, mc.ncas, mc.nelecas)
        mc.e_tot, mc.e_cas, mc.ci = energy, ecas, ci
        if callback is not None:
            callback(dict(row))
        lib.chkfile.save(mc.chkfile, f'mo_coeff_iter_{macro+1}', mo)

    mo_energy = None
    if mc.canonicalization:
        mo, ci, mo_energy = mc.canonicalize(
            mo, ci, eris=eris, sort=mc.sorting_mo_energy,
            cas_natorb=False, casdm1=dm1, verbose=verbose)
    else:
        mc.canonicalization_diagnostics = {
            'enabled': False, 'reason': 'mc.canonicalization is False'}
    mc.mo_coeff, mc.mo_energy = mo, mo_energy
    mc.final_orbital_gradient_norm = gradient_norm
    diagnostics = dict(adaptive=False, orbital_method='second_order',
                       converged=converged, final_gradient_norm=gradient_norm,
                       energy_tolerance=float(conv_tol),
                       gradient_tolerance=float(conv_tol_grad),
                       linear_solver=last_linear_info,
                       metric=dict(mc.superci_metric_diagnostics),
                       integrals=dict(provenance), cholesky=dict(provenance),
                       canonicalization=dict(mc.canonicalization_diagnostics),
                       kramers_restricted=bool(kramers),
                       macro_iterations=sum(not row['converged'] for row in mc.macro_history))
    mc.second_order_diagnostics = diagnostics
    mc.superci_diagnostics = diagnostics  # existing callers inspect this key
    return converged, energy, ecas, ci, mo, mo_energy


def validate(mc, mo, *, solver='davidson', kramers=False):
    """AH restrictions, independent of the adaptive Super-CI policy."""
    if solver != 'davidson':
        raise ValueError("Second-order AH requires solver='davidson'")
    if mc.natorb or mc.canonicalize_:
        raise ValueError('Second-order AH requires natorb=False and canonicalize_=False')
    if not np.isfinite(mc.max_stepsize) or mc.max_stepsize <= 0:
        raise ValueError('Second-order AH requires a positive finite max_stepsize')
    if kramers:
        zmc_utils._identify_kramers_mapping(mc, mo)


def gen_g_hop(mc, mo, dm1, dm2, eris, *, kramers=None):
    if kramers is None:
        kramers = zmc_utils._resolve_kramers_mode(mc)
    validate(mc, mo, kramers=kramers)
    if not isinstance(eris, (zmc_ao2mo._ERIS, zmc_ao2mo._CDERIS)):
        raise TypeError('Second-order AH requires a full or factorized ERI container')
    nc, na = mc.ncore, mc.ncas
    no, nm = nc + na, mo.shape[1]
    active = slice(nc, no)
    q = build_orbital_quantities(mc, mo, dm1, dm2, eris)
    density, metric_info = zmc_utils._physical_active_density(dm1.T)
    mc.superci_metric_diagnostics = metric_info
    factorized = isinstance(eris, zmc_ao2mo._CDERIS)
    if not factorized:
        block_mb = 3 * nm**2 * na**2 * 16 / 1e6
        if block_mb > max(0., mc.max_memory - lib.current_memory()[0]) * .6:
            raise MemoryError('Second-order MO blocks need %.0f MB; increase max_memory' % block_mb)
    blocks = []
    core = np.zeros((nm, nm), complex)
    core[:nc, :nc] = np.eye(nc)
    total = core.copy()
    total[active, active] = dm1.T
    gradient = q.gradient
    mapping = zmc_utils._identify_kramers_mapping(mc, mo) if kramers else None

    def project(v):
        if mapping is None:
            return v
        matrix, _ = zmc_utils._project_kramers_rotation(
            mc, mo, mc.unpack_uniq_var(v), force=True, mapping=mapping)
        return mc.pack_uniq_var(matrix)

    projected_gradient = project(mc.pack_uniq_var(gradient))

    def hop(v):
        if not factorized and not blocks:
            lib.logger.info(mc, 'Second-order shared integral blocks: aapp, papa, paap')
            blocks.extend(eris.second_order_blocks(reserved_mb=block_mb))
        x = mc.unpack_uniq_var(project(v))
        responses = np.array([x @ core - core @ x, x @ total - total @ x])
        j, k = eris.get_jk(mo @ responses @ mo.conj().T)
        dc, dt = mo.conj().T @ (j-k) @ mo
        fc = q.fock_core @ x - x @ q.fock_core + dc
        ft = q.fock_effective @ x - x @ q.fock_effective + dt
        # Differentiate all four orbital indices of (p u|v w). This includes
        # the Q' and Q'' terms absent from a mixed-one-body Super-CI operator.
        if factorized:
            # Bilinear CD reconstruction: (pu|vw) = sum_P L[P,p,u] L[P,v,w].
            # Only the rotated active columns need another AO half transform.
            trial = eris.transform_trial_active(mo @ x[:, active])
            delta_aa = -lib.einsum('vq,Pqw->Pvw', x[active, :], eris.cd_pa)
            delta_aa += lib.einsum('Pqv,qw->Pvw', eris.cd_pa.conj(), x[:, active])
            first = lib.einsum('Pvw,tuvw->Ptu', eris.cd_aa, dm2)
            second = lib.einsum('Pvw,tuvw->Ptu', delta_aa, dm2)
            dq = -x @ q.two_rdm_contraction
            dq += lib.einsum('Ppu,Ptu->pt', trial, first)
            dq += lib.einsum('Ppu,Ptu->pt', eris.cd_pa, second)
        else:
            ppaa, papa, paap = blocks
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
        return project(mc.pack_uniq_var(hg))

    # Natural/semicanonical coordinates are used only by the preconditioner:
    # the physical active orbitals, RDMs and finite-M MPS never rotate.
    parts = [slice(0, nc), active, slice(no, nm)]
    matrices = [q.fock_effective[parts[0], parts[0]], density,
                q.fock_effective[parts[2], parts[2]]]
    units = [zmc_utils._kramers_subspace_eigh(mc, matrix, mo[:, part])[1]
             if mapping is not None else eigh(matrix)[1]
             for matrix, part in zip(matrices, parts)]
    fd = np.concatenate([np.diag(u.conj().T @ q.fock_effective[p, p] @ u).real
                         for p, u in zip(parts, units)])
    ld = np.diag(units[1].conj().T @ q.lagrangian[active, active] @ units[1]).real.copy()
    occ = np.diag(units[1].conj().T @ density @ units[1]).real.copy()
    if mapping is not None:
        # The internal eigencoordinates are explicitly pair ordered.
        for values in (fd, ld, occ):
            pair_mean = .5*(values[::2] + values[1::2])
            values[::2] = values[1::2] = pair_mean
    diagonal = np.zeros((nm, nm))
    diagonal[no:, :nc] = fd[no:, None] - fd[None, :nc]
    diagonal[active, :nc] = (fd[active]-ld)[:, None] - (1-occ)[:, None]*fd[None, :nc]
    diagonal[no:, active] = fd[no:, None]*occ[None, :] - ld[None, :]
    # BAGEL compute_denom replaces an unshifted near-zero denominator by 1.
    diagonal[abs(diagonal) < 1e-15] = 1.
    hd = mc.pack_uniq_var(diagonal)

    def precondition(vector, shift, floor):
        mat = mc.unpack_uniq_var(project(vector))
        out = np.zeros_like(mat)
        for i, j in ((1, 0), (2, 0), (2, 1)):
            pi, pj = parts[i], parts[j]
            ui, uj = units[i], units[j]
            block = ui.conj().T @ mat[pi, pj] @ uj
            denominator = diagonal[pi, pj] - shift
            block /= np.where(abs(denominator) > floor, denominator, 1.)
            out[pi, pj] = ui @ block @ uj.conj().T
        return project(mc.pack_uniq_var(out))
    hop.precondition = precondition
    hop.project = project
    hop.orbital_hessian = True
    return projected_gradient, hd, hop, lambda v: v, None, mo


def davidson(hop, g, hdiag, sop=None, max_stepsize=.2, tol=1e-8,
             neig=1, mmax=40, lindep=1e-12, log=None, micro_step_tol=0.):
    """Scaled AH in a real Krylov space, including the complex conjugate part.

    For each projected problem solve [[0,g^T],[g,H/lambda]], then set
    x = v_orb/(lambda*v_ref). Adjust lambda in that small problem to bound
    ||x|| before taking another Hessian product, as in BAGEL AugHess.
    """
    if neig != 1 or mmax < 1 or not np.isfinite(max_stepsize) or max_stepsize <= 0:
        raise ValueError('Scaled AH requires one root, positive space and step radius')
    project = getattr(hop, 'project', lambda v: v)
    g = project(np.asarray(g, complex))
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
    def diagonal_precondition(v, shift, floor):
        denominator = hd-shift
        return v / np.where(abs(denominator) > floor, denominator, 1.)

    precondition = getattr(hop, 'precondition', diagonal_precondition)
    trial = project(precondition(g, -.001, 1e-12))
    limit = min(mmax, 2*len(g))
    vectors = np.empty((len(g), limit), complex)
    products = np.empty_like(vectors)
    matrix = np.zeros((limit, limit))
    projected_g = np.zeros(limit)
    history = []
    for iteration in range(1, limit + 1):
        trial = project(trial)
        trial_norm = np.linalg.norm(trial)
        if not np.isfinite(trial_norm) or trial_norm <= lindep:
            raise AHNumericalError('AH trial vector has zero or nonfinite norm')
        trial /= trial_norm
        k = iteration-1
        vectors[:, k] = trial
        products[:, k] = project(hop(trial))
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
                raise AHNumericalError('Scaled AH has no root with a usable reference component')
            root = roots[0]
            return coeffs[1:, root] / (scale*coeffs[0, root]), float(energies[root])

        # BAGEL AugHess::compute_lambda_: at most ten secant-like updates.
        scale, last_scale, last_size, below = 1., 0., 0., 0
        for scale_iter in range(10):
            if not np.isfinite(scale):
                raise AHNumericalError('AH scale is not finite')
            coeff, energy = solve(scale)
            size = float(np.linalg.norm(coeff))
            if not np.isfinite(size):
                raise AHNumericalError('AH step has nonfinite norm')
            if scale_iter == 0:
                if size <= cap:
                    break
                last_scale, scale = scale, size/cap
            else:
                if abs(size-cap)/cap < .01:
                    break
                if size > cap:
                    last_scale, scale = scale, scale*size/cap
                else:
                    if below > 2:
                        break
                    below += 1
                    d1, d2 = cap-size, last_size-cap
                    if d1 == 0. or d1 == -d2:
                        break
                    old_scale, last_scale = last_scale, scale
                    scale = (d1*old_scale + d2*scale)/(d1+d2)
            scale = max(1., scale)
            last_size = size
        else:
            coeff, energy = solve(scale)
        step = project(basis @ coeff)
        hstep = project(sigma @ coeff)
        residual = project(g + hstep - scale*energy*step)
        residual_norm = float(np.linalg.norm(residual))
        history.append(residual_norm)
        scaled_residual = residual_norm / scale
        effective_tol = max(tol, float(np.linalg.norm(step))*micro_step_tol)
        converged = scaled_residual <= effective_tol
        info = dict(converged=converged, iterations=iteration,
            residual_norm=residual_norm, residual_history=history.copy(),
            step_norm=float(np.linalg.norm(step)), ah_scale=scale,
            orbital_shift=-scale*energy, total_davidson_iterations=iteration,
            reason=('converged' if scaled_residual <= tol else 'converged_inexact') if converged else 'maximum_space',
            scaled_residual_norm=scaled_residual, effective_tolerance=effective_tol,
            strict_converged=scaled_residual <= tol, micro_step_tolerance=micro_step_tol,
            quadratic_form=float(np.vdot(step, hstep).real),
            preconditioner='natural-semicanonical' if hasattr(hop, 'precondition') else 'diagonal')
        if log is not None:
            log.info('Second-order AH %d: residual=%.3e lambda=%.5g step=%.5g',
                     iteration, residual_norm, scale, np.linalg.norm(step))
        if converged:
            return step, energy, info
        trial = project(precondition(residual, scale*energy, 1e-12))
        # Real coefficients are essential: hop(i*x) need not equal i*hop(x).
        for _ in range(10):
            trial -= basis @ (basis.conj().T @ trial).real
            trial_norm = np.linalg.norm(trial)
            if trial_norm <= lindep:
                info['reason'] = 'linear_dependence'
                return step, energy, info
            trial /= trial_norm
            if trial_norm > .25:
                break
    return step, energy, info
