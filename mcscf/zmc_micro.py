"""One-step complex-spinor CASSCF with CIAH and finite-sweep DMRG response.

Control-flow reference: Q. Sun, J. Yang and G. K.-L. Chan, Chem. Phys. Lett.
683, 291 (2017), doi:10.1016/j.cplett.2017.03.004; PySCF mc1step/ciah
(Apache-2.0, copyright The PySCF Developers, author Qiming Sun).

The orbital derivatives are those of this repository's zmc_ah, not PySCF's
spin-free formulas. RDM1[p,q]=<p+ q>, RDM2[p,q,r,s]=<p+ r+ s q>.
chemist ERI[p,q,r,s]=(pq|rs). One site is one spinor.

A macro fixes an orbital chart C(x)=C_anchor exp(kappa(x)) and one real-linear
Hessian. CIAH updates its source but never changes cached Hx products. At a
keyframe a short CI/DMRG solve updates BOTH RDMs. The moving-body gradient is
pulled back through dexp to the SAME anchor chart. The default fast model is
exact 1e/core + DEP1 active two-electron integrals (T=U-I). Exact micro mode
is a deliberately expensive reference. Neither mode needs 3/4-RDMs.

Single electronic root only in this version. General complex and optional
Kramers-restricted orbital rotations, frozen/irrep masks, full and CD ERIs are
supported. SA is rejected, not silently treated as a single state.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import gc
import time

import numpy as np
from scipy.linalg import eigh
from pyscf import lib
from pyscf.lib import logger
from pyscf.soscf import ciah

from socutils.mcscf import zmc_ah, zmc_ao2mo, zmc_utils


DEFAULTS = dict(
    max_cycle_micro=3,
    micro_integral_mode='dep1',
    micro_ci_sweeps=2,
    micro_ci_davidson_threshold=1e-12,
    micro_max_component=.02,
    micro_kf_interval=2,
    micro_kf_trust=3.,
    micro_ah_start_cycle=2,
    micro_ah_start_tol=1e-3,
    micro_ah_lindep=1e-14,
    micro_ah_level_shift=0.,
    micro_max_trials=6,
    micro_rdm_tol=1e-7,
    micro_diagnostics=None,
)


class MicroNumericalError(RuntimeError):
    """A tentative numerical state can be discarded and retried."""


def realify(z):
    z = np.asarray(z, dtype=np.complex128)
    if z.ndim != 1:
        raise ValueError('Expected a one-dimensional complex orbital vector')
    return np.concatenate((z.real, z.imag))


def unrealify(x):
    x = np.asarray(x)
    if x.ndim != 1 or x.size % 2 or np.iscomplexobj(x):
        raise ValueError('Expected a real vector of even length')
    n = x.size // 2
    return x[:n] + 1j*x[n:]


@dataclass
class RotationChart:
    """Fixed exponential chart and exact adjoint differential of exp."""
    generator: np.ndarray
    rotation: np.ndarray
    eigenvectors: np.ndarray
    phases: np.ndarray

    @classmethod
    def from_generator(cls, kappa):
        kappa = np.asarray(kappa, dtype=complex)
        if not np.all(np.isfinite(kappa)):
            raise MicroNumericalError('Nonfinite orbital generator')
        if np.linalg.norm(kappa+kappa.conj().T) > 1e-10*max(1., np.linalg.norm(kappa)):
            raise ValueError('Orbital generator must be anti-Hermitian')
        phases, vectors = eigh(-1j*kappa)
        rotation = (vectors*np.exp(1j*phases)) @ vectors.conj().T
        return cls(kappa.copy(), rotation, vectors, phases)

    def pullback(self, body_gradient):
        """Integral_0^1 exp(sK) G exp(-sK) ds; no log(U) approximation."""
        vectors = self.eigenvectors
        difference = self.phases[:, None]-self.phases[None, :]
        factor = np.exp(.5j*difference)*np.sinc(difference/(2*np.pi))
        transformed = vectors.conj().T @ body_gradient @ vectors
        return vectors @ (factor*transformed) @ vectors.conj().T


@dataclass
class ActiveHamiltonian:
    mo: np.ndarray
    h1: np.ndarray
    h2: np.ndarray
    ecore: float
    paaa: np.ndarray
    core_potential: np.ndarray
    eris: object = None
    provenance: dict | None = None


@dataclass
class Reference:
    mo: np.ndarray
    eris: object
    provenance: dict
    energy: float
    ecas: float
    dm1: np.ndarray
    dm2: np.ndarray
    # Exact CI arrays are safe copies; native MPS handles always stay in solver.
    ci: object = None


def _option(mc, name):
    return getattr(mc, name, DEFAULTS[name])


def _real(value, name, tol=1e-8):
    value = np.asarray(value)
    if value.ndim or not np.isfinite(value) or abs(value.imag) > tol:
        raise MicroNumericalError('%s must be a finite real scalar' % name)
    return float(value.real)


def _new_stats():
    return dict(full_ao2mo=0, exact_micro_ao2mo=0, cd_trial_transforms=0,
                ah_hops=0, jk_calls=0, jk_densities=0, jk_seconds=0.,
                strict_ci_calls=0, strict_ci_seconds=0., micro_ci_calls=0,
                micro_ci_sweeps=0, micro_ci_seconds=0., keyframes=0,
                rdm_seconds=0., max_model_cache_mb=0.)


@contextmanager
def _count_jk(mc, stats):
    mf = mc._scf
    previous = mf.__dict__.get('get_jk')
    was_local = 'get_jk' in mf.__dict__
    original = mf.get_jk

    def counted(*args, **kwargs):
        start = time.perf_counter()
        stats['jk_calls'] += 1
        dm = kwargs.get('dm', args[1] if len(args) > 1 else None)
        if dm is not None:
            shape = np.shape(dm)
            stats['jk_densities'] += int(np.prod(shape[:-2])) if len(shape) > 2 else 1
        try:
            return original(*args, **kwargs)
        finally:
            stats['jk_seconds'] += time.perf_counter()-start
    mf.get_jk = counted
    try:
        yield
    finally:
        if was_local:
            mf.get_jk = previous
        else:
            del mf.get_jk


def _rdms(mc, ci, stats):
    start = time.perf_counter()
    one, two = mc.fcisolver.make_rdm12(ci, mc.ncas, mc.nelecas)
    one, two = np.array(one, dtype=complex, copy=True), np.array(two, dtype=complex, copy=True)
    stats['rdm_seconds'] += time.perf_counter()-start
    if one.shape != (mc.ncas,)*2 or two.shape != (mc.ncas,)*4:
        raise ValueError('Unexpected spinor RDM shape')
    if not np.all(np.isfinite(one)) or not np.all(np.isfinite(two)):
        raise MicroNumericalError('Nonfinite micro RDM')
    n = int(mc.nelecas)
    errors = dict(trace=abs(np.trace(one)-n),
        hermiticity=np.max(abs(one-one.conj().T)),
        dm2_hermiticity=np.max(abs(two.conj()-two.transpose(1,0,3,2))),
        creation_antisymmetry=np.max(abs(two+two.transpose(2,1,0,3))),
        annihilation_antisymmetry=np.max(abs(two+two.transpose(0,3,2,1))),
    )
    # The interleaved convention gives sum_r Gamma[p,q,r,r]=(N-1)D[p,q].
    errors['contraction'] = float(np.max(abs(np.einsum('pqrr->pq', two)-(n-1)*one)))
    if max(errors.values()) > _option(mc, 'micro_rdm_tol'):
        raise MicroNumericalError('RDM convention/normalization check failed: %s' % errors)
    zmc_utils._physical_active_density(one.T, tolerance=_option(mc, 'micro_rdm_tol'))
    return one, two


def _build(mc, mo, cderi, stats, *, micro=False):
    eris, provenance = zmc_utils._build_eris(mc, mo, cderi=cderi)
    stats['full_ao2mo'] += 1
    stats['exact_micro_ao2mo'] += int(micro)
    # Fixed-anchor full blocks are shared by all Hx and DEP1 keyframes.
    eris._micro_cache_enabled = True
    return eris, provenance


def _close_eris(eris):
    """Release only containers created by this optimizer, never the SCF source."""
    if eris is None:
        return
    eris._micro_second_order_cache = None
    for name in ('_aa_half', 'feri'):
        handle = getattr(eris, name, None)
        if handle is not None:
            handle.close()
            setattr(eris, name, None)


def _full_reference(mc, mo, adapter, cderi, stats, verbose, *, ci0=None):
    from socutils.mcscf import zmcscf
    eris, provenance = _build(mc, mo, cderi, stats)
    try:
        cas = zmcscf._fake_h_for_fast_casci(mc, mo, eris)
        start = time.perf_counter()
        mc.ci = None  # do not keep a stale native MPS through driver replacement
        hint = adapter.prepare_strict(ci0)
        stats['strict_ci_calls'] += 1
        try:
            energy, ecas, ci = cas.kernel(mo, ci0=hint, verbose=verbose)
        finally:
            stats['strict_ci_seconds'] += time.perf_counter()-start
        # Do NOT use _ci_usable: its compatibility fallback accepts finite but
        # unconverged DMRG states. Macro acceptance here requires strict CI.
        if not bool(np.all(getattr(mc.fcisolver, 'converged', False))):
            raise MicroNumericalError('Strict macro CI/DMRG did not converge')
        energy, ecas = _real(energy, 'total energy'), _real(ecas, 'CAS energy')
        one, two = _rdms(mc, ci, stats)
        adapter.set_ci(ci)
        return Reference(np.array(mo, copy=True), eris, provenance, energy, ecas,
                         one, two, None if adapter.is_dmrg else deepcopy(ci))
    except Exception:
        _close_eris(eris)
        raise


def _one_core(mc, mo, eris):
    """Exact 1e/core for the fixed SCF integral source, with no stale core K cache."""
    nc, no = mc.ncore, mc.ncore+mc.ncas
    core, active = mo[:, :nc], mo[:, nc:no]
    density = core @ core.conj().T
    # No mo_coeff/mo_occ tags: a macro CDERIS cached K belongs to the old core.
    j, k = eris.get_jk(density)
    potential = j-k
    h = mc.get_hcore()
    h1 = active.conj().T @ (h+potential) @ active
    ecore = (mc.energy_nuc() + np.einsum('ij,ji', density, h)
             + .5*np.einsum('ij,ji', density, potential))
    return h1, _real(ecore, 'core energy'), potential


def dep1_paaa(eris, rotation, *, stats=None):
    """Four-slot DEP1 of (p u|v w), T=U-I, in the macro anchor basis.

    Full route reuses PPAA/PAPA/PAAP. CD uses one trial-active half transform,
    never a full P,P factor or nmo**4 tensor. Both implement the same bilinear
    Coulomb formula to first order in T (not in log(U)).
    """
    u = np.asarray(rotation, dtype=complex)
    t = u-np.eye(u.shape[0])
    a = slice(eris.ncore, eris.ncore+eris.ncas)
    paaa = eris.paaa
    if isinstance(eris, zmc_ao2mo._CDERIS):
        lpa, laa = eris.cd_pa, eris.cd_aa
        trial = eris.transform_trial_active(eris.mo @ t[:, a])
        if stats is not None:
            stats['cd_trial_transforms'] += 1
        delta_pa = lib.einsum('qp,Pqu->Ppu', t.conj(), lpa) + trial
        delta_aa = (lib.einsum('qv,Pqw->Pvw', t[:, a].conj(), lpa)
                    + lib.einsum('Pqv,qw->Pvw', lpa.conj(), t[:, a]))
        result = (paaa + lib.einsum('Ppu,Pvw->puvw', delta_pa, laa)
                  + lib.einsum('Ppu,Pvw->puvw', lpa, delta_aa))
        if stats is not None:
            stats['max_model_cache_mb'] = max(stats['max_model_cache_mb'],
                (lpa.nbytes+laa.nbytes+trial.nbytes+delta_pa.nbytes)/1e6)
    else:
        reserve = 3*u.shape[0]**2*eris.ncas**2*16/1e6
        ppaa, papa, paap = eris.second_order_blocks(reserved_mb=reserve)
        result = np.array(paaa, copy=True)
        result += lib.einsum('qp,quvw->puvw', t.conj(), paaa)
        result += lib.einsum('pqvw,qu->puvw', ppaa, t[:, a])
        result += lib.einsum('puqw,qv->puvw', papa, t[:, a].conj())
        result += lib.einsum('puvq,qw->puvw', paap, t[:, a])
        if stats is not None:
            stats['max_model_cache_mb'] = max(stats['max_model_cache_mb'],
                (ppaa.nbytes+papa.nbytes+paap.nbytes+paaa.nbytes+result.nbytes)/1e6)
    return result


def update_active_hamiltonian(mc, anchor, rotation, *, mode='dep1', cderi=None, stats=None):
    """Make one micro Hamiltonian; exact mode is a non-fast control."""
    if stats is None:
        stats = _new_stats()
    mo = anchor.mo @ rotation
    eris = None
    if mode == 'exact':
        eris, provenance = _build(mc, mo, cderi, stats, micro=True)
        paaa = eris.paaa
        source = eris
    elif mode == 'dep1':
        provenance = dict(anchor.provenance, micro_approximation='exact-1e-core+DEP1-2e')
        paaa = dep1_paaa(anchor.eris, rotation, stats=stats)
        source = anchor.eris
    else:
        raise ValueError("micro_integral_mode must be 'exact' or 'dep1'")
    try:
        h1, ecore, core_potential = _one_core(mc, mo, source)
        a = slice(mc.ncore, mc.ncore+mc.ncas)
        h2 = np.array(paaa[a], order='C', copy=True)
        error = max(np.max(abs(h1-h1.conj().T)),
                    np.max(abs(h2-h2.transpose(2,3,0,1))),
                    np.max(abs(h2.conj()-h2.transpose(1,0,3,2))))
        scale = max(1., float(np.max(abs(h1))), float(np.max(abs(h2))))
        if error > 1e-9*scale:
            raise MicroNumericalError('Micro integrals violate complex Coulomb symmetry: %.3e' % error)
        return ActiveHamiltonian(mo, h1, h2, ecore, paaa, core_potential, eris, provenance)
    except Exception:
        _close_eris(eris)
        raise


def gradient_at_keyframe(mc, hcas, dm1, dm2, chart, project=lambda x: x):
    """New-RDM gradient at current orbitals, pulled back to the fixed chart.

    This evaluates a fresh 2-RDM contraction, not g_old + H_old*x. In DEP1
    mode it is an approximate keyframe gradient of the true Hamiltonian.
    """
    mo = hcas.mo
    nc, no = mc.ncore, mc.ncore+mc.ncas
    active = mo[:, nc:no]
    density_a = active @ dm1.T @ active.conj().T
    j, k = mc._scf.get_jk(mc.mol, density_a)
    fc = mo.conj().T @ (mc.get_hcore()+hcas.core_potential) @ mo
    ft = fc + mo.conj().T @ (j-k) @ mo
    lagrangian = np.zeros((mo.shape[1],)*2, dtype=complex)
    lagrangian[:, :nc] = ft[:, :nc]
    lagrangian[:, nc:no] = (fc[:, nc:no] @ dm1.T
        + lib.einsum('puvw,tuvw->pt', hcas.paaa, dm2))
    body = lagrangian-lagrangian.conj().T
    return project(mc.pack_uniq_var(chart.pullback(body)))


class _CIAdapter:
    """Narrow audited protocol; no implicit ndarray operations on native MPS."""
    def __init__(self, mc):
        from socutils.dmrg import DMRGCI
        from socutils.fci import zfci
        self.mc, self.solver = mc, mc.fcisolver
        self.is_dmrg = isinstance(self.solver, DMRGCI)
        self.is_exact = isinstance(self.solver, zfci.FCISolver)
        if not self.is_dmrg and not self.is_exact:
            raise NotImplementedError('micro() supports socutils DMRGCI or exact zfci.FCISolver')
        self.ci = None
        self.session = None

    def __enter__(self):
        if self.is_dmrg:
            from socutils.dmrg.micro import DMRGOrbitalSession
            self.session = DMRGOrbitalSession(self.solver)
            self.session.__enter__()
        return self

    def __exit__(self, *exc):
        if self.session is not None:
            self.session.__exit__(*exc)

    def set_ci(self, ci):
        if not self.is_dmrg:
            self.ci = deepcopy(ci)
        self.mc.ci = ci

    def prepare_strict(self, initial=None):
        if self.is_dmrg:
            # A tentative MPS is a guess, not an accepted wavefunction. Strict
            # kernel validates it at its own tolerance (and can cold-retry).
            if self.solver.driver is not None and not self.solver.resume:
                self.solver.restart = self.solver._restart = True
            return None
        return initial if self.ci is None else self.ci

    def snapshot(self):
        if self.is_dmrg:
            return self.session.snapshot()
        return dict(ci=deepcopy(self.ci), converged=self.solver.converged)

    def restore(self, snapshot):
        self.mc.ci = None
        if self.is_dmrg:
            self.session.restore(snapshot, max_memory=self.mc.max_memory, verbose=self.mc.verbose)
            self.mc.ci = self.solver.ci
        else:
            self.ci = deepcopy(snapshot['ci'])
            self.solver.converged = snapshot['converged']
            self.mc.ci = self.ci

    def discard(self, snapshot):
        if self.is_dmrg:
            snapshot.close()

    def publish(self):
        if self.is_dmrg:
            self.session.publish()


def update_casdm(mc, hcas, adapter, stats, *, verbose=None):
    """Finite-sweep DMRG response, or explicitly labelled small exact-CI control."""
    solver = mc.fcisolver
    context = getattr(solver, 'set_orbital_context', None)
    if context is not None:
        context(hcas.mo[:, mc.ncore:mc.ncore+mc.ncas], mc._scf.get_ovlp(), mol=mc.mol)
    start = time.perf_counter()
    stats['micro_ci_calls'] += 1
    mc.ci = None
    try:
        if adapter.is_dmrg:
            energy, ci = solver.approx_kernel(hcas.h1, hcas.h2, mc.ncas, mc.nelecas,
                ecore=hcas.ecore, sweeps=_option(mc, 'micro_ci_sweeps'),
                davidson_threshold=_option(mc, 'micro_ci_davidson_threshold'),
                max_memory=mc.max_memory, verbose=verbose)
            info = dict(solver.convergence_info)
            stats['micro_ci_sweeps'] += info['sweeps']
        else:
            energy, ci = solver.kernel(hcas.h1, hcas.h2, mc.ncas, mc.nelecas,
                ci0=adapter.ci, ecore=hcas.ecore, max_memory=mc.max_memory, verbose=verbose)
            if not bool(np.all(solver.converged)):
                raise MicroNumericalError('Small exact-CI control failed')
            info = dict(approximate=False, mode='exact-small-CI-control', sweeps=0)
    finally:
        stats['micro_ci_seconds'] += time.perf_counter()-start
    energy = _real(energy, 'micro energy')
    dm1, dm2 = _rdms(mc, ci, stats)
    adapter.set_ci(ci)
    return energy, dm1, dm2, info


def _ball_scale(current, direction, radius):
    """Largest alpha in [0,1] with ||current+alpha*direction|| <= radius."""
    if np.linalg.norm(current+direction) <= radius:
        return 1.
    aa = np.vdot(direction, direction).real
    bb = 2*np.vdot(current, direction).real
    cc = np.vdot(current, current).real-radius**2
    if aa == 0:
        return 0.
    return float(np.clip((-bb+np.sqrt(max(0., bb*bb-4*aa*cc)))/(2*aa), 0., 1.))


def rotate_orb_cc(mc, gradient, hdiag, hop, keyframe, *, radius, stats,
                  davidson_tol=1e-8, davidson_maxiter=40, verbose=None):
    """CIAH co-iteration with a frozen operator in one fixed exponential chart.

    keyframe(chart, model_gradient) must update CI/RDM and return a fresh chart
    gradient and diagnostics. The callback never changes the Hessian closure.
    """
    log = logger.new_logger(mc, verbose)
    project = getattr(hop, 'project', lambda v: v)
    g0 = project(np.asarray(gradient, complex))
    model = realify(g0)
    accumulated = np.zeros_like(g0)
    haccumulated = np.zeros_like(g0)
    records = []
    nsteps, since_kf, nhop = 0, 0, 0
    last_keyframe = 0
    kfnorm = np.linalg.norm(g0)
    gtol = mc.conv_tol_grad
    cap = radius/np.sqrt(2.)  # ||kappa||_F = sqrt(2) ||packed||_2
    identity = RotationChart.from_generator(mc.unpack_uniq_var(accumulated))
    if not g0.size or np.linalg.norm(g0) < .2*gtol:
        return identity, accumulated, 0., records

    def hreal(vector):
        nonlocal nhop
        nhop += 1
        stats['ah_hops'] += 1
        return realify(project(hop(project(unrealify(vector)))))

    def precondition(vector, eigenvalue):
        source = project(unrealify(vector))
        shift = eigenvalue-_option(mc, 'micro_ah_level_shift')
        if hasattr(hop, 'precondition'):
            output = hop.precondition(source, shift, 1e-10)
        else:
            denom = np.asarray(hdiag)-shift
            denom = np.where(abs(denom) > 1e-10, denom, 1e-10)
            output = source/denom
        output = realify(project(output))
        size = np.linalg.norm(output)
        if not np.isfinite(size) or size < 1e-15:
            # This is a legitimate identity preconditioner, not a zero direction.
            output = realify(source)
            size = np.linalg.norm(output)
        if size < 1e-15:
            raise MicroNumericalError('CIAH preconditioner has no independent direction')
        return output/size

    initial = precondition(model, 0.)
    limit = min(int(davidson_maxiter), model.size)
    generator = ciah.davidson_cc(hreal, lambda: model, precondition, initial,
        tol=davidson_tol, max_cycle=limit,
        lindep=_option(mc, 'micro_ah_lindep'), verbose=log)
    chart = identity
    try:
        for ah_end, iterations, eigenvalue, dx_real, hdx_real, residual, seig in generator:
            resnorm = float(np.linalg.norm(residual))
            usable = (ah_end or iterations >= limit or
                      (iterations >= _option(mc, 'micro_ah_start_cycle') and
                       resnorm <= _option(mc, 'micro_ah_start_tol')))
            if not usable:
                continue
            dx = project(unrealify(dx_real))
            hdx = project(unrealify(hdx_real))
            if not np.all(np.isfinite(dx)) or not np.all(np.isfinite(hdx)):
                raise MicroNumericalError('CIAH returned a nonfinite trial')
            scale = min(1., _option(mc, 'micro_max_component')/max(1e-300, np.max(abs(dx))))
            scale *= _ball_scale(accumulated, scale*dx, cap)
            dx, hdx = scale*dx, scale*hdx
            if np.linalg.norm(dx) < 1e-14:
                break
            if 2*np.vdot(unrealify(model), dx).real + np.vdot(dx, hdx).real >= 0:
                # Do not pretend a non-descent AH proposal is safe. Outer logic
                # will restore the accepted MPS before trying a smaller radius.
                if nsteps == 0:
                    raise MicroNumericalError('CIAH model did not produce a descent direction')
                break
            accumulated += dx
            haccumulated += hdx
            model = model+realify(hdx)
            nsteps += 1
            since_kf += 1
            at_boundary = np.linalg.norm(accumulated) >= .999*cap
            small = np.linalg.norm(model) < .3*gtol
            update = (since_kf >= _option(mc, 'micro_kf_interval') or
                      small or at_boundary or iterations >= limit or ah_end)
            if not update:
                continue
            chart = RotationChart.from_generator(mc.unpack_uniq_var(accumulated))
            fresh, record = keyframe(chart, unrealify(model))
            fresh = project(np.asarray(fresh, complex))
            if not np.all(np.isfinite(fresh)):
                raise MicroNumericalError('Nonfinite keyframe gradient')
            discrepancy = float(np.linalg.norm(fresh-unrealify(model)))
            freshnorm = float(np.linalg.norm(fresh))
            record.update(ah_steps=nsteps, ah_hops=nhop, ah_residual=resnorm,
                ah_end=bool(ah_end), chart_gradient_norm=freshnorm,
                keyframe_discrepancy=discrepancy,
                step_norm=float(np.sqrt(2)*np.linalg.norm(accumulated)))
            records.append(record)
            stats['keyframes'] += 1
            last_keyframe = nsteps
            since_kf = 0
            log.info('  micro %d | g(chart)=%.3e | dg=%.3e | Hx=%d | CI sweeps=%s',
                     len(records), freshnorm, discrepancy, nhop, record.get('sweeps', 0))
            bad = (discrepancy > _option(mc, 'micro_kf_trust')*max(np.linalg.norm(model), gtol)
                   and freshnorm > kfnorm)
            model = realify(fresh)  # source changes; H and all cached Hx do not
            kfnorm = freshnorm
            if (bad or at_boundary or freshnorm < .3*gtol or
                    len(records) >= _option(mc, 'max_cycle_micro')):
                record['end_micro_reason'] = ('keyframe_discrepancy' if bad else
                    'radius' if at_boundary else 'gradient' if freshnorm < .3*gtol else 'micro_limit')
                break
    finally:
        generator.close()
    # If the Davidson generator ended between keyframes, do the final response
    # explicitly so the outgoing MPS and candidate orbitals have the same basis.
    if nsteps and last_keyframe != nsteps:
        chart = RotationChart.from_generator(mc.unpack_uniq_var(accumulated))
        fresh, record = keyframe(chart, unrealify(model))
        record.update(ah_steps=nsteps, ah_hops=nhop,
            chart_gradient_norm=float(np.linalg.norm(fresh)),
            keyframe_discrepancy=float(np.linalg.norm(fresh-unrealify(model))),
            end_micro_reason='ah_space_end')
        records.append(record)
        stats['keyframes'] += 1
    predicted = float(2*np.vdot(g0, accumulated).real + np.vdot(accumulated, haccumulated).real)
    return chart, accumulated, predicted, records


def _install_state(mc, state, adapter):
    mc.mo_coeff = np.array(state.mo, copy=True)
    mc.e_tot, mc.e_cas = state.energy, state.ecas
    mc.ci = adapter.solver.ci if adapter.is_dmrg else deepcopy(adapter.ci)


def _gradient(mc, state, kramers):
    q = zmc_utils.build_orbital_quantities(mc, state.mo, state.dm1, state.dm2, state.eris)
    g = mc.pack_uniq_var(q.gradient.copy())
    if kramers:
        matrix, _ = zmc_utils._project_kramers_rotation(
            mc, state.mo, mc.unpack_uniq_var(g), force=True)
        g = mc.pack_uniq_var(matrix)
    return g


def validate(mc, mo, *, symm=None):
    if int(getattr(mc.fcisolver, 'nroots', 1)) != 1 or hasattr(mc.fcisolver, 'weights'):
        raise NotImplementedError('micro() currently requires genuine single-root state-specific CASSCF; no SA wrapper')
    if mc.natorb or mc.canonicalize_:
        raise ValueError('micro() requires natorb=False and canonicalize_=False during iterations')
    if not np.isfinite(mc.max_stepsize) or mc.max_stepsize <= 0:
        raise ValueError('max_stepsize must be a positive generator Frobenius radius')
    for name in ('max_cycle_micro', 'micro_ci_sweeps', 'micro_kf_interval',
                 'micro_ah_start_cycle', 'micro_max_trials'):
        value = _option(mc, name)
        if int(value) != value or value <= 0:
            raise ValueError('%s must be a positive integer' % name)
    if _option(mc, 'micro_ci_sweeps') < 2:
        raise ValueError('At least two micro CI sweeps are required')
    for name in ('micro_max_component', 'micro_ah_start_tol', 'micro_ah_lindep',
                 'micro_kf_trust', 'micro_ci_davidson_threshold', 'micro_rdm_tol'):
        if not np.isfinite(_option(mc, name)) or _option(mc, name) <= 0:
            raise ValueError('%s must be positive and finite' % name)
    if _option(mc, 'micro_integral_mode') not in ('dep1', 'exact'):
        raise ValueError("micro_integral_mode must be 'dep1' or 'exact'")
    if not np.isfinite(_option(mc, 'micro_ah_level_shift')):
        raise ValueError('micro_ah_level_shift must be finite')
    if int(mc.max_cycle_macro) != mc.max_cycle_macro or mc.max_cycle_macro < 0:
        raise ValueError('max_cycle_macro must be a nonnegative integer')
    if mo.ndim != 2 or not np.all(np.isfinite(mo)):
        raise ValueError('Invalid initial orbitals')
    kramers = zmc_utils._resolve_kramers_mode(mc, symm)
    if kramers:
        zmc_utils._identify_kramers_mapping(mc, mo)
    return kramers


def kernel(mc, mo_coeff, *, max_stepsize=.2, conv_tol=None, conv_tol_grad=None,
           verbose=5, cderi=None, bfgs=False, solver='davidson',
           davidson_maxiter=40, davidson_tol=1e-8, davidson_strict=True,
           symm=None, callback=None, ci0=None):
    """Return (converged, E, Ecas, CI, MO, MO_energy), like peer optimizers.

    `davidson_strict` is accepted for dispatcher compatibility; a micro AH need
    not meet the final residual. Only complete macro CI and the actual orbital
    gradient determine final convergence. Fatal operator/programming errors are
    never swallowed as a successful step. Repeated numerical trial failures
    restore the last accepted MPS and raise MicroNumericalError.
    """
    if solver != 'davidson' or bfgs:
        raise ValueError('micro() uses realified CIAH; GMRES/BFGS is not supported')
    mo = np.array(mo_coeff, dtype=complex, copy=True)
    mc.mo_coeff = mo.copy()
    kramers = validate(mc, mo, symm=symm)
    conv_tol = mc.conv_tol if conv_tol is None else conv_tol
    conv_tol_grad = mc.conv_tol_grad if conv_tol_grad is None else conv_tol_grad
    if conv_tol_grad is None:
        conv_tol_grad = np.sqrt(conv_tol)
    if min(conv_tol, conv_tol_grad) <= 0 or not np.isfinite(conv_tol+conv_tol_grad):
        raise ValueError('Convergence tolerances must be positive and finite')
    mc.conv_tol_grad = conv_tol_grad
    radius = float(max_stepsize)
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError('max_stepsize must be positive and finite')
    if int(davidson_maxiter) != davidson_maxiter or davidson_maxiter < 1:
        raise ValueError('davidson_maxiter must be a positive integer')
    if not np.isfinite(davidson_tol) or davidson_tol <= 0:
        raise ValueError('davidson_tol must be positive and finite')
    max_radius = radius
    log = logger.new_logger(mc, verbose)
    stats = _new_stats()
    mc.macro_history, mc.orbital_trial_history = [], []
    mc.converged = False
    mc.canonicalization_diagnostics = dict(enabled=False, reason='not yet finalized')
    state, snapshot = None, None
    previous_change, previous_step = np.inf, 0.
    converged = False
    gradient_norm = np.inf
    wall_start = time.perf_counter()
    log.info('Orbital optimizer = complex-spinor one-step micro / CIAH')
    log.info('Micro integrals = %s; keyframes/macro <= %d; CI sweeps = %d',
             _option(mc, 'micro_integral_mode'), _option(mc, 'max_cycle_micro'),
             _option(mc, 'micro_ci_sweeps'))
    with _count_jk(mc, stats), _CIAdapter(mc) as adapter:
        try:
            state = _full_reference(mc, mo, adapter, cderi, stats, verbose, ci0=ci0)
            _install_state(mc, state, adapter)
            for macro in range(mc.max_cycle_macro):
                start = time.perf_counter()
                baseline = dict(stats)
                g, hd, hop, _, _, _ = zmc_ah.gen_g_hop(mc, state.mo,
                    state.dm1, state.dm2, state.eris, kramers=kramers)
                gradient_norm = float(np.linalg.norm(g))
                row = dict(macro_iteration=macro, total_energy=state.energy,
                    cas_energy=state.ecas, energy_change=float(previous_change),
                    orbital_gradient_norm=gradient_norm, orbital_step_norm=previous_step,
                    orbital_method='micro', accepted=True, converged=False,
                    ci_solver_converged=True, integral_representation=state.provenance['representation'],
                    integral_provenance=dict(state.provenance), micro_integral_mode=_option(mc, 'micro_integral_mode'))
                mc.macro_history.append(row)
                log.info('MCSCF macro = %4d | E = %22.15f | dE = %.3e | Grad norm = %.3e',
                         macro, state.energy, previous_change, gradient_norm)
                if abs(previous_change) < conv_tol and gradient_norm < conv_tol_grad:
                    converged = row['converged'] = True
                    if callback is not None:
                        callback(dict(row))
                    break
                # Already stationary at the initial point: one strict repeat
                # establishes an actual energy change, not a fabricated zero.
                if gradient_norm < .2*conv_tol_grad:
                    snapshot = adapter.snapshot()
                    candidate = _full_reference(mc, state.mo, adapter, cderi, stats, verbose)
                    previous_change = candidate.energy-state.energy
                    _close_eris(state.eris)
                    state = candidate
                    _install_state(mc, state, adapter)
                    previous_step = 0.
                    adapter.discard(snapshot)
                    snapshot = None
                    continue
                snapshot = adapter.snapshot()
                trials = []
                accepted = False
                for attempt in range(_option(mc, 'micro_max_trials')):
                    if attempt:
                        adapter.restore(snapshot)
                        _install_state(mc, state, adapter)
                    candidate = None
                    dm_previous = state.dm1.copy()

                    def keyframe(chart, model_gradient):
                        nonlocal dm_previous
                        hcas = update_active_hamiltonian(mc, state, chart.rotation,
                            mode=_option(mc, 'micro_integral_mode'), cderi=cderi, stats=stats)
                        try:
                            energy, one, two, ci_info = update_casdm(mc, hcas, adapter, stats, verbose=verbose)
                            fresh = gradient_at_keyframe(mc, hcas, one, two, chart,
                                                         getattr(hop, 'project', lambda x: x))
                            record = dict(energy_model=energy,
                                rdm1_change=float(np.linalg.norm(one-dm_previous)),
                                sweeps=ci_info.get('sweeps', 0),
                                ci_approximate=ci_info.get('approximate', False),
                                ci_energy_change=ci_info.get('energy_change'),
                                local_squared_residual_threshold=ci_info.get('local_squared_residual_threshold'))
                            dm_previous = one
                            return fresh, record
                        finally:
                            _close_eris(hcas.eris)

                    record = dict(attempt=attempt, radius=radius, accepted=False)
                    trials.append(record)
                    try:
                        chart, dx, predicted, micro_rows = rotate_orb_cc(mc, g, hd, hop, keyframe,
                            radius=radius, stats=stats, davidson_tol=davidson_tol,
                            davidson_maxiter=davidson_maxiter, verbose=verbose)
                        record.update(microiterations=micro_rows, predicted_energy_change=predicted,
                                      step_norm=float(np.linalg.norm(chart.generator)))
                        candidate = _full_reference(mc, state.mo @ chart.rotation, adapter,
                                                     cderi, stats, verbose)
                        change = candidate.energy-state.energy
                        ratio = change/predicted if predicted < -1e-16 else None
                        slack = conv_tol if gradient_norm < conv_tol_grad else 0.
                        accepted = (change <= slack and
                            (abs(change) < conv_tol or (ratio is not None and ratio > .1)))
                        record.update(energy=candidate.energy, energy_change=change,
                                      ratio=ratio, accepted=bool(accepted))
                    except (MicroNumericalError, zmc_ah.AHNumericalError, np.linalg.LinAlgError) as error:
                        record['error'] = str(error)
                        accepted = False
                    if accepted:
                        _close_eris(state.eris)
                        state = candidate
                        _install_state(mc, state, adapter)
                        previous_change, previous_step = change, record['step_norm']
                        if ratio is not None and ratio < .25:
                            radius *= .5
                        elif ratio is not None and ratio > .75 and previous_step > .8*radius:
                            radius = min(max_radius, 1.5*radius)
                        break
                    if candidate is not None:
                        _close_eris(candidate.eris)
                    log.info('Micro macro trial rejected: radius=%.4g, error=%s; restore accepted state',
                             radius, record.get('error', record.get('energy_change')))
                    radius *= .5
                mc.orbital_trial_history = trials
                if not accepted:
                    raise MicroNumericalError('All micro macro trials failed; accepted reference restored')
                adapter.discard(snapshot)
                snapshot = None
                row.update(orbital_trials=trials, next_total_energy=state.energy,
                    accepted_energy_change=previous_change, applied_orbital_step_norm=previous_step,
                    predicted_energy_change=predicted, trust_ratio=ratio, trust_radius=radius,
                    microiterations=micro_rows, rejected_trials=len(trials)-1,
                    macro_wall_time=time.perf_counter()-start,
                    work={k: stats[k]-baseline[k] for k in stats if k != 'max_model_cache_mb'})
                if callback is not None:
                    callback(dict(row))
                if mc.chkfile:
                    lib.chkfile.save(mc.chkfile, 'mo_coeff_iter_%d' % (macro+1), state.mo)
                log.info('MCSCF micro update | dE=%.3e | step=%.3e | keyframes=%d | time=%.2fs',
                         previous_change, previous_step, len(micro_rows), row['macro_wall_time'])
                # H and its ERI buffers have changed: no Krylov data cross this boundary.
                hop = None

            # Always refresh on the RETURNED state, including max_cycle_macro exhaustion.
            gradient_norm = float(np.linalg.norm(_gradient(mc, state, kramers)))
            converged = (abs(previous_change) < conv_tol and gradient_norm < conv_tol_grad
                         and bool(np.all(mc.fcisolver.converged)))
            mo_energy = None
            if converged and mc.canonicalization:
                snapshot = adapter.snapshot()
                final_mo, _, mo_energy = mc.canonicalize(state.mo, mc.ci, eris=state.eris,
                    sort=mc.sorting_mo_energy, cas_natorb=False, casdm1=state.dm1, verbose=verbose)
                # Bind the final live MPS/checkpoint to exactly these orbitals,
                # not an approximately equal pre-canonical Hamiltonian.
                final = _full_reference(mc, final_mo, adapter, cderi, stats, verbose)
                canonical_change = final.energy-state.energy
                try:
                    final_gradient = float(np.linalg.norm(_gradient(mc, final, kramers)))
                except Exception:
                    _close_eris(final.eris)
                    raise
                _close_eris(state.eris)
                state = final
                _install_state(mc, state, adapter)
                gradient_norm = final_gradient
                converged = (abs(canonical_change) < conv_tol and
                             gradient_norm < conv_tol_grad and bool(np.all(mc.fcisolver.converged)))
                adapter.discard(snapshot)
                snapshot = None
            else:
                mc.canonicalization_diagnostics = dict(enabled=False,
                    reason='disabled' if not mc.canonicalization else 'reference_not_converged')
            _install_state(mc, state, adapter)
            mc.mo_energy, mc.converged = mo_energy, bool(converged)
            if converged:
                adapter.publish()
            log.info('MCSCF micro final | converged=%s | E=%.15f | true gradient=%.3e',
                     converged, state.energy, gradient_norm)
            return bool(converged), state.energy, state.ecas, mc.ci, mc.mo_coeff, mo_energy
        except Exception:
            mc.converged = False
            if snapshot is not None:
                adapter.restore(snapshot)
                _install_state(mc, state, adapter)
            if state is not None:
                gradient_norm = float(np.linalg.norm(_gradient(mc, state, kramers)))
            raise
        finally:
            if snapshot is not None:
                adapter.discard(snapshot)
            mc.final_orbital_gradient_norm = gradient_norm
            mc.micro_diagnostics = dict(orbital_method='micro', converged=bool(mc.converged),
                final_gradient_norm=gradient_norm, energy_tolerance=conv_tol,
                gradient_tolerance=conv_tol_grad, integral_mode=_option(mc, 'micro_integral_mode'),
                kramers_restricted=bool(kramers), single_root=True, work=stats,
                wall_seconds=time.perf_counter()-wall_start,
                macro_iterations=len(mc.macro_history))
            mc.superci_diagnostics = mc.micro_diagnostics
            if state is not None:
                mc.cholesky_diagnostics = dict(state.provenance)
                _close_eris(state.eris)
            gc.collect()
