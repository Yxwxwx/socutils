# Author: Xubo Wang <xubo.wang@outlook.com>
# Date: 2023/12/4
import sys
import numpy
import numpy as np
from pyscf import lib, scf, gto, mcscf
from pyscf.lib import logger
from functools import reduce
from scipy.sparse.linalg import gmres

# from .hf_superci import GMRES
from socutils.mcscf import zcahf, zcasci, zmcscf, zmc_ao2mo
from scipy.sparse.linalg import LinearOperator
import scipy
from numpy.linalg import norm
from socutils.mcscf.zmc_utils import (
    _physical_active_density,
    _contract_dm2_gradient,
    _build_eris,
    _resolve_kramers_mode,
    _identify_kramers_mapping,
    _project_kramers_rotation,
    _ci_convergence_snapshot,
    _schedule_orbital_trial,
    _bounded_orbital_update,
    _kramers_subspace_eigh,
)
from socutils.mcscf.hf_superci import precondition_grad, postprocess_x


def expmat(x):
    expm = np.eye(x.shape[0], dtype=complex)
    xx = np.eye(x.shape[0], dtype=complex)
    for i in range(10):
        print("xx", i, np.linalg.norm(xx))
        xx = np.dot(xx, x) / (i + 1)
        expm += xx
    return expm


from scipy.linalg import expm as expmat


def _canonical_generalized_eigh(projected_h, projected_s, lindep):
    """Solve a Hermitian generalized problem on the reliable metric range.

    Super-CI has exact zero-metric orbital directions when an active natural
    occupation is zero or one.  Finite-accuracy RDMs can turn those zeros into
    tiny eigenvalues of either sign, for which a Cholesky-based generalized
    eigensolver is neither defined nor numerically meaningful.  Canonical
    orthogonalization removes only that unresolved null space and retains a
    strict guard against a materially indefinite metric.
    """
    metric_eigenvalues, metric_eigenvectors = scipy.linalg.eigh(projected_s)
    metric_scale = max(1.0, float(np.max(np.abs(metric_eigenvalues), initial=0.0)))
    minimum_metric_eigenvalue = float(metric_eigenvalues[0])
    negative_noise = max(0.0, -minimum_metric_eigenvalue)
    negative_tolerance = np.sqrt(lindep) * metric_scale
    if negative_noise > negative_tolerance:
        raise RuntimeError(
            "Super-CI projected metric is materially indefinite: "
            "minimum eigenvalue %.6e, tolerance %.6e"
            % (minimum_metric_eigenvalue, negative_tolerance)
        )

    metric_cutoff = max(lindep * metric_scale, 10.0 * negative_noise)
    keep = metric_eigenvalues > metric_cutoff
    if not np.any(keep):
        raise RuntimeError(
            "Super-CI projected metric has no numerically independent directions"
        )
    orthogonalizer = (
        metric_eigenvectors[:, keep] / np.sqrt(metric_eigenvalues[keep])[None, :]
    )
    orthogonal_h = reduce(
        np.dot,
        (orthogonalizer.T.conj(), projected_h, orthogonalizer),
    )
    orthogonal_h = (orthogonal_h + orthogonal_h.T.conj()) * 0.5
    eigenvalues, orthogonal_eigenvectors = scipy.linalg.eigh(orthogonal_h)
    eigenvectors = orthogonalizer.dot(orthogonal_eigenvectors)
    diagnostics = {
        "metric_rank": int(np.count_nonzero(keep)),
        "metric_dimension": int(projected_s.shape[0]),
        "metric_min_eigenvalue": minimum_metric_eigenvalue,
        "metric_cutoff": float(metric_cutoff),
        "metric_discarded_directions": int(np.count_nonzero(~keep)),
    }
    return eigenvalues, eigenvectors, diagnostics




def _apply_superci_metric(generator, active_density, ncore, nocc):
    """Apply the covariant Super-CI overlap metric to a MO generator."""
    generator = np.asarray(generator, dtype=complex)
    active_density = np.asarray(active_density, dtype=complex)
    ncas = nocc - ncore
    metric_hole = np.eye(ncas, dtype=complex) - active_density
    result = np.array(generator, copy=True)
    virtual_active = generator[nocc:, ncore:nocc].dot(active_density)
    active_core = metric_hole.dot(generator[ncore:nocc, :ncore])
    result[nocc:, ncore:nocc] = virtual_active
    result[ncore:nocc, nocc:] = -virtual_active.T.conj()
    result[ncore:nocc, :ncore] = active_core
    result[:ncore, ncore:nocc] = -active_core.T.conj()
    return result


def form_kramers(mo_coeff):
    nao = mo_coeff.shape[0] // 2
    # a, b for alpha and beta atomic orbitals
    # u, b for unbarred and barred spinors
    mo_au = mo_coeff[::2, ::2]  # U
    mo_bu = mo_coeff[1::2, ::2]  # -V^*
    mo_coeff[::2, 1::2] = mo_bu.conj()  # V
    mo_coeff[1::2, 1::2] = -mo_au.conj()  # U^*
    return mo_coeff


def ensure_kramers(mat):
    a = mat[::2, ::2].copy()
    at = mat[1::2, 1::2].copy()
    b = mat[::2, 1::2].copy()
    bt = mat[1::2, ::2].copy()
    out = numpy.zeros_like(mat)
    aa = (a + at.conj()) * 0.5
    out[::2, ::2] = aa
    out[1::2, 1::2] = aa.conj()
    bb = (b - bt.conj()) * 0.5
    print(norm(a), norm(at), norm(b), norm(bt))
    print(norm(a - at.T), norm(b + bt.T))
    out[::2, 1::2] = bb
    out[1::2, ::2] = -bb.conj()
    print("ensure kramers", numpy.linalg.norm(mat - out), numpy.linalg.norm(mat))
    return out


def compute_lambda_(mat1, mat2, x_):
    nlast = mat1.shape[0]
    assert nlast > 1
    lambda_test = 1.0
    lambda_lasttest = 0.0
    stepsize_lasttest = 0.0
    stepsize = 0.0
    maxstepsize = 1.0
    iok = 0
    for i in range(10):
        scr = mat1 + mat2 * (1.0 / lambda_test)
        e, c = np.linalg.eigh(scr)

        ivec = -1
        for j in range(nlast):
            if abs(c[0, j]) <= 1.1 and abs(c[0, j]) > 0.1:
                ivec = j
                break
        if ivec < 0:
            raise Exception("logical error in AugHess")
        c[:, ivec] = c[:, ivec] / (c[0, ivec])
        step = np.dot(x_[1:, :nlast], c[:nlast, ivec])
        stepsize = np.linalg.norm(step[1:]) / abs(lambda_test)
        # print(ivec, e, stepsize, lambda_test)

        if i == 0:
            if stepsize <= maxstepsize:
                break
            lambda_lasttest = lambda_test
            lambda_test = stepsize / maxstepsize
        else:
            if abs(stepsize - maxstepsize) / maxstepsize < 0.01:
                break
            if stepsize > maxstepsize:
                lambda_lasttest = lambda_test
                lambda_test = lambda_test * (stepsize / maxstepsize)
            else:
                if iok > 2:
                    break
                iok += 1
                d1 = maxstepsize - stepsize
                d2 = stepsize_lasttest - maxstepsize
                if d1 == 0.0 or d1 == -d2:
                    break
                lambda_lasttest_ = lambda_lasttest
                lambda_lasttest = lambda_test
                lambda_test = (
                    d1 / (d1 + d2) * lambda_lasttest_ + d2 / (d1 + d2) * lambda_test
                )
            if lambda_test < 1.0:
                lambda_test = 1.0
            stepsize_lasttest = stepsize
    return lambda_test, stepsize


def davidson(
    hop,
    g,
    hdiag,
    sop=None,
    max_stepsize=1.0,
    tol=5e-6,
    neig=1,
    mmax=10,
    lindep=1e-12,
    log=None,
):
    """Solve the generalized augmented-Hessian Super-CI equation.

    This Davidson solve is the orbital-equation solve.  Its tolerance and
    residual are deliberately independent of Block2's local eigensolver.
    The returned residual is that of the full augmented equation after the
    eigenvector has been normalized to unit reference component.
    """
    if sop is None:
        raise ValueError("The Super-CI overlap operator is required")
    if neig != 1:
        raise NotImplementedError("Only one augmented-Hessian root is supported")
    if mmax < 1:
        raise ValueError("mmax must be at least one")

    g = np.asarray(g, dtype=complex)
    hdiag = np.asarray(hdiag)
    if g.ndim != 1 or hdiag.shape != g.shape:
        raise ValueError("g and hdiag must be one-dimensional arrays of equal size")

    if hasattr(hop, '_superci_frame'):
        # Conditioning belongs to the unshifted solver too.  In particular,
        # do not require adaptive orbital regularization to resolve tiny
        # particle/hole occupations in the Super-CI overlap.
        expand, restrict, diagonal, inverse_metric, _ = hop._superci_frame
        y, energy, info = davidson(
            lambda v: restrict(hop(expand(v))), restrict(g), diagonal,
            sop=lambda v: v, max_stepsize=max_stepsize, tol=tol * .2,
            neig=neig, mmax=mmax, lindep=lindep, log=log,
        )
        step = expand(y)
        residual = hop(step) + g - energy * sop(step)
        reference = abs(np.vdot(g, step) - energy)
        raw_norm = float(np.hypot(np.linalg.norm(residual), reference))
        white_norm = float(np.hypot(np.linalg.norm(restrict(residual)), reference))
        residual_norm = max(raw_norm, white_norm)
        info.update(
            converged=residual_norm <= tol, residual_norm=residual_norm,
            raw_full_residual=raw_norm, retained_residual=white_norm,
            orbital_shift=0., metric_condition_bound=float(np.max(inverse_metric, initial=1.)),
            step_norm=float(np.linalg.norm(step)),
            retained_dimension=len(diagonal), full_dimension=len(g),
            total_davidson_iterations=info['iterations'],
        )
        if info['converged']:
            info['reason'] = 'converged_unshifted_metric_frame'
        elif info['reason'] in ('converged', 'zero_gradient'):
            info['reason'] = 'full_metric_residual'
        if log is not None:
            log.info('Super-CI unshifted metric solve: retained_res=%.3e full_res=%.3e step=%.6g',
                     white_norm, raw_norm, info['step_norm'])
        return step, energy, info

    gnorm = np.linalg.norm(g)
    if gnorm <= tol:
        info = {
            "converged": True,
            "iterations": 0,
            "residual_norm": float(gnorm),
            "residual_history": [float(gnorm)],
            "step_norm": 0.0,
            "max_stepsize": float(max_stepsize),
            "reason": "zero_gradient",
        }
        return np.zeros_like(g), 0.0, info

    n = g.size + 1
    x = np.zeros((n, mmax + 1), dtype=complex)
    sigma = np.zeros_like(x)
    sdotx = np.zeros_like(x)
    x[0, 0] = 1.0
    sigma[1:, 0] = g
    sdotx[0, 0] = 1.0

    denom = hdiag + 1e-3
    denom = np.where(np.abs(denom) > 1e-8, denom, np.where(denom.real < 0, -1e-8, 1e-8))
    x[1:, 1] = -g / denom
    trial_norm = np.linalg.norm(x[:, 1])
    if trial_norm <= lindep:
        x[1:, 1] = -g / gnorm
    else:
        x[:, 1] /= trial_norm

    residual_history = []
    last_step = None
    last_eig = None
    last_info = None
    for m in range(1, mmax + 1):
        sigma[1:, m] = hop(x[1:, m])
        sigma[0, m] = np.vdot(g, x[1:, m])
        sdotx[1:, m] = sop(x[1:, m])

        xsub = x[:, : m + 1]
        sigsub = sigma[:, : m + 1]
        ssub = sdotx[:, : m + 1]
        projected_h = xsub.T.conj().dot(sigsub)
        projected_s = xsub.T.conj().dot(ssub)
        projected_h = (projected_h + projected_h.T.conj()) * 0.5
        projected_s = (projected_s + projected_s.T.conj()) * 0.5
        eigvals, eigvecs, metric_info = _canonical_generalized_eigh(
            projected_h, projected_s, lindep
        )

        root = next(
            (i for i in range(eigvecs.shape[1]) if 0.1 < abs(eigvecs[0, i]) <= 1.1),
            None,
        )
        if root is None:
            raise RuntimeError(
                "No root with a usable reference component "
                "was found in the augmented-Hessian subspace"
            )

        coeff = eigvecs[:, root]
        augvec = xsub.dot(coeff)
        reference = augvec[0]
        if abs(reference) <= lindep:
            raise RuntimeError(
                "Augmented-Hessian root has a vanishing reference component"
            )
        step = augvec[1:] / reference
        residual = (sigsub.dot(coeff) - eigvals[root] * ssub.dot(coeff)) / reference
        residual_norm = float(np.linalg.norm(residual))
        step_norm = float(np.linalg.norm(step))
        residual_history.append(residual_norm)
        converged = residual_norm <= tol
        last_step = step
        last_eig = float(eigvals[root].real)
        last_info = {
            "converged": converged,
            "iterations": m,
            "residual_norm": residual_norm,
            "residual_history": residual_history.copy(),
            "step_norm": step_norm,
            "max_stepsize": float(max_stepsize),
            "root": root,
            "reason": "converged" if converged else "maximum_space",
            **metric_info,
        }
        if log is not None:
            log.debug(
                "Super-CI Davidson iter %d root %d eigenvalue %.12g "
                "residual %.6g step %.6g",
                m,
                root,
                last_eig,
                residual_norm,
                step_norm,
            )
        if converged:
            return last_step, last_eig, last_info
        if m == mmax:
            return last_step, last_eig, last_info

        correction = np.zeros(n, dtype=complex)
        correction_denom = hdiag - eigvals[root]
        correction_denom = np.where(
            np.abs(correction_denom) > 1e-8,
            correction_denom,
            np.where(correction_denom.real < 0, -1e-8, 1e-8),
        )
        correction[1:] = -residual[1:] / correction_denom
        # Two passes make the complex Euclidean Gram-Schmidt robust enough for
        # the small Super-CI spaces used here.  np.vdot supplies the required
        # conjugation of the existing subspace vectors.
        for _ in range(2):
            for idx in range(m + 1):
                correction -= np.vdot(x[:, idx], correction) * x[:, idx]
        correction_norm = np.linalg.norm(correction)
        if correction_norm <= lindep:
            last_info["reason"] = "linear_dependence"
            return last_step, last_eig, last_info
        x[:, m + 1] = correction / correction_norm

    return last_step, last_eig, last_info






def _subspace_eigh(casscf, matrix, mo_subspace):
    """Diagonalize an MO-space matrix using reference symmetry when needed."""
    from socutils.scf import spinor_hf

    mf = casscf._scf
    if isinstance(mf, spinor_hf.KRHF):
        return mf.eig(matrix)
    if isinstance(mf, spinor_hf.SymmSpinorSCF):
        return mf.eig(matrix, mo=mo_subspace)
    return scipy.linalg.eigh(matrix)




def _active_natural_orbitals(casscf, casdm1, mo_active):
    """Diagonalize the active 1-RDM using the reference's symmetry when needed."""
    return _subspace_eigh(casscf, -casdm1, mo_active)










# note: the ncas, nelecas, ncore should all be counted as the number of spin orbitals
def _full_eri_operators(mc, mo, raw_dm1, dm2, fock, lag):
    """Hermitian mixed-one-body Super-CI projection, not an orbital Hessian.

    The natural-occupation frame conditions the linear algebra without
    rotating active orbitals or the DMRG MPS. Both step policies use it.
    """
    nc, na = mc.ncore, mc.ncas
    no, nm = nc + na, mo.shape[1]
    nv = nm - no
    density, diagnostics = _physical_active_density(raw_dm1.T)
    mc.superci_metric_diagnostics = diagnostics
    d = raw_dm1.T
    fc, fv = fock[:nc, :nc], fock[no:, no:]
    fa = lag[nc:no, nc:no]
    defect = np.linalg.norm(fa - fa.conj().T)
    # Define the active block of the Hermitian mixed one-body operator.
    # This is not symmetrization of a faulty full/projected Super-CI matrix.
    fa = (fa + fa.conj().T) * .5
    removal = np.einsum('tuvw,vw->tu', dm2, fa) - raw_dm1 * np.einsum('vw,vw->', raw_dm1, fa)
    addition = fa - removal.T - fa @ d - d @ fa
    lc = lag[no:, nc:no] @ (np.eye(na) - d)
    bc = d @ lag[nc:no, :nc]

    def action(x):
        out = np.zeros_like(x)
        cv, ca, va = x[no:, :nc], x[nc:no, :nc], x[no:, nc:no]
        out[no:, :nc] = fv @ cv - cv @ fc + lc @ ca - va @ bc
        out[nc:no, :nc] = (d - np.eye(na)) @ ca @ fc + addition @ ca + lc.conj().T @ cv
        out[no:, nc:no] = fv @ va @ d + va @ removal.T - cv @ bc.conj().T
        return out

    gradient = mc.pack_uniq_var(lag - lag.conj().T)
    size = gradient.size
    if size != nc * (na + nv) + nv * na:
        raise ValueError('Full-ERI Super-CI supports unscreened, unfrozen spinor rotations only')
    hop = LinearOperator((size, size), matvec=lambda x: mc.pack_uniq_var(action(mc.unpack_uniq_var(x))), dtype=complex)
    sop = LinearOperator((size, size), matvec=lambda x: mc.pack_uniq_var(
        _apply_superci_metric(mc.unpack_uniq_var(x), density, nc, no)), dtype=complex)
    diagonal = np.zeros((nm, nm))
    diagonal[no:, :nc] = (np.diag(fv)[:, None] - np.diag(fc)[None, :]).real
    diagonal[nc:no, :nc] = ((np.diag(d) - 1)[:, None] * np.diag(fc)[None, :] + np.diag(addition)[:, None]).real
    diagonal[no:, nc:no] = (np.diag(fv)[:, None] * np.diag(d)[None, :] + np.diag(removal)[None, :]).real
    hd = mc.pack_uniq_var(diagonal)

    p, q = scipy.linalg.eigh(density)
    cutoff = 1e-14
    holes = 1 - p
    wh = q[:, holes > cutoff] / np.sqrt(holes[holes > cutoff])
    wp = q[:, p > cutoff] / np.sqrt(p[p > cutoff])
    cv_size, ca_size = nv * nc, wh.shape[1] * nc

    def expand(z):
        out = np.zeros((nm, nm), complex)
        out[no:, :nc] = z[:cv_size].reshape(nv, nc)
        out[nc:no, :nc] = wh @ z[cv_size:cv_size+ca_size].reshape(wh.shape[1], nc)
        out[no:, nc:no] = z[cv_size+ca_size:].reshape(nv, wp.shape[1]) @ wp.conj().T
        return mc.pack_uniq_var(out)

    def restrict(x):
        mat = mc.unpack_uniq_var(x)
        return np.concatenate((mat[no:, :nc].ravel(),
            (wh.conj().T @ mat[nc:no, :nc]).ravel(), (mat[no:, nc:no] @ wp).ravel()))

    white_hd = np.concatenate((diagonal[no:, :nc].ravel(),
        (np.diag(wh.conj().T @ (d-np.eye(na)) @ wh)[:, None] * np.diag(fc)[None, :]
         + np.diag(wh.conj().T @ addition @ wh)[:, None]).ravel(),
        (np.diag(fv)[:, None] * np.diag(wp.conj().T @ d @ wp)[None, :]
         + np.diag(wp.conj().T @ removal.T @ wp)[None, :]).ravel())).real
    # B^H B is diagonal in this frame; an orbital shift is NOT mu * S.
    shift_diag = np.concatenate((np.ones(cv_size),
        np.repeat(1 / holes[holes > cutoff], nc), np.tile(1 / p[p > cutoff], nv)))
    hop._superci_frame = (expand, restrict, white_hd, shift_diag, mc.max_stepsize / np.sqrt(2))
    logger.info(mc, 'Super-CI metric frame: full=%d retained=%d cutoff=%.1e active-L antihermiticity=%.3e',
                size, white_hd.size, cutoff, defect)
    precond = LinearOperator((size, size), matvec=lambda x: x / np.maximum(abs(hd), 1e-8), dtype=complex)
    return gradient, hd, hop, sop, precond, mo


def _gen_g_hop_full(mc, mo, raw_dm1, dm2, eris):
    if not isinstance(eris, zmc_ao2mo._ERIS) or _resolve_kramers_mode(mc):
        raise ValueError('Full-ERI Super-CI requires full ERI and no Kramers restriction')
    if mc.canonicalize_:
        raise ValueError('Full-ERI Super-CI requires canonicalize_=False')
    nc, no = mc.ncore, mc.ncore + mc.ncas
    core = mo[:, :nc]
    occ = np.zeros(mo.shape[1]); occ[:nc] = 1
    jc, kc = eris.get_jk(core @ core.conj().T, mo_coeff=mo, mo_occ=occ)
    eris.vj_c, eris.vk_c = jc, kc
    fc = mo.conj().T @ (mc.get_hcore() + jc - kc) @ mo
    ja, ka = eris.get_jk_active_mo(raw_dm1.T)
    fock = fc + ja - ka
    lag = np.zeros_like(fock)
    lag[:, :nc] = fock[:, :nc]
    lag[:, nc:no] = fc[:, nc:no] @ raw_dm1.T + _contract_dm2_gradient(eris, dm2)
    return _full_eri_operators(mc, mo, raw_dm1, dm2, fock, lag)

def gen_g_hop(casscf, mo, casdm1, casdm2, eris):
    if casscf.mo_coeff is None:
        casscf.mo_coeff = mo
    ncas = casscf.ncas
    nelecas = casscf.nelecas
    ncore = casscf.ncore
    nocc = ncas + ncore
    nmo = mo.shape[1]

    # The corrected complex operator and full occupation metric are shared
    # by both step policies on the unrestricted full-ERI route.
    mask = casscf.uniq_var_indices(nmo, ncore, ncas, casscf.frozen)
    if (isinstance(eris, zmc_ao2mo._ERIS) and not casscf.canonicalize_
            and not _resolve_kramers_mode(casscf)
            and np.count_nonzero(mask) == ncore * (nmo - ncore) + (nmo - nocc) * ncas):
        return _gen_g_hop_full(casscf, mo, casdm1, casdm2, eris)

    # casdm1 = np.diag(np.diag(casdm1))

    ################# gradient #################
    dm_core = np.zeros((nmo, nmo), dtype=complex)
    dm_active = np.zeros((nmo, nmo), dtype=complex)
    idx = np.arange(ncore)
    dm_core[idx, idx] = 1
    dm_active[ncore:nocc, ncore:nocc] = casdm1
    dm1 = dm_core + dm_active
    h1e_mo = reduce(np.dot, (mo.T.conj(), casscf.get_hcore(), mo))
    core_occ = np.zeros(nmo)
    core_occ[:ncore] = 1
    dm_core_ao = reduce(np.dot, (mo, dm_core, mo.T.conj()))
    vj_c, vk_c = eris.get_jk(dm_core_ao, mo_coeff=mo, mo_occ=core_occ)
    eris.vj_c = vj_c
    eris.vk_c = vk_c
    vhf_c = reduce(np.dot, (mo.T.conj(), vj_c - vk_c, mo))
    vj_a_mo, vk_a_mo = eris.get_jk_active_mo(casdm1)
    vhf_a = vj_a_mo - vk_a_mo
    vhf_ca = vhf_c + vhf_a

    c_core = np.eye(ncore)
    c_vir = np.eye(nmo - nocc)
    # canonicalization begins here
    if casscf.canonicalize_:
        fock_eff = h1e_mo + vhf_ca
        fock_core = fock_eff[:ncore, :ncore]
        fock_vir = fock_eff[nocc:, nocc:]
        e_core, c_core = _subspace_eigh(casscf, fock_core, mo[:, :ncore])
        e_vir, c_vir = _subspace_eigh(casscf, fock_vir, mo[:, nocc:])
        mo[:, :ncore] = np.dot(mo[:, :ncore], c_core)
        mo[:, nocc:] = np.dot(mo[:, nocc:], c_vir)
        h1e_mo = reduce(np.dot, (mo.T.conj(), casscf.get_hcore(), mo))
        vhf_c = reduce(np.dot, (mo.T.conj(), vj_c - vk_c, mo))
        # Rotate vhf_a to new MO basis using canonicalization rotation
        c = np.eye(nmo, dtype=complex)
        c[:ncore, :ncore] = c_core
        c[nocc:, nocc:] = c_vir
        vhf_a = reduce(np.dot, (c.T.conj(), vhf_a, c))
        vhf_ca = vhf_c + vhf_a
        logger.debug(casscf, "Super-CI canonical core energies = %s", e_core)
        logger.debug(casscf, "Super-CI canonical virtual energies = %s", e_vir)
        logger.debug(
            casscf,
            "Super-CI active-active effective Fock block =\n%s",
            fock_eff[ncore:nocc, ncore:nocc],
        )
    # canonicalization ends
    fock_eff = h1e_mo + vhf_ca
    fock_eff_core = fock_eff[:ncore, :ncore]
    fock_eff_vir = fock_eff[nocc:, nocc:]
    fock_offdiag_core = np.linalg.norm(
        fock_eff_core - np.diag(fock_eff_core.diagonal())
    )
    fock_offdiag_vir = np.linalg.norm(fock_eff_vir - np.diag(fock_eff_vir.diagonal()))
    logger.info(
        casscf,
        "\nSuper-CI Fock | Core offdiag = %.3e | Virtual offdiag = %.3e",
        fock_offdiag_core,
        fock_offdiag_vir,
    )
    g = np.zeros((nmo, nmo), dtype=complex)
    g[:, :ncore] = h1e_mo[:, :ncore] + vhf_ca[:, :ncore]
    g[:, ncore:nocc] = np.dot(h1e_mo[:, ncore:nocc] + vhf_c[:, ncore:nocc], casdm1)

    g_new = np.zeros((nmo, nmo), dtype=complex)
    # g[:, :ncore] = h1e_mo[:, :ncore] + vhf_ca[:, :ncore]
    # g[:, ncore:nocc] = np.dot(
    # h1e_mo[:, ncore:nocc] + vhf_c[:, ncore:nocc], casdm1)
    g_new[ncore:, :ncore] = h1e_mo[ncore:, :ncore] + vhf_ca[ncore:, :ncore]
    g_new[ncore:, ncore:nocc] = np.dot(
        h1e_mo[:, ncore:nocc] + vhf_c[:, ncore:nocc], casdm1
    )[ncore:, :]
    g_new[nocc:, ncore:nocc] = np.dot(
        h1e_mo[nocc:, ncore:nocc] + vhf_c[nocc:, ncore:nocc], casdm1
    )
    g_new[ncore:nocc, :ncore] = np.dot(
        casdm1, h1e_mo[ncore:nocc, :ncore] + vhf_c[ncore:nocc, :ncore]
    )
    g_dm2 = _contract_dm2_gradient(eris, casdm2)

    # transform g_dm2 to canonical basis
    c = np.eye(nmo, dtype=complex)
    c[:ncore, :ncore] = c_core
    c[nocc:, nocc:] = c_vir
    g_dm2 = np.dot(c.T.conj(), g_dm2)
    # transformation done

    g[:, ncore:nocc] += g_dm2
    g_new[ncore:nocc, :ncore] += g_dm2.T[:, :ncore]
    g_new[nocc:, ncore:nocc] += g_dm2[nocc:, :]
    g_orb = casscf.pack_uniq_var(g - g.T.conj())
    # g_orb = casscf.pack_uniq_var(g_new-g_new.T.conj())

    fock_eff = h1e_mo + vhf_ca
    # g_orb = casscf.pack_uniq_var(g - g.T.conj())
    # g = g - g.T.conj()

    # term1 h_ai,bj = (delta_ij F_ab - delta_ab F_ji)
    f_oo = fock_eff[:ncore, :ncore]
    f_vv = fock_eff[nocc:, nocc:]
    f_aa = fock_eff[ncore:nocc, ncore:nocc]
    # intermediate for hessian calculation
    # g = np.zeros((nmo, nmo), dtype=complex)
    # g[:, :ncore] = h1e_mo[:, :ncore] + vhf_ca[:, :ncore]
    # g[:, ncore:nocc] = np.dot(
    #     h1e_mo[:, ncore:nocc] + vhf_c[:, ncore:nocc], casdm1)
    # paaa = eris.paaa
    # g_dm2 = lib.einsum('puvw,tuvw->pt', paaa, casdm2)
    # g_tu = d_tu,vw F_vw - F_vw,D_vw,D_tu
    g_tu = lib.einsum("tuvw,vw->tu", casdm2, g[ncore:nocc, ncore:nocc]) - lib.einsum(
        "tu,vw,vw->tu", casdm1, casdm1, g[ncore:nocc, ncore:nocc]
    )
    # g_tu2 = d_vw,ut F_vw - F_vw,D_vw,D_tu
    g_tu2 = lib.einsum("vwut,vw->tu", casdm2, g[ncore:nocc, ncore:nocc]) - lib.einsum(
        "tu,vw,vw->tu", casdm1, casdm1, g[ncore:nocc, ncore:nocc]
    )

    y = lib.einsum("pu,qu->pq", (h1e_mo + vhf_c)[ncore:nocc, ncore:nocc], casdm1)
    h_diag = np.ones((nmo, nmo), dtype=complex)
    for v_idx in range(nocc, nmo):
        for i_idx in range(ncore):
            h_diag[v_idx, i_idx] = fock_eff[v_idx, v_idx] - fock_eff[i_idx, i_idx]

    for a_idx in range(ncore, nocc):
        for i_idx in range(ncore):
            d_tt = dm1[a_idx, a_idx]
            h_diag[a_idx, i_idx] = (
                fock_eff[a_idx, a_idx]
                - (1.0 - d_tt) * fock_eff[i_idx, i_idx]
                - y[a_idx - ncore, a_idx - ncore]
                - g_dm2[a_idx, a_idx - ncore]
            )

    for v_idx in range(nocc, nmo):
        for a_idx in range(ncore, nocc):
            h_diag[v_idx, a_idx] = (
                fock_eff[v_idx, v_idx] * dm1[a_idx, a_idx]
                - y[a_idx - ncore, a_idx - ncore]
                - g_dm2[a_idx, a_idx - ncore]
            )
    h_diag = h_diag.real * (1.0 + 0.0j)

    def h_op(x):
        x1 = casscf.unpack_uniq_var(x)
        # super-ci hessian
        sigma = np.zeros_like(x1)
        f_oo = fock_eff[:ncore, :ncore]
        f_vv = fock_eff[nocc:, nocc:]
        f_aa = fock_eff[ncore:nocc, ncore:nocc]

        f_ov = fock_eff[:ncore, nocc:]
        dm1_aa = dm1[ncore:nocc, ncore:nocc]

        n_tt = dm1_aa.diagonal()
        m_tt = 1.0 - dm1_aa.diagonal()
        n_tt_sqrt = np.sqrt(dm1_aa.diagonal())
        m_tt_sqrt = np.sqrt(1.0 - dm1_aa.diagonal())

        one = np.ones((nocc - ncore, nocc - ncore))
        scale = (
            one - np.einsum("ij,j->ij", one, n_tt) - np.einsum("ij,i->ij", one, n_tt)
        )

        # core-virtual block
        # term1 h_ai,bj = (delta_ij F_ab - delta_ab F_ji)
        sigma[nocc:, :ncore] += lib.einsum(
            "ab,bi->ai", f_vv, x1[nocc:, :ncore]
        ) - lib.einsum("ji,aj->ai", f_oo, x1[nocc:, :ncore])

        # term 2 h_ai,bu = -delta_ab*f_vi*D_vu
        sigma[nocc:, :ncore] -= lib.einsum(
            "vi,vu,au->ai", g[ncore:nocc, :ncore], dm1_aa, x1[nocc:, ncore:nocc]
        )

        # term3 h_ai,uj = delta_ij(f_au-f_av*D_uv)
        sigma[nocc:, :ncore] += lib.einsum(
            "au,ui->ai", g[nocc:, ncore:nocc], x1[ncore:nocc, :ncore]
        ) - lib.einsum(
            "av,uv,ui->ai", g[nocc:, ncore:nocc], dm1_aa, x1[ncore:nocc, :ncore]
        )

        # core-active block
        # term5 h_ti,uj =
        # f_ji * (D_ut - delta_tu)
        sigma[ncore:nocc, :ncore] += lib.einsum(
            "ut,ji,uj->ti", dm1_aa, f_oo, x1[ncore:nocc, :ncore]
        ) - lib.einsum("ji,tj->ti", f_oo, x1[ncore:nocc, :ncore])
        # h_ti,uj += f_tu delta_ij - f_tv,d_uv,delta_ij
        sigma[ncore:nocc, :ncore] += (
            lib.einsum(
                "tu,ui->ti", g[ncore:nocc, ncore:nocc] - g_tu, x1[ncore:nocc, :ncore]
            )
            - lib.einsum(
                "tv,uv,ui->ti",
                g[ncore:nocc, ncore:nocc],
                dm1_aa,
                x1[ncore:nocc, :ncore],
            )
            - lib.einsum(
                "tv,uv,ui->ti",
                dm1_aa,
                g[ncore:nocc, ncore:nocc],
                x1[ncore:nocc, :ncore],
            )
        )
        # term5 continued
        # g_tu = d_tu,vw F_vw - F_vw,D_vw,D_tu
        # + delta_ij(f_tu-(d_tu,vw-d_ut*d_vw)*f_vw-f_tv*d_uv
        # the last two term differs from molpro's expression since molpro
        # suppose a symmetrized form of 2rdm while we don't.

        # term3 h_ai,uj = delta_ij(f_au-f_av*D_uv)
        # adjoint of term 3 h_ti,bj x_bj->sigma_ti
        sigma[ncore:nocc, :ncore] += lib.einsum(
            "au,ai->ui", g[nocc:, ncore:nocc], x1[nocc:, :ncore]
        ) - lib.einsum("av,uv,ai->ui", g[nocc:, ncore:nocc], dm1_aa, x1[nocc:, :ncore])
        # sigma[ncore:nocc,:ncore] += x1[ncore:nocc,:ncore] * h_diag[ncore:nocc,:ncore]
        # virtual-active block
        # adjoint of term2
        # h_bu,ai = -delta_ab * f_vi*D_vu
        sigma[nocc:, ncore:nocc] -= lib.einsum(
            "vi,vu,ai->au", g[ncore:nocc, :ncore], dm1_aa, x1[nocc:, :ncore]
        )
        # term4 h_ti,bu = 0

        # term 6 h_at,bu=delta_ab(d_tu,vw-d_tu*d_vw)f_vw+d_tu*f_ab
        # sigma[nocc:,ncore:nocc] += x1[nocc:,ncore:nocc] * h_diag[nocc:,ncore:nocc]

        sigma[nocc:, ncore:nocc] += lib.einsum(
            "tu,ab,bu->at", dm1_aa, f_vv, x1[nocc:, ncore:nocc]
        ) + lib.einsum("tu,au->at", g_tu, x1[nocc:, ncore:nocc])

        sigma_pack = casscf.pack_uniq_var(sigma)
        return sigma_pack

    n_uniq_var = g_orb.shape[0]
    hop = LinearOperator((n_uniq_var, n_uniq_var), matvec=h_op)
    metric_dm1, metric_diagnostics = _physical_active_density(casdm1)
    casscf.superci_metric_diagnostics = metric_diagnostics

    def s_op(x):
        x1 = casscf.unpack_uniq_var(x)
        # In a natural-orbital basis these products reduce to the historical
        # elementwise factors n_t and 1-n_t.  Keeping the full density makes
        # the Super-CI metric covariant under arbitrary active-space rotations
        # (including rotations inside exactly degenerate Kramers manifolds).
        sx1 = _apply_superci_metric(x1, metric_dm1, ncore, nocc)
        return casscf.pack_uniq_var(sx1)

    sop = LinearOperator((n_uniq_var, n_uniq_var), matvec=s_op)

    def h_diag_inv(x):
        return x / (casscf.pack_uniq_var(h_diag + h_diag.T))

    precond = LinearOperator((n_uniq_var, n_uniq_var), h_diag_inv)
    return g_orb, casscf.pack_uniq_var((h_diag + h_diag.T).real), hop, sop, precond, mo


def precondition_grad0(grad, xs, ys, rhos, bfgs_space=10):
    assert len(ys) <= bfgs_space, "size of xs greater than bfgs space size"
    gbar = grad.copy()
    niter = len(xs) if bfgs_space > len(xs) else bfgs_space
    a = np.zeros(niter)
    for ii in range(niter):
        i = niter - ii - 1
        a[i] = np.dot(xs[i].conj(), gbar).real / rhos[i]
        gbar = gbar - ys[i] * a[i]
        # print(f'precond_grad, {ii}, {i}, {np.linalg.norm(gbar):.4e}, {np.linalg.norm(xs[i]):.4f} {np.linalg.norm(ys[i]):.4f}, {rhos[i]:.4f}, {a[i]:.4e}, {np.dot(xs[i].conj(), gbar).real:.4e} ')
    print(
        f"{len(ys)} {len(xs)} {np.linalg.norm(gbar):.4e} {np.linalg.norm(gbar - grad):.4e} bfgs precond"
    )
    return gbar, a


def postprocess_x0(xbar, xs, ys, rhos, a, bfgs_space=10):
    assert len(xs) <= bfgs_space, "size of xs greater than bfgs space size"
    niter = len(xs) if bfgs_space > len(xs) else bfgs_space
    xorig = xbar.copy()
    for i in range(niter):
        b = np.dot(ys[i].conj(), xbar).real / rhos[i]
        # print(f'postprocess {a[i]:.4e}, {b:.4e}, {np.linalg.norm(xs[i]):.4f}')
        xbar = xbar - xs[i] * (a[i] - b)
    print(f"bfgs post {np.linalg.norm(xorig - xbar):.4e}, {np.linalg.norm(xorig):.4e}")
    return 0.5 * xbar




class _FixedRDMOrbitalObjective:
    """Forte2-style orbital objective: rotate MOs while keeping CI RDMs fixed."""

    def __init__(self, mc, mo, dm1, dm2, eris, provenance, x, quantities):
        self.mc, self.mo, self.dm1, self.dm2 = mc, mo.copy(), dm1, dm2
        self.eris, self.provenance = eris, provenance
        self.x = x.copy()
        self.unitary = np.eye(mo.shape[1], dtype=complex)
        self.quantities = quantities

    def evaluate(self, x):
        delta = x - self.x
        if np.any(delta):
            rotation = expmat(self.mc.unpack_uniq_var(delta))
            self.unitary = self.unitary @ rotation
            self.mo = self.mo @ rotation
            self.eris, self.provenance = _build_eris(self.mc, self.mo)
            self.x = x.copy()
            self.quantities = None
        mci = zmcscf._fake_h_for_fast_casci(self.mc, self.mo, self.eris)
        h1, ecore = mci.get_h1eff(self.mo)
        energy = (np.einsum('pq,pq->', h1, self.dm1)
                  + .5*np.einsum('pqrs,pqrs->', self.eris.aaaa, self.dm2)
                  + ecore)
        if not np.isfinite(energy) or abs(energy.imag) > 1e-8:
            raise RuntimeError('Fixed-RDM orbital energy is not real and finite')
        return float(energy.real)

    def gradient(self, x):
        if self.quantities is None:
            from socutils.mcscf.zmc_utils import build_orbital_quantities
            self.quantities = build_orbital_quantities(
                self.mc, self.mo, self.dm1, self.dm2, self.eris)
        # dE = 2 Re(g^H dR) in the independent complex rotation variables.
        return 2*self.mc.pack_uniq_var(self.quantities.gradient)

    def hess_diag(self, x):
        q = self.quantities
        nc, no, nm = self.mc.ncore, self.mc.ncore+self.mc.ncas, self.mo.shape[1]
        f = np.diag(q.fock_effective).real
        occupation = np.diag(self.dm1).real
        lagrangian = np.diag(q.lagrangian)[nc:no].real
        h = np.zeros((nm, nm))
        h[no:, :nc] = 2*(f[no:, None] - f[None, :nc])
        h[no:, nc:no] = 2*(f[no:, None]*occupation[None, :] - lagrangian[None, :])
        h[nc:no, :nc] = 2*((f[nc:no]-lagrangian)[:, None]
                            - (1-occupation)[:, None]*f[None, :nc])
        return self.mc.pack_uniq_var(h)


def _fixed_rdm_lbfgs_step(mc, mo, dm1, dm2, eris, provenance, x, quantities,
                         max_dir, grad_tol):
    """Forte2's six max-correction L-BFGS microsteps, reset every macro."""
    objective = _FixedRDMOrbitalObjective(
        mc, mo, dm1, dm2, eris, provenance, x, quantities)
    x = x.copy()
    initial_energy = energy = objective.evaluate(x)
    gradient = objective.gradient(x)
    initial_gradient_norm = float(np.linalg.norm(gradient))
    diagonal = objective.hess_diag(x)
    cutoff = max(1e-12, 1e-10*np.max(abs(diagonal), initial=0.))
    mask = abs(diagonal) > cutoff
    pairs = []
    last_x, last_gradient = x.copy(), gradient.copy()
    converged = initial_gradient_norm <= grad_tol*max(1., np.linalg.norm(x))
    skipped = 0
    iterations = 0
    while not converged and iterations < 6:
        direction = gradient.copy()
        alphas = []
        for step, change, rho in reversed(pairs):
            alpha = rho*np.vdot(step, direction)
            direction -= alpha*change
            alphas.append(alpha)
        direction[mask] /= diagonal[mask]
        for (step, change, rho), alpha in zip(pairs, reversed(alphas)):
            beta = rho*np.vdot(change, direction)
            direction += (alpha-beta)*step
        direction *= -1
        if not np.all(np.isfinite(direction)):
            raise RuntimeError('Fixed-RDM L-BFGS direction is nonfinite')
        largest = float(np.max(abs(direction), initial=0.))
        if largest == 0:
            break
        x += min(1., max_dir/largest)*direction
        energy = objective.evaluate(x)
        iterations += 1
        if iterations == 6:
            break
        gradient = objective.gradient(x)
        converged = np.linalg.norm(gradient) <= grad_tol*max(1., np.linalg.norm(x))
        if converged:
            break
        step, change = x-last_x, gradient-last_gradient
        curvature = np.vdot(change, step)
        if curvature.real > 0:
            pairs.append((step.copy(), change.copy(), 1/curvature))
            pairs = pairs[-6:]
            last_x, last_gradient = x.copy(), gradient.copy()
        else:
            skipped += 1
    generator = scipy.linalg.logm(objective.unitary)
    generator = (generator-generator.conj().T)*.5
    return objective.mo, objective.eris, objective.provenance, x, generator, {
        'solver': 'fixed_rdm_lbfgs', 'converged': bool(converged),
        'iterations': iterations, 'history_size': len(pairs),
        'skipped_curvature_pairs': skipped, 'initial_gradient_norm': initial_gradient_norm,
        'fixed_rdm_energy_start': initial_energy, 'fixed_rdm_energy_end': energy,
        'fixed_rdm_energy_change': energy-initial_energy, 'max_correction': max_dir,
    }




def mcscf_superci(
    mc,
    mo_coeff,
    max_stepsize=0.2,
    conv_tol=None,
    conv_tol_grad=None,
    verbose=5,
    cderi=None,
    bfgs=False,
    solver="davidson",
    davidson_maxiter=10,
    davidson_tol=5e-6,
    davidson_strict=True,
    symm=None,
    callback=None,
    forte2=False,
):
    # cderi is retained for compatibility with callers that supply vectors
    # directly; normal calculations use the CD object attached to the SCF.
    davidson_mmax = davidson_maxiter
    log = logger.new_logger(mc, verbose)
    cput0 = (logger.process_clock(), logger.perf_counter())
    mc.canonicalization_diagnostics = None
    if conv_tol is None:
        conv_tol = mc.conv_tol
    mol = mc.mol
    # if mc.irrep is None:
    #    mo = form_kramers(mo_coeff)
    mo = np.array(mo_coeff, dtype=np.complex128, copy=True)
    nmo = mo_coeff.shape[1]
    ncore = mc.ncore
    ncas = mc.ncas
    nocc = ncore + ncas

    if solver not in ("davidson", "gmres"):
        raise ValueError("Super-CI solver must be 'davidson' or 'gmres'")
    kramers = _resolve_kramers_mode(mc, symm)
    adaptive = bool(getattr(mc, "superci_adaptive", False)) and not forte2
    build_operators, solve_davidson = gen_g_hop, davidson
    if forte2:
        from socutils.mcscf import zmc_superci_adaptive
        zmc_superci_adaptive.validate(mc, mo, solver=solver, cderi=cderi, kramers=kramers)
        if bfgs:
            raise ValueError('Forte2 orbital L-BFGS requires Super-CI BFGS disabled')
        mc.superci_metric_diagnostics = {}
        log.info('Orbital optimizer = Forte2 fixed-RDM complex L-BFGS (max 6 microsteps)')
    elif adaptive:
        from socutils.mcscf import zmc_superci_adaptive

        zmc_superci_adaptive.validate(
            mc, mo, solver=solver, cderi=cderi, kramers=kramers,
        )
        build_operators = zmc_superci_adaptive.gen_g_hop
        solve_davidson = zmc_superci_adaptive.davidson
        log.info("Super-CI adaptive = True (orbital shift selected by step radius)")
    bounded = adaptive
    if bounded and bfgs:
        raise ValueError('Bounded orbital optimization requires BFGS disabled')
    initial_radius = getattr(mc, 'orbital_trust_start', .2)
    if bounded and (not np.isfinite(initial_radius) or initial_radius <= 0):
        raise ValueError('Invalid orbital trust radius or micro-step tolerance')
    orbital_radius = min(max_stepsize, initial_radius)
    method_name = 'Forte2' if forte2 else 'Super-CI'
    log.info("%s orbital solver = %s", method_name,
             'fixed-RDM L-BFGS' if forte2 else solver)
    log.info("MCSCF Kramers = %s", kramers)
    if solver == "davidson" and not forte2:
        log.info(
            "%s Davidson tolerance = %.3g, maximum space = %d, strict = %s",
            method_name,
            davidson_tol,
            davidson_mmax,
            davidson_strict,
        )

    mci = mc.view(zcasci.CASCI)
    eris, integral_info = _build_eris(mc, mo, cderi=cderi)
    mc.cholesky_diagnostics = dict(integral_info)
    log.info(
        "MCSCF ERI route: representation = %s, source = %s, "
        "container = %s, naux = %s, Cholesky = %s, threshold = %s",
        integral_info["representation"],
        integral_info["source"],
        integral_info["container"],
        integral_info["naux"],
        integral_info["active"],
        integral_info["threshold"],
    )
    mci = zmcscf._fake_h_for_fast_casci(mc, mo, eris)
    log.info("******** Initial %s CASCI ********", method_name)
    e_tot, e_cas, fcivec = mci.kernel(mo, verbose=verbose)
    ci_converged = bool(np.all(getattr(mc.fcisolver, "converged", True)))
    if not ci_converged:
        raise RuntimeError("The active-space CI solver did not converge")
    mc.e_tot, mc.e_cas = e_tot, e_cas
    # mc._finalize()
    # e_tot, e_cas, fcivec = mc.casci(mo, ci0=None, eris=eris)
    if conv_tol_grad is None:
        conv_tol_grad = np.sqrt(conv_tol)
        logger.info(mc, "Set conv_tol_grad to %g", conv_tol_grad)

    conv = False
    norm_gorb = norm_gci = -1
    de, elast = np.inf, e_tot

    t1m = log.timer("Initializing MCSCF", *cput0)
    casdm1, casdm2 = mc.fcisolver.make_rdm12(fcivec, ncas, mc.nelecas)

    norm_rot = 0.0
    norm_ddm = 1e2
    casdm1_prev = casdm1_last = casdm1
    t3m = t2m = log.timer("CAS DM", *t1m)

    imacro = 0
    xs = []
    ys = []
    rhos = []
    g_prev = None
    x_prev = None
    rejected = False
    trust_radii = 0.5
    orbital_rotation_vector = (np.zeros_like(mc.pack_uniq_var(np.zeros((nmo, nmo))), dtype=complex)
                               if forte2 else None)
    e_last = e_tot
    dr = None
    macro_history = []
    mc.macro_history = macro_history
    last_linear_info = None
    while not conv and imacro < mc.max_cycle_macro:
        macro_wall_start = logger.perf_counter()
        # compute natural orbital and transform ci to natural orbtial basis
        # no transform function available now so re do a ci calculation
        # do it in gen_g_hop
        if mc.natorb is True:
            moa = mo[:, ncore:nocc]
            if kramers:
                natocc, c = _kramers_subspace_eigh(
                    mc,
                    -casdm1,
                    moa,
                )
            else:
                natocc, c = _active_natural_orbitals(mc, casdm1, moa)
            moa_new = np.dot(moa, c)
            mo[:, ncore:nocc] = moa_new

            eris, integral_info = _build_eris(mc, mo, cderi=cderi)
            t2m = log.timer("update eris", *t2m)
            mci = zmcscf._fake_h_for_fast_casci(mc, mo, eris)
            e_nat_tot, e_cas, fcivec = mci.kernel(mo, ci0=None, verbose=verbose)
            ci_converged = bool(np.all(getattr(mc.fcisolver, "converged", True)))
            if not ci_converged:
                raise RuntimeError(
                    "The active-space CI solver did not converge "
                    "after the natural-orbital rotation"
                )
            log.debug(
                "Super-CI natural-orbital CASCI energy %.15f -> %.15f",
                e_tot,
                e_nat_tot,
            )
            e_tot = e_nat_tot
            casdm1, casdm2 = mci.fcisolver.make_rdm12(fcivec, ncas, mc.nelecas)
            log.debug(
                "Super-CI active natural occupations = %s",
                casdm1.diagonal(),
            )

        if forte2:
            from socutils.mcscf.zmc_utils import build_orbital_quantities
            quantities = build_orbital_quantities(mc, mo, casdm1, casdm2, eris)
            g = mc.pack_uniq_var(quantities.screened_gradient)
        else:
            g, h_diag, hop, sop, precond, mo = build_operators(mc, mo, casdm1, casdm2, eris)
        norm_gorb = norm(g)
        de_text = "inf" if not np.isfinite(de) else "%.3e" % de
        log.info(
            "MCSCF macro = %4d | E = %22.15f | dE = %11s | "
            "Grad norm = %9.3e | Step norm = %9.3e",
            imacro,
            e_tot,
            de_text,
            norm_gorb,
            norm_rot,
        )
        t2m = log.timer("Compute gradient", *t2m)
        norm_gorb = np.linalg.norm(g)

        natural_occupations = np.linalg.eigvalsh((casdm1 + casdm1.T.conj()) * 0.5).real[
            ::-1
        ]
        history_entry = {
            "macro_iteration": imacro,
            "total_energy": float(np.real(e_tot)),
            "energy_change": None if not np.isfinite(de) else float(np.real(de)),
            "cas_energy": float(np.real(e_cas)),
            "orbital_gradient_norm": float(norm_gorb),
            "orbital_step_norm": float(norm_rot),
            "natural_occupations": natural_occupations.tolist(),
            "converged": False,
            "accepted": True,
            "ci_solver_converged": ci_converged,
            "ci_solver_diagnostics": _ci_convergence_snapshot(mc.fcisolver),
            "integral_representation": integral_info["representation"],
            "integral_factorized": integral_info["factorized"],
            "integral_source": integral_info["source"],
            "cholesky_active": integral_info["active"],
            "cholesky_naux": integral_info["naux"],
            "superci_metric": dict(mc.superci_metric_diagnostics),
            "adaptive": adaptive,
            "orbital_method": "forte2" if forte2 else "superci",
        }
        macro_history.append(history_entry)

        g_unpack = mc.unpack_uniq_var(g)
        row, col = np.unravel_index(np.argmax(abs(g_unpack)), g_unpack.shape)
        if log.verbose >= logger.DEBUG:
            for i in range(nmo):
                for j in range(i):
                    if abs(g_unpack[i, j]) > 1e-2:
                        log.debug(
                            "Super-CI gradient (%d,%d) = %s",
                            i,
                            j,
                            g_unpack[i, j],
                        )
            if mc.irrep is not None:
                log.debug(
                    "Super-CI largest gradient (%d,%d) = %s [%s,%s]",
                    row,
                    col,
                    g_unpack[row, col],
                    mc.irrep[row],
                    mc.irrep[col],
                )
            else:
                log.debug(
                    "Super-CI largest gradient (%d,%d) = %s",
                    row,
                    col,
                    g_unpack[row, col],
                )
        if abs(de) < conv_tol and norm_gorb < conv_tol_grad:
            conv = True
        if conv:
            _schedule_orbital_trial(mc, norm_gorb, None)
            history_entry["converged"] = True
            log.info(
                "MCSCF converged | Macro = %4d | E = %22.15f | Grad norm = %.3e",
                imacro,
                e_tot,
                norm_gorb,
            )
            if callback is not None:
                callback(dict(history_entry))
            break

        if forte2:
            (mo_new, eris, integral_info, orbital_rotation_vector, dr,
             last_linear_info) = _fixed_rdm_lbfgs_step(
                mc, mo, casdm1, casdm2, eris, integral_info,
                orbital_rotation_vector, quantities, max_stepsize, conv_tol_grad)
            applied_x = mc.pack_uniq_var(dr)
            step_norm = float(norm(dr))
            history_entry['restart_before_trial'] = _schedule_orbital_trial(
                mc, norm_gorb, step_norm)
            mci = zmcscf._fake_h_for_fast_casci(mc, mo_new, eris)
            e_tot, e_cas, fcivec = mci.kernel(mo_new, ci0=None, verbose=verbose)
            ci_converged = bool(np.all(getattr(mc.fcisolver, 'converged', True)))
            if not ci_converged or not np.isfinite(e_tot):
                raise RuntimeError('The active-space CI solver did not converge after Forte2 orbital microsteps')
            de = float(e_tot - e_last)
            e2 = last_linear_info['fixed_rdm_energy_change']
            r = de/e2 if abs(e2) > 1e-16 else np.nan
            step_rescaled = False
            trust_radii = max_stepsize
            trust_action = 'fixed-RDM L-BFGS / CI updated'
            history_entry.update(
                linear_solver=last_linear_info,
                proposed_orbital_step_norm=step_norm,
                applied_orbital_step_norm=step_norm,
                step_rescaled=False,
                next_total_energy=float(e_tot),
                accepted_energy_change=de,
                predicted_energy_change=e2,
                prediction_model='fixed_rdm_energy',
                next_ci_solver_diagnostics=_ci_convergence_snapshot(mc.fcisolver),
            )
        elif bounded:
            (mo_new, e_tot, e_cas, fcivec, eris, integral_info, dr, applied_x,
             last_linear_info, e2, de, r, orbital_radius, trials) = _bounded_orbital_update(
                mc, mo, e_last, g, h_diag, hop, sop, solve_davidson,
                orbital_radius, max_stepsize, davidson_tol, davidson_mmax,
                conv_tol, conv_tol_grad, False, verbose, log, cderi=cderi)
            ci_converged = True
            step_rescaled = False
            trust_radii = orbital_radius
            trust_action = 'accepted / bounded Super-CI'
            history_entry.update(linear_solver=last_linear_info, orbital_trials=trials,
                proposed_orbital_step_norm=float(norm(dr)), applied_orbital_step_norm=float(norm(dr)),
                step_rescaled=False, next_total_energy=float(e_tot), accepted_energy_change=de,
                predicted_energy_change=e2,
                prediction_model='linear_orbital_gradient',
                next_ci_solver_diagnostics=_ci_convergence_snapshot(mc.fcisolver),
                restart_before_trial=trials[-1]['restart'],
                rejected_trials=sum(trial['stage'] == 'casci' and not trial['accepted'] for trial in trials),
                inner_solver_failures=sum(trial['stage'] == 'orbital_solve' for trial in trials),
                trial_radius=trials[-1]['radius'])
            if r is None:
                r = np.nan
        else:
            gbar = g

            t_gmres = (logger.process_clock(), logger.perf_counter())
            BFGS_SUBSPACE = 6
            apply_bfgs = False
            if imacro > 0:
                bfgs_on = 1.0
                if not rejected and norm_gorb < bfgs_on:
                    ys.append(g - g_prev)
                    xs.append(x_prev)
                    rhos.append(2 * np.dot(ys[-1].conj(), xs[-1]).real)
                if len(ys) > BFGS_SUBSPACE:
                    ys.pop(0)
                    xs.pop(0)
                    rhos.pop(0)
                # if np.linalg.norm(g_prev) < norm_gorb or de > 0.0:
                if de > 0.0:
                    log.debug(
                        "Super-CI BFGS history reset: gradient %.6g -> %.6g",
                        np.linalg.norm(g_prev),
                        norm_gorb,
                    )
                    xs = []
                    ys = []
                    rhos = []
                if bfgs is True and norm_gorb < bfgs_on:
                    gbar, a = precondition_grad(g, xs, ys, rhos, bfgs_space=BFGS_SUBSPACE)
                    apply_bfgs = True

            if solver == "gmres":
                residuals = []

                def linear_callback(rk):
                    residuals.append(float(rk))

                x, gmres_info = gmres(
                    hop,
                    -trust_radii * gbar,
                    M=precond,
                    maxiter=50,
                    callback=linear_callback,
                    callback_type="pr_norm",
                )
                last_linear_info = {
                    "solver": "gmres",
                    "converged": gmres_info == 0,
                    "iterations": len(residuals),
                    "residual_norm": residuals[-1] if residuals else None,
                    "residual_history": residuals,
                    "reason": "converged" if gmres_info == 0 else "maximum_iterations",
                }
            else:
                if imacro > 0:
                    trust_radii = max(trust_radii, 0.2)
                x, e, last_linear_info = solve_davidson(
                    hop,
                    trust_radii * gbar,
                    h_diag,
                    sop=sop,
                    max_stepsize=trust_radii,
                    tol=davidson_tol,
                    mmax=davidson_mmax,
                    log=log,
                )
                last_linear_info = dict(last_linear_info, solver="davidson")
            linear_residual = last_linear_info["residual_norm"]
            residual_text = "n/a" if linear_residual is None else "%.3e" % linear_residual
            log.info(
                "Super-CI solve = %4d | Solver = %-8s | Iterations = %4d | "
                "Residual = %9s | Converged = %s \n",
                imacro,
                solver.capitalize(),
                last_linear_info["iterations"],
                residual_text,
                last_linear_info["converged"],
            )
            history_entry["linear_solver"] = last_linear_info
            if not last_linear_info["converged"]:
                message = (
                    "Super-CI %s did not converge: residual %s after %d iterations"
                    % (
                        solver,
                        last_linear_info["residual_norm"],
                        last_linear_info["iterations"],
                    )
                )
                if davidson_strict:
                    mc.superci_diagnostics = {
                        "adaptive": adaptive,
                        "linear_solver": last_linear_info,
                        "final_gradient_norm": float(norm_gorb),
                        "converged": False,
                        "integrals": dict(integral_info),
                    }
                    raise RuntimeError(message)
                log.warn(message)
            if apply_bfgs:
                x = 0.5 * postprocess_x(x, xs, ys, rhos, a, bfgs_space=BFGS_SUBSPACE)
            t2m = log.timer("Solving Super-CI equation", *t_gmres)

            dr = mc.unpack_uniq_var(x)
            kramers_mapping = (
                _identify_kramers_mapping(mc, mo) if kramers else None
            )
            dr, kramers_rotation = _project_kramers_rotation(
                mc,
                mo,
                dr,
                force=kramers,
                mapping=kramers_mapping,
            )
            step_control = max_stepsize
            proposed_step_norm = float(norm(dr))
            history_entry["proposed_orbital_step_norm"] = proposed_step_norm
            if log.verbose >= logger.DEBUG:
                for i in range(nmo):
                    for j in range(i):
                        if abs(dr[i, j]) > 1e-2:
                            log.debug(
                                "Super-CI orbital step (%d,%d) = %s",
                                i,
                                j,
                                dr[i, j],
                            )
            step_rescaled = proposed_step_norm > step_control
            if step_rescaled:
                dr = dr * (step_control / proposed_step_norm)
            if kramers_rotation is not None:
                history_entry["kramers_rotation"] = kramers_rotation
                log.info(
                    "Kramers-projected orbital generator: input residual "
                    "%.6g, output residual %.6g, change %.6g",
                    kramers_rotation["input_generator_residual"],
                    kramers_rotation["output_generator_residual"],
                    kramers_rotation["projection_change_norm"],
                )
            rotation = expmat(dr)
            mo_new = np.dot(mo, rotation)
            history_entry["applied_orbital_step_norm"] = float(norm(dr))
            history_entry["step_rescaled"] = bool(step_rescaled)
            applied_x = mc.pack_uniq_var(dr)

            norm_rot = np.linalg.norm(rotation - np.eye(nmo, dtype=complex))
            # e_tot, e_cas, fcivec, _, _ = mci.kernel(mo)
            eris, integral_info = _build_eris(mc, mo_new, cderi=cderi)
            t2m = log.timer("update eris", *t2m)
            mci = zmcscf._fake_h_for_fast_casci(mc, mo_new, eris)
            history_entry['restart_before_trial'] = _schedule_orbital_trial(
                mc, norm_gorb, float(norm(dr)))
            e_tot, e_cas, fcivec = mci.kernel(mo_new, ci0=None, verbose=verbose)
            ci_converged = bool(np.all(getattr(mc.fcisolver, "converged", True)))
            if not ci_converged:
                raise RuntimeError(
                    "The active-space CI solver did not converge after the orbital step"
                )

            # trus radius control
            # g_new, h_diag, hop, precondition = gen_g_hop(mc, mo_new, casdm1, casdm2, eris)
            # dg = norm(g_new) - norm(g)
            de = e_tot - e_last  # + dg * .1
            # On the corrected route g is half the real-coordinate gradient:
            # dE = 2 Re(g^H x). The numerator is not an orbital Hessian.
            # Preserve the old heuristic for the other integral/rotation routes.
            metric_frame = hasattr(hop, '_superci_frame')
            e2 = float((2.0 if metric_frame else 0.5) * np.vdot(applied_x, g).real)
            r = de / e2 if abs(e2) > 1e-16 else np.inf

            history_entry["next_total_energy"] = float(np.real(e_tot))
            history_entry["accepted_energy_change"] = float(np.real(de))
            history_entry["predicted_energy_change"] = float(np.real(e2))
            history_entry["prediction_model"] = (
                "linear_orbital_gradient" if metric_frame else "legacy_superci_estimate"
            )
            history_entry["next_ci_solver_diagnostics"] = _ci_convergence_snapshot(
                mc.fcisolver
            )
            if de > 0.0:
                trust_radii *= 0.5
                trust_action = "accepted energy rise / trust reduced"
            elif r < 0.25:  # and de < 0.0:
                trust_radii *= 0.5
                trust_action = "accepted / trust reduced"
            elif r > 0.75 and de < 0.0:
                trust_radii = min(1.4 * trust_radii, 1.0)
                trust_action = "accepted / trust increased"
            else:
                trust_action = "accepted / trust unchanged"
            if trust_radii < 1e-2 * max_stepsize:
                trust_radii = 1e-2 * max_stepsize
                rejected = False
        macro_wall = logger.perf_counter() - macro_wall_start
        history_entry["trust_radius"] = float(trust_radii)
        history_entry["trust_action"] = trust_action
        history_entry["macro_wall_time"] = float(macro_wall)
        log.info(
            "\nMCSCF update = %3d | E = %22.15f | dE = %10.3e | "
            "Pred = %10.3e | Ratio = %8.3f | Step = %.3e%s | "
            "Trust = %.3e | Time = %.2f s | %s",
            imacro,
            e_tot,
            de,
            e2,
            r,
            history_entry["applied_orbital_step_norm"],
            " (rescaled)" if step_rescaled else "",
            trust_radii,
            macro_wall,
            trust_action,
        )
        norm_rot = np.linalg.norm(dr)

        rejected = False
        mo = mo_new
        e_last = e_tot
        x_prev = applied_x
        g_prev = g
        casdm1, casdm2 = mc.fcisolver.make_rdm12(fcivec, ncas, mc.nelecas)
        history_entry["next_natural_occupations"] = (
            np.linalg.eigvalsh((casdm1 + casdm1.T.conj()) * 0.5).real[::-1].tolist()
        )
        if callback is not None:
            callback(dict(history_entry))
        # from socutils.tools import analyze
        # analyze.analyze(mol, mo[:, ncore:nocc], casdm1.diagonal())
        nact = casdm1.shape[0]
        # for i in range(nact):
        #    for j in range(i):
        #        if abs(casdm1[i, j]) > 1e-5:
        #            print(f'{i}, {j}, {casdm1[i,j]}')
        imacro += 1
        t1m = log.timer(f"macro iter {imacro}", *t1m)
        lib.chkfile.save(mc.chkfile, f"mo_coeff_iter_{imacro}", mo)
        if verbose >= logger.INFO:
            mc.e_tot = e_tot
            mc.e_cas = e_cas
            mc._finalize()
    mo_energy = None
    if mc.canonicalization:
        log.info("CASSCF final core/virtual canonicalization")
        mo, fcivec, mo_energy = mc.canonicalize(
            mo,
            fcivec,
            eris=eris,
            sort=mc.sorting_mo_energy,
            # Active natural-orbital changes are already performed inside the
            # macroiterations and followed by a fresh CI/DMRG solve.
            cas_natorb=False,
            casdm1=casdm1,
            verbose=verbose,
        )
    else:
        mc.canonicalization_diagnostics = {
            "enabled": False,
            "reason": "mc.canonicalization is False",
        }
    mc.mo_coeff = mo
    mc.mo_energy = mo_energy
    mc.final_orbital_gradient_norm = float(norm_gorb)
    mc.superci_diagnostics = {
        "adaptive": adaptive,
        "orbital_method": "forte2" if forte2 else "superci",
        "converged": bool(conv),
        "final_gradient_norm": float(norm_gorb),
        "energy_tolerance": float(conv_tol),
        "gradient_tolerance": float(conv_tol_grad),
        "linear_solver": last_linear_info,
        "metric": dict(mc.superci_metric_diagnostics),
        "integrals": dict(integral_info),
        "canonicalization": dict(mc.canonicalization_diagnostics),
        # Retain the historical key for callers that inspect CD provenance.
        "cholesky": dict(integral_info),
        "kramers_restricted": bool(kramers),
        "macro_iterations": int(imacro),
    }
    return conv, e_tot, e_cas, fcivec, mo, mo_energy


if __name__ == "__main__":
    mol = gto.M(
        atom="""
C -0.600  0.000  0.000
C  0.600  0.000  0.000
H   -1.4523499293        0.8996235720         .0000000000
H   -1.4523499293       -0.8996235720         .0000000000
H    1.4523499293        0.8996235720         .0000000000
H    1.4523499293       -0.8996235720         .0000000000
""",
        basis="ccpvdz",
        verbose=4,
        charge=0,
        max_memory=40000,
        nucmod="G",
    )

    from socutils.scf import spinor_hf, x2camf_hf
    from pyscf.x2c import x2c

    mf = x2c.RHF(mol)

    # mf.with_x2c = x2camf_hf.SpinorX2CAMFHelper(mol)
    mf.max_cycle = 50
    mf.kernel()
    print(mf.mo_coeff[:, 0], mf.mo_coeff[:, 1])
    for ene, occ in zip(mf.mo_energy, mf.mo_occ):
        print(f"{ene:20.15f} {occ:8.4g}")

    mf.mol.charge = 0
    mf.mol.build()
    mc = zmcscf.CASSCF(mf, ncas=6, nelecas=4)
    # mc = mc.state_average_(numpy.ones(9)/9.)
    mc.superci()
    print(mc.e_tot)
