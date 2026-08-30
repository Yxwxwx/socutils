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
from socutils.mcscf.hf_superci import precondition_grad, postprocess_x
from socutils.mcscf.orbital_diis import OrbitalDIIS


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
            "Super-CI active 1-RDM violates spinor N-representability by "
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
    """Build the Super-CI integral container selected by the SCF source.

    An attached ``with_df`` object (or the legacy explicit ``cderi`` argument)
    selects the existing factorized route.  Otherwise the direct four-index
    spinor transformation is used.  Keeping this decision in one helper is
    important because Super-CI rebuilds the transformed integrals after every
    accepted orbital rotation and, optionally, after active natural-orbital
    rotations.

    Returns
    -------
    eris
        A :class:`~socutils.mcscf.zmc_ao2mo._CDERIS` or
        :class:`~socutils.mcscf.zmc_ao2mo._ERIS` instance.
    diagnostics : dict
        Stable provenance fields shared by the macroiteration history and the
        final Super-CI diagnostics.
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


def _subspace_eigh(casscf, matrix, mo_subspace):
    """Diagonalize an MO-space matrix using reference symmetry when needed."""
    from socutils.scf import spinor_hf

    mf = casscf._scf
    if isinstance(mf, spinor_hf.KRHF):
        return mf.eig(matrix)
    if isinstance(mf, spinor_hf.SymmSpinorSCF):
        return mf.eig(matrix, mo=mo_subspace)
    return scipy.linalg.eigh(matrix)


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


def _active_natural_orbitals(casscf, casdm1, mo_active):
    """Diagonalize the active 1-RDM using the reference's symmetry when needed."""
    return _subspace_eigh(casscf, -casdm1, mo_active)


def _resolve_kramers_mode(casscf, symm=None, *, use_diis=False):
    """Resolve explicit/automatic Kramers mode and guard DIIS usage."""
    from socutils.scf import spinor_hf

    if symm is not None:
        symm = str(symm).lower()
        if symm != "kramers":
            raise ValueError("symm must be None or 'kramers'")
    native = isinstance(casscf._scf, spinor_hf.KRHF) or getattr(
        casscf.fcisolver, "kramers_adapter", None
    ) is not None
    if use_diis and native and symm != "kramers":
        raise ValueError(
            "symm='kramers' is required when DIIS is used with a "
            "Kramers-restricted reference or active-space solver"
        )
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
    """Project an orbital generator onto the time-reversal invariant space."""
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

    input_residual = float(
        np.max(abs(generator - time_reverse_one_body(time_reversal, generator)))
    )
    projected = (generator + time_reverse_one_body(time_reversal, generator)) * 0.5
    projected = (projected - projected.T.conj()) * 0.5
    # Remove any redundant within-space elements after projection.  Because
    # every Kramers pair lies wholly inside one orbital space, this mask
    # commutes with time reversal.
    projected = casscf.unpack_uniq_var(casscf.pack_uniq_var(projected))
    projected = (projected + time_reverse_one_body(time_reversal, projected)) * 0.5
    projected = (projected - projected.T.conj()) * 0.5
    output_residual = float(
        np.max(abs(projected - time_reverse_one_body(time_reversal, projected)))
    )
    return projected, {
        "input_generator_residual": input_residual,
        "output_generator_residual": output_residual,
        "projection_change_norm": float(norm(projected - generator)),
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
        "schedule_mode",
        "effective_twosite_to_onesite",
    )
    return {key: info[key] for key in keys if key in info}


# note: the ncas, nelecas, ncore should all be counted as the number of spin orbitals
def gen_g_hop(casscf, mo, casdm1, casdm2, eris):
    if casscf.mo_coeff is None:
        casscf.mo_coeff = mo
    ncas = casscf.ncas
    nelecas = casscf.nelecas
    ncore = casscf.ncore
    nocc = ncas + ncore
    nmo = mo.shape[1]

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
    use_diis=False,
    symm=None,
    diis_space=15,
    diis_start_cycle=3,
    diis_start_gradient=0.02,
    callback=None,
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
    if use_diis and bfgs:
        raise ValueError("Super-CI DIIS and BFGS acceleration are mutually exclusive")
    kramers = _resolve_kramers_mode(mc, symm, use_diis=use_diis)
    orbital_diis = None
    if use_diis:
        orbital_diis = OrbitalDIIS(
            mo,
            mc._scf.get_ovlp(),
            space=diis_space,
            start_cycle=diis_start_cycle,
            start_gradient=diis_start_gradient,
        )
    log.info("Super-CI orbital solver = %s", solver)
    log.info(
        "Super-CI Kramers = %s, orbital DIIS = %s",
        kramers,
        bool(use_diis),
    )
    if solver == "davidson":
        log.info(
            "Super-CI Davidson tolerance = %.3g, maximum space = %d, strict = %s",
            davidson_tol,
            davidson_mmax,
            davidson_strict,
        )

    mci = mc.view(zcasci.CASCI)
    eris, integral_info = _build_eris(mc, mo, cderi=cderi)
    mc.cholesky_diagnostics = dict(integral_info)
    log.info(
        "Super-CI ERI route: representation = %s, source = %s, "
        "container = %s, naux = %s, Cholesky = %s, threshold = %s",
        integral_info["representation"],
        integral_info["source"],
        integral_info["container"],
        integral_info["naux"],
        integral_info["active"],
        integral_info["threshold"],
    )
    mci = zmcscf._fake_h_for_fast_casci(mc, mo, eris)
    log.info("******** Initial Super-CI CASCI ********")
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

    t1m = log.timer("Initializing Super-CI based MCSCF", *cput0)
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

        g, h_diag, hop, sop, precond, mo = gen_g_hop(mc, mo, casdm1, casdm2, eris)
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
        g_unpack = mc.unpack_uniq_var(g)

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
            "ci_solver_converged": ci_converged,
            "ci_solver_diagnostics": _ci_convergence_snapshot(mc.fcisolver),
            "integral_representation": integral_info["representation"],
            "integral_factorized": integral_info["factorized"],
            "integral_source": integral_info["source"],
            "cholesky_active": integral_info["active"],
            "cholesky_naux": integral_info["naux"],
            "superci_metric": dict(mc.superci_metric_diagnostics),
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
            x, e, last_linear_info = davidson(
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
        if use_diis:
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

            diis_result = orbital_diis.update(
                mo,
                mo_new,
                g_unpack,
                cycle=imacro,
                gradient_norm=norm_gorb,
                max_stepsize=max_stepsize,
                step_metric="frobenius",
                projector=project_generator,
            )
            dr = diis_result.generator
            mo_new = diis_result.mo_coeff
            rotation = expmat(dr)
            step_rescaled = bool(
                step_rescaled or diis_result.diagnostics["step_scale"] < 1.0
            )
            history_entry["diis"] = diis_result.diagnostics
        history_entry["applied_orbital_step_norm"] = float(norm(dr))
        history_entry["step_rescaled"] = bool(step_rescaled)
        applied_x = mc.pack_uniq_var(dr)

        norm_rot = np.linalg.norm(rotation - np.eye(nmo, dtype=complex))
        # e_tot, e_cas, fcivec, _, _ = mci.kernel(mo)
        eris, integral_info = _build_eris(mc, mo_new, cderi=cderi)
        t2m = log.timer("update eris", *t2m)
        mci = zmcscf._fake_h_for_fast_casci(mc, mo_new, eris)
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
        e2 = float(0.5 * np.vdot(applied_x, g).real)
        r = de / e2 if abs(e2) > 1e-16 else np.inf

        history_entry["next_total_energy"] = float(np.real(e_tot))
        history_entry["accepted_energy_change"] = float(np.real(de))
        history_entry["predicted_energy_change"] = float(np.real(e2))
        history_entry["next_ci_solver_diagnostics"] = _ci_convergence_snapshot(
            mc.fcisolver
        )
        # while(True):
        #    if de < 0.0:
        #        print('energy lowered, exit iteration')
        #        break
        #    elif (abs(r) < 2.0):
        #        print('normal step')
        #        break
        #    dr = 0.5*dr
        #    rotation = expmat(dr)
        #    mo_new = np.dot(mo, rotation)
        #    eris = zmc_ao2mo._CDERIS(mc, mo_new, cderi=cderi, level=2)
        #    mci = zmcscf._fake_h_for_fast_casci(mc, mo_new, eris)
        #    e_tot, e_cas, fcivec = mci.kernel(mo_new, ci0=None, verbose=verbose)
        #    de = e_tot - e_last
        #    e2 = 0.5 * np.dot(x.T.conj(), g)
        #    r = de / e2
        #    print(f'Energy change {de:.4e}, predicted change {e2:.4e}')
        if False:  # r < -10 and de > 0.0:
            trust_radii *= 0.7
            log.debug("Super-CI rejected orbital step = %s", dr)
            rejected = True
            new_rot = expmat(0.01 * dr)
            mo_new = np.dot(mo, new_rot)
            eris, integral_info = _build_eris(mc, mo_new, cderi=cderi)
            t2m = log.timer("update eris", *t2m)
            mci = zmcscf._fake_h_for_fast_casci(mc, mo, eris)
            log.debug(
                "Super-CI rejected step norms: generator %.6g, rotation %.6g",
                np.linalg.norm(dr),
                np.linalg.norm(rotation),
            )
            # print(rotation[np.where(abs(rotation) > 1e-4)])
            # for dri in dr:
            #    print(dri)
            # for roti in rotation:
            #    print(roti)
            e_tot, e_cas, fcivec = mci.kernel(mo, ci0=None, verbose=verbose)
            casdm1, casdm2 = mci.fcisolver.make_rdm12(fcivec, ncas, mc.nelecas)
            de = e_tot - e_last
            e_last = e_tot
            # continue
            trust_action = "rejected / trust reduced"
        elif de > 0.0:
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
        #'''
        # print(trust_radii)
        # dr[::2,::2] = dr[1::2,1::2]
        # dr[::2,1::2] = dr[1::2,::2]

        rotation = expmat(dr)
        norm_rot = np.linalg.norm(dr)
        nvar = rotation.shape[0]
        # for i in range(nvar):
        #    if abs(rotation[i, i]) > 1.01 or abs(rotation[i, i]) < 0.99:
        #        print(rotation[i, i] > 1.01, rotation[i, i] < 0.99, i, i,
        #              rotation[i, i])
        #    for j in range(i):
        #        if abs(rotation[i, j]) > 0.01:
        #            continue

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
        "diis": bool(use_diis),
        "diis_space": int(diis_space) if use_diis else None,
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
