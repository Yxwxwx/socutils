# SPDX-License-Identifier: GPL-3.0-or-later
"""Fully uncontracted complex-spinor NEVPT2 by native Block2 MPS response.

All eight classes use their full active Fock sectors. The reference and
integrals are fixed; no high-order RDM, IC basis, or time propagation is used.
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass
from tempfile import TemporaryDirectory

import numpy as np
from pyscf import lib
from pyscf.lib import logger

from . import nevpt2_mps_response as response
from . import nevpt2_utils as u


@dataclass(frozen=True)
class UCControls(response.ResponseControls):
    max_bond_dimension: int = 500
    n_sweeps: int = 12
    tol: float = 0.
    linear_threshold: float = 1e-24
    noise: float = 0.
    cutoff: float = 1e-24
    global_residual_tol: float = 1e-8
    diagnostic: bool = False
    strict_cas: bool = False


@contextmanager
def _pt_driver(reference_driver, ncas, nelec, *, scratch=None, stack_memory=1024, n_threads=None):
    """An independent PT driver, sequentially owning Block2's global frame.

    Block2 permits only one active driver per process. Reference tensors must
    be copied before entering; no reference-driver operation is made inside.
    Restore the original frame, allocators and threading even on exceptions.
    """
    from pyblock2.driver.core import DMRGDriver
    if not np.isfinite(stack_memory) or stack_memory <= 0:
        raise ValueError("PT stack_memory must be positive (MB)")
    if n_threads is not None and (isinstance(n_threads, (bool, np.bool_))
                                 or not isinstance(n_threads, (int, np.integer)) or n_threads <= 0):
        raise ValueError("PT n_threads must be a positive integer")
    global_state = reference_driver.bw.b.Global
    names = ("frame", "frame_float", "ialloc", "dalloc", "threading")
    saved = {name: getattr(global_state, name) for name in names}
    if saved["frame"] is not reference_driver.frame:
        raise RuntimeError("the reference driver must own the active Block2 frame before PT")
    try:
        with TemporaryDirectory(prefix="uc-nevpt2-", dir=scratch or lib.param.TMPDIR) as temporary:
            driver = DMRGDriver(
                symm_type=reference_driver.symm_type, scratch=temporary,
                stack_mem=int(stack_memory * 1024**2),
                n_threads=lib.num_threads() if n_threads is None else n_threads,
                n_mkl_threads=1, clean_scratch=True)
            try:
                driver.initialize_system(n_sites=ncas, n_elec=nelec)
                yield driver
            finally:
                driver.finalize()
                driver = None
    finally:
        for name, value in saved.items():
            setattr(global_state, name, value)


def _norm(mps):
    """QR before taking the norm: no difference of large squared norms."""
    mps.canonicalize(0)
    return float(np.sqrt(sum(np.vdot(b.reduced, b.reduced).real
                             for b in mps.tensors[0].blocks)))


def _algebra_mpo(mpo):
    from pyblock2.algebra.io import MPOTools
    while hasattr(mpo, "prim_mpo"):
        mpo = mpo.prim_mpo
    return MPOTools.from_block2(mpo)


def _qc_mpo(driver, h1e, g2e, constant):
    """Block2 owns the integral-to-MPO construction; no custom expressions.

    FastBipartite avoids Conventional's complex-SGF middle-transform issue.
    Integral screening is disabled, including the separate fast cutoff.
    IdentityAddedMPO is applied only at native contraction boundaries so the
    same operator can also be inspected by the uncompressed residual audit.
    """
    from pyblock2.driver.core import MPOAlgorithmTypes
    return driver.get_qc_mpo(
        h1e, g2e, ecore=constant, algo_type=MPOAlgorithmTypes.FastBipartite,
        symmetrize=False, integral_cutoff=0., fast_cutoff=0., cutoff=0.,
        add_ident=False, iprint=0)


@contextmanager
def _qc_problem(driver, active_mps, eris, order, core, virtual, eactive, *, source_scale=1.):
    """Full-chain L=E0-HD and R=(H-E0)*source_scale, as in block2main.

    Nuclear energy cancels from both shifts. The Dyall one-body active block
    is core dressed; only AAAA two-body integrals remain. H uses the ORIGINAL
    bare one-/two-body integrals, not core-dressed source blocks.
    """
    from .spinor_helper import _SpinorERIs
    if not isinstance(eris, _SpinorERIs):
        raise TypeError("native full-H UC requires full _SpinorERIs; omit eris in kernel() "
                        "to transform the complete MO integrals (compact Wick blocks are insufficient)")
    nc, nv, na, nm = eris.ncore, eris.nvirt, eris.ncas, eris.nmo
    permutation = np.r_[np.arange(nc), nc + order, np.arange(eris.nocc, nm)]
    h1, g2 = eris.h1e, eris.pppp
    if not np.array_equal(permutation, np.arange(nm)):
        h1 = h1[np.ix_(permutation, permutation)]
        g2 = g2[np.ix_(permutation, permutation, permutation, permutation)]
    ecore = .5 * np.trace((eris.h1e + eris.h1eff)[:nc, :nc]).real
    saved = driver.__dict__.copy()
    reference = right = left = None
    try:
        driver.initialize_system(n_sites=nm, n_elec=int(saved["target"].n) + nc)
        driver.reorder_idx = None
        reference = response._embed_reference(driver, active_mps, nc, nv, cas_info=True)
        right = _qc_mpo(driver, source_scale * h1, source_scale * g2,
                         -source_scale * (eactive + ecore))
        # Construct the negative Dyall coefficients directly, so no scalar
        # MPO wrapper can be lost when exporting the operator for diagnostics.
        hdyall = np.diag(-np.r_[core, np.zeros(na), virtual]).astype(complex)
        hdyall[nc:eris.nocc, nc:eris.nocc] = -eris.get_h1eff("AA")[np.ix_(order, order)]
        gdyall = np.zeros((nm,) * 4, dtype=complex)
        gdyall[nc:eris.nocc, nc:eris.nocc, nc:eris.nocc, nc:eris.nocc] = (
            -eris.get_chem("AAAA")[np.ix_(order, order, order, order)])
        left = _qc_mpo(driver, hdyall, gdyall, eactive + sum(core))
        del h1, g2, hdyall, gdyall
        yield reference, right, left, nc, nv
    finally:
        try:
            if reference is not None:
                response._release_response_mps(driver, reference)
        finally:
            del right, left
            driver.__dict__.clear()
            driver.__dict__.update(saved)


def _project_class(mps, ncore, nvirt, nelec, key):
    """Exact occupation projector at the two active-space boundary bonds."""
    holes = sum(key.count(x) for x in "ij")
    particles = sum(key.count(x) for x in "rs")
    result = mps.deep_copy()
    if holes > ncore or particles > nvirt:
        result.tensors[0] = result.tensors[0] * 0.
        return result
    for site, number in ((ncore - 1, ncore - holes),
                         (mps.n_sites - nvirt - 1, nelec - particles)):
        if 0 <= site < mps.n_sites - 1:
            for block in result.tensors[site].blocks:
                if block.q_labels[-1].n != number:
                    # Keep the bond structure: algebra inner products do not
                    # support empty intermediate tensors in zero sectors.
                    # deep_copy shares arrays, so replace rather than mutate.
                    block.reduced = np.zeros_like(block.reduced)
    return result


def _cas_part(mps, ncore, nvirt):
    """P_CAS = occupied core times identity on CAS times empty virtuals."""
    result = mps.deep_copy()
    for i, tensor in enumerate(result.tensors):
        occupation = 1 if i < ncore else (0 if i >= mps.n_sites - nvirt else None)
        if occupation is not None:
            # Omit exactly forbidden occupation blocks, rather than retaining
            # their large zero arrays through subsequent QR/direct sums.
            tensor.blocks = [block for block in tensor.blocks
                             if block.q_labels[0 if i == 0 else 1].n == occupation]
    return result


def _external_part(mps, ncore, nvirt):
    """Exact tensor-network Q application; no truncation or root subtraction."""
    cas = _cas_part(mps, ncore, nvirt)
    # Remove redundant product-state bonds BEFORE the direct sum. This is
    # untruncated QR, not a fitted source or an approximate CAS projector.
    # In particular, occupied-core/empty-virtual CAS tails need only rank 1.
    cas.canonicalize(0)
    return mps - cas


def _truncate_external(mps, bond, cutoff, ncore, nvirt):
    """A constrained truncation, not a cleanup of the final response."""
    mps.compress(k=bond, cutoff=cutoff)
    raw_leak = _norm(_cas_part(mps, ncore, nvirt))
    kept = _external_part(mps, ncore, nvirt)
    norm = _norm(kept)
    leak = _norm(_cas_part(kept, ncore, nvirt))
    relative = leak / norm if norm else leak
    if relative > 1e-10 or not np.isfinite(relative):
        raise RuntimeError(f"post-truncation CAS constraint failed: {relative:.3e}")
    return kept, dict(raw_truncation_cas_norm=raw_leak,
                     retained_cas_relative_norm=relative,
                     retained_bond_dimension=max(sum(x.values()) for x in kept.get_bond_dims()))


def _cas_boundary_factors(mps, ncore, nvirt, site):
    """QR factors of P_CAS times the left/right renormalized basis maps.

    Only O(M^2) matrices are stored. Rank decisions use singular values of
    these maps, NOT eigenvalues of their squared Gram matrices. There is no
    enumeration of CAS determinants or of reference roots.
    """
    total = sum(q.n for q in mps.tensors[-1].blocks[0].q_labels)

    def propagate(indices, forward):
        factors = {0 if forward else total: np.ones((1, 1), complex)}
        for i in indices:
            collected = {}
            wanted = 1 if i < ncore else (0 if i >= mps.n_sites - nvirt else None)
            for block in mps.tensors[i].blocks:
                labels, a = block.q_labels, block.reduced
                if i == 0:
                    ql, qm, qr = 0, labels[0].n, labels[1].n
                    a = a[None, ...]
                elif i == mps.n_sites - 1:
                    ql, qm, qr = labels[0].n, labels[1].n, total
                    a = a[..., None]
                else:
                    ql, qm, qr = (q.n for q in labels)
                if wanted is not None and qm != wanted:
                    continue
                old, new = (ql, qr) if forward else (qr, ql)
                if old in factors:
                    for p in range(a.shape[1]):
                        mapped = factors[old] @ (a[:, p, :] if forward else a[:, p, :].T)
                        collected.setdefault(new, []).append(mapped)
            factors = {q: np.linalg.qr(np.concatenate(parts), mode="r")
                       for q, parts in collected.items()}
        return factors

    return (propagate(range(site), True),
            propagate(range(mps.n_sites - 1, site + 1, -1), False))


def _cas_aligned_gauge(mps, ncore, nvirt, site):
    """Unitary bond gauges make the local CAS constraint a coordinate mask.

    A dense floating-point Q repeatedly applied in Krylov iteration can
    accumulate forbidden directions. In this gauge those coordinates are
    exactly zero throughout native Linear's vector algebra. This changes
    neither the physical state nor any orbital/integral.
    """
    mps.canonicalize(site)
    left, right = _cas_boundary_factors(mps, ncore, nvirt, site)
    ranks = [{}, {}]
    for cut, factors, forward in ((site, left, True), (site + 2, right, False)):
        rotations = {}
        for q, factor in factors.items():
            _, singular, vh = np.linalg.svd(factor, full_matrices=True)
            threshold = np.finfo(float).eps * max(factor.shape) * max(1., singular[0]) * 8 if len(singular) else 0.
            ranks[0 if forward else 1][q] = int(np.count_nonzero(singular > threshold))
            rotations[q] = vh.conj().T
        if not 0 < cut < mps.n_sites:
            continue
        for block in mps.tensors[cut - 1].blocks:
            if block.q_labels[-1].n in rotations:
                v = rotations[block.q_labels[-1].n]
                block.reduced = np.tensordot(block.reduced, v if forward else v.conj(), axes=(-1, 0))
        for block in mps.tensors[cut].blocks:
            if block.q_labels[0].n in rotations:
                v = rotations[block.q_labels[0].n]
                block.reduced = np.tensordot(v.conj().T if forward else v.T, block.reduced, axes=(1, 0))
    return ranks


def _cas_local_projector(driver, state, site, factors, ncore, nvirt, ranks):
    """Orthogonal projector onto ker(P_CAS V) for a two-site native update.

    V is the current renormalized-basis embedding. V† P_CAS V is generally
    NOT a projector. Its nullspace is computed from the boundary maps above.
    The returned Q acts on native packed wavefunctions, without a dense local
    Hamiltonian or a dense many-electron projector.
    """
    info, target = state.info, int(driver.target.n)
    support_cache = {}
    mapping_error = 0.

    def support(factor, dim, rank):
        nonlocal mapping_error
        if factor is None:
            assert rank == 0
            return np.zeros((dim, 0), complex)
        # Retain the rank selected during alignment. Re-ranking a nearly
        # null direction after another QR can make inconsistent decisions.
        error = float(np.linalg.norm(factor[:, rank:]))
        mapping_error = max(mapping_error, error)
        if error > 1e-11:
            raise RuntimeError(f"CAS-aligned basis leaks at site {site}: {error:.3e}")
        return np.eye(dim, dtype=complex)[:, :rank]

    def fused_support(left):
        i = site if left else site + 1
        info.load_left_dims(site) if left else info.load_right_dims(site + 2)
        boundary = info.left_dims[site] if left else info.right_dims[site + 2]
        physical = info.basis[i]
        first, second = (boundary, physical) if left else (physical, boundary)
        fci = info.left_dims_fci[site + 1] if left else info.right_dims_fci[site + 1]
        fused = physical.__class__.tensor_product_ref(first, second, fci)
        connection = physical.__class__.get_connection_info(first, second, fused)
        result = {}
        wanted = 1 if i < ncore else (0 if i >= state.n_sites - nvirt else None)
        for qindex in range(fused.n):
            pieces, offset = [], 0
            for j in range(connection.acc_n_states[qindex], connection.acc_n_states[qindex + 1]):
                ja, jb = connection.ij_indices[j]
                bound_index, phys_index = (ja, jb) if left else (jb, ja)
                dim = int(boundary.n_states[bound_index])
                assert physical.n_states[phys_index] == 1, "SGF spinor site must have dimension one per occupation"
                number = int(boundary.quanta[bound_index].n)
                allowed = wanted is None or physical.quanta[phys_index].n == wanted
                cache_key = left, number
                if allowed:
                    if cache_key not in support_cache:
                        boundary_number = number if left else target - number
                        support_cache[cache_key] = support(factors[0 if left else 1].get(
                            boundary_number), dim, ranks[0 if left else 1].get(boundary_number, 0))
                    pieces.append((offset, support_cache[cache_key]))
                offset += dim
            result[int(fused.quanta[qindex].n)] = np.zeros(
                (offset, sum(x.shape[1] for _, x in pieces)), complex)
            col = 0
            for row, x in pieces:
                result[int(fused.quanta[qindex].n)][row:row + x.shape[0], col:col + x.shape[1]] = x
                col += x.shape[1]
        fused.deallocate()
        return result

    ls, rs = fused_support(True), fused_support(False)
    tensor = state.tensors[site]
    blocks, offset = [], 0
    for j in range(tensor.info.n):
        q = tensor.info.quanta[j]
        left = ls[int(q.get_bra(tensor.info.delta_quantum).n)]
        right = rs[int((-q.get_ket()).n)]
        shape = np.asarray(tensor[j]).shape
        assert shape == (left.shape[0], right.shape[0])
        blocks.append((offset, shape, left, right))
        offset += int(np.prod(shape))

    def project(array):
        result = np.array(array, copy=True)
        flat = result.reshape(-1)
        assert len(flat) == offset
        for start, shape, left, right in blocks:
            x = flat[start:start + int(np.prod(shape))].reshape(shape)
            if left.shape[1] and right.shape[1]:
                # Exact coordinate zeros, not a subtraction of nearly equal
                # floating-point vectors in a singular Krylov problem.
                x[np.ix_(np.any(left != 0, axis=1), np.any(right != 0, axis=1))] = 0.
        return result

    project.cas_basis_mapping_error = mapping_error
    return project


def _linear_response(driver, state, reference, dyall, source_mpo, controls, bond, iprint):
    """block2main-style native Linear, with the source MPO and reference ket.

    Retain the native Automatic solver and density-matrix decomposition.
    Supply the ENTIRE
    threshold schedule: unspecified sweeps default to 1e-10, not its last item.
    """
    bw = driver.bw
    driver.align_mps_center(state, reference)
    left = bw.bs.MovingEnvironment(dyall, state, state, "UC-LINEAR-L")
    right = bw.bs.MovingEnvironment(source_mpo, state, reference, "UC-LINEAR-R")
    try:
        left.init_environments(False)
        left.delayed_contraction = bw.b.OpNamesSet.normal_ops()
        left.cached_contraction = False
        right.init_environments(False)
        linear = bw.bs.Linear(left, right, bw.b.VectorUBond([bond]),
                              bw.b.VectorUBond([reference.info.bond_dim + 400]),
                              bw.VectorFP([controls.noise]))
        linear.noise_type = bw.b.NoiseTypes.ReducedPerturbativeCollected
        linear.cutoff = controls.cutoff
        linear.linear_conv_thrds = bw.VectorFP([controls.linear_threshold] * controls.n_sweeps)
        linear.linear_rel_conv_thrd = controls.linear_rel_conv_thrd
        linear.linear_max_iter = controls.linear_max_iter + 100
        linear.linear_soft_max_iter = controls.linear_max_iter
        linear.iprint = iprint
        value = linear.solve(controls.n_sweeps, reference.center == 0, controls.tol)
        return value, len(linear.targets), list(linear.discarded_weights)
    finally:
        if driver.clean_scratch:
            left.remove_partition_files()
            right.remove_partition_files()


def _mrci_from_algebra(driver, mps, ncore, nvirt, center):
    """Import tensors while retaining the native nevpt2sd space descriptor."""
    from pyblock2.algebra.io import MPSTools
    native = MPSTools.to_block2(mps, driver.basis, center=center,
                                tag=response._response_tag())
    old = native.info
    old.load_mutable()
    info = driver.bw.brs.MRCIMPSInfo(driver.n_sites, ncore, nvirt, 2,
                                    driver.vacuum, driver.target, driver.ghamil.basis)
    info.tag, info.bond_dim = old.tag, old.bond_dim
    info.left_dims, info.right_dims = old.left_dims, old.right_dims
    native.info = info
    info.save_mutable()
    native.save_data()
    info.save_data(str(response.Path(driver.scratch) / f"{info.tag}-mps_info.bin"))
    native.dot = 2
    return native


def _constrained_linear_response(driver, state, reference, dyall, source_mpo,
                                 controls, bond, ncore, nvirt, constant):
    """Native two-site Linear updates restricted to ker(P_CAS V).

    After EVERY native truncation, apply Q exactly and rebuild the environments.
    This explicit benchmark path trades speed for an auditable invariant;
    projection may increase the stored bond dimension beyond the SVD cap.
    It does not replace Block2's local solver or truncation algorithm.
    """
    from pyblock2.algebra.io import MPSTools
    if controls.noise != 0:
        raise NotImplementedError("CAS-constrained Linear currently requires noise=0")
    bw = driver.bw
    reference_python = response._active_mps(driver, reference)
    y = _external_part(response._active_mps(driver, state), ncore, nvirt)
    history, discarded = [], []
    for sweep in range(controls.n_sweeps):
        forward = sweep % 2 == 0
        sites = range(driver.n_sites - 1) if forward else range(driver.n_sites - 2, -1, -1)
        for site in sites:
            ket = bra = left = right = None
            try:
                ranks = _cas_aligned_gauge(y, ncore, nvirt, site)
                bra = _mrci_from_algebra(driver, y, ncore, nvirt, site)
                # Import canonicalizes y in place; use precisely this gauge.
                factors = _cas_boundary_factors(y, ncore, nvirt, site)
                ket = MPSTools.to_block2(reference_python.deep_copy(), driver.basis, center=site,
                                         tag=response._response_tag())
                ket.dot = 2
                left = bw.bs.MovingEnvironment(dyall, bra, bra, "UC-Q-L")
                right = bw.bs.MovingEnvironment(source_mpo, bra, ket, "UC-Q-R")
                for me in (left, right):
                    me.init_environments(False)
                    me.prepare(0, driver.n_sites)
                left.delayed_contraction = bw.b.OpNamesSet.normal_ops()
                left.cached_contraction = False
                linear = bw.bs.Linear(left, right, bw.b.VectorUBond([bond]),
                                       bw.b.VectorUBond([ket.info.bond_dim + 400]), bw.VectorFP([0.]))
                linear.noise_type = bw.b.NoiseTypes.ReducedPerturbativeCollected
                # A diagonal preconditioner need not preserve the Q subspace.
                linear.linear_use_precondition = False
                linear.linear_max_iter = controls.linear_max_iter + 100
                linear.linear_soft_max_iter = controls.linear_max_iter
                linear.linear_rel_conv_thrd = controls.linear_rel_conv_thrd
                linear.cutoff, linear.iprint = controls.cutoff, 0
                project = None
                domain_defect = 0.

                def right_kernel(beta, hop, a, b, xs):
                    nonlocal project
                    project = _cas_local_projector(driver, bra, site, factors, ncore, nvirt, ranks)
                    # Constrain the initial guess as well as the local RHS.
                    tensor = bra.tensors[site]
                    initial = np.concatenate([np.asarray(tensor[j]).ravel()
                                              for j in range(tensor.info.n)])
                    initial = project(initial)
                    offset = 0
                    for j in range(tensor.info.n):
                        view = np.asarray(tensor[j])
                        view[:] = initial[offset:offset + view.size].reshape(view.shape)
                        offset += view.size
                    temporary = np.zeros_like(b)
                    hop(a, temporary, beta)
                    b += project(temporary)

                def left_kernel(beta, hop, a, b, xs):
                    nonlocal domain_defect
                    qa = project(a)
                    norm = np.linalg.norm(a)
                    if norm:
                        domain_defect = max(domain_defect, float(np.linalg.norm(a - qa) / norm))
                    temporary = np.zeros_like(b)
                    hop(qa, temporary, beta)
                    # Include the scalar shift INSIDE Q, not as an ambient
                    # constant times I added by the native iterative solver.
                    temporary += beta * constant * qa
                    b += project(temporary)

                rk, lk = driver.make_kernel(right_kernel), driver.make_kernel(left_kernel)
                linear.reff_kernel, linear.leff_kernel = rk, lk
                result = linear.blocking(site, forward, bond, ket.info.bond_dim + 400,
                                          0., controls.linear_threshold)
                discarded.append(float(result.error))
                # A single blocking call leaves two separate tensors, with
                # the new one-site center prepared by native propagate_wfn.
                bra.dot, bra.center = 1, site + 1 if forward else site
                fused = ("S" if bra.center == driver.n_sites - 1 else "K") if forward else (
                    "K" if bra.center == 0 else "S")
                bra.canonical_form = "L" * bra.center + fused + "R" * (driver.n_sites - bra.center - 1)
                bra.save_data()
                raw = MPSTools.from_block2(bra)
                # Bound all bonds before the exact Q lift (at most doubles M).
                # Neither this truncated candidate nor the native update output
                # is reused as an iterate before the occupation constraint.
                y, truncation = _truncate_external(raw, bond, controls.cutoff, ncore, nvirt)
                history.append(dict(sweep=sweep, site=site,
                                    local_krylov_cas_relative_norm=domain_defect,
                                    cas_basis_mapping_error=project.cas_basis_mapping_error, **truncation))
            finally:
                for me in (left, right):
                    if me is not None and driver.clean_scratch:
                        me.remove_partition_files()
                for owned in (bra, ket):
                    if owned is not None:
                        response._release_response_mps(driver, owned)
    return y, history, discarded


def _solve_native(driver, active_mps, eris, order, core, virtual,
                  eactive, controls, bond, iprint):
    """block2main-style whole response; diagnostics never supply the RHS.

    R is the complete H-E0, not only its action restricted to CAS kets.
    L=E0-H_D and the unknown is Psi1. No Q is inserted into the solve.
    """
    from pyblock2.algebra.io import MPSTools
    if eris.ncore == 0 and eris.nvirt == 0:
        return 0., dict(zero_source=True, empty_external_space=True, source_norm2=0.,
                        global_relative_residual=0., converged=True)
    # Only determine the scalar scale; do not retain copies of all eight
    # source integral blocks throughout the full-H MPO construction/solve.
    scale = max((u._maximum_abs(t) for key in u.SUBSPACE_ORDER
                 for _, t, _ in response._source_tensors(eris, order, key)), default=0.)
    if scale == 0:
        # No external coupling does not imply a zero CAS residual. Only an
        # identically zero full operator can be skipped without measuring it.
        scale = max(u._maximum_abs(eris.h1e), u._maximum_abs(eris.pppp), abs(eactive))
        if scale == 0:
            return 0., dict(zero_source=True, source_norm2=0.,
                            global_relative_residual=0., converged=True)
    reference_norm = _norm(active_mps.deep_copy())
    mpo_start = time.perf_counter()
    if iprint:
        print("UC building full H and Dyall MPOs with get_qc_mpo/FastBipartite", flush=True)
    with _qc_problem(
        driver, active_mps, eris, order, core, virtual, eactive,
        source_scale=1. / (scale * reference_norm),
    ) as (reference, source, dyall, nc, nv):
        mpo_seconds = time.perf_counter() - mpo_start
        if iprint:
            print(f"UC full-chain MPOs ready in {mpo_seconds:.3f} s; starting native Linear.solve", flush=True)
        state = None
        try:
            state = response._nevpt_mps(driver, nc, nv, "all", bond)
            right = driver.bw.bs.IdentityAddedMPO(source)
            left = driver.bw.bs.IdentityAddedMPO(dyall)
            reported, sweeps, discarded = _linear_response(
                driver, state, reference, left, right, controls, bond, iprint)
            if not np.isfinite(reported):
                raise RuntimeError("nonfinite native whole-chain response")
            if iprint:
                print("UC native Linear.solve finished; measuring energies and CAS diagnostics", flush=True)

            # Only response tensors are read here, never R|reference>.
            # The right environment continues to contract R and reference.
            y = response._active_mps(driver, state)
            ynorm = _norm(y)
            ref = response._active_mps(driver, reference) / reference_norm
            reference_overlap = abs(ref.conj() @ y)
            cas_y_norm = _norm(_cas_part(y, nc, nv))
            bx = driver.expectation(state, right, reference)
            xx = driver.expectation(state, left, state)
            energy = float(scale**2 * (-xx.real + 2 * bx.real))
            classes = {}
            for key in u.SUBSPACE_ORDER:
                yi = _project_class(y, nc, nv, int(driver.target.n), key)
                norm = _norm(yi)
                part = None
                try:
                    if norm:
                        part = MPSTools.to_block2(yi, driver.basis, center=0,
                                                  tag=response._response_tag())
                        bxi = driver.expectation(part, right, reference)
                        xxi = driver.expectation(part, left, part)
                    else:
                        bxi = xxi = 0j
                    classes[key] = dict(
                        hylleraas_energy=float(scale**2 * (-xxi.real + 2 * bxi.real)),
                        projected_energy=float(scale**2 * bxi.real),
                        overlap=[float(scale**2 * bxi.real), float(scale**2 * bxi.imag)],
                        quadratic=[float(scale**2 * xxi.real), float(scale**2 * xxi.imag)],
                        source_norm2=None, residual_norm2=None,
                        global_relative_residual=None, converged=False)
                finally:
                    if part is not None:
                        response._release_response_mps(driver, part)

            entry = dict(
                classes=classes, source_norm2=None, global_relative_residual=None,
                source_representation="on-the-fly (H-E0) MPO on CAS reference",
                source_scaling="integral coefficient scale; RHS norm not assumed unity",
                source_relative_compression_error=0.,
                projected_energy=float(scale**2 * bx.real), hylleraas_energy=energy,
                native_reported_overlap=[float(np.real(reported) * scale**2),
                                         float(np.imag(reported) * scale**2)],
                overlap=[float(scale**2 * bx.real), float(scale**2 * bx.imag)],
                quadratic=[float(scale**2 * xx.real), float(scale**2 * xx.imag)],
                reference_zero_mode_leakage=float(reference_overlap / ynorm) if ynorm else 0.,
                cas_response_relative_norm=float(cas_y_norm / ynorm) if ynorm else 0.,
                cas_purity_certified=cas_y_norm <= controls.global_residual_tol * ynorm,
                reference_eigenstate_certified=None,
                cas_source_norm=None, cas_source_relative_norm=None,
                cas_residual_squared_raw=None,
                cas_residual_squared_resolution_estimate=None,
                cas_residual_method="not measured; enable diagnostic=True",
                response_bond_dimension=max(sum(x.values()) for x in y.get_bond_dims()),
                actual_sweeps=sweeps, requested_sweeps=controls.n_sweeps,
                discarded_weights=discarded, controls=controls.__dict__,
                solver="pyblock2.Linear.solve/Automatic",
                mpo_construction="DMRGDriver.get_qc_mpo/FastBipartite; full H and spinor Dyall",
                mpo_wall_seconds=mpo_seconds,
                equation="(E0-HD) Psi1 = (H-E0) Psi0; scalar RHS scaling restored in energies",
                space_restriction="MRCIMPSInfo(ci_order=2), CAS included",
                projector="none; block2main-style unconstrained response",
                positive_definiteness="not certified", converged=False,
                reason="global residual not measured")
            entry["class_sum_hylleraas_energy"] = sum(d["hylleraas_energy"] for d in classes.values())
            entry["cas_hylleraas_energy"] = energy - entry["class_sum_hylleraas_energy"]
            if iprint:
                print(f"UC energy estimators: Hylleraas={energy:.14f} "
                      f"projected={entry['projected_energy']:.14f} "
                      f"CAS={entry['cas_hylleraas_energy']:.3e}; "
                      "global residual not yet measured", flush=True)
            if controls.diagnostic:
                if iprint:
                    print("UC opt-in global residual audit: applying actual left/right MPOs", flush=True)
                # Explicit opt-in audit AFTER the native solve, never its input.
                # These are exactly the left/right operators passed to Linear,
                # including the CAS RHS residual; no Q or final cleanup.
                b = _algebra_mpo(source) @ response._active_mps(driver, reference)
                ay = _algebra_mpo(dyall) @ y
                sn = _norm(b)
                if iprint:
                    print(f"UC audit: actual RHS norm={scale * sn:.6e}; "
                          "evaluating residual norm", flush=True)
                residual_mps = ay - b
                residual = _norm(residual_mps)
                rho = residual / sn if sn else (0. if residual == 0. else float("inf"))
                if iprint:
                    print(f"UC audit: global relative residual={rho:.6e}; "
                          "checking CAS and eight sectors", flush=True)
                cas_norm = _norm(_cas_part(b, nc, nv))
                entry.update(source_norm2=float(scale**2 * sn**2),
                             global_relative_residual=float(rho),
                             cas_source_norm=float(scale * cas_norm),
                             cas_source_relative_norm=float(cas_norm / sn) if sn else None,
                             reference_eigenstate_certified=cas_norm <= controls.global_residual_tol * sn,
                             cas_residual_method="explicit uncompressed audit of actual RHS",
                             converged=rho <= controls.global_residual_tol,
                             reason=None if rho <= controls.global_residual_tol else "global residual exceeds target")
                for key, d in classes.items():
                    bi = _project_class(b, nc, nv, int(driver.target.n), key)
                    ri = _project_class(residual_mps, nc, nv, int(driver.target.n), key)
                    bn, rn = _norm(bi), _norm(ri)
                    d.update(source_norm2=float(scale**2 * bn**2),
                             residual_norm2=float(scale**2 * rn**2),
                             global_relative_residual=(rn / bn if bn else (0. if rn == 0. else float("inf"))),
                             converged=rn <= controls.global_residual_tol * bn if bn else rn == 0.)
                if not all(d["converged"] for d in classes.values()):
                    entry.update(converged=False, reason="class residual exceeds target")
            return energy, entry
        finally:
            if state is not None:
                response._release_response_mps(driver, state)


def _solve_class(driver, active_mps, eris, order, key, core, virtual,
                 eactive, controls, bond, iprint=0):
    """One normalized source and native response solve, shared by all classes."""
    if key == "all" and not controls.strict_cas:
        return _solve_native(driver, active_mps, eris, order, core, virtual,
                             eactive, controls, bond, iprint)
    # Scale the operator before any source compression. Preserve this physical
    # amplitude separately; a weak nonzero source is never screened by norm.
    tensors = response._source_tensors(eris, order, key)
    coefficient_scale = max((u._maximum_abs(t) for _, t, _ in tensors), default=0.)
    if coefficient_scale == 0:
        return 0., {"zero_source": True, "source_norm2": 0.,
                    "global_relative_residual": 0., "converged": True}
    with response._class_problem(
        driver, active_mps, eris, order, key, core, virtual, eactive,
        source_scale=1. / coefficient_scale,
    ) as (reference, source_mpo, dyall_mpo, ncore, nvirt):
        state = None
        try:
            reference_python = response._active_mps(driver, reference)
            reference_norm = _norm(reference_python)
            # Exact MPO application in tensor-network form; never CI expansion.
            # This is deliberately a correctness-first benchmark: exact product
            # bonds can be large. This product is used for normalization and
            # diagnostics only: the solver applies source_mpo to the reference
            # directly, as in block2main, with no fitted source MPS.
            b = _algebra_mpo(source_mpo) @ (reference_python / reference_norm)
            source_cas_norm = _norm(_cas_part(b, ncore, nvirt)) if key == "all" else None
            if key == "all":
                b = _external_part(b, ncore, nvirt)
            source_norm = _norm(b)
            if source_norm == 0:
                return 0., {"zero_source": True, "source_norm2": 0.,
                            "global_relative_residual": 0., "converged": True}
            b = b / source_norm
            physical_norm = coefficient_scale * source_norm
            native_source = driver.bw.bs.IdentityAddedMPO(
                source_mpo * (1. / (reference_norm * source_norm)))
            state = response._nevpt_mps(driver, ncore, nvirt, key, bond)
            native_dyall = driver.bw.bs.IdentityAddedMPO(dyall_mpo)
            constraint_history = None
            if key == "all":
                constant = native_dyall.const_e
                native_dyall.const_e = 0.
                try:
                    y, constraint_history, discarded = _constrained_linear_response(
                        driver, state, reference, native_dyall, native_source,
                        controls, bond, ncore, nvirt, constant)
                finally:
                    native_dyall.const_e = constant
                actual_sweeps = controls.n_sweeps
            else:
                reported, actual_sweeps, discarded = _linear_response(
                    driver, state, reference, native_dyall, native_source, controls, bond, iprint)
                if not np.isfinite(reported):
                    raise RuntimeError(f"nonfinite {key} response")
                y = response._active_mps(driver, state)
            overlap = b.conj() @ y
            ay = _algebra_mpo(dyall_mpo) @ y if key == "all" else None
            quadratic = y.conj() @ ay if ay is not None else driver.expectation(state, native_dyall, state)
            projected = -physical_norm**2 * overlap.real
            functional = physical_norm**2 * (quadratic.real - 2 * overlap.real)
            rho = None
            if controls.diagnostic:
                # Construct the residual itself, using the UNCOMPRESSED source.
                if ay is None:
                    ay = _algebra_mpo(dyall_mpo) @ y
                residual = ay - b
                rho = _norm(residual)
                if not np.isfinite(rho):
                    raise RuntimeError(f"nonfinite global {key} residual")
            converged = rho is not None and rho <= controls.global_residual_tol
            entry = dict(
                source_norm2=physical_norm**2,
                source_relative_compression_error=0.,
                source_representation="direct source MPO acting on reference MPS",
                global_relative_residual=rho,
                projected_energy=float(projected), hylleraas_energy=float(functional),
                overlap=[float(overlap.real * physical_norm**2),
                         float(overlap.imag * physical_norm**2)],
                quadratic=[float(quadratic.real * physical_norm**2),
                           float(quadratic.imag * physical_norm**2)],
                converged=converged,
                reason=None if converged else ("global residual not measured" if rho is None
                                              else "global residual exceeds target"),
                response_bond_dimension=max(sum(x.values()) for x in y.get_bond_dims()),
                actual_sweeps=actual_sweeps, discarded_weights=discarded,
                requested_sweeps=controls.n_sweeps, solver="pyblock2.Linear/Automatic",
                positive_definiteness="not certified", controls=controls.__dict__,
            )
            if key == "all":
                if ay is None:
                    ay = _algebra_mpo(dyall_mpo) @ y
                classes = {}
                for subspace in u.SUBSPACE_ORDER:
                    def project(mps):
                        return _project_class(mps, ncore, nvirt, int(driver.target.n), subspace)
                    bi, yi, ayi = project(b), project(y), project(ay)
                    sn = _norm(bi)
                    bx, xx = bi.conj() @ yi, yi.conj() @ ayi
                    ri = _norm(ayi - bi) if controls.diagnostic else None
                    classes[subspace] = dict(
                        source_norm2=physical_norm**2 * sn**2,
                        residual_norm2=None if ri is None else physical_norm**2 * ri**2,
                        global_relative_residual=(None if ri is None else (ri / sn if sn else (0. if ri == 0. else float("inf")))),
                        projected_energy=float(-physical_norm**2 * bx.real),
                        hylleraas_energy=float(physical_norm**2 * (xx.real - 2 * bx.real)),
                        overlap=[float(physical_norm**2 * bx.real), float(physical_norm**2 * bx.imag)],
                        quadratic=[float(physical_norm**2 * xx.real), float(physical_norm**2 * xx.imag)],
                        zero_source=sn == 0.,
                        converged=ri is not None and (ri <= controls.global_residual_tol * sn if sn else ri == 0.),
                    )
                cas = _cas_part(y, ncore, nvirt)
                cas_norm = _norm(cas)
                entry.update(classes=classes, cas_response_norm=physical_norm * cas_norm,
                             cas_response_norm_over_source=cas_norm,
                             space_restriction="MRCIMPSInfo(ci_order=2)",
                             projector="ker(P_CAS V) local solves; Q after each truncated update",
                             cas_source_norm_before_projection=coefficient_scale * source_cas_norm,
                             cas_source_norm=physical_norm * _norm(_cas_part(b, ncore, nvirt)),
                             constraint_history=constraint_history)
                entry["class_sum_hylleraas_energy"] = sum(d["hylleraas_energy"] for d in classes.values())
                if not all(d["converged"] for d in classes.values()):
                    entry.update(converged=False, reason=("class residual exceeds target"
                                 if controls.diagnostic else "global residual not measured"))
            return float(functional), entry
        finally:
            if state is not None:
                response._release_response_mps(driver, state)


def evaluate_uc(mc, eris, pdms, core_energy, virtual_energy, *, root=0, options=None,
                class_resolved=False, scratch=None, stack_memory=1024, n_threads=None,
                response_mode="full_chain"):
    """Full nevpt2sd response; optional eight-sector independent cross-check."""
    from pyblock2.driver.core import SymmetryTypes
    controls = UCControls(**({} if options is None else options))
    if response_mode not in ("full_chain", "external_tuples"):
        raise ValueError("response_mode must be 'full_chain' or 'external_tuples'")
    if response_mode == "external_tuples" and (class_resolved or controls.strict_cas):
        raise ValueError("external_tuples already excludes CAS; class_resolved/strict_cas are full-chain checks")
    for name in ("strict_cas", "diagnostic"):
        if not isinstance(getattr(controls, name), (bool, np.bool_)):
            raise TypeError(f"{name} must be a boolean")
    _, bond, _ = response._controls(mc.fcisolver, {
        k: v for k, v in controls.__dict__.items()
        if k in response.ResponseControls.__dataclass_fields__})
    u._finite_nonnegative(controls.global_residual_tol, name="global_residual_tol")
    driver, reference = u._root_ket(mc.fcisolver, root)
    if not (SymmetryTypes.SGF in driver.symm_type and SymmetryTypes.CPX in driver.symm_type):
        raise TypeError("full UC requires a native SGFCPX MPS")
    if driver.mpi is not None or np.any(np.asarray(driver.orb_sym) != 0):
        raise NotImplementedError("full UC currently requires threaded C1 SGFCPX")
    if len(pdms) != 2:
        raise ValueError("full UC uses only raw 1/2-RDMs")
    order = np.arange(eris.ncas) if driver.reorder_idx is None else np.asarray(driver.reorder_idx)
    if not np.array_equal(np.sort(order), np.arange(eris.ncas)):
        raise ValueError("invalid retained MPS orbital permutation")
    eactive = response._real_scalar(
        np.einsum("pq,pq", eris.get_h1eff("AA"), pdms[0])
        + .5 * np.einsum("pqrs,pqsr", eris.get_phys("AAAA"), pdms[1]),
        name="active reference energy")[0]
    active_mps = response._active_mps(driver, reference)
    nelec = int(driver.target.n)
    with _pt_driver(driver, eris.ncas, nelec, scratch=scratch,
                    stack_memory=stack_memory, n_threads=n_threads) as pt_driver:
        if response_mode == "external_tuples":
            from .nevpt2_external_response import evaluate_external
            return evaluate_external(pt_driver, active_mps, eris, order,
                                     np.asarray(core_energy), np.asarray(virtual_energy),
                                     eactive, controls, bond, mc)
        energies, diagnostics = {}, {}
        if not class_resolved:
            energy, entry = _solve_class(pt_driver, active_mps, eris, order, "all",
                                         np.asarray(core_energy), np.asarray(virtual_energy),
                                         eactive, controls, bond,
                                         iprint=1 if getattr(mc, "verbose", 0) >= logger.INFO else 0)
            if entry.get("zero_source"):
                return ({key: 0. for key in u.SUBSPACE_ORDER},
                        {key: dict(entry) for key in u.SUBSPACE_ORDER})
            entry.update(reference_requested_bond_dimension=getattr(mc.fcisolver, "max_bond_dimension", None),
                         reference_actual_bond_dimension=max(sum(x.values()) for x in active_mps.get_bond_dims()),
                         response_requested_bond_dimension=bond,
                         driver_lifecycle="independent PT driver and scratch; reference frame restored")
            for key, d in entry["classes"].items():
                energies[key] = d["hylleraas_energy"]
                diagnostics[key] = {**d, "converged": d["converged"] and entry["converged"],
                                    "joint_response": {k: v for k, v in entry.items() if k != "classes"}}
            return energies, diagnostics
        for key in u.SUBSPACE_ORDER:
            start = time.perf_counter()
            holes = sum(key.count(x) for x in "ij")
            particles = sum(key.count(x) for x in "rs")
            if (holes > eris.ncore or particles > eris.nvirt
                    or not 0 <= nelec + holes - particles <= eris.ncas):
                energies[key], entry = 0., dict(empty_sector=True, source_norm2=0.,
                                               global_relative_residual=0., converged=True)
            else:
                energies[key], entry = _solve_class(
                    pt_driver, active_mps, eris, order, key, np.asarray(core_energy),
                    np.asarray(virtual_energy), eactive, controls, bond)
            entry.update(wall_seconds=time.perf_counter() - start,
                         active_electrons=nelec + holes - particles,
                         inactive_holes=holes, virtual_electrons=particles)
            diagnostics[key] = entry
            logger.note(mc, "E(X2CUCNEVPT2-%s) = %.14f  residual = %s",
                        key, energies[key], entry["global_relative_residual"])
        return energies, diagnostics


class X2CUCNEVPT2(lib.StreamObject):
    """State-specific eight-class Sharma--Chan MPS-PT with a spinor Dyall H0.

    ``mps_response_options`` uses the existing hybrid solver option names.
    Energies are the measured Hylleraas functional. ``converged`` requires
    an actual global residual measurement; local Linear stopping is not a
    global certificate. The native default does not constrain CAS or certify
    external-space purity. strict_cas=True selects the separate validation
    path. diagnostic=True opts into explicit post-solve residual MPS audits.
    response_mode="external_tuples" instead solves all fixed external
    occupation blocks on the active chain, excluding (0,0) by construction.
    """

    def __init__(self, mc, frozen=0):
        if u._has_frozen_orbitals(frozen) or u._has_frozen_orbitals(getattr(mc, "frozen", None)):
            raise NotImplementedError("nonzero frozen spinors are not supported")
        self._mc, self._scf, self.mol = mc, mc._scf, mc._scf.mol
        self.verbose, self.stdout = mc.verbose, mc.stdout
        self.root, self.canonicalized = 0, False
        self.mo_coeff, self.mo_energy = mc.mo_coeff, getattr(mc, "mo_energy", None)
        self.mps_response_options = None
        self.response_mode = "full_chain"
        self.scratch, self.stack_memory, self.n_threads = None, 1024, lib.num_threads()
        self.reference_energy = self.e_corr = None
        self.sub_eners, self.diagnostics = {}, {}
        self.converged = False
        self._keys = set(self.__dict__)

    @property
    def e_tot(self):
        return None if self.e_corr is None else self.reference_energy + self.e_corr

    def kernel(self, root=None, mo_coeff=None, pdms=None, eris=None, eris_basis="input_mo"):
        from .spinor_helper import _SpinorERIs
        self.converged = False
        mc = u._full_integral_mc(self._mc)
        if u._has_frozen_orbitals(getattr(mc, "frozen", None)):
            raise NotImplementedError("nonzero frozen spinors are not supported")
        root = self.root if root is None else root
        if isinstance(root, bool) or not isinstance(root, (int, np.integer)):
            raise TypeError("root must be an integer")
        u._root_ket(mc.fcisolver, root)  # fail before integral/RDM work
        eris_basis = u._normalize_eris_basis(eris_basis)
        if pdms is None:
            pdms = tuple(u._make_rdm(mc.fcisolver, root, rank) for rank in (1, 2))
        pdms, rdm_audit = u.validate_pdms(pdms, mc.ncas, u._total_nelec(mc.nelecas), max_rank=2)
        original = np.asarray(mc.mo_coeff if mo_coeff is None else mo_coeff)
        mo, eps = u.semicanonicalize(mc, original, pdms[0], root,
                                     canonicalized=self.canonicalized, mo_energy=self.mo_energy)
        # Even canonicalized=True must not silently drop external Fock mixing.
        fock = mo.conj().T @ mc.get_fock(mo_coeff=mo, casdm1=pdms[0]) @ mo
        fock_error = 0.
        for sl in (slice(0, mc.ncore), slice(mc.ncore + mc.ncas, len(eps))):
            fock_error = max(fock_error, u._maximum_abs(fock[sl, sl] - np.diag(eps[sl])))
        if fock_error > 1e-9:
            raise ValueError(f"external Dyall Fock blocks are not semicanonical: {fock_error:.3e}")
        if eris is None:
            if self.response_mode == "external_tuples":
                from .nevpt2_eris import wick_eris_from_mc
                eris = wick_eris_from_mc(mc, mo)
            else:
                eris = u._dense_eris_from_mc(mc, mo)
        elif eris_basis == "input_mo" and not np.array_equal(mo, original):
            rotation = original.conj().T @ mc._scf.get_ovlp() @ mo
            eris = (u._rotate_eris(eris, rotation) if isinstance(eris, _SpinorERIs)
                    else u._rotate_wick_eris(eris, rotation))
        if (eris.ncore, eris.ncas, eris.nmo) != (mc.ncore, mc.ncas, mo.shape[1]):
            raise ValueError("ERI partition does not match the reference")
        eactive = response._real_scalar(
            np.einsum("pq,pq", eris.get_h1eff("AA"), pdms[0])
            + .5 * np.einsum("pqrs,pqsr", eris.get_phys("AAAA"), pdms[1]),
            name="active energy")[0]
        if isinstance(eris, _SpinorERIs):
            ecore = .5 * np.trace((eris.h1e + eris.h1eff)[:eris.ncore, :eris.ncore]).real
        else:
            audit = eris.symmetry_diagnostics or {}
            if "electronic_core_energy" not in audit:
                raise ValueError("compact ERIs need electronic_core_energy provenance")
            ecore = audit["electronic_core_energy"]
        enuc = float(mc.mol.energy_nuc())
        e_reference, reference_audit = u._pt_reference_energy(mc, root, mo, pdms, eris)
        e_dyall = eactive + ecore + enuc
        reference_difference = float(e_dyall - e_reference)
        energies, diagnostics = evaluate_uc(
            mc, eris, pdms, eps[:eris.ncore], eps[eris.nocc:],
            root=root, options=self.mps_response_options, scratch=self.scratch,
            stack_memory=self.stack_memory, n_threads=self.n_threads, response_mode=self.response_mode)
        measured = all(d["global_relative_residual"] is not None for d in diagnostics.values())
        total_norm = sum(d["source_norm2"] for d in diagnostics.values()) if measured else None
        # Sum actual squared residuals, not norm2*rho2: a zero-source sector
        # can leak (rho=inf), and 0*inf would hide it in a NaN.
        residual_norm2 = (sum(d.get("residual_norm2", 0.) for d in diagnostics.values())
                          if measured else None)
        rho = (np.sqrt(residual_norm2 / total_norm) if measured and total_norm
               else (0. if residual_norm2 == 0. else float("inf")) if measured else None)
        joint = next(iter(diagnostics.values())).get("joint_response")
        if joint is not None:
            # The native default also contains CAS. Do not silently replace
            # its result/residual by the external-sector sums (a final Q).
            rho = joint["global_relative_residual"]
        self.root, self.mo_coeff, self.mo_energy = int(root), mo, eps
        self.reference_energy = float(e_reference)
        self.e_corr = float(joint["hylleraas_energy"] if joint is not None else sum(energies.values()))
        self.sub_eners, self.eris = energies, eris
        max_residual = (max(d.get("max_channel_relative_residual", d["global_relative_residual"])
                            for d in diagnostics.values())
                        if measured else None)
        self.diagnostics = dict(classes=diagnostics, global_relative_residual=rho,
                                max_channel_relative_residual=max_residual,
                                response_partition=("fixed external tuples; active-only responses"
                                    if self.response_mode == "external_tuples" else "single MRCIMPSInfo(ci_order=2) response"),
                                cas_constraint=("fixed external occupation; no (0,0) tuple"
                                    if self.response_mode == "external_tuples" else joint.get("projector") if joint else None),
                                reference=reference_audit, rdms=rdm_audit,
                                active_energy=eactive, electronic_core_energy=float(ecore),
                                nuclear_energy=enuc, dyall_reference_energy=e_dyall,
                                reference_energy_difference=reference_difference,
                                dyall_constant=float(ecore + enuc - sum(eps[:eris.ncore])),
                                external_fock_error=fock_error)
        if joint is not None:
            for name in ("reference_zero_mode_leakage", "cas_response_relative_norm",
                         "cas_purity_certified", "reference_eigenstate_certified",
                         "cas_source_norm", "cas_source_relative_norm", "cas_hylleraas_energy"):
                if name in joint:
                    self.diagnostics[name] = joint[name]
        self.converged = (measured and all(d["converged"] for d in diagnostics.values())
                          and abs(reference_difference) <= 1e-8)
        if not self.converged:
            u._warn_numerical(f"full UC not certified: global residual={rho}, "
                              f"max class residual={max_residual}, "
                              f"Dyall/reference energy difference={reference_difference:.3e} Eh")
        logger.note(self, "E(X2CUCNEVPT2) = %.14f  E_corr = %.14f  converged = %s",
                    self.e_tot, self.e_corr, self.converged)
        return self.e_corr


UCNEVPT2 = X2CUCNEVPT2
