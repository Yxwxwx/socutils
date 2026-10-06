# SPDX-License-Identifier: GPL-3.0-or-later
r"""One-step complex-spinor MS-FIC-NEVPT2, SS-SR and MS-MR.

Reynolds--Shiozaki, JCTC 15, 1560 (2019), Eqs. (12),(13),(23),(27),(28).
Both ansatzes use one SA Dyall partition and the same rank-zero--four
transition densities. No UC response, spin adaptation, or Kramers reduction.
Amplitudes represent the true first-order wavefunction: (A+eta S)t=-V.
"""

import errno
import hashlib
import json
import logging
import shutil
import time
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp
from types import SimpleNamespace

import numpy as np
from pyscf import lib
from scipy.linalg import eigh

from . import nevpt2_utils as u
from . import x2cficnevpt2 as fic
from ._ic_metric import congruence, refine_metric_basis
from .nevpt2_eris import wick_eris_from_mc


def _gate(error, scale, atol, rtol, name, *, warning=False):
    limit = atol + rtol * scale
    message = f"{name}: defect={error:.3e}, limit={limit:.3e}"
    if not np.isfinite(error):
        raise FloatingPointError(message)
    if error > limit:
        if warning:
            u._warn_numerical(message)
        else:
            raise FloatingPointError(message)
    return float(error)


def _hermitian(matrix, atol, rtol, name):
    matrix = np.asarray(matrix, dtype=complex)
    if not np.all(np.isfinite(matrix)):
        raise FloatingPointError(f"non-finite {name}")
    error = _gate(
        u._maximum_abs(matrix - matrix.conj().T),
        u._maximum_abs(matrix),
        atol,
        rtol,
        name,
        warning=True,
    )
    return (matrix + matrix.conj().T) * 0.5, error


@dataclass
class MSFICClass:
    metric: np.ndarray
    right: np.ndarray
    left: np.ndarray
    active: np.ndarray
    source: np.ndarray
    dimension_per_reference: int
    diagnostics: dict


@dataclass
class MSFICPrepared:
    eris: object
    mo_coeff: np.ndarray
    mo_energy: np.ndarray
    sa_density: np.ndarray
    model_roots: tuple
    overlap: np.ndarray
    active_reference: np.ndarray
    reference: np.ndarray
    classes: dict
    diagnostics: dict

    def rotated(self, rotation):
        """Covariant model-coordinate change, keeping SA orbitals/Fock fixed."""
        unitary = np.asarray(rotation, dtype=complex)
        n = len(self.model_roots)
        if unitary.shape != (n, n) or not np.allclose(
            unitary.conj().T @ unitary, np.eye(n), atol=1e-12, rtol=0
        ):
            raise ValueError("reference rotation must be unitary")
        classes = {}
        for key, block in self.classes.items():
            transform = np.kron(unitary, np.eye(block.dimension_per_reference))
            matrices = [
                transform.conj().T @ a @ transform
                for a in (block.metric, block.right, block.left, block.active)
            ]
            source = np.einsum(
                "ab,...bc,cd->...ad",
                transform.conj().T,
                block.source,
                unitary,
                optimize=True,
            )
            classes[key] = MSFICClass(
                *matrices,
                source,
                block.dimension_per_reference,
                dict(block.diagnostics),
            )
        return MSFICPrepared(
            self.eris,
            self.mo_coeff,
            self.mo_energy,
            self.sa_density,
            self.model_roots,
            unitary.conj().T @ self.overlap @ unitary,
            unitary.conj().T @ self.active_reference @ unitary,
            unitary.conj().T @ self.reference @ unitary,
            classes,
            {**self.diagnostics, "reference_rotation": unitary},
        )


def _contract_pair(eris, pdms, overlap, equations, backend):
    """Reuse single-state IC operators/selectors; only the density is transition."""
    context = u._execution_context(eris, pdms, overlap=overlap)
    context.update({f"ident{k}": np.ones((1,) * k, dtype=complex) for k in (1, 2, 3)})
    active_eris = SimpleNamespace(ncore=1, nvirt=1, ncas=eris.ncas)
    active_context = {
        **context,
        "deltaII": np.zeros((1, 1)),
        "deltaEE": np.zeros((1, 1)),
    }
    globals_ = {"np": u._wick_einsum_namespace(backend)}
    result = {}
    for key in fic.SUBSPACE_ORDER:
        components = fic._IC_COMPONENTS[key]
        free_shape = fic._shape_for_labels(tuple(key), eris)
        if 0 in free_shape:
            continue
        tensors = {name: {} for name in ("metric", "right", "left")}
        sources = {}
        for bra in components:
            sources[bra.name] = fic._execute_tensor(
                equations.rhs_code[key, bra.name],
                "rhs",
                tuple(key) + bra.bra_active,
                context,
                globals_,
                np.complex128,
                eris,
            )
            for ket in components:
                pair = (key, bra.name, ket.name)
                labels = tuple(key) + bra.bra_active + ket.ket_active
                for name, target in tensors.items():
                    target[bra.name, ket.name] = fic._execute_tensor(
                        getattr(equations, name + "_code")[pair],
                        name,
                        labels,
                        active_context,
                        globals_,
                        np.complex128,
                        active_eris,
                    )
        zero = (0,) * len(key)
        matrices = [
            fic._assemble_matrix(tensors[name], zero, components, eris.ncas)
            for name in tensors
        ]
        # Keep all free tuples in source storage; unique-pair selectors below
        # are the same as the audited single-state FIC implementation.
        pieces = [
            sources[c.name].reshape(free_shape + (-1,))[
                ..., fic._component_selection(c, eris.ncas)
            ]
            for c in components
        ]
        result[key] = (*matrices, np.concatenate(pieces, axis=-1))
    return result


def _capacity_error(error):
    # NumPy tofile can report a short write without preserving errno.
    return error.errno in (errno.ENOSPC, errno.EDQUOT) or (
        error.errno is None
        and " requested and " in str(error)
        and " written" in str(error)
    )


def _write_transition_pair(pdms, bra, ket, ncas, directory, *, keep):
    """Commit one raw ordered pair without copying/scanning rank four."""
    if not isinstance(pdms, (tuple, list)) or len(pdms) != 4:
        raise ValueError("MS-FIC requires full transition ranks 1--4")
    arrays = tuple(np.asarray(dm) for dm in pdms)
    for rank, dm in enumerate(arrays, start=1):
        if dm.shape != (ncas,) * (2 * rank):
            raise ValueError(f"transition dm{rank} has wrong shape")
        if not np.issubdtype(dm.dtype, np.number):
            raise TypeError(f"transition dm{rank} must be numeric")
    if not keep and all(
        isinstance(dm, np.memmap) and not dm.flags.writeable for dm in pdms
    ):
        return tuple(pdms)  # Borrow existing file maps; do not rewrite their contents.
    for rank, dm in enumerate(arrays, start=1):
        path = directory / f"dm{rank}.npy"
        temporary = path.with_suffix(".npy.tmp")
        with temporary.open("wb") as handle:
            np.save(handle, dm, allow_pickle=False)
        temporary.replace(path)
    temporary = directory / "READY.json.tmp"
    temporary.write_text(
        json.dumps(
            {
                "format": "socutils.msfic.transition-rdms",
                "bra": int(bra),
                "ket": int(ket),
                "ncas": int(ncas),
                "shapes": [list(dm.shape) for dm in arrays],
                "dtypes": [str(dm.dtype) for dm in arrays],
            }
        ),
        encoding="utf-8",
    )
    temporary.replace(directory / "READY.json")


@contextmanager
def _disk_transition_pair(
    provider,
    bra,
    ket,
    ncas,
    directory,
    *,
    keep,
    fallback_directory=None,
):
    """Release generated arrays before opening read-only contraction maps."""
    pdms = provider(bra, ket)
    directories = [directory]
    if fallback_directory is not None:
        directories.append(fallback_directory)
    with ExitStack() as chosen:
        for index, base in enumerate(directories):
            pair_directory = None
            with ExitStack() as trial:
                try:
                    if not keep and all(
                        isinstance(dm, np.memmap) and not dm.flags.writeable
                        for dm in pdms
                    ):
                        # The writer validates metadata but does not write borrowed maps.
                        existing = _write_transition_pair(
                            pdms, bra, ket, ncas, None, keep=False
                        )
                        break
                    required = sum(np.asarray(dm).nbytes for dm in pdms) + 2**20
                    if shutil.disk_usage(base).free < required:
                        raise OSError(
                            errno.ENOSPC,
                            f"need {required / 2**30:.3f} GiB for one RDM pair",
                            str(base),
                        )
                    storage = (
                        nullcontext(str(Path(base) / f"bra-{bra}_ket-{ket}"))
                        if keep
                        else TemporaryDirectory(prefix=f"pair-{bra}-{ket}_", dir=base)
                    )
                    pair_directory = Path(trial.enter_context(storage))
                    pair_directory.mkdir(parents=True, exist_ok=True)
                    existing = _write_transition_pair(
                        pdms,
                        bra,
                        ket,
                        ncas,
                        pair_directory,
                        keep=keep,
                    )
                except OSError as error:
                    if keep and pair_directory is not None and pair_directory.is_dir():
                        names = ["READY.json", "READY.json.tmp"] + [
                            f"dm{rank}.npy{suffix}"
                            for rank in range(1, 5)
                            for suffix in ("", ".tmp")
                        ]
                        for name in names:
                            (pair_directory / name).unlink(missing_ok=True)
                        pair_directory.rmdir()
                    if not _capacity_error(error) or index + 1 == len(directories):
                        raise
                    logging.getLogger(__name__).warning(
                        "MS-FIC transition RDM (%s,%s): %s; falling back to %s",
                        bra,
                        ket,
                        error,
                        directories[index + 1],
                    )
                    continue
                chosen.enter_context(trial.pop_all())
                break
        del pdms
        if existing is not None:
            yield existing
            return
        mapped = []
        try:
            for rank in range(1, 5):
                mapped.append(
                    np.load(
                        pair_directory / f"dm{rank}.npy",
                        mmap_mode="r",
                        allow_pickle=False,
                    )
                )
            yield tuple(mapped)
        finally:
            for dm in mapped:
                dm._mmap.close()


def build_msfic_classes(
    eris,
    model_roots,
    transition_pdms,
    overlaps,
    *,
    contraction_backend="pytblis",
    matrix_atol=1e-10,
    matrix_rtol=1e-10,
    transition_rdm_dir=None,
    transition_rdm_fallback_dir=None,
):
    """Stream ordered model-root pairs, not high-order pairs for all SA roots.

    ``transition_pdms(bra_root,ket_root)`` returns raw SGF ranks 1--4.
    Both directions are independently measured and contracted without
    elementwise RDM audits or alterations. Only array metadata is checked.
    Each pair is written then read through mmap, with no other pair live.
    An explicit directory retains files in a fresh preparation subdirectory;
    otherwise only the current pair occupies ``lib.param.TMPDIR``.
    A configured fallback directory is used only for capacity/quota failures.
    """
    roots = tuple(model_roots)
    n = len(roots)
    overlap = np.asarray(overlaps, dtype=complex)
    if overlap.shape != (n, n):
        raise ValueError("model overlap shape mismatch")
    _gate(
        u._maximum_abs(overlap - np.eye(n)),
        1.0,
        matrix_atol,
        matrix_rtol,
        "model states are not orthonormal; tighten reference",
    )
    backend = u._normalize_contraction_backend(contraction_backend)
    equations = fic._compile_fic_equations(include_overlap=True)
    classes = {}
    active_ref = np.zeros((n, n), dtype=complex)
    keep_rdms = transition_rdm_dir is not None
    locations = {}
    with ExitStack() as sessions:

        def session(base):
            Path(base).mkdir(parents=True, exist_ok=True)
            if keep_rdms:
                return mkdtemp(prefix="msfic_rdms_", dir=base)
            return sessions.enter_context(
                TemporaryDirectory(prefix="msfic_rdms_", dir=base)
            )

        fallback_directory = (
            session(transition_rdm_fallback_dir)
            if transition_rdm_fallback_dir is not None
            else None
        )
        try:
            rdm_directory = session(
                transition_rdm_dir if keep_rdms else lib.param.TMPDIR
            )
        except OSError as error:
            if not _capacity_error(error) or fallback_directory is None:
                raise
            logging.getLogger(__name__).warning(
                "MS-FIC RDM temporary directory unavailable: %s; using %s",
                error,
                fallback_directory,
            )
            rdm_directory, fallback_directory = fallback_directory, None
        for i, bra in enumerate(roots):
            for j, ket in enumerate(roots):
                with _disk_transition_pair(
                    transition_pdms,
                    bra,
                    ket,
                    eris.ncas,
                    rdm_directory,
                    keep=keep_rdms,
                    fallback_directory=fallback_directory,
                ) as densities:
                    if keep_rdms:
                        locations[f"{bra},{ket}"] = str(
                            Path(densities[0].filename).parent
                        )
                    active_ref[i, j] = np.einsum(
                        "pq,pq", eris.get_h1eff("AA"), densities[0]
                    ) + 0.5 * np.einsum(
                        "pqrs,pqsr", eris.get_phys("AAAA"), densities[1]
                    )
                    contracted = _contract_pair(
                        eris, densities, overlap[i, j], equations, backend
                    )
                    for key, (s, r, l, v) in contracted.items():
                        d = len(s)
                        if key not in classes:
                            classes[key] = MSFICClass(
                                *[
                                    np.zeros((n * d, n * d), dtype=complex)
                                    for _ in range(4)
                                ],
                                np.zeros(v.shape[:-1] + (n * d, n), dtype=complex),
                                d,
                                {},
                            )
                        block = classes[key]
                        row, col = slice(i * d, (i + 1) * d), slice(j * d, (j + 1) * d)
                        (
                            block.metric[row, col],
                            block.right[row, col],
                            block.left[row, col],
                        ) = s, r, l
                        block.source[..., row, j] = v
                del densities, contracted
    active_ref, ref_error = _hermitian(
        active_ref, matrix_atol, matrix_rtol, "active Href"
    )
    for key, block in classes.items():
        s, r, l = block.metric, block.right, block.left
        d = block.dimension_per_reference
        s4 = s.reshape(n, d, n, d)
        se = np.einsum("najb,jp->napb", s4, active_ref).reshape(s.shape)
        es = np.einsum("nj,japb->napb", active_ref, s4).reshape(s.shape)
        scale = max(u._maximum_abs(r), u._maximum_abs(l), u._maximum_abs(es))
        adjoint = _gate(
            u._maximum_abs(r - l.conj().T),
            scale,
            matrix_atol,
            matrix_rtol,
            key + " R=L-adjoint",
        )
        restoration = _gate(
            u._maximum_abs((r + se) - (l + es)),
            scale,
            matrix_atol,
            matrix_rtol,
            key + " F energy restoration",
        )
        block.metric, s_error = _hermitian(s, matrix_atol, matrix_rtol, key + " S")
        # Record numerical reference/IC defects before Hermitian averaging.
        block.active, f_error = _hermitian(
            0.5 * (r + l + se + es), matrix_atol, matrix_rtol, key + " F"
        )
        block.diagnostics = {
            "metric_hermiticity": s_error,
            "active_hermiticity": f_error,
            "commutator_adjoint": adjoint,
            "energy_restoration": restoration,
        }
    return (
        classes,
        active_ref,
        {
            "transition_rdm_audited": False,
            "transition_rdm_storage": "per-pair-readonly-mmap",
            "transition_rdm_directory": str(rdm_directory) if keep_rdms else None,
            "transition_rdm_locations": locations,
            "active_reference_hermiticity": ref_error,
        },
    )


def prepare_msfic(
    mc,
    *,
    sa_roots,
    sa_weights,
    model_roots,
    mo_coeff=None,
    transition_pdms=None,
    model_overlap=None,
    contraction_backend="pytblis",
    integral_max_memory=2000,
    matrix_atol=1e-10,
    matrix_rtol=1e-10,
    transition_rdm_dir=None,
    transition_rdm_fallback_dir=None,
):
    """One public SA-Fock preparation, active basis unchanged, full Coulomb.

    A supplied transition provider is an explicit testing/restart hook; the
    default uses the actual project's DMRG MPS/NPDM interfaces.
    Transition RDMs are streamed through disk, one ordered pair at a time.
    Set ``transition_rdm_dir`` to retain the files instead of deleting them
    after contraction. Each preparation creates an isolated subdirectory.
    Configure ``transition_rdm_fallback_dir`` for capacity/quota spillover.
    """
    start = time.perf_counter()
    if u._has_frozen_orbitals(getattr(mc, "frozen", None)):
        raise NotImplementedError("MS-FIC currently requires frozen=0")
    mc = u._full_integral_mc(mc)
    sa_roots, model_roots = tuple(sa_roots), tuple(model_roots)
    if not sa_roots or not model_roots or len(set(model_roots)) != len(model_roots):
        raise ValueError("nonempty distinct root sets required")
    if any(r not in sa_roots for r in model_roots):
        raise ValueError("model roots must be in SA roots")
    weights = np.asarray(sa_weights, dtype=float)
    if (
        weights.shape != (len(sa_roots),)
        or np.any(weights < 0)
        or not np.all(np.isfinite(weights))
    ):
        raise ValueError("invalid SA weights")
    if not np.isclose(weights.sum(), 1.0, atol=1e-12, rtol=0):
        raise ValueError("SA weights must sum to one")
    sa_density = sum(
        w * u.make_rdm1(mc.fcisolver, r) for w, r in zip(weights, sa_roots)
    )
    input_mo = np.asarray(mc.mo_coeff if mo_coeff is None else mo_coeff)
    mo, eps = u.semicanonicalize(mc, input_mo, sa_density, sa_roots[0])
    fock = mo.conj().T @ mc.get_fock(mo_coeff=mo, casdm1=sa_density) @ mo
    nc, no = int(mc.ncore), int(mc.ncore + mc.ncas)
    off = max(
        u._maximum_abs(fock[:nc, :nc] - np.diag(eps[:nc])),
        u._maximum_abs(fock[no:, no:] - np.diag(eps[no:])),
    )
    _gate(
        off, u._maximum_abs(fock), matrix_atol, matrix_rtol, "common external SA Fock"
    )
    eris = wick_eris_from_mc(mc, mo, max_memory=integral_max_memory)
    if model_overlap is None:
        model_overlap = np.array(
            [
                [u.make_transition_overlap(mc.fcisolver, a, b) for b in model_roots]
                for a in model_roots
            ]
        )
    if transition_pdms is None:
        transition_pdms = lambda a, b: u.make_transition_dm1234(mc.fcisolver, a, b)
    classes, active_ref, audits = build_msfic_classes(
        eris,
        model_roots,
        transition_pdms,
        model_overlap,
        contraction_backend=contraction_backend,
        matrix_atol=matrix_atol,
        matrix_rtol=matrix_rtol,
        transition_rdm_dir=transition_rdm_dir,
        transition_rdm_fallback_dir=transition_rdm_fallback_dir,
    )
    core = eris.symmetry_diagnostics["electronic_core_energy"] + mc.mol.energy_nuc()
    reference = active_ref + core * np.asarray(model_overlap)
    fingerprint = hashlib.sha256()
    for array in (mo, eps, sa_density, weights, reference):
        fingerprint.update(np.ascontiguousarray(array).tobytes())
    return MSFICPrepared(
        eris,
        mo,
        eps,
        sa_density,
        model_roots,
        np.asarray(model_overlap),
        active_ref,
        reference,
        classes,
        {
            **audits,
            "sa_roots": sa_roots,
            "sa_weights": weights,
            "external_fock_defect": off,
            "fingerprint": fingerprint.hexdigest(),
            "preparation_seconds": time.perf_counter() - start,
        },
    )


def _metric_basis(
    s, *, metric_atol, metric_rcond, matrix_atol, matrix_rtol, metric_refinement=True
):
    s, error = _hermitian(s, matrix_atol, matrix_rtol, "IC metric")
    vals, vecs = eigh(s)
    scale = max(1.0, u._maximum_abs(vals))
    _gate(
        max(0.0, -float(vals.min(initial=0))),
        scale,
        matrix_atol,
        matrix_rtol,
        "negative metric eigenvalue",
    )
    cutoff = max(metric_atol, metric_rcond * vals.max(initial=0))
    keep = vals > cutoff
    x = vecs[:, keep] / np.sqrt(vals[keep])
    refinement = {"gram_refinement_steps": 0}
    if metric_refinement:
        x, refinement = refine_metric_basis(
            x,
            s,
            tolerance=max(1e-13, matrix_atol * 0.1),
            threads=lib.num_threads(),
        )
    return (
        x,
        vecs[:, ~keep],
        {
            "rank": int(keep.sum()),
            "dimension": len(s),
            "minimum_eigenvalue": float(vals[0]) if vals.size else 0.0,
            "maximum_eigenvalue": float(vals[-1]) if vals.size else 0.0,
            "minimum_retained_eigenvalue": float(vals[keep][0]) if keep.any() else None,
            "condition_number": float(vals[keep][-1] / vals[keep][0])
            if keep.any()
            else None,
            "cutoff": float(cutoff),
            "hermiticity": error,
            **refinement,
        },
    )


def solve_msfic(
    prepared,
    *,
    ansatz="ms_mr",
    shift=0.2,
    metric_atol=1e-12,
    metric_rcond=1e-11,
    metric_refinement=True,
    matrix_atol=1e-10,
    matrix_rtol=1e-10,
    null_atol=1e-10,
    null_rtol=1e-8,
    residual_tol=1e-10,
    singular_tol=1e-12,
):
    """Solve IC projected equations, including non-diagonal model Href.

    MS-MR solves the common-space Sylvester equation in the model eigenbasis.
    SS-SR restricts each target column to its own reference-generated span;
    off-diagonal Href is retained in the coupled projected equations.
    """
    if ansatz not in ("ss_sr", "ms_mr"):
        raise ValueError("ansatz must be ss_sr or ms_mr")
    for name, value in locals().copy().items():
        if name not in ("prepared", "ansatz"):
            u._finite_nonnegative(value, name=name)
    n = len(prepared.model_roots)
    e, rotation = eigh(prepared.active_reference)
    energy_origin = (
        float(np.mean(e)) if ansatz == "ms_mr" and metric_refinement else 0.0
    )
    e = e - energy_origin
    ytotal = np.zeros((n, n), dtype=complex)
    ntotal = np.zeros_like(ytotal)
    corrections, norms, audits = {}, {}, {}
    start = time.perf_counter()
    eris = prepared.eris
    for key, block in prepared.classes.items():
        s, f, v = block.metric, block.active, block.source
        d = block.dimension_per_reference
        args = {
            "metric_atol": metric_atol,
            "metric_rcond": metric_rcond,
            "matrix_atol": matrix_atol,
            "matrix_rtol": matrix_rtol,
            "metric_refinement": metric_refinement,
        }
        if ansatz == "ms_mr":
            x, null, metric = _metric_basis(s, **args)
            factors = [(x, null)]
            refined = metric["gram_refinement_steps"] > 0
            connected = (
                np.asarray(f, dtype=np.clongdouble)
                - energy_origin * np.asarray(s, dtype=np.clongdouble)
                if refined
                else f - energy_origin * s
            )
            f_orth = congruence(
                x, connected, extended=refined, threads=lib.num_threads()
            )
        else:
            factors, metrics = [], []
            for root in range(n):
                sl = slice(root * d, (root + 1) * d)
                xr, nr, mr = _metric_basis(s[sl, sl], **args)
                factors.append((xr, nr))
                metrics.append(mr)
            ranks = [len(xr.T) for xr, _ in factors]
            offsets = np.cumsum([0] + ranks)
            f_orth = np.zeros((offsets[-1], offsets[-1]), dtype=complex)
            for k, (xk, _) in enumerate(factors):
                rows, kr = slice(k * d, (k + 1) * d), slice(offsets[k], offsets[k + 1])
                # Cancel the reference energy before an ill-conditioned
                # metric congruence, not between two amplified products.
                refined = metrics[k]["gram_refinement_steps"] > 0
                dtype = np.clongdouble if refined else np.complex128
                connected = np.asarray(
                    f[rows, rows], dtype=dtype
                ) - prepared.active_reference[k, k] * np.asarray(
                    s[rows, rows], dtype=dtype
                )
                f_orth[kr, kr] = congruence(
                    xk,
                    connected,
                    extended=refined,
                    threads=lib.num_threads(),
                )
                for j, (xj, _) in enumerate(factors):
                    if j == k:
                        continue
                    cols, jr = (
                        slice(j * d, (j + 1) * d),
                        slice(offsets[j], offsets[j + 1]),
                    )
                    coupling = prepared.active_reference[j, k]
                    # Tiny model couplings retain the full term on the BLAS path.
                    extend_cross = abs(coupling) > matrix_atol and (
                        refined or metrics[j]["gram_refinement_steps"] > 0
                    )
                    f_orth[kr, jr] -= coupling * (
                        congruence(
                            xk,
                            s[rows, cols],
                            xj,
                            extended=extend_cross,
                            threads=lib.num_threads(),
                        )
                    )
            metric = {"per_reference": metrics, "rank": int(offsets[-1])}
        raw_f_orth = f_orth
        f_orth, f_error = _hermitian(
            f_orth, matrix_atol, matrix_rtol, key + " retained IC F"
        )
        spectrum, z = eigh(f_orth)
        ysum, nsum = np.zeros_like(ytotal), np.zeros_like(ntotal)
        summary = {
            **block.diagnostics,
            "metric": metric,
            "number_of_tuples": 0,
            "maximum_ic_relative_residual": 0.0,
            "maximum_ic_absolute_residual": 0.0,
            "maximum_projected_source_norm": 0.0,
            "minimum_nonzero_projected_source_norm": None,
            "maximum_discarded_source": 0.0,
            "maximum_condition_number": 0.0,
            "minimum_absolute_denominator": None,
            "minimum_denominator": None,
            "maximum_denominator": None,
            "retained_hermiticity": f_error,
        }
        for indices in fic._iter_free_tuples(key, eris):
            source = v[indices]
            delta = fic._orbital_gap_at(
                key,
                indices,
                prepared.mo_energy[: eris.ncore],
                prepared.mo_energy[eris.nocc :],
            )
            t = np.zeros_like(source)
            if ansatz == "ms_mr":
                discarded = np.linalg.norm(null.conj().T @ source)
                rhs = x.conj().T @ source @ rotation
                denoms = spectrum[:, None] + delta + shift - e[None, :]
            else:
                discarded = np.sqrt(
                    sum(
                        np.linalg.norm(nr.conj().T @ source[k * d : (k + 1) * d, k])
                        ** 2
                        for k, (_, nr) in enumerate(factors)
                    )
                )
                rhs = np.concatenate(
                    [
                        xr.conj().T @ source[k * d : (k + 1) * d, k]
                        for k, (xr, _) in enumerate(factors)
                    ]
                )
                denoms = spectrum + delta + shift
            _gate(
                discarded,
                np.linalg.norm(source),
                null_atol,
                null_rtol,
                key + " source in discarded metric space",
            )
            if denoms.size:
                minabs = float(np.min(np.abs(denoms)))
                if minabs <= singular_tol:
                    raise FloatingPointError(
                        f"{key}/{indices}: near-singular IC block {minabs:.3e}"
                    )
                response = -z @ ((z.conj().T @ rhs) / denoms)
                if ansatz == "ms_mr":
                    t = x @ response @ rotation.conj().T
                    residual = (
                        raw_f_orth @ response
                        + (delta + shift) * response
                        - response * e
                        + rhs
                    )
                else:
                    for k, (xr, _) in enumerate(factors):
                        t[k * d : (k + 1) * d, k] = (
                            xr @ response[offsets[k] : offsets[k + 1]]
                        )
                    residual = raw_f_orth @ response + (delta + shift) * response + rhs
                if rhs.ndim == 2:
                    lengths = np.linalg.norm(rhs, axis=0)
                    errors = np.linalg.norm(residual, axis=0)
                else:
                    lengths, errors = (
                        np.atleast_1d(np.linalg.norm(rhs)),
                        np.atleast_1d(np.linalg.norm(residual)),
                    )
                rho = float(
                    np.max(
                        np.divide(
                            errors,
                            lengths,
                            out=np.zeros_like(errors),
                            where=lengths != 0,
                        )
                    )
                )
                if not np.isfinite(rho):
                    raise FloatingPointError(key + " non-finite IC relative residual")
                summary["maximum_ic_absolute_residual"] = max(
                    float(errors.max(initial=0)),
                    summary["maximum_ic_absolute_residual"],
                )
                summary["maximum_projected_source_norm"] = max(
                    float(lengths.max(initial=0)),
                    summary["maximum_projected_source_norm"],
                )
                nonzero = lengths[lengths > 0]
                if nonzero.size:
                    summary["minimum_nonzero_projected_source_norm"] = min(
                        float(nonzero.min()),
                        summary["minimum_nonzero_projected_source_norm"] or np.inf,
                    )
                condition = float(np.max(np.abs(denoms)) / minabs)
                summary["minimum_absolute_denominator"] = min(
                    minabs, summary["minimum_absolute_denominator"] or np.inf
                )
                summary["minimum_denominator"] = min(
                    float(denoms.min()),
                    summary["minimum_denominator"]
                    if summary["minimum_denominator"] is not None
                    else np.inf,
                )
                summary["maximum_denominator"] = max(
                    float(denoms.max()),
                    summary["maximum_denominator"]
                    if summary["maximum_denominator"] is not None
                    else -np.inf,
                )
                summary["maximum_condition_number"] = max(
                    condition, summary["maximum_condition_number"]
                )
                summary["maximum_ic_relative_residual"] = max(
                    rho, summary["maximum_ic_relative_residual"]
                )
            yy, nn = source.conj().T @ t, t.conj().T @ s @ t
            summary["maximum_discarded_source"] = max(
                float(discarded), summary["maximum_discarded_source"]
            )
            summary["number_of_tuples"] += 1
            ysum += yy
            nsum += nn
        _gate(
            summary["maximum_ic_relative_residual"],
            1.0,
            0.0,
            residual_tol,
            key + " IC relative residual",
            warning=True,
        )
        corrections[key] = 0.5 * (ysum + ysum.conj().T) - shift * nsum
        norms[key], audits[key] = nsum, summary
        ytotal += ysum
        ntotal += nsum
    heff, herm_error = _hermitian(
        prepared.reference + 0.5 * (ytotal + ytotal.conj().T) - shift * ntotal,
        matrix_atol,
        matrix_rtol,
        "Heff",
    )
    energies, mixing = eigh(heff)
    eigen_residual = float(np.linalg.norm(heff @ mixing - mixing * energies))
    return MSFICResult(
        ansatz,
        shift,
        energies,
        mixing,
        heff,
        ytotal,
        ntotal,
        corrections,
        norms,
        {
            "classes": audits,
            "heff_hermiticity": herm_error,
            "heff_eigen_residual": eigen_residual,
            "solve_seconds": time.perf_counter() - start,
            "metric_refinement": bool(metric_refinement),
            "active_energy_origin": energy_origin,
        },
    )


@dataclass
class MSFICResult:
    ansatz: str
    shift: float
    energies: np.ndarray
    mixing: np.ndarray
    heff: np.ndarray
    source_response: np.ndarray
    shift_norm: np.ndarray
    class_corrections: dict
    class_norms: dict
    diagnostics: dict

    @property
    def spread(self):
        return float(np.ptp(self.energies))


class WickX2CMSFICNEVPT2(lib.StreamObject):
    """PySCF-style MS-FIC entry; a prepared data object can serve both ansatzes."""

    def __init__(self, mc, frozen=0):
        if u._has_frozen_orbitals(frozen):
            raise NotImplementedError("MS-FIC currently requires frozen=0")
        self._mc = mc
        self.mol = mc.mol
        self.verbose, self.stdout = mc.verbose, mc.stdout
        self.sa_roots = tuple(range(getattr(mc.fcisolver, "nroots", 1)))
        self.sa_weights = np.asarray(
            getattr(mc, "weights", np.ones(len(self.sa_roots)) / len(self.sa_roots))
        )
        self.model_roots = None
        self.ansatz, self.shift = "ms_mr", 0.2
        self.contraction_backend = "pytblis"
        self.transition_rdm_dir = None
        self.transition_rdm_fallback_dir = None
        self.prepared, self.result = None, None

    def kernel(
        self,
        *,
        prepared=None,
        sa_roots=None,
        sa_weights=None,
        model_roots=None,
        ansatz=None,
        shift=None,
        **solver_options,
    ):
        if prepared is None:
            roots = self.model_roots if model_roots is None else model_roots
            if roots is None:
                raise ValueError("model_roots must be explicitly identified")
            prepared = prepare_msfic(
                self._mc,
                sa_roots=self.sa_roots if sa_roots is None else sa_roots,
                sa_weights=self.sa_weights if sa_weights is None else sa_weights,
                model_roots=roots,
                contraction_backend=self.contraction_backend,
                transition_rdm_dir=self.transition_rdm_dir,
                transition_rdm_fallback_dir=self.transition_rdm_fallback_dir,
            )
        self.prepared = prepared
        self.result = solve_msfic(
            prepared,
            ansatz=self.ansatz if ansatz is None else ansatz,
            shift=self.shift if shift is None else shift,
            **solver_options,
        )
        return self.result

    @property
    def e_tot(self):
        return None if self.result is None else self.result.energies


X2CMSFICNEVPT2 = WickX2CMSFICNEVPT2
