#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared infrastructure for the dense spinor NEVPT2 implementations.

This module contains method-independent data handling only: raw SGF RDM I/O
and validation, transition RDMs, contraction-backend selection, Wick parser
plumbing, semicanonical integral transport, and numerical audit helpers.
SC, FIC, and QD equations and solvers remain in their dedicated modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import itertools
import warnings
from types import SimpleNamespace

import numpy as np
from pyscf.lib import logger

try:
    from . import spinor_helper
except ImportError:  # pragma: no cover - repository-snapshot execution
    import spinor_helper


__all__ = [
    "MRPTNumericalWarning",
    "SUBSPACE_ORDER",
    "adjoint_transition_pdm",
    "make_dm1234",
    "make_rdm1",
    "make_rdm2",
    "make_rdm3",
    "make_rdm4",
    "make_transition_dm1234",
    "make_transition_overlap",
    "make_transition_rdm1",
    "make_transition_rdm2",
    "make_transition_rdm3",
    "make_transition_rdm4",
    "semicanonicalize",
    "validate_pdms",
    "validate_transition_pdms",
]


SUBSPACE_ORDER = ("ijrs", "rsi", "ijr", "rs", "ij", "ir", "r", "i")

_ERIS_BASES = frozenset(("input_mo", "semicanonical"))
_DENOMINATOR_MODES = frozenset(("hermitianized", "strict_si"))
_CONTRACTION_BACKENDS = frozenset(("numpy", "pytblis"))
_DEFAULT_CONTRACTION_BACKEND = "pytblis"
_DEFAULT_RDM_ATOL = 1.0e-9
_DEFAULT_RDM_RTOL = 1.0e-8
_DEFAULT_RDM_WORK_MEMORY = 512 * 2**20
_DEFAULT_DENOMINATOR_TOL = 1.0e-12


class MRPTNumericalWarning(RuntimeWarning):
    """A finite MRPT audit residual exceeded its configured tolerance."""


def _warn_numerical(message: str) -> None:
    """Report a recoverable finite-residual audit failure."""

    warnings.warn(message, MRPTNumericalWarning, stacklevel=2)


def _select_root_ci(ci, root, *, nroots=1):
    """Select a multiroot CI handle without slicing a single CI vector."""

    root = int(root)
    if isinstance(ci, (list, tuple, range)):
        return ci[root]
    if isinstance(ci, np.ndarray) and ci.ndim == 1:
        if ci.dtype == object or (int(nroots) > 1 and len(ci) == int(nroots)):
            return ci[root]
    return ci


def _total_nelec(nelecas) -> int:
    """Normalize scalar or ``(nalpha, nbeta)`` active-electron counts."""

    if np.isscalar(nelecas):
        total = int(nelecas)
    else:
        values = tuple(nelecas)
        if not values:
            raise ValueError("nelecas must not be empty")
        total = sum(int(value) for value in values)
    if total < 0:
        raise ValueError("nelecas must be non-negative")
    return total


def _has_frozen_orbitals(frozen) -> bool:
    """Return whether a PySCF-style frozen-orbital specification is active."""

    if frozen is None:
        return False
    array = np.asarray(frozen)
    if array.ndim == 0:
        return bool(array.item())
    return bool(array.size)


def _normalize_eris_basis(eris_basis) -> str:
    if not isinstance(eris_basis, str) or eris_basis not in _ERIS_BASES:
        choices = ", ".join(sorted(_ERIS_BASES))
        raise ValueError(f"eris_basis must be one of: {choices}")
    return eris_basis


def _normalize_denominator_mode(denominator_mode) -> str:
    if (
        not isinstance(denominator_mode, str)
        or denominator_mode not in _DENOMINATOR_MODES
    ):
        choices = ", ".join(sorted(_DENOMINATOR_MODES))
        raise ValueError(f"denominator_mode must be one of: {choices}")
    return denominator_mode


def _normalize_contraction_backend(contraction_backend) -> str:
    if (
        not isinstance(contraction_backend, str)
        or contraction_backend not in _CONTRACTION_BACKENDS
    ):
        choices = ", ".join(sorted(_CONTRACTION_BACKENDS))
        raise ValueError(f"contraction_backend must be one of: {choices}")
    return contraction_backend


def _validate_tblis_operand_dtypes(named_arrays):
    """Fail before TBLIS when an operand has unsupported base precision."""

    unsupported = []
    allowed = {np.dtype(np.float64), np.dtype(np.complex128)}
    for name, value in named_arrays:
        dtype = np.asarray(value).dtype
        if dtype not in allowed:
            unsupported.append(f"{name}={dtype}")
    if unsupported:
        details = ", ".join(unsupported)
        raise TypeError(
            "pytblis contractions require float64/complex128 operands; "
            f"unsupported base precision: {details}. Convert explicitly or "
            "use contraction_backend='numpy'."
        )


@lru_cache(maxsize=2)
def _wick_einsum_namespace(contraction_backend):
    """Return the late-bound ``np.einsum`` namespace for Wick source."""

    contraction_backend = _normalize_contraction_backend(contraction_backend)
    if contraction_backend == "numpy":
        einsum = np.einsum
    else:
        try:
            from pytblis import einsum
        except ImportError as error:  # pragma: no cover - required dependency
            raise ImportError(
                "contraction_backend='pytblis' requires the pytblis package"
            ) from error
    return SimpleNamespace(einsum=einsum)


# Each expression is the fixed-free-index component of P_omega H |Phi>.
# The factors and signs follow the unantisymmetrized Hamiltonian above.  In
# particular, a fixed pair is represented by its explicit exchange
# difference, whereas a fully summed active pair carries no extra 1/2.
# These are the operator counterparts of SI Sections I.E--I.L.
_PERTURBER_EXPRESSIONS = {
    # S_ij,rs^(0), SI Eqs. (13)--(14); i<j and r<s.
    "ijrs": (
        "w[rsij] C[r] C[s] D[j] D[i]\n"
        " - w[rsji] C[r] C[s] D[j] D[i]"
    ),
    # S_i,rs^(-1), SI Eqs. (21)--(23); r<s.
    "rsi": (
        "SUM <a> w[rsia] C[r] C[s] D[a] D[i]\n"
        " - SUM <a> w[sria] C[r] C[s] D[a] D[i]"
    ),
    # S_ij,r^(1), SI Eqs. (15)--(17); i<j.
    "ijr": (
        "SUM <a> w[raij] C[r] C[a] D[j] D[i]\n"
        " - SUM <a> w[raji] C[r] C[a] D[j] D[i]"
    ),
    # S_rs^(-2), SI Eqs. (31)--(33); r<s.
    "rs": "SUM <ab> w[rsab] C[r] C[s] D[b] D[a]",
    # S_ij^(2), SI Eqs. (18)--(20); i<j.
    "ij": "SUM <ab> w[abij] C[a] C[b] D[j] D[i]",
    # S_i,r^(0), SI Eqs. (24)--(27).
    "ir": (
        "SUM <ab> w[raib] C[r] C[a] D[b] D[i]\n"
        " - SUM <ab> w[rabi] C[r] C[a] D[b] D[i]\n"
        " + h[ri] C[r] D[i]"
    ),
    # S_r^(-1), SI Eqs. (34)--(36).
    "r": (
        "SUM <abc> w[rabc] C[r] C[a] D[c] D[b]\n"
        " + SUM <a> h[ra] C[r] D[a]"
    ),
    # S_i^(1), SI Eqs. (28)--(30).  The dummy ordering uses the AAIA block
    # without invoking an ERI permutation rule.
    "i": (
        "SUM <abc> w[baic] C[b] C[a] D[c] D[i]\n"
        " + SUM <a> h[ai] C[a] D[i]"
    ),
}

_PAIR_RESTRICTIONS = {
    "ijrs": ((0, 1), (2, 3)),
    "rsi": ((0, 1),),
    "ijr": ((0, 1),),
    "rs": ((0, 1),),
    "ij": ((0, 1),),
    "ir": (),
    "r": (),
    "i": (),
}

_H1_KEYS = ("AA", "AI", "EI", "EA")
_W_KEYS = (
    "AAAA",
    "EAAA",
    "EAIA",
    "EAAI",
    "AAIA",
    "EEIA",
    "EAII",
    "EEAA",
    "AAII",
    "EEII",
)


@dataclass(frozen=True)
class _WickERIBlocks:
    """Detached integral blocks required by the generated Wick equations."""

    ncore: int
    ncas: int
    nvirt: int
    h1eff_blocks: dict[str, np.ndarray]
    phys_blocks: dict[str, np.ndarray]

    @property
    def nocc(self):
        return self.ncore + self.ncas

    @property
    def nmo(self):
        return self.nocc + self.nvirt

    @property
    def nbytes(self):
        return int(
            sum(array.nbytes for array in self.h1eff_blocks.values())
            + sum(array.nbytes for array in self.phys_blocks.values())
        )

    def get_h1eff(self, key):
        return self.h1eff_blocks[key]

    def get_phys(self, key):
        return self.phys_blocks[key]


def _compact_wick_eris(eris):
    """Copy only the integral blocks consumed by the Wick source."""

    if isinstance(eris, _WickERIBlocks):
        return eris
    return _WickERIBlocks(
        ncore=int(eris.ncore),
        ncas=int(eris.ncas),
        nvirt=int(eris.nvirt),
        h1eff_blocks={
            key: np.array(eris.get_h1eff(key), copy=True, order="C")
            for key in _H1_KEYS
        },
        phys_blocks={
            key: np.array(eris.get_phys(key), copy=True, order="C")
            for key in _W_KEYS
        },
    )


def _rotate_wick_eris(eris, rotation, *, block_atol=1.0e-10):
    """Rotate only required blocks under a core/active/virtual block unitary.

    CanonStep 1 leaves the active orbitals fixed and independently rotates
    the core and virtual spaces.  In that case every requested output block
    depends on the identically labelled input block, so forming a full
    ``nmo**4`` row tensor is unnecessary.
    """

    if not isinstance(eris, spinor_helper._SpinorERIs):
        raise TypeError("blockwise Wick rotation requires dense input ERIs")
    rotation = np.asarray(rotation)
    if rotation.shape != (eris.nmo, eris.nmo):
        raise ValueError("orbital rotation has the wrong shape")
    block_atol = float(block_atol)
    if not np.isfinite(block_atol) or block_atol < 0.0:
        raise ValueError("block_atol must be finite and non-negative")
    unitary_error = _maximum_abs(
        rotation.T.conj() @ rotation - np.eye(eris.nmo)
    )
    if unitary_error > block_atol:
        raise RuntimeError(
            "CanonStep-1 orbital transformation is not unitary; maximum "
            f"error={unitary_error:.3e}"
        )
    slices = {
        "I": slice(0, eris.ncore),
        "A": slice(eris.ncore, eris.nocc),
        "E": slice(eris.nocc, eris.nmo),
    }
    maximum_off_block = 0.0
    for left, right in itertools.product("IAE", repeat=2):
        if left != right:
            maximum_off_block = max(
                maximum_off_block,
                _maximum_abs(rotation[slices[left], slices[right]]),
            )
    if maximum_off_block > block_atol:
        raise RuntimeError(
            "CanonStep-1 rotation mixes core, active, or virtual spaces; "
            f"maximum off-block element={maximum_off_block:.3e}"
        )
    blocks = {
        label: rotation[space, space]
        for label, space in slices.items()
    }
    active_identity_error = _maximum_abs(
        blocks["A"] - np.eye(eris.ncas)
    )
    if active_identity_error > block_atol:
        raise RuntimeError(
            "CanonStep-1 rotation changed the active basis; maximum active "
            f"rotation error={active_identity_error:.3e}"
        )
    h1eff_blocks = {}
    for key in _H1_KEYS:
        left, right = (blocks[label] for label in key)
        h1eff_blocks[key] = np.ascontiguousarray(
            np.einsum(
                "ap,ab,bq->pq",
                left.conj(),
                eris.get_h1eff(key),
                right,
                optimize=True,
            )
        )
    phys_blocks = {}
    for key in _W_KEYS:
        first, second, third, fourth = (blocks[label] for label in key)
        phys_blocks[key] = np.ascontiguousarray(
            np.einsum(
                "ap,bq,cr,ds,abcd->pqrs",
                first.conj(),
                second.conj(),
                third,
                fourth,
                eris.get_phys(key),
                optimize=True,
            )
        )
    return _WickERIBlocks(
        ncore=int(eris.ncore),
        ncas=int(eris.ncas),
        nvirt=int(eris.nvirt),
        h1eff_blocks=h1eff_blocks,
        phys_blocks=phys_blocks,
    )

def _block2_wick_types():
    try:
        from block2 import (
            MapPStrIntVectorWickPermutation,
            MapWickIndexTypesSet,
            VectorWickIndex,
            VectorWickString,
            VectorWickTensor,
            WickExpr,
            WickIndex,
            WickIndexTypes,
            WickString,
            WickTensor,
            WickTensorTypes,
        )
    except ImportError as error:  # pragma: no cover - installation error
        raise RuntimeError(
            "block2 must be installed with the internal-contraction Wick bindings"
        ) from error
    return {
        "MapPStrIntVectorWickPermutation": MapPStrIntVectorWickPermutation,
        "MapWickIndexTypesSet": MapWickIndexTypesSet,
        "VectorWickIndex": VectorWickIndex,
        "VectorWickString": VectorWickString,
        "VectorWickTensor": VectorWickTensor,
        "WickExpr": WickExpr,
        "WickIndex": WickIndex,
        "WickIndexTypes": WickIndexTypes,
        "WickString": WickString,
        "WickTensor": WickTensor,
        "WickTensorTypes": WickTensorTypes,
    }


def _wick_parsers(types):
    index_types = types["WickIndexTypes"]
    index = types["WickIndex"]
    index_map = types["MapWickIndexTypesSet"]()
    index_map[index_types.Inactive] = index.parse_set("ijklmnoz")
    index_map[index_types.Active] = index.parse_set("abcdefghpq")
    index_map[index_types.External] = index.parse_set("rstuvwxy")

    # Deliberately empty.  qc_phys/qc_chem contain ordinary permutations that
    # are false for general complex spinors when the required conjugation is
    # omitted.  Hermitian ERI identities are numerical checks, not Wick rules.
    permutation_map = types["MapPStrIntVectorWickPermutation"]()
    parse = lambda expression: types["WickExpr"].parse(
        expression, index_map, permutation_map
    )
    parse_tensor = lambda expression: types["WickTensor"].parse(
        expression, index_map, permutation_map
    )
    return parse, parse_tensor


def _conjugate_with_coefficients(expression, types):
    """Hermitian-conjugate operators and explicitly rename coefficient data.

    Block2 0.5.4rc16 reverses/adjoints C/D in ``WickExpr.conjugate()`` but
    intentionally does not conjugate ordinary coefficient tensors.  The
    renamed ``hc`` and ``wc`` arrays are bound to elementwise conjugates at
    execution time.
    """

    tensor_type = types["WickTensorTypes"]
    result = expression.conjugate()
    names = {"h": "hc", "w": "wc"}
    for term in result.terms:
        for tensor in term.tensors:
            if tensor.type == tensor_type.Tensor and tensor.name in names:
                tensor.name = names[tensor.name]
    return result


def _vacuum_reduce(expression):
    """Contract the external vacuum and filled inactive determinant."""

    return (
        expression.expand()
        .remove_external()
        .remove_inactive()
        .simplify()
    )


def _lower_active_operators(expression, types, *, include_overlap=False):
    """Lower active strings to raw RDMs, optionally including rank zero.

    Ordinary expectation values may leave an active-space identity implicit.
    A transition matrix element may not: its rank-zero density is the model
    state overlap.  ``include_overlap`` therefore appends an explicit scalar
    ``dm0[]`` tensor to every term without active creation/destruction
    operators.
    """

    tensor_type = types["WickTensorTypes"]
    vector_index = types["VectorWickIndex"]
    vector_string = types["VectorWickString"]
    vector_tensor = types["VectorWickTensor"]
    wick_expr = types["WickExpr"]
    wick_string = types["WickString"]
    wick_tensor = types["WickTensor"]

    lowered_terms = vector_string()
    for term in expression.terms:
        coefficients = vector_tensor()
        operators = []
        for tensor in term.tensors:
            if tensor.type in (
                tensor_type.CreationOperator,
                tensor_type.DestroyOperator,
            ):
                operators.append(tensor)
            else:
                coefficients.append(tensor)

        if operators:
            kinds = [tensor.type for tensor in operators]
            ncreation = kinds.count(tensor_type.CreationOperator)
            ndestroy = kinds.count(tensor_type.DestroyOperator)
            expected = [tensor_type.CreationOperator] * ncreation
            expected += [tensor_type.DestroyOperator] * ndestroy
            if ncreation != ndestroy or kinds != expected:
                raise RuntimeError(
                    "Wick left an active operator string that is not a "
                    f"particle RDM: {term!r}"
                )
            if ncreation > 4:
                raise RuntimeError(
                    "NEVPT2 Wick expansion requested an active "
                    f"{ncreation}-RDM; recheck the commutator: {term!r}"
                )
            indices = vector_index()
            for tensor in operators:
                indices.append(tensor.indices[0])
            coefficients.append(wick_tensor(f"dm{ncreation}", indices))
        elif include_overlap:
            coefficients.append(wick_tensor("dm0", vector_index()))

        lowered_terms.append(
            wick_string(coefficients, term.ctr_indices, term.factor)
        )
    return wick_expr(lowered_terms).simplify()


def _execution_context(eris, pdms, *, overlap=None):
    context = {
        "deltaAA": np.eye(eris.ncas),
        "deltaII": np.eye(eris.ncore),
        "deltaEE": np.eye(eris.nvirt),
    }
    for rank, density in enumerate(pdms, start=1):
        context[f"dm{rank}{'A' * (2 * rank)}"] = density
    if overlap is not None:
        context["dm0"] = np.asarray(overlap)
    for key in _H1_KEYS:
        value = eris.get_h1eff(key)
        context[f"h{key}"] = value
        context[f"hc{key}"] = value.conj()
    for key in _W_KEYS:
        value = eris.get_phys(key)
        context[f"w{key}"] = value
        context[f"wc{key}"] = value.conj()
    return context


def _free_index_shape(key: str, eris):
    return tuple(eris.ncore if char in "ij" else eris.nvirt for char in key)


def _strict_pair_mask(key: str, shape):
    mask = np.ones(shape, dtype=bool)
    if not shape or any(size == 0 for size in shape):
        return mask
    grid = np.indices(shape, sparse=True)
    for left, right in _PAIR_RESTRICTIONS[key]:
        mask &= grid[left] < grid[right]
    return mask


def _require_real(values, *, root, subspace, quantity, atol, rtol):
    values = np.asarray(values)
    if not np.all(np.isfinite(values)):
        raise FloatingPointError(
            f"root {root} subspace {subspace}: {quantity} contains "
            "non-finite values"
        )
    if not np.iscomplexobj(values):
        return values.astype(float, copy=False)
    limit = atol + rtol * np.maximum(1.0, np.abs(values.real))
    violation = np.abs(values.imag) - limit
    if np.any(violation > 0.0):
        maximum = float(np.max(np.abs(values.imag)))
        _warn_numerical(
            f"root {root} subspace {subspace}: {quantity} is not real; "
            f"maximum imaginary part = {maximum:.3e}"
        )
    return values.real
def _maximum_abs(array) -> float:
    array = np.asarray(array)
    return float(np.max(np.abs(array), initial=0.0))


def _leading_chunks(shape, dtype, work_memory):
    """Yield leading-index chunks whose dense temporaries fit in memory.

    A transposed RDM is generally non-contiguous, so flattening it may itself
    allocate the full tensor.  Leading-index tuples keep both the original and
    transposed operands as views while bounding every arithmetic temporary.
    """

    shape = tuple(int(size) for size in shape)
    if not shape or 0 in shape:
        yield (...,)
        return
    itemsize = np.dtype(dtype).itemsize
    # Allow two arithmetic temporaries (e.g. conjugate plus difference).
    max_elements = max(1, int(work_memory) // (2 * itemsize))
    trailing = int(np.prod(shape, dtype=np.int64))
    prefix_rank = 0
    while trailing > max_elements and prefix_rank < len(shape):
        trailing //= shape[prefix_rank]
        prefix_rank += 1
    if prefix_rank == 0:
        yield (...,)
        return
    suffix = (slice(None),) * (len(shape) - prefix_rank)
    for prefix in np.ndindex(shape[:prefix_rank]):
        yield prefix + suffix


def _all_finite_chunked(array, work_memory):
    array = np.asarray(array)
    return all(
        bool(np.all(np.isfinite(array[index])))
        for index in _leading_chunks(array.shape, array.dtype, work_memory)
    )


def _maximum_abs_chunked(array, work_memory) -> float:
    array = np.asarray(array)
    maximum = 0.0
    for index in _leading_chunks(array.shape, array.dtype, work_memory):
        maximum = max(maximum, _maximum_abs(array[index]))
    return maximum


def _maximum_abs_relation(
    left,
    right,
    *,
    sign,
    conjugate_right=False,
    work_memory,
) -> float:
    """Return max(abs(left + sign * right)) using bounded temporaries."""

    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape:
        raise ValueError("RDM relation operands have different shapes")
    maximum = 0.0
    for index in _leading_chunks(left.shape, left.dtype, work_memory):
        right_chunk = right[index]
        if conjugate_right:
            right_chunk = right_chunk.conj()
        difference = left[index] + sign * right_chunk
        maximum = max(maximum, _maximum_abs(difference))
    return maximum


def validate_pdms(
    pdms,
    ncas,
    nelec,
    *,
    atol=_DEFAULT_RDM_ATOL,
    rtol=_DEFAULT_RDM_RTOL,
    work_memory=_DEFAULT_RDM_WORK_MEMORY,
):
    """Validate raw SGF 1--4 particle RDM shapes, order, and contractions.

    ``work_memory`` bounds arithmetic temporaries used by the symmetry checks.
    This matters for a 16-spinor complex 4-RDM, whose dense storage alone is
    64 GiB.  It does not relax or sample any validation condition.

    Finite relation residuals beyond tolerance emit
    :class:`MRPTNumericalWarning` and are retained in the diagnostics.  Invalid
    shapes, nonnumeric arrays, and non-finite values remain hard errors.
    """

    if not isinstance(pdms, (tuple, list)) or len(pdms) != 4:
        raise ValueError("pdms must be a (dm1, dm2, dm3, dm4) sequence")
    ncas = int(ncas)
    nelec = int(nelec)
    work_memory = int(work_memory)
    if work_memory <= 0:
        raise ValueError("work_memory must be positive")
    checked = []
    diagnostics = {}
    for rank, density in enumerate(pdms, start=1):
        density = np.asarray(density)
        expected_shape = (ncas,) * (2 * rank)
        if density.shape != expected_shape:
            raise ValueError(
                f"dm{rank} must have shape {expected_shape}, got {density.shape}"
            )
        if not np.issubdtype(density.dtype, np.number):
            raise TypeError(f"dm{rank} must be numeric")
        if not _all_finite_chunked(density, work_memory):
            raise ValueError(f"dm{rank} contains non-finite values")

        creator_error = 0.0
        annihilator_error = 0.0
        for axis in range(rank - 1):
            creator_error = max(
                creator_error,
                _maximum_abs_relation(
                    density,
                    density.swapaxes(axis, axis + 1),
                    sign=1.0,
                    work_memory=work_memory,
                ),
            )
            annihilator_axis = rank + axis
            annihilator_error = max(
                annihilator_error,
                _maximum_abs_relation(
                    density,
                    density.swapaxes(
                        annihilator_axis, annihilator_axis + 1
                    ),
                    sign=1.0,
                    work_memory=work_memory,
                ),
            )

        hermitian_axes = tuple(range(2 * rank - 1, rank - 1, -1))
        hermitian_axes += tuple(range(rank - 1, -1, -1))
        hermiticity_error = _maximum_abs_relation(
            density,
            density.transpose(hermitian_axes),
            sign=-1.0,
            conjugate_right=True,
            work_memory=work_memory,
        )
        scale = max(1.0, _maximum_abs_chunked(density, work_memory))
        tolerance = atol + rtol * scale
        symmetry_gate_passed = bool(
            max(creator_error, annihilator_error, hermiticity_error)
            <= tolerance
        )
        if not symmetry_gate_passed:
            _warn_numerical(
                f"dm{rank} violates raw SGF antisymmetry/Hermiticity: "
                f"creator={creator_error:.3e}, annihilator={annihilator_error:.3e}, "
                f"Hermiticity={hermiticity_error:.3e}"
            )
        diagnostics[f"dm{rank}"] = {
            "shape": list(density.shape),
            "creator_antisymmetry_error": creator_error,
            "annihilator_antisymmetry_error": annihilator_error,
            "hermiticity_error": hermiticity_error,
            "symmetry_tolerance": float(tolerance),
            "symmetry_gate_passed": symmetry_gate_passed,
        }
        checked.append(density)

    trace = np.trace(checked[0])
    trace_error = abs(trace - nelec)
    trace_tolerance = atol + rtol * max(1, nelec)
    trace_gate_passed = bool(trace_error <= trace_tolerance)
    if not trace_gate_passed:
        _warn_numerical(
            f"dm1 trace is inconsistent with {nelec} electrons: "
            f"error={trace_error:.3e}"
        )
    diagnostics["dm1"]["trace_error"] = float(trace_error)
    diagnostics["dm1"]["trace"] = float(trace.real)
    diagnostics["dm1"]["trace_imag"] = float(trace.imag)
    diagnostics["dm1"]["trace_tolerance"] = float(trace_tolerance)
    diagnostics["dm1"]["trace_gate_passed"] = trace_gate_passed

    for rank in range(2, 5):
        contracted = np.trace(
            checked[rank - 1], axis1=rank - 1, axis2=rank
        )
        expected = (nelec - rank + 1) * checked[rank - 2]
        error = _maximum_abs_relation(
            contracted,
            expected,
            sign=-1.0,
            work_memory=work_memory,
        )
        scale = max(1.0, _maximum_abs_chunked(expected, work_memory))
        contraction_tolerance = atol + rtol * scale
        contraction_gate_passed = bool(error <= contraction_tolerance)
        if not contraction_gate_passed:
            _warn_numerical(
                f"dm{rank}->dm{rank - 1} particle-number contraction "
                f"failed: error={error:.3e}"
            )
        diagnostics[f"dm{rank}"]["contraction_error"] = error
        diagnostics[f"dm{rank}"]["contraction_tolerance"] = float(
            contraction_tolerance
        )
        diagnostics[f"dm{rank}"][
            "contraction_gate_passed"
        ] = contraction_gate_passed
    return tuple(checked), diagnostics


def _root_ket(dmrgci, root):
    driver = getattr(dmrgci, "driver", None)
    kets = getattr(dmrgci, "kets", None)
    if driver is None or kets is None:
        raise RuntimeError(
            "the converged pyblock2 driver/MPS must remain open while forming RDMs"
        )
    root = int(root)
    if not 0 <= root < len(kets):
        raise IndexError(
            f"DMRG root {root} is outside the available range [0, {len(kets)})"
        )
    return driver, kets[root]


def _make_rdm(dmrgci, root, rank):
    driver, ket = _root_ket(dmrgci, root)
    method = getattr(driver, f"get_{rank}pdm", None)
    kwargs = {
        "site_type": int(getattr(dmrgci, "npdm_site_type", 0)),
        "cutoff": float(getattr(dmrgci, "npdm_cutoff", 1.0e-24)),
    }
    if callable(method):
        density = method(ket, **kwargs)
    else:  # Compatibility fallback, using the inspected installed signature.
        method = getattr(driver, "get_npdm", None)
        if not callable(method):
            raise RuntimeError(
                f"the installed pyblock2 driver has no {rank}-RDM interface"
            )
        density = method(ket, pdm_type=rank, **kwargs)
    density = np.asarray(density)
    ncas = int(getattr(dmrgci, "ncas"))
    expected_shape = (ncas,) * (2 * rank)
    if density.shape != expected_shape:
        raise ValueError(
            f"Block2 root {root} {rank}-RDM has shape {density.shape}; "
            f"expected raw SGF shape {expected_shape}"
        )
    if not np.iscomplexobj(density):
        density = density.astype(np.complex128)
    return density


def make_rdm1(dmrgci, root=0):
    """Return raw ``<C[p]D[q]>`` for one selected SGF MPS root."""

    return _make_rdm(dmrgci, root, 1)


def make_rdm2(dmrgci, root=0):
    """Return raw ``<C[p]C[q]D[s]D[r]>`` for one selected MPS root."""

    return _make_rdm(dmrgci, root, 2)


def make_rdm3(dmrgci, root=0):
    """Return the explicit raw SGF 3-particle RDM for one MPS root."""

    return _make_rdm(dmrgci, root, 3)


def make_rdm4(dmrgci, root=0):
    """Return the explicit raw SGF 4-particle RDM for one MPS root."""

    return _make_rdm(dmrgci, root, 4)


def make_dm1234(dmrgci, root=0):
    """Obtain explicit, unapproximated raw 1--4 RDMs for one selected root."""

    return tuple(_make_rdm(dmrgci, root, rank) for rank in range(1, 5))


def _rotate_eris(eris, rotation):
    rotation = np.asarray(rotation)
    h1e = rotation.T.conj() @ eris.h1e @ rotation
    h1eff = rotation.T.conj() @ eris.h1eff @ rotation
    eri = np.einsum(
        "ap,bq,cr,ds,abcd->pqrs",
        rotation.conj(),
        rotation,
        rotation.conj(),
        rotation,
        eris.pppp,
        optimize=True,
    )
    return spinor_helper.init_eris(
        h1e,
        eri,
        eris.ncore,
        eris.ncas,
        h1eff=h1eff,
        frozen=0,
        copy=False,
        check=True,
    )


_DEFAULT_AO2MO_ROUNDOFF_FACTOR = 1.0
_SUPPORTED_AO2MO_DTYPES = frozenset(
    np.dtype(dtype)
    for dtype in (np.float32, np.float64, np.complex64, np.complex128)
)
_SYMMETRY_COMPARISON_PATHS = 2
_REAL_OPS_PER_REAL_MULTIPLY_ADD = 2
_REAL_OPS_PER_COMPLEX_MULTIPLY_ADD = 8
def _require_supported_ao2mo_dtype(values, *, name):
    dtype = np.asarray(values).dtype
    if dtype not in _SUPPORTED_AO2MO_DTYPES:
        supported = ", ".join(
            sorted(item.name for item in _SUPPORTED_AO2MO_DTYPES)
        )
        raise TypeError(
            f"{name} has unsupported dtype {dtype}; supported AO2MO dtypes: "
            f"{supported}"
        )
    return dtype


def _roundoff_gamma(operation_count, epsilon, *, name):
    try:
        accumulated_roundoff = int(operation_count) * float(epsilon)
    except OverflowError as error:
        raise ValueError(
            f"{name} roundoff model requires 0 < operation_count * eps < 1"
        ) from error
    if (
        not np.isfinite(accumulated_roundoff)
        or not 0.0 < accumulated_roundoff < 1.0
    ):
        raise ValueError(
            f"{name} roundoff model requires 0 < operation_count * eps < 1"
        )
    return accumulated_roundoff / (1.0 - accumulated_roundoff)


def _ao2mo_roundoff_policy(
    values,
    *,
    accumulation_length,
    contraction_stages,
    roundoff_factor,
):
    """Return an operation-aware floating-point gate and its provenance.

    ``gamma(k) = k*eps / (1 - k*eps)`` is the standard accumulation factor for
    ``k`` rounded operations.  AO-to-MO transformation applies two dense
    contractions to a one-electron tensor and four to a two-electron tensor,
    each accumulating over the AO dimension.  A real multiply-add is modelled
    as two real rounded operations and a complex multiply-add as at most eight.
    A symmetry residual compares two independently rounded transformation
    paths, hence the explicit factor of two.  Scaling ``gamma(k)`` by
    ``max(1, maximum_absolute_value)`` gives an operation-aware gate, not a
    rigorous forward error bound in the presence of cancellation.
    ``roundoff_factor`` is an explicit backend-specific multiplier on this
    model.
    """

    if isinstance(accumulation_length, (bool, np.bool_)) or not isinstance(
        accumulation_length, (int, np.integer)
    ):
        raise TypeError("roundoff_accumulation_length must be an integer")
    accumulation_length = int(accumulation_length)
    if accumulation_length <= 0:
        raise ValueError("roundoff_accumulation_length must be positive")
    if isinstance(contraction_stages, (bool, np.bool_)) or not isinstance(
        contraction_stages, (int, np.integer)
    ):
        raise TypeError("contraction_stages must be an integer")
    contraction_stages = int(contraction_stages)
    if contraction_stages <= 0:
        raise ValueError("contraction_stages must be positive")
    try:
        roundoff_factor = float(roundoff_factor)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "roundoff_factor must be finite and positive"
        ) from error
    if not np.isfinite(roundoff_factor) or roundoff_factor <= 0.0:
        raise ValueError("roundoff_factor must be finite and positive")

    input_dtype = _require_supported_ao2mo_dtype(values, name="values")
    is_complex = bool(np.issubdtype(input_dtype, np.complexfloating))
    real_dtype = np.asarray(np.asarray(values).real).dtype
    epsilon = float(np.finfo(real_dtype).eps)
    real_operations_per_multiply_add = (
        _REAL_OPS_PER_COMPLEX_MULTIPLY_ADD
        if is_complex
        else _REAL_OPS_PER_REAL_MULTIPLY_ADD
    )
    operation_count = (
        contraction_stages
        * real_operations_per_multiply_add
        * accumulation_length
    )
    gamma = _roundoff_gamma(operation_count, epsilon, name="AO2MO")
    with np.errstate(over="ignore", invalid="ignore"):
        maximum_absolute_value = _maximum_abs(values)
    if not np.isfinite(maximum_absolute_value):
        raise ValueError(
            "AO2MO maximum absolute value is non-finite; the roundoff gate "
            "cannot be constructed"
        )
    effective_scale = max(1.0, maximum_absolute_value)
    gate = (
        roundoff_factor
        * _SYMMETRY_COMPARISON_PATHS
        * gamma
        * effective_scale
    )
    if not np.isfinite(gate):
        raise ValueError(
            "AO2MO roundoff gate is non-finite; reduce roundoff_factor or "
            "the transformed-integral scale"
        )
    return float(gate), {
        "formula": (
            "roundoff_factor * symmetry_comparison_paths * "
            "gamma(operation_count) * effective_scale"
        ),
        "input_dtype": input_dtype.name,
        "is_complex": is_complex,
        "real_dtype": np.dtype(real_dtype).name,
        "machine_epsilon": epsilon,
        "maximum_absolute_value": maximum_absolute_value,
        "effective_scale": effective_scale,
        "accumulation_length": accumulation_length,
        "contraction_stages": contraction_stages,
        "real_operations_per_multiply_add": real_operations_per_multiply_add,
        "operation_count": operation_count,
        "gamma": float(gamma),
        "symmetry_comparison_paths": _SYMMETRY_COMPARISON_PATHS,
        "symmetry_comparison_path_provenance": (
            "raw symmetry residual compares two independently rounded "
            "AO2MO paths"
        ),
        "roundoff_factor": roundoff_factor,
    }


def _project_dense_mo_integrals(
    h1e,
    eri,
    *,
    roundoff_accumulation_length=None,
    roundoff_factor=_DEFAULT_AO2MO_ROUNDOFF_FACTOR,
):
    """Project AO2MO roundoff back onto the physical spinor symmetries.

    ``s1`` means that the full tensor is stored; it does not remove the
    Coulomb identities ``(pq|rs) = (rs|pq)`` and
    ``(pq|rs) = (qp|sr)*``.  The raw residual is audited before projection;
    values beyond the operation-aware roundoff estimate emit
    ``MRPTNumericalWarning`` so an axis/conjugation problem is visible while
    the symmetry projection can still proceed.  ``eri`` is a private writable
    AO2MO buffer and is consumed as scratch to keep the production peak at two
    full tensors.  A read-only injected buffer is copied first and can
    therefore transiently require a third full tensor.
    """

    h1e = np.asarray(h1e)
    eri = np.asarray(eri)
    if h1e.ndim != 2 or h1e.shape[0] != h1e.shape[1]:
        raise ValueError("h1e must be square before symmetry projection")
    nmo = h1e.shape[0]
    if eri.shape != (nmo,) * 4:
        raise ValueError("eri has the wrong shape before symmetry projection")
    _require_supported_ao2mo_dtype(h1e, name="h1e")
    _require_supported_ao2mo_dtype(eri, name="eri")
    if not np.all(np.isfinite(h1e)):
        raise ValueError("raw transformed h1e contains non-finite values")
    if not eri.flags.writeable:
        eri = np.array(eri, copy=True)
    if roundoff_accumulation_length is None:
        roundoff_accumulation_length = nmo

    h1e_scale = _maximum_abs(h1e)
    h1e_error = _maximum_abs(h1e - h1e.T.conj())
    eri_scale = 0.0
    pair_error = 0.0
    conjugate_error = 0.0
    for p in range(nmo):
        block = eri[p]
        if not np.all(np.isfinite(block)):
            raise ValueError("raw transformed eri contains non-finite values")
        eri_scale = max(eri_scale, _maximum_abs(block))
        pair = eri[:, :, p, :].transpose(2, 0, 1)
        conjugate = eri[:, p, :, :].transpose(0, 2, 1).conj()
        pair_error = max(pair_error, _maximum_abs(block - pair))
        conjugate_error = max(
            conjugate_error, _maximum_abs(block - conjugate)
        )

    # Model plausible roundoff from the raw AO2MO contractions.  The policy
    # derives epsilon from each tensor dtype, scales with its output magnitude,
    # and uses the actual AO accumulation length supplied by the transformation.
    # Gross index/conjugation residuals remain visible as warnings before
    # projection, while their complete numerical values remain in diagnostics.
    h1e_tolerance, h1e_roundoff_policy = _ao2mo_roundoff_policy(
        h1e,
        accumulation_length=roundoff_accumulation_length,
        contraction_stages=2,
        roundoff_factor=roundoff_factor,
    )
    eri_tolerance, eri_roundoff_policy = _ao2mo_roundoff_policy(
        eri,
        accumulation_length=roundoff_accumulation_length,
        contraction_stages=4,
        roundoff_factor=roundoff_factor,
    )
    if h1e_error > h1e_tolerance:
        _warn_numerical(
            "raw transformed h1e violates Hermiticity beyond roundoff: "
            f"error={h1e_error:.3e}, tolerance={h1e_tolerance:.3e}"
        )
    if max(pair_error, conjugate_error) > eri_tolerance:
        _warn_numerical(
            "raw transformed eri violates complex Coulomb symmetry beyond "
            f"roundoff: pair={pair_error:.3e}, "
            f"conjugate={conjugate_error:.3e}, "
            f"tolerance={eri_tolerance:.3e}"
        )

    # Scale before every pairwise sum so symmetry-perfect values near the dtype
    # maximum remain finite.  ``eri`` is the documented scratch tensor.
    projected_h1e = np.array(h1e, copy=True)
    projected_h1e *= 0.5
    projected_h1e += 0.5 * h1e.T.conj()
    if not np.all(np.isfinite(projected_h1e)):
        raise FloatingPointError(
            "h1e symmetry projection produced non-finite values"
        )

    eri *= 0.5
    projected_eri = np.empty_like(eri)
    np.add(eri, eri.transpose(2, 3, 0, 1), out=projected_eri)
    projected_eri *= 0.5
    np.conjugate(projected_eri.transpose(1, 0, 3, 2), out=eri)
    projected_eri += eri

    post_h1e_error = _maximum_abs(projected_h1e - projected_h1e.T.conj())
    post_pair_error = 0.0
    post_conjugate_error = 0.0
    for p in range(nmo):
        block = projected_eri[p]
        if not np.all(np.isfinite(block)):
            raise FloatingPointError(
                "eri symmetry projection produced non-finite values"
            )
        pair = projected_eri[:, :, p, :].transpose(2, 0, 1)
        conjugate = (
            projected_eri[:, p, :, :].transpose(0, 2, 1).conj()
        )
        post_pair_error = max(
            post_pair_error, _maximum_abs(block - pair)
        )
        post_conjugate_error = max(
            post_conjugate_error, _maximum_abs(block - conjugate)
        )

    # Each stable pair average uses at most four rounded scalar operations per
    # component.  The allowances below conservatively cover one such average
    # for h1e and the two sequential averages forming the four-element ERI
    # group projection.  They describe projection arithmetic only; the raw
    # AO2MO gate above has its own operation-aware policy.
    h1e_projection_arithmetic_allowance = _roundoff_gamma(
        4,
        h1e_roundoff_policy["machine_epsilon"],
        name="h1e projection",
    ) * h1e_roundoff_policy["effective_scale"]
    eri_projection_arithmetic_allowance = _roundoff_gamma(
        8,
        eri_roundoff_policy["machine_epsilon"],
        name="eri projection",
    ) * eri_roundoff_policy["effective_scale"]

    diagnostics = {
        "h1e": {
            "raw_hermiticity_error": h1e_error,
            "maximum_absolute_value": h1e_scale,
            "roundoff_gate": h1e_tolerance,
            "roundoff_gate_passed": bool(h1e_error <= h1e_tolerance),
            "roundoff_policy": h1e_roundoff_policy,
            "projection_change_upper_bound": 0.5 * h1e_error
            + h1e_projection_arithmetic_allowance,
            "projection_arithmetic_allowance": (
                h1e_projection_arithmetic_allowance
            ),
            "post_projection_hermiticity_error": post_h1e_error,
        },
        "eri": {
            "raw_pair_exchange_error": pair_error,
            "raw_conjugate_exchange_error": conjugate_error,
            "maximum_absolute_value": eri_scale,
            "roundoff_gate": eri_tolerance,
            "roundoff_gate_passed": bool(
                max(pair_error, conjugate_error) <= eri_tolerance
            ),
            "roundoff_policy": eri_roundoff_policy,
            "projection_change_upper_bound": 0.5
            * (pair_error + conjugate_error)
            + eri_projection_arithmetic_allowance,
            "projection_arithmetic_allowance": (
                eri_projection_arithmetic_allowance
            ),
            "post_projection_pair_exchange_error": post_pair_error,
            "post_projection_conjugate_exchange_error": post_conjugate_error,
        },
    }
    return projected_h1e, projected_eri, diagnostics


def _dense_eris_from_mc(
    mc,
    mo_coeff,
    *,
    roundoff_factor=_DEFAULT_AO2MO_ROUNDOFF_FACTOR,
):
    from pyscf.ao2mo import nrr_outcore

    mo_coeff = np.asarray(mo_coeff)
    hcore = np.asarray(mc.get_hcore())
    h1e = mo_coeff.T.conj() @ hcore @ mo_coeff
    dense = nrr_outcore.full_iofree(
        mc.mol,
        mo_coeff,
        motype="j-spinor",
        verbose=getattr(mc, "verbose", logger.NOTE),
    )
    nmo = mo_coeff.shape[1]
    raw_eri = np.asarray(dense).reshape((nmo,) * 4)
    del dense
    h1e, eri, symmetry_diagnostics = _project_dense_mo_integrals(
        h1e,
        raw_eri,
        roundoff_accumulation_length=mo_coeff.shape[0],
        roundoff_factor=roundoff_factor,
    )
    del raw_eri
    result = spinor_helper.init_eris(
        h1e,
        eri,
        int(mc.ncore),
        int(mc.ncas),
        frozen=0,
        copy=False,
        check=True,
    )
    result.symmetry_diagnostics = symmetry_diagnostics
    return result


def _reference_energy(mc, root):
    if hasattr(mc, "e_states") and getattr(mc, "e_states") is not None:
        energies = np.asarray(mc.e_states)
        if energies.ndim:
            return float(energies[root])
    energies = np.asarray(mc.e_tot)
    if energies.ndim:
        return float(energies[root])
    if int(getattr(getattr(mc, "fcisolver", None), "nroots", 1)) > 1:
        solver_energies = np.asarray(getattr(mc.fcisolver, "e_tot", []))
        if solver_energies.ndim and len(solver_energies) > root:
            # DMRGCI.e_tot already includes the active-space core constant
            # used by its most recent CASSCF Hamiltonian.
            return float(solver_energies[root])
        raise RuntimeError("state-specific CASSCF reference energies are unavailable")
    return float(energies)



def semicanonicalize(
    mc,
    mo_coeff,
    dm1,
    root,
    *,
    canonicalized=False,
    mo_energy=None,
    verbose=None,
):
    """Return CanonStep-1 orbitals and energies without rotating the CAS.

    When canonicalized is true, validate and reuse the supplied orbital
    energies. Otherwise call the reference object's canonicalizer for the
    selected root and reject any active-orbital rotation, because the RDMs
    remain expressed in the original active basis.
    """

    mo_coeff = np.asarray(mo_coeff)
    if canonicalized:
        if mo_energy is None:
            mo_energy = getattr(mc, "mo_energy", None)
        if mo_energy is None:
            raise ValueError(
                "canonicalized=True requires prepared spinor orbital energies"
            )
        energies = np.asarray(mo_energy)
        if energies.shape != (mo_coeff.shape[1],):
            raise ValueError("mo_energy has the wrong shape")
        if not np.all(np.isfinite(energies)):
            raise ValueError("mo_energy contains non-finite values")
        if np.iscomplexobj(energies) and _maximum_abs(energies.imag) > 1e-12:
            _warn_numerical(
                "semicanonical orbital energies have imaginary components; "
                "using their real parts"
            )
        return mo_coeff, np.asarray(energies.real, dtype=float)

    ci = _select_root_ci(
        getattr(mc, "ci", None),
        root,
        nroots=getattr(getattr(mc, "fcisolver", None), "nroots", 1),
    )
    if verbose is None:
        verbose = getattr(mc, "verbose", logger.NOTE)
    rotated, _ci, energies = mc.canonicalize(
        mo_coeff,
        ci=ci,
        cas_natorb=False,
        casdm1=dm1,
        verbose=verbose,
    )
    rotated = np.asarray(rotated)
    ncore = int(mc.ncore)
    nocc = ncore + int(mc.ncas)
    active_change = _maximum_abs(
        rotated[:, ncore:nocc] - mo_coeff[:, ncore:nocc]
    )
    if active_change > 1.0e-12:
        raise RuntimeError(
            "root-specific semicanonicalization changed active orbitals "
            f"by {active_change:.3e}; the MPS/RDM basis would be invalid"
        )
    return rotated, np.asarray(energies, dtype=float)


def adjoint_transition_pdm(dm):
    """Return the reverse raw-SGF transition PDM.

    For axes ``p1,...,pk,qk,...,q1`` the adjoint relation reverses all
    ``2*k`` axes and complex conjugates the tensor.
    """

    density = np.asarray(dm)
    if density.ndim == 0 or density.ndim % 2:
        raise ValueError("a transition PDM must have a positive even rank")
    axes = tuple(range(density.ndim - 1, -1, -1))
    return density.transpose(axes).conj()


def _transition_root_kets(dmrgci, bra_root, ket_root):
    bra_driver, bra = _root_ket(dmrgci, bra_root)
    ket_driver, ket = _root_ket(dmrgci, ket_root)
    if bra_driver is not ket_driver:
        raise RuntimeError("transition MPS roots do not share one Block2 driver")
    return bra_driver, bra, ket


def _make_transition_rdm(dmrgci, bra_root, ket_root, rank):
    bra_root = int(bra_root)
    ket_root = int(ket_root)
    rank = int(rank)
    if rank not in (1, 2, 3, 4):
        raise ValueError("transition RDM rank must be between one and four")
    if bra_root == ket_root:
        return _make_rdm(dmrgci, bra_root, rank)

    driver, bra, ket = _transition_root_kets(dmrgci, bra_root, ket_root)
    kwargs = {
        "site_type": int(getattr(dmrgci, "npdm_site_type", 0)),
        "cutoff": float(getattr(dmrgci, "npdm_cutoff", 1.0e-24)),
    }
    method = getattr(driver, f"get_trans_{rank}pdm", None)
    if callable(method):
        density = method(bra, ket, **kwargs)
    else:
        method = getattr(driver, "get_npdm", None)
        if not callable(method):
            raise RuntimeError(
                f"the installed pyblock2 driver has no transition {rank}-RDM "
                "interface"
            )
        # Installed block2 0.5.4rc16 signature:
        # get_npdm(ket, pdm_type=1, bra=None, ..., site_type=0, cutoff=1e-24)
        density = method(ket, pdm_type=rank, bra=bra, **kwargs)
    density = np.asarray(density)
    ncas = int(getattr(dmrgci, "ncas"))
    expected_shape = (ncas,) * (2 * rank)
    if density.shape != expected_shape:
        raise ValueError(
            f"Block2 transition ({bra_root},{ket_root}) {rank}-RDM has "
            f"shape {density.shape}; expected raw SGF shape {expected_shape}"
        )
    if not np.issubdtype(density.dtype, np.number):
        raise TypeError("Block2 returned a non-numeric transition PDM")
    return np.asarray(density, dtype=np.complex128)


def make_transition_rdm1(dmrgci, bra_root, ket_root):
    """Return ``<bra|C[p]D[q]|ket>`` in raw SGF order."""

    return _make_transition_rdm(dmrgci, bra_root, ket_root, 1)


def make_transition_rdm2(dmrgci, bra_root, ket_root):
    """Return the explicit raw SGF transition 2-particle RDM."""

    return _make_transition_rdm(dmrgci, bra_root, ket_root, 2)


def make_transition_rdm3(dmrgci, bra_root, ket_root):
    """Return the explicit raw SGF transition 3-particle RDM."""

    return _make_transition_rdm(dmrgci, bra_root, ket_root, 3)


def make_transition_rdm4(dmrgci, bra_root, ket_root):
    """Return the explicit, uncompressed transition 4-particle RDM."""

    return _make_transition_rdm(dmrgci, bra_root, ket_root, 4)


def make_transition_dm1234(dmrgci, bra_root, ket_root):
    """Form raw transition 1--4 PDMs for one ordered MPS root pair."""

    return tuple(
        _make_transition_rdm(dmrgci, bra_root, ket_root, rank)
        for rank in range(1, 5)
    )


def _make_transition_dm123(dmrgci, bra_root, ket_root):
    """Form only the ranks required by Angeli Eq. (18) SC couplings."""

    return tuple(
        _make_transition_rdm(dmrgci, bra_root, ket_root, rank)
        for rank in range(1, 4)
    )


def _transition_contraction_pdms(pdms):
    """Normalize only production ranks 1--3 to the TBLIS complex ABI."""

    if not isinstance(pdms, (tuple, list)) or len(pdms) < 3:
        raise ValueError("transition contractions require RDM ranks 1--3")
    return tuple(
        np.asarray(density, dtype=np.complex128) for density in pdms[:3]
    )


def _complex_scalar(value, *, name):
    array = np.asarray(value)
    if array.size != 1 or not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be one numeric scalar")
    result = complex(array.reshape(()))
    if not np.isfinite(result):
        raise ValueError(f"{name} is not finite")
    return result


def _finite_nonnegative(value, *, name):
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def _make_transition_overlap(dmrgci, bra_root, ket_root, *, dm1=None):
    driver, bra, ket = _transition_root_kets(dmrgci, bra_root, ket_root)
    get_identity = getattr(driver, "get_identity_mpo", None)
    expectation = getattr(driver, "expectation", None)
    if callable(get_identity) and callable(expectation):
        identity = get_identity()
        overlap = expectation(bra, identity, ket)
        return _complex_scalar(overlap, name="model-state overlap"), "identity_mpo"

    if dm1 is None:
        dm1 = make_transition_rdm1(dmrgci, bra_root, ket_root)
    nelecas = getattr(dmrgci, "nelecas", None)
    if nelecas is None:
        raise RuntimeError(
            "cannot infer overlap from transition dm1 without active electron count"
        )
    nelec = _total_nelec(nelecas)
    if nelec == 0:
        if int(bra_root) == int(ket_root):
            return 1.0 + 0.0j, "zero_electron_identity"
        raise RuntimeError("a zero-electron transition overlap needs identity MPO")
    overlap = np.trace(np.asarray(dm1)) / nelec
    return _complex_scalar(overlap, name="dm1-derived overlap"), "dm1_trace_fallback"


def make_transition_overlap(dmrgci, bra_root, ket_root):
    """Return ``<bra_root|ket_root>`` using the Block2 identity MPO."""

    return _make_transition_overlap(dmrgci, bra_root, ket_root)[0]


def _complex_pair(value):
    value = complex(value)
    return [float(value.real), float(value.imag)]


def validate_transition_pdms(
    pdms_ij,
    ncas,
    nelec,
    *,
    overlap_ij,
    pdms_ji=None,
    atol=_DEFAULT_RDM_ATOL,
    rtol=_DEFAULT_RDM_RTOL,
    work_memory=_DEFAULT_RDM_WORK_MEMORY,
):
    """Validate raw SGF transition ranks 1--3 or 1--4.

    This deliberately does not normalize a possible rank-4 tensor's dtype:
    doing so could create a hidden tens-of-GiB copy for larger active spaces.
    The production contraction path slices to ranks 1--3 and converts those
    arrays to complex128 explicitly at its backend boundary.

    Finite relation residuals beyond tolerance emit
    :class:`MRPTNumericalWarning` and are retained in the
    diagnostics.  Invalid shapes, nonnumeric arrays, and non-finite values
    remain hard errors.
    """

    if not isinstance(pdms_ij, (tuple, list)) or len(pdms_ij) not in (3, 4):
        raise ValueError(
            "pdms_ij must contain consecutive transition RDM ranks 1--3 "
            "or 1--4"
        )
    if pdms_ji is not None and (
        not isinstance(pdms_ji, (tuple, list))
        or len(pdms_ji) != len(pdms_ij)
    ):
        raise ValueError("pdms_ji must contain the same ranks as pdms_ij")
    ncas = int(ncas)
    nelec = int(nelec)
    work_memory = int(work_memory)
    if work_memory <= 0:
        raise ValueError("work_memory must be positive")
    overlap_ij = _complex_scalar(overlap_ij, name="overlap_ij")

    checked = []
    reverse_checked = [] if pdms_ji is not None else None
    diagnostics = {"overlap": _complex_pair(overlap_ij)}
    for rank, density in enumerate(pdms_ij, start=1):
        density = np.asarray(density)
        expected_shape = (ncas,) * (2 * rank)
        if density.shape != expected_shape:
            raise ValueError(
                f"transition dm{rank} must have shape {expected_shape}, "
                f"got {density.shape}"
            )
        if not np.issubdtype(density.dtype, np.number):
            raise TypeError(f"transition dm{rank} must be numeric")
        if not _all_finite_chunked(density, work_memory):
            raise ValueError(f"transition dm{rank} contains non-finite values")

        creator_error = 0.0
        annihilator_error = 0.0
        for axis in range(rank - 1):
            creator_error = max(
                creator_error,
                _maximum_abs_relation(
                    density,
                    density.swapaxes(axis, axis + 1),
                    sign=1.0,
                    work_memory=work_memory,
                ),
            )
            annihilator_axis = rank + axis
            annihilator_error = max(
                annihilator_error,
                _maximum_abs_relation(
                    density,
                    density.swapaxes(
                        annihilator_axis, annihilator_axis + 1
                    ),
                    sign=1.0,
                    work_memory=work_memory,
                ),
            )
        scale = max(1.0, _maximum_abs_chunked(density, work_memory))
        tolerance = atol + rtol * scale
        symmetry_gate_passed = bool(
            max(creator_error, annihilator_error) <= tolerance
        )
        if not symmetry_gate_passed:
            _warn_numerical(
                f"transition dm{rank} violates raw SGF antisymmetry: "
                f"creator={creator_error:.3e}, "
                f"annihilator={annihilator_error:.3e}"
            )
        rank_diagnostics = {
            "shape": list(density.shape),
            "creator_antisymmetry_error": creator_error,
            "annihilator_antisymmetry_error": annihilator_error,
            "symmetry_tolerance": float(tolerance),
            "symmetry_gate_passed": symmetry_gate_passed,
        }

        if pdms_ji is not None:
            reverse = np.asarray(pdms_ji[rank - 1])
            if reverse.shape != expected_shape:
                raise ValueError(
                    f"reverse transition dm{rank} has shape {reverse.shape}; "
                    f"expected {expected_shape}"
                )
            if not np.issubdtype(reverse.dtype, np.number):
                raise TypeError(f"reverse transition dm{rank} must be numeric")
            if not _all_finite_chunked(reverse, work_memory):
                raise ValueError(
                    f"reverse transition dm{rank} contains non-finite values"
                )
            axes = tuple(range(2 * rank - 1, -1, -1))
            adjoint_error = _maximum_abs_relation(
                reverse,
                density.transpose(axes),
                sign=-1.0,
                conjugate_right=True,
                work_memory=work_memory,
            )
            reverse_scale = max(
                scale, _maximum_abs_chunked(reverse, work_memory)
            )
            adjoint_tolerance = atol + rtol * max(1.0, reverse_scale)
            adjoint_gate_passed = bool(adjoint_error <= adjoint_tolerance)
            if not adjoint_gate_passed:
                _warn_numerical(
                    f"transition dm{rank} reverse-adjoint relation failed: "
                    f"error={adjoint_error:.3e}"
                )
            rank_diagnostics["reverse_adjoint_error"] = adjoint_error
            rank_diagnostics["reverse_adjoint_tolerance"] = float(
                adjoint_tolerance
            )
            rank_diagnostics[
                "reverse_adjoint_gate_passed"
            ] = adjoint_gate_passed
            reverse_checked.append(reverse)

        diagnostics[f"dm{rank}"] = rank_diagnostics
        checked.append(density)

    trace = np.trace(checked[0])
    trace_error = abs(trace - nelec * overlap_ij)
    trace_tolerance = atol + rtol * max(1.0, nelec * abs(overlap_ij))
    trace_gate_passed = bool(trace_error <= trace_tolerance)
    if not trace_gate_passed:
        _warn_numerical(
            "transition dm1 trace is inconsistent with electron number and "
            f"overlap: error={trace_error:.3e}"
        )
    diagnostics["dm1"]["trace"] = _complex_pair(trace)
    diagnostics["dm1"]["trace_error"] = float(trace_error)
    diagnostics["dm1"]["trace_tolerance"] = float(trace_tolerance)
    diagnostics["dm1"]["trace_gate_passed"] = trace_gate_passed

    for rank in range(2, len(checked) + 1):
        contracted = np.trace(
            checked[rank - 1], axis1=rank - 1, axis2=rank
        )
        expected = (nelec - rank + 1) * checked[rank - 2]
        error = _maximum_abs_relation(
            contracted,
            expected,
            sign=-1.0,
            work_memory=work_memory,
        )
        scale = max(1.0, _maximum_abs_chunked(expected, work_memory))
        contraction_tolerance = atol + rtol * scale
        contraction_gate_passed = bool(error <= contraction_tolerance)
        if not contraction_gate_passed:
            _warn_numerical(
                f"transition dm{rank}->dm{rank - 1} particle-number "
                f"contraction failed: error={error:.3e}"
            )
        diagnostics[f"dm{rank}"]["contraction_error"] = error
        diagnostics[f"dm{rank}"]["contraction_tolerance"] = float(
            contraction_tolerance
        )
        diagnostics[f"dm{rank}"][
            "contraction_gate_passed"
        ] = contraction_gate_passed

    return tuple(checked), diagnostics
