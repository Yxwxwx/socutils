#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
"""General-complex-spinor X2C time-dependent MPS NEVPT2.

The production implementation in this module follows Sokolov--Chan
t-NEVPT2 and Sokolov--Guo--Ronca--Chan t-MPS-NEVPT2.  In particular, its
source operators are parsed from the same ``_PERTURBER_EXPRESSIONS`` object
used by :mod:`x2cscnevpt2`; this module does not maintain a second hand-written
set of NEVPT2 source equations.

The public MPS driver is built in stages below.  The source parser and dense
numerical utilities are deliberately independent of Block2 so they can be
tested against an exact complex Fock-space oracle before any MPS propagation
is trusted.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from contextlib import contextmanager
import gc
import hashlib
import itertools
import math
import os
from pathlib import Path
import re
import resource
import time
from typing import Any, Iterable, Iterator, Sequence
import uuid

import numpy as np
from pyscf import lib
from pyscf.lib import logger

from . import nevpt2_utils as _utils
from . import spinor_helper


__all__ = ["X2CTMPSNEVPT2", "X2CTNEVPT2", "TMPSNEVPT2"]


SUBSPACE_ORDER = _utils.SUBSPACE_ORDER
_PERTURBER_EXPRESSIONS = _utils._PERTURBER_EXPRESSIONS
_PAIR_RESTRICTIONS = _utils._PAIR_RESTRICTIONS

_SOURCE_LINE = re.compile(
    r"^\s*([+-]?)\s*(?:SUM\s+<([a-z]+)>\s+)?" r"([hw])\[([a-z]+)\]\s+(.+?)\s*$"
)
_SOURCE_OPERATOR = re.compile(r"([CD])\[([a-z])\]")
_ACTIVE_SYMBOLS = frozenset("abc")
_CORE_SYMBOLS = frozenset("ij")
_VIRTUAL_SYMBOLS = frozenset("rs")


@dataclass(frozen=True)
class _SymbolicSourceTerm:
    """One parsed line of the shared Wick perturber specification."""

    factor: int
    tensor: str
    tensor_indices: tuple[str, ...]
    summed_indices: tuple[str, ...]
    active_operators: tuple[tuple[str, str], ...]
    external_operators: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class _ActiveOperatorTerm:
    """One numeric active-space fermion string."""

    operators: tuple[tuple[str, int], ...]
    coefficient: complex


@dataclass(frozen=True)
class _ActiveSource:
    """The fixed-external-index source acting on the active reference."""

    key: str
    free_indices: tuple[int, ...]
    delta_n_active: int
    constant: complex
    terms: tuple[_ActiveOperatorTerm, ...]
    discarded_coefficient_l1: float = 0.0

    @property
    def is_scalar(self) -> bool:
        return not self.terms


def _validate_source_key(key: str) -> str:
    if key not in SUBSPACE_ORDER:
        choices = ", ".join(SUBSPACE_ORDER)
        raise ValueError(f"unknown NEVPT2 source key {key!r}; expected {choices}")
    return key


@lru_cache(maxsize=16)
def _parse_source_expression(
    key: str,
    expression: str,
) -> tuple[_SymbolicSourceTerm, ...]:
    """Parse one shared Wick expression into its active and external parts."""

    key = _validate_source_key(key)
    free_symbols = frozenset(key)
    parsed: list[_SymbolicSourceTerm] = []
    for line_number, line in enumerate(expression.splitlines(), start=1):
        if not line.strip():
            continue
        match = _SOURCE_LINE.fullmatch(line)
        if match is None:
            raise RuntimeError(
                f"cannot parse shared perturber {key!r} line {line_number}: "
                f"{line!r}"
            )
        sign_text, summed_text, tensor, tensor_text, operator_text = match.groups()
        factor = -1 if sign_text == "-" else 1
        summed_indices = tuple(summed_text or "")
        if len(summed_indices) != len(set(summed_indices)):
            raise RuntimeError(f"shared perturber {key!r} repeats a summed index")
        if any(index not in _ACTIVE_SYMBOLS for index in summed_indices):
            raise RuntimeError(
                f"shared perturber {key!r} has a non-active summed index"
            )
        tensor_indices = tuple(tensor_text)
        expected_rank = 2 if tensor == "h" else 4
        if len(tensor_indices) != expected_rank:
            raise RuntimeError(
                f"shared perturber {key!r} has a rank-{len(tensor_indices)} "
                f"{tensor} tensor"
            )

        matches = tuple(_SOURCE_OPERATOR.finditer(operator_text))
        residual = _SOURCE_OPERATOR.sub("", operator_text)
        if not matches or residual.strip():
            raise RuntimeError(
                f"shared perturber {key!r} has an unsupported operator "
                f"expression {operator_text!r}"
            )
        active_operators = []
        external_operators = []
        for operator_match in matches:
            operator, index = operator_match.groups()
            item = (operator, index)
            if index in free_symbols:
                external_operators.append(item)
            elif index in summed_indices:
                active_operators.append(item)
            else:
                raise RuntimeError(
                    f"shared perturber {key!r} operator index {index!r} "
                    "is neither free nor summed"
                )
        allowed_indices = free_symbols | frozenset(summed_indices)
        if any(index not in allowed_indices for index in tensor_indices):
            raise RuntimeError(f"shared perturber {key!r} tensor has an unbound index")
        if set(summed_indices) - set(tensor_indices):
            raise RuntimeError(
                f"shared perturber {key!r} declares an unused summed index"
            )
        parsed.append(
            _SymbolicSourceTerm(
                factor=factor,
                tensor=tensor,
                tensor_indices=tensor_indices,
                summed_indices=summed_indices,
                active_operators=tuple(active_operators),
                external_operators=tuple(external_operators),
            )
        )

    if not parsed:
        raise RuntimeError(f"shared perturber {key!r} is empty")

    # Removing the fixed core/virtual operators is valid only when every term
    # creates the same external occupation pattern in the same relative order.
    external_pattern = parsed[0].external_operators
    if any(term.external_operators != external_pattern for term in parsed[1:]):
        raise RuntimeError(
            f"shared perturber {key!r} has inconsistent external strings"
        )
    return tuple(parsed)


def _symbolic_source_terms(key: str) -> tuple[_SymbolicSourceTerm, ...]:
    """Return a parser result tied to the current shared expression text."""

    key = _validate_source_key(key)
    return _parse_source_expression(key, _PERTURBER_EXPRESSIONS[key])


def _source_free_shape(key: str, eris: spinor_helper._SpinorERIs) -> tuple[int, ...]:
    key = _validate_source_key(key)
    return tuple(
        eris.ncore if symbol in _CORE_SYMBOLS else eris.nvirt for symbol in key
    )


def _source_pair_allowed(key: str, free_indices: Sequence[int]) -> bool:
    key = _validate_source_key(key)
    free_indices = tuple(int(index) for index in free_indices)
    if len(free_indices) != len(key):
        raise ValueError(f"source {key!r} needs {len(key)} free indices")
    return all(
        free_indices[left] < free_indices[right]
        for left, right in _PAIR_RESTRICTIONS[key]
    )


def _iter_source_blocks(
    key: str,
    eris: spinor_helper._SpinorERIs,
) -> Iterator[tuple[int, ...]]:
    """Yield all pair-restricted fixed core/virtual tuples for one class."""

    shape = _source_free_shape(key, eris)
    for free_indices in np.ndindex(shape):
        if _source_pair_allowed(key, free_indices):
            yield tuple(int(index) for index in free_indices)


def _permutation_sign(values: Sequence[int]) -> int:
    inversions = sum(
        values[left] > values[right]
        for left in range(len(values))
        for right in range(left + 1, len(values))
    )
    return -1 if inversions % 2 else 1


def _canonicalize_active_operators(
    operators: Sequence[tuple[str, int]],
) -> tuple[int, tuple[tuple[str, int], ...]]:
    """Canonicalize contiguous equal-kind fermion runs with their parity."""

    operators = tuple((str(kind), int(index)) for kind, index in operators)
    if any(kind not in ("C", "D") for kind, _index in operators):
        raise ValueError("active source operators must be C or D")
    result: list[tuple[str, int]] = []
    total_sign = 1
    start = 0
    while start < len(operators):
        stop = start + 1
        kind = operators[start][0]
        while stop < len(operators) and operators[stop][0] == kind:
            stop += 1
        indices = [index for _kind, index in operators[start:stop]]
        if len(indices) != len(set(indices)):
            return 0, ()
        order = sorted(indices, reverse=kind == "D")
        positions = {value: position for position, value in enumerate(order)}
        total_sign *= _permutation_sign([positions[value] for value in indices])
        result.extend((kind, index) for index in order)
        start = stop
    return total_sign, tuple(result)


def _global_orbital_index(
    symbol: str,
    value: int,
    eris: spinor_helper._SpinorERIs,
) -> int:
    value = int(value)
    if symbol in _CORE_SYMBOLS:
        if not 0 <= value < eris.ncore:
            raise IndexError(f"core source index {symbol}={value} is out of range")
        return value
    if symbol in _ACTIVE_SYMBOLS:
        if not 0 <= value < eris.ncas:
            raise IndexError(f"active source index {symbol}={value} is out of range")
        return eris.ncore + value
    if symbol in _VIRTUAL_SYMBOLS:
        if not 0 <= value < eris.nvirt:
            raise IndexError(f"virtual source index {symbol}={value} is out of range")
        return eris.nocc + value
    raise RuntimeError(f"unknown source index symbol {symbol!r}")


def _source_tensor_element(
    term: _SymbolicSourceTerm,
    assignment: dict[str, int],
    eris: spinor_helper._SpinorERIs,
) -> complex:
    indices = tuple(
        _global_orbital_index(symbol, assignment[symbol], eris)
        for symbol in term.tensor_indices
    )
    if term.tensor == "h":
        return complex(eris.h1eff[indices])
    p, q, r, s = indices
    return complex(eris.pppp[p, r, q, s])


def _source_delta_n(
    constant: complex,
    terms: Iterable[_ActiveOperatorTerm],
) -> int:
    changes = {
        sum(1 if kind == "C" else -1 for kind, _index in term.operators)
        for term in terms
        if term.coefficient != 0.0
    }
    if constant != 0.0:
        changes.add(0)
    if not changes:
        return 0
    if len(changes) != 1:
        raise RuntimeError(f"source mixes active particle-number changes {changes}")
    return changes.pop()


def _active_source(
    key: str,
    free_indices: Sequence[int],
    eris: spinor_helper._SpinorERIs,
    *,
    coefficient_cutoff: float = 0.0,
) -> _ActiveSource:
    """Expand one fixed block directly from the shared Wick specification."""

    key = _validate_source_key(key)
    if not isinstance(eris, spinor_helper._SpinorERIs):
        raise TypeError("t-MPS sources require dense spinor_helper._SpinorERIs")
    coefficient_cutoff = float(coefficient_cutoff)
    if not np.isfinite(coefficient_cutoff) or coefficient_cutoff != 0.0:
        raise ValueError(
            "coefficient_cutoff must be finite and exactly zero; "
            "magnitude-pruned "
            "active sources are not supported"
        )
    free_indices = tuple(int(index) for index in free_indices)
    shape = _source_free_shape(key, eris)
    if len(free_indices) != len(shape):
        raise ValueError(f"source {key!r} needs {len(shape)} free indices")
    if any(not 0 <= index < size for index, size in zip(free_indices, shape)):
        raise IndexError(f"source {key!r} free indices are out of range")
    if not _source_pair_allowed(key, free_indices):
        raise ValueError(f"source {key!r} violates its strict pair restriction")

    fixed_assignment = dict(zip(key, free_indices))
    coefficients: dict[tuple[tuple[str, int], ...], complex] = {}
    for symbolic in _symbolic_source_terms(key):
        summed_ranges = (range(eris.ncas) for _ in symbolic.summed_indices)
        for values in itertools.product(*summed_ranges):
            assignment = dict(fixed_assignment)
            assignment.update(zip(symbolic.summed_indices, values))
            coefficient = symbolic.factor * _source_tensor_element(
                symbolic,
                assignment,
                eris,
            )
            operators = tuple(
                (kind, assignment[index]) for kind, index in symbolic.active_operators
            )
            sign, canonical = _canonicalize_active_operators(operators)
            if sign:
                coefficients[canonical] = coefficients.get(canonical, 0.0j) + (
                    sign * coefficient
                )

    constant = complex(coefficients.pop((), 0.0j))
    discarded_l1 = 0.0
    numeric_terms = []
    for operators, coefficient in sorted(coefficients.items()):
        coefficient = complex(coefficient)
        if abs(coefficient) <= coefficient_cutoff:
            discarded_l1 += abs(coefficient)
            continue
        numeric_terms.append(_ActiveOperatorTerm(operators, coefficient))
    if abs(constant) <= coefficient_cutoff:
        discarded_l1 += abs(constant)
        constant = 0.0j
    terms = tuple(numeric_terms)
    delta_n = _source_delta_n(constant, terms)
    expected_delta = {
        "ijrs": 0,
        "rsi": -1,
        "ijr": 1,
        "rs": -2,
        "ij": 2,
        "ir": 0,
        "r": -1,
        "i": 1,
    }[key]
    if delta_n != expected_delta and (constant != 0.0 or terms):
        raise RuntimeError(
            f"parsed source {key!r} changes active particle number by "
            f"{delta_n}, expected {expected_delta}"
        )
    return _ActiveSource(
        key=key,
        free_indices=free_indices,
        delta_n_active=expected_delta,
        constant=constant,
        terms=terms,
        discarded_coefficient_l1=float(discarded_l1),
    )


def _operator_text(
    operators: Sequence[tuple[str, int]],
    *,
    site_map: Sequence[int] | None = None,
) -> str:
    """Convert active C/D operators to Block2's ``+_p -_q`` syntax."""

    if site_map is None:
        site_map = tuple(range(1 + max((x[1] for x in operators), default=-1)))
    site_map = tuple(int(value) for value in site_map)
    pieces = []
    for kind, index in operators:
        if not 0 <= index < len(site_map):
            raise IndexError("active operator index is outside the site map")
        symbol = "+" if kind == "C" else "-"
        pieces.append(f"{symbol}_{site_map[index]}")
    return " ".join(pieces)


def _source_mpo_terms(
    source: _ActiveSource,
    *,
    site_map: Sequence[int] | None = None,
) -> tuple[list[tuple[str, complex]], complex]:
    """Return the exact input contract for ``get_mpo_any_fermionic``."""

    terms = [
        (_operator_text(term.operators, site_map=site_map), term.coefficient)
        for term in source.terms
    ]
    return terms, complex(source.constant)


def _adjoint_active_source(source: _ActiveSource) -> _ActiveSource:
    """Return the exact operator adjoint, including order reversal."""

    return _ActiveSource(
        key=source.key,
        free_indices=source.free_indices,
        delta_n_active=-source.delta_n_active,
        constant=source.constant.conjugate(),
        terms=tuple(
            _ActiveOperatorTerm(
                tuple(
                    ("D" if kind == "C" else "C", index)
                    for kind, index in reversed(term.operators)
                ),
                term.coefficient.conjugate(),
            )
            for term in source.terms
        ),
        discarded_coefficient_l1=source.discarded_coefficient_l1,
    )


def _unwrap_general_mpo(mpo):
    """Strip Block2 convenience wrappers for ``MPOTools.from_block2``."""

    while hasattr(mpo, "prim_mpo"):
        mpo = mpo.prim_mpo
    return mpo


def _to_block2_mpo_lossless(mpo, basis, *, tag="PYMPO", add_ident=True):
    r"""Translate a pyblock2 MPO without magnitude-based block deletion.

    ``pyblock2.algebra.io.MPOTools.to_block2`` currently omits every local
    reduced block whose norm is below ``1e-12``.  That convenience cutoff is
    not admissible for the exact :math:`B^\dagger B` source-norm audit: a tiny
    source can still have a finite resolvent contribution when its external
    gap is tiny or its Table-I coefficient is large.  This is the upstream
    conversion algorithm with the sole semantic change that only identically
    zero blocks are omitted.
    """

    from collections import Counter

    from pyblock2.algebra.io import init_block2_types

    quantum_type = mpo.tensors[0].blocks[0].q_labels[0].__class__
    dtype = mpo.tensors[0].blocks[0].reduced.dtype
    b, bs, brs, _bx = init_block2_types(quantum_type, dtype)
    n_sites = len(mpo.tensors)
    block2_mpo = bs.MPO(n_sites, tag)
    tensors, left_operators, right_operators, site_op_infos = [], [], [], []
    site_basis = [None] * n_sites
    for site, basis_states in enumerate(basis):
        state_info = brs.StateInfo()
        state_info.allocate(len(basis_states))
        for index, (quantum, count) in enumerate(basis_states.items()):
            state_info.quanta[index] = quantum
            state_info.n_states[index] = count
        site_basis[site] = state_info
        state_info.sort_states()
    vacuum = (
        mpo.tensors[0].blocks[0].q_labels[0]
        - mpo.tensors[0].blocks[0].q_labels[0]
    )[0]
    middle_dimensions = mpo.get_bond_dims()
    left_dimensions = [Counter({vacuum: 1})] + middle_dimensions
    right_dimensions = middle_dimensions + [Counter({vacuum: 1})]
    for site in range(n_sites):
        py_tensor = mpo.tensors[site]
        tensors.append(bs.OperatorTensor())
        site_op_infos.append({})
        double_allocator = b.DoubleVectorAllocator()
        int_allocator = b.IntVectorAllocator()
        n_rows = sum(left_dimensions[site].values())
        n_columns = sum(right_dimensions[site].values())
        left_quantum_numbers = [
            quantum
            for quantum, count in sorted(left_dimensions[site].items())
            for _ in range(count)
        ]
        right_quantum_numbers = [
            quantum
            for quantum, count in sorted(right_dimensions[site].items())
            for _ in range(count)
        ]
        left_offsets = Counter()
        right_offsets = Counter()
        offset = 0
        for quantum, count in sorted(left_dimensions[site].items()):
            left_offsets[quantum] = offset
            offset += count
        offset = 0
        for quantum, count in sorted(right_dimensions[site].items()):
            right_offsets[quantum] = offset
            offset += count
        data = {}
        for block in py_tensor.blocks:
            quantum_labels, reduced = block.q_labels, block.reduced
            row_quantum = column_quantum = vacuum
            if site != n_sites - 1:
                column_quantum, quantum_labels = (
                    block.q_labels[-1],
                    quantum_labels[:-1],
                )
            else:
                reduced = reduced[..., None]
            if site != 0:
                row_quantum, quantum_labels = block.q_labels[0], quantum_labels[1:]
            else:
                reduced = reduced[None, ...]
            row_count = left_dimensions[site][row_quantum]
            column_count = right_dimensions[site][column_quantum]
            row_offset = left_offsets[row_quantum]
            column_offset = right_offsets[column_quantum]
            for row in range(row_offset, row_offset + row_count):
                for column in range(column_offset, column_offset + column_count):
                    matrix = reduced[
                        row - row_offset,
                        ...,
                        column - column_offset,
                    ]
                    if not np.any(matrix != 0.0):
                        continue
                    data.setdefault((row, column), []).append(
                        (quantum_labels, matrix)
                    )
        for operator_index, (indices, blocks) in enumerate(sorted(data.items())):
            delta_quantum = (blocks[0][0][0] - blocks[0][0][1])[0]
            if site == 0:
                expression = bs.OpElement(
                    b.OpNames.XL,
                    b.SiteIndex(
                        (indices[1] // 1000, indices[1] % 1000),
                        (),
                    ),
                    delta_quantum,
                    1.0,
                )
            elif site == n_sites - 1:
                expression = bs.OpElement(
                    b.OpNames.XR,
                    b.SiteIndex(
                        (indices[0] // 1000, indices[0] % 1000),
                        (),
                    ),
                    delta_quantum,
                    1.0,
                )
            else:
                expression = bs.OpElement(
                    b.OpNames.X,
                    b.SiteIndex(
                        (operator_index // 1000, operator_index % 1000),
                        (),
                    ),
                    delta_quantum,
                    1.0,
                )
            if delta_quantum not in site_op_infos[site]:
                site_op_infos[site][delta_quantum] = brs.SparseMatrixInfo(
                    int_allocator
                )
                site_op_infos[site][delta_quantum].initialize(
                    site_basis[site],
                    site_basis[site],
                    delta_quantum,
                    delta_quantum.is_fermion,
                )
            matrix_info = site_op_infos[site][delta_quantum]
            sparse_matrix = bs.SparseMatrix(double_allocator)
            sparse_matrix.allocate(matrix_info)
            for (left_quantum, right_quantum), matrix in blocks:
                state_index = sparse_matrix.info.find_state(
                    delta_quantum.combine(left_quantum, right_quantum)
                )
                sparse_matrix[state_index] = np.asarray(matrix).ravel()
            tensors[site].ops[expression] = sparse_matrix
        identity_quantum = vacuum
        identity_operator = bs.OpElement(
            b.OpNames.I,
            b.SiteIndex(),
            identity_quantum,
            1.0,
        )
        if identity_operator not in tensors[site].ops:
            if identity_quantum not in site_op_infos[site]:
                site_op_infos[site][identity_quantum] = brs.SparseMatrixInfo(
                    int_allocator
                )
                site_op_infos[site][identity_quantum].initialize(
                    site_basis[site],
                    site_basis[site],
                    identity_quantum,
                    identity_quantum.is_fermion,
                )
            matrix_info = site_op_infos[site][identity_quantum]
            sparse_matrix = bs.SparseMatrix(double_allocator)
            sparse_matrix.allocate(matrix_info)
            for state_index in range(matrix_info.n):
                sparse_matrix[state_index] = np.identity(
                    matrix_info.n_states_ket[state_index]
                ).ravel()
            tensors[site].ops[identity_operator] = sparse_matrix
        left_row = [
            bs.OpElement(
                b.OpNames.XL,
                b.SiteIndex((index // 1000, index % 1000), ()),
                quantum,
                1.0,
            )
            for index, quantum in enumerate(right_quantum_numbers)
        ]
        right_column = [
            bs.OpElement(
                b.OpNames.XR,
                b.SiteIndex((index // 1000, index % 1000), ()),
                -quantum,
                1.0,
            )
            for index, quantum in enumerate(left_quantum_numbers)
        ]
        if site == 0:
            tensors[site].lmat = brs.SymbolicRowVector(n_columns)
            tensors[site].lmat.data = brs.VectorOpExpr(left_row)
            for index, expression in enumerate(left_row):
                if expression not in tensors[site].ops:
                    tensors[site].lmat.data[index] = brs.OpExpr()
        elif site == n_sites - 1:
            tensors[site].lmat = brs.SymbolicColumnVector(n_rows)
            tensors[site].lmat.data = brs.VectorOpExpr(right_column)
            for index, expression in enumerate(right_column):
                if expression not in tensors[site].ops:
                    tensors[site].lmat.data[index] = brs.OpExpr()
        else:
            expressions = [
                bs.OpElement(
                    b.OpNames.X,
                    b.SiteIndex((index // 1000, index % 1000), ()),
                    (blocks[0][0][0] - blocks[0][0][1])[0],
                    1.0,
                )
                for index, (_indices, blocks) in enumerate(sorted(data.items()))
            ]
            tensors[site].lmat = brs.SymbolicMatrix(n_rows, n_columns)
            tensors[site].lmat.indices = b.VectorPIntInt(
                [indices for indices in sorted(data)]
            )
            tensors[site].lmat.data = brs.VectorOpExpr(expressions)
        tensors[site].rmat = tensors[site].lmat
        right_operators.append(brs.SymbolicColumnVector(len(right_column)))
        left_operators.append(brs.SymbolicRowVector(len(left_row)))
        right_operators[site].data = brs.VectorOpExpr(right_column)
        left_operators[site].data = brs.VectorOpExpr(left_row)
        site_op_infos[site] = brs.VectorPLMatInfo(sorted(site_op_infos[site].items()))
    block2_mpo.const_e = mpo.const_e
    block2_mpo.tf = bs.TensorFunctions(bs.OperatorFunctions(brs.CG()))
    block2_mpo.site_op_infos = brs.VectorVectorPLMatInfo(site_op_infos)
    block2_mpo.basis = brs.VectorStateInfo(site_basis)
    block2_mpo.sparse_form = "N" * n_sites
    block2_mpo.op = bs.OpElement(
        b.OpNames.H,
        b.SiteIndex(),
        left_operators[-1][0].q_label,
        1.0,
    )
    block2_mpo.right_operator_names = brs.VectorSymbolic(right_operators)
    block2_mpo.left_operator_names = brs.VectorSymbolic(left_operators)
    block2_mpo.tensors = bs.VectorOpTensor(tensors)
    block2_mpo.left_vacuum = vacuum
    for site in range(block2_mpo.n_sites):
        for expression, matrix in block2_mpo.tensors[site].ops.items():
            if expression.q_label != matrix.info.delta_quantum:
                raise RuntimeError("lossless MPO conversion changed a quantum label")
        symbolic = block2_mpo.tensors[site].lmat
        left_row = block2_mpo.left_operator_names[site].data
        right_column = block2_mpo.right_operator_names[site].data
        if site == 0:
            for index in range(len(left_row)):
                if (
                    symbolic.data[index].get_type() != b.OpTypes.Zero
                    and symbolic.data[index].q_label != left_row[index].q_label
                ):
                    raise RuntimeError("invalid left MPO boundary quantum label")
        elif site == block2_mpo.n_sites - 1:
            for index in range(len(right_column)):
                if (
                    symbolic.data[index].get_type() != b.OpTypes.Zero
                    and symbolic.data[index].q_label != right_column[index].q_label
                ):
                    raise RuntimeError("invalid right MPO boundary quantum label")
        else:
            previous_left = block2_mpo.left_operator_names[site - 1].data
            next_right = block2_mpo.right_operator_names[site + 1].data
            if len(left_row) != len(next_right):
                raise RuntimeError("inconsistent lossless MPO bond dimensions")
            for index in range(len(left_row)):
                if left_row[index].q_label != -next_right[index].q_label:
                    raise RuntimeError("inconsistent lossless MPO bond quantum labels")
            for matrix_index in range(len(symbolic.data)):
                left_quantum = previous_left[
                    symbolic.indices[matrix_index][0]
                ].q_label
                right_quantum = left_row[
                    symbolic.indices[matrix_index][1]
                ].q_label
                operator_quantum = symbolic.data[matrix_index].q_label
                if left_quantum + operator_quantum != right_quantum:
                    raise RuntimeError("invalid internal MPO quantum-label flow")
    for site in range(block2_mpo.n_sites):
        block2_mpo.save_tensor(site)
        block2_mpo.unload_tensor(site)
        block2_mpo.save_left_operators(site)
        block2_mpo.unload_left_operators(site)
        block2_mpo.save_right_operators(site)
        block2_mpo.unload_right_operators(site)
    block2_mpo = bs.SimplifiedMPO(block2_mpo, bs.Rule(), False, False)
    if add_ident:
        block2_mpo = bs.IdentityAddedMPO(block2_mpo)
    return block2_mpo


def _orbital_gap_for_block(
    key: str,
    free_indices: Sequence[int],
    core_energy: Sequence[float],
    virtual_energy: Sequence[float],
) -> float:
    """Return the existing positive-gap convention for one fixed block."""

    key = _validate_source_key(key)
    free_indices = tuple(int(index) for index in free_indices)
    if len(free_indices) != len(key):
        raise ValueError(f"source {key!r} needs {len(key)} free indices")
    core_energy = np.asarray(core_energy, dtype=float)
    virtual_energy = np.asarray(virtual_energy, dtype=float)
    gap = 0.0
    for symbol, index in zip(key, free_indices):
        values = core_energy if symbol in _CORE_SYMBOLS else virtual_energy
        if not 0 <= index < len(values):
            raise IndexError(f"orbital-gap index {symbol}={index} is out of range")
        gap += -values[index] if symbol in _CORE_SYMBOLS else values[index]
    return float(gap)


def _validate_time_grid(time_grid: Sequence[float]) -> np.ndarray:
    grid = np.asarray(time_grid, dtype=float)
    if grid.ndim != 1 or grid.size < 2:
        raise ValueError("time_grid must be a one-dimensional array with >=2 points")
    if not np.all(np.isfinite(grid)):
        raise ValueError("time_grid contains non-finite values")
    if grid[0] != 0.0:
        raise ValueError("time_grid must begin at zero")
    if np.any(np.diff(grid) <= 0.0):
        raise ValueError("time_grid must be strictly increasing")
    return grid


def _integrate_samples(
    time_grid: Sequence[float],
    values: Sequence[complex],
    *,
    mode: str,
) -> complex:
    """Integrate sampled values without silently discarding complex parts."""

    grid = _validate_time_grid(time_grid)
    values = np.asarray(values)
    if values.shape != grid.shape:
        raise ValueError("integration values must have the time-grid shape")
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError("integration values must be numeric")
    if not np.all(np.isfinite(values)):
        raise ValueError("integration values contain non-finite data")
    if mode == "trapezoid_tail":
        if hasattr(np, "trapezoid"):
            return complex(np.trapezoid(values, grid))
        return complex(np.trapz(values, grid))
    if mode not in ("simpson_tail", "newton_cotes_8_tail"):
        raise ValueError(
            "integration_mode must be 'simpson_tail', "
            "'newton_cotes_8_tail', or 'trapezoid_tail'"
        )
    spacing = np.diff(grid)
    uniform_tol = 64.0 * np.finfo(float).eps * max(1.0, abs(grid[-1]))
    if np.max(np.abs(spacing - spacing[0])) > uniform_tol:
        raise ValueError(f"{mode} requires a uniform time grid")
    intervals = grid.size - 1
    if mode == "newton_cotes_8_tail":
        # Composite closed nine-point Newton--Cotes (degree nine).  It is
        # particularly useful for the steep but smooth short-time spectral
        # decay in molecular t-NEVPT2.  Any final incomplete eight-interval
        # panel is integrated by the already audited Simpson/trapezoid rule.
        panel_intervals = intervals - intervals % 8
        weights = np.asarray(
            (989, 5888, -928, 10496, -4540, 10496, -928, 5888, 989),
            dtype=float,
        )
        result = 0.0j
        for start in range(0, panel_intervals, 8):
            result += (
                4.0 * spacing[0] / 14175.0 * np.dot(weights, values[start : start + 9])
            )
        if panel_intervals != intervals:
            suffix_grid = grid[panel_intervals:]
            result += _integrate_samples(
                suffix_grid - suffix_grid[0],
                values[panel_intervals:],
                mode="simpson_tail",
            )
        return complex(result)

    simpson_intervals = intervals if intervals % 2 == 0 else intervals - 1
    result = 0.0j
    if simpson_intervals:
        stop = simpson_intervals + 1
        result = (
            spacing[0]
            / 3.0
            * (
                values[0]
                + values[stop - 1]
                + 4.0 * np.sum(values[1 : stop - 1 : 2])
                + 2.0 * np.sum(values[2 : stop - 2 : 2])
            )
        )
    if simpson_intervals != intervals:
        result += 0.5 * spacing[-1] * (values[-2] + values[-1])
    return complex(result)


def _completed_integration_gate(
    diagnostics: dict[str, dict[str, Any]],
    field: str,
) -> bool:
    """Fail closed unless every nontrivial completed integral passes ``field``."""

    if not isinstance(diagnostics, dict) or not diagnostics:
        return False
    return all(
        isinstance(item, dict)
        and (
            item.get("analytic") is True
            or (
                item.get("zero_source") is True
                and item.get("exact_zero_source_certified") is True
            )
            or item.get(field) is True
        )
        for item in diagnostics.values()
    )


_INTEGRATION_GATE_EVIDENCE_FIELDS = (
    "pass_name",
    "requested_time_step",
    "explicit_time_grid",
    "scheduled_time_points",
    "scheduled_final_time",
    "sampled_time_points",
    "sampled_final_time",
    "analytic",
    "zero_source",
    "exact_zero_source_certified",
    "accepted",
    "reason",
    "tail",
    "decay_rate",
    "window",
    "terminal_window_maximum",
    "negligible_tolerance",
    "imaginary_residual",
    "imaginary_limit",
    "log_fit_max_residual",
    "rate_relative_sensitivity",
    "log_fit_residual_tolerance",
    "rate_relative_sensitivity_tolerance",
    "tail_used",
    "tail_limit",
    "tail_gate_passed",
    "quadrature",
    "raw_quadrature",
    "reference_normalization_divisor",
    "external_energy",
    "signed_energy_contribution",
    "mode",
    "energy_imaginary_residual",
    "energy_real_magnitude",
    "energy_imaginary_scale",
    "energy_imaginary_tolerance",
    "energy_imaginary_gate_passed",
    "maximum_raw_integrand_imaginary_residual",
    "maximum_raw_integrand_real_magnitude",
    "raw_integrand_imaginary_scale",
    "raw_integrand_imaginary_tolerance",
    "raw_integrand_imaginary_gate_passed",
    "maximum_prefactor_hermiticity_residual",
    "maximum_prefactor_magnitude",
    "prefactor_hermiticity_scale",
    "prefactor_hermiticity_tolerance",
    "prefactor_hermiticity_gate_passed",
    "maximum_correlation_imaginary_residual",
    "maximum_correlation_real_magnitude",
    "correlation_imaginary_scale",
    "correlation_imaginary_tolerance",
    "correlation_imaginary_gate_passed",
)


def _integration_gate_evidence(
    diagnostics: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Retain the numerical evidence needed to re-audit an integration pass."""

    if not isinstance(diagnostics, dict):
        raise TypeError("integration diagnostics must be a dictionary")
    return {
        str(label): {
            field: item[field]
            for field in _INTEGRATION_GATE_EVIDENCE_FIELDS
            if field in item
        }
        for label, item in diagnostics.items()
        if isinstance(item, dict) and label != "dt_refinement"
    }


def _estimate_exponential_tail(
    time_grid: Sequence[float],
    values: Sequence[complex],
    *,
    window: int = 5,
    imaginary_tolerance: float = 1.0e-10,
    negligible_tolerance: float | None = None,
    log_residual_tolerance: float = 0.1,
    rate_sensitivity_tolerance: float = 0.25,
) -> dict[str, Any]:
    """Estimate a positive single-exponential tail and window sensitivity."""

    grid = _validate_time_grid(time_grid)
    samples = np.asarray(values, dtype=np.complex128)
    if samples.shape != grid.shape or not np.all(np.isfinite(samples)):
        raise ValueError("tail samples must be finite and match the time grid")
    window = int(window)
    if window < 3:
        raise ValueError("tail fit window must contain at least three points")
    if negligible_tolerance is not None:
        negligible_tolerance = float(negligible_tolerance)
        if not np.isfinite(negligible_tolerance) or negligible_tolerance < 0.0:
            raise ValueError(
                "negligible tail tolerance must be finite and non-negative"
            )
    log_residual_tolerance = float(log_residual_tolerance)
    rate_sensitivity_tolerance = float(rate_sensitivity_tolerance)
    if not np.isfinite(log_residual_tolerance) or log_residual_tolerance <= 0.0:
        raise ValueError("tail log-residual tolerance must be finite and positive")
    if not np.isfinite(rate_sensitivity_tolerance) or rate_sensitivity_tolerance <= 0.0:
        raise ValueError("tail rate-sensitivity tolerance must be finite and positive")
    fit_controls = {
        "log_fit_residual_tolerance": log_residual_tolerance,
        "rate_relative_sensitivity_tolerance": (rate_sensitivity_tolerance),
    }
    if len(grid) < window:
        return {
            **fit_controls,
            "accepted": False,
            "reason": "insufficient_points",
            "tail": math.inf,
            "decay_rate": math.nan,
            "window": window,
        }
    terminal_window_maximum = float(np.max(np.abs(samples[-window:]), initial=0.0))
    if (
        negligible_tolerance is not None
        and terminal_window_maximum <= negligible_tolerance
    ):
        return {
            **fit_controls,
            "accepted": True,
            "reason": "numerically_zero_window",
            "tail": 0.0,
            # There is no numerically meaningful fitted rate once the whole
            # window is below the declared absolute floor.  ``None`` makes
            # that absence explicit and keeps strict JSON finite-value gates
            # from confusing an intentional sentinel with failed arithmetic.
            "decay_rate": None,
            "terminal_window_maximum": terminal_window_maximum,
            "negligible_tolerance": negligible_tolerance,
            "imaginary_residual": float(
                np.max(np.abs(samples.imag[-window:]), initial=0.0)
            ),
            "window": window,
        }
    scale = max(
        float(np.max(np.abs(samples.real[-window:]))),
        np.finfo(float).tiny,
    )
    imag_residual = float(np.max(np.abs(samples.imag[-window:])))
    imaginary_limit = imaginary_tolerance * (1.0 + scale)
    if imag_residual > imaginary_limit:
        return {
            **fit_controls,
            "accepted": False,
            "reason": "complex_tail",
            "tail": math.inf,
            "decay_rate": math.nan,
            "imaginary_residual": imag_residual,
            "imaginary_limit": imaginary_limit,
            "window": window,
        }
    real_values = samples.real
    if np.any(real_values[-window:] <= 0.0):
        return {
            **fit_controls,
            "accepted": False,
            "reason": "nonpositive_tail",
            "tail": math.inf,
            "decay_rate": math.nan,
            "window": window,
        }

    def fit(count: int) -> tuple[float, float]:
        x = grid[-count:]
        y = np.log(real_values[-count:])
        slope, intercept = np.polyfit(x, y, 1)
        predicted = slope * x + intercept
        residual = float(np.max(np.abs(y - predicted)))
        return float(-slope), residual

    decay_rate, log_residual = fit(window)
    # A two-point fit has no residual estimate but does provide the only
    # independent slope available for the minimum legal three-point window.
    # Comparing against it is preferable to silently disabling the required
    # fit-window sensitivity gate when ``window == 3``.
    shorter_rate, _shorter_residual = fit(window - 1)
    rate_sensitivity = abs(decay_rate - shorter_rate) / max(
        abs(decay_rate),
        1.0e-300,
    )
    if not np.isfinite(decay_rate) or decay_rate <= 0.0:
        reason = "nondecaying_tail"
    elif not np.isfinite(log_residual):
        reason = "nonfinite_decay_fit"
    elif log_residual > log_residual_tolerance:
        reason = "poor_exponential_fit"
    elif rate_sensitivity > rate_sensitivity_tolerance:
        reason = "unstable_decay_rate"
    else:
        reason = None
    accepted = reason is None
    tail = float(real_values[-1] / decay_rate) if accepted else math.inf
    return {
        **fit_controls,
        "accepted": accepted,
        "reason": reason,
        "tail": tail,
        "decay_rate": decay_rate,
        "log_fit_max_residual": log_residual,
        "rate_relative_sensitivity": float(rate_sensitivity),
        "imaginary_residual": imag_residual,
        "imaginary_limit": imaginary_limit,
        "window": window,
    }


@dataclass(frozen=True)
class _PreparedTMPSRoot:
    root: int
    reference_energy: float
    mo_coeff: np.ndarray
    orbital_energy: np.ndarray
    eris: spinor_helper._SpinorERIs
    core_energy: np.ndarray
    virtual_energy: np.ndarray
    dm1: np.ndarray
    dm1_diagnostics: dict[str, Any]
    integral_symmetry_diagnostics: dict[str, Any] | None
    driver: Any
    reference_mps: Any
    active_mpo: Any
    reference_norm: float
    active_reference_energy: float
    active_mpo_diagnostics: dict[str, Any]


@dataclass
class _DirectPassResult:
    sub_eners: dict[str, float]
    sub_times: dict[str, float]
    sub_diagnostics: dict[str, Any]
    source_diagnostics: dict[str, Any]
    propagation_diagnostics: dict[str, Any]
    integration_diagnostics: dict[str, Any]
    source_counts: dict[str, int]
    time_series: dict[str, Any]


@dataclass(frozen=True)
class _FactorizedClassPlan:
    """Table-I source basis and complex coefficient outer products."""

    key: str
    basis_sources: tuple[_ActiveSource, ...]
    coefficients: np.ndarray
    gaps: np.ndarray
    block_labels: tuple[str, ...]
    external_energy: float
    external_block_energies: dict[str, float]


class _TagFactory:
    """Generate collision-resistant, filesystem-safe Block2 MPS tags."""

    def __init__(self, root: int, algorithm: str):
        self._prefix = f"TMPS-{uuid.uuid4().hex[:10]}-R{root}-{algorithm[:3]}"
        self._counter = 0

    def __call__(self, key: str, free_indices: Sequence[int], stage: str) -> str:
        digest = hashlib.blake2s(
            repr(tuple(int(value) for value in free_indices)).encode(),
            digest_size=4,
        ).hexdigest()
        self._counter += 1
        return f"{self._prefix}-{key}-{digest}-{stage}-{self._counter}"


def _checked_real_scalar(
    value,
    *,
    name: str,
    imaginary_tolerance: float,
) -> tuple[float, float]:
    value = _utils._complex_scalar(value, name=name)
    imaginary_residual = abs(value.imag)
    if imaginary_residual > imaginary_tolerance * max(1.0, abs(value.real)):
        _utils._warn_numerical(
            f"{name} has an imaginary residual of {imaginary_residual:.3e}"
        )
    return float(value.real), float(imaginary_residual)


def _integrand_reality_diagnostics(
    values,
    *,
    imaginary_tolerance: float,
) -> dict[str, Any]:
    """Audit a physical complex integrand before any Hermitian averaging."""

    samples = np.asarray(values, dtype=np.complex128)
    imaginary_tolerance = float(imaginary_tolerance)
    if samples.ndim != 1 or not np.all(np.isfinite(samples)):
        raise ValueError("integrand samples must be a finite one-dimensional array")
    if not np.isfinite(imaginary_tolerance) or imaginary_tolerance < 0.0:
        raise ValueError(
            "integrand imaginary tolerance must be finite and non-negative"
        )
    scale = max(
        1.0,
        float(np.max(np.abs(samples.real), initial=0.0)),
    )
    maximum_real_magnitude = float(np.max(np.abs(samples.real), initial=0.0))
    residual = float(np.max(np.abs(samples.imag), initial=0.0))
    limit = float(imaginary_tolerance * scale)
    return {
        "maximum_raw_integrand_imaginary_residual": residual,
        "maximum_raw_integrand_real_magnitude": maximum_real_magnitude,
        "raw_integrand_imaginary_scale": scale,
        "raw_integrand_imaginary_tolerance": limit,
        "raw_integrand_imaginary_gate_passed": bool(residual <= limit),
    }


def _identity_overlap(driver, bra, identity, ket):
    """Return ``<bra|ket>`` using Block2's stable center alignment.

    In the current SGF/CPX Block2 build, moving a high-particle-number MPS
    canonical center from the left edge to the right edge during a cross
    expectation can lose allowed bond sectors (and can pass a zero leading
    dimension to ZGEMM).  The adjoint overlap is mathematically identical,
    so when the bra lies to the left of the ket we evaluate ``<ket|bra>`` and
    conjugate it.  Block2 then only moves a temporary bra copy to the left.
    Equal-center contractions retain the ordinary orientation.
    """

    if int(bra.center) < int(ket.center):
        return np.conjugate(driver.expectation(ket, identity, bra))
    return driver.expectation(bra, identity, ket)


def _measure_hermitian_mps_gram(driver, states, identity, *, name):
    """Measure a Hermitian MPS Gram matrix with safe center alignment.

    Equal-center pairs permit two genuinely independent contraction
    orientations and are Hermitian-averaged.  For unequal centers, Block2's
    unsafe rightward alignment prevents measuring both orientations: evaluate
    the one safe contraction once and fill its exact conjugate partner.  The
    diagnostics distinguish measured from unavailable directional checks.
    """

    count = len(states)
    gram = np.zeros((count, count), dtype=np.complex128)
    maximum_directional_residual = 0.0
    maximum_diagonal_imaginary = 0.0
    expectation_count = 0
    measured_directional_pair_count = 0
    unavailable_directional_pair_count = 0
    diagonal_expectations = [None] * count
    pair_expectations = []
    state_centers = [
        None if state is None else int(state.center) for state in states
    ]
    for left, bra in enumerate(states):
        if bra is None:
            continue
        diagonal = _utils._complex_scalar(
            _identity_overlap(driver, bra, identity, bra),
            name=f"{name} diagonal ({left},{left})",
        )
        expectation_count += 1
        maximum_diagonal_imaginary = max(
            maximum_diagonal_imaginary,
            abs(diagonal.imag),
        )
        diagonal_expectations[left] = [
            float(diagonal.real),
            float(diagonal.imag),
        ]
        gram[left, left] = diagonal.real
        for right in range(left + 1, count):
            ket = states[right]
            if ket is None:
                continue
            forward = _utils._complex_scalar(
                _identity_overlap(driver, bra, identity, ket),
                name=f"{name} forward ({left},{right})",
            )
            expectation_count += 1
            if int(bra.center) == int(ket.center):
                reverse = _utils._complex_scalar(
                    _identity_overlap(driver, ket, identity, bra),
                    name=f"{name} reverse ({right},{left})",
                )
                expectation_count += 1
                measured_directional_pair_count += 1
                maximum_directional_residual = max(
                    maximum_directional_residual,
                    abs(forward - reverse.conjugate()),
                )
                overlap = 0.5 * (forward + reverse.conjugate())
                pair_expectations.append(
                    {
                        "left": left,
                        "right": right,
                        "forward": [float(forward.real), float(forward.imag)],
                        "reverse": [float(reverse.real), float(reverse.imag)],
                    }
                )
            else:
                unavailable_directional_pair_count += 1
                overlap = forward
                pair_expectations.append(
                    {
                        "left": left,
                        "right": right,
                        "forward": [float(forward.real), float(forward.imag)],
                        "reverse": None,
                    }
                )
            gram[left, right] = overlap
            gram[right, left] = overlap.conjugate()
    scale = max(1.0, float(np.max(np.abs(gram), initial=0.0)))
    eigenvalues = np.linalg.eigvalsh(gram)
    minimum_eigenvalue = float(np.min(eigenvalues, initial=0.0))
    negative_eigenvalue_residual = float(max(0.0, -minimum_eigenvalue))
    positive_semidefinite_tolerance = float(
        128.0 * np.finfo(float).eps * max(1, count) * scale
    )
    return gram, {
        "method": "center_safe_hermitian_measurement",
        "expectation_count": expectation_count,
        "measured_directional_pair_count": measured_directional_pair_count,
        "unavailable_directional_pair_count": (unavailable_directional_pair_count),
        "directional_residual_scope": "equal_center_pairs_only",
        "state_centers": state_centers,
        "diagonal_expectations": diagonal_expectations,
        "pair_expectations": pair_expectations,
        "maximum_directional_residual": float(maximum_directional_residual),
        "maximum_directional_relative_residual": float(
            maximum_directional_residual / scale
        ),
        "maximum_diagonal_imaginary_residual": float(maximum_diagonal_imaginary),
        "minimum_eigenvalue": minimum_eigenvalue,
        "negative_eigenvalue_residual": negative_eigenvalue_residual,
        "positive_semidefinite_tolerance": positive_semidefinite_tolerance,
        "positive_semidefinite_gate_passed": bool(
            negative_eigenvalue_residual <= positive_semidefinite_tolerance
        ),
    }


def _measure_mps_cross_gram(
    driver,
    left_states,
    right_states,
    identity,
    *,
    name,
):
    """Measure ``<left_i|right_j>`` without assuming equal MPS bases."""

    values = np.zeros(
        (len(left_states), len(right_states)),
        dtype=np.complex128,
    )
    expectation_count = 0
    for left_index, bra in enumerate(left_states):
        if bra is None:
            continue
        for right_index, ket in enumerate(right_states):
            if ket is None:
                continue
            values[left_index, right_index] = _utils._complex_scalar(
                _identity_overlap(driver, bra, identity, ket),
                name=(f"{name} cross overlap " f"({left_index},{right_index})"),
            )
            expectation_count += 1
    return values, {"expectation_count": expectation_count}


def _validate_dm1_only(dm1, ncas: int, nelec: int, *, atol: float, rtol: float):
    """Validate the sole RDM permitted on the production t-MPS path."""

    dm1 = np.asarray(dm1)
    if dm1.shape != (ncas, ncas):
        raise ValueError(f"dm1 must have shape ({ncas}, {ncas})")
    if not np.issubdtype(dm1.dtype, np.number):
        raise TypeError("dm1 must be numeric")
    if not np.all(np.isfinite(dm1)):
        raise ValueError("dm1 contains non-finite values")
    dm1 = np.asarray(dm1, dtype=np.complex128)
    hermiticity_error = float(np.max(np.abs(dm1 - dm1.conj().T), initial=0.0))
    trace = complex(np.trace(dm1))
    trace_error = abs(trace - nelec)
    scale = max(1.0, float(np.max(np.abs(dm1), initial=0.0)), float(nelec))
    tolerance = float(atol + rtol * scale)
    if hermiticity_error > tolerance:
        _utils._warn_numerical(
            f"dm1 Hermiticity residual {hermiticity_error:.3e} exceeds "
            f"{tolerance:.3e}"
        )
    if trace_error > tolerance:
        _utils._warn_numerical(
            f"dm1 particle-number residual {trace_error:.3e} exceeds "
            f"{tolerance:.3e}"
        )
    diagnostics = {
        "shape": list(dm1.shape),
        "hermiticity_error": hermiticity_error,
        "trace": [float(trace.real), float(trace.imag)],
        "trace_error": float(trace_error),
        "tolerance": tolerance,
        "passed": bool(max(hermiticity_error, trace_error) <= tolerance),
    }
    return dm1, diagnostics


def _maximum_array_error(left, right, *, name: str) -> float:
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape:
        raise ValueError(f"{name} shapes differ: {left.shape} != {right.shape}")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError(f"{name} contains non-finite values")
    return float(np.max(np.abs(left - right), initial=0.0))


def _active_site_map(driver, ncas: int) -> tuple[int, ...]:
    """Map original active orbital indices to physical Block2 sites."""

    reorder = getattr(driver, "reorder_idx", None)
    if reorder is None:
        return tuple(range(ncas))
    reorder = np.asarray(reorder, dtype=int)
    expected = np.arange(ncas)
    if reorder.shape != (ncas,) or not np.array_equal(np.sort(reorder), expected):
        raise RuntimeError("the retained DMRG orbital permutation is invalid")
    return tuple(int(value) for value in np.argsort(reorder))


@contextmanager
def _shifted_mpo_constant(mpo, shift: float):
    """Temporarily add one source-specific scalar to a shared MPO."""

    shift = float(shift)
    if not np.isfinite(shift):
        raise ValueError("MPO constant shift must be finite")
    original = complex(mpo.const_e)
    mpo.const_e = original + shift
    try:
        yield mpo
    finally:
        mpo.const_e = original


def _mps_scratch_root(driver) -> Path:
    path = getattr(driver, "mps_dir", None)
    if path is None:
        path = getattr(driver, "scratch", None)
    if callable(path):
        path = path()
    if path is None:
        raise RuntimeError("Block2 driver does not expose its MPS scratch path")
    return Path(path).resolve()


def _remove_owned_mps_files(driver, tag: str) -> None:
    """Remove only scratch files carrying one validated module-owned tag."""

    if not re.fullmatch(r"TMPS-[A-Za-z0-9-]+", tag):
        raise RuntimeError(f"refusing to remove a non-t-MPS tag {tag!r}")
    scratch = _mps_scratch_root(driver)
    patterns = (
        f"{tag}-mps_info.bin",
        f"F.MPS.{tag}.*",
        f"F.MPS.INFO.{tag}.LEFT.*",
        f"F.MPS.INFO.{tag}.RIGHT.*",
    )
    for pattern in patterns:
        for path in scratch.glob(pattern):
            resolved = path.resolve()
            if resolved.parent != scratch or not resolved.is_file():
                raise RuntimeError("refusing to remove an unexpected MPS path")
            resolved.unlink()


def _release_owned_mps(driver, mps, tag: str, *, remove_files: bool) -> None:
    """Release one module-owned MPS without touching the reference/checkpoint."""

    if not re.fullmatch(r"TMPS-[A-Za-z0-9-]+", tag):
        raise RuntimeError(f"refusing to release a non-t-MPS tag {tag!r}")
    # The driver returns disk-backed MPS objects with their mutable tensors
    # unloaded.  In Block2 0.5.4rc16, calling ``deallocate`` directly on that
    # state can segfault; reload each mutable layer before releasing it.  This
    # exact sequence is exercised in the lifecycle integration test.
    info = mps.info
    mps.load_mutable()
    mps.deallocate()
    info.load_mutable()
    info.deallocate_mutable()
    if not remove_files:
        return
    _remove_owned_mps_files(driver, tag)


def _mps_determinant_coefficients(
    driver,
    mps,
    *,
    given_determinants,
    temporary_tag: str,
    remove_files: bool,
):
    """Return all SGF determinant coefficients without changing ``mps``.

    ``DMRGDriver.get_csf_coefficients`` moves a non-left-canonical input by
    making an internally tagged copy that it does not expose for lifecycle
    cleanup.  Make that copy explicitly with a module-owned tag instead.  The
    original time-evolved MPS therefore retains both its center and its exact
    amplitude for the next imaginary-time step.
    """

    working = mps
    temporary = None
    try:
        if int(mps.center) != 0:
            temporary = driver.copy_mps(mps, tag=temporary_tag)
            driver.align_mps_center(temporary, ref=0)
            working = temporary
        determinants, coefficients = driver.get_csf_coefficients(
            working,
            cutoff=0.0,
            given_dets=given_determinants,
            max_print=0,
            fci_conv=True,
            iprint=0,
        )
        determinants = np.asarray(determinants, dtype=np.uint8)
        coefficients = np.asarray(coefficients, dtype=np.complex128)
        if determinants.ndim != 2 or determinants.shape[1] != int(mps.n_sites):
            raise RuntimeError("Block2 returned an invalid determinant table")
        if coefficients.shape != (len(determinants),):
            raise RuntimeError("Block2 returned invalid determinant coefficients")
        if not np.all(np.isfinite(coefficients)):
            raise RuntimeError("Block2 returned non-finite determinant coefficients")
        if given_determinants is not None:
            expected = np.asarray(given_determinants, dtype=np.uint8)
            reorder = getattr(driver, "reorder_idx", None)
            if reorder is not None:
                reorder = np.asarray(reorder, dtype=np.int64)
                if (
                    reorder.shape != (int(mps.n_sites),)
                    or not np.array_equal(
                        np.sort(reorder),
                        np.arange(int(mps.n_sites)),
                    )
                ):
                    raise RuntimeError("Block2 has an invalid orbital reordering")
                # ``given_dets`` is interpreted in Block2's internal site
                # order, while the returned table is mapped back to the
                # caller's orbital order.  The coefficient array retains the
                # input row order.
                expected = expected[:, np.argsort(reorder)]
            if determinants.shape != expected.shape or not np.array_equal(
                determinants,
                expected,
            ):
                raise RuntimeError("Block2 changed the requested determinant ordering")
        return determinants, coefficients
    finally:
        if temporary is not None:
            _release_owned_mps(
                driver,
                temporary,
                temporary_tag,
                remove_files=remove_files,
            )


def _deallocate_mpo(mpo) -> None:
    # The installed Block2 wrappers share low-level symbolic/operator storage
    # between MPOs.  Explicit ``deallocate`` on a temporary source/identity
    # MPO can invalidate a still-live Hamiltonian MPO and segfault on the next
    # expectation.  Driver finalization is the supported lifetime boundary.
    del mpo


def _directory_size(path) -> int:
    if path is None or not os.path.isdir(path):
        return 0
    total = 0
    for root, _directories, filenames in os.walk(path):
        for filename in filenames:
            try:
                total += os.path.getsize(os.path.join(root, filename))
            except FileNotFoundError:
                pass
    return total


def _factorized_basis_operators(
    key: str,
    ncas: int,
) -> tuple[tuple[tuple[str, int], ...], ...]:
    """Return the nonredundant active operator basis of 2017 Table I."""

    key = _validate_source_key(key)
    if key == "ijr":
        return tuple((("C", active),) for active in range(ncas))
    if key == "rsi":
        return tuple((("D", active),) for active in range(ncas))
    if key == "ij":
        return tuple(
            (("C", left), ("C", right))
            for left in range(ncas)
            for right in range(left + 1, ncas)
        )
    if key == "rs":
        return tuple(
            (("D", right), ("D", left))
            for left in range(ncas)
            for right in range(left + 1, ncas)
        )
    if key == "ir":
        return tuple(
            (("C", creator), ("D", annihilator))
            for creator in range(ncas)
            for annihilator in range(ncas)
        )
    if key in ("ijrs", "i", "r"):
        return ()
    raise AssertionError(key)


def _factorized_class_plan(
    key: str,
    eris: spinor_helper._SpinorERIs,
    core_energy: Sequence[float],
    virtual_energy: Sequence[float],
    dm1,
    *,
    denominator_tolerance: float = 1.0e-12,
) -> _FactorizedClassPlan:
    """Derive one complex Table-I plan from the direct source specification."""

    key = _validate_source_key(key)
    dm1 = np.asarray(dm1, dtype=np.complex128)
    if dm1.shape != (eris.ncas, eris.ncas):
        raise ValueError("factorized Table-I plan needs an active 1-RDM")
    blocks = []
    gaps = []
    labels = []
    for free_indices in _iter_source_blocks(key, eris):
        source = _active_source(key, free_indices, eris)
        gap = _orbital_gap_for_block(
            key,
            free_indices,
            core_energy,
            virtual_energy,
        )
        if gap <= denominator_tolerance:
            raise RuntimeError(
                f"source {key}{free_indices} has non-positive gap {gap:.6e}"
            )
        blocks.append(source)
        gaps.append(gap)
        labels.append(key + ":" + ",".join(map(str, free_indices)))
    gaps_array = np.asarray(gaps, dtype=float)
    external_block_energies = {}

    if key == "ijrs":
        for label, source, gap in zip(labels, blocks, gaps_array):
            external_block_energies[label] = float(-abs(source.constant) ** 2 / gap)
        return _FactorizedClassPlan(
            key=key,
            basis_sources=(),
            coefficients=np.zeros((len(blocks), 0), dtype=np.complex128),
            gaps=gaps_array,
            block_labels=tuple(labels),
            external_energy=float(sum(external_block_energies.values())),
            external_block_energies=external_block_energies,
        )

    if key in ("i", "r"):
        coefficients = np.eye(len(blocks), dtype=np.complex128)
        return _FactorizedClassPlan(
            key=key,
            basis_sources=tuple(blocks),
            coefficients=coefficients,
            gaps=gaps_array,
            block_labels=tuple(labels),
            external_energy=0.0,
            external_block_energies={},
        )

    basis_operators = _factorized_basis_operators(key, eris.ncas)
    basis_index = {
        operators: position for position, operators in enumerate(basis_operators)
    }
    coefficients = np.zeros(
        (len(blocks), len(basis_operators)),
        dtype=np.complex128,
    )
    for block_index, (label, source, gap) in enumerate(zip(labels, blocks, gaps_array)):
        for term in source.terms:
            try:
                operator_index = basis_index[term.operators]
            except KeyError as error:
                raise RuntimeError(
                    f"source {label} contains an operator outside its Table-I basis"
                ) from error
            coefficients[block_index, operator_index] += term.coefficient
        if key == "ir":
            active_reference_amplitude = sum(
                term.coefficient * dm1[term.operators[0][1], term.operators[1][1]]
                for term in source.terms
            )
            numerator = abs(source.constant) ** 2 + 2.0 * np.real(
                source.constant.conjugate() * active_reference_amplitude
            )
            external_block_energies[label] = float(-numerator / gap)
        elif source.constant != 0.0:
            raise RuntimeError(f"source {label} has an unexpected scalar term")

    basis_sources = tuple(
        _ActiveSource(
            key=key,
            free_indices=(basis_position,),
            delta_n_active={
                "rsi": -1,
                "ijr": 1,
                "rs": -2,
                "ij": 2,
                "ir": 0,
            }[key],
            constant=0.0j,
            terms=(_ActiveOperatorTerm(operators, 1.0 + 0.0j),),
        )
        for basis_position, operators in enumerate(basis_operators)
    )
    return _FactorizedClassPlan(
        key=key,
        basis_sources=basis_sources,
        coefficients=coefficients,
        gaps=gaps_array,
        block_labels=tuple(labels),
        external_energy=float(sum(external_block_energies.values())),
        external_block_energies=external_block_energies,
    )


def _factorized_prefactor(
    plan: _FactorizedClassPlan,
    tau: float,
) -> np.ndarray:
    """Build `sum_eta c_eta* c_eta exp(-gap_eta tau)` explicitly."""

    tau = float(tau)
    if not np.isfinite(tau) or tau < 0.0:
        raise ValueError("imaginary time must be finite and non-negative")
    weighted = np.exp(-plan.gaps * tau)[:, None] * plan.coefficients
    return plan.coefficients.conj().T @ weighted


def _coefficient_coupling_mask(coefficients) -> np.ndarray:
    """Return Table-I GF entries that can be nonzero by exact support.

    No magnitude threshold is used: an entry ``(mu, nu)`` is omitted only
    when no external block contains both coefficients.  Its prefactor is then
    identically zero for every imaginary time, irrespective of orbital gaps.
    """

    coefficients = np.asarray(coefficients, dtype=np.complex128)
    if coefficients.ndim != 2 or not np.all(np.isfinite(coefficients)):
        raise ValueError("factorized coefficients must be a finite matrix")
    support = coefficients != 0.0
    incidence = support.astype(np.int64, copy=False)
    return (incidence.T @ incidence) != 0


def _orthogonalize_factorized_plan(
    plan: _FactorizedClassPlan,
    gram,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
    discarded_weight_atol: float = 1.0e-12,
    discarded_weight_rtol: float = 1.0e-10,
    coefficient_cutoff: float = 0.0,
    return_transform: bool = False,
):
    """Löwdin-condition a Table-I source basis without using an RDM.

    The primitive states are the columns of ``B`` and ``gram = B^dagger B``.
    For retained eigenpairs ``S U = U s`` this routine uses

    ``Phi = B U s**(-1/2)`` and ``c_orth = c U s**(1/2)``.

    Stable directions use the normalized Löwdin form above. Every remaining
    direction, including a zero eigenvalue of the *fitted* MPS Gram matrix, is
    retained as ``B U`` with row coefficient ``c^T U*``. Thus the threshold
    never projects a physical source. A composite direction may be skipped
    later only when its operator action is certified as an exact structural or
    particle-sector zero. The reported source weight remains useful
    diagnostics but is *not* an inverse-resolvent energy bound. Production
    propagates this conditioned basis because finite-bond-dimension time-step
    targeting is nonlinear and otherwise depends on an immaterial primitive
    basis choice.
    """

    primitive_count = len(plan.basis_sources)
    gram = np.asarray(gram, dtype=np.complex128)
    if gram.shape != (primitive_count, primitive_count):
        raise ValueError("factorized source Gram matrix has an inconsistent shape")
    if not np.all(np.isfinite(gram)):
        raise ValueError("factorized source Gram matrix is not finite")
    absolute_tolerance = float(absolute_tolerance)
    relative_tolerance = float(relative_tolerance)
    coefficient_cutoff = float(coefficient_cutoff)
    discarded_weight_atol = float(discarded_weight_atol)
    discarded_weight_rtol = float(discarded_weight_rtol)
    if (
        not np.isfinite(absolute_tolerance)
        or not np.isfinite(relative_tolerance)
        or absolute_tolerance < 0.0
        or relative_tolerance < 0.0
    ):
        raise ValueError("factorized rank tolerances must be finite and non-negative")
    if not np.isfinite(coefficient_cutoff) or coefficient_cutoff != 0.0:
        raise ValueError(
            "factorized coefficient cutoff must be finite and exactly zero"
        )
    if not np.all(np.isfinite(plan.coefficients)):
        raise ValueError("factorized source coefficients are not finite")
    if (
        not np.isfinite(discarded_weight_atol)
        or not np.isfinite(discarded_weight_rtol)
        or discarded_weight_atol < 0.0
        or discarded_weight_rtol < 0.0
    ):
        raise ValueError(
            "factorized discarded-source-weight tolerances must be finite "
            "and non-negative"
        )

    hermiticity_error = float(np.max(np.abs(gram - gram.conj().T), initial=0.0))
    hermitian_gram = 0.5 * (gram + gram.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(hermitian_gram)
    maximum_eigenvalue = float(max(0.0, eigenvalues[-1])) if eigenvalues.size else 0.0
    spectral_scale = max(
        1.0,
        float(np.max(np.abs(eigenvalues), initial=0.0)),
    )
    # Positive semidefiniteness is a mathematical property of a Gram matrix,
    # independent of the user-selected threshold that decides whether a
    # direction is safe to Löwdin-normalize.  Use only an eigensolver
    # backward-error allowance here; a large rank tolerance must never make a
    # genuinely indefinite matrix admissible.
    psd_tolerance = float(
        128.0
        * np.finfo(float).eps
        * max(1, primitive_count)
        * spectral_scale
    )
    with np.errstate(over="ignore", invalid="ignore"):
        rank_threshold = float(
            absolute_tolerance + relative_tolerance * maximum_eigenvalue
        )
    if not np.isfinite(rank_threshold):
        raise RuntimeError("factorized Gram rank threshold overflowed")
    minimum_eigenvalue = float(min(0.0, eigenvalues[0])) if eigenvalues.size else 0.0
    if minimum_eigenvalue < -psd_tolerance:
        raise RuntimeError(
            "factorized source Gram matrix is not positive semidefinite: "
            f"lambda_min={minimum_eigenvalue:.3e}, "
            f"tolerance={psd_tolerance:.3e}"
        )
    lowdin_normalized = eigenvalues > rank_threshold
    # The rank threshold controls conditioning, never source deletion.  The
    # Gram matrix is measured from independently fitted primitive MPSs, so
    # even an exactly zero *fitted* eigenvalue is not proof that the
    # corresponding exact operator combination annihilates the reference.
    # Carry every eigen-direction forward.  Stable directions are normalized;
    # all remaining directions are realized as unscaled composite sources and
    # can be skipped later only when the operator/MPO construction certifies an
    # exact structural or particle-sector zero.
    included = np.ones(eigenvalues.shape, dtype=bool)
    included_values = eigenvalues[included]
    included_vectors = eigenvectors[:, included]
    included_lowdin = lowdin_normalized[included]
    retained_count = int(np.count_nonzero(included))

    if retained_count:
        source_scales = np.ones(retained_count, dtype=float)
        source_scales[included_lowdin] = 1.0 / np.sqrt(included_values[included_lowdin])
        basis_transform = included_vectors * source_scales[None, :]
        # Rows of ``plan.coefficients`` store ket coefficients c^T.  For
        # Phi = B U D, the row coordinates are d^T = c^T U* D^-1.
        coefficient_transform = included_vectors.conj() / source_scales[None, :]
        conditioned_coefficients = plan.coefficients @ coefficient_transform
    else:
        source_scales = np.zeros(0, dtype=float)
        basis_transform = np.zeros((primitive_count, 0), dtype=np.complex128)
        conditioned_coefficients = np.zeros(
            (len(plan.gaps), 0),
            dtype=np.complex128,
        )

    delta_numbers = {source.delta_n_active for source in plan.basis_sources}
    if len(delta_numbers) > 1:
        raise RuntimeError("one factorized source basis mixes particle sectors")
    delta_n_active = next(iter(delta_numbers), 0)
    conditioned_sources = []
    discarded_l1 = 0.0
    for conditioned_index in range(retained_count):
        constant = 0.0j
        source_discarded_l1 = 0.0
        terms_by_string: dict[tuple[tuple[str, int], ...], complex] = {}
        for primitive_index, source in enumerate(plan.basis_sources):
            weight = basis_transform[primitive_index, conditioned_index]
            constant += weight * source.constant
            for term in source.terms:
                terms_by_string[term.operators] = (
                    terms_by_string.get(term.operators, 0.0j)
                    + weight * term.coefficient
                )
        terms = []
        for operators, coefficient in terms_by_string.items():
            if abs(coefficient) <= coefficient_cutoff:
                discarded_l1 += abs(coefficient)
                source_discarded_l1 += abs(coefficient)
                continue
            terms.append(_ActiveOperatorTerm(operators, coefficient))
        conditioned_sources.append(
            _ActiveSource(
                key=plan.key,
                free_indices=(conditioned_index,),
                delta_n_active=delta_n_active,
                constant=constant,
                terms=tuple(terms),
                discarded_coefficient_l1=source_discarded_l1,
            )
        )

    positive_values = eigenvalues[eigenvalues > 0.0]
    condition_number = (
        float(positive_values[-1] / positive_values[0])
        if positive_values.size
        else None
    )
    discarded_positive_weight = float(
        np.sum(np.clip(eigenvalues[~included], 0.0, None))
    )
    coefficient_eigen_coordinates = plan.coefficients @ eigenvectors.conj()
    positive_sqrt = np.sqrt(np.clip(eigenvalues, 0.0, None))
    physical_coordinates = coefficient_eigen_coordinates * positive_sqrt[None, :]
    total_source_weight = float(np.sum(np.abs(physical_coordinates) ** 2))
    discarded_positive = (~included) & (eigenvalues > 0.0)
    discarded_nonzero = ~included
    coefficient_supported = np.any(coefficient_eigen_coordinates != 0.0, axis=0)
    discarded_energy_relevant = discarded_nonzero & coefficient_supported
    discarded_source_weight = float(
        np.sum(np.abs(physical_coordinates[:, discarded_positive]) ** 2)
    )
    discarded_source_weight_limit = float(
        discarded_weight_atol + discarded_weight_rtol * total_source_weight
    )
    if not np.all(
        np.isfinite(
            (
                total_source_weight,
                discarded_source_weight,
                discarded_source_weight_limit,
            )
        )
    ):
        raise RuntimeError("factorized coefficient-weighted source norms overflowed")
    discarded_source_weight_gate = bool(
        discarded_source_weight <= discarded_source_weight_limit
    )
    diagnostics = {
        "method": "lossless_hybrid_lowdin",
        "primitive_count": primitive_count,
        "retained_rank": retained_count,
        "lowdin_normalized_rank": int(np.count_nonzero(lowdin_normalized)),
        "near_null_unscaled_rank": int(np.count_nonzero(included & ~lowdin_normalized)),
        "rank_threshold_controls_scaling_only": True,
        "all_fitted_gram_directions_retained": True,
        "conditioned_source_scales": [float(value) for value in source_scales],
        "conditioned_lowdin_normalized": [bool(value) for value in included_lowdin],
        "diagnostic_lowdin_rank_truncation_applied": bool(
            retained_count < primitive_count
        ),
        "discarded_direction_count": int(primitive_count - retained_count),
        "discarded_positive_direction_count": int(np.count_nonzero(discarded_positive)),
        "discarded_coefficient_supported_direction_count": int(
            np.count_nonzero(discarded_energy_relevant)
        ),
        "discarded_energy_projection_applied": bool(np.any(discarded_energy_relevant)),
        "rank_threshold": rank_threshold,
        "positive_semidefinite_tolerance": psd_tolerance,
        "positive_semidefinite_gate_passed": bool(
            minimum_eigenvalue >= -psd_tolerance
        ),
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "gram_hermiticity_error": hermiticity_error,
        "gram_eigenvalues": [float(value) for value in eigenvalues],
        "minimum_eigenvalue": minimum_eigenvalue,
        "maximum_eigenvalue": maximum_eigenvalue,
        "retained_condition_number": condition_number,
        "discarded_positive_weight": discarded_positive_weight,
        "total_coefficient_weighted_source_norm_squared": (total_source_weight),
        "discarded_coefficient_weighted_source_norm_squared": (discarded_source_weight),
        "discarded_source_weight_tolerance": (discarded_source_weight_limit),
        "discarded_source_weight_gate_passed": (discarded_source_weight_gate),
        "discarded_coefficient_l1": float(discarded_l1),
    }
    conditioned_plan = _FactorizedClassPlan(
        key=plan.key,
        basis_sources=tuple(conditioned_sources),
        coefficients=np.asarray(conditioned_coefficients),
        gaps=plan.gaps,
        block_labels=plan.block_labels,
        external_energy=plan.external_energy,
        external_block_energies=plan.external_block_energies,
    )
    if return_transform:
        return conditioned_plan, diagnostics, basis_transform
    return conditioned_plan, diagnostics


class X2CTMPSNEVPT2(lib.StreamObject):
    """State-specific fully uncontracted X2C-t-MPS-NEVPT2.

    ``algorithm='factorized_gf'`` is the paper-factorized production backend;
    ``algorithm='direct_source'`` retains every fixed external source as the
    correctness-first reference route.
    """

    def __init__(self, mc, frozen=0):
        if _utils._has_frozen_orbitals(frozen) or _utils._has_frozen_orbitals(
            getattr(mc, "frozen", None)
        ):
            raise NotImplementedError("nonzero frozen spinors are outside v1")
        self._mc = mc
        self._scf = mc._scf
        self.mol = self._scf.mol
        self.verbose = getattr(mc, "verbose", self.mol.verbose)
        self.stdout = getattr(mc, "stdout", self.mol.stdout)

        self.root = 0
        self.algorithm = "factorized_gf"
        self.te_type = "rk4"
        self.mo_coeff = getattr(mc, "mo_coeff", None)
        self.mo_energy = getattr(mc, "mo_energy", None)
        self.canonicalized = False
        self.eris = None
        self.eris_basis = None

        self.source_bond_dims = None
        self.source_n_sweeps = 8
        self.source_tol = 1.0e-10
        inherited_seed = getattr(
            getattr(mc, "fcisolver", None),
            "random_seed",
            1234,
        )
        self.source_random_seed = int(
            1234 if inherited_seed is None else inherited_seed
        )
        # Block2's ``multiply`` default is only 1e-6/1e-7 (squared linear
        # residual), which is too loose for the source norm/moment gates.
        self.source_thrds = [1.0e-12]
        self.source_cutoff = 1.0e-20
        self.source_coefficient_cutoff = 0.0
        self.source_residual_mode = "exact_mpo"
        self.source_residual_atol = 1.0e-8
        self.source_residual_rtol = 1.0e-8
        self.gf_rank_atol = 1.0e-12
        self.gf_rank_rtol = 1.0e-10
        self.gf_discarded_weight_atol = 1.0e-12
        self.gf_discarded_weight_rtol = 1.0e-10

        self.time_step = 0.05
        self.max_time = 20.0
        self.min_steps = 8
        self.max_steps = None
        self.time_bond_dims = None
        self.n_sub_sweeps = 2
        self.time_cutoff = 1.0e-20
        self.krylov_conv_thrd = 5.0e-8
        self.krylov_subspace_size = 20
        # The ordinary path evaluates every Green-function overlap directly
        # as an MPS contraction.  ``dense_csf`` is an exact small-sector
        # accelerator intended for molecular regression: it extracts all
        # determinant coefficients from each propagated MPS once, contracts
        # the full overlap vector with BLAS, and hard-cross-checks one native
        # MPS overlap at every time point.  It is never selected implicitly.
        self.factorized_overlap_backend = "mps"
        self.dense_csf_max_dimension = 4096
        self.dense_csf_overlap_atol = 1.0e-10
        self.dense_csf_overlap_rtol = 1.0e-9

        self.integration_mode = "simpson_tail"
        self.tail_fit_window = 5
        self.tail_fit_log_residual_tol = 0.1
        self.tail_fit_rate_sensitivity_tol = 0.25
        self.integrand_abs_tol = 1.0e-12
        self.tail_abs_tol = 1.0e-10
        self.tail_rel_tol = 1.0e-8
        self.energy_imag_tol = 1.0e-10
        self.correlation_residual_tol = 1.0e-9
        self.denominator_tol = 1.0e-12
        self.norm_tol = 1.0e-14

        self.dt_refinement = True
        self.dt_refinement_factor = 2
        self.dt_energy_tolerance = 1.0e-8
        self.store_time_series = False
        self.cleanup_mps = True
        self.audit_against_sc = False
        self.sc_audit_atol = 1.0e-9
        self.sc_audit_rtol = 1.0e-8

        self.rdm_atol = _utils._DEFAULT_RDM_ATOL
        self.rdm_rtol = _utils._DEFAULT_RDM_RTOL
        self.integral_roundoff_factor = _utils._DEFAULT_AO2MO_ROUNDOFF_FACTOR
        self.active_mpo_atol = 1.0e-10
        self.active_mpo_rtol = 1.0e-9

        self.reference_energy = None
        self.active_reference_energy = None
        self.e_corr = None
        self.sub_eners = {}
        self.sub_times = {}
        self.sub_diagnostics = {}
        self.source_diagnostics = {}
        self.propagation_diagnostics = {}
        self.integration_diagnostics = {}
        self.integral_symmetry_diagnostics = None
        self.dm1_diagnostics = None
        self.dm1 = None
        self.active_mpo_diagnostics = None
        self.sc_moment_diagnostics = None
        self.source_counts = {}
        self.time_series = {}
        self._pass_time_series = {}
        self.resource_diagnostics = {}
        self.control_diagnostics = {}
        # Populated afresh by ``kernel`` and reused only between its primary
        # and dt-refined passes.  The exact B^dagger B MPO can be much wider
        # than B itself, so do not build the identical diagnostic twice.
        self._rhs_norm_cache = {}
        self._keys = set(self.__dict__)

    def _reset_run_results(self):
        """Clear result state so a failed rerun cannot expose stale energies."""

        self.reference_energy = None
        self.active_reference_energy = None
        self.e_corr = None
        self.sub_eners = {}
        self.sub_times = {}
        self.sub_diagnostics = {}
        self.source_diagnostics = {}
        self.propagation_diagnostics = {}
        self.integration_diagnostics = {}
        self.integral_symmetry_diagnostics = None
        self.dm1_diagnostics = None
        self.dm1 = None
        self.active_mpo_diagnostics = None
        self.sc_moment_diagnostics = None
        self.source_counts = {}
        self.time_series = {}
        self._pass_time_series = {}
        self.resource_diagnostics = {}
        self.control_diagnostics = {}
        self._rhs_norm_cache = {}

    @property
    def e_tot(self):
        if self.e_corr is None:
            return None
        return np.asarray(self.reference_energy) + np.asarray(self.e_corr)

    def _validate_controls(self, *, algorithm, te_type, time_grid):
        if algorithm not in ("direct_source", "factorized_gf"):
            raise ValueError("algorithm must be 'direct_source' or 'factorized_gf'")
        if te_type not in ("rk4", "tdvp"):
            raise ValueError("te_type must be 'rk4' or 'tdvp'")
        if self.factorized_overlap_backend not in ("mps", "dense_csf"):
            raise ValueError(
                "factorized_overlap_backend must be 'mps' or 'dense_csf'"
            )
        if (
            self.factorized_overlap_backend == "dense_csf"
            and algorithm != "factorized_gf"
        ):
            raise ValueError(
                "dense_csf overlap acceleration is only defined for factorized_gf"
            )
        if self.source_residual_mode not in ("exact_mpo", "radial_only"):
            raise ValueError(
                "source_residual_mode must be 'exact_mpo' or 'radial_only'"
            )
        positive_float_names = (
            "source_tol",
            "time_step",
            "max_time",
            "krylov_conv_thrd",
            "integrand_abs_tol",
            "tail_abs_tol",
            "tail_rel_tol",
            "energy_imag_tol",
            "correlation_residual_tol",
            "denominator_tol",
            "norm_tol",
            "dt_energy_tolerance",
            "tail_fit_log_residual_tol",
            "tail_fit_rate_sensitivity_tol",
        )
        for name in positive_float_names:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        # Block2 explicitly accepts a zero density-matrix cutoff.  This is a
        # useful exact-small-system limit: tiny Schmidt directions can carry
        # appreciable weight after contraction of an ill-conditioned source
        # representation even when the reported discarded weight is tiny.
        for name in ("source_cutoff", "time_cutoff"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        coefficient_cutoff = float(self.source_coefficient_cutoff)
        if not np.isfinite(coefficient_cutoff) or coefficient_cutoff != 0.0:
            raise ValueError(
                "source_coefficient_cutoff must be exactly zero; production "
                "t-MPS-NEVPT2 does not permit magnitude-based source pruning"
            )
        for name in (
            "source_residual_atol",
            "source_residual_rtol",
            "gf_rank_atol",
            "gf_rank_rtol",
            "gf_discarded_weight_atol",
            "gf_discarded_weight_rtol",
            "dense_csf_overlap_atol",
            "dense_csf_overlap_rtol",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.source_residual_atol == 0.0 and self.source_residual_rtol == 0.0:
            raise ValueError("at least one source residual tolerance must be positive")
        if self.gf_rank_atol == 0.0 and self.gf_rank_rtol == 0.0:
            raise ValueError("at least one factorized rank tolerance must be positive")
        if (
            self.gf_discarded_weight_atol == 0.0
            and self.gf_discarded_weight_rtol == 0.0
        ):
            raise ValueError(
                "at least one factorized discarded-source-weight tolerance "
                "must be positive"
            )
        if (
            self.dense_csf_overlap_atol == 0.0
            and self.dense_csf_overlap_rtol == 0.0
        ):
            raise ValueError(
                "at least one dense-CSF overlap tolerance must be positive"
            )
        for name in ("sc_audit_atol", "sc_audit_rtol"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        for pair_name, absolute_name, relative_name in (
            ("RDM", "rdm_atol", "rdm_rtol"),
            ("active-MPO", "active_mpo_atol", "active_mpo_rtol"),
        ):
            absolute = float(getattr(self, absolute_name))
            relative = float(getattr(self, relative_name))
            if (
                not np.isfinite(absolute)
                or not np.isfinite(relative)
                or absolute < 0.0
                or relative < 0.0
            ):
                raise ValueError(
                    f"{absolute_name}/{relative_name} must be finite and "
                    "non-negative"
                )
            if absolute == 0.0 and relative == 0.0:
                raise ValueError(f"at least one {pair_name} tolerance must be positive")
        for name in (
            "source_n_sweeps",
            "min_steps",
            "n_sub_sweeps",
            "krylov_subspace_size",
            "tail_fit_window",
            "dense_csf_max_dimension",
        ):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if int(self.tail_fit_window) < 3:
            raise ValueError("tail_fit_window must be at least three")
        if self.max_steps is not None and int(self.max_steps) <= 0:
            raise ValueError("max_steps must be positive when supplied")
        if int(self.dt_refinement_factor) < 2:
            raise ValueError("dt_refinement_factor must be at least two")
        source_random_seed = int(self.source_random_seed)
        if not 0 <= source_random_seed <= np.iinfo(np.int32).max:
            raise ValueError("source_random_seed must be a non-negative 32-bit integer")
        if self.integration_mode not in (
            "simpson_tail",
            "newton_cotes_8_tail",
            "trapezoid_tail",
        ):
            raise ValueError("unknown integration_mode")
        # Validate user-supplied schedules here; source-dependent defaults are
        # resolved only after the retained reference/source MPS is available.
        if self.source_bond_dims is not None:
            self._bond_schedule(
                self.source_bond_dims,
                default=1,
                name="source_bond_dims",
            )
        if self.time_bond_dims is not None:
            self._bond_schedule(
                self.time_bond_dims,
                default=1,
                name="time_bond_dims",
            )
        if self.source_thrds is not None:
            thresholds = np.atleast_1d(np.asarray(self.source_thrds, dtype=float))
            if (
                thresholds.ndim != 1
                or thresholds.size == 0
                or not np.all(np.isfinite(thresholds))
                or np.any(thresholds <= 0.0)
            ):
                raise ValueError("source_thrds must contain finite positive thresholds")
        if time_grid is not None:
            grid = _validate_time_grid(time_grid)
            if self.integration_mode in (
                "simpson_tail",
                "newton_cotes_8_tail",
            ):
                spacing = np.diff(grid)
                uniform_tol = (
                    64.0
                    * np.finfo(float).eps
                    * max(
                        1.0,
                        abs(grid[-1]),
                    )
                )
                if np.max(np.abs(spacing - spacing[0])) > uniform_tol:
                    raise ValueError(
                        f"{self.integration_mode} requires a uniform time_grid"
                    )

    def _prepare_root(
        self,
        mc,
        *,
        root: int,
        mo_coeff,
        dm1,
        eris,
        eris_basis: str,
    ) -> _PreparedTMPSRoot:
        solver = mc.fcisolver
        driver, reference_mps = _utils._root_ket(solver, root)
        from pyblock2.driver.core import SymmetryTypes

        if SymmetryTypes.SGF not in driver.symm_type:
            raise RuntimeError("t-MPS-NEVPT2 requires Block2 SGF symmetry")
        if SymmetryTypes.CPX not in driver.symm_type:
            raise RuntimeError("t-MPS-NEVPT2 requires Block2 complex mode")
        if getattr(driver, "mpi", None) is not None:
            raise NotImplementedError(
                "this Block2 build's source-MPO/td_dmrg MPI path has not been "
                "validated; refusing a silent serial or replacement-driver fallback"
            )
        active_mpo = getattr(solver, "_active_mpo", None)
        if active_mpo is None:
            raise RuntimeError(
                "the converged DMRGCI active Hamiltonian MPO is unavailable"
            )

        ncas = int(mc.ncas)
        nelec = _utils._total_nelec(
            getattr(solver, "nelecas", getattr(mc, "nelecas", None))
        )
        if dm1 is None:
            dm1 = _utils.make_rdm1(solver, root=root)
        dm1, dm1_diagnostics = _validate_dm1_only(
            dm1,
            ncas,
            nelec,
            atol=float(self.rdm_atol),
            rtol=float(self.rdm_rtol),
        )

        original_mo = np.asarray(mo_coeff)
        semicanonical_mo, orbital_energy = _utils.semicanonicalize(
            mc,
            original_mo,
            dm1,
            root,
            canonicalized=bool(self.canonicalized),
            mo_energy=self.mo_energy,
            verbose=self.verbose,
        )
        if eris is None:
            prepared_eris = _utils._dense_eris_from_mc(
                mc,
                semicanonical_mo,
                roundoff_factor=self.integral_roundoff_factor,
            )
        else:
            if not isinstance(eris, spinor_helper._SpinorERIs):
                raise TypeError("eris must be a spinor_helper._SpinorERIs")
            if (eris.ncore, eris.ncas, eris.nmo) != (
                int(mc.ncore),
                ncas,
                semicanonical_mo.shape[1],
            ):
                raise ValueError("eris partition does not match the CASSCF object")
            prepared_eris = eris
            if eris_basis == "input_mo" and not np.array_equal(
                semicanonical_mo, original_mo
            ):
                overlap = np.asarray(mc._scf.get_ovlp())
                rotation = original_mo.T.conj() @ overlap @ semicanonical_mo
                prepared_eris = _utils._rotate_eris(eris, rotation)
        integral_diagnostics = getattr(
            prepared_eris,
            "symmetry_diagnostics",
            None,
        )

        snapshot = getattr(solver, "checkpoint_hamiltonian", None)
        if snapshot is None:
            raise RuntimeError(
                "DMRGCI checkpoint Hamiltonian is required to audit the retained MPO"
            )
        h1_error = _maximum_array_error(
            snapshot["h1e"],
            prepared_eris.get_h1eff("AA"),
            name="active one-electron Hamiltonian",
        )
        eri_error = _maximum_array_error(
            snapshot["eri"],
            prepared_eris.get_chem("AAAA"),
            name="active two-electron Hamiltonian",
        )
        active_scale = max(
            1.0,
            float(np.max(np.abs(snapshot["h1e"]), initial=0.0)),
            float(np.max(np.abs(snapshot["eri"]), initial=0.0)),
        )
        active_tolerance = float(
            self.active_mpo_atol + self.active_mpo_rtol * active_scale
        )
        if max(h1_error, eri_error) > active_tolerance:
            raise RuntimeError(
                "the retained DMRG active MPO is not in the prepared active "
                f"basis: h1={h1_error:.3e}, eri={eri_error:.3e}, "
                f"tolerance={active_tolerance:.3e}"
            )

        identity_mpo = driver.get_identity_mpo()
        try:
            norm_value = driver.expectation(
                reference_mps,
                identity_mpo,
                reference_mps,
            )
        finally:
            _deallocate_mpo(identity_mpo)
        reference_norm, norm_imaginary = _checked_real_scalar(
            norm_value,
            name="active reference norm",
            imaginary_tolerance=self.energy_imag_tol,
        )
        norm_imaginary_limit = float(
            self.energy_imag_tol * max(1.0, abs(reference_norm))
        )
        if reference_norm <= self.norm_tol:
            raise RuntimeError("active reference MPS has zero norm")
        reference_norm_error = abs(reference_norm - 1.0)
        reference_norm_tolerance = float(self.rdm_atol + self.rdm_rtol)
        reference_norm_gate = bool(reference_norm_error <= reference_norm_tolerance)
        if not reference_norm_gate:
            _utils._warn_numerical(
                "active reference MPS norm differs from one by "
                f"{reference_norm_error:.3e}"
            )
        active_value = driver.expectation(
            reference_mps,
            active_mpo,
            reference_mps,
        )
        active_energy, active_imaginary = _checked_real_scalar(
            active_value / reference_norm,
            name="active Dyall reference energy",
            imaginary_tolerance=self.energy_imag_tol,
        )
        active_imaginary_limit = float(
            self.energy_imag_tol * max(1.0, abs(active_energy))
        )
        active_mpo_diagnostics = {
            "reference_norm_expectation": [
                float(complex(norm_value).real),
                float(complex(norm_value).imag),
            ],
            "h1_input_error": h1_error,
            "eri_input_error": eri_error,
            "input_tolerance": active_tolerance,
            "input_gate_passed": True,
            "reference_norm": reference_norm,
            "reference_norm_imaginary_residual": norm_imaginary,
            "reference_norm_imaginary_tolerance": norm_imaginary_limit,
            "reference_norm_imaginary_gate_passed": bool(
                norm_imaginary <= norm_imaginary_limit
            ),
            "reference_norm_error": float(reference_norm_error),
            "reference_norm_tolerance": reference_norm_tolerance,
            "reference_norm_gate_passed": reference_norm_gate,
            "active_reference_energy": active_energy,
            "active_energy_expectation": [
                float(complex(active_value).real),
                float(complex(active_value).imag),
            ],
            "active_reference_energy_imaginary_residual": active_imaginary,
            "active_reference_energy_imaginary_tolerance": (active_imaginary_limit),
            "active_reference_energy_imaginary_gate_passed": bool(
                active_imaginary <= active_imaginary_limit
            ),
            "site_map": list(_active_site_map(driver, ncas)),
            "mpo_source": "mc.fcisolver._active_mpo",
        }
        ncore = prepared_eris.ncore
        nocc = prepared_eris.nocc
        return _PreparedTMPSRoot(
            root=root,
            reference_energy=_utils._reference_energy(mc, root),
            mo_coeff=np.asarray(semicanonical_mo),
            orbital_energy=np.asarray(orbital_energy, dtype=float),
            eris=prepared_eris,
            core_energy=np.asarray(orbital_energy[:ncore], dtype=float),
            virtual_energy=np.asarray(orbital_energy[nocc:], dtype=float),
            dm1=dm1,
            dm1_diagnostics=dm1_diagnostics,
            integral_symmetry_diagnostics=integral_diagnostics,
            driver=driver,
            reference_mps=reference_mps,
            active_mpo=active_mpo,
            reference_norm=reference_norm,
            active_reference_energy=active_energy,
            active_mpo_diagnostics=active_mpo_diagnostics,
        )

    def _bond_schedule(self, values, *, default: int, name: str) -> list[int]:
        if values is None:
            return [int(default)]
        if np.isscalar(values):
            values = [values]
        result = [int(value) for value in values]
        if not result or any(value <= 0 for value in result):
            raise ValueError(f"{name} must contain positive bond dimensions")
        return result

    def _control_snapshot(self, prepared, *, time_grid) -> dict[str, Any]:
        """Return the complete numerical setup recorded for this run."""

        reference_bond = int(prepared.reference_mps.info.bond_dim)
        source_bonds = self._bond_schedule(
            self.source_bond_dims,
            default=max(1, 2 * reference_bond),
            name="source_bond_dims",
        )
        requested_time_bonds = (
            None
            if self.time_bond_dims is None
            else self._bond_schedule(
                self.time_bond_dims,
                default=source_bonds[-1],
                name="time_bond_dims",
            )
        )
        explicit_grid = None if time_grid is None else _validate_time_grid(time_grid)
        grid_diagnostics = {
            "explicit": explicit_grid is not None,
            "point_count": (None if explicit_grid is None else int(len(explicit_grid))),
            "final_time": (None if explicit_grid is None else float(explicit_grid[-1])),
            "minimum_step": (
                None if explicit_grid is None else float(np.min(np.diff(explicit_grid)))
            ),
            "maximum_step": (
                None if explicit_grid is None else float(np.max(np.diff(explicit_grid)))
            ),
        }
        return {
            "source": {
                "reference_bond_dimension": reference_bond,
                "bond_dimensions": source_bonds,
                "n_sweeps": int(self.source_n_sweeps),
                "random_seed": int(self.source_random_seed),
                "tolerance": float(self.source_tol),
                "thresholds": (
                    None
                    if self.source_thrds is None
                    else np.atleast_1d(
                        np.asarray(self.source_thrds, dtype=float)
                    ).tolist()
                ),
                "cutoff": float(self.source_cutoff),
                "coefficient_cutoff": float(self.source_coefficient_cutoff),
                "full_residual_mode": self.source_residual_mode,
                "full_residual_atol": float(self.source_residual_atol),
                "full_residual_rtol": float(self.source_residual_rtol),
                "factorized_rank_atol": float(self.gf_rank_atol),
                "factorized_rank_rtol": float(self.gf_rank_rtol),
                "factorized_discarded_weight_atol": float(
                    self.gf_discarded_weight_atol
                ),
                "factorized_discarded_weight_rtol": float(
                    self.gf_discarded_weight_rtol
                ),
            },
            "propagation": {
                "te_type": self.te_type,
                "time_step": float(self.time_step),
                "max_time": float(self.max_time),
                "min_steps": int(self.min_steps),
                "max_steps": (None if self.max_steps is None else int(self.max_steps)),
                "bond_dimensions": requested_time_bonds,
                "bond_dimension_default": (
                    None
                    if requested_time_bonds is not None
                    else "fitted source MPS bond dimension"
                ),
                "n_sub_sweeps": int(self.n_sub_sweeps),
                "cutoff": float(self.time_cutoff),
                "krylov_convergence_threshold": float(self.krylov_conv_thrd),
                "krylov_subspace_size": int(self.krylov_subspace_size),
                "normalize_mps": False,
                "factorized_overlap_backend": self.factorized_overlap_backend,
                "dense_csf_max_dimension": int(self.dense_csf_max_dimension),
                "dense_csf_overlap_atol": float(self.dense_csf_overlap_atol),
                "dense_csf_overlap_rtol": float(self.dense_csf_overlap_rtol),
                "time_grid": grid_diagnostics,
            },
            "integration": {
                "mode": self.integration_mode,
                "tail_fit_window": int(self.tail_fit_window),
                "tail_fit_log_residual_tol": float(self.tail_fit_log_residual_tol),
                "tail_fit_rate_sensitivity_tol": float(
                    self.tail_fit_rate_sensitivity_tol
                ),
                "integrand_abs_tol": float(self.integrand_abs_tol),
                "tail_abs_tol": float(self.tail_abs_tol),
                "tail_rel_tol": float(self.tail_rel_tol),
                "energy_imag_tol": float(self.energy_imag_tol),
                "correlation_residual_tol": float(self.correlation_residual_tol),
                "denominator_tol": float(self.denominator_tol),
                "norm_tol": float(self.norm_tol),
                "dt_refinement_requested": bool(self.dt_refinement),
                "dt_refinement_effective": bool(
                    self.dt_refinement and explicit_grid is None
                ),
                "dt_refinement_factor": int(self.dt_refinement_factor),
                "dt_energy_tolerance": float(self.dt_energy_tolerance),
            },
            "audits": {
                "rdm_atol": float(self.rdm_atol),
                "rdm_rtol": float(self.rdm_rtol),
                "active_mpo_atol": float(self.active_mpo_atol),
                "active_mpo_rtol": float(self.active_mpo_rtol),
                "integral_roundoff_factor": float(self.integral_roundoff_factor),
                "audit_against_sc": bool(self.audit_against_sc),
                "sc_audit_atol": float(self.sc_audit_atol),
                "sc_audit_rtol": float(self.sc_audit_rtol),
            },
            "storage": {
                "store_time_series": bool(self.store_time_series),
                "cleanup_mps": bool(self.cleanup_mps),
                "sequential_sources": True,
            },
        }

    def _make_source_mpo(self, prepared, source, *, add_ident=True):
        site_map = _active_site_map(prepared.driver, prepared.eris.ncas)
        terms, constant = _source_mpo_terms(source, site_map=site_map)
        mpo = prepared.driver.get_mpo_any_fermionic(
            terms,
            ecore=constant if constant != 0.0 else None,
            cutoff=float(self.source_cutoff),
            iprint=1 if self.verbose >= logger.DEBUG else 0,
            add_ident=bool(add_ident),
        )
        q_label = mpo.op.q_label
        if int(q_label.n) != source.delta_n_active:
            _deallocate_mpo(mpo)
            raise RuntimeError(
                f"source {source.key} MPO has particle change {q_label.n}; "
                f"expected {source.delta_n_active}"
            )
        return mpo

    def _exact_source_fit_diagnostics(self, *, kind, rhs_norm_squared):
        """Return one uniform diagnostic schema for analytic exact sources."""

        rhs_norm_squared = float(rhs_norm_squared)
        if not np.isfinite(rhs_norm_squared) or rhs_norm_squared < 0.0:
            raise ValueError("exact source norm squared must be non-negative")
        requested_tolerance = float(
            self.source_residual_atol
            + self.source_residual_rtol * math.sqrt(rhs_norm_squared)
        )
        return {
            "exact_rhs_norm_squared": rhs_norm_squared,
            "rhs_norm_expectation": [rhs_norm_squared, 0.0],
            "rhs_norm_imaginary_residual": 0.0,
            "rhs_norm_imaginary_tolerance": float(
                self.energy_imag_tol * max(1.0, rhs_norm_squared)
            ),
            "rhs_norm_imaginary_gate_passed": True,
            "radial_stationarity_residual": 0.0,
            "radial_stationarity_relative_residual": 0.0,
            "radial_stationarity_tolerance": 0.0,
            "radial_stationarity_gate_passed": True,
            "full_fit_residual_available": True,
            "full_fit_residual_squared_raw": 0.0,
            "full_fit_residual_squared": 0.0,
            "full_fit_residual_norm": 0.0,
            "full_fit_relative_residual": 0.0,
            "full_fit_residual_tolerance": requested_tolerance,
            "full_fit_residual_effective_tolerance": requested_tolerance,
            "full_fit_residual_roundoff_tolerance_squared": 0.0,
            "full_fit_residual_nonnegative_gate_passed": True,
            "full_fit_residual_reason": None,
            "fit_gate_kind": str(kind),
            "fit_gate_passed": True,
        }

    def _exact_rhs_norm_squared(self, prepared, source, *, tag):
        """Measure ``<Psi0|B^dagger B|Psi0>`` as an MPO product."""

        from pyblock2.algebra.io import MPOTools

        cache = getattr(self, "_rhs_norm_cache", None)
        cache_key = (
            id(prepared.driver),
            id(prepared.reference_mps),
            source,
        )
        if cache is not None and cache_key in cache:
            norm, cached_diagnostics = cache[cache_key]
            diagnostics = dict(cached_diagnostics)
            diagnostics["rhs_norm_cache_hit"] = True
            return norm, diagnostics

        driver = prepared.driver
        source_raw = None
        adjoint_raw = None
        norm_mpo = None
        started = time.perf_counter()
        try:
            source_raw = self._make_source_mpo(
                prepared,
                source,
                add_ident=False,
            )
            adjoint_raw = self._make_source_mpo(
                prepared,
                _adjoint_active_source(source),
                add_ident=False,
            )
            python_source = MPOTools.from_block2(_unwrap_general_mpo(source_raw))
            python_adjoint = MPOTools.from_block2(_unwrap_general_mpo(adjoint_raw))
            python_norm = python_adjoint @ python_source
            python_bond_dimensions = [
                int(sum(blocks.values())) for blocks in python_norm.get_bond_dims()
            ]
            norm_mpo = _to_block2_mpo_lossless(
                python_norm,
                basis=list(driver.basis),
                tag=f"{tag}-BDB",
                add_ident=True,
            )
            if int(norm_mpo.op.q_label.n) != 0:
                raise RuntimeError("B^dagger B MPO is not particle conserving")
            value = driver.expectation(
                prepared.reference_mps,
                norm_mpo,
                prepared.reference_mps,
            )
            norm, imaginary = _checked_real_scalar(
                value,
                name=f"{source.key}{source.free_indices} exact RHS norm squared",
                imaginary_tolerance=self.energy_imag_tol,
            )
            imaginary_limit = float(self.energy_imag_tol * max(1.0, abs(norm)))
            if norm < -self.norm_tol:
                raise RuntimeError(
                    f"source {source.key}{source.free_indices} has negative "
                    f"exact RHS norm squared {norm:.3e}"
                )
            norm = max(norm, 0.0)
            diagnostics = {
                "rhs_norm_expectation": [
                    float(complex(value).real),
                    float(complex(value).imag),
                ],
                "rhs_norm_imaginary_residual": imaginary,
                "rhs_norm_imaginary_tolerance": imaginary_limit,
                "rhs_norm_imaginary_gate_passed": bool(imaginary <= imaginary_limit),
                "rhs_norm_mpo_max_bond_dimension": max(
                    python_bond_dimensions,
                    default=1,
                ),
                "rhs_norm_mpo_bond_dimensions": python_bond_dimensions,
                "rhs_norm_mpo_conversion_block_floor": 0.0,
                "rhs_norm_mpo_conversion_zero_policy": "exact_zero_only",
                "rhs_norm_wall_seconds": time.perf_counter() - started,
                "rhs_norm_cache_hit": False,
            }
            if cache is not None:
                cache[cache_key] = (norm, dict(diagnostics))
            return norm, diagnostics
        finally:
            _deallocate_mpo(norm_mpo)
            _deallocate_mpo(adjoint_raw)
            _deallocate_mpo(source_raw)

    def _build_source_mps(
        self,
        prepared,
        source,
        *,
        tag_factory,
        residual_mode=None,
    ):
        start = time.perf_counter()
        driver = prepared.driver
        reference = prepared.reference_mps
        tag = tag_factory(source.key, source.free_indices, "SRC")
        residual_mode = (
            self.source_residual_mode if residual_mode is None else str(residual_mode)
        )
        if residual_mode not in ("exact_mpo", "radial_only"):
            raise ValueError("residual_mode must be 'exact_mpo' or 'radial_only'")
        if source.discarded_coefficient_l1 != 0.0:
            raise RuntimeError(
                f"source {source.key}{source.free_indices} contains discarded "
                "coefficients; magnitude-pruned sources are not a production "
                "t-MPS-NEVPT2 path"
            )
        # Symmetry and spin selection rules can make an otherwise valid
        # fixed-external block identically zero.  Block2 0.5.4rc16 rejects an
        # empty ``get_mpo_any_fermionic`` input before it can attach the
        # theoretical particle-number label, so recognize the mathematical
        # zero before constructing an MPO.  This is an exact structural
        # zero, not coefficient-magnitude screening.
        if source.constant == 0.0 and not source.terms:
            target_n = int(reference.info.target.n) + source.delta_n_active
            return None, {
                "tag": tag,
                "particle_number": target_n,
                "particle_number_change": source.delta_n_active,
                "zero_by_particle_sector": not (0 <= target_n <= prepared.eris.ncas),
                "zero_by_coefficients": True,
                "source_term_count": 0,
                "constant": [0.0, 0.0],
                "discarded_coefficient_l1": source.discarded_coefficient_l1,
                **self._exact_source_fit_diagnostics(
                    kind="exact_structural_zero",
                    rhs_norm_squared=0.0,
                ),
                "wall_seconds": time.perf_counter() - start,
            }

        source_mpo = self._make_source_mpo(prepared, source)
        source_mps = None
        try:
            target = reference.info.target + source_mpo.op.q_label
            target_n = int(target.n)
            if not 0 <= target_n <= prepared.eris.ncas:
                return None, {
                    "tag": tag,
                    "particle_number": target_n,
                    "particle_number_change": source.delta_n_active,
                    "zero_by_particle_sector": True,
                    "zero_by_coefficients": False,
                    "source_term_count": len(source.terms),
                    **self._exact_source_fit_diagnostics(
                        kind="exact_particle_sector_zero",
                        rhs_norm_squared=0.0,
                    ),
                    "wall_seconds": time.perf_counter() - start,
                }
            reference_bond = int(reference.info.bond_dim)
            source_bonds = self._bond_schedule(
                self.source_bond_dims,
                default=max(1, 2 * reference_bond),
                name="source_bond_dims",
            )
            source_mps = driver.get_random_mps(
                tag=tag,
                bond_dim=source_bonds[0],
                center=int(reference.center),
                dot=int(reference.dot),
                target=target,
                left_vacuum=source_mpo.left_vacuum,
            )
            reported_norm = driver.multiply(
                source_mps,
                source_mpo,
                reference,
                n_sweeps=int(self.source_n_sweeps),
                tol=float(self.source_tol),
                bond_dims=[reference_bond],
                bra_bond_dims=source_bonds,
                thrds=self.source_thrds,
                cutoff=float(self.source_cutoff),
                iprint=1 if self.verbose >= logger.DEBUG else 0,
            )
            identity = driver.get_identity_mpo()
            try:
                norm_value = driver.expectation(source_mps, identity, source_mps)
                cross_value = driver.expectation(
                    source_mps,
                    source_mpo,
                    reference,
                )
            finally:
                _deallocate_mpo(identity)
            source_particle_change = int(source_mpo.op.q_label.n)
            # The wrapped B MPO is no longer needed after the mixed overlap.
            # Release it before constructing the wider B^dagger B diagnostic.
            _deallocate_mpo(source_mpo)
            source_mpo = None
            norm, norm_imaginary = _checked_real_scalar(
                norm_value,
                name=f"{source.key}{source.free_indices} source norm squared",
                imaginary_tolerance=self.energy_imag_tol,
            )
            norm_imaginary_limit = float(self.energy_imag_tol * max(1.0, abs(norm)))
            if norm < -self.norm_tol:
                raise RuntimeError("source MPS has a negative norm squared")
            norm = max(norm, 0.0)
            cross = _utils._complex_scalar(
                cross_value,
                name=f"{source.key}{source.free_indices} source fit overlap",
            )
            radial_stationarity_residual = abs(norm - cross)
            fit_scale = max(1.0, norm, abs(cross))
            radial_stationarity_tolerance = (
                max(
                    float(self.source_tol),
                    100.0 * np.finfo(float).eps,
                )
                * fit_scale
            )
            radial_stationarity_gate_passed = bool(
                radial_stationarity_residual <= radial_stationarity_tolerance
            )
            if not radial_stationarity_gate_passed:
                _utils._warn_numerical(
                    f"source {source.key}{source.free_indices} radial "
                    "Galerkin stationarity residual "
                    f"{radial_stationarity_residual:.3e} exceeds "
                    f"{radial_stationarity_tolerance:.3e}"
                )

            rhs_norm = None
            rhs_diagnostics = {}
            full_residual_squared = None
            raw_residual_squared = None
            full_residual = None
            full_relative_residual = None
            full_residual_tolerance = None
            full_residual_effective_tolerance = None
            residual_nonnegative_gate = None
            source_operator_l1 = float(
                abs(source.constant)
                + math.fsum(abs(term.coefficient) for term in source.terms)
            )
            source_norm_upper_bound = float(
                prepared.reference_norm * source_operator_l1**2
            )
            if not np.isfinite(source_norm_upper_bound):
                raise RuntimeError(
                    f"source {source.key}{source.free_indices} operator-norm "
                    "roundoff bound overflowed"
                )
            if residual_mode == "exact_mpo":
                rhs_norm, rhs_diagnostics = self._exact_rhs_norm_squared(
                    prepared,
                    source,
                    tag=tag,
                )
                raw_residual_squared = float(norm + rhs_norm - 2.0 * cross.real)
                if not np.isfinite(raw_residual_squared):
                    raise RuntimeError(
                        f"source {source.key}{source.free_indices} has a "
                        "non-finite full fit residual"
                    )
                cancellation_scale = max(
                    norm + rhs_norm + 2.0 * abs(cross),
                    source_norm_upper_bound,
                    self.norm_tol**2,
                )
                residual_roundoff_tolerance = float(
                    512.0 * np.finfo(float).eps * cancellation_scale
                )
                full_residual_squared = max(raw_residual_squared, 0.0)
                full_residual = math.sqrt(full_residual_squared)
                rhs_amplitude = math.sqrt(rhs_norm)
                full_relative_residual = float(
                    full_residual / max(rhs_amplitude, self.norm_tol)
                )
                full_residual_tolerance = float(
                    self.source_residual_atol
                    + self.source_residual_rtol * rhs_amplitude
                )
                effective_residual_squared_limit = float(
                    full_residual_tolerance**2 + residual_roundoff_tolerance
                )
                residual_nonnegative_gate = bool(
                    raw_residual_squared >= -effective_residual_squared_limit
                )
                full_residual_effective_tolerance = math.sqrt(
                    effective_residual_squared_limit
                )
                fit_gate_passed = bool(
                    residual_nonnegative_gate
                    and rhs_diagnostics["rhs_norm_imaginary_gate_passed"]
                    and raw_residual_squared <= effective_residual_squared_limit
                )
                if not fit_gate_passed:
                    _utils._warn_numerical(
                        f"source {source.key}{source.free_indices} full "
                        f"fit residual {full_residual:.3e} exceeds the "
                        f"effective {full_residual_effective_tolerance:.3e} "
                        f"(requested {full_residual_tolerance:.3e}) or "
                        "failed its non-negativity/reality audit; "
                        f"raw squared residual={raw_residual_squared:.3e}, "
                        "negative allowance="
                        f"{effective_residual_squared_limit:.3e}, RHS Im="
                        f"{rhs_diagnostics['rhs_norm_imaginary_residual']:.3e}"
                    )
                full_residual_available = True
                full_residual_reason = None
                fit_gate_kind = "full_source_residual"
            else:
                residual_roundoff_tolerance = None
                fit_gate_passed = False
                full_residual_available = False
                full_residual_reason = (
                    "residual_mode='radial_only' disables the exact "
                    "B^dagger B right-hand-side norm"
                )
                fit_gate_kind = "full_source_residual_unavailable"
            reported = complex(reported_norm)
            if not np.isfinite(reported):
                raise RuntimeError("Block2 multiply returned a non-finite norm")
            # The installed Block2 contract explicitly returns ||x||.
            # Compare that complex scalar to sqrt(<x|x>); do not hide an
            # imaginary component or accept the squared norm by ambiguity.
            reported_norm_error = abs(reported - math.sqrt(norm))
            reported_norm_tolerance = max(
                float(self.source_tol),
                100.0 * np.finfo(float).eps,
            ) * max(1.0, abs(reported), math.sqrt(norm))
            reported_norm_gate_passed = bool(
                reported_norm_error <= reported_norm_tolerance
            )
            if not reported_norm_gate_passed:
                _utils._warn_numerical(
                    f"source {source.key}{source.free_indices} Block2 "
                    f"reported-norm mismatch {reported_norm_error:.3e} "
                    f"exceeds {reported_norm_tolerance:.3e}"
                )
            diagnostics = {
                "tag": tag,
                "particle_number": target_n,
                "particle_number_change": source_particle_change,
                "source_term_count": len(source.terms),
                "constant": [source.constant.real, source.constant.imag],
                "discarded_coefficient_l1": source.discarded_coefficient_l1,
                "reported_multiply_norm": [reported.real, reported.imag],
                "mps_norm_squared": norm,
                "mps_norm_expectation": [
                    float(complex(norm_value).real),
                    float(complex(norm_value).imag),
                ],
                "mps_norm_imaginary_residual": norm_imaginary,
                "mps_norm_imaginary_tolerance": norm_imaginary_limit,
                "mps_norm_imaginary_gate_passed": bool(
                    norm_imaginary <= norm_imaginary_limit
                ),
                "fit_overlap": [cross.real, cross.imag],
                "radial_stationarity_residual": float(radial_stationarity_residual),
                "radial_stationarity_relative_residual": float(
                    radial_stationarity_residual / max(norm, self.norm_tol)
                ),
                "radial_stationarity_tolerance": float(radial_stationarity_tolerance),
                "radial_stationarity_gate_passed": (radial_stationarity_gate_passed),
                "exact_rhs_norm_squared": rhs_norm,
                "full_fit_residual_available": full_residual_available,
                "full_fit_residual_squared_raw": raw_residual_squared,
                "full_fit_residual_squared": full_residual_squared,
                "full_fit_residual_norm": full_residual,
                "full_fit_relative_residual": full_relative_residual,
                "full_fit_residual_tolerance": full_residual_tolerance,
                "full_fit_residual_effective_tolerance": (
                    full_residual_effective_tolerance
                ),
                "full_fit_residual_roundoff_tolerance_squared": (
                    residual_roundoff_tolerance
                ),
                "source_operator_coefficient_l1": source_operator_l1,
                "source_norm_triangle_upper_bound": source_norm_upper_bound,
                "full_fit_residual_nonnegative_gate_passed": (
                    residual_nonnegative_gate
                ),
                "full_fit_residual_reason": full_residual_reason,
                "fit_gate_kind": fit_gate_kind,
                "fit_gate_passed": fit_gate_passed,
                "reported_norm_consistency_error": float(reported_norm_error),
                "reported_norm_consistency_tolerance": float(reported_norm_tolerance),
                "reported_norm_consistency_gate_passed": (reported_norm_gate_passed),
                "bond_dimension": int(source_mps.info.bond_dim),
                "zero_by_particle_sector": False,
                "zero_by_coefficients": False,
                "wall_seconds": time.perf_counter() - start,
            }
            diagnostics.update(rhs_diagnostics)
            fitted_dot = int(source_mps.dot)
            propagation_dot = 2 if int(source_mps.n_sites) > 1 else 1
            if fitted_dot != propagation_dot:
                source_mps, _forward = driver.adjust_mps(
                    source_mps,
                    dot=propagation_dot,
                )
            diagnostics["fitted_dot"] = fitted_dot
            diagnostics["propagation_dot"] = int(source_mps.dot)
            return source_mps, diagnostics
        except Exception:
            if source_mps is not None:
                _release_owned_mps(
                    driver,
                    source_mps,
                    tag,
                    remove_files=bool(self.cleanup_mps),
                )
            raise
        finally:
            _deallocate_mpo(source_mpo)

    def _build_dense_csf_overlap_context(
        self,
        prepared,
        basis_mps,
        initial_gram,
        *,
        tag_factory,
        key,
    ):
        """Build an exact determinant representation of a small MPS basis."""

        if self.factorized_overlap_backend != "dense_csf":
            return None
        driver = prepared.driver
        first_state = next((state for state in basis_mps if state is not None), None)
        if first_state is None:
            raise RuntimeError(
                f"factorized class {key} has no MPS for dense-CSF overlap"
            )
        n_sites = int(first_state.n_sites)
        target_n = int(first_state.info.target.n)
        if not 0 <= target_n <= n_sites:
            raise RuntimeError(
                f"factorized class {key} has invalid SGF particle sector "
                f"N={target_n} for {n_sites} sites"
            )
        determinant_count = math.comb(n_sites, target_n)
        if determinant_count > int(self.dense_csf_max_dimension):
            raise RuntimeError(
                f"factorized class {key} dense-CSF sector dimension "
                f"{determinant_count} exceeds dense_csf_max_dimension="
                f"{int(self.dense_csf_max_dimension)}"
            )
        # In SGF every spinor site is either empty (0) or occupied (1).
        # Supplying the fixed-N list is essential: Block2's cutoff=0 search
        # otherwise returns all 2**n_sites bit strings, including exact-zero
        # coefficients outside the MPS target sector.
        determinants = np.zeros(
            (determinant_count, n_sites),
            dtype=np.uint8,
        )
        for row, occupied in enumerate(
            itertools.combinations(range(n_sites), target_n)
        ):
            determinants[row, list(occupied)] = 1
        columns = [None] * len(basis_mps)
        extraction_count = 0
        for basis_index, state in enumerate(basis_mps):
            if state is None:
                continue
            if (
                int(state.n_sites) != n_sites
                or int(state.info.target.n) != target_n
            ):
                raise RuntimeError(
                    f"factorized class {key} mixes dense-CSF particle sectors"
                )
            temporary_tag = tag_factory(
                key,
                (basis_index, -1),
                "CSFBASIS",
            )
            _state_determinants, coefficients = _mps_determinant_coefficients(
                driver,
                state,
                given_determinants=determinants,
                temporary_tag=temporary_tag,
                remove_files=bool(self.cleanup_mps),
            )
            extraction_count += 1
            columns[basis_index] = coefficients
        coefficient_matrix = np.zeros(
            (len(determinants), len(basis_mps)),
            dtype=np.complex128,
        )
        for basis_index, coefficients in enumerate(columns):
            if coefficients is not None:
                coefficient_matrix[:, basis_index] = coefficients
        determinant_gram = coefficient_matrix.conj().T @ coefficient_matrix
        initial_gram = np.asarray(initial_gram, dtype=np.complex128)
        errors = np.abs(determinant_gram - initial_gram)
        limits = float(self.dense_csf_overlap_atol) + float(
            self.dense_csf_overlap_rtol
        ) * np.maximum(1.0, np.maximum(np.abs(determinant_gram), np.abs(initial_gram)))
        maximum_error = float(np.max(errors, initial=0.0))
        maximum_limit = float(np.max(limits, initial=0.0))
        gate = bool(np.all(errors <= limits))
        if not gate:
            maximum_ratio = float(
                np.max(
                    np.divide(
                        errors,
                        limits,
                        out=np.full_like(errors, np.inf, dtype=float),
                        where=limits > 0.0,
                    ),
                    initial=0.0,
                )
            )
            raise RuntimeError(
                f"factorized class {key} dense-CSF determinant Gram does not "
                "match the independently measured MPS Gram: maximum error "
                f"{maximum_error:.3e}, maximum error/tolerance ratio "
                f"{maximum_ratio:.3e}"
            )
        return {
            "determinants": determinants,
            "coefficient_matrix": coefficient_matrix,
            "determinant_count": int(len(determinants)),
            "coefficient_extraction_count": int(extraction_count),
            "initial_gram_maximum_abs_error": maximum_error,
            "initial_gram_maximum_tolerance": maximum_limit,
            "initial_gram_gate_passed": gate,
            "native_crosscheck_count": 0,
            "native_crosscheck_maximum_abs_error": 0.0,
            "native_crosscheck_maximum_tolerance": 0.0,
            "native_crosscheck_gate_passed": True,
        }

    def _dense_csf_overlap_vector(
        self,
        prepared,
        evolved,
        basis_mps,
        identity,
        context,
        *,
        tag_factory,
        key,
        ket_index,
        step_index,
    ):
        """Measure one full overlap vector and audit its physical phase."""

        temporary_tag = tag_factory(
            key,
            (ket_index, step_index),
            "CSFSTEP",
        )
        _determinants, coefficients = _mps_determinant_coefficients(
            prepared.driver,
            evolved,
            given_determinants=context["determinants"],
            temporary_tag=temporary_tag,
            remove_files=bool(self.cleanup_mps),
        )
        context["coefficient_extraction_count"] += 1
        overlaps = context["coefficient_matrix"].conj().T @ coefficients
        required = np.asarray([bra is not None for bra in basis_mps], dtype=bool)
        overlaps = np.where(required, overlaps, 0.0)
        check_indices = np.flatnonzero(required)
        if not len(check_indices):
            raise RuntimeError(
                f"factorized class {key} dense-CSF overlap has no native check bra"
            )
        check_index = int(check_indices[0])
        native = _utils._complex_scalar(
            _identity_overlap(
                prepared.driver,
                basis_mps[check_index],
                identity,
                evolved,
            ),
            name=(
                f"factorized {key} dense-CSF native overlap "
                f"({check_index},{ket_index},{step_index})"
            ),
        )
        error = float(abs(overlaps[check_index] - native))
        tolerance = float(
            self.dense_csf_overlap_atol
            + self.dense_csf_overlap_rtol
            * max(1.0, abs(overlaps[check_index]), abs(native))
        )
        gate = bool(np.isfinite(error) and error <= tolerance)
        context["native_crosscheck_count"] += 1
        context["native_crosscheck_maximum_abs_error"] = max(
            context["native_crosscheck_maximum_abs_error"],
            error,
        )
        context["native_crosscheck_maximum_tolerance"] = max(
            context["native_crosscheck_maximum_tolerance"],
            tolerance,
        )
        context["native_crosscheck_gate_passed"] = bool(
            context["native_crosscheck_gate_passed"] and gate
        )
        if not gate:
            raise RuntimeError(
                f"factorized class {key} dense-CSF overlap disagrees with "
                f"native MPS contraction at ket {ket_index}, time point "
                f"{step_index}: error={error:.3e}, tolerance={tolerance:.3e}"
            )
        return overlaps

    def _propagate_overlap_grid(
        self,
        prepared,
        initial_ket,
        basis_mps,
        identity,
        grid,
        initial_overlaps,
        *,
        time_bonds,
        tag_factory,
        key,
        ket_index,
        active_shift=None,
        stop_callback=None,
        dense_overlap_context=None,
    ):
        """Continuously propagate one source and sample all bra overlaps."""

        driver = prepared.driver
        bw = driver.bw
        samples = np.zeros(
            (len(basis_mps), len(grid)),
            dtype=np.complex128,
        )
        samples[:, 0] = np.asarray(initial_overlaps, dtype=np.complex128)
        overlap_count = 0
        maximum_discarded_weight = 0.0
        discarded_weights = []
        evolution_tag = tag_factory(key, (ket_index,), "GFSTREAM")
        evolved = None
        environment = None
        evolution = None
        completed_points = 1
        if active_shift is None:
            active_shift = -prepared.active_reference_energy
        active_shift = float(active_shift)
        if not np.isfinite(active_shift):
            raise ValueError("active propagation shift must be finite")
        try:
            evolved = driver.copy_mps(initial_ket, tag=evolution_tag)
            with _shifted_mpo_constant(
                prepared.active_mpo,
                active_shift,
            ):
                environment = bw.bs.MovingEnvironment(
                    prepared.active_mpo,
                    evolved,
                    evolved,
                    f"TDDMRG-{ket_index}",
                )
                environment.delayed_contraction = bw.b.OpNamesSet.normal_ops()
                environment.cached_contraction = True
                environment.init_environments(self.verbose >= logger.DEBUG)
                evolution = bw.bs.TimeEvolution(
                    environment,
                    bw.b.VectorUBond(time_bonds),
                    (
                        bw.b.TETypes.RK4
                        if self.te_type == "rk4"
                        else bw.b.TETypes.TangentSpace
                    ),
                )
                evolution.hermitian = self.te_type == "tdvp"
                evolution.iprint = 1 if self.verbose >= logger.DEBUG else 0
                evolution.n_sub_sweeps = (
                    int(self.n_sub_sweeps) if self.te_type == "rk4" else 1
                )
                evolution.normalize_mps = False
                evolution.cutoff = float(self.time_cutoff)
                evolution.krylov_conv_thrd = float(self.krylov_conv_thrd)
                evolution.krylov_subspace_size = int(self.krylov_subspace_size)

                for step_index, target_time in enumerate(grid[1:], start=1):
                    delta_t = float(target_time - grid[step_index - 1])
                    if delta_t <= 0.0:
                        raise ValueError("imaginary-time increments must be positive")
                    # ``solve`` restarts its local sweep index on every call.
                    # Since one call advances one sampled time interval, set
                    # the requested per-step bond dimension explicitly rather
                    # than leaving the full vector on ``TimeEvolution`` (which
                    # would use its first entry at every interval).
                    step_bond = int(
                        time_bonds[min(step_index - 1, len(time_bonds) - 1)]
                    )
                    evolution.bond_dims = bw.b.VectorUBond([step_bond])
                    if self.te_type == "tdvp":
                        evolution.solve(2, delta_t / 2.0, evolved.center == 0)
                    else:
                        evolution.solve(1, delta_t, evolved.center == 0)
                    if evolution.discarded_weights:
                        discarded_weight = float(evolution.discarded_weights[-1])
                        discarded_weights.append(discarded_weight)
                        maximum_discarded_weight = max(
                            maximum_discarded_weight,
                            discarded_weight,
                        )
                    else:
                        discarded_weights.append(0.0)
                    if dense_overlap_context is None:
                        for bra_index, initial_bra in enumerate(basis_mps):
                            if initial_bra is not None:
                                samples[bra_index, step_index] = _identity_overlap(
                                    driver,
                                    initial_bra,
                                    identity,
                                    evolved,
                                )
                                overlap_count += 1
                    else:
                        dense_overlaps = self._dense_csf_overlap_vector(
                            prepared,
                            evolved,
                            basis_mps,
                            identity,
                            dense_overlap_context,
                            tag_factory=tag_factory,
                            key=key,
                            ket_index=ket_index,
                            step_index=step_index,
                        )
                        required = np.asarray(
                            [bra is not None for bra in basis_mps],
                            dtype=bool,
                        )
                        samples[:, step_index] = dense_overlaps
                        overlap_count += int(np.count_nonzero(required))
                    completed_points = step_index + 1
                    if stop_callback is not None and stop_callback(
                        grid[:completed_points],
                        samples[:, :completed_points],
                    ):
                        break
            return (
                samples[:, :completed_points],
                maximum_discarded_weight,
                overlap_count,
                np.asarray(
                    discarded_weights[: max(0, completed_points - 1)],
                    dtype=float,
                ),
            )
        finally:
            if environment is not None and driver.clean_scratch:
                environment.remove_partition_files()
            del evolution
            del environment
            if evolved is not None:
                _release_owned_mps(
                    driver,
                    evolved,
                    evolution_tag,
                    remove_files=bool(self.cleanup_mps),
                )
            elif self.cleanup_mps:
                _remove_owned_mps_files(driver, evolution_tag)

    def _audit_sc_source_moments(self, prepared, mc=None) -> dict[str, Any]:
        """Compare MPS source C(0)/active moments with the Wick SC oracle.

        This explicitly opt-in path is the sole place where t-MPS-NEVPT2 may
        request active RDMs above rank one.  The RDMs are used only to evaluate
        the already generated SC norm/commutator arrays; no SC energy enters
        the t-MPS result.
        """

        from . import x2cscnevpt2 as sc

        if mc is None:
            mc = self._mc
        start = time.perf_counter()
        pdms = sc.make_dm1234(mc.fcisolver, root=prepared.root)
        _validated, rdm_diagnostics = sc.validate_pdms(
            pdms,
            prepared.eris.ncas,
            _utils._total_nelec(
                getattr(
                    mc.fcisolver,
                    "nelecas",
                    getattr(mc, "nelecas", None),
                )
            ),
            atol=float(self.rdm_atol),
            rtol=float(self.rdm_rtol),
        )
        _sub_eners, _sub_norms, _sub_gaps, arrays = sc._evaluate_wick_subspaces(
            prepared.eris,
            pdms,
            prepared.core_energy,
            prepared.virtual_energy,
            root=prepared.root,
            scalar_atol=float(self.sc_audit_atol),
            scalar_rtol=float(self.sc_audit_rtol),
            denominator_mode="hermitianized",
            return_arrays=True,
        )

        driver = prepared.driver
        tag_factory = _TagFactory(prepared.root, "sc-audit")
        entries = {}
        maximum_norm_error = 0.0
        maximum_moment_error = 0.0
        maximum_norm_relative_error = 0.0
        maximum_moment_relative_error = 0.0
        failed_labels = []
        for key in SUBSPACE_ORDER:
            for free_indices in _iter_source_blocks(key, prepared.eris):
                label = f"{key}:" + ",".join(map(str, free_indices))
                source = _active_source(
                    key,
                    free_indices,
                    prepared.eris,
                    coefficient_cutoff=float(self.source_coefficient_cutoff),
                )
                source_mps = None
                source_tag = None
                if source.is_scalar:
                    actual_norm = abs(source.constant) ** 2
                    actual_moment = 0.0j
                    source_fit = {
                        "analytic_scalar_source": True,
                        "particle_number": int(prepared.reference_mps.info.target.n),
                    }
                else:
                    source_mps, source_fit = self._build_source_mps(
                        prepared,
                        source,
                        tag_factory=tag_factory,
                    )
                    if source_mps is None:
                        actual_norm = 0.0
                        actual_moment = 0.0j
                    else:
                        source_tag = source_fit["tag"]
                        try:
                            raw_norm = float(source_fit["mps_norm_squared"])
                            actual_norm = raw_norm / prepared.reference_norm
                            active_value = driver.expectation(
                                source_mps,
                                prepared.active_mpo,
                                source_mps,
                            )
                            actual_moment = (
                                complex(active_value)
                                - prepared.active_reference_energy * raw_norm
                            ) / prepared.reference_norm
                        finally:
                            _release_owned_mps(
                                driver,
                                source_mps,
                                source_tag,
                                remove_files=bool(self.cleanup_mps),
                            )
                            source_mps = None
                            source_tag = None
                expected_norm = complex(arrays[key]["norm"][free_indices])
                expected_moment = complex(arrays[key]["commutator"][free_indices])
                norm_error = abs(actual_norm - expected_norm)
                moment_error = abs(actual_moment - expected_moment)
                norm_scale = max(1.0, abs(actual_norm), abs(expected_norm))
                moment_scale = max(
                    1.0,
                    abs(actual_moment),
                    abs(expected_moment),
                )
                norm_relative_error = norm_error / norm_scale
                moment_relative_error = moment_error / moment_scale
                norm_limit = self.sc_audit_atol + self.sc_audit_rtol * norm_scale
                moment_limit = self.sc_audit_atol + self.sc_audit_rtol * moment_scale
                passed = bool(norm_error <= norm_limit and moment_error <= moment_limit)
                if not passed:
                    failed_labels.append(label)
                entries[label] = {
                    "source": source_fit,
                    "mps_norm": [
                        float(np.real(actual_norm)),
                        float(np.imag(actual_norm)),
                    ],
                    "sc_norm": [expected_norm.real, expected_norm.imag],
                    "norm_error": float(norm_error),
                    "norm_tolerance": float(norm_limit),
                    "mps_active_first_moment": [
                        actual_moment.real,
                        actual_moment.imag,
                    ],
                    "sc_active_commutator": [
                        expected_moment.real,
                        expected_moment.imag,
                    ],
                    "active_first_moment_error": float(moment_error),
                    "active_first_moment_tolerance": float(moment_limit),
                    "passed": passed,
                }
                maximum_norm_error = max(maximum_norm_error, norm_error)
                maximum_moment_error = max(
                    maximum_moment_error,
                    moment_error,
                )
                maximum_norm_relative_error = max(
                    maximum_norm_relative_error,
                    norm_relative_error,
                )
                maximum_moment_relative_error = max(
                    maximum_moment_relative_error,
                    moment_relative_error,
                )
                if source_mps is not None and source_tag is not None:
                    _release_owned_mps(
                        driver,
                        source_mps,
                        source_tag,
                        remove_files=bool(self.cleanup_mps),
                    )

        passed = not failed_labels
        if not passed:
            _utils._warn_numerical(
                "optional SC source-moment audit failed for "
                + ", ".join(failed_labels[:8])
                + (" ..." if len(failed_labels) > 8 else "")
            )
        return {
            "enabled": True,
            "rdm_ranks_used_only_for_audit": [1, 2, 3, 4],
            "rdm_diagnostics": rdm_diagnostics,
            "entries": entries,
            "maximum_norm_error": float(maximum_norm_error),
            "maximum_norm_relative_error": float(maximum_norm_relative_error),
            "maximum_active_first_moment_error": float(maximum_moment_error),
            "maximum_active_first_moment_relative_error": float(
                maximum_moment_relative_error
            ),
            "failed_labels": failed_labels,
            "passed": passed,
            "wall_seconds": time.perf_counter() - start,
        }

    def _default_time_grid(self, *, time_step: float) -> np.ndarray:
        steps = int(math.floor(self.max_time / time_step + 1.0e-12))
        if self.max_steps is not None:
            steps = min(steps, int(self.max_steps))
        steps = max(1, steps)
        return np.arange(steps + 1, dtype=float) * time_step

    def _propagate_direct_block(
        self,
        prepared,
        source,
        *,
        gap: float,
        time_grid,
        time_step: float,
        tag_factory,
    ):
        start = time.perf_counter()
        if source.discarded_coefficient_l1 != 0.0:
            raise RuntimeError(
                f"source {source.key}{source.free_indices} contains discarded "
                "coefficients; refusing a magnitude-pruned direct source"
            )
        if gap <= self.denominator_tol:
            raise RuntimeError(
                f"source {source.key}{source.free_indices} has non-positive "
                f"orbital gap {gap:.6e}"
            )

        # A pure scalar active source is proportional to the normalized
        # reference, hence H_act-E_act annihilates it and its resolvent is
        # analytic.  This covers both exact-zero blocks and, for example, an
        # ``ir`` block whose active particle-hole coefficients vanish while
        # h[ri] does not.  Passing either case as an empty fermionic MPO is
        # unsupported by Block2 and would be unnecessary work in any case.
        if source.is_scalar:
            coefficient = complex(source.constant)
            source_norm = float(abs(coefficient) ** 2)
            zero_source = coefficient == 0.0
            target_n = int(prepared.reference_mps.info.target.n) + source.delta_n_active
            source_diagnostics = {
                "analytic": True,
                "analytic_scalar_source": True,
                "coefficient": [coefficient.real, coefficient.imag],
                "source_norm": source_norm,
                "mps_norm_squared": source_norm,
                "particle_number": target_n,
                "particle_number_change": source.delta_n_active,
                "source_term_count": 0,
                "discarded_coefficient_l1": source.discarded_coefficient_l1,
                "zero_by_particle_sector": not (0 <= target_n <= prepared.eris.ncas),
                "zero_by_coefficients": zero_source,
                **self._exact_source_fit_diagnostics(
                    kind="analytic_scalar_source",
                    rhs_norm_squared=source_norm,
                ),
                "wall_seconds": time.perf_counter() - start,
            }
            propagation = {
                "analytic": True,
                "zero_source": zero_source,
                "time_points": 0,
                "wall_seconds": time.perf_counter() - start,
            }
            integration = {
                "analytic": True,
                "zero_source": zero_source,
                "tail": 0.0,
                "tail_gate_passed": True,
            }
            return (
                -source_norm / gap,
                source_diagnostics,
                propagation,
                integration,
                None,
            )

        driver = prepared.driver
        source_mps, source_diagnostics = self._build_source_mps(
            prepared,
            source,
            tag_factory=tag_factory,
        )
        if source_mps is None:
            exact_zero_certified = bool(
                source_diagnostics.get("fit_gate_kind")
                in {"exact_particle_sector_zero", "exact_structural_zero"}
                and source_diagnostics.get("fit_gate_passed") is True
            )
            return (
                0.0,
                source_diagnostics,
                {
                    "zero_source": True,
                    "exact_zero_source_certified": exact_zero_certified,
                    "wall_seconds": time.perf_counter() - start,
                },
                {
                    "tail": 0.0,
                    "zero_source": True,
                    "exact_zero_source_certified": exact_zero_certified,
                },
                None,
            )
        source_tag = source_diagnostics["tag"]
        identity = driver.get_identity_mpo()
        try:
            norm = float(source_diagnostics["mps_norm_squared"])
            if norm == 0.0:
                # A numerically zero fitted MPS (or an RHS norm clipped within
                # floating-point tolerance) is not an algebraic zero proof.
                # Only the structural/particle-sector branches above may set
                # an exact-zero certificate and bypass propagation gates.
                exact_zero_certified = False
                _utils._warn_numerical(
                    f"source {source.key}{source.free_indices} fitted to a "
                    "zero MPS without a structural exact-zero certificate"
                )
                return (
                    0.0,
                    source_diagnostics,
                    {
                        "zero_source": True,
                        "zero_by_mps_action": True,
                        "exact_zero_source_certified": exact_zero_certified,
                        "wall_seconds": time.perf_counter() - start,
                    },
                    {
                        "tail": 0.0,
                        "zero_source": True,
                        "exact_zero_source_certified": exact_zero_certified,
                    },
                    None,
                )
            active_expectation = driver.expectation(
                source_mps,
                prepared.active_mpo,
                source_mps,
            )
            active_expectation = _utils._complex_scalar(
                active_expectation,
                name=f"{source.key}{source.free_indices} active first moment",
            )
            first_moment = (
                active_expectation - (prepared.active_reference_energy - gap) * norm
            )
            explicit_grid = time_grid is not None
            grid = (
                _validate_time_grid(time_grid)
                if explicit_grid
                else self._default_time_grid(time_step=time_step)
            )
            early_stop = False
            tail_diagnostics = None
            shift = gap - prepared.active_reference_energy
            time_bonds = self._bond_schedule(
                self.time_bond_dims,
                default=int(source_mps.info.bond_dim),
                name="time_bond_dims",
            )

            def stop_callback(sampled_grid, sampled_matrix):
                nonlocal early_stop, tail_diagnostics
                if explicit_grid or len(sampled_grid) < self.min_steps + 1:
                    return False
                correlations = sampled_matrix[0]
                tail_diagnostics = _estimate_exponential_tail(
                    sampled_grid,
                    correlations,
                    window=int(self.tail_fit_window),
                    imaginary_tolerance=float(self.energy_imag_tol),
                    negligible_tolerance=float(self.integrand_abs_tol),
                    log_residual_tolerance=float(self.tail_fit_log_residual_tol),
                    rate_sensitivity_tolerance=float(
                        self.tail_fit_rate_sensitivity_tol
                    ),
                )
                partial = _integrate_samples(
                    sampled_grid,
                    correlations,
                    mode=self.integration_mode,
                )
                tail_limit = self.tail_abs_tol + self.tail_rel_tol * abs(partial)
                early_stop = bool(
                    abs(correlations[-1]) < self.integrand_abs_tol
                    and tail_diagnostics["accepted"]
                    and tail_diagnostics["tail"] < tail_limit
                )
                return early_stop

            (
                sampled_matrix,
                maximum_discarded_weight,
                _sampled_overlap_count,
                discarded_weights,
            ) = self._propagate_overlap_grid(
                prepared,
                source_mps,
                [source_mps],
                identity,
                grid,
                [complex(norm)],
                time_bonds=time_bonds,
                tag_factory=tag_factory,
                key=source.key,
                ket_index=0,
                active_shift=shift,
                stop_callback=stop_callback,
            )
            times = np.asarray(grid[: sampled_matrix.shape[1]], dtype=float)
            values = np.asarray(sampled_matrix[0], dtype=np.complex128)
            maximum_imaginary = float(np.max(np.abs(values.imag), initial=0.0))
            correlation_scale = max(
                1.0,
                float(np.max(np.abs(values.real), initial=0.0)),
            )
            maximum_correlation_real_magnitude = float(
                np.max(np.abs(values.real), initial=0.0)
            )
            correlation_imaginary_tolerance = float(
                self.energy_imag_tol * correlation_scale
            )
            correlation_imaginary_gate = bool(
                maximum_imaginary <= correlation_imaginary_tolerance
            )
            if not correlation_imaginary_gate:
                _utils._warn_numerical(
                    f"source {source.key}{source.free_indices} correlation has "
                    f"Im residual {maximum_imaginary:.3e}"
                )
            positivity_violation = float(
                max(0.0, -float(np.min(values.real, initial=0.0)))
            )
            monotonicity_violation = float(
                max(0.0, float(np.max(np.diff(values.real), initial=0.0)))
            )
            if max(positivity_violation, monotonicity_violation) > (
                self.correlation_residual_tol * correlation_scale
            ):
                _utils._warn_numerical(
                    f"source {source.key}{source.free_indices} violates positive "
                    "monotone imaginary-time decay"
                )
            if tail_diagnostics is None:
                tail_diagnostics = _estimate_exponential_tail(
                    times,
                    values,
                    window=int(self.tail_fit_window),
                    imaginary_tolerance=float(self.energy_imag_tol),
                    negligible_tolerance=float(self.integrand_abs_tol),
                    log_residual_tolerance=float(self.tail_fit_log_residual_tol),
                    rate_sensitivity_tolerance=float(
                        self.tail_fit_rate_sensitivity_tol
                    ),
                )
            integral = _integrate_samples(
                times,
                values,
                mode=self.integration_mode,
            )
            tail_limit = self.tail_abs_tol + self.tail_rel_tol * abs(integral)
            tail_accepted = bool(
                tail_diagnostics["accepted"] and tail_diagnostics["tail"] < tail_limit
            )
            if tail_diagnostics["accepted"]:
                tail = float(tail_diagnostics["tail"])
                if not tail_accepted:
                    _utils._warn_numerical(
                        f"source {source.key}{source.free_indices} reached "
                        f"tau={times[-1]:.6g} with tail {tail:.3e} above "
                        f"the requested {tail_limit:.3e}"
                    )
            else:
                tail = 0.0
                _utils._warn_numerical(
                    f"source {source.key}{source.free_indices} has no stable "
                    f"tail estimate ({tail_diagnostics['reason']}); the reported "
                    "energy is truncated at max_time"
                )
            # The retained reference MPS is expected to be normalized, but
            # divide by its measured norm so scalar and operator sources use
            # one scale-invariant wavefunction convention.
            energy_value = -(integral + tail) / prepared.reference_norm
            energy, energy_imaginary = _checked_real_scalar(
                energy_value,
                name=f"{source.key}{source.free_indices} t-MPS energy",
                imaginary_tolerance=self.energy_imag_tol,
            )
            energy_imaginary_limit = float(self.energy_imag_tol * max(1.0, abs(energy)))
            energy_imaginary_scale = float(max(1.0, abs(energy)))
            if energy > self.energy_imag_tol:
                _utils._warn_numerical(
                    f"source {source.key}{source.free_indices} has positive "
                    f"second-order energy {energy:.3e}"
                )
            finite_difference_moment = None
            if len(times) > 1:
                finite_difference_moment = complex(
                    -(values[1] - values[0]) / (times[1] - times[0])
                )
            propagation = {
                "zero_source": False,
                "time_points": len(times),
                "final_time": float(times[-1]),
                "early_stop": early_stop,
                "te_type": self.te_type,
                "normalize_mps": False,
                "maximum_discarded_weight": float(max(discarded_weights, default=0.0)),
                "maximum_correlation_imaginary_residual": maximum_imaginary,
                "maximum_correlation_real_magnitude": (
                    maximum_correlation_real_magnitude
                ),
                "correlation_imaginary_scale": correlation_scale,
                "correlation_imaginary_tolerance": (correlation_imaginary_tolerance),
                "correlation_imaginary_gate_passed": correlation_imaginary_gate,
                "positivity_violation": positivity_violation,
                "monotonicity_violation": monotonicity_violation,
                "source_norm": norm,
                "active_plus_orbital_first_moment": [
                    first_moment.real,
                    first_moment.imag,
                ],
                "finite_difference_first_moment": (
                    None
                    if finite_difference_moment is None
                    else [
                        finite_difference_moment.real,
                        finite_difference_moment.imag,
                    ]
                ),
                "wall_seconds": time.perf_counter() - start,
            }
            integration = dict(tail_diagnostics)
            integration.update(
                {
                    "quadrature": [integral.real, integral.imag],
                    "tail_used": tail,
                    "tail_limit": float(tail_limit),
                    "tail_gate_passed": tail_accepted,
                    "reference_normalization_divisor": (prepared.reference_norm),
                    "energy_imaginary_residual": energy_imaginary,
                    "energy_real_magnitude": abs(energy),
                    "energy_imaginary_scale": energy_imaginary_scale,
                    "energy_imaginary_tolerance": energy_imaginary_limit,
                    "energy_imaginary_gate_passed": bool(
                        energy_imaginary <= energy_imaginary_limit
                    ),
                    "maximum_correlation_imaginary_residual": maximum_imaginary,
                    "maximum_correlation_real_magnitude": (
                        maximum_correlation_real_magnitude
                    ),
                    "correlation_imaginary_scale": correlation_scale,
                    "correlation_imaginary_tolerance": (
                        correlation_imaginary_tolerance
                    ),
                    "correlation_imaginary_gate_passed": (correlation_imaginary_gate),
                    "mode": self.integration_mode,
                }
            )
            series = None
            if self.store_time_series:
                series = {
                    "time": times,
                    "correlation": values,
                    "discarded_weights": np.asarray(
                        discarded_weights,
                        dtype=float,
                    ),
                }
            return energy, source_diagnostics, propagation, integration, series
        finally:
            _deallocate_mpo(identity)
            _release_owned_mps(
                driver,
                source_mps,
                source_tag,
                remove_files=bool(self.cleanup_mps),
            )
            gc.collect()

    @staticmethod
    def _block_label(key: str, free_indices: Sequence[int]) -> str:
        return key + ":" + ",".join(str(int(value)) for value in free_indices)

    def _run_direct_pass(
        self,
        prepared,
        *,
        time_grid,
        time_step: float,
        pass_name: str,
    ) -> _DirectPassResult:
        prepared.driver.bw.b.Random.rand_seed(int(self.source_random_seed))
        tag_factory = _TagFactory(prepared.root, "direct")
        sub_eners = {}
        sub_times = {}
        sub_diagnostics = {}
        source_diagnostics = {}
        propagation_diagnostics = {}
        integration_diagnostics = {}
        source_counts = {}
        time_series = {}
        scheduled_grid = (
            _validate_time_grid(time_grid)
            if time_grid is not None
            else self._default_time_grid(time_step=time_step)
        )
        for key in SUBSPACE_ORDER:
            class_start = time.perf_counter()
            class_energy = 0.0
            block_energies = {}
            count = 0
            for free_indices in _iter_source_blocks(key, prepared.eris):
                source = _active_source(
                    key,
                    free_indices,
                    prepared.eris,
                    coefficient_cutoff=self.source_coefficient_cutoff,
                )
                gap = _orbital_gap_for_block(
                    key,
                    free_indices,
                    prepared.core_energy,
                    prepared.virtual_energy,
                )
                label = self._block_label(key, free_indices)
                if gap <= self.denominator_tol:
                    raise RuntimeError(
                        f"source {label} has non-positive orbital gap {gap:.6e}"
                    )
                if key == "ijrs":
                    energy = -abs(source.constant) ** 2 / gap
                    source_norm = float(abs(source.constant) ** 2)
                    zero_source = source.constant == 0.0
                    source_diag = {
                        "analytic": True,
                        "analytic_scalar_source": True,
                        "coefficient": [
                            source.constant.real,
                            source.constant.imag,
                        ],
                        "source_norm": source_norm,
                        "mps_norm_squared": source_norm,
                        "source_term_count": 0,
                        "discarded_coefficient_l1": 0.0,
                        "particle_number_change": 0,
                        "zero_by_particle_sector": False,
                        "zero_by_coefficients": zero_source,
                        **self._exact_source_fit_diagnostics(
                            kind="analytic_scalar_source",
                            rhs_norm_squared=source_norm,
                        ),
                    }
                    propagation_diag = {
                        "analytic": True,
                        "zero_source": zero_source,
                        "time_points": 0,
                    }
                    integration_diag = {
                        "analytic": True,
                        "zero_source": zero_source,
                        "tail": 0.0,
                    }
                    series = None
                else:
                    (
                        energy,
                        source_diag,
                        propagation_diag,
                        integration_diag,
                        series,
                    ) = self._propagate_direct_block(
                        prepared,
                        source,
                        gap=gap,
                        time_grid=time_grid,
                        time_step=time_step,
                        tag_factory=tag_factory,
                    )
                    count += int(
                        not propagation_diag.get("analytic", False)
                        and not propagation_diag.get("zero_source", False)
                    )
                source_diag["operator_constant"] = [
                    source.constant.real,
                    source.constant.imag,
                ]
                source_diag["operator_terms"] = [
                    {
                        "operators": list(term.operators),
                        "coefficient": [
                            term.coefficient.real,
                            term.coefficient.imag,
                        ],
                    }
                    for term in source.terms
                ]
                energy = float(energy)
                integration_diag.setdefault("external_energy", 0.0)
                integration_diag["signed_energy_contribution"] = energy
                sampled_time_points = int(propagation_diag.get("time_points", 0))
                sampled_final_time = float(
                    propagation_diag.get("final_time", 0.0)
                )
                integration_diag.update(
                    {
                        "pass_name": str(pass_name),
                        "requested_time_step": float(time_step),
                        "explicit_time_grid": bool(time_grid is not None),
                        "scheduled_time_points": int(len(scheduled_grid)),
                        "scheduled_final_time": float(scheduled_grid[-1]),
                        "sampled_time_points": sampled_time_points,
                        "sampled_final_time": sampled_final_time,
                    }
                )
                class_energy += energy
                block_energies[label] = energy
                source_diag["orbital_gap"] = gap
                source_diagnostics[label] = source_diag
                propagation_diagnostics[label] = propagation_diag
                integration_diagnostics[label] = integration_diag
                if series is not None:
                    time_series[label] = series
            sub_eners[key] = float(class_energy)
            sub_times[key] = time.perf_counter() - class_start
            sub_diagnostics[key] = {
                "block_energies": block_energies,
                "block_count": len(block_energies),
                "propagated_source_count": count,
                "pass": pass_name,
            }
            source_counts[key] = count
            logger.info(
                self,
                "root %d direct-source %-4s E = %.15f sources = %d time = %.2f s",
                prepared.root,
                key,
                class_energy,
                count,
                sub_times[key],
            )
        return _DirectPassResult(
            sub_eners=sub_eners,
            sub_times=sub_times,
            sub_diagnostics=sub_diagnostics,
            source_diagnostics=source_diagnostics,
            propagation_diagnostics=propagation_diagnostics,
            integration_diagnostics=integration_diagnostics,
            source_counts=source_counts,
            time_series=time_series,
        )

    def _factorized_class_integrand(
        self,
        prepared,
        plan: _FactorizedClassPlan,
        *,
        time_grid,
        time_step: float,
        tag_factory,
    ):
        """Evaluate one 2017 Table-I active Green-function contribution."""

        start = time.perf_counter()
        driver = prepared.driver
        if not plan.basis_sources:
            return (
                plan.external_energy,
                {},
                {
                    "analytic": True,
                    "propagated_source_count": 0,
                    "wall_seconds": time.perf_counter() - start,
                },
                {
                    "analytic": True,
                    "external_energy": plan.external_energy,
                },
                None,
                0,
            )

        grid = (
            _validate_time_grid(time_grid)
            if time_grid is not None
            else self._default_time_grid(time_step=time_step)
        )
        primitive_count = len(plan.basis_sources)
        source_diagnostics = {}
        identity = driver.get_identity_mpo()
        primitive_mps: list[Any | None] = []
        primitive_tags: list[str | None] = []
        basis_mps: list[Any | None] = []
        basis_tags: list[str | None] = []
        conditioning = None
        maximum_discarded_weight = 0.0
        overlap_count = 0
        try:
            for basis_index, source in enumerate(plan.basis_sources):
                label = f"{plan.key}:TableI:{basis_index}"
                mps, diagnostics = self._build_source_mps(
                    prepared,
                    source,
                    tag_factory=tag_factory,
                )
                diagnostics["basis_role"] = "primitive_table_i"
                diagnostics["table_i_basis_index"] = basis_index
                diagnostics["operator_constant"] = [
                    source.constant.real,
                    source.constant.imag,
                ]
                diagnostics["operator_terms"] = [
                    {
                        "operators": list(term.operators),
                        "coefficient": [
                            term.coefficient.real,
                            term.coefficient.imag,
                        ],
                    }
                    for term in source.terms
                ]
                source_diagnostics[label] = diagnostics
                if mps is None or diagnostics.get("mps_norm_squared", 0.0) == 0.0:
                    primitive_mps.append(None)
                    primitive_tags.append(None)
                    if mps is not None:
                        _release_owned_mps(
                            driver,
                            mps,
                            diagnostics["tag"],
                            remove_files=bool(self.cleanup_mps),
                        )
                else:
                    primitive_mps.append(mps)
                    primitive_tags.append(diagnostics["tag"])

            (
                primitive_gram,
                primitive_overlap_diagnostics,
            ) = _measure_hermitian_mps_gram(
                driver,
                primitive_mps,
                identity,
                name=f"{plan.key} primitive Gram",
            )
            overlap_count += primitive_overlap_diagnostics["expectation_count"]
            if (
                primitive_overlap_diagnostics["maximum_directional_relative_residual"]
                > self.correlation_residual_tol
            ):
                _utils._warn_numerical(
                    f"factorized class {plan.key} primitive Gram has a "
                    "directional overlap residual of "
                    f"{primitive_overlap_diagnostics['maximum_directional_residual']:.3e}"
                )
            if (
                primitive_overlap_diagnostics["maximum_diagonal_imaginary_residual"]
                > self.energy_imag_tol
            ):
                _utils._warn_numerical(
                    f"factorized class {plan.key} primitive Gram diagonal "
                    "has an imaginary residual of "
                    f"{primitive_overlap_diagnostics['maximum_diagonal_imaginary_residual']:.3e}"
                )

            source_norms = np.sqrt(np.clip(np.diag(primitive_gram).real, 0.0, None))
            # Apply the rank test to the physical, unscaled Gram matrix.
            # Normalizing every merely nonzero fitted primitive first would
            # turn a numerically null source into an O(1) vector and amplify
            # an allowed absolute fit error by its inverse norm before that
            # null direction can be removed.
            conditioned_plan, conditioning, lowdin_transform = (
                _orthogonalize_factorized_plan(
                    plan,
                    primitive_gram,
                    absolute_tolerance=float(self.gf_rank_atol),
                    relative_tolerance=float(self.gf_rank_rtol),
                    discarded_weight_atol=float(self.gf_discarded_weight_atol),
                    discarded_weight_rtol=float(self.gf_discarded_weight_rtol),
                    coefficient_cutoff=float(self.source_coefficient_cutoff),
                    return_transform=True,
                )
            )
            conditioning["primitive_overlap_measurement"] = (
                primitive_overlap_diagnostics
            )
            primitive_nonzero_count = int(sum(mps is not None for mps in primitive_mps))
            conditioning["primitive_nonzero_count"] = primitive_nonzero_count
            conditioning["primitive_propagation_norms"] = [
                float(value) for value in source_norms
            ]
            lowdin_normalized_mask = np.asarray(
                conditioning["conditioned_lowdin_normalized"],
                dtype=bool,
            )
            numerical_rank_reduction_applied = bool(
                conditioning["retained_rank"] < primitive_nonzero_count
            )
            # A source-norm threshold cannot bound an inverse resolvent when
            # a denominator is small.  Consequently every fitted-Gram
            # direction is retained above; any future discarded nonzero
            # coefficient-supported direction would be reported as an energy
            # projection, irrespective of the separate source-weight
            # diagnostic.
            rank_projection_applied = bool(
                conditioning["discarded_energy_projection_applied"]
            )
            conditioning["numerical_rank_reduction_applied"] = (
                numerical_rank_reduction_applied
            )
            conditioning["table_i_energy_projection_applied"] = rank_projection_applied
            if rank_projection_applied:
                _utils._warn_numerical(
                    f"factorized class {plan.key} numerical-rank reduction "
                    "projects a nonzero coefficient-supported source "
                    "direction (discarded weighted norm "
                    f"{conditioning['discarded_coefficient_weighted_source_norm_squared']:.3e}, "
                    "diagnostic tolerance "
                    f"{conditioning['discarded_source_weight_tolerance']:.3e}); "
                    "source norm alone provides no resolvent-energy bound"
                )

            # Time-step-targeting is nonlinear at finite M. Propagating a
            # nonorthogonal Table-I basis independently therefore makes the
            # result depend strongly on an otherwise immaterial source-basis
            # choice. Realize the lossless hybrid Löwdin/eigen-source basis as
            # physical MPSs and propagate those instead. Stable directions are
            # Löwdin-normalized; nonzero near-null directions remain unscaled,
            # so the transform stays invertible. Every transformed operator is
            # audited against its exact composite MPO: a radial-only fit would
            # not control primitive fit errors amplified by 1/sqrt(lambda).
            conditioned_source_diagnostics = []
            exact_zero_fit_kinds = {
                "exact_particle_sector_zero",
                "exact_structural_zero",
            }
            for basis_index, source in enumerate(conditioned_plan.basis_sources):
                mps, diagnostics = self._build_source_mps(
                    prepared,
                    source,
                    tag_factory=tag_factory,
                    residual_mode="exact_mpo",
                )
                if (
                    mps is None
                    and diagnostics.get("fit_gate_kind") not in exact_zero_fit_kinds
                ):
                    raise RuntimeError(
                        f"conditioned factorized source {plan.key} "
                        f"{basis_index} is absent without an exact-zero certificate"
                    )
                if (
                    mps is not None
                    and diagnostics.get("fit_gate_kind") not in exact_zero_fit_kinds
                    and float(diagnostics.get("mps_norm_squared", 0.0)) <= 0.0
                ):
                    _release_owned_mps(
                        driver,
                        mps,
                        diagnostics["tag"],
                        remove_files=bool(self.cleanup_mps),
                    )
                    raise RuntimeError(
                        f"conditioned factorized source {plan.key} "
                        f"{basis_index} has a numerically zero fitted MPS "
                        "without an exact structural/particle-sector certificate"
                    )
                diagnostics["basis_role"] = "lowdin_propagation"
                diagnostics["lowdin_basis_index"] = basis_index
                diagnostics["operator_constant"] = [
                    source.constant.real,
                    source.constant.imag,
                ]
                diagnostics["operator_terms"] = [
                    {
                        "operators": list(term.operators),
                        "coefficient": [
                            term.coefficient.real,
                            term.coefficient.imag,
                        ],
                    }
                    for term in source.terms
                ]
                diagnostics["lowdin_normalized_direction"] = bool(
                    lowdin_normalized_mask[basis_index]
                )
                conditioned_source_diagnostics.append(diagnostics)
                basis_mps.append(mps)
                basis_tags.append(diagnostics["tag"] if mps is not None else None)

            conditioned_gram, conditioned_overlap_diagnostics = (
                _measure_hermitian_mps_gram(
                    driver,
                    basis_mps,
                    identity,
                    name=f"{plan.key} conditioned Gram",
                )
            )
            mixed_gram, mixed_overlap_diagnostics = _measure_mps_cross_gram(
                driver,
                primitive_mps,
                basis_mps,
                identity,
                name=f"{plan.key} primitive/conditioned Gram",
            )
            overlap_count += conditioned_overlap_diagnostics["expectation_count"]
            overlap_count += mixed_overlap_diagnostics["expectation_count"]
            expected_conditioned_gram = (
                lowdin_transform.conj().T @ primitive_gram @ lowdin_transform
            )
            expected_mixed_gram = primitive_gram @ lowdin_transform
            mixed_gram_error = float(
                np.max(
                    np.abs(mixed_gram - expected_mixed_gram),
                    initial=0.0,
                )
            )
            realization_residuals = []
            realization_gates = []
            for basis_index in range(len(basis_mps)):
                target_norm_raw = float(
                    expected_conditioned_gram[basis_index, basis_index].real
                )
                target_norm = max(target_norm_raw, 0.0)
                actual_norm = float(conditioned_gram[basis_index, basis_index].real)
                actual_target_overlap = np.dot(
                    lowdin_transform[:, basis_index],
                    mixed_gram[:, basis_index].conj(),
                )
                raw_squared = float(
                    actual_norm + target_norm - 2.0 * actual_target_overlap.real
                )
                cancellation_scale = max(
                    actual_norm + target_norm + 2.0 * abs(actual_target_overlap),
                    self.norm_tol**2,
                )
                roundoff_squared = float(
                    512.0 * np.finfo(float).eps * cancellation_scale
                )
                requested = float(
                    self.source_residual_atol
                    + self.source_residual_rtol * math.sqrt(max(target_norm, 0.0))
                )
                effective_squared = requested**2 + roundoff_squared
                residual = math.sqrt(max(raw_squared, 0.0))
                mixed_gram_gate = bool(
                    raw_squared >= -effective_squared
                    and raw_squared <= effective_squared
                )
                fit_kind = conditioned_source_diagnostics[basis_index].get(
                    "fit_gate_kind"
                )
                exact_zero = fit_kind in exact_zero_fit_kinds
                exact_fit_gate = bool(
                    conditioned_source_diagnostics[basis_index].get("fit_gate_passed")
                    is True
                    and (
                        exact_zero
                        or (
                            fit_kind == "full_source_residual"
                            and conditioned_source_diagnostics[basis_index].get(
                                "radial_stationarity_gate_passed"
                            )
                            is True
                            and conditioned_source_diagnostics[basis_index].get(
                                "mps_norm_imaginary_gate_passed"
                            )
                            is True
                            and conditioned_source_diagnostics[basis_index].get(
                                "rhs_norm_imaginary_gate_passed"
                            )
                            is True
                            and conditioned_source_diagnostics[basis_index].get(
                                "reported_norm_consistency_gate_passed"
                            )
                            is True
                        )
                    )
                )
                gate = exact_fit_gate
                gate_kind = (
                    "exact_certified_zero_source"
                    if exact_zero
                    else (
                        "exact_conditioned_composite_source_residual"
                        if lowdin_normalized_mask[basis_index]
                        else "exact_near_null_composite_source_residual"
                    )
                )
                selected_raw_squared = float(
                    conditioned_source_diagnostics[basis_index].get(
                        "full_fit_residual_squared_raw",
                        0.0,
                    )
                )
                selected_residual = float(
                    conditioned_source_diagnostics[basis_index].get(
                        "full_fit_residual_norm",
                        0.0,
                    )
                )
                selected_tolerance = float(
                    conditioned_source_diagnostics[basis_index].get(
                        "full_fit_residual_tolerance",
                        0.0,
                    )
                )
                selected_effective_tolerance = float(
                    conditioned_source_diagnostics[basis_index].get(
                        "full_fit_residual_effective_tolerance",
                        0.0,
                    )
                )
                selected_nonnegative_gate = bool(
                    conditioned_source_diagnostics[basis_index].get(
                        "full_fit_residual_nonnegative_gate_passed"
                    )
                    is True
                )
                realization_residuals.append(float(selected_residual))
                realization_gates.append(gate)
                conditioned_source_diagnostics[basis_index].update(
                    {
                        "realization_target": "exact_transformed_table_i_operator",
                        "realization_gate_kind": gate_kind,
                        "mixed_realization_target_norm_squared_raw": (target_norm_raw),
                        "mixed_realization_fit_residual_squared_raw": raw_squared,
                        "mixed_realization_fit_residual_norm": float(residual),
                        "mixed_realization_fit_residual_tolerance": requested,
                        "mixed_realization_fit_residual_effective_tolerance": (
                            math.sqrt(effective_squared)
                        ),
                        "mixed_realization_fit_gate_passed": mixed_gram_gate,
                        "realization_fit_residual_squared_raw": (selected_raw_squared),
                        "realization_fit_residual_norm": float(selected_residual),
                        "realization_fit_residual_tolerance": (selected_tolerance),
                        "realization_fit_residual_effective_tolerance": (
                            selected_effective_tolerance
                        ),
                        "realization_fit_residual_nonnegative_gate_passed": (
                            selected_nonnegative_gate
                        ),
                        "realization_fit_gate_passed": gate,
                    }
                )
                if not gate:
                    _utils._warn_numerical(
                        f"factorized class {plan.key} conditioned source "
                        f"{basis_index} realization residual "
                        f"{selected_residual:.3e} exceeds/fails "
                        f"{selected_effective_tolerance:.3e} ({gate_kind})"
                    )

            conditioned_norms = np.sqrt(
                np.clip(np.diag(conditioned_gram).real, 0.0, None)
            )
            available_conditioned_mask = np.asarray(
                [mps is not None for mps in basis_mps],
                dtype=bool,
            )
            if np.any(
                lowdin_normalized_mask
                & available_conditioned_mask
                & (conditioned_norms <= 0.0)
            ):
                raise RuntimeError(
                    f"factorized class {plan.key} has a zero-norm "
                    "Löwdin-normalized propagation source"
                )
            propagation_norms = np.where(
                lowdin_normalized_mask & available_conditioned_mask,
                conditioned_norms,
                1.0,
            )
            for basis_index, mps in enumerate(basis_mps):
                if lowdin_normalized_mask[basis_index] and mps is not None:
                    mps.load_mutable()
                    mps.iscale(1.0 / propagation_norms[basis_index])
                    mps.save_data()
                conditioned_source_diagnostics[basis_index][
                    "propagation_normalization"
                ] = float(propagation_norms[basis_index])
                conditioned_source_diagnostics[basis_index][
                    "near_null_source_left_unscaled"
                ] = bool(not lowdin_normalized_mask[basis_index])
            inverse_conditioned_norms = 1.0 / propagation_norms
            conditioned_gram = (
                inverse_conditioned_norms[:, None]
                * conditioned_gram
                * inverse_conditioned_norms[None, :]
            )
            expected_propagation_gram = (
                inverse_conditioned_norms[:, None]
                * expected_conditioned_gram
                * inverse_conditioned_norms[None, :]
            )
            conditioned_sources = tuple(
                _ActiveSource(
                    key=source.key,
                    free_indices=source.free_indices,
                    delta_n_active=source.delta_n_active,
                    constant=(source.constant * inverse_conditioned_norms[index]),
                    terms=tuple(
                        _ActiveOperatorTerm(
                            term.operators,
                            term.coefficient * inverse_conditioned_norms[index],
                        )
                        for term in source.terms
                    ),
                    discarded_coefficient_l1=(
                        source.discarded_coefficient_l1
                        * inverse_conditioned_norms[index]
                    ),
                )
                for index, source in enumerate(conditioned_plan.basis_sources)
            )
            coefficient_plan = _FactorizedClassPlan(
                key=conditioned_plan.key,
                basis_sources=conditioned_sources,
                coefficients=(
                    conditioned_plan.coefficients * propagation_norms[None, :]
                ),
                gaps=conditioned_plan.gaps,
                block_labels=conditioned_plan.block_labels,
                external_energy=conditioned_plan.external_energy,
                external_block_energies=(conditioned_plan.external_block_energies),
            )
            conditioning.update(
                {
                    "realization": "lossless_hybrid_lowdin_propagation_basis",
                    "rank_projection_applied": rank_projection_applied,
                    "conditioned_source_diagnostics": (conditioned_source_diagnostics),
                    "conditioned_overlap_measurement": (
                        conditioned_overlap_diagnostics
                    ),
                    "mixed_overlap_measurement": (mixed_overlap_diagnostics),
                    "mixed_gram_maximum_abs_error": mixed_gram_error,
                    "maximum_realization_fit_residual": max(
                        realization_residuals,
                        default=0.0,
                    ),
                    "conditioned_realization_gate_passed": bool(all(realization_gates)),
                    "expected_conditioned_gram_deviation_from_identity": float(
                        np.max(
                            np.abs(
                                expected_conditioned_gram
                                - np.eye(
                                    expected_conditioned_gram.shape[0],
                                    dtype=np.complex128,
                                )
                            ),
                            initial=0.0,
                        )
                    ),
                    "null_space_projection_error": None,
                }
            )

            initial_gram = conditioned_gram.copy()
            conditioning.update(
                {
                    "primitive_gram_real": primitive_gram.real.tolist(),
                    "primitive_gram_imag": primitive_gram.imag.tolist(),
                    "lowdin_transform_real": lowdin_transform.real.tolist(),
                    "lowdin_transform_imag": lowdin_transform.imag.tolist(),
                    "conditioned_gram_measured_after_propagation_normalization_real": (
                        initial_gram.real.tolist()
                    ),
                    "conditioned_gram_measured_after_propagation_normalization_imag": (
                        initial_gram.imag.tolist()
                    ),
                    "conditioned_gram_expected_after_propagation_normalization_real": (
                        expected_propagation_gram.real.tolist()
                    ),
                    "conditioned_gram_expected_after_propagation_normalization_imag": (
                        expected_propagation_gram.imag.tolist()
                    ),
                    "primitive_coefficients_real": plan.coefficients.real.tolist(),
                    "primitive_coefficients_imag": plan.coefficients.imag.tolist(),
                    "external_gaps": plan.gaps.tolist(),
                    "external_energy": float(plan.external_energy),
                    "propagation_coefficients_real": (
                        coefficient_plan.coefficients.real.tolist()
                    ),
                    "propagation_coefficients_imag": (
                        coefficient_plan.coefficients.imag.tolist()
                    ),
                }
            )
            primitive_fit_gates = []
            primitive_fit_error_bounds = []
            for primitive_index in range(primitive_count):
                diagnostics = source_diagnostics[f"{plan.key}:TableI:{primitive_index}"]
                exact_zero = diagnostics.get("fit_gate_kind") in exact_zero_fit_kinds
                primitive_fit_gates.append(
                    bool(
                        diagnostics.get("fit_gate_passed") is True
                        and (
                            exact_zero
                            or diagnostics.get("fit_gate_kind")
                            == "full_source_residual"
                        )
                    )
                )
                primitive_fit_error_bounds.append(
                    0.0
                    if exact_zero
                    else float(
                        diagnostics.get(
                            "full_fit_residual_effective_tolerance",
                            math.inf,
                        )
                    )
                )
            primitive_fit_error_bounds = np.asarray(
                primitive_fit_error_bounds,
                dtype=float,
            )
            transformed_primitive_error_bounds = (
                np.abs(lowdin_transform).T @ primitive_fit_error_bounds
            )
            conditioned_fit_error_bounds = np.asarray(
                [
                    (
                        0.0
                        if diagnostics.get("fit_gate_kind") in exact_zero_fit_kinds
                        else float(
                            diagnostics.get(
                                "full_fit_residual_effective_tolerance",
                                math.inf,
                            )
                        )
                    )
                    for diagnostics in conditioned_source_diagnostics
                ],
                dtype=float,
            )
            # For x_j the fitted conditioned MPS, b_i the fitted primitive
            # MPSs, and t_i the exact Table-I sources,
            # ||x_j-sum_i T_ij b_i|| <= ||x_j-sum_i T_ij t_i||
            #   + sum_i |T_ij| ||t_i-b_i||.
            # This is the strict error bound needed when T contains a large
            # inverse square root of a small fitted-Gram eigenvalue.
            realization_error_bounds = (
                conditioned_fit_error_bounds + transformed_primitive_error_bounds
            ) / propagation_norms
            if not np.all(np.isfinite(realization_error_bounds)):
                raise RuntimeError(
                    f"factorized class {plan.key} has a non-finite "
                    "conditioned-source realization error bound"
                )
            target_norms = np.sqrt(
                np.clip(np.diag(expected_propagation_gram).real, 0.0, None)
            )
            gram_roundoff_limits = self.gf_rank_atol + self.gf_rank_rtol * np.maximum(
                1.0,
                np.abs(expected_propagation_gram),
            )
            # If x_i=t_i+d_i and ||d_i||<=r_i, Cauchy--Schwarz gives
            # |<x_i|x_j>-<t_i|t_j>| <= r_i||t_j|| + ||t_i||r_j + r_i r_j.
            # The measured conditioned Gram therefore cannot be required to
            # agree more tightly than the independently certified source-fit
            # residuals used to construct it.
            gram_limits = (
                gram_roundoff_limits
                + realization_error_bounds[:, None] * target_norms[None, :]
                + target_norms[:, None] * realization_error_bounds[None, :]
                + realization_error_bounds[:, None] * realization_error_bounds[None, :]
            )
            gram_errors = np.abs(initial_gram - expected_propagation_gram)
            conditioned_gram_error = float(np.max(gram_errors, initial=0.0))
            gram_gate = bool(np.all(gram_errors <= gram_limits))
            gram_ratios = np.zeros_like(gram_errors, dtype=float)
            np.divide(
                gram_errors,
                gram_limits,
                out=gram_ratios,
                where=gram_limits > 0.0,
            )
            gram_ratios[(gram_limits == 0.0) & (gram_errors > 0.0)] = np.inf
            maximum_gram_ratio = float(np.max(gram_ratios, initial=0.0))
            conditioning["conditioned_gram_error"] = conditioned_gram_error
            conditioning["conditioned_nonzero_count"] = int(
                np.count_nonzero(np.diag(initial_gram).real > 0.0)
            )
            conditioning["conditioned_realization_error_bounds"] = [
                float(value) for value in realization_error_bounds
            ]
            conditioning["conditioned_exact_fit_error_bounds"] = [
                float(value / propagation_norms[index])
                for index, value in enumerate(conditioned_fit_error_bounds)
            ]
            conditioning["transformed_primitive_fit_error_bounds"] = [
                float(value / propagation_norms[index])
                for index, value in enumerate(transformed_primitive_error_bounds)
            ]
            conditioning["primitive_full_residual_gate_passed"] = bool(
                all(primitive_fit_gates)
            )
            conditioning["conditioned_realization_gate_passed"] = bool(
                conditioning["conditioned_realization_gate_passed"]
                and all(primitive_fit_gates)
            )
            conditioning["conditioned_gram_minimum_tolerance"] = (
                float(np.min(gram_limits)) if gram_limits.size else 0.0
            )
            conditioning["conditioned_gram_tolerance"] = float(
                np.max(gram_limits, initial=0.0)
            )
            conditioning["conditioned_gram_elementwise_errors"] = gram_errors.tolist()
            conditioning["conditioned_gram_elementwise_tolerances"] = (
                gram_limits.tolist()
            )
            conditioning["conditioned_gram_maximum_error_to_tolerance_ratio"] = (
                maximum_gram_ratio
            )
            conditioning["conditioned_gram_gate_passed"] = bool(
                gram_gate and all(primitive_fit_gates)
            )
            if not gram_gate:
                _utils._warn_numerical(
                    f"factorized class {plan.key} conditioned source Gram/"
                    f"target error {conditioned_gram_error:.3e} has maximum "
                    f"error/tolerance ratio {maximum_gram_ratio:.3e}"
                )

            for mps, tag in zip(primitive_mps, primitive_tags):
                if mps is not None and tag is not None:
                    _release_owned_mps(
                        driver,
                        mps,
                        tag,
                        remove_files=bool(self.cleanup_mps),
                    )
            primitive_mps = []
            primitive_tags = []
            gc.collect()

            available_mask = np.asarray(
                [mps is not None for mps in basis_mps],
                dtype=bool,
            )
            coupling_mask = _coefficient_coupling_mask(
                coefficient_plan.coefficients
            )
            coupling_mask &= available_mask[:, None] & available_mask[None, :]
            conditioning["conditioned_available_mask"] = available_mask.tolist()
            conditioning["coefficient_coupling_mask"] = coupling_mask.tolist()
            conditioning["propagated_basis_indices"] = []
            available_source_count = int(np.count_nonzero(available_mask))
            if available_source_count == 0:
                exact_zero_certified = bool(
                    all(
                        diagnostics.get("fit_gate_kind") in exact_zero_fit_kinds
                        and diagnostics.get("fit_gate_passed") is True
                        for diagnostics in conditioned_source_diagnostics
                    )
                )
                if not exact_zero_certified:
                    raise RuntimeError(
                        f"factorized class {plan.key} has no realizable source "
                        "without an exact-zero certificate"
                    )
                return (
                    plan.external_energy,
                    source_diagnostics,
                    {
                        "analytic": False,
                        "zero_source": True,
                        "exact_zero_source_certified": True,
                        "propagated_source_count": 0,
                        "declared_source_count": len(basis_mps),
                        "table_i_source_count": primitive_count,
                        "green_function_rank": len(coefficient_plan.basis_sources),
                        "green_function_coupled_entry_count": int(
                            np.count_nonzero(coupling_mask)
                        ),
                        "green_function_evaluated_entry_count": int(
                            np.count_nonzero(coupling_mask)
                        ),
                        "green_function_structural_zero_count": int(
                            coupling_mask.size - np.count_nonzero(coupling_mask)
                        ),
                        "normalize_mps": False,
                        "basis_conditioning": conditioning,
                        "wall_seconds": time.perf_counter() - start,
                    },
                    {
                        "analytic": False,
                        "zero_source": True,
                        "exact_zero_source_certified": True,
                        "tail": 0.0,
                        "tail_gate_passed": True,
                        "active_energy": 0.0,
                        "external_energy": plan.external_energy,
                    },
                    None,
                    0,
                )

            dense_overlap_context = self._build_dense_csf_overlap_context(
                prepared,
                basis_mps,
                initial_gram,
                tag_factory=tag_factory,
                key=plan.key,
            )

            prefactors = np.asarray(
                [_factorized_prefactor(coefficient_plan, tau) for tau in grid],
                dtype=np.complex128,
            )
            prefactor_hermiticity_residual = float(
                np.max(
                    np.abs(prefactors - prefactors.conj().transpose(0, 2, 1)),
                    initial=0.0,
                )
            )
            prefactor_scale = max(
                1.0,
                float(np.max(np.abs(prefactors), initial=0.0)),
            )
            maximum_prefactor_magnitude = float(np.max(np.abs(prefactors), initial=0.0))
            prefactor_hermiticity_tolerance = float(
                self.energy_imag_tol * prefactor_scale
            )
            prefactor_hermiticity_gate = bool(
                prefactor_hermiticity_residual <= prefactor_hermiticity_tolerance
            )
            if not prefactor_hermiticity_gate:
                _utils._warn_numerical(
                    f"factorized class {plan.key} Table-I prefactor has "
                    f"Hermiticity residual {prefactor_hermiticity_residual:.3e}, "
                    f"above {prefactor_hermiticity_tolerance:.3e}"
                )
            propagated_basis_count = len(basis_mps)
            components = np.zeros(
                (propagated_basis_count, len(grid)),
                dtype=np.complex128,
            )
            raw_components = np.zeros_like(components)
            coupled_pairs = np.argwhere(coupling_mask).astype(
                np.int64,
                copy=False,
            )
            raw_green_function_overlaps = np.zeros(
                (len(coupled_pairs), len(grid)),
                dtype=np.complex128,
            )
            propagated_source_count = 0
            propagated_basis_indices = []
            for ket_index, initial_ket in enumerate(basis_mps):
                if initial_ket is None:
                    continue
                required_bras = coupling_mask[:, ket_index]
                if not np.any(required_bras):
                    continue
                overlap_bras = [
                    bra if required else None
                    for bra, required in zip(basis_mps, required_bras)
                ]
                overlaps = np.where(
                    required_bras,
                    initial_gram[:, ket_index],
                    0.0,
                )
                time_bonds = self._bond_schedule(
                    self.time_bond_dims,
                    default=int(initial_ket.info.bond_dim),
                    name="time_bond_dims",
                )
                (
                    samples,
                    discarded_weight,
                    sampled_overlaps,
                    _discarded_weight_series,
                ) = self._propagate_overlap_grid(
                    prepared,
                    initial_ket,
                    overlap_bras,
                    identity,
                    grid,
                    overlaps,
                    time_bonds=time_bonds,
                    tag_factory=tag_factory,
                    key=plan.key,
                    ket_index=ket_index,
                    dense_overlap_context=dense_overlap_context,
                )
                maximum_discarded_weight = max(
                    maximum_discarded_weight,
                    discarded_weight,
                )
                overlap_count += sampled_overlaps
                propagated_source_count += 1
                propagated_basis_indices.append(ket_index)
                pair_rows = np.flatnonzero(
                    coupled_pairs[:, 1] == ket_index
                )
                raw_green_function_overlaps[pair_rows, :] = samples[
                    coupled_pairs[pair_rows, 0],
                    :,
                ]
                for step_index, prefactor in enumerate(prefactors):
                    forward = np.dot(
                        prefactor[:, ket_index],
                        samples[:, step_index],
                    )
                    adjoint = np.dot(
                        prefactor[ket_index, :],
                        samples[:, step_index].conj(),
                    )
                    raw_components[ket_index, step_index] = forward
                    components[ket_index, step_index] = 0.5 * (forward + adjoint)

            conditioning["propagated_basis_indices"] = propagated_basis_indices

            raw_integrand = np.sum(raw_components, axis=0)
            hermitianized_integrand = np.sum(components, axis=0)
            reality = _integrand_reality_diagnostics(
                raw_integrand,
                imaginary_tolerance=float(self.energy_imag_tol),
            )
            maximum_imaginary = reality["maximum_raw_integrand_imaginary_residual"]
            integrand_scale = reality["raw_integrand_imaginary_scale"]
            if not reality["raw_integrand_imaginary_gate_passed"]:
                _utils._warn_numerical(
                    f"factorized class {plan.key} integrand has Im residual "
                    f"{maximum_imaginary:.3e}, above "
                    f"{reality['raw_integrand_imaginary_tolerance']:.3e}"
                )
            positivity_violation = float(
                max(0.0, -float(np.min(raw_integrand.real, initial=0.0)))
            )
            monotonicity_violation = float(
                max(0.0, float(np.max(np.diff(raw_integrand.real), initial=0.0)))
            )
            if max(positivity_violation, monotonicity_violation) > (
                self.correlation_residual_tol * integrand_scale
            ):
                _utils._warn_numerical(
                    f"factorized class {plan.key} violates positive monotone "
                    "imaginary-time decay"
                )
            quadrature = _integrate_samples(
                grid,
                raw_integrand,
                mode=self.integration_mode,
            )
            hermitianized_quadrature = _integrate_samples(
                grid,
                hermitianized_integrand,
                mode=self.integration_mode,
            )
            tail_diagnostics = _estimate_exponential_tail(
                grid,
                raw_integrand,
                window=int(self.tail_fit_window),
                imaginary_tolerance=float(self.energy_imag_tol),
                negligible_tolerance=float(self.integrand_abs_tol),
                log_residual_tolerance=float(self.tail_fit_log_residual_tol),
                rate_sensitivity_tolerance=float(self.tail_fit_rate_sensitivity_tol),
            )
            tail_limit = self.tail_abs_tol + self.tail_rel_tol * abs(quadrature)
            tail_gate = bool(
                tail_diagnostics["accepted"] and tail_diagnostics["tail"] < tail_limit
            )
            if tail_diagnostics["accepted"]:
                tail = float(tail_diagnostics["tail"])
                if not tail_gate:
                    _utils._warn_numerical(
                        f"factorized class {plan.key} tail {tail:.3e} exceeds "
                        f"the requested {tail_limit:.3e}"
                    )
            else:
                tail = 0.0
                _utils._warn_numerical(
                    f"factorized class {plan.key} has no stable tail estimate "
                    f"({tail_diagnostics['reason']}); active GF is truncated"
                )
            # Primitive Green functions inherit the squared norm of the
            # retained reference MPS.  Remove it before adding the analytic
            # external contribution, which is defined for normalized Psi0.
            active_value = -(quadrature + tail) / prepared.reference_norm
            active_energy, energy_imaginary = _checked_real_scalar(
                active_value,
                name=f"factorized {plan.key} active GF energy",
                imaginary_tolerance=self.energy_imag_tol,
            )
            energy_imaginary_limit = float(
                self.energy_imag_tol * max(1.0, abs(active_energy))
            )
            energy_imaginary_scale = float(max(1.0, abs(active_energy)))
            total_energy = float(active_energy + plan.external_energy)
            dense_overlap_diagnostics = {
                "enabled": dense_overlap_context is not None,
            }
            if dense_overlap_context is not None:
                dense_overlap_diagnostics.update(
                    {
                        name: value
                        for name, value in dense_overlap_context.items()
                        if name not in ("determinants", "coefficient_matrix")
                    }
                )
            propagation = {
                "analytic": False,
                "factorized_overlap_backend": self.factorized_overlap_backend,
                "dense_csf_overlap": dense_overlap_diagnostics,
                "propagated_source_count": propagated_source_count,
                "declared_source_count": propagated_basis_count,
                "table_i_source_count": primitive_count,
                "green_function_rank": len(coefficient_plan.basis_sources),
                "basis_conditioning": conditioning,
                "green_function_overlap_count": overlap_count,
                "green_function_coupled_entry_count": int(
                    np.count_nonzero(coupling_mask)
                ),
                "green_function_evaluated_entry_count": int(
                    np.count_nonzero(coupling_mask)
                ),
                "green_function_structural_zero_count": int(
                    coupling_mask.size - np.count_nonzero(coupling_mask)
                ),
                "green_function_adjoint_reconstruction": False,
                "time_points": len(grid),
                "final_time": float(grid[-1]),
                "te_type": self.te_type,
                "normalize_mps": False,
                "maximum_discarded_weight": maximum_discarded_weight,
                "maximum_integrand_imaginary_residual": maximum_imaginary,
                "maximum_prefactor_hermiticity_residual": (
                    prefactor_hermiticity_residual
                ),
                "maximum_prefactor_magnitude": maximum_prefactor_magnitude,
                "prefactor_hermiticity_scale": prefactor_scale,
                "prefactor_hermiticity_tolerance": (prefactor_hermiticity_tolerance),
                "prefactor_hermiticity_gate_passed": prefactor_hermiticity_gate,
                **reality,
                "hermitian_green_function_assembly": False,
                "hermitianized_integrand_is_diagnostic_only": True,
                "maximum_hermitianization_correction": float(
                    np.max(
                        np.abs(raw_integrand - hermitianized_integrand),
                        initial=0.0,
                    )
                ),
                "positivity_violation": positivity_violation,
                "monotonicity_violation": monotonicity_violation,
                "wall_seconds": time.perf_counter() - start,
            }
            integration = dict(tail_diagnostics)
            integration.update(
                {
                    "quadrature": [quadrature.real, quadrature.imag],
                    "raw_quadrature": [quadrature.real, quadrature.imag],
                    "hermitianized_quadrature": [
                        hermitianized_quadrature.real,
                        hermitianized_quadrature.imag,
                    ],
                    **reality,
                    "maximum_prefactor_hermiticity_residual": (
                        prefactor_hermiticity_residual
                    ),
                    "maximum_prefactor_magnitude": maximum_prefactor_magnitude,
                    "prefactor_hermiticity_scale": prefactor_scale,
                    "prefactor_hermiticity_tolerance": (
                        prefactor_hermiticity_tolerance
                    ),
                    "prefactor_hermiticity_gate_passed": (prefactor_hermiticity_gate),
                    "tail_used": tail,
                    "tail_limit": float(tail_limit),
                    "tail_gate_passed": tail_gate,
                    "reference_normalization_divisor": (prepared.reference_norm),
                    "active_energy": active_energy,
                    "external_energy": plan.external_energy,
                    "energy_imaginary_residual": energy_imaginary,
                    "energy_real_magnitude": abs(active_energy),
                    "energy_imaginary_scale": energy_imaginary_scale,
                    "energy_imaginary_tolerance": energy_imaginary_limit,
                    "energy_imaginary_gate_passed": bool(
                        energy_imaginary <= energy_imaginary_limit
                    ),
                    "mode": self.integration_mode,
                }
            )
            series = None
            if self.store_time_series:
                series = {
                    "time": np.asarray(grid, dtype=float),
                    "integrand": raw_integrand,
                    "components": raw_components,
                    "green_function_overlap_pairs": coupled_pairs,
                    "green_function_overlaps": raw_green_function_overlaps,
                    "raw_integrand": raw_integrand,
                    "hermitianized_integrand": hermitianized_integrand,
                    "hermitianized_components": components,
                }
            return (
                total_energy,
                source_diagnostics,
                propagation,
                integration,
                series,
                propagated_source_count,
            )
        finally:
            _deallocate_mpo(identity)
            for mps, tag in zip(primitive_mps, primitive_tags):
                if mps is not None and tag is not None:
                    _release_owned_mps(
                        driver,
                        mps,
                        tag,
                        remove_files=bool(self.cleanup_mps),
                    )
            for mps, tag in zip(basis_mps, basis_tags):
                if mps is not None and tag is not None:
                    _release_owned_mps(
                        driver,
                        mps,
                        tag,
                        remove_files=bool(self.cleanup_mps),
                    )
            gc.collect()

    def _run_factorized_pass(
        self,
        prepared,
        *,
        time_grid,
        time_step: float,
        pass_name: str,
    ) -> _DirectPassResult:
        prepared.driver.bw.b.Random.rand_seed(int(self.source_random_seed))
        tag_factory = _TagFactory(prepared.root, "factorized")
        sub_eners = {}
        sub_times = {}
        sub_diagnostics = {}
        source_diagnostics = {}
        propagation_diagnostics = {}
        integration_diagnostics = {}
        source_counts = {}
        time_series = {}
        scheduled_grid = (
            _validate_time_grid(time_grid)
            if time_grid is not None
            else self._default_time_grid(time_step=time_step)
        )
        for key in SUBSPACE_ORDER:
            class_start = time.perf_counter()
            plan = _factorized_class_plan(
                key,
                prepared.eris,
                prepared.core_energy,
                prepared.virtual_energy,
                prepared.dm1,
                denominator_tolerance=self.denominator_tol,
            )
            (
                energy,
                class_sources,
                propagation,
                integration,
                series,
                source_count,
            ) = self._factorized_class_integrand(
                prepared,
                plan,
                time_grid=time_grid,
                time_step=time_step,
                tag_factory=tag_factory,
            )
            sub_eners[key] = float(energy)
            integration["signed_energy_contribution"] = float(energy)
            integration.update(
                {
                    "pass_name": str(pass_name),
                    "requested_time_step": float(time_step),
                    "explicit_time_grid": bool(time_grid is not None),
                    "scheduled_time_points": int(len(scheduled_grid)),
                    "scheduled_final_time": float(scheduled_grid[-1]),
                    "sampled_time_points": int(propagation.get("time_points", 0)),
                    "sampled_final_time": float(
                        propagation.get("final_time", 0.0)
                    ),
                }
            )
            sub_times[key] = time.perf_counter() - class_start
            sub_diagnostics[key] = {
                "pass": pass_name,
                "paper_class": {
                    "ijrs": "[0]",
                    "rsi": "[-1]",
                    "ijr": "[+1]",
                    "rs": "[-2]",
                    "ij": "[+2]",
                    "ir": "[0']",
                    "r": "[-1']",
                    "i": "[+1']",
                }[key],
                "external_energy": plan.external_energy,
                "external_block_energies": dict(plan.external_block_energies),
                "external_block_count": len(plan.block_labels),
                "coefficient_shape": list(plan.coefficients.shape),
                "declared_source_count": len(plan.basis_sources),
                "propagated_source_count": source_count,
            }
            source_diagnostics.update(class_sources)
            propagation_diagnostics[key] = propagation
            integration_diagnostics[key] = integration
            source_counts[key] = source_count
            if series is not None:
                time_series[key] = series
            logger.info(
                self,
                "root %d factorized-GF %-4s E = %.15f sources = %d time = %.2f s",
                prepared.root,
                key,
                energy,
                source_count,
                sub_times[key],
            )
        return _DirectPassResult(
            sub_eners=sub_eners,
            sub_times=sub_times,
            sub_diagnostics=sub_diagnostics,
            source_diagnostics=source_diagnostics,
            propagation_diagnostics=propagation_diagnostics,
            integration_diagnostics=integration_diagnostics,
            source_counts=source_counts,
            time_series=time_series,
        )

    def kernel(
        self,
        mc=None,
        mo_coeff=None,
        dm1=None,
        eris=None,
        eris_basis="input_mo",
        root=None,
        *,
        algorithm=None,
        te_type=None,
        time_grid=None,
        store_time_series=None,
    ):
        """Compute one state-specific fully uncontracted t-MPS correction."""

        start_wall = time.perf_counter()
        start_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self._reset_run_results()
        if mc is None:
            mc = self._mc
        if _utils._has_frozen_orbitals(getattr(mc, "frozen", None)):
            raise NotImplementedError("nonzero frozen spinors are outside v1")
        if root is None:
            root = self.root
        root = int(root)
        if root < 0:
            raise IndexError("root must be non-negative")
        if algorithm is None:
            algorithm = self.algorithm
        if te_type is None:
            te_type = self.te_type
        if store_time_series is not None:
            self.store_time_series = bool(store_time_series)
        eris_basis = _utils._normalize_eris_basis(eris_basis)
        self._validate_controls(
            algorithm=algorithm,
            te_type=te_type,
            time_grid=time_grid,
        )
        self.root = root
        self.algorithm = algorithm
        self.te_type = te_type
        if mo_coeff is None:
            mo_coeff = mc.mo_coeff

        prepared = self._prepare_root(
            mc,
            root=root,
            mo_coeff=np.asarray(mo_coeff),
            dm1=dm1,
            eris=eris,
            eris_basis=eris_basis,
        )
        # Cache exact B^dagger B expectations only for this prepared
        # root/reference.  The optional SC audit and both time grids can then
        # share them without ever carrying values into a later kernel call.
        self._rhs_norm_cache = {}
        self.reference_energy = prepared.reference_energy
        self.active_reference_energy = prepared.active_reference_energy
        self.mo_coeff = prepared.mo_coeff
        self.mo_energy = prepared.orbital_energy
        self.eris = prepared.eris
        self.eris_basis = "semicanonical"
        self.dm1_diagnostics = prepared.dm1_diagnostics
        self.dm1 = np.asarray(prepared.dm1, dtype=np.complex128).copy()
        self.integral_symmetry_diagnostics = prepared.integral_symmetry_diagnostics
        self.active_mpo_diagnostics = prepared.active_mpo_diagnostics
        if self.audit_against_sc:
            self.sc_moment_diagnostics = self._audit_sc_source_moments(
                prepared,
                mc=mc,
            )
        else:
            self.sc_moment_diagnostics = {
                "enabled": False,
                "production_rdm_ranks": [1],
            }

        self.control_diagnostics = self._control_snapshot(
            prepared,
            time_grid=time_grid,
        )

        logger.info(
            self,
            "X2C-t-MPS-NEVPT2 root=%d algorithm=%s te_type=%s dt=%.6g "
            "max_time=%.6g sequential_sources=True",
            root,
            algorithm,
            te_type,
            self.time_step,
            self.max_time,
        )
        logger.info(
            self,
            "source controls: %s",
            self.control_diagnostics["source"],
        )
        logger.info(
            self,
            "propagation/integration controls: %s / %s",
            self.control_diagnostics["propagation"],
            self.control_diagnostics["integration"],
        )
        run_pass = (
            self._run_direct_pass
            if algorithm == "direct_source"
            else self._run_factorized_pass
        )
        primary = run_pass(
            prepared,
            time_grid=time_grid,
            time_step=float(self.time_step),
            pass_name="primary",
        )
        selected = primary
        refinement = None
        if self.dt_refinement and time_grid is None:
            refined_step = self.time_step / int(self.dt_refinement_factor)
            refined = run_pass(
                prepared,
                time_grid=None,
                time_step=float(refined_step),
                pass_name="dt_refined",
            )
            class_errors = {
                key: abs(refined.sub_eners[key] - primary.sub_eners[key])
                for key in SUBSPACE_ORDER
            }
            total_error = abs(
                sum(refined.sub_eners.values()) - sum(primary.sub_eners.values())
            )
            primary_tail_gate = _completed_integration_gate(
                primary.integration_diagnostics,
                "tail_gate_passed",
            )
            refined_tail_gate = _completed_integration_gate(
                refined.integration_diagnostics,
                "tail_gate_passed",
            )
            primary_energy_reality_gate = _completed_integration_gate(
                primary.integration_diagnostics,
                "energy_imaginary_gate_passed",
            )
            refined_energy_reality_gate = _completed_integration_gate(
                refined.integration_diagnostics,
                "energy_imaginary_gate_passed",
            )
            if algorithm == "factorized_gf":
                primary_integrand_reality_gate = _completed_integration_gate(
                    primary.integration_diagnostics,
                    "raw_integrand_imaginary_gate_passed",
                )
                refined_integrand_reality_gate = _completed_integration_gate(
                    refined.integration_diagnostics,
                    "raw_integrand_imaginary_gate_passed",
                )
                primary_prefactor_hermiticity_gate = _completed_integration_gate(
                    primary.integration_diagnostics,
                    "prefactor_hermiticity_gate_passed",
                )
                refined_prefactor_hermiticity_gate = _completed_integration_gate(
                    refined.integration_diagnostics,
                    "prefactor_hermiticity_gate_passed",
                )
            else:
                primary_integrand_reality_gate = _completed_integration_gate(
                    primary.integration_diagnostics,
                    "correlation_imaginary_gate_passed",
                )
                refined_integrand_reality_gate = _completed_integration_gate(
                    refined.integration_diagnostics,
                    "correlation_imaginary_gate_passed",
                )
                primary_prefactor_hermiticity_gate = True
                refined_prefactor_hermiticity_gate = True
            refinement = {
                "factor": int(self.dt_refinement_factor),
                "primary_time_step": float(self.time_step),
                "refined_time_step": float(refined_step),
                "primary_subspace_energies": dict(primary.sub_eners),
                "refined_subspace_energies": dict(refined.sub_eners),
                "primary_total_energy_correction": float(
                    sum(primary.sub_eners.values())
                ),
                "refined_total_energy_correction": float(
                    sum(refined.sub_eners.values())
                ),
                "class_errors": class_errors,
                "maximum_class_error": max(class_errors.values(), default=0.0),
                "total_error": total_error,
                "tolerance": float(self.dt_energy_tolerance),
                "primary_integration_gate_evidence": _integration_gate_evidence(
                    primary.integration_diagnostics
                ),
                "refined_integration_gate_evidence": _integration_gate_evidence(
                    refined.integration_diagnostics
                ),
                "primary_source_gate_evidence": {
                    "source_diagnostics": dict(primary.source_diagnostics),
                    "propagation_diagnostics": dict(primary.propagation_diagnostics),
                    "source_counts": dict(primary.source_counts),
                },
                "refined_source_gate_evidence": {
                    "source_diagnostics": dict(refined.source_diagnostics),
                    "propagation_diagnostics": dict(refined.propagation_diagnostics),
                    "source_counts": dict(refined.source_counts),
                },
                "primary_tail_gate_passed": primary_tail_gate,
                "refined_tail_gate_passed": refined_tail_gate,
                "primary_energy_reality_gate_passed": (primary_energy_reality_gate),
                "refined_energy_reality_gate_passed": (refined_energy_reality_gate),
                "primary_raw_integrand_reality_gate_passed": (
                    primary_integrand_reality_gate
                ),
                "refined_raw_integrand_reality_gate_passed": (
                    refined_integrand_reality_gate
                ),
                "primary_prefactor_hermiticity_gate_passed": (
                    primary_prefactor_hermiticity_gate
                ),
                "refined_prefactor_hermiticity_gate_passed": (
                    refined_prefactor_hermiticity_gate
                ),
                "passed": bool(
                    max(max(class_errors.values(), default=0.0), total_error)
                    <= self.dt_energy_tolerance
                    and primary_tail_gate
                    and refined_tail_gate
                    and primary_energy_reality_gate
                    and refined_energy_reality_gate
                    and primary_integrand_reality_gate
                    and refined_integrand_reality_gate
                    and primary_prefactor_hermiticity_gate
                    and refined_prefactor_hermiticity_gate
                ),
            }
            if not refinement["passed"]:
                _utils._warn_numerical(
                    "t-MPS time-step refinement failed: max class error "
                    f"{refinement['maximum_class_error']:.3e}, total error "
                    f"{total_error:.3e}"
                )
            selected = refined

        self.sub_eners = dict(selected.sub_eners)
        self.sub_times = dict(selected.sub_times)
        self.sub_diagnostics = dict(selected.sub_diagnostics)
        self.source_diagnostics = dict(selected.source_diagnostics)
        self.propagation_diagnostics = dict(selected.propagation_diagnostics)
        self.integration_diagnostics = dict(selected.integration_diagnostics)
        if refinement is not None:
            self.integration_diagnostics["dt_refinement"] = refinement
        self.source_counts = dict(selected.source_counts)
        self.time_series = dict(selected.time_series)
        self._pass_time_series = (
            {
                "primary": dict(primary.time_series),
                "dt_refined": dict(refined.time_series),
            }
            if refinement is not None and self.store_time_series
            else (
                {"primary": dict(primary.time_series)}
                if self.store_time_series
                else {}
            )
        )
        self.e_corr = float(sum(self.sub_eners.values()))

        end_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        scratch = getattr(prepared.driver, "scratch", None)
        self.resource_diagnostics = {
            "wall_seconds": time.perf_counter() - start_wall,
            "max_rss_gib": float(end_rss) / 1024.0**2,
            "max_rss_delta_gib": float(max(0, end_rss - start_rss)) / 1024.0**2,
            "scratch_gib": float(_directory_size(scratch)) / 1024.0**3,
            "scratch_path": scratch,
        }
        logger.note(
            self,
            "root %d X2C-t-MPS-NEVPT2 E_corr = %.15f E_tot = %.15f",
            root,
            self.e_corr,
            float(self.e_tot),
        )
        return self.e_corr


X2CTNEVPT2 = X2CTMPSNEVPT2
TMPSNEVPT2 = X2CTMPSNEVPT2
