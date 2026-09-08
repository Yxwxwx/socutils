#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
r"""General-complex-spinor X2C fully internally contracted MRCISD.

This module implements the one-reference fully internally contracted (FIC)
MRCISD ansatz of Sivalingam, Krupicka, Auer, and Neese, J. Chem. Phys. 145,
054104 (2016), for the general complex spinors used by :mod:`socutils`.
The reference and all internally contracted single/double excitation vectors
are retained in a nonorthogonal basis,

.. math::

   |\Phi_0\rangle=|\Psi_0\rangle,\qquad
   |\Phi_\lambda\rangle=O_\lambda|\Psi_0\rangle.

Block2 Wick algebra derives the overlap and shifted physical-Hamiltonian
matrix elements directly in the raw SGF one- through four-particle RDM
convention.  The Hamiltonian convention is

.. math::

   H=\sum_{pq}h_{pq}C_pD_q
     +\frac12\sum_{pqrs}w_{pqrs}C_pC_qD_sD_r,
   \qquad w_{pqrs}=\mathrm{eri}_{prqs}.

In particular, ``h`` below is the *bare* one-electron Hamiltonian.  Inactive
contractions are performed by Wick; using the core-dressed ``h1eff`` here
would double count the inactive contribution.

The implementation is structurally based on the GPL-3.0-or-later Block2
``pyblock2.icmr.icmrcisd_full`` driver (Copyright 2020-2021 Huanchen Zhai),
but replaces its spin-free ``E1/E2``, real-transpose, ``qc_phys`` and real
reference-coefficient assumptions by explicit fermionic ``C/D`` operators,
Hermitian conjugation, and an empty ERI permutation table appropriate for
general complex spinors.

FIC-MRCISD is variational within its contracted CI space, but truncated MRCI
is not size extensive.  The Davidson ``+Q`` value exposed by this module is a
nonvariational a posteriori approximation, not a variational energy.

The same-sector commutator reduction assumes that the supplied RDMs belong to
a stationary eigenstate of the projected CAS Hamiltonian in the supplied MO
basis.  The driver audits this prerequisite through the active-Hamiltonian
variance and rejects a materially nonstationary reference.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache, wraps
import gc
import itertools
import math
import re
import time
from types import MappingProxyType
from typing import Any, Sequence

import numpy as np
from pyscf import lib
from pyscf.lib import logger
from scipy import linalg

from . import nevpt2_utils as _utils
from . import spinor_helper
from . import x2cficnevpt2 as _fic


__all__ = [
    "WickX2CICMRCISD",
    "X2CICMRCISD",
    "dump_x2cicmrcisd_wick_equations",
]


SECTOR_ORDER = ("ref",) + tuple(_utils.SUBSPACE_ORDER)
_KERNEL_KEYWORDS = frozenset(
    (
        "mc",
        "mo_coeff",
        "pdms",
        "eris",
        "eris_basis",
        "root",
        "nroots",
        "state_selection",
    )
)
_FREE_BRA_LABEL = MappingProxyType(
    {"i": "k", "j": "l", "r": "t", "s": "u"}
)
_SPACE_DUMMY_LABELS = MappingProxyType(
    {"I": "mnoz", "A": "defh", "E": "vwxy"}
)
_INDEX_SPACE = MappingProxyType(
    {
        **{label: "I" for label in "ijklmnoz"},
        **{label: "A" for label in "abcdefghpq"},
        **{label: "E" for label in "rstuvwxy"},
    }
)
_SPACE_ORDER = ("I", "A", "E")

# Canonical metric orthogonalization inherits the backward error of the
# Hermitian eigensolver and amplifies it by the condition number of the
# retained metric.  This fixed factor covers the eigensolve, column scaling,
# and BLAS products without making the acceptance threshold grow with rank.
_CANONICAL_ORTHOGONALIZATION_ROUNDOFF_FACTOR = 64.0

# This is deliberately a read-only adapter over the already regression-tested
# FIC-NEVPT2 FOIS definition.  MRCISD must not maintain a drifting second set
# of coefficient-free excitation operators.
_IC_COMPONENTS = MappingProxyType(
    {key: tuple(_fic._IC_COMPONENTS[key]) for key in _utils.SUBSPACE_ORDER}
)
_PAIR_RESTRICTIONS = MappingProxyType(
    {key: tuple(_utils._PAIR_RESTRICTIONS[key]) for key in _utils.SUBSPACE_ORDER}
)


@dataclass(frozen=True)
class _ReferenceComponent:
    name: str = "reference"
    expression: str = "1.0"
    ket_active: tuple[str, ...] = ()
    bra_active: tuple[str, ...] = ()
    active_pairs: tuple[tuple[int, int], ...] = ()


_REFERENCE_COMPONENT = _ReferenceComponent()


@dataclass(frozen=True)
class _EquationBundle:
    """Lazy Block2-Wick equations for all metric and Hamiltonian blocks."""

    metric_expr: dict[tuple[str, str, str], Any]
    hamiltonian_expr: dict[tuple[str, str, str, str], Any]
    metric_text: dict[tuple[str, str, str], str]
    hamiltonian_text: dict[tuple[str, str, str, str], str]
    metric_code: dict[tuple[str, str, str], str]
    hamiltonian_code: dict[tuple[str, str, str, str], str]
    required_h1_blocks: tuple[str, ...]
    required_eri_blocks: tuple[str, ...]
    maximum_rdm_rank: int
    cross_sector_overlap_count: int
    reference_first_moment: Any
    reference_second_moment: Any
    reference_moment_text: str


@dataclass(frozen=True)
class _SectorLayout:
    """Raw contracted basis layout for one external-occupation sector."""

    key: str
    components: tuple[Any, ...]
    free_tuples: tuple[tuple[int, ...], ...]
    active_selections: tuple[np.ndarray, ...]
    active_tuples: tuple[tuple[tuple[int, ...], ...], ...]
    component_slices: tuple[slice, ...]
    local_dimension: int
    raw_dimension: int
    raw_slice: slice


@dataclass(frozen=True)
class _MetricSpace:
    """One repeated free-index block of a sector metric."""

    layout: _SectorLayout
    metric: np.ndarray
    orthogonalizer: np.ndarray
    eigenvalues: np.ndarray
    cutoff: float
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class _OrthogonalBlock:
    """One fixed core/virtual occupation block in the global basis."""

    layout: _SectorLayout
    free_indices: tuple[int, ...]
    block_index: int
    raw_slice: slice
    orth_slice: slice
    orthogonalizer: np.ndarray
    holes: frozenset[int]
    particles: frozenset[int]


def _components(key: str) -> tuple[Any, ...]:
    if key == "ref":
        return (_REFERENCE_COMPONENT,)
    try:
        return _IC_COMPONENTS[key]
    except KeyError as error:
        raise ValueError(f"unknown FIC-MRCISD sector {key!r}") from error


def _replace_operator_labels(expression: str, mapping: dict[str, str]) -> str:
    """Replace only bracketed one-index fermion labels."""

    result = expression
    for source, target in mapping.items():
        result = result.replace(f"[{source}]", f"[{target}]")
    return result


def _ket_labels(key: str, component: Any) -> tuple[str, ...]:
    if key == "ref":
        return ()
    return tuple(key) + tuple(component.ket_active)


def _bra_labels(key: str, component: Any) -> tuple[str, ...]:
    if key == "ref":
        return ()
    return tuple(_FREE_BRA_LABEL[label] for label in key) + tuple(
        component.bra_active
    )


def _bra_expression(parse, key: str, component: Any):
    if key == "ref":
        return parse(component.expression)
    mapping = {label: _FREE_BRA_LABEL[label] for label in key}
    mapping.update(dict(zip(component.ket_active, component.bra_active)))
    return parse(
        _replace_operator_labels(component.expression, mapping)
    ).conjugate()


def _full_bare_hamiltonian(parse):
    """Build all 3**2 one-body and 3**4 two-body I/A/E blocks."""

    result = parse("")
    for spaces in itertools.product(_SPACE_ORDER, repeat=2):
        indices = "".join(
            _SPACE_DUMMY_LABELS[space][axis]
            for axis, space in enumerate(spaces)
        )
        result += parse(
            f"SUM <{indices}> h[{indices}] "
            f"C[{indices[0]}] D[{indices[1]}]"
        )
    for spaces in itertools.product(_SPACE_ORDER, repeat=4):
        indices = "".join(
            _SPACE_DUMMY_LABELS[space][axis]
            for axis, space in enumerate(spaces)
        )
        result += parse(
            f"0.5 SUM <{indices}> w[{indices}] "
            f"C[{indices[0]}] C[{indices[1]}] "
            f"D[{indices[3]}] D[{indices[2]}]"
        )
    return result


def _target(parse_tensor, name: str, labels: Sequence[str]):
    return parse_tensor(f"{name}[{''.join(labels)}]")


def _expression_rdm_rank(expression) -> int:
    maximum = 0
    for term in expression.terms:
        for tensor in term.tensors:
            match = re.fullmatch(r"dm([1-9][0-9]*)", tensor.name)
            if match is not None:
                maximum = max(maximum, int(match.group(1)))
    return maximum


def _expression_integral_blocks(expression, types):
    index_types = types["WickIndexTypes"]
    space = {
        index_types.Inactive: "I",
        index_types.Active: "A",
        index_types.External: "E",
    }
    h1 = set()
    eri = set()
    for term in expression.terms:
        for tensor in term.tensors:
            if tensor.name not in ("h", "hc", "w", "wc"):
                continue
            try:
                key = "".join(space[index.types] for index in tensor.indices)
            except KeyError as error:
                raise RuntimeError(
                    f"integral tensor {tensor!r} has a non-orbital index"
                ) from error
            (h1 if tensor.name in ("h", "hc") else eri).add(key)
    return h1, eri


def _preserve_omp_threads(function):
    """Keep Block2 Wick's global-thread activation local to compilation."""

    @wraps(function)
    def wrapped(*args, **kwargs):
        with lib.with_omp_threads(lib.num_threads()):
            return function(*args, **kwargs)

    return wrapped


@lru_cache(maxsize=1)
@_preserve_omp_threads
def _compile_equations() -> _EquationBundle:
    """Generate the complete complex-spinor FIC-MRCISD Wick system."""

    types = _utils._block2_wick_types()
    parse, parse_tensor = _utils._wick_parsers(types)
    hamiltonian = _full_bare_hamiltonian(parse)
    active_hamiltonian = parse(
        "SUM <ab> f[ab] C[a] D[b]"
    ) + parse(
        "0.5 SUM <abcd> w[abcd] C[a] C[b] D[d] D[c]"
    )
    reference_first_moment = _utils._lower_active_operators(
        _utils._vacuum_reduce(active_hamiltonian), types
    )
    reference_second_moment = _utils._lower_active_operators(
        _utils._vacuum_reduce(active_hamiltonian * active_hamiltonian), types
    )

    metric_expr = {}
    hamiltonian_expr = {}
    metric_text = {}
    hamiltonian_text = {}
    metric_code = {}
    hamiltonian_code = {}
    required_h1 = set()
    required_eri = set()
    maximum_rdm_rank = max(
        _expression_rdm_rank(reference_first_moment),
        _expression_rdm_rank(reference_second_moment),
    )

    # A nonzero overlap between two distinct external-occupation sectors would
    # invalidate the block metric construction and is therefore a compile-time
    # hard error, not a numerical assumption.
    cross_sector_overlap_count = 0
    for bra_key in SECTOR_ORDER:
        for bra_component in _components(bra_key):
            bra = _bra_expression(parse, bra_key, bra_component)
            for ket_key in SECTOR_ORDER:
                for ket_component in _components(ket_key):
                    ket = parse(ket_component.expression)
                    if bra_key == ket_key and bra_key != "ref":
                        metric = _utils._lower_active_operators(
                            _utils._vacuum_reduce(bra * ket), types
                        )
                        pair = (
                            bra_key,
                            bra_component.name,
                            ket_component.name,
                        )
                        labels = _bra_labels(
                            bra_key, bra_component
                        ) + _ket_labels(ket_key, ket_component)
                        metric_expr[pair] = metric
                        metric_text[pair] = repr(metric)
                        metric_code[pair] = metric.to_einsum(
                            _target(parse_tensor, "metric", labels)
                        )
                        maximum_rdm_rank = max(
                            maximum_rdm_rank,
                            _expression_rdm_rank(metric),
                        )
                    elif bra_key != ket_key:
                        overlap = _utils._lower_active_operators(
                            _utils._vacuum_reduce(bra * ket), types
                        )
                        if len(overlap.terms):
                            cross_sector_overlap_count += 1
                            raise RuntimeError(
                                "distinct FIC-MRCISD sectors have a nonzero "
                                f"symbolic overlap: {bra_key}/{bra_component.name} "
                                f"vs {ket_key}/{ket_component.name}: {overlap!r}"
                            )

                    if bra_key == "ref" and ket_key == "ref":
                        continue
                    if bra_key == ket_key:
                        # Expanding [H,O] before attaching the bra removes the
                        # disconnected H2*O2 term and is what keeps the active
                        # density rank at four or below.
                        connected = (
                            (hamiltonian ^ ket).expand(6).simplify()
                        )
                        raw = bra * connected
                        organization = "commutator"
                    else:
                        raw = bra * hamiltonian * ket
                        organization = "expectation"
                    element = _utils._lower_active_operators(
                        _utils._vacuum_reduce(raw), types
                    )
                    pair_h = (
                        bra_key,
                        bra_component.name,
                        ket_key,
                        ket_component.name,
                    )
                    labels_h = _bra_labels(
                        bra_key, bra_component
                    ) + _ket_labels(ket_key, ket_component)
                    hamiltonian_expr[pair_h] = element
                    hamiltonian_text[pair_h] = (
                        f"organization={organization}\n{element!r}"
                    )
                    hamiltonian_code[pair_h] = element.to_einsum(
                        _target(parse_tensor, "hamiltonian", labels_h)
                    )
                    maximum_rdm_rank = max(
                        maximum_rdm_rank,
                        _expression_rdm_rank(element),
                    )
                    hkeys, wkeys = _expression_integral_blocks(element, types)
                    required_h1.update(hkeys)
                    required_eri.update(wkeys)

    if maximum_rdm_rank > 4:
        raise RuntimeError(
            "FIC-MRCISD Wick lowering requested an active RDM above rank four"
        )
    return _EquationBundle(
        metric_expr=metric_expr,
        hamiltonian_expr=hamiltonian_expr,
        metric_text=metric_text,
        hamiltonian_text=hamiltonian_text,
        metric_code=metric_code,
        hamiltonian_code=hamiltonian_code,
        required_h1_blocks=tuple(sorted(required_h1)),
        required_eri_blocks=tuple(sorted(required_eri)),
        maximum_rdm_rank=maximum_rdm_rank,
        cross_sector_overlap_count=cross_sector_overlap_count,
        reference_first_moment=reference_first_moment,
        reference_second_moment=reference_second_moment,
        reference_moment_text=(
            f"<H_active>\n{reference_first_moment!r}\n\n"
            f"<H_active^2>\n{reference_second_moment!r}"
        ),
    )


def dump_x2cicmrcisd_wick_equations(filename: str | None = None) -> str:
    """Return, and optionally write, every generated MRCISD equation."""

    equations = _compile_equations()
    sections = [
        "[conventions]",
        "Hamiltonian: bare h[pq] C[p]D[q] + "
        "0.5 w[pqrs] C[p]C[q]D[s]D[r]",
        f"maximum active RDM rank: {equations.maximum_rdm_rank}",
        "required h blocks: " + ", ".join(equations.required_h1_blocks),
        "required w blocks: " + ", ".join(equations.required_eri_blocks),
        "[reference stationarity moments]\n"
        + equations.reference_moment_text,
    ]
    for key in _utils.SUBSPACE_ORDER:
        for component in _components(key):
            sections.append(
                f"[{key}/{component.name}] basis\n{component.expression}"
            )
        for bra_component in _components(key):
            for ket_component in _components(key):
                pair = (key, bra_component.name, ket_component.name)
                sections.extend(
                    (
                        f"[{key}/{bra_component.name},{ket_component.name}] "
                        f"metric expression\n{equations.metric_text[pair]}",
                        f"[{key}/{bra_component.name},{ket_component.name}] "
                        f"metric einsum\n{equations.metric_code[pair]}",
                    )
                )
    for bra_key in SECTOR_ORDER:
        for bra_component in _components(bra_key):
            for ket_key in SECTOR_ORDER:
                for ket_component in _components(ket_key):
                    if bra_key == ket_key == "ref":
                        continue
                    pair = (
                        bra_key,
                        bra_component.name,
                        ket_key,
                        ket_component.name,
                    )
                    title = (
                        f"{bra_key}/{bra_component.name} <- "
                        f"{ket_key}/{ket_component.name}"
                    )
                    sections.extend(
                        (
                            f"[{title}] shifted-H expression\n"
                            f"{equations.hamiltonian_text[pair]}",
                            f"[{title}] shifted-H einsum\n"
                            f"{equations.hamiltonian_code[pair]}",
                        )
                    )
    text = "\n\n".join(sections) + "\n"
    if filename is not None:
        with open(filename, "w", encoding="utf-8") as handle:
            handle.write(text)
    return text


def _label_dimension(label: str, eris) -> int:
    try:
        space = _INDEX_SPACE[label]
    except KeyError as error:
        raise ValueError(f"unknown Wick index label {label!r}") from error
    return {
        "I": int(eris.ncore),
        "A": int(eris.ncas),
        "E": int(eris.nvirt),
    }[space]


def _index_space_from_type(index_type, types) -> str:
    index_types = types["WickIndexTypes"]
    mapping = {
        index_types.Inactive: "I",
        index_types.Active: "A",
        index_types.External: "E",
    }
    try:
        return mapping[index_type]
    except KeyError as error:
        raise RuntimeError(f"unsupported Wick orbital-index type {index_type}") from error


@contextmanager
def _tblis_thread_limit(threads=None):
    """Explicit, restoring TBLIS-only control for serial validation drivers.

    Does not change BLAS/OpenMP settings or any numerical threshold. Like
    pytblis itself, this process-wide setting must not be switched concurrently
    by multiple Python threads. None preserves the caller's current setting.
    """

    if threads is None:
        yield
        return
    if (
        isinstance(threads, bool)
        or not isinstance(threads, (int, np.integer))
        or threads <= 0
    ):
        raise ValueError("TBLIS thread count must be a positive integer")
    import pytblis

    previous = pytblis.get_num_threads()
    pytblis.set_num_threads(int(threads))
    try:
        yield
    finally:
        pytblis.set_num_threads(previous)


class _ExpressionEvaluator:
    """Evaluate lowered Wick expressions with fixed nonactive indices.

    Block2's generated full-output einsums are retained for equation export.
    Production evaluation fixes a core/virtual occupation block first, which
    avoids allocating tensors containing every pair of external tuples.
    """

    def __init__(self, eris, pdms, contraction_backend: str):
        self.eris = eris
        input_pdms = tuple(np.asarray(dm) for dm in pdms)
        self.contraction_backend = _utils._normalize_contraction_backend(
            contraction_backend
        )
        self.types = _utils._block2_wick_types()
        self.einsum = _utils._wick_einsum_namespace(
            self.contraction_backend
        ).einsum
        arrays = (np.asarray(eris.h1e), np.asarray(eris.pppp), *input_pdms)
        self.strictly_real = all(
            not np.iscomplexobj(array)
            or not bool(np.any(np.asarray(array).imag != 0.0))
            for array in arrays
        )
        self.dtype = np.dtype(
            np.float64
            if self.strictly_real
            else np.result_type(
                np.complex128,
                eris.h1e.dtype,
                eris.pppp.dtype,
                *(density.dtype for density in input_pdms),
            )
        )
        self.pdms = tuple(
            np.asarray(density.real, dtype=self.dtype)
            if self.strictly_real
            else density
            for density in input_pdms
        )
        self._operand_cache: dict[tuple[str, str], np.ndarray] = {}
        # Plans belong to this input/evaluator, not the global Wick cache:
        # their operands are views of this calculation's integrals and RDMs.
        self._expression_plans = {}
        self.evaluation_diagnostics = {
            "expression_plans": 0,
            "expression_calls": 0,
            "terms_visited": 0,
            "exact_zero_terms": 0,
            "einsum_calls": 0,
        }

    def _coerce(self, value) -> np.ndarray:
        value = np.asarray(value)
        if self.strictly_real:
            return np.asarray(value.real, dtype=self.dtype)
        return value

    def _integral_key(self, tensor) -> str:
        return "".join(
            _index_space_from_type(index.types, self.types)
            for index in tensor.indices
        )

    def _operand(self, tensor) -> np.ndarray:
        tensor_types = self.types["WickTensorTypes"]
        if tensor.type == tensor_types.KroneckerDelta:
            first_space = _index_space_from_type(
                tensor.indices[0].types, self.types
            )
            if any(
                _index_space_from_type(index.types, self.types) != first_space
                for index in tensor.indices[1:]
            ):
                raise RuntimeError("Kronecker delta mixes orbital spaces")
            dimension = {
                "I": self.eris.ncore,
                "A": self.eris.ncas,
                "E": self.eris.nvirt,
            }[first_space]
            cache_key = ("delta", first_space)
            if cache_key not in self._operand_cache:
                self._operand_cache[cache_key] = np.eye(
                    int(dimension), dtype=self.dtype
                )
            return self._operand_cache[cache_key]

        name = tensor.name
        match = re.fullmatch(r"dm([1-4])", name)
        if match is not None:
            return self.pdms[int(match.group(1)) - 1]
        if name in ("h", "hc"):
            key = self._integral_key(tensor)
            cache_key = (name, key)
            if cache_key not in self._operand_cache:
                value = self.eris.get_h1(key)
                if name == "hc":
                    value = value.conj()
                self._operand_cache[cache_key] = self._coerce(value)
            return self._operand_cache[cache_key]
        if name in ("f", "fc"):
            cache_key = (name, "AA")
            if cache_key not in self._operand_cache:
                value = self.eris.get_h1eff("AA")
                if name == "fc":
                    value = value.conj()
                self._operand_cache[cache_key] = self._coerce(value)
            return self._operand_cache[cache_key]
        if name in ("w", "wc"):
            key = self._integral_key(tensor)
            cache_key = (name, key)
            if cache_key not in self._operand_cache:
                value = self.eris.get_phys(key)
                if name == "wc":
                    value = value.conj()
                self._operand_cache[cache_key] = self._coerce(value)
            return self._operand_cache[cache_key]
        raise RuntimeError(f"unsupported lowered Wick tensor {tensor!r}")

    def _prepare_expression(self, expression, output_labels, fixed_labels):
        """Parse immutable Wick metadata once for all external-index values.

        No numerical contraction or spin selection is done here. In particular,
        a plan may be reused for *different* external tuples, but its numerical
        result is never cached. Keep the expression alive to make its id safe.
        """

        key = (id(expression), output_labels, fixed_labels)
        cached = self._expression_plans.get(key)
        if cached is not None:
            return cached[1]
        plans = []
        for term in expression.terms:
            raw_factor = complex(term.factor)
            if self.strictly_real:
                if raw_factor.imag != 0.0:
                    raise FloatingPointError(
                        "a generated Wick coefficient is unexpectedly complex"
                    )
                factor = float(raw_factor.real)
            else:
                factor = raw_factor
            scalar_specs = []
            operand_specs = []
            subscripts = []
            carried_labels = set()
            for tensor in term.tensors:
                value = np.asarray(self._operand(tensor))
                indexer = []
                remaining = []
                for index in tensor.indices:
                    label = index.name
                    if label in fixed_labels:
                        indexer.append(label)
                    else:
                        indexer.append(slice(None))
                        remaining.append(label)
                        carried_labels.add(label)
                if not remaining:
                    scalar_specs.append((value, tuple(indexer)))
                else:
                    # An unsliced active-only operand is shared by every
                    # external tuple; avoid making even a new view of it.
                    sliced = any(
                        isinstance(item, str) for item in indexer
                    )
                    operand_specs.append((value, tuple(indexer) if sliced else ()))
                    subscripts.append("".join(remaining))
            present_output = tuple(
                label for label in output_labels if label in carried_labels
            )
            equation = ",".join(subscripts) + "->" + "".join(present_output)
            broadcast_shape = tuple(
                int(self.eris.ncas) if label in present_output else 1
                for label in output_labels
            )
            plans.append(
                (factor, scalar_specs, operand_specs, equation, broadcast_shape)
            )
        self._expression_plans[key] = (expression, plans)
        self.evaluation_diagnostics["expression_plans"] += 1
        return plans

    def evaluate(
        self,
        expression,
        output_labels: Sequence[str],
        fixed_indices: dict[str, int],
    ) -> np.ndarray:
        """Evaluate fixed-external blocks with exact scalar-zero screening.

        Fixed Kronecker deltas and scalar integral coefficients are evaluated
        before the active contractions. Only exact zero is screened; no energy,
        integral, RDM or matrix-element threshold is introduced.
        """

        output_labels = tuple(output_labels)
        if len(output_labels) != len(set(output_labels)):
            raise ValueError("Wick output labels must be unique")
        if any(_INDEX_SPACE[label] != "A" for label in output_labels):
            raise ValueError(
                "all remaining block-local Wick outputs must be active indices"
            )
        fixed_indices = {
            label: int(value) for label, value in fixed_indices.items()
        }
        for label, fixed in fixed_indices.items():
            if not 0 <= fixed < _label_dimension(label, self.eris):
                raise IndexError(f"fixed Wick index {label}={fixed} is out of range")
        plans = self._prepare_expression(
            expression, output_labels, tuple(sorted(fixed_indices))
        )
        target_shape = (int(self.eris.ncas),) * len(output_labels)
        result = np.zeros(target_shape, dtype=self.dtype)
        counts = self.evaluation_diagnostics
        counts["expression_calls"] += 1
        counts["terms_visited"] += len(plans)
        for factor, scalar_specs, operand_specs, equation, broadcast_shape in plans:
            for operand, indices in scalar_specs:
                if factor == 0:
                    break
                indexer = tuple(fixed_indices[label] for label in indices)
                factor *= operand[indexer].item()
            if factor == 0:
                counts["exact_zero_terms"] += 1
                continue
            operands = [
                operand[tuple(
                    fixed_indices[item] if isinstance(item, str) else item
                    for item in indices
                )] if indices else operand
                for operand, indices in operand_specs
            ]
            if operands:
                counts["einsum_calls"] += 1
                value = self.einsum(equation, *operands, optimize=True)
            else:
                value = np.asarray(1.0, dtype=self.dtype)
            value = factor * np.asarray(value)
            if output_labels:
                value = np.asarray(value).reshape(broadcast_shape)
                value = np.broadcast_to(value, target_shape)
            result += value
        return result


def _free_tuples(key: str, eris) -> tuple[tuple[int, ...], ...]:
    if key == "ref":
        return ((),)
    shape = tuple(_label_dimension(label, eris) for label in key)
    if any(dimension == 0 for dimension in shape):
        return ()
    restrictions = _PAIR_RESTRICTIONS[key]
    return tuple(
        tuple(int(value) for value in indices)
        for indices in itertools.product(*(range(size) for size in shape))
        if all(indices[left] < indices[right] for left, right in restrictions)
    )


def _make_layouts(eris) -> tuple[_SectorLayout, ...]:
    layouts = []
    raw_offset = 0
    for key in SECTOR_ORDER:
        components = _components(key)
        selections = []
        active_tuples = []
        component_slices = []
        local_offset = 0
        for component in components:
            if key == "ref":
                selection = np.array([0], dtype=np.int64)
                tuples = ((),)
            else:
                selection, tuples = _fic._active_selection(
                    int(eris.ncas),
                    len(component.ket_active),
                    tuple(component.active_pairs),
                )
            selection = np.asarray(selection, dtype=np.int64)
            selections.append(selection)
            active_tuples.append(tuple(tuples))
            component_slices.append(
                slice(local_offset, local_offset + len(selection))
            )
            local_offset += len(selection)
        free = _free_tuples(key, eris)
        raw_dimension = int(local_offset * len(free))
        layout = _SectorLayout(
            key=key,
            components=components,
            free_tuples=free,
            active_selections=tuple(selections),
            active_tuples=tuple(active_tuples),
            component_slices=tuple(component_slices),
            local_dimension=int(local_offset),
            raw_dimension=raw_dimension,
            raw_slice=slice(raw_offset, raw_offset + raw_dimension),
        )
        layouts.append(layout)
        raw_offset += raw_dimension
    return tuple(layouts)


def _fixed_indices(
    bra_layout: _SectorLayout,
    bra_free: Sequence[int],
    ket_layout: _SectorLayout,
    ket_free: Sequence[int],
) -> dict[str, int]:
    fixed = {}
    if bra_layout.key != "ref":
        fixed.update(
            {
                _FREE_BRA_LABEL[label]: int(value)
                for label, value in zip(bra_layout.key, bra_free)
            }
        )
    if ket_layout.key != "ref":
        fixed.update(
            {
                label: int(value)
                for label, value in zip(ket_layout.key, ket_free)
            }
        )
    return fixed


def _selected_component_matrix(
    tensor,
    bra_selection: np.ndarray,
    ket_selection: np.ndarray,
    bra_rank: int,
    ket_rank: int,
    ncas: int,
) -> np.ndarray:
    nbra = int(ncas) ** int(bra_rank)
    nket = int(ncas) ** int(ket_rank)
    matrix = np.asarray(tensor).reshape(nbra, nket)
    return matrix[np.ix_(bra_selection, ket_selection)]


def _evaluate_local_matrix(
    evaluator: _ExpressionEvaluator,
    equations: _EquationBundle,
    bra_layout: _SectorLayout,
    bra_free: Sequence[int],
    ket_layout: _SectorLayout,
    ket_free: Sequence[int],
    *,
    metric: bool,
) -> np.ndarray:
    if metric and bra_layout.key != ket_layout.key:
        return np.zeros(
            (bra_layout.local_dimension, ket_layout.local_dimension),
            dtype=evaluator.dtype,
        )
    if (
        not metric
        and bra_layout.key == "ref"
        and ket_layout.key == "ref"
    ):
        return np.zeros((1, 1), dtype=evaluator.dtype)

    result = np.zeros(
        (bra_layout.local_dimension, ket_layout.local_dimension),
        dtype=evaluator.dtype,
    )
    fixed = _fixed_indices(
        bra_layout, bra_free, ket_layout, ket_free
    )
    for bra_index, bra_component in enumerate(bra_layout.components):
        for ket_index, ket_component in enumerate(ket_layout.components):
            if metric:
                pair = (
                    bra_layout.key,
                    bra_component.name,
                    ket_component.name,
                )
                expression = equations.metric_expr[pair]
            else:
                pair = (
                    bra_layout.key,
                    bra_component.name,
                    ket_layout.key,
                    ket_component.name,
                )
                expression = equations.hamiltonian_expr[pair]
            output = tuple(bra_component.bra_active) + tuple(
                ket_component.ket_active
            )
            tensor = evaluator.evaluate(expression, output, fixed)
            block = _selected_component_matrix(
                tensor,
                bra_layout.active_selections[bra_index],
                ket_layout.active_selections[ket_index],
                len(bra_component.bra_active),
                len(ket_component.ket_active),
                int(evaluator.eris.ncas),
            )
            result[
                bra_layout.component_slices[bra_index],
                ket_layout.component_slices[ket_index],
            ] = block
    return result


def _maximum_abs(values) -> float:
    return float(np.max(np.abs(np.asarray(values)), initial=0.0))


def _canonical_orthonormality_tolerance(
    matrix_atol: float,
    matrix_rtol: float,
    retained_condition_number: float,
    dtype,
) -> tuple[float, float, float]:
    """Return effective, roundoff, and user tolerances for whitening.

    For a backward-stable Hermitian eigensolve, forming ``X^H S X`` after
    canonical whitening has a first-order error proportional to machine
    epsilon times the condition number of the retained metric.  A rank-based
    allowance is deliberately excluded: it could hide a defective
    orthogonalizer merely because a sector is large.
    """

    user_tolerance = float(matrix_atol + matrix_rtol)
    condition = float(retained_condition_number)
    if not np.isfinite(condition) or condition < 1.0:
        raise ValueError(
            "a nonempty retained metric must have a finite condition number "
            "greater than or equal to one"
        )
    real_dtype = np.empty((), dtype=np.dtype(dtype)).real.dtype
    machine_epsilon = float(np.finfo(real_dtype).eps)
    roundoff_bound = float(
        _CANONICAL_ORTHOGONALIZATION_ROUNDOFF_FACTOR
        * machine_epsilon
        * condition
    )
    if not np.isfinite(roundoff_bound) or roundoff_bound >= 1.0:
        raise FloatingPointError(
            "the retained metric is too ill-conditioned for reliable "
            "canonical orthogonalization"
        )
    return (
        max(user_tolerance, roundoff_bound),
        roundoff_bound,
        machine_epsilon,
    )


def _audit_projection_residual(
    name: str,
    residual: float,
    tolerance: float,
    *,
    projection: str,
) -> float:
    """Apply the common normal/warn/hard policy before a numerical projection."""

    residual = float(residual)
    tolerance = float(tolerance)
    hard_tolerance = float(1.0e3 * tolerance)
    if not np.isfinite(residual) or residual > hard_tolerance:
        raise FloatingPointError(
            f"{name} residual {residual:.3e} exceeds the hard limit "
            f"{hard_tolerance:.3e}"
        )
    if residual > tolerance:
        _utils._warn_numerical(
            f"{name} residual {residual:.3e} exceeds {tolerance:.3e}; "
            f"{projection}"
        )
    return hard_tolerance


def _finite_nonnegative(value, *, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def _metric_space(
    layout: _SectorLayout,
    evaluator: _ExpressionEvaluator,
    equations: _EquationBundle,
    *,
    metric_atol: float,
    metric_rtol: float,
    metric_rcond: float,
    matrix_atol: float,
    matrix_rtol: float,
) -> _MetricSpace:
    if layout.key == "ref":
        metric = np.ones((1, 1), dtype=evaluator.dtype)
        orthogonalizer = np.ones((1, 1), dtype=evaluator.dtype)
        diagnostics = {
            "sector": "ref",
            "free_block_count": 1,
            "local_raw_dimension": 1,
            "raw_dimension": 1,
            "local_rank": 1,
            "rank": 1,
            "metric_minimum_eigenvalue": 1.0,
            "metric_maximum_eigenvalue": 1.0,
            "metric_cutoff": 0.0,
            "retained_condition_number": 1.0,
            "metric_hermiticity_error": 0.0,
            "orthonormality_error": 0.0,
            "repeated_block_error": 0.0,
        }
        return _MetricSpace(
            layout, metric, orthogonalizer, np.ones(1), 0.0, diagnostics
        )

    if not layout.free_tuples or layout.local_dimension == 0:
        metric = np.zeros(
            (layout.local_dimension, layout.local_dimension),
            dtype=evaluator.dtype,
        )
        orthogonalizer = np.zeros(
            (layout.local_dimension, 0), dtype=evaluator.dtype
        )
        diagnostics = {
            "sector": layout.key,
            "free_block_count": 0,
            "local_raw_dimension": layout.local_dimension,
            "raw_dimension": 0,
            "local_rank": 0,
            "rank": 0,
            "metric_minimum_eigenvalue": 0.0,
            "metric_maximum_eigenvalue": 0.0,
            "metric_cutoff": float(metric_atol),
            "retained_condition_number": math.inf,
            "metric_hermiticity_error": 0.0,
            "orthonormality_error": 0.0,
            "repeated_block_error": 0.0,
        }
        return _MetricSpace(
            layout,
            metric,
            orthogonalizer,
            np.empty(0),
            float(metric_atol),
            diagnostics,
        )

    representative = layout.free_tuples[0]
    metric = _evaluate_local_matrix(
        evaluator,
        equations,
        layout,
        representative,
        layout,
        representative,
        metric=True,
    )
    metric_hermiticity_error = _maximum_abs(metric - metric.conj().T)
    metric_scale = max(1.0, _maximum_abs(metric))
    metric_hermiticity_tolerance = float(
        matrix_atol + matrix_rtol * metric_scale
    )
    if metric_hermiticity_error > metric_hermiticity_tolerance:
        if metric_hermiticity_error > 1.0e3 * metric_hermiticity_tolerance:
            raise FloatingPointError(
                f"FIC-MRCISD sector {layout.key} metric Hermiticity residual "
                f"{metric_hermiticity_error:.3e} is too large to project "
                f"(hard limit {1.0e3 * metric_hermiticity_tolerance:.3e})"
            )
        _utils._warn_numerical(
            f"FIC-MRCISD sector {layout.key} metric Hermiticity residual "
            f"{metric_hermiticity_error:.3e} exceeds "
            f"{metric_hermiticity_tolerance:.3e}"
        )
    metric_h = 0.5 * (metric + metric.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(metric_h)
    maximum = float(np.max(eigenvalues, initial=0.0))
    minimum = float(eigenvalues[0])
    negative_tolerance = float(
        metric_atol + metric_rtol * max(1.0, abs(maximum))
    )
    if minimum < -negative_tolerance:
        raise FloatingPointError(
            f"FIC-MRCISD sector {layout.key} metric has a significant "
            f"negative eigenvalue {minimum:.3e} (limit {-negative_tolerance:.3e})"
        )
    cutoff = float(max(metric_atol, metric_rcond * max(maximum, 0.0)))
    retained = eigenvalues > cutoff
    retained_values = eigenvalues[retained]
    condition = (
        float(retained_values[-1] / retained_values[0])
        if retained_values.size
        else math.inf
    )
    orthogonalizer = eigenvectors[:, retained] / np.sqrt(
        retained_values
    )[None, :]
    identity = orthogonalizer.conj().T @ metric_h @ orthogonalizer
    orthonormality_error = _maximum_abs(
        identity - np.eye(identity.shape[0], dtype=identity.dtype)
    )
    orthonormality_user_tolerance = float(matrix_atol + matrix_rtol)
    if retained_values.size:
        (
            orthonormality_tolerance,
            orthonormality_roundoff_bound,
            orthonormality_machine_epsilon,
        ) = _canonical_orthonormality_tolerance(
            matrix_atol,
            matrix_rtol,
            condition,
            metric_h.dtype,
        )
    else:
        orthonormality_tolerance = orthonormality_user_tolerance
        orthonormality_roundoff_bound = 0.0
        orthonormality_machine_epsilon = float(
            np.finfo(metric_h.real.dtype).eps
        )
    if orthonormality_error > orthonormality_tolerance:
        raise FloatingPointError(
            f"FIC-MRCISD sector {layout.key} canonical metric "
            f"orthonormalization failed: {orthonormality_error:.3e} "
            f"exceeds {orthonormality_tolerance:.3e}"
        )

    repeated_block_error = 0.0
    if len(layout.free_tuples) > 1:
        last = layout.free_tuples[-1]
        repeated = _evaluate_local_matrix(
            evaluator,
            equations,
            layout,
            last,
            layout,
            last,
            metric=True,
        )
        repeated_block_error = _maximum_abs(repeated - metric)
        if repeated_block_error > metric_hermiticity_tolerance:
            raise FloatingPointError(
                f"FIC-MRCISD sector {layout.key} metric depends on the "
                f"fixed external tuple ({repeated_block_error:.3e})"
            )

    local_rank = int(np.count_nonzero(retained))
    diagnostics = {
        "sector": layout.key,
        "free_block_count": len(layout.free_tuples),
        "local_raw_dimension": layout.local_dimension,
        "raw_dimension": layout.raw_dimension,
        "component_dimensions": {
            component.name: int(len(selection))
            for component, selection in zip(
                layout.components, layout.active_selections
            )
        },
        "local_rank": local_rank,
        "rank": int(local_rank * len(layout.free_tuples)),
        "metric_eigenvalues": [float(value) for value in eigenvalues],
        "metric_minimum_eigenvalue": minimum,
        "metric_maximum_eigenvalue": maximum,
        "metric_negative_tolerance": negative_tolerance,
        "metric_cutoff": cutoff,
        "retained_condition_number": condition,
        "metric_hermiticity_error": metric_hermiticity_error,
        "metric_hermiticity_tolerance": metric_hermiticity_tolerance,
        "orthonormality_error": orthonormality_error,
        "orthonormality_tolerance": orthonormality_tolerance,
        "orthonormality_user_tolerance": orthonormality_user_tolerance,
        "orthonormality_roundoff_bound": orthonormality_roundoff_bound,
        "orthonormality_machine_epsilon": (
            orthonormality_machine_epsilon
        ),
        "repeated_block_error": repeated_block_error,
        "metric_block_structure": "identical direct sum over fixed external tuples",
    }
    return _MetricSpace(
        layout,
        metric_h,
        np.asarray(orthogonalizer),
        np.asarray(eigenvalues),
        cutoff,
        diagnostics,
    )


def _external_signature(
    key: str, free_indices: Sequence[int]
) -> tuple[frozenset[int], frozenset[int]]:
    if key == "ref":
        return frozenset(), frozenset()
    holes = frozenset(
        int(value)
        for label, value in zip(key, free_indices)
        if label in "ij"
    )
    particles = frozenset(
        int(value)
        for label, value in zip(key, free_indices)
        if label in "rs"
    )
    return holes, particles


def _minimum_external_excitation_rank(
    bra: _OrthogonalBlock, ket: _OrthogonalBlock
) -> int:
    """Necessary Slater-Condon rank between two external occupations."""

    bra_only = len(ket.holes - bra.holes) + len(
        bra.particles - ket.particles
    )
    ket_only = len(bra.holes - ket.holes) + len(
        ket.particles - bra.particles
    )
    return max(bra_only, ket_only)


def _build_orthogonal_blocks(
    metric_spaces: Sequence[_MetricSpace],
) -> tuple[tuple[_OrthogonalBlock, ...], int]:
    blocks = []
    orth_offset = 0
    for space in metric_spaces:
        layout = space.layout
        local_rank = int(space.orthogonalizer.shape[1])
        if local_rank == 0:
            continue
        for block_index, free_indices in enumerate(layout.free_tuples):
            raw_start = layout.raw_slice.start + block_index * layout.local_dimension
            holes, particles = _external_signature(layout.key, free_indices)
            blocks.append(
                _OrthogonalBlock(
                    layout=layout,
                    free_indices=free_indices,
                    block_index=block_index,
                    raw_slice=slice(
                        raw_start, raw_start + layout.local_dimension
                    ),
                    orth_slice=slice(orth_offset, orth_offset + local_rank),
                    orthogonalizer=space.orthogonalizer,
                    holes=holes,
                    particles=particles,
                )
            )
            orth_offset += local_rank
    return tuple(blocks), int(orth_offset)


def _matrix_memory_estimate(
    raw_dimension: int,
    orth_dimension: int,
    nroots: int,
    *,
    retain_raw_matrices: bool,
    dtype=np.complex128,
) -> dict[str, int]:
    itemsize = np.dtype(dtype).itemsize
    h_orth = itemsize * orth_dimension * orth_dimension
    # The unsymmetrized matrix, Hermitian projection temporaries, LAPACK
    # working copy/output, and a conservative implementation margin.  This
    # intentionally overestimates ordinary runs rather than allowing a dense
    # eigensystem to pass preflight and then be killed by the allocator.
    eigensystem = 6 * h_orth
    raw = (
        2 * itemsize * raw_dimension * raw_dimension
        if retain_raw_matrices
        else 0
    )
    selected_vectors = itemsize * raw_dimension * max(1, nroots)
    total = eigensystem + raw + selected_vectors
    return {
        "raw_dimension": int(raw_dimension),
        "orthogonal_dimension": int(orth_dimension),
        "h_orth_bytes": int(h_orth),
        "eigensystem_bytes": int(eigensystem),
        "raw_matrix_bytes": int(raw),
        "selected_vector_bytes": int(selected_vectors),
        "estimated_peak_bytes": int(total),
    }


def _effective_memory_limit(calculation, mc) -> int:
    if calculation.matrix_memory_limit is not None:
        value = int(calculation.matrix_memory_limit)
    else:
        max_memory_mb = float(
            getattr(mc, "max_memory", getattr(calculation.mol, "max_memory", 2000.0))
        )
        current_memory_mb = float(lib.current_memory()[0])
        value = int((max_memory_mb - current_memory_mb) * 1_000_000)
    if value <= 0:
        raise MemoryError(
            "no positive memory budget remains for the dense MRCISD matrix"
        )
    return value


def _assemble_raw_metric(
    metric_spaces: Sequence[_MetricSpace], raw_dimension: int
) -> np.ndarray:
    dtype = np.result_type(
        *(space.metric.dtype for space in metric_spaces)
    )
    result = np.zeros((raw_dimension, raw_dimension), dtype=dtype)
    for space in metric_spaces:
        layout = space.layout
        for block_index in range(len(layout.free_tuples)):
            start = layout.raw_slice.start + block_index * layout.local_dimension
            block_slice = slice(start, start + layout.local_dimension)
            result[block_slice, block_slice] = space.metric
    return result


def _sector_pair_has_equation(
    equations: _EquationBundle, bra_key: str, ket_key: str
) -> bool:
    if bra_key == ket_key == "ref":
        return False
    return any(
        len(
            equations.hamiltonian_expr[
                bra_key,
                bra_component.name,
                ket_key,
                ket_component.name,
            ].terms
        )
        for bra_component in _components(bra_key)
        for ket_component in _components(ket_key)
    )


def _assemble_shifted_hamiltonian(
    evaluator: _ExpressionEvaluator,
    equations: _EquationBundle,
    blocks: Sequence[_OrthogonalBlock],
    raw_dimension: int,
    orth_dimension: int,
    *,
    retain_raw: bool,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    h_orth = np.zeros(
        (orth_dimension, orth_dimension), dtype=evaluator.dtype
    )
    raw_h = (
        np.zeros(
            (raw_dimension, raw_dimension), dtype=evaluator.dtype
        )
        if retain_raw
        else None
    )
    sector_nonzero = {
        (bra_key, ket_key): _sector_pair_has_equation(
            equations, bra_key, ket_key
        )
        for bra_key in SECTOR_ORDER
        for ket_key in SECTOR_ORDER
    }
    evaluated_block_pairs = 0
    structural_zero_block_pairs = 0
    slater_condon_zero_block_pairs = 0
    maximum_raw_block_element = 0.0
    for bra_block in blocks:
        for ket_block in blocks:
            pair_key = (bra_block.layout.key, ket_block.layout.key)
            if not sector_nonzero[pair_key]:
                structural_zero_block_pairs += 1
                continue
            if _minimum_external_excitation_rank(bra_block, ket_block) > 2:
                slater_condon_zero_block_pairs += 1
                continue
            raw_block = _evaluate_local_matrix(
                evaluator,
                equations,
                bra_block.layout,
                bra_block.free_indices,
                ket_block.layout,
                ket_block.free_indices,
                metric=False,
            )
            evaluated_block_pairs += 1
            maximum_raw_block_element = max(
                maximum_raw_block_element, _maximum_abs(raw_block)
            )
            if raw_h is not None:
                raw_h[bra_block.raw_slice, ket_block.raw_slice] = raw_block
            transformed = (
                bra_block.orthogonalizer.conj().T
                @ raw_block
                @ ket_block.orthogonalizer
            )
            h_orth[bra_block.orth_slice, ket_block.orth_slice] = transformed
    return h_orth, raw_h, {
        "total_ordered_block_pairs": int(len(blocks) ** 2),
        "evaluated_ordered_block_pairs": int(evaluated_block_pairs),
        "symbolic_zero_ordered_block_pairs": int(structural_zero_block_pairs),
        "slater_condon_zero_ordered_block_pairs": int(
            slater_condon_zero_block_pairs
        ),
        "maximum_raw_block_element": float(maximum_raw_block_element),
    }


def _raw_coefficients(
    blocks: Sequence[_OrthogonalBlock],
    orthogonal_vectors: np.ndarray,
    raw_dimension: int,
) -> np.ndarray:
    orthogonal_vectors = np.asarray(orthogonal_vectors)
    result = np.zeros(
        (raw_dimension, orthogonal_vectors.shape[1]),
        dtype=np.result_type(
            orthogonal_vectors.dtype,
            *(block.orthogonalizer.dtype for block in blocks),
        ),
    )
    for block in blocks:
        result[block.raw_slice] = (
            block.orthogonalizer @ orthogonal_vectors[block.orth_slice]
        )
    return result


def _physical_norms(
    blocks: Sequence[_OrthogonalBlock],
    metric_spaces: dict[str, _MetricSpace],
    coefficients: np.ndarray,
) -> np.ndarray:
    norms = np.zeros(
        coefficients.shape[1],
        dtype=np.result_type(
            coefficients.dtype,
            *(space.metric.dtype for space in metric_spaces.values()),
        ),
    )
    for block in blocks:
        local = coefficients[block.raw_slice]
        metric = metric_spaces[block.layout.key].metric
        norms += np.einsum(
            "ik,ij,jk->k", local.conj(), metric, local, optimize=True
        )
    return norms


def _align_degenerate_eigenspaces(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    reference_overlap: np.ndarray,
    *,
    atol: float,
    rtol: float,
):
    """Choose the reference-following gauge inside numerical degeneracies."""

    values = np.array(eigenvalues, dtype=float, copy=True)
    vectors = np.array(eigenvectors, copy=True)
    clusters = []
    start = 0
    while start < len(values):
        stop = start + 1
        scale = max(1.0, abs(values[start]))
        tolerance = float(atol + rtol * scale)
        while stop < len(values) and values[stop] - values[start] <= tolerance:
            stop += 1
        if stop - start > 1:
            block = vectors[:, start:stop]
            amplitudes = reference_overlap @ block
            norm = float(np.linalg.norm(amplitudes))
            spread = float(values[stop - 1] - values[start])
            rotated = False
            if norm > np.finfo(float).eps:
                direction = amplitudes.conj() / norm
                seed = np.column_stack(
                    (
                        direction,
                        np.eye(stop - start, dtype=vectors.dtype),
                    )
                )
                rotation, _ = np.linalg.qr(seed)
                vectors[:, start:stop] = block @ rotation
                # Treat a cluster narrower than the declared numerical
                # degeneracy threshold as one eigenspace.  The resulting
                # residual is explicitly audited below and bounded by spread.
                values[start:stop] = float(np.mean(values[start:stop]))
                rotated = True
            clusters.append(
                {
                    "start": int(start),
                    "stop": int(stop),
                    "size": int(stop - start),
                    "energy_spread": spread,
                    "reference_subspace_weight": float(norm**2),
                    "reference_aligned": rotated,
                }
            )
        start = stop
    return values, vectors, clusters


def _reference_stationarity_audit(
    evaluator: _ExpressionEvaluator,
    equations: _EquationBundle,
    mc,
    reference_energy: float,
    *,
    residual_warn: float,
    residual_error: float,
    energy_atol: float,
) -> dict[str, Any]:
    r"""Audit the projected-CAS eigenstate condition through ``Var(H_act)``.

    The same-sector four-RDM commutator organization is a physical shifted
    expectation only for a stationary CAS reference.  The active Hamiltonian
    variance uses the supplied raw dm1--dm4 and is therefore available for
    exact CI and DMRG references without access to their internal CI/MPS
    representation.
    """

    first = complex(
        np.asarray(
            evaluator.evaluate(
                equations.reference_first_moment, (), {}
            )
        ).item()
    )
    second = complex(
        np.asarray(
            evaluator.evaluate(
                equations.reference_second_moment, (), {}
            )
        ).item()
    )
    variance = second - first.conjugate() * first
    variance_scale = max(1.0, abs(second), abs(first) ** 2)
    roundoff = 100.0 * np.finfo(float).eps * variance_scale
    if variance.real < -roundoff or abs(variance.imag) > roundoff:
        raise FloatingPointError(
            "reference active-Hamiltonian variance is not a finite "
            f"non-negative real number: {variance!r}"
        )
    residual = float(np.sqrt(max(0.0, variance.real)))
    if residual > residual_error:
        raise FloatingPointError(
            "the supplied reference is not a sufficiently stationary CAS "
            f"root: ||(Hact-<Hact>)Psi||={residual:.3e} exceeds the hard "
            f"limit {residual_error:.3e} Eh"
        )
    if residual > residual_warn:
        _utils._warn_numerical(
            "finite reference stationarity residual "
            f"||(Hact-<Hact>)Psi||={residual:.3e} Eh exceeds "
            f"{residual_warn:.3e} Eh"
        )

    reconstructed = None
    reconstruction_error = None
    molecule = getattr(mc, "mol", getattr(getattr(mc, "_scf", None), "mol", None))
    nuclear_energy = getattr(molecule, "energy_nuc", None)
    if callable(nuclear_energy):
        core = slice(0, int(evaluator.eris.ncore))
        h_core = evaluator.eris.h1e[core, core]
        w_core = evaluator.eris.get_phys("IIII")
        core_energy = np.trace(h_core)
        core_energy += 0.5 * (
            np.einsum("ijij->", w_core, optimize=True)
            - np.einsum("ijji->", w_core, optimize=True)
        )
        reconstructed = complex(nuclear_energy()) + core_energy + first
        reconstruction_error = float(abs(reconstructed - reference_energy))
        if reconstruction_error > 1.0e3 * energy_atol:
            raise FloatingPointError(
                "reference energy reconstructed from bare integrals and "
                f"raw RDMs differs by {reconstruction_error:.3e} Eh "
                f"(hard limit {1.0e3 * energy_atol:.3e} Eh)"
            )
        if reconstruction_error > energy_atol:
            _utils._warn_numerical(
                "reference energy reconstructed from bare integrals and "
                f"raw RDMs differs by {reconstruction_error:.3e} Eh"
            )

    return {
        "active_hamiltonian_first_moment": [first.real, first.imag],
        "active_hamiltonian_second_moment": [second.real, second.imag],
        "active_hamiltonian_variance": [variance.real, variance.imag],
        "active_eigen_residual_norm": residual,
        "residual_warning_threshold": float(residual_warn),
        "residual_hard_threshold": float(residual_error),
        "stationarity_gate_passed": bool(residual <= residual_error),
        "reference_energy_reconstructed": (
            None
            if reconstructed is None
            else [reconstructed.real, reconstructed.imag]
        ),
        "reference_energy_reconstruction_abs_error": reconstruction_error,
        "reference_energy_tolerance": float(energy_atol),
    }


def _permutation_sign(values: Sequence[int]) -> int:
    inversions = sum(
        values[left] > values[right]
        for left in range(len(values))
        for right in range(left + 1, len(values))
    )
    return -1 if inversions % 2 else 1


@lru_cache(maxsize=None)
def _ordered_tuple_map(ncas: int, rank: int, reverse: bool):
    combinations = tuple(itertools.combinations(range(ncas), rank))
    combination_index = {
        values: index for index, values in enumerate(combinations)
    }
    indices = np.full(ncas**rank, -1, dtype=np.int64)
    signs = np.zeros(ncas**rank, dtype=np.int8)
    for flat, values in enumerate(
        itertools.product(range(ncas), repeat=rank)
    ):
        if len(set(values)) != rank:
            continue
        ordered = tuple(reversed(values)) if reverse else values
        indices[flat] = combination_index[tuple(sorted(ordered))]
        signs[flat] = _permutation_sign(ordered)
    return combinations, indices, signs


def _exact_fci_raw_rdm(
    coefficients: np.ndarray,
    ncas: int,
    nelec: int,
    rank: int,
    *,
    row_chunk: int = 256,
) -> np.ndarray:
    """Form one raw SGF RDM from a complete spinor-FCI coefficient vector."""

    shape = (ncas,) * (2 * rank)
    if rank > nelec or rank > ncas:
        return np.zeros(shape, dtype=np.complex128)
    from pyscf.fci import cistring

    combinations, creator_indices, creator_signs = _ordered_tuple_map(
        ncas, rank, True
    )
    _, annihilator_indices, annihilator_signs = _ordered_tuple_map(
        ncas, rank, False
    )
    determinant_strings = np.asarray(
        cistring.make_strings(range(ncas), nelec), dtype=np.int64
    )
    remaining_strings = np.asarray(
        cistring.make_strings(range(ncas), nelec - rank), dtype=np.int64
    )
    compact = np.zeros(
        (len(combinations), len(remaining_strings)), dtype=np.complex128
    )
    combination_index = {
        values: index for index, values in enumerate(combinations)
    }
    for determinant_index, (state, amplitude) in enumerate(
        zip(determinant_strings, coefficients)
    ):
        if amplitude == 0:
            continue
        occupied = tuple(
            orbital for orbital in range(ncas) if int(state) >> orbital & 1
        )
        for removed in itertools.combinations(occupied, rank):
            reduced_state = int(state)
            phase = 1
            for orbital in reversed(removed):
                mask = 1 << orbital
                phase *= (
                    -1 if (reduced_state & (mask - 1)).bit_count() % 2 else 1
                )
                reduced_state ^= mask
            remaining_index = cistring.str2addr(
                ncas, nelec - rank, reduced_state
            )
            compact[combination_index[removed], remaining_index] = (
                phase * coefficients[determinant_index]
            )
    gram = compact.conj() @ compact.T
    creator_rows = np.flatnonzero(creator_indices >= 0)
    annihilator_columns = np.flatnonzero(annihilator_indices >= 0)
    density = np.zeros((ncas**rank, ncas**rank), dtype=np.complex128)
    for start in range(0, creator_rows.size, int(row_chunk)):
        rows = creator_rows[start : start + int(row_chunk)]
        block = gram[
            creator_indices[rows, None],
            annihilator_indices[annihilator_columns][None, :],
        ]
        block *= creator_signs[rows, None]
        block *= annihilator_signs[annihilator_columns][None, :]
        density[rows[:, None], annihilator_columns[None, :]] = block
    return density.reshape(shape)


def _exact_fci_pdms(ci, ncas: int, nelec: int):
    """Return raw dm1--dm4 for a full spinor-FCI vector in cistring order."""

    from pyscf.fci import cistring

    coefficients = np.asarray(ci)
    expected = int(cistring.num_strings(int(ncas), int(nelec)))
    if coefficients.ndim != 1 or coefficients.size != expected:
        raise ValueError(
            "the automatic exact-FCI RDM path requires a one-dimensional "
            f"complete spinor-CI vector of length {expected}"
        )
    return tuple(
        _exact_fci_raw_rdm(coefficients, int(ncas), int(nelec), rank)
        for rank in range(1, 5)
    )


def _reference_pdms(mc, root: int):
    """Dispatch between the retained Block2 MPS and exact spinor-FCI paths."""

    solver = mc.fcisolver
    if getattr(solver, "driver", None) is not None and getattr(
        solver, "kets", None
    ) is not None:
        return _utils.make_dm1234(solver, root=root)

    ci = getattr(mc, "ci", None)
    if ci is None:
        ci = getattr(solver, "ci", None)
    if ci is None:
        raise RuntimeError(
            "automatic 1--4 RDM construction requires a retained Block2 "
            "MPS or mc.ci; alternatively pass raw SGF pdms explicitly"
        )
    selected_ci = _utils._select_root_ci(
        ci, root, nroots=int(getattr(solver, "nroots", 1))
    )
    nelec_source = getattr(solver, "nelecas", None)
    if nelec_source is None:
        nelec_source = mc.nelecas
    ncas = int(mc.ncas)
    estimated_rdm_bytes = int(
        np.dtype(np.complex128).itemsize
        * sum(ncas ** (2 * rank) for rank in range(1, 5))
    )
    max_memory_mb = float(
        getattr(mc, "max_memory", getattr(mc._scf.mol, "max_memory", 2000.0))
    )
    available_bytes = int(
        max(0.0, max_memory_mb - float(lib.current_memory()[0])) * 1_000_000
    )
    if estimated_rdm_bytes > available_bytes:
        raise MemoryError(
            "automatic exact-FCI raw dm1--dm4 storage estimate "
            f"{estimated_rdm_bytes / 2**30:.2f} GiB exceeds the available "
            f"PySCF memory budget {available_bytes / 2**30:.2f} GiB"
        )
    return _exact_fci_pdms(
        selected_ci,
        ncas,
        _utils._total_nelec(nelec_source),
    )


class WickX2CICMRCISD(lib.StreamObject):
    """State-specific general-complex-spinor fully IC MRCISD driver."""

    def __init__(self, mc, frozen=0):
        if _utils._has_frozen_orbitals(frozen) or _utils._has_frozen_orbitals(
            getattr(mc, "frozen", None)
        ):
            raise NotImplementedError(
                "nonzero frozen spinors are outside X2C-FIC-MRCISD v1"
            )
        self._mc = mc
        self._scf = mc._scf
        self.mol = self._scf.mol
        self.verbose = getattr(mc, "verbose", self.mol.verbose)
        self.stdout = getattr(mc, "stdout", self.mol.stdout)

        self.root = 0
        self.nroots = 1
        self.state_selection = "max_reference_weight"
        self.contraction_backend = _utils._DEFAULT_CONTRACTION_BACKEND
        self.metric_atol = 1.0e-12
        self.metric_rtol = 1.0e-10
        self.metric_rcond = 1.0e-10
        self.matrix_atol = 1.0e-10
        self.matrix_rtol = 1.0e-9
        self.reference_weight_tol = 1.0e-12
        self.reference_residual_warn = 1.0e-7
        self.reference_residual_error = 1.0e-2
        self.reference_energy_atol = 1.0e-8
        self.eigenvalue_degeneracy_atol = 1.0e-10
        self.eigenvalue_degeneracy_rtol = 1.0e-12
        self.rdm_atol = _utils._DEFAULT_RDM_ATOL
        self.rdm_rtol = _utils._DEFAULT_RDM_RTOL
        self.rdm_work_memory = _utils._DEFAULT_RDM_WORK_MEMORY
        self.integral_roundoff_factor = _utils._DEFAULT_AO2MO_ROUNDOFF_FACTOR
        self.matrix_memory_limit = None
        self.retain_raw_matrices = False
        self.frozen = 0

        self.mo_coeff = getattr(mc, "mo_coeff", None)
        self.eris = None
        self.eris_basis = None
        self._clear_results()
        self._keys = set(self.__dict__)

    def run(self, *args, **kwargs):
        """Run :meth:`kernel` and retain PySCF's fluent object interface.

        PySCF's generic ``StreamObject.run`` turns keyword arguments into
        attributes and then calls ``kernel`` without forwarding them.  RDMs,
        ERIs, a replacement CASSCF object, and its MO basis are per-call
        inputs here, so silently dropping them would change the calculation.
        """

        kernel_kwargs = {
            name: value
            for name, value in kwargs.items()
            if name in _KERNEL_KEYWORDS
        }
        setting_kwargs = {
            name: value
            for name, value in kwargs.items()
            if name not in _KERNEL_KEYWORDS or name in self._keys
        }
        self.set(**setting_kwargs)
        self.kernel(*args, **kernel_kwargs)
        return self

    @property
    def e_tot(self):
        if self.e_corr is None:
            return None
        return float(self.reference_energy + self.e_corr)

    @property
    def e_tot_q(self):
        if self.e_tot is None or self.de_dav_q is None:
            return None
        return float(self.e_tot + self.de_dav_q)

    def _clear_results(self):
        self.reference_energy = None
        self.e_states = None
        self.e_corr_states = None
        self.ci = None
        self.ci_orth = None
        self.reference_weights = None
        self.selected_state = None
        self.state_indices = None
        self.e_corr = None
        self.de_dav_q_states = None
        self.e_states_q = None
        self.de_dav_q = None
        self.sector_diagnostics = {}
        self.matrix_diagnostics = {}
        self.rdm_diagnostics = None
        self.integral_symmetry_diagnostics = None
        self.sub_times = {}
        self.raw_metric = None
        self.raw_shifted_hamiltonian = None
        self.orthogonal_shifted_hamiltonian = None
        self.basis_layouts = None

    def _validate_controls(self, *, root, nroots, state_selection):
        if root < 0:
            raise IndexError("root must be non-negative")
        if nroots <= 0:
            raise ValueError("nroots must be positive")
        if state_selection not in ("max_reference_weight", "lowest"):
            raise ValueError(
                "state_selection must be 'max_reference_weight' or 'lowest'"
            )
        self.contraction_backend = _utils._normalize_contraction_backend(
            self.contraction_backend
        )
        for name in (
            "metric_atol",
            "metric_rtol",
            "metric_rcond",
            "matrix_atol",
            "matrix_rtol",
            "reference_weight_tol",
            "reference_residual_warn",
            "reference_residual_error",
            "reference_energy_atol",
            "eigenvalue_degeneracy_atol",
            "eigenvalue_degeneracy_rtol",
            "rdm_atol",
            "rdm_rtol",
        ):
            setattr(
                self,
                name,
                _finite_nonnegative(getattr(self, name), name=name),
            )
        if self.metric_atol == 0.0 and self.metric_rcond == 0.0:
            raise ValueError(
                "at least one metric absolute/relative cutoff must be positive"
            )
        if self.reference_weight_tol == 0.0:
            raise ValueError("reference_weight_tol must be positive")
        if self.reference_residual_error < self.reference_residual_warn:
            raise ValueError(
                "reference_residual_error must be at least "
                "reference_residual_warn"
            )
        if int(self.rdm_work_memory) <= 0:
            raise ValueError("rdm_work_memory must be positive")

    def _prepare_inputs(
        self,
        mc,
        *,
        root: int,
        mo_coeff,
        pdms,
        eris,
        eris_basis: str,
    ):
        mo_coeff = np.asarray(mo_coeff)
        if mo_coeff.ndim != 2:
            raise ValueError("mo_coeff must be a two-dimensional array")
        if pdms is None:
            reference_mo = np.asarray(getattr(mc, "mo_coeff", mo_coeff))
            if not np.array_equal(mo_coeff, reference_mo):
                raise ValueError(
                    "automatic reference RDMs are in mc.mo_coeff's active "
                    "basis; pass explicitly transformed raw SGF pdms when "
                    "supplying a different mo_coeff"
                )
            pdms = _reference_pdms(mc, root)
        nelec_source = getattr(mc.fcisolver, "nelecas", None)
        if nelec_source is None:
            nelec_source = mc.nelecas
        pdms, rdm_diagnostics = _utils.validate_pdms(
            pdms,
            int(mc.ncas),
            _utils._total_nelec(nelec_source),
            atol=self.rdm_atol,
            rtol=self.rdm_rtol,
            work_memory=self.rdm_work_memory,
        )
        if eris is None:
            eris = _utils._dense_eris_from_mc(
                mc,
                mo_coeff,
                roundoff_factor=self.integral_roundoff_factor,
            )
        elif not isinstance(eris, spinor_helper._SpinorERIs):
            raise TypeError("eris must be a spinor_helper._SpinorERIs")
        if (eris.ncore, eris.ncas, eris.nmo) != (
            int(mc.ncore),
            int(mc.ncas),
            int(mo_coeff.shape[1]),
        ):
            raise ValueError("eris partition does not match the CASSCF object")
        return mo_coeff, pdms, eris, rdm_diagnostics

    def kernel(
        self,
        mc=None,
        mo_coeff=None,
        pdms=None,
        eris=None,
        eris_basis="input_mo",
        root=None,
        nroots=None,
        state_selection=None,
    ):
        """Compute one reference-specific fully IC X2C-MRCISD spectrum."""

        total_start = time.perf_counter()
        self._clear_results()
        if mc is None:
            mc = self._mc
        if _utils._has_frozen_orbitals(getattr(mc, "frozen", None)):
            raise NotImplementedError(
                "nonzero frozen spinors are outside X2C-FIC-MRCISD v1"
            )
        solver_converged = getattr(getattr(mc, "fcisolver", None), "converged", None)
        if solver_converged is not None and not bool(
            np.all(np.asarray(solver_converged))
        ):
            raise RuntimeError("the selected CASSCF/DMRG reference is not converged")
        if root is None:
            root = self.root
        if nroots is None:
            nroots = self.nroots
        if state_selection is None:
            state_selection = self.state_selection
        root = int(root)
        nroots = int(nroots)
        state_selection = str(state_selection)
        eris_basis = _utils._normalize_eris_basis(eris_basis)
        if eris_basis != "input_mo":
            raise ValueError(
                "X2C-FIC-MRCISD does not semicanonicalize orbitals; pass "
                "mo_coeff and any supplied eris in the same basis with "
                "eris_basis='input_mo'"
            )
        self._validate_controls(
            root=root,
            nroots=nroots,
            state_selection=state_selection,
        )
        reference_energies = np.asarray(getattr(mc, "e_states", None))
        if reference_energies.ndim:
            available_reference_roots = len(reference_energies)
        else:
            available_reference_roots = int(
                getattr(getattr(mc, "fcisolver", None), "nroots", 1)
            )
        if root >= available_reference_roots:
            raise IndexError(
                f"reference root {root} is outside the available range "
                f"[0, {available_reference_roots})"
            )
        self.root = root
        self.nroots = nroots
        self.state_selection = state_selection
        if mo_coeff is None:
            mo_coeff = mc.mo_coeff

        input_start = time.perf_counter()
        mo_coeff, pdms, prepared_eris, rdm_diagnostics = self._prepare_inputs(
            mc,
            root=root,
            mo_coeff=mo_coeff,
            pdms=pdms,
            eris=eris,
            eris_basis=eris_basis,
        )
        self.sub_times["inputs"] = time.perf_counter() - input_start
        logger.info(self, "native MRCISD inputs ready in %.3f s",
                    self.sub_times["inputs"])
        self.mo_coeff = mo_coeff
        self.eris = prepared_eris
        self.eris_basis = "input_mo"
        self.reference_energy = _utils._reference_energy(mc, root)
        self.rdm_diagnostics = rdm_diagnostics
        self.integral_symmetry_diagnostics = getattr(
            prepared_eris, "symmetry_diagnostics", None
        )

        equation_start = time.perf_counter()
        equations = _compile_equations()
        self.sub_times["equations"] = time.perf_counter() - equation_start
        logger.info(self, "native MRCISD Wick equations ready in %.3f s",
                    self.sub_times["equations"])
        if equations.maximum_rdm_rank > 4:
            raise RuntimeError("generated MRCISD equations require >4-RDM")
        if self.contraction_backend == "pytblis":
            _utils._validate_tblis_operand_dtypes(
                [
                    (f"h{key}", prepared_eris.get_h1(key))
                    for key in equations.required_h1_blocks
                ]
                + [
                    (f"w{key}", prepared_eris.get_phys(key))
                    for key in equations.required_eri_blocks
                ]
                + [
                    (f"dm{rank}", density)
                    for rank, density in enumerate(pdms, start=1)
                ]
            )
        evaluator = _ExpressionEvaluator(
            prepared_eris, pdms, self.contraction_backend
        )
        reference_audit = _reference_stationarity_audit(
            evaluator,
            equations,
            mc,
            self.reference_energy,
            residual_warn=self.reference_residual_warn,
            residual_error=self.reference_residual_error,
            energy_atol=self.reference_energy_atol,
        )
        layouts = _make_layouts(prepared_eris)
        self.basis_layouts = layouts

        metric_start = time.perf_counter()
        metric_spaces = tuple(
            _metric_space(
                layout,
                evaluator,
                equations,
                metric_atol=self.metric_atol,
                metric_rtol=self.metric_rtol,
                metric_rcond=self.metric_rcond,
                matrix_atol=self.matrix_atol,
                matrix_rtol=self.matrix_rtol,
            )
            for layout in layouts
        )
        self.sub_times["metric"] = time.perf_counter() - metric_start
        self.sector_diagnostics = {
            space.layout.key: dict(space.diagnostics)
            for space in metric_spaces
        }
        metric_by_key = {space.layout.key: space for space in metric_spaces}
        blocks, orth_dimension = _build_orthogonal_blocks(metric_spaces)
        raw_dimension = int(sum(layout.raw_dimension for layout in layouts))
        if orth_dimension == 0:
            raise RuntimeError("FIC-MRCISD metric has no retained basis vectors")
        if nroots > orth_dimension:
            raise ValueError(
                f"nroots={nroots} exceeds retained MRCISD dimension "
                f"{orth_dimension}"
            )

        memory = _matrix_memory_estimate(
            raw_dimension,
            orth_dimension,
            nroots,
            retain_raw_matrices=bool(self.retain_raw_matrices),
            dtype=evaluator.dtype,
        )
        memory["matrix_dtype"] = evaluator.dtype.name
        memory_limit = _effective_memory_limit(self, mc)
        memory["limit_bytes"] = memory_limit
        memory["passed"] = bool(memory["estimated_peak_bytes"] <= memory_limit)
        if not memory["passed"]:
            raise MemoryError(
                "FIC-MRCISD dense eigensystem estimate "
                f"{memory['estimated_peak_bytes'] / 2**30:.2f} GiB exceeds "
                f"matrix_memory_limit={memory_limit / 2**30:.2f} GiB; "
                f"raw dimension={raw_dimension}, retained dimension={orth_dimension}"
            )

        matrix_start = time.perf_counter()
        logger.info(self, "native MRCISD assembling Hamiltonian: "
                    "raw=%d retained=%d blocks=%d dtype=%s",
                    raw_dimension, orth_dimension, len(blocks), evaluator.dtype.name)
        h_orth_raw, raw_h, assembly_diagnostics = (
            _assemble_shifted_hamiltonian(
                evaluator,
                equations,
                blocks,
                raw_dimension,
                orth_dimension,
                retain_raw=bool(self.retain_raw_matrices),
            )
        )
        self.sub_times["hamiltonian"] = time.perf_counter() - matrix_start
        raw_scale = max(1.0, _maximum_abs(h_orth_raw))
        raw_hermiticity_error = _maximum_abs(
            h_orth_raw - h_orth_raw.conj().T
        )
        raw_hermiticity_tolerance = float(
            self.matrix_atol + self.matrix_rtol * raw_scale
        )
        if raw_hermiticity_error > raw_hermiticity_tolerance:
            hard_hermiticity_tolerance = 1.0e3 * raw_hermiticity_tolerance
            if (
                not np.isfinite(raw_hermiticity_error)
                or raw_hermiticity_error > hard_hermiticity_tolerance
            ):
                raise FloatingPointError(
                    "FIC-MRCISD orthogonal Hamiltonian Hermiticity residual "
                    f"{raw_hermiticity_error:.3e} is too large to project "
                    f"(hard limit {hard_hermiticity_tolerance:.3e})"
                )
            _utils._warn_numerical(
                "FIC-MRCISD orthogonal Hamiltonian Hermiticity residual "
                f"{raw_hermiticity_error:.3e} exceeds "
                f"{raw_hermiticity_tolerance:.3e}; applying the audited "
                "Hermitian projection"
            )
        h_orth = 0.5 * (h_orth_raw + h_orth_raw.conj().T)
        self.orthogonal_shifted_hamiltonian = h_orth
        if self.retain_raw_matrices:
            self.raw_metric = _assemble_raw_metric(
                metric_spaces, raw_dimension
            )
            self.raw_shifted_hamiltonian = raw_h

        diagonal_start = time.perf_counter()
        logger.info(self, "native MRCISD Hamiltonian ready in %.3f s; "
                    "starting full eigensolve",
                    self.sub_times["hamiltonian"])
        eigenvalues, eigenvectors = linalg.eigh(
            h_orth,
            check_finite=False,
            overwrite_a=False,
        )
        self.sub_times["diagonalization"] = (
            time.perf_counter() - diagonal_start
        )
        # The first orthogonal basis vector is the explicit normalized
        # reference.  Form this as the physical metric overlap S[0,:] X v,
        # rather than assuming a real coefficient or squaring v[0].
        reference_overlap_in_orth = np.zeros(
            orth_dimension, dtype=eigenvectors.dtype
        )
        reference_block = blocks[0]
        if reference_block.layout.key != "ref":
            raise RuntimeError("the explicit reference is not first in the basis")
        reference_overlap_in_orth[reference_block.orth_slice] = (
            np.ones(1, dtype=eigenvectors.dtype)
            @ reference_block.orthogonalizer
        )
        eigenvalues, eigenvectors, degenerate_clusters = (
            _align_degenerate_eigenspaces(
                eigenvalues,
                eigenvectors,
                reference_overlap_in_orth,
                atol=self.eigenvalue_degeneracy_atol,
                rtol=self.eigenvalue_degeneracy_rtol,
            )
        )
        all_reference_amplitudes = (
            reference_overlap_in_orth @ eigenvectors
        )
        all_reference_weights = np.abs(all_reference_amplitudes) ** 2
        if state_selection == "lowest":
            selected_indices = np.arange(nroots, dtype=int)
        else:
            selected_indices = np.argsort(
                -all_reference_weights, kind="stable"
            )[:nroots]
        selected_vectors = eigenvectors[:, selected_indices]
        selected_eigenvalues = eigenvalues[selected_indices]
        eigen_residual = _maximum_abs(
            h_orth @ selected_vectors
            - selected_vectors * selected_eigenvalues[None, :]
        )
        eigen_residual_scale = max(
            1.0,
            _maximum_abs(h_orth),
            _maximum_abs(selected_eigenvalues),
        )
        eigen_residual_tolerance = float(
            self.matrix_atol + self.matrix_rtol * eigen_residual_scale
        )
        eigen_residual_hard_tolerance = _audit_projection_residual(
            "returned-state eigen-equation",
            eigen_residual,
            eigen_residual_tolerance,
            projection="retaining the declared numerical-degeneracy gauge",
        )
        physical_ci = _raw_coefficients(
            blocks, selected_vectors, raw_dimension
        )
        physical_norms = _physical_norms(
            blocks, metric_by_key, physical_ci
        )
        norm_error = _maximum_abs(physical_norms - 1.0)
        norm_tolerance = float(
            self.matrix_atol + self.matrix_rtol * max(1, nroots)
        )
        if norm_error > norm_tolerance:
            raise FloatingPointError(
                "physical FIC-MRCISD eigenvectors are not metric normalized: "
                f"maximum error={norm_error:.3e}"
            )

        self.state_indices = np.asarray(selected_indices, dtype=int)
        self.selected_state = int(selected_indices[0])
        self.ci_orth = np.asarray(selected_vectors)
        self.ci = physical_ci
        self.e_corr_states = np.asarray(
            selected_eigenvalues, dtype=float
        )
        self.e_states = self.reference_energy + self.e_corr_states
        self.reference_weights = np.asarray(
            all_reference_weights[selected_indices], dtype=float
        )
        self.e_corr = float(self.e_corr_states[0])

        de_dav_q = np.full(nroots, np.nan, dtype=float)
        davidson_reasons = []
        for state, (correlation, weight) in enumerate(
            zip(self.e_corr_states, self.reference_weights)
        ):
            if weight <= self.reference_weight_tol:
                davidson_reasons.append(
                    {
                        "returned_state": state,
                        "eigenstate_index": int(selected_indices[state]),
                        "defined": False,
                        "reason": "reference_weight_at_or_below_tolerance",
                        "reference_weight": float(weight),
                    }
                )
                continue
            de_dav_q[state] = float(
                correlation * (1.0 - weight) / weight
            )
            davidson_reasons.append(
                {
                    "returned_state": state,
                    "eigenstate_index": int(selected_indices[state]),
                    "defined": True,
                    "reason": None,
                    "reference_weight": float(weight),
                }
            )
        self.de_dav_q_states = de_dav_q
        self.e_states_q = self.e_states + de_dav_q
        self.de_dav_q = float(de_dav_q[0])

        self.matrix_diagnostics = {
            "equations": {
                "maximum_rdm_rank": equations.maximum_rdm_rank,
                "required_h1_blocks": list(equations.required_h1_blocks),
                "required_eri_blocks": list(equations.required_eri_blocks),
                "cross_sector_nonzero_overlap_equations": (
                    equations.cross_sector_overlap_count
                ),
                "same_sector_organization": "commutator",
                "cross_sector_organization": "full_expectation",
                "one_electron_integrals": "bare_h1e",
                "strictly_real_input_path": evaluator.strictly_real,
                "matrix_dtype": evaluator.dtype.name,
            },
            "reference_stationarity": reference_audit,
            "memory": memory,
            "assembly": assembly_diagnostics,
            "contractions": dict(evaluator.evaluation_diagnostics),
            "raw_orthogonal_hamiltonian_hermiticity_error": (
                raw_hermiticity_error
            ),
            "raw_orthogonal_hamiltonian_hermiticity_tolerance": (
                raw_hermiticity_tolerance
            ),
            "raw_orthogonal_hamiltonian_hermiticity_hard_tolerance": (
                1.0e3 * raw_hermiticity_tolerance
            ),
            "raw_orthogonal_hamiltonian_hermiticity_gate_passed": bool(
                raw_hermiticity_error <= raw_hermiticity_tolerance
            ),
            "eigen_residual_maximum": eigen_residual,
            "eigen_residual_scale": eigen_residual_scale,
            "eigen_residual_tolerance": eigen_residual_tolerance,
            "eigen_residual_hard_tolerance": eigen_residual_hard_tolerance,
            "eigen_residual_gate_passed": bool(
                eigen_residual <= eigen_residual_tolerance
            ),
            "eigen_residual_scope": "returned_states",
            "physical_metric_norms": [
                [float(value.real), float(value.imag)]
                for value in physical_norms
            ],
            "physical_metric_norm_error": norm_error,
            "state_selection": state_selection,
            "degenerate_eigenspace_alignment": {
                "absolute_tolerance": self.eigenvalue_degeneracy_atol,
                "relative_tolerance": self.eigenvalue_degeneracy_rtol,
                "clusters": degenerate_clusters,
            },
            "selected_eigenstate_indices": [
                int(value) for value in selected_indices
            ],
            "reference_amplitudes": [
                [
                    float(all_reference_amplitudes[index].real),
                    float(all_reference_amplitudes[index].imag),
                ]
                for index in selected_indices
            ],
            "davidson_q": davidson_reasons,
        }
        self.sub_times["total"] = time.perf_counter() - total_start

        for key in SECTOR_ORDER:
            diagnostics = self.sector_diagnostics[key]
            logger.info(
                self,
                "root %d FIC-MRCISD metric %-4s raw=%d rank=%d "
                "local-rank=%d cutoff=%.3e",
                root,
                key,
                diagnostics["raw_dimension"],
                diagnostics["rank"],
                diagnostics["local_rank"],
                diagnostics["metric_cutoff"],
            )
        logger.note(
            self,
            "root %d X2C-FIC-MRCISD state=%d E_corr=%.16g "
            "E_tot=%.16g w_ref=%.12g +Q=%.16g E_tot(+Q)=%.16g",
            root,
            self.selected_state,
            self.e_corr,
            self.e_tot,
            self.reference_weights[0],
            self.de_dav_q,
            self.e_tot_q,
        )
        logger.info(
            self,
            "FIC-MRCISD dimensions raw=%d retained=%d; H Hermiticity "
            "residual=%.3e; timings=%s",
            raw_dimension,
            orth_dimension,
            raw_hermiticity_error,
            " | ".join(
                f"{name}={value:.2f}s"
                for name, value in self.sub_times.items()
            ),
        )
        gc.collect()
        return self.e_corr


X2CICMRCISD = WickX2CICMRCISD
