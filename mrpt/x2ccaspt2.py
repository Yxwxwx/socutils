# SPDX-License-Identifier: GPL-3.0-or-later
r"""Single-state, fully internally contracted, exact-4-RDM spinor CASPT2.

Conventions match socutils.mrpt.spinor_helper::

    H = sum h[p,q] C[p] D[q]
        + 1/2 sum w[p,q,r,s] C[p] C[q] D[s] D[r]
    w[p,q,r,s] = eri[p,r,q,s]; eri[p,q,r,s] = (pq|rs)
    dm[k][p1,...,pk,qk,...,q1] = <C[p1]...C[pk]D[qk]...D[q1]>

approximation='full' (default) solves all allowed Fock couplings.
approximation='caspt2d' solves amplitudes using the class-diagonal Fock A_D.
This is not the diagonal of an arbitrary orthogonalization basis: complete
intraclass matrices, including the active Fock block, remain. The default
energy_functional='model' reports the A_D functional (OpenMolcas 24.10).
energy_functional='full' reports the full-Fock functional of these same
amplitudes (the older BLOCK-enabled OpenMolcas 48f9d80 build). Both are
available as e_model_hylleraas/e_full_hylleraas; e_projected is nonvariational.
approximation='partial_ccvv' fixes ijrs/CCVV amplitudes to their diagonal-
model values and solves the OTHER rows with A_RC t_C as a fixed driving
term. Its energy is the full-model Hylleraas functional. No amplitude or
matrix-storage savings are claimed for this dense reference implementation.

The four mutually exclusive regularizers are 'real', 'imaginary', 'sigma2',
and 'sigma1'. For epsilon=level_shift>0 (Eh), the reciprocal filters are

    real:       1/(d+epsilon)
    imaginary:  d/(d*d+epsilon*epsilon)
    sigma_p:    [1-exp(-(|d|/epsilon)**p)]/d, p=1,2.

regularization_basis='caspt2d' (default) only uses the class-diagonal model
for denominator estimates, while retaining couplings permitted by the
SELECTED approximation. regularization_basis='spectral' filters the full
selected matrix spectrum instead. These prescriptions differ for coupled
matrices. approximation and regularization_basis are independent options.
All zero-strength regularizers use the unchanged unshifted solver.

IPEA (ipea_shift, Eh) modifies H0; it is NOT a regulator subtracted from the
Hylleraas energy. ipea_definition='spinor_diagonal' implements a specified
native-spinor occupation prescription in the primitive _FAMILIES basis:

    w_mu = sum_(active creators p) n_p + sum_(active annihilators q)(1-n_q)
    Delta_IPEA[mu,nu] = delta_mu,nu * ipea_shift * w_mu * S[mu,mu].

Here n_p=<C_p D_p> is in [0,1]. It is a capacity-one extension of the
spatial occupation formula; it is NOT asserted to reproduce OpenMolcas
spin-free IPEA. IPEA depends on the primitive contraction basis and metric
projection, even when two generating sets have the same span. Explicit
singles and ordered spinor doubles are retained in this implementation.
Nonzero IPEA emits CASPT2IPEAWarning to make this methodological choice
visible. This native-spinor prescription has NOT been validated against
OpenMolcas.

For a scalar Hamiltonian only, the matrix-free backend also provides
ipea_definition='molcas'. It expands the conventional spin-adapted primitive
IC generators into the retained native-spinor response space and applies the
OpenMolcas occupation correction there. Hamiltonian, source, metric, and raw
1--4 RDM contractions still use the native spinor equations: this is not a
scalar solver backend and no spin trace is taken. The adjacent alpha/beta
ordering is checked against h1e, the supplied Fock matrix, and ERIs before it
is used. An arbitrary SOC Hamiltonian is rejected because OpenMolcas' scalar
IPEA prescription has no unique relativistic extension. Use ipea_basis='input'
with the same semicanonical scalar orbitals as the external reference.

The default ipea_basis='pseudocanonical' diagonalizes F_AA, constructs this
prescription in its induced IC coordinates, and pulls it back to the input
basis. No active MPS or rank-4 RDM rotates. ipea_basis='input' uses the
supplied orbital gauge verbatim. Degenerate active Fock eigenspaces do not
uniquely fix a pseudocanonical IPEA gauge. No partner labels are guessed.

For Phi_mu=O_mu|0>, exact-RDM matrix construction is

    S[mu,nu]=<Phi_mu|Phi_nu>,  EF=<0|F|0>  (NOT E_CASSCF),
    A[mu,nu]=<Phi_mu|F|Phi_nu>-EF*S[mu,nu],  b[mu]=<Phi_mu|H|0>.

Full O_mu^dagger F O_nu is evaluated. A commutator alone would lose terms
since |0> is not generally a Fock eigenvector, including rank-4 RDM terms.
Hylleraas energy uses the requested A including IPEA, without the regularizer:
t^dagger A t+2 Re(t^dagger b). e_projected=Re(b^dagger t).
For full-functional CASPT2-D/partial_ccvv an e_constraint_correction is present:

    e_corr = e_projected + e_shift_correction + e_constraint_correction.

In full CASPT2 the constraint correction is zero. A partially frozen
complex response need not have real b^dagger t; its imaginary part is a
reported diagnostic, not silently interpreted as a real expectation value.

block2 (default) uses Block2 Wick. python is an exact recursive CAR/RDM
algebra backend requiring only NumPy/SciPy, without a mocked Block2 module.
The low-level integral API can run without PySCF. The StreamObject adapter
requires the existing socutils/PySCF environment. No automatic fallback.

The dense path is a correctness oracle, with memory preflight. The native
matrix-free path factors external indices and uses local active metrics.
Both consume exact raw spinor 1--4 RDMs. No IPEA=0
baseline, SC/FIC/QD production file or user input tensor is modified in place.
MS/XMS, cumulants, Kramers compression and nonzero frozen spinors are absent.

References (definitions, not claims of completed cross-code validation):
  Andersson et al., JPC 94, 5483 (1990), 10.1021/j100377a012.
  Andersson et al., JCP 96, 1218 (1992), 10.1063/1.462209.
  Ghigo et al., CPL 396, 142 (2004), 10.1016/j.cplett.2004.08.032.
  Shiozaki and Mizukami, JCTC 11, 4733 (2015), 10.1021/acs.jctc.5b00754.
  Yanai et al., JCTC 13, 4829 (2017), 10.1021/acs.jctc.7b00735, Eq.15.
  Battaglia et al., JCTC 18, 4814 (2022), 10.1021/acs.jctc.2c00368.
  Nishimoto, JCP 158, 174112 (2023), 10.1063/5.0147611.
  https://sebwouters.github.io/CheMPS2/caspt2.html
  https://molcas.gitlab.io/OpenMolcas/sphinx/users.guide/programs/caspt2.html
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from functools import lru_cache
import itertools
import json
import math
from pathlib import Path
import re
import string
import time
import warnings
from typing import Any

import numpy as np
from scipy import linalg

try:
    from pyscf import lib as _pyscf_lib
except ImportError:
    _pyscf_lib = None

__all__ = [
    "WickX2CCASPT2", "X2CCASPT2", "CASPT2Result", "CASPT2Matrices",
    "CASPT2NumericalError", "CASPT2IntruderError", "caspt2_from_integrals",
    "build_caspt2_matrices", "build_generalized_fock", "solve_caspt2",
    "dump_caspt2_wick_equations", "SUBSPACE_ORDER", "build_ipea_matrix",
    "CASPT2IPEAWarning",
]

SUBSPACE_ORDER = ("ijrs", "rsi", "ijr", "rs", "ij", "ir", "r", "i")


class CASPT2NumericalError(ValueError):
    """Input or contracted-matrix consistency failure (not a level shift)."""


class CASPT2IntruderError(CASPT2NumericalError):
    """An unshifted CASPT2 denominator is unresolved near zero."""


class CASPT2IPEAWarning(UserWarning):
    """A native-spinor, basis-defined IPEA prescription is being used."""


@dataclass(frozen=True)
class _Family:
    sector: str
    name: str
    expression: str
    active: tuple[str, ...]
    active_pairs: tuple[tuple[int, int], ...] = ()

    @property
    def variables(self):
        return tuple(self.sector) + self.active

    @property
    def spaces(self):
        return tuple("I" if x in "ij" else "E" for x in self.sector) + (
            "A",
        ) * len(self.active)

    @property
    def restrictions(self):
        variables = self.variables
        pairs = []
        for first, second in (("i", "j"), ("r", "s")):
            if first in self.sector and second in self.sector:
                pairs.append((variables.index(first), variables.index(second)))
        offset = len(self.sector)
        pairs += [(offset + p, offset + q) for p, q in self.active_pairs]
        return tuple(pairs)


# Same unweighted IC span as the repository's x2cficnevpt2._IC_COMPONENTS.
# Keep singles in i/r/ir explicitly: metric removal handles exact redundancy.
_FAMILIES = (
    _Family("ijrs", "double", "C[r] C[s] D[j] D[i]", ()),
    _Family("rsi", "double", "C[r] C[s] D[a] D[i]", ("a",)),
    _Family("ijr", "double", "C[r] C[a] D[j] D[i]", ("a",)),
    _Family("rs", "double", "C[r] C[s] D[b] D[a]", ("a", "b"), ((0, 1),)),
    _Family("ij", "double", "C[a] C[b] D[j] D[i]", ("a", "b"), ((0, 1),)),
    _Family("ir", "single", "C[r] D[i]", ()),
    _Family("ir", "double", "C[r] C[a] D[b] D[i]", ("a", "b")),
    _Family("r", "single", "C[r] D[a]", ("a",)),
    _Family("r", "double", "C[r] C[a] D[c] D[b]", ("a", "b", "c"), ((1, 2),)),
    _Family("i", "single", "C[a] D[i]", ("a",)),
    _Family("i", "double", "C[a] C[b] D[c] D[i]", ("a", "b", "c"), ((0, 1),)),
)


def _int(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return int(value)


def _positive_float(value, name, *, allow_zero=True):
    value = float(value)
    if not np.isfinite(value) or value < 0 or (not allow_zero and value == 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")
    return value


def _maxabs(a):
    return float(np.max(np.abs(a), initial=0.0))


def _finite(a, name):
    a = np.asarray(a)
    if a.dtype.kind not in "fciub":
        raise TypeError(f"{name} must be numeric")
    # Avoid an array-sized boolean temporary for the 4-RDM/ERI.
    chunks = a if a.ndim else (a,)
    for chunk in chunks:
        if not np.all(np.isfinite(chunk)):
            raise CASPT2NumericalError(f"{name} contains a non-finite value")
    return a


def _hermitian(a, name, atol, rtol):
    error = _maxabs(a - a.conj().T)
    limit = atol + rtol * max(1.0, _maxabs(a))
    if error > limit:
        raise CASPT2NumericalError(f"{name} is not Hermitian: {error:.3e} > {limit:.3e}")
    return error


def _validate_input(h1e, eri, ncore, ncas, nelecas, pdms, atol, rtol, check_rdms):
    ncore, ncas, nelecas = (_int(x, name) for x, name in (
        (ncore, "ncore"), (ncas, "ncas"), (nelecas, "nelecas")))
    h1e, eri = _finite(h1e, "h1e"), _finite(eri, "eri")
    if h1e.ndim != 2 or h1e.shape[0] != h1e.shape[1]:
        raise ValueError("h1e must be square")
    nmo = len(h1e)
    if eri.shape != (nmo,) * 4:
        raise ValueError("eri must be dense, unantisymmetrized chemist (nmo,)*4")
    if ncore + ncas > nmo or nelecas > ncas:
        raise ValueError("invalid core/active/virtual partition or active electron count")
    diag = {"h1e_hermiticity_error": _hermitian(h1e, "h1e", atol, rtol)}
    pair_error = conj_error = scale = 0.0
    for p in range(nmo):
        pair_error = max(pair_error, _maxabs(eri[p] - eri[:, :, p, :].transpose(2, 0, 1)))
        conj_error = max(conj_error, _maxabs(eri[p] - eri[:, p, :, :].transpose(0, 2, 1).conj()))
        scale = max(scale, _maxabs(eri[p]))
    if max(pair_error, conj_error) > atol + rtol * max(1.0, scale):
        raise CASPT2NumericalError("eri violates complex Coulomb symmetry")
    diag.update(eri_pair_error=pair_error, eri_conjugate_error=conj_error)
    if not isinstance(pdms, (tuple, list)) or len(pdms) != 4:
        raise ValueError("pdms must be the exact raw SGF (dm1,dm2,dm3,dm4)")
    checked = []
    for k, dm in enumerate(pdms, 1):
        dm = _finite(dm, f"dm{k}")
        if dm.shape != (ncas,) * (2 * k):
            raise ValueError(f"dm{k} has shape {dm.shape}; expected {(ncas,) * (2*k)}")
        checked.append(dm)
        if check_rdms and dm.size:
            herm = anti = 0.0
            adj = dm.transpose(tuple(reversed(range(dm.ndim))))
            # Chunk the residuals along the first axis; no full dm4 copy.
            for p in range(ncas):
                herm = max(herm, _maxabs(dm[p] - adj[p].conj()))
            for axis in (*range(k - 1), *range(k, 2 * k - 1)):
                trans = dm.swapaxes(axis, axis + 1)
                for p in range(ncas):
                    anti = max(anti, _maxabs(dm[p] + trans[p]))
            dm_scale = max(1.0, max((_maxabs(x) for x in dm), default=0.0))
            if max(herm, anti) > atol + rtol * dm_scale:
                raise CASPT2NumericalError(f"dm{k} raw-order Hermiticity/antisymmetry failure")
            diag[f"dm{k}"] = {"hermiticity_error": herm, "antisymmetry_error": anti}
    trace_error = abs(np.trace(checked[0]) - nelecas)
    if trace_error > atol + rtol * max(1, nelecas):
        raise CASPT2NumericalError("dm1 trace differs from nelecas")
    diag["dm1_trace_error"] = float(trace_error)
    if check_rdms:
        for k in range(2, 5):
            actual = np.trace(checked[k-1], axis1=k-1, axis2=k)
            expected = (nelecas-k+1) * checked[k-2]
            err = _maxabs(actual - expected)
            if err > atol + rtol * max(1.0, _maxabs(expected)):
                raise CASPT2NumericalError(f"dm{k}->dm{k-1} contraction failure: {err:.3e}")
            diag.setdefault(f"dm{k}", {})["contraction_error"] = err
    return h1e, eri, tuple(checked), ncore, ncas, nelecas, diag


def build_generalized_fock(h1e, eri, dm1, ncore):
    r"""Return the native spinor Fock matrix and full raw density.

    gamma[p,q] = <C[p]D[q]> (transpose of the usual AO density convention).
    F[p,q] = h[p,q] + sum_rs gamma[r,s] ((pq|rs) - (ps|rq)).

    The raw spin density is used. For an open-shell *spin-free* scalar
    comparison, explicitly inject the spin-free Fock operator via fock_mo;
    an Ms-polarized spinor Fock need not equal F_spatial tensor I_spin.
    """
    h1e, eri, dm1 = np.asarray(h1e), np.asarray(eri), np.asarray(dm1)
    ncore = _int(ncore, "ncore")
    nmo = len(h1e)
    if h1e.shape != (nmo, nmo) or eri.shape != (nmo,) * 4:
        raise ValueError("inconsistent h1e/eri shapes")
    if dm1.ndim != 2 or dm1.shape[0] != dm1.shape[1] or ncore + len(dm1) > nmo:
        raise ValueError("invalid active dm1 or ncore")
    gamma = np.zeros((nmo, nmo), dtype=np.complex128)
    gamma[:ncore, :ncore] = np.eye(ncore)
    a = slice(ncore, ncore + len(dm1))
    gamma[a, a] = dm1
    fock = np.array(h1e, dtype=np.complex128, copy=True)
    fock += np.einsum("pqrs,rs->pq", eri, gamma, optimize=True)
    fock -= np.einsum("psrq,rs->pq", eri, gamma, optimize=True)
    return fock, gamma


def _electronic_reference_energy(h1e, eri, ncore, ncas, pdms):
    h_eff = np.array(h1e, dtype=np.complex128, copy=True)
    e_core = 0j
    for i in range(ncore):
        h_eff += eri[:, :, i, i] - eri[:, i, i, :]
        e_core += h1e[i, i]
        for j in range(ncore):
            e_core += 0.5 * (eri[i, i, j, j] - eri[i, j, j, i])
    a = slice(ncore, ncore + ncas)
    e_active = np.einsum("pq,pq->", h_eff[a, a], pdms[0])
    e_active += 0.5 * np.einsum("pqrs,prsq->", eri[a, a, a, a], pdms[1], optimize=True)
    return e_core + e_active


@dataclass(frozen=True)
class _Tensor:
    name: str
    indices: tuple[str, ...]
    spaces: tuple[str, ...]


@dataclass(frozen=True)
class _Term:
    coefficient: complex
    tensors: tuple[_Tensor, ...]
    equalities: tuple[tuple[str, str], ...]
    summed_dimensions: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Kernel:
    rows: tuple[str, ...]
    columns: tuple[str, ...]
    spaces: tuple[tuple[str, str], ...]
    terms: tuple[_Term, ...]


@lru_cache(maxsize=None)
def _gaussian_expectation(operators, filled):
    """Exact core/full or external/vacuum Wick contractions, no active Wick rule."""
    if not operators:
        return ((1, ()),)
    first_kind, first_index = operators[0]
    if (first_kind == "C") != filled:
        return ()
    terms = []
    for j in range(1, len(operators)):
        kind, index = operators[j]
        if kind == first_kind:
            continue
        rest = operators[1:j] + operators[j+1:]
        for sign, deltas in _gaussian_expectation(rest, filled):
            terms.append(((-1) ** (j-1) * sign, ((first_index, index),) + deltas))
    return tuple(terms)


@lru_cache(maxsize=None)
def _normal_active(operators):
    """CAR: D[p] C[q] = delta[p,q] - C[q] D[p]."""
    for j in range(len(operators) - 1):
        if operators[j][0] == "D" and operators[j+1][0] == "C":
            result = []
            short = operators[:j] + operators[j+2:]
            for sign, ds, ops in _normal_active(short):
                result.append((sign, ((operators[j][1], operators[j+1][1]),) + ds, ops))
            swapped = operators[:j] + (operators[j+1], operators[j]) + operators[j+2:]
            for sign, ds, ops in _normal_active(swapped):
                result.append((-sign, ds, ops))
            return tuple(result)
    return ((1, (), operators),)


def _python_reduce(operators, spaces):
    """Exact expectation over |filled I> tensor |active> tensor |empty E>."""
    counts = Counter()
    for kind, idx in operators:
        counts[spaces[idx]] += 1 if kind == "C" else -1
    if any(counts.values()):
        return ()
    order = {"I": 0, "A": 1, "E": 2}
    ranks = [order[spaces[idx]] for _, idx in operators]
    sign = (-1) ** sum(ranks[i] > ranks[j] for i in range(len(ranks)) for j in range(i+1, len(ranks)))
    core = tuple(op for op in operators if spaces[op[1]] == "I")
    active = tuple(op for op in operators if spaces[op[1]] == "A")
    external = tuple(op for op in operators if spaces[op[1]] == "E")
    return tuple(
        (sign*si*se*sa, di+de+da, oa)
        for si, di in _gaussian_expectation(core, True)
        for se, de in _gaussian_expectation(external, False)
        for sa, da, oa in _normal_active(active)
    )


def _canonical_term(coefficient, tensors, deltas, active, spaces, output, summed):
    """Eliminate Kronecker deltas while preserving free-index equality masks."""
    parent = {x: x for x in spaces}

    def find(x):
        while x != parent[x]:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for x, y in deltas:
        if spaces[x] != spaces[y]:
            return None
        parent[find(y)] = find(x)
    classes = {}
    for x in spaces:
        classes.setdefault(find(x), []).append(x)
    mapping, equalities, repeated_dims = {}, [], []
    for members in classes.values():
        free = [x for x in output if x in members]
        representative = free[0] if free else min(members)
        mapping.update((x, representative) for x in members)
        equalities.extend((free[0], x) for x in free[1:])
    creations = [mapping[x] for kind, x in active if kind == "C"]
    destructions = [mapping[x] for kind, x in active if kind == "D"]
    if len(creations) != len(destructions):
        raise RuntimeError("non-number-conserving active expectation survived")
    k = len(creations)
    if k > 4:
        raise RuntimeError(f"CASPT2 equation unexpectedly needs a {k}-RDM")
    for seq in (creations, destructions):
        if len(set(seq)) != len(seq):
            return None
        coefficient *= (-1) ** sum(seq[i] > seq[j] for i in range(len(seq)) for j in range(i+1, len(seq)))
        seq.sort()
    mapped_tensors = []
    for name, indices in tensors:
        ix = tuple(mapping[x] for x in indices)
        mapped_tensors.append(_Tensor(name, ix, tuple(spaces[x] for x in ix)))
    dm_ix = tuple(creations + destructions)
    mapped_tensors.append(_Tensor(f"dm{k}", dm_ix, ("A",) * (2*k)))
    used = {x for tensor in mapped_tensors for x in tensor.indices}
    for members in classes.values():
        rep = mapping[members[0]]
        if not any(x in output for x in members) and rep not in used:
            # Closed delta loop with no tensor occurrence contributes dimension.
            if any(x in summed for x in members):
                repeated_dims.append(spaces[rep])
    return _Term(complex(coefficient), tuple(mapped_tensors), tuple(equalities), tuple(repeated_dims))


def _block2_reduce(coefficient, tensor, operators, spaces, summed, output):
    """Compile actual Block2 Wick output into the common tensor evaluator."""
    try:
        from block2 import (WickIndexTypes as IT, WickIndex, WickExpr,
                            WickTensorTypes as TT, MapWickIndexTypesSet,
                            MapPStrIntVectorWickPermutation)
    except ImportError as exc:
        raise ImportError("Block2 Wick is unavailable; install the repository dependencies, "
                          "or explicitly select wick_backend='python' for the CAR reference engine") from exc
    index_map = MapWickIndexTypesSet()
    for space, enum in (("I", IT.Inactive), ("A", IT.Active), ("E", IT.External)):
        labels = "".join(x for x, sp in spaces.items() if sp == space)
        index_map[enum] = WickIndex.parse_set(labels)
    # Deliberately NO real-spin-free ERI permutation assumptions.
    perms = MapPStrIntVectorWickPermutation()
    text = f"{float(coefficient):.17g} "
    if summed:
        text += "SUM <" + "".join(summed) + "> "
    if tensor is not None:
        text += tensor[0] + "[" + "".join(tensor[1]) + "] "
    text += " ".join(f"{kind}[{idx}]" for kind, idx in operators)
    expr = WickExpr.parse(text, index_map, perms)
    expr = expr.expand().remove_external().remove_inactive().simplify()
    terms = []
    for wt in expr.terms:
        local_spaces = dict(spaces)
        plain, deltas, active = [], [], []
        for wten in wt.tensors:
            indices = tuple(ix.name for ix in wten.indices)
            for ix in wten.indices:
                matches = [space for space, enum in (("I", IT.Inactive), ("A", IT.Active), ("E", IT.External))
                           if (ix.types & enum) != IT.Nothing]
                if len(matches) != 1:
                    raise RuntimeError(f"ambiguous Block2 index type: {ix}")
                local_spaces[ix.name] = matches[0]
            if wten.type == TT.CreationOperator or wten.type == TT.DestroyOperator:
                if any(local_spaces[x] != "A" for x in indices):
                    raise RuntimeError("Block2 left an inactive/external operator")
                active.append(("C" if wten.type == TT.CreationOperator else "D", indices[0]))
            elif str(wten.name).lower() == "delta":
                deltas.append(indices)
            else:
                plain.append((wten.name, indices))
        summed_local = tuple(ix.name for ix in wt.ctr_indices)
        for ix in wt.ctr_indices:
            if ix.name not in local_spaces:
                for space, enum in (("I", IT.Inactive), ("A", IT.Active), ("E", IT.External)):
                    if (ix.types & enum) != IT.Nothing:
                        local_spaces[ix.name] = space
        term = _canonical_term(wt.factor, plain, deltas, active, local_spaces, output, summed_local)
        if term is not None:
            terms.append(term)
    return terms


@lru_cache(maxsize=None)
def _compile_kernel(kind, bra_id, ket_id, backend):
    if backend not in ("block2", "python"):
        raise ValueError("wick_backend must be 'block2' or 'python'")
    bra = _FAMILIES[bra_id]
    ket = None if ket_id is None else _FAMILIES[ket_id]
    available = iter(string.ascii_lowercase + string.ascii_uppercase)
    rows = tuple(next(available) for _ in bra.variables)
    columns = () if ket is None else tuple(next(available) for _ in ket.variables)
    spaces = dict(zip(rows, bra.spaces))
    if ket is not None:
        spaces.update(zip(columns, ket.spaces))
    bmap = dict(zip(bra.variables, rows))
    ops_bra = [("D" if op == "C" else "C", bmap[ix])
               for op, ix in reversed(re.findall(r"([CD])\[(\w+)\]", bra.expression))]
    if ket is not None:
        kmap = dict(zip(ket.variables, columns))
        ops_ket = [(op, kmap[ix]) for op, ix in re.findall(r"([CD])\[(\w+)\]", ket.expression)]
    else:
        ops_ket = []
    dummy = tuple(next(available) for _ in range(4))
    candidates = []
    if kind == "S":
        candidates.append((1.0, None, (), (), ()))
    else:
        for combination in itertools.product("IAE", repeat=2):
            x, y = dummy[:2]
            candidates.append((1.0, ("f" if kind == "F" else "h", (x, y)),
                               (("C", x), ("D", y)), dummy[:2], combination))
        if kind == "b":
            for combination in itertools.product("IAE", repeat=4):
                x, y, z, t = dummy
                candidates.append((0.5, ("w", dummy),
                                   (("C", x), ("C", y), ("D", t), ("D", z)), dummy, combination))
    terms = []
    for coefficient, tensor, middle, summed, types in candidates:
        local_spaces = {**spaces, **dict(zip(summed, types))}
        ops = tuple(ops_bra) + tuple(middle) + tuple(ops_ket)
        balance = Counter()
        for op, idx in ops:
            balance[local_spaces[idx]] += 1 if op == "C" else -1
        if any(balance.values()):
            continue
        if backend == "block2":
            terms.extend(_block2_reduce(coefficient, tensor, ops, local_spaces, summed, rows+columns))
        else:
            for sign, deltas, active in _python_reduce(ops, local_spaces):
                term = _canonical_term(coefficient*sign, () if tensor is None else (tensor,),
                                       deltas, active, local_spaces, rows+columns, summed)
                if term is not None:
                    terms.append(term)
    combined = {}
    for term in terms:
        key = (term.tensors, term.equalities, term.summed_dimensions)
        combined[key] = combined.get(key, 0j) + term.coefficient
    reduced = tuple(_Term(c, tensors, equalities, summed_dims)
                    for (tensors, equalities, summed_dims), c in combined.items() if c != 0)
    return _Kernel(rows, columns, tuple(spaces.items()), reduced)


def dump_caspt2_wick_equations(filename=None, *, wick_backend="block2"):
    """Dump the actual evaluated S/F/b contractions, not a formal template."""
    lines = [f"CASPT2: exact 4-RDM; backend={wick_backend}; IPEA=0; no shift"]
    for i, bra in enumerate(_FAMILIES):
        for kind, partners in (("b", (None,)), ("S", range(len(_FAMILIES))), ("F", range(len(_FAMILIES)))):
            for j in partners:
                k = _compile_kernel(kind, i, j, wick_backend)
                if not k.terms:
                    continue
                label = f"{bra.sector}/{bra.name}"
                if j is not None:
                    label += f" | {_FAMILIES[j].sector}/{_FAMILIES[j].name}"
                lines.append(f"\n{kind}: {label}; row={k.rows}; column={k.columns}")
                for term in k.terms:
                    tensors = " * ".join(f"{t.name}{''.join(t.spaces)}[{''.join(t.indices)}]" for t in term.tensors)
                    lines.append(f"{term.coefficient} {tensors}; equal={term.equalities}; loops={term.summed_dimensions}")
    text = "\n".join(lines) + "\n"
    if filename is not None:
        Path(filename).write_text(text, encoding="utf-8")
    return text


def _family_dimension(family, dimensions):
    value = math.prod(dimensions[sp] for sp in family.spaces)
    for p, q in family.restrictions:
        n = dimensions[family.spaces[p]]
        if n < 2:
            return 0
        value = value // (n*n) * (n*(n-1)//2)
    return value


def _coordinates(family, dimensions):
    tuples = [ix for ix in itertools.product(*(range(dimensions[sp]) for sp in family.spaces))
              if all(ix[p] < ix[q] for p, q in family.restrictions)]
    return np.asarray(tuples, dtype=np.int64).reshape(-1, len(family.variables))


class _Data:
    def __init__(self, h, eri, fock, pdms, ncore, ncas):
        self.h, self.w, self.f = h, eri.transpose(0, 2, 1, 3), fock
        self.pdms = (np.asarray(1.0 + 0j),) + tuple(pdms)
        self.slices = {"I": slice(0, ncore), "A": slice(ncore, ncore+ncas),
                       "E": slice(ncore+ncas, len(h))}
        self.dimensions = {"I": ncore, "A": ncas, "E": len(h)-ncore-ncas}

    def get(self, tensor):
        if tensor.name.startswith("dm"):
            return self.pdms[int(tensor.name[2:])]
        array = {"h": self.h, "w": self.w, "f": self.f}[tensor.name]
        return array[tuple(self.slices[sp] for sp in tensor.spaces)]


def _evaluate_tile(kernel, rows, columns, data, einsum):
    """Gather constrained free indices before einsum; never build 8-axis H blocks."""
    nr, nc = len(rows), len(columns)
    result = np.zeros((nr, nc), dtype=np.complex128)
    fixed = {x: (0, rows[:, i]) for i, x in enumerate(kernel.rows)}
    fixed.update({x: (1, columns[:, i]) for i, x in enumerate(kernel.columns)})
    for term in kernel.terms:
        dummy_spaces = {}
        for tensor in term.tensors:
            for x, sp in zip(tensor.indices, tensor.spaces):
                if x not in fixed:
                    dummy_spaces[x] = sp
        dummy = {x: (i+2, np.arange(data.dimensions[sp]))
                 for i, (x, sp) in enumerate(sorted(dummy_spaces.items()))}
        if any(len(indices) == 0 for _, indices in dummy.values()):
            continue
        indexers = {**fixed, **dummy}
        subscripts, operands = [], []
        for tensor in term.tensors:
            array = data.get(tensor)
            axes = sorted({indexers[x][0] for x in tensor.indices})
            inds = []
            for x in tensor.indices:
                axis, vec = indexers[x]
                shape = [1] * len(axes)
                shape[axes.index(axis)] = len(vec)
                inds.append(vec.reshape(shape))
            gathered = array[tuple(inds)] if inds else array
            # String subscripts are supported by both NumPy and PyTBLIS.
            subscripts.append("".join(string.ascii_letters[axis] for axis in axes))
            operands.append(np.asarray(gathered, dtype=np.complex128))
        # Explicit output axes also cover constants / otherwise absent indices.
        mask = np.ones((nr, nc), dtype=float)
        for x, y in term.equalities:
            ax, ix = fixed[x]
            ay, iy = fixed[y]
            vx = ix[:, None] if ax == 0 else ix[None, :]
            vy = iy[:, None] if ay == 0 else iy[None, :]
            mask *= (vx == vy)
        if not np.any(mask):
            continue
        subscripts.append("ab")
        operands.append(np.asarray(mask, dtype=np.complex128))
        expression = ",".join(subscripts) + "->ab"
        coefficient = term.coefficient * math.prod(data.dimensions[sp] for sp in term.summed_dimensions)
        result += coefficient * einsum(expression, *operands, optimize=True)
    return result


@dataclass
class CASPT2Matrices:
    overlap: np.ndarray
    fock_matrix: np.ndarray
    rhs: np.ndarray
    fock_mo: np.ndarray
    fock_reference_energy: float
    reference_energy: float
    sector_slices: dict[str, slice]
    family_slices: tuple[slice, ...]
    coordinates: tuple[np.ndarray, ...]
    diagnostics: dict[str, Any] = field(default_factory=dict)
    active_dm1: np.ndarray | None = None

    @property
    def a(self):
        return self.fock_matrix - self.fock_reference_energy * self.overlap


@dataclass
class CASPT2Result:
    reference_energy: float
    fock_reference_energy: float
    e_corr: float
    e_hylleraas: float
    amplitudes: np.ndarray
    reference_weight: float
    sub_eners: dict[str, float]
    diagnostics: dict[str, Any]
    matrices: CASPT2Matrices | None = None
    e_projected: float | None = None
    e_shift_correction: float = 0.0
    regularizer: str = "none"
    level_shift: float = 0.0
    regularization_basis: str = "caspt2d"
    sub_projected_eners: dict[str, float] = field(default_factory=dict)
    approximation: str = "full"
    ipea_shift: float = 0.0
    ipea_definition: str = "spinor_diagonal"
    ipea_basis: str = "pseudocanonical"
    e_constraint_correction: float = 0.0
    e_full_hylleraas: float | None = None
    e_no_ipea_hylleraas: float | None = None
    e_model_hylleraas: float | None = None
    energy_functional: str = 'model'

    @property
    def e_tot(self):
        return self.reference_energy + self.e_corr

    def summary(self):
        return dict(method="SS-X2C-IC-CASPT2", ipea_shift=self.ipea_shift,
                    approximation=self.approximation, ipea_definition=self.ipea_definition,
                    ipea_basis=self.ipea_basis, e_constraint_correction=self.e_constraint_correction,
                    e_full_hylleraas=self.e_full_hylleraas,
                    e_model_hylleraas=self.e_model_hylleraas,
                    energy_functional=self.energy_functional,
                    e_no_ipea_hylleraas=self.e_no_ipea_hylleraas,
                    regularizer=self.regularizer, level_shift=self.level_shift,
                    regularization_basis=self.regularization_basis,
                    energy_definition=self.diagnostics.get("energy_definition", "model_Hylleraas_shift_corrected"),
                    e_projected=self.e_projected, e_shift_correction=self.e_shift_correction,
                    sub_projected_eners=self.sub_projected_eners,
                    reference_energy=self.reference_energy,
                    fock_reference_energy=self.fock_reference_energy,
                    e_corr=self.e_corr, e_tot=self.e_tot, e_hylleraas=self.e_hylleraas,
                    reference_weight=self.reference_weight, sub_eners=self.sub_eners,
                    diagnostics=self.diagnostics)

    def to_json(self, filename):
        Path(filename).write_text(json.dumps(self.summary(), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def build_caspt2_matrices(
    h1e, eri, ncore, ncas, nelecas, pdms, *, fock_mo=None, constant=0.0,
    reference_energy=None, wick_backend="block2", contraction_backend="numpy",
    atol=1e-10, rtol=1e-9, check_rdms=True, max_memory_mb=2048.0,
    tile_size=16, tile_memory_mb=32.0,
):
    r"""Build full IC S, <Phi|F|Phi>, and <Phi|H|0> with exact RDMs.

    ``fock_mo`` optionally supplies a *full* Hermitian Fock matrix in the same
    MO basis, useful for matched-H0 scalar/open-shell regressions. No blocks
    are discarded or implicitly diagonalized. ``constant`` is the nuclear
    or other state-independent scalar in the physical Hamiltonian, NOT a
    folded-core energy: all ncore spinors remain explicit in Wick.
    """
    start = time.perf_counter()
    atol = _positive_float(atol, "atol")
    rtol = _positive_float(rtol, "rtol")
    max_memory_mb = _positive_float(max_memory_mb, "max_memory_mb", allow_zero=False)
    tile_memory_mb = _positive_float(tile_memory_mb, "tile_memory_mb", allow_zero=False)
    tile_size = _int(tile_size, "tile_size")
    if tile_size == 0:
        raise ValueError("tile_size must be positive")
    if wick_backend not in ("block2", "python"):
        raise ValueError("wick_backend must be 'block2' or 'python'")
    if contraction_backend == "numpy":
        einsum = np.einsum
    elif contraction_backend == "pytblis":
        from pytblis import einsum
    else:
        raise ValueError("contraction_backend must be numpy or pytblis")
    h, eri, pdms, ncore, ncas, nelecas, diagnostics = _validate_input(
        h1e, eri, ncore, ncas, nelecas, pdms, atol, rtol, check_rdms)
    native_fock, gamma = build_generalized_fock(h, eri, pdms[0], ncore)
    fock = native_fock if fock_mo is None else _finite(fock_mo, "fock_mo")
    if fock.shape != h.shape:
        raise ValueError("fock_mo has wrong shape")
    diagnostics["fock_hermiticity_error"] = _hermitian(fock, "fock_mo", atol, rtol)
    fock = np.asarray(fock, dtype=np.complex128)
    ef = complex(np.einsum("pq,pq->", fock, gamma))
    ee = complex(_electronic_reference_energy(h, eri, ncore, ncas, pdms))
    constant = complex(constant)
    for name, value in (("<F>", ef), ("electronic reference energy", ee), ("constant", constant)):
        if not np.isfinite(value) or abs(value.imag) > atol + rtol * abs(value.real):
            raise CASPT2NumericalError(f"{name} is not finite and real: {value}")
    eref = ee.real + constant.real
    if reference_energy is not None:
        value = complex(reference_energy)
        if not np.isfinite(value) or abs(value.imag) > atol:
            raise ValueError("reference_energy must be finite and real")
        error = abs(eref - value.real)
        diagnostics["reference_energy_error"] = float(error)
        # Use an absolute, not |total energy|-scaled, reference-energy gate.
        if error > 10 * atol:
            raise CASPT2NumericalError(f"RDM reference energy differs by {error:.3e} Eh")
    data = _Data(h, eri, fock, pdms, ncore, ncas)
    dimensions = data.dimensions
    sizes = [_family_dimension(family, dimensions) for family in _FAMILIES]
    total = sum(sizes)
    # Include both full and CASPT2-D denominator eigensystems plus the
    # scaled regularized solve/SVD workspaces in the dense memory preflight.
    estimate = 16 * (34 * total * total + 24 * total)
    diagnostics.update(raw_dimension=total, matrix_peak_estimate_bytes=estimate,
                       wick_backend=wick_backend, contraction_backend=contraction_backend,
                       fock_source="native_spinor" if fock_mo is None else "explicit_mo",
                       ncore=ncore, ncas=ncas, nvirt=dimensions["E"], nelecas=nelecas)
    if estimate > max_memory_mb * 2**20:
        raise MemoryError(f"dense CASPT2 dimension {total} needs approximately {estimate/2**20:.1f} MiB "
                          f"for matrices/solve, above max_memory_mb={max_memory_mb}; "
                          "use a smaller validation problem (this backend is not matrix-free)")
    coords = tuple(_coordinates(f, dimensions) for f in _FAMILIES)
    if sizes != [len(c) for c in coords]:
        raise RuntimeError("IC basis count and enumeration disagree")
    offsets = np.cumsum([0] + sizes)
    slices = tuple(slice(int(offsets[i]), int(offsets[i+1])) for i in range(len(sizes)))
    sector_slices = {}
    for sector in SUBSPACE_ORDER:
        indices = [i for i, f in enumerate(_FAMILIES) if f.sector == sector]
        sector_slices[sector] = slice(slices[indices[0]].start, slices[indices[-1]].stop)
    S = np.zeros((total, total), dtype=np.complex128)
    F = np.zeros_like(S)
    b = np.zeros(total, dtype=np.complex128)
    biggest = max(dimensions.values(), default=1)
    actual_tile = min(tile_size, max(1, int(math.sqrt(tile_memory_mb*2**20/(64*max(1, biggest)**2)))))
    max_rank, term_count = 0, 0
    for i, bra in enumerate(_FAMILIES):
        if not sizes[i]:
            continue
        kernel = _compile_kernel("b", i, None, wick_backend)
        term_count += len(kernel.terms)
        for first in range(0, sizes[i], actual_tile):
            last = min(first + actual_tile, sizes[i])
            b[slices[i].start+first:slices[i].start+last] = _evaluate_tile(
                kernel, coords[i][first:last], np.empty((1, 0), dtype=int), data, einsum)[:, 0]
        for j, ket in enumerate(_FAMILIES):
            if not sizes[j]:
                continue
            for kind, matrix in (("S", S), ("F", F)):
                kernel = _compile_kernel(kind, i, j, wick_backend)
                term_count += len(kernel.terms)
                for term in kernel.terms:
                    max_rank = max(max_rank, *(int(t.name[2:]) for t in term.tensors if t.name.startswith("dm")))
                if not kernel.terms:
                    continue
                for r in range(0, sizes[i], actual_tile):
                    for c in range(0, sizes[j], actual_tile):
                        rr, cc = min(r+actual_tile, sizes[i]), min(c+actual_tile, sizes[j])
                        matrix[slices[i].start+r:slices[i].start+rr,
                               slices[j].start+c:slices[j].start+cc] = _evaluate_tile(
                            kernel, coords[i][r:rr], coords[j][c:cc], data, einsum)
    diagnostics.update(maximum_rdm_rank=max_rank, algebra_term_count=term_count,
                       actual_tile_size=actual_tile,
                       overlap_hermiticity_error=_hermitian(S, "IC overlap", atol, rtol),
                       projected_fock_hermiticity_error=_hermitian(F, "IC Fock", atol, rtol),
                       matrix_seconds=time.perf_counter()-start)
    return CASPT2Matrices(S, F, b, fock, float(ef.real), float(eref), sector_slices,
                         slices, coords, diagnostics, np.array(pdms[0], copy=True))


_REGULARIZERS = frozenset(("none", "real", "imaginary", "sigma2", "sigma1"))
_REGULARIZATION_BASES = frozenset(("caspt2d", "spectral"))


def _regularization_options(regularizer, level_shift, regularization_basis):
    """Normalize one exclusive method choice; epsilon is always in Hartree."""
    if regularizer is None:
        regularizer = "none"
    if not isinstance(regularizer, str) or regularizer.lower() not in _REGULARIZERS:
        raise ValueError("regularizer must be none, real, imaginary, sigma2, or sigma1")
    regularizer = regularizer.lower()
    if not isinstance(regularization_basis, str) or regularization_basis.lower() not in _REGULARIZATION_BASES:
        raise ValueError("regularization_basis must be 'caspt2d' or 'spectral'")
    regularization_basis = regularization_basis.lower()
    epsilon = _positive_float(level_shift, "level_shift")
    if regularizer == "none" and epsilon != 0.0:
        raise ValueError("nonzero level_shift requires an explicit regularizer; shifts are not inferred")
    return regularizer, epsilon, regularization_basis


def _regularized_filter(d, regularizer, epsilon):
    r"""Real reciprocal-denominator filter, WITHOUT discarding the phase of b.

    Used only for positive epsilon and imaginary/sigma regularizers. Sigma
    uses sigma = epsilon**(-p). expm1 and scaled small-d branches preserve
    the correct limits without overflow in d**2 or catastrophic subtraction.
    Exact sigma1 zeros are checked by the caller before assigning f(0)=0 to
    directions that are provably decoupled and undriven.
    """
    d = np.asarray(d, dtype=float)
    if regularizer not in ("imaginary", "sigma1", "sigma2") or epsilon <= 0:
        raise ValueError("smooth-filter evaluation requires a positive regularization parameter")
    f = np.zeros_like(d)
    if regularizer == "imaginary":
        scale = np.maximum(np.abs(d), epsilon)
        q, e = d / scale, epsilon / scale
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            f = (q / (q*q + e*e)) / scale
    else:
        p = 1 if regularizer == "sigma1" else 2
        with np.errstate(over="ignore", under="ignore"):
            ratio = np.abs(d) / epsilon
        small = ratio <= 0.5
        # The large-ratio branch cannot lose significant digits in 1-exp(-x).
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            x = ratio[~small] ** p
            f[~small] = -np.expm1(-x) / d[~small]
            x = ratio[small] ** p
            exprel = np.ones_like(x)
            np.divide(-np.expm1(-x), x, out=exprel, where=x != 0.0)
            if p == 1:
                f[small] = (np.sign(d[small]) * exprel) / epsilon
            else:
                f[small] = ((d[small] / epsilon) * exprel) / epsilon
    if not np.all(np.isfinite(f)):
        raise CASPT2NumericalError("regularized reciprocal filter overflow; use a resolvable level_shift")
    return f


def _checked_sector_slices(matrices):
    """An exact, disjoint partition of raw IC rows, required for A_D/energy allocation."""
    n = len(matrices.rhs)
    cover = np.zeros(n, dtype=np.int64)
    output = {}
    for key, sl in matrices.sector_slices.items():
        if not isinstance(sl, slice) or sl.step not in (None, 1):
            raise ValueError("sector_slices must contain contiguous slices")
        lo = 0 if sl.start is None else sl.start
        hi = n if sl.stop is None else sl.stop
        if not isinstance(lo, (int, np.integer)) or not isinstance(hi, (int, np.integer)) or not 0 <= lo <= hi <= n:
            raise ValueError("sector slice is outside the contracted basis")
        output[key] = slice(int(lo), int(hi))
        cover[lo:hi] += 1
    if np.any(cover != 1):
        raise ValueError("sector_slices must cover every contracted direction exactly once")
    return output


def _caspt2d_operator(A, S, sectors, atol, rtol):
    r"""A_D = sum_g P_g A P_g before canonical metric orthogonalization.

    g denotes the eight occupation classes (all families within a class are
    kept together). This is equivalent to suppressing the cross-I/A/E Fock
    blocks, NOT taking the diagonal of an arbitrary metric eigenvector basis.
    Intraclass active Fock couplings remain and are diagonalized exactly.
    """
    ad = np.zeros_like(A)
    soff = S.copy()
    for sl in sectors.values():
        ad[sl, sl] = A[sl, sl]
        soff[sl, sl] = 0.0
    error = _maxabs(soff)
    if error > atol + rtol * max(1.0, _maxabs(S)):
        raise CASPT2NumericalError("CASPT2-D denominators require mutually orthogonal occupation sectors")
    return ad


def _metric_congruence(X, operator, source, metric_info):
    """Transform an IC operator/source, extending accumulation when needed."""
    refined = any(
        detail.get("gram_refinement_steps", 0)
        for detail in metric_info.get("sector_metric_ranks", {}).values()
    )
    if refined:
        xl = np.asarray(X, dtype=np.clongdouble)
        al = np.asarray(operator, dtype=np.clongdouble)
        transformed_operator = np.asarray(xl.conj().T @ al @ xl, dtype=np.complex128)
        transformed_source = (
            None if source is None else
            np.asarray(xl.conj().T @ np.asarray(source, dtype=np.clongdouble),
                       dtype=np.complex128)
        )
        metric_info["metric_congruence_accumulation_dtype"] = np.dtype(np.clongdouble).name
    else:
        transformed_operator = X.conj().T @ operator @ X
        transformed_source = None if source is None else X.conj().T @ source
        metric_info["metric_congruence_accumulation_dtype"] = np.dtype(np.complex128).name
    return transformed_operator, transformed_source


def _pole_response(d, z, epsilon, *, denominator_atol, source_atol, atol):
    """Unshifted/real-shift solve with a bound on a discarded pole's ENERGY.

    For a real shift the pole is at d=-epsilon, not d=0. The bound includes
    the shift-correction term; a tiny source can never hide an arbitrarily
    large first-order norm times epsilon.
    """
    shifted = d + epsilon
    near = np.abs(shifted) <= denominator_atol
    zn, qn, dn = z[near], shifted[near], d[near]
    if np.any((qn == 0.0) & (zn != 0.0)):
        bound = math.inf
    else:
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            u = np.zeros_like(zn)
            np.divide(zn, qn, out=u, where=qn != 0.0)
            if epsilon == 0.0:
                terms = np.zeros(len(zn), dtype=float)
                np.divide(np.abs(zn)**2, np.abs(qn), out=terms, where=qn != 0.0)
            else:
                terms = np.abs(dn) * np.abs(u)**2 + 2*np.abs(np.conj(zn)*u)
            bound = float(np.sum(terms))
    coupled = float(linalg.norm(zn))
    if not np.isfinite(bound) or coupled > source_atol or bound > atol:
        raise CASPT2IntruderError(
            f"{np.count_nonzero(near)} near-zero {'real-shifted' if epsilon else 'unshifted'} "
            f"Fock denominators; coupled source norm={coupled:.3e}, "
            f"discarded Hylleraas energy bound={bound:.3e}. "
            "No extra shift or absolute denominator was substituted.")
    response = np.zeros_like(z)
    np.divide(-z, shifted, out=response, where=~near)
    return response, near, coupled, bound


def _solve_caspt2_full(matrices, *, metric_atol=1e-12, metric_rcond=1e-10,
                 denominator_atol=1e-10, source_atol=1e-10,
                 atol=1e-10, rtol=1e-9, keep_matrices=False,
                 regularizer="none", level_shift=0.0,
                 regularization_basis="caspt2d", regularized_rcond=1e-12):
    r"""Solve full dense CASPT2 with an optional, explicitly defined regularizer.

    epsilon=``level_shift`` is nonnegative, in Eh, including for sigma1/2.
    Every zero-strength option is exactly the original unshifted solve.

    In the CASPT2-D denominator basis let A=D+O, D=diag(d). The regularized
    equation is (diag(1/f(d))+O)t=-b. We solve the equivalent scaled equation
    (I+diag(f(d))O)t=-diag(f(d))b, including the well-defined t_i=0 limits for
    imaginary/sigma2 at d_i=0. All off-diagonal couplings O are retained.
    Near-singular coupled systems are errors, not silently pseudoinverted.

    ``spectral`` uses the complete A eigensystem instead; it is NOT in
    general the same as the diagonal-denominator approximation above.
    For complex spinors 'imaginary' means the Hermitian reciprocal filter,
    not componentwise Re of a complex amplitude.

    e_corr is the ORIGINAL Hylleraas functional. e_projected is Re(b^dagger t).
    They coincide only in the unshifted stationary solution (or special cases).
    """
    regularizer, epsilon, basis = _regularization_options(
        regularizer, level_shift, regularization_basis)
    controls = dict(metric_atol=metric_atol, metric_rcond=metric_rcond,
                    denominator_atol=denominator_atol, source_atol=source_atol,
                    atol=atol, rtol=rtol, regularized_rcond=regularized_rcond)
    controls = {name: _positive_float(value, name) for name, value in controls.items()}
    if controls['regularized_rcond'] >= 1.0:
        raise ValueError("regularized_rcond must be smaller than one")
    denominator_atol, source_atol = controls["denominator_atol"], controls["source_atol"]
    atol, rtol = controls["atol"], controls["rtol"]
    start = time.perf_counter()
    S, F, b = (np.asarray(matrices.overlap), np.asarray(matrices.fock_matrix),
               np.asarray(matrices.rhs))
    if b.ndim != 1:
        raise ValueError("contracted source must be a vector")
    n = len(b)
    for name, a in (("overlap", S), ("projected Fock", F), ("source", b)):
        _finite(a, name)
    if S.shape != (n, n) or F.shape != (n, n):
        raise ValueError("contracted matrix shapes are inconsistent")
    _hermitian(S, "overlap", atol, rtol)
    _hermitian(F, "projected Fock", atol, rtol)
    ef = complex(matrices.fock_reference_energy)
    if not np.isfinite(ef) or abs(ef.imag) > atol:
        raise ValueError("fock_reference_energy must be finite and real")
    A = 0.5 * (F + F.conj().T) - ef.real * (0.5 * (S + S.conj().T))
    sectors = _checked_sector_slices(matrices)
    # Keep the globally thresholded metric range but orthogonalize each exact
    # occupation class separately.  _sector_orthogonalizer also re-whitens
    # the retained Gram matrix when a small eigenvalue amplifies eigensolver
    # roundoff; it never changes the selected rank.
    X, _, metric_info = _sector_orthogonalizer(matrices, controls)
    ao, bo = _metric_congruence(X, A, b, metric_info)
    _hermitian(ao, "orthogonal CASPT2 A", atol, rtol)
    ao = 0.5 * (ao + ao.conj().T)
    if len(bo):
        d_full, v_full = linalg.eigh(ao)
    else:
        d_full, v_full = np.empty(0), np.empty((0, 0), dtype=complex)

    effective = "none" if epsilon == 0.0 else regularizer
    extra = {}
    near = np.zeros(len(bo), dtype=bool)
    discarded_source, discarded_bound = 0.0, 0.0
    suppressed_count = 0
    if effective in ("none", "real"):
        # epsilon*I in the orthonormal basis corresponds to epsilon*S in
        # the original nonorthogonal IC basis, NOT epsilon*I there.
        d, v = d_full, v_full
        z = v.conj().T @ bo
        eps = epsilon if effective == "real" else 0.0
        response, near, discarded_source, discarded_bound = _pole_response(
            d, z, eps, denominator_atol=denominator_atol,
            source_atol=source_atol, atol=atol)
        to = v @ response
        equation_residual = ao @ to + eps*to + bo
        residual_definition = "(A_orth + epsilon I)t + b (Eh)"
        equation_limit = atol + rtol * max(1.0, float(linalg.norm(bo)))
        actual_basis = "full_spectrum"  # uniform shift is basis independent
    else:
        if basis == "spectral":
            d, v = d_full, v_full
            off = np.zeros_like(ao)
            actual_basis = "full_spectrum"
        else:
            ad = _caspt2d_operator(A, S, sectors, atol, rtol)
            ado, _ = _metric_congruence(X, ad, None, metric_info)
            _hermitian(ado, "CASPT2-D denominator operator", atol, rtol)
            if len(bo):
                d, v = linalg.eigh(0.5*(ado+ado.conj().T))
            else:
                d, v = np.empty(0), np.empty((0, 0), dtype=complex)
            off = v.conj().T @ ao @ v - np.diag(d)
            actual_basis = "class_diagonal_CASPT2D_spectrum"
            extra["retained_offdiagonal_fock_norm"] = float(linalg.norm(off))
        z = v.conj().T @ bo
        zero = d == 0.0
        if effective == "sigma1" and np.any(zero):
            # An exactly zero, driven denominator has no two-sided sigma1
            # limit. Also reject indirect driving through O, not only b_i.
            potentially_driven = ((np.abs(z[zero]) != 0.0)
                                  | (np.max(np.abs(off[zero]), axis=1, initial=0.0) != 0.0))
            if np.any(potentially_driven):
                raise CASPT2IntruderError(
                    "sigma1 has no unique limit at an exactly zero driven denominator; "
                    "use sigma2/imaginary or change the reference, not an arbitrary sign")
        f = _regularized_filter(d, effective, epsilon)
        suppressed_count = int(np.count_nonzero(f == 0.0))
        if basis == "spectral":
            response = -f*z
            # This is the filter-defining residual; the unshifted residual
            # is normally NONZERO and is recorded separately below.
            equation_residual = response + f*z
            equation_limit = atol + rtol * max(1.0, float(linalg.norm(response)))
            residual_definition = "t_spectral + f(d) b_spectral (dimensionless)"
        else:
            B = np.eye(len(d), dtype=np.complex128) + f[:, None]*off
            rhs = -f*z
            if len(rhs):
                sv = linalg.svdvals(B)
                rcond = float(sv[-1]/sv[0]) if sv[0] else 0.0
                if rcond <= max(controls['regularized_rcond'], np.finfo(float).eps*len(rhs)):
                    raise CASPT2IntruderError(
                        f"coupled regularized CASPT2 equation is nearly singular (rcond={rcond:.3e}); "
                        "diagonal-denominator regularization is not an intruder-free guarantee")
                response = linalg.solve(B, rhs, assume_a="gen")
            else:
                rcond, response = 1.0, np.empty(0, dtype=np.complex128)
            equation_residual = B @ response - rhs
            equation_limit = atol + rtol * max(1.0, float(linalg.norm(rhs)))
            residual_definition = "[I+diag(f)O]t + diag(f)b (dimensionless)"
            extra["regularized_equation_rcond"] = rcond
            extra["regularized_equation_condition_number"] = 1.0/rcond
        to = v @ response
        extra.update(exact_zero_regularization_denominators=int(np.count_nonzero(zero)),
                     near_regularization_denominators=int(np.count_nonzero(np.abs(d)<=denominator_atol)),
                     sigma1_zero_policy="raise_if_driven" if effective == "sigma1" else None)

    t = X @ to
    if not np.all(np.isfinite(t)):
        raise CASPT2NumericalError("non-finite regularized CASPT2 amplitudes")
    res = float(linalg.norm(equation_residual))
    if not np.isfinite(res) or res > equation_limit:
        raise CASPT2NumericalError(f"CASPT2 regularized-equation residual too large: {res:.3e}")
    unshifted_residual = float(linalg.norm(ao @ to + bo))
    projected = np.vdot(b, t)
    At = A @ t
    hyll = np.vdot(t, At) + 2*np.vdot(t, b).real
    energy_limit = atol + rtol * max(1.0, abs(projected.real), abs(hyll.real))
    if (not np.isfinite(projected) or not np.isfinite(hyll)
            or max(abs(projected.imag), abs(hyll.imag)) > energy_limit):
        raise CASPT2NumericalError("CASPT2 projected / Hylleraas energy is not finite and real")
    if effective == "none" and abs(hyll.real-projected.real) > energy_limit:
        raise CASPT2NumericalError("unshifted energy / Hylleraas consistency check failed")
    norm1 = float(np.vdot(to, to).real)
    if not np.isfinite(norm1):
        raise CASPT2NumericalError("first-order norm overflow")
    projected_sectors = {key: float(np.vdot(b[sl], t[sl]).real) for key,sl in sectors.items()}
    if effective == "none":
        e2, sector_energies = float(projected.real), dict(projected_sectors)
    else:
        e2 = float(hyll.real)
        # Symmetric allocation of cross-class quadratic terms: the class
        # pieces add to the full functional; they are NOT independent solves.
        sector_energies = {
            key: float((np.vdot(t[sl], At[sl]) + 2*np.vdot(t[sl], b[sl]).real).real)
            for key,sl in sectors.items()}
    if abs(sum(sector_energies.values())-e2) > energy_limit:
        raise CASPT2NumericalError("CASPT2 class energy allocations do not sum to the total")
    diagnostics = dict(matrices.diagnostics)
    diagnostics.update(
        raw_dimension=n,
        discarded_denominator_source_norm=discarded_source,
        discarded_denominator_energy_abs_bound=discarded_bound,
        minimum_denominator=float(np.min(d_full)) if d_full.size else None,
        minimum_abs_denominator=float(np.min(np.abs(d_full))) if d_full.size else None,
        negative_denominator_count=int(np.count_nonzero(d_full < -denominator_atol)),
        uncoupled_near_zero_count=int(np.count_nonzero(near)),
        residual_norm=res, residual_definition=residual_definition,
        residual_acceptance_limit=equation_limit, unshifted_residual_norm=unshifted_residual,
        energy_imaginary=float(hyll.imag), projected_energy_imaginary=float(projected.imag),
        first_order_norm=norm1, solve_seconds=time.perf_counter()-start, controls=controls,
        regularizer=regularizer, effective_regularizer=effective, level_shift=epsilon,
        regularization_basis=basis, actual_denominator_basis=actual_basis,
        minimum_regularization_denominator=float(np.min(d)) if d.size else None,
        minimum_abs_regularization_denominator=float(np.min(np.abs(d))) if d.size else None,
        negative_regularization_denominator_count=int(np.count_nonzero(d < 0.0)),
        suppressed_regularization_direction_count=suppressed_count,
        energy_definition="original_Hylleraas_shift_corrected",
        sub_energy_definition="projected" if effective == "none" else "Hylleraas_row_allocation",
        sigma_power=(1 if regularizer == "sigma1" else 2 if regularizer == "sigma2" else None),
        sigma_parameterization="sigma=epsilon**(-p); epsilon is in Eh" if regularizer.startswith("sigma") else None,
        **metric_info, **extra)
    return CASPT2Result(
        matrices.reference_energy, matrices.fock_reference_energy, e2, float(hyll.real),
        t, 1.0/(1.0+norm1), sector_energies, diagnostics,
        matrices if keep_matrices else None, e_projected=float(projected.real),
        e_shift_correction=float(hyll.real-projected.real), regularizer=regularizer,
        level_shift=epsilon, regularization_basis=basis,
        sub_projected_eners=projected_sectors)


_APPROXIMATIONS = frozenset(("full", "caspt2d", "partial_ccvv"))
_IPEA_DEFINITIONS = frozenset(("spinor_diagonal", "molcas"))
_IPEA_BASES = frozenset(("input", "pseudocanonical"))


def _variant_options(approximation, ipea_shift, ipea_definition, ipea_basis):
    aliases = {"diagonal": "caspt2d", "caspt2-d": "caspt2d"}
    if isinstance(approximation, str):
        approximation = aliases.get(approximation.lower(), approximation.lower())
    if not isinstance(approximation, str) or approximation not in _APPROXIMATIONS:
        raise ValueError("approximation must be full, caspt2d, or partial_ccvv")
    ipea_shift = _positive_float(ipea_shift, "ipea_shift")
    if not isinstance(ipea_definition, str) or ipea_definition not in _IPEA_DEFINITIONS:
        raise ValueError("ipea_definition must be 'spinor_diagonal' or 'molcas'")
    if not isinstance(ipea_basis, str) or ipea_basis not in _IPEA_BASES:
        raise ValueError("ipea_basis must be 'input' or 'pseudocanonical'")
    return approximation, ipea_shift, ipea_definition, ipea_basis


def _family_active_rotation(family, coordinates, active_unitary):
    r"""Induced IC coordinate transformation |Phi_new> = |Phi_input> T.

    C'_p=sum_q C_q U_qp, D'_p=sum_q U_qp* D_q. Ordered same-kind
    pairs transform by an exterior product. Core/external columns and the
    reference state are fixed. No high-rank RDM or MPS is rotated.
    """
    coords = np.asarray(coordinates)
    size = len(coords)
    if size == 0:
        return np.empty((0, 0), dtype=np.complex128)
    out = np.ones((size, size), dtype=np.complex128)
    nf = len(family.sector)
    for k in range(nf):
        out *= coords[:, k, None] == coords[None, :, k]
    kinds = dict((label, kind) for kind, label in re.findall(r"([CD])\[([a-z])\]", family.expression))
    done = set()
    for p, q in family.active_pairs:
        if kinds[family.active[p]] != kinds[family.active[q]]:
            raise CASPT2NumericalError("antisymmetry pair mixes creation and annihilation indices")
        u = active_unitary if kinds[family.active[p]] == "C" else active_unitary.conj()
        op, oq = coords[:, nf+p, None], coords[:, nf+q, None]
        np_, nq = coords[None, :, nf+p], coords[None, :, nf+q]
        out *= u[op, np_] * u[oq, nq] - u[op, nq] * u[oq, np_]
        done.update((p, q))
    for p, label in enumerate(family.active):
        if p not in done:
            u = active_unitary if kinds[label] == "C" else active_unitary.conj()
            out *= u[coords[:, nf+p, None], coords[None, :, nf+p]]
    return out


def build_ipea_matrix(matrices, ipea_shift, *, definition="spinor_diagonal",
                      basis="pseudocanonical", atol=1e-10, rtol=1e-9):
    r"""Return (Delta_IPEA, diagnostics) in the original IC coordinates.

    This is an EXPLICITLY DEFINED native-spinor extension of the diagonal
    occupation prescription, NOT a claim of OpenMolcas scalar equivalence.
    In the specified orbital gauge let n_p=<C_p D_p> in [0,1]. For each
    primitive generator O_mu in _FAMILIES,

      w_mu = sum_(active creation p) n_p
             + sum_(active annihilation q) (1-n_q),
      Delta_mu,nu = delta_mu,nu * lambda * w_mu * S_mu,mu.

    The capacity-one factors reduce to lambda/2 * [sum D_created +
    sum(2-D_removed)] when n_{p alpha}=n_{p beta}=D_pp/2. This statement
    concerns occupation factors, NOT equality of two differently contracted
    spin-free/spinor matrices. IPEA is a coordinate-defined modification:
    changing an overcomplete IC generating set or its orthogonalization can
    change its numerical value (Nishimoto, JCP 158, 174112, 2023).

    _FAMILIES (including explicit singles), ordered same-kind pairs, and
    canonical metric projection fix that ambiguity for THIS implementation.
    Delta is added before the original metric-null-space projection. It is
    part of H0, not a level shift to be subtracted from the energy.

    With basis='pseudocanonical', diagonalize F_AA, form the diagonal
    prescription in the corresponding induced IC coordinates, then pull it
    back unitarily. The reference MPS and input RDMs are never modified.
    Exactly degenerate F_AA eigenvectors do not define a unique IPEA gauge;
    use basis='input' with a deliberately prepared orbital basis if needed.
    """
    _, lam, definition, basis = _variant_options("full", ipea_shift, definition, basis)
    atol = _positive_float(atol, "atol")
    rtol = _positive_float(rtol, "rtol")
    S = np.asarray(matrices.overlap)
    n = len(matrices.rhs)
    if S.shape != (n, n):
        raise ValueError("IPEA requires a square IC metric matching the source")
    _hermitian(S, "IPEA input overlap", atol, rtol)
    delta = np.zeros((n, n), dtype=np.complex128)
    info = dict(ipea_shift=lam, ipea_definition=definition, ipea_basis=basis,
                ipea_scalar_openmolcas_equivalence_validated=False,
                ipea_generator_basis="_FAMILIES:explicit_singles+ordered_spinor_doubles",
                ipea_metric_policy="unmodified_S_canonical_range",
                ipea_added_to_h0=True)
    if lam == 0.0:
        return delta, {**info, "ipea_matrix_norm": 0.0}
    if definition == 'molcas':
        raise NotImplementedError(
            "ipea_definition='molcas' is available only in the matrix-free "
            "native-spinor scalar-limit path")
    required = ("ncore", "ncas", "nvirt")
    if any(key not in matrices.diagnostics for key in required):
        raise ValueError("nonzero IPEA needs ncore/ncas/nvirt and IC generator metadata from build_caspt2_matrices")
    nc, na, nv = (_int(matrices.diagnostics[key], key) for key in required)
    f = _finite(matrices.fock_mo, "IPEA Fock matrix")
    if f.shape != (nc+na+nv, nc+na+nv):
        raise ValueError("IPEA Fock and orbital partition disagree")
    if matrices.active_dm1 is None or np.shape(matrices.active_dm1) != (na, na):
        raise ValueError("nonzero IPEA needs the raw active one-particle RDM in CASPT2Matrices.active_dm1")
    gamma = _finite(matrices.active_dm1, "IPEA active density")
    _hermitian(gamma, "IPEA active density", atol, rtol)
    if len(matrices.family_slices) != len(_FAMILIES) or len(matrices.coordinates) != len(_FAMILIES):
        raise ValueError("IPEA requires the exact primitive IC family metadata; an arbitrary basis is insufficient")
    if na == 0 or n == 0:
        return delta, {**info, "ipea_matrix_norm": 0.0, "ipea_active_occupations": []}
    fa = f[nc:nc+na, nc:nc+na]
    _hermitian(fa, "IPEA active Fock", atol, rtol)
    if basis == "pseudocanonical":
        eigenvalues, u = linalg.eigh(0.5*(fa+fa.conj().T))
    else:
        eigenvalues, u = np.diag(fa).real.copy(), np.eye(na, dtype=np.complex128)
    minimum_separation = (float(np.min(np.abs(np.diff(np.sort(eigenvalues)))))
                          if na > 1 else None)
    gauge_tolerance = atol + rtol*max(1.0, _maxabs(eigenvalues))
    gauge_degenerate = bool(
        basis == "pseudocanonical"
        and minimum_separation is not None
        and minimum_separation <= gauge_tolerance
    )
    warning = (
        "Nonzero IPEA uses the documented native-spinor diagonal prescription in a fixed "
        "primitive IC basis. This is not a verified reproduction of OpenMolcas spin-free IPEA; "
        "IPEA is basis/contraction dependent. See build_ipea_matrix()."
    )
    if gauge_degenerate:
        warning += (
            " The active Fock spectrum is degenerate within the numerical tolerance, so the "
            "pseudocanonical IPEA gauge is not unique; prepare a fixed orbital gauge and use "
            "ipea_basis='input' for reproducible comparisons."
        )
    warnings.warn(warning, CASPT2IPEAWarning, stacklevel=2)
    # raw SGF gamma[p,q]=<C_p D_q>, NOT the transposed chemists' density.
    transformed_gamma = u.T @ gamma @ u.conj()
    occup0 = np.diag(transformed_gamma).real
    if np.min(occup0) < -atol-rtol or np.max(occup0) > 1.0+atol+rtol:
        raise CASPT2NumericalError("IPEA spinor occupations are outside [0,1]")
    occup = np.clip(occup0, 0.0, 1.0)
    info.update(ipea_active_occupations=occup.tolist(),
                ipea_occupation_roundoff_projection=float(np.max(abs(occup-occup0), initial=0.0)),
                ipea_active_fock_eigenvalues=eigenvalues.tolist(),
                ipea_min_active_fock_separation=minimum_separation,
                ipea_active_fock_gauge_tolerance=gauge_tolerance,
                ipea_pseudocanonical_gauge_degenerate=gauge_degenerate)
    covered = np.zeros(n, dtype=np.int64)
    weights_by_family = {}
    max_transform_error = 0.0
    dimensions = dict(I=nc, A=na, E=nv)
    for family, sl, coords in zip(_FAMILIES, matrices.family_slices, matrices.coordinates):
        if not isinstance(sl, slice) or sl.step not in (None, 1) or sl.start is None or sl.stop is None:
            raise ValueError("invalid IPEA family slice")
        if not 0 <= sl.start <= sl.stop <= n:
            raise ValueError("IPEA family slice out of range")
        coords = np.asarray(coords)
        if coords.shape != (sl.stop-sl.start, len(family.variables)) or coords.dtype.kind not in 'iu':
            raise ValueError("IPEA family coordinates disagree with its slice")
        if not np.array_equal(coords, _coordinates(family, dimensions)):
            raise ValueError("IPEA requires the documented primitive generator order")
        covered[sl] += 1
        if sl.start == sl.stop:
            continue
        kinds = dict((label, kind) for kind, label in re.findall(r"([CD])\[([a-z])\]", family.expression))
        weight = np.zeros(len(coords))
        for k, label in enumerate(family.active):
            nk = occup[coords[:, len(family.sector)+k]]
            weight += nk if kinds[label] == 'C' else 1.0-nk
        weights_by_family[family.sector+"/"+family.name] = {
            "minimum": float(np.min(weight)), "maximum": float(np.max(weight)),
            "dimension": int(len(weight))}
        if not np.any(weight):
            continue
        sf = S[sl, sl]
        if basis == 'input':
            diag_s = np.diag(sf).real
            t = None
        else:
            t = _family_active_rotation(family, coords, u)
            err = _maxabs(t.conj().T @ t - np.eye(len(t)))
            max_transform_error = max(max_transform_error, err)
            if err > 10*(atol+rtol):
                raise CASPT2NumericalError("induced IC active-orbital transform is not unitary")
            diag_s = np.einsum('ij,ij->j', t.conj(), sf @ t).real
        if np.min(diag_s, initial=0.0) < -atol-rtol*max(1.0, _maxabs(sf)):
            raise CASPT2NumericalError("IPEA primitive norm is negative")
        diagonal = lam*weight*np.maximum(diag_s, 0.0)
        if t is None:
            idx = np.arange(sl.start, sl.stop)
            delta[idx, idx] = diagonal
        else:
            delta[sl, sl] = (t*diagonal[None, :]) @ t.conj().T
    if np.any(covered != 1):
        raise ValueError("IPEA family slices must cover each primitive direction exactly once")
    _hermitian(delta, "IPEA correction", atol, rtol)
    info.update(ipea_matrix_norm=float(linalg.norm(delta)),
                ipea_max_induced_unitary_error=max_transform_error,
                ipea_weights_by_family=weights_by_family,
                ipea_ccvv_correction_norm=float(linalg.norm(delta[matrices.sector_slices['ijrs'], :])))
    return delta, info


def _sector_orthogonalizer(matrices, controls):
    """Keep class labels through metric removal; use one global eigenvalue cutoff."""
    S, b = np.asarray(matrices.overlap), np.asarray(matrices.rhs)
    sectors = _checked_sector_slices(matrices)
    atol, rtol = controls['atol'], controls['rtol']
    _hermitian(S, 'IC overlap', atol, rtol)
    # Also validates that the sectors really are orthogonal.
    _caspt2d_operator(S, S, sectors, atol, rtol)
    parts, largest = {}, 0.0
    for key, sl in sectors.items():
        sg = 0.5*(S[sl, sl]+S[sl, sl].conj().T)
        eg, ug = linalg.eigh(sg) if len(sg) else (np.empty(0), np.empty((0,0), complex))
        parts[key] = (sl, eg, ug)
        largest = max(largest, float(np.max(eg, initial=0.0)))
    cutoff = max(controls['metric_atol'], controls['metric_rcond']*largest)
    rank = sum(int(np.count_nonzero(e > cutoff)) for _, e, _ in parts.values())
    X = np.zeros((len(b), rank), dtype=np.complex128)
    out_slices, details = {}, {}
    pos, null_source_sq = 0, 0.0
    for key, (sl, eg, ug) in parts.items():
        if np.min(eg, initial=0.0) < -(atol+rtol*max(1.0, largest)):
            raise CASPT2NumericalError('IC overlap has a significant negative eigenvalue')
        keep = eg > cutoff
        n = int(np.count_nonzero(keep))
        out_slices[key] = slice(pos, pos+n)
        xg = ug[:, keep]/np.sqrt(eg[keep])[None, :]
        # Small retained metric eigenvalues amplify eigenvector/eigenvalue
        # rounding. Re-whiten the ACTUAL Gram matrix without dropping any
        # additional direction or loosening the rank/reality thresholds.
        # Extended accumulation is local to this numerical check/refinement;
        # production tensors and amplitudes remain complex128.
        sg = 0.5*(S[sl, sl]+S[sl, sl].conj().T)
        before = _maxabs(xg.conj().T @ sg @ xg-np.eye(n))
        refinements = 0
        if n and before > max(1e-13, atol*0.1):
            sg_long = np.asarray(sg, dtype=np.clongdouble)
            for _ in range(2):
                xl = np.asarray(xg, dtype=np.clongdouble)
                gram = np.asarray(xl.conj().T @ sg_long @ xl, dtype=np.complex128)
                gram = 0.5*(gram+gram.conj().T)
                ge, gu = linalg.eigh(gram)
                if ge[0] <= 0.5 or ge[-1] >= 1.5:
                    raise CASPT2NumericalError('retained metric Gram is too ill-conditioned to refine safely')
                xg = xg @ ((gu/np.sqrt(ge)[None, :]) @ gu.conj().T)
                refinements += 1
        X[sl, pos:pos+n] = xg
        null_source_sq += float(linalg.norm(ug[:, ~keep].conj().T @ b[sl]))**2
        details[key] = dict(raw_dimension=sl.stop-sl.start, metric_rank=n,
                            gram_error_before_refinement=before,
                            gram_refinement_steps=refinements)
        pos += n
    null_source = math.sqrt(null_source_sq)
    if null_source > controls['source_atol']+rtol*linalg.norm(b):
        raise CASPT2NumericalError(f'source outside retained metric span: {null_source:.3e}')
    # Evaluate the final audit with extended accumulation as well. A BLAS
    # product in double precision is itself inaccurate for highly scaled X.
    xl = np.asarray(X, dtype=np.clongdouble)
    gram = xl.conj().T @ np.asarray(S, dtype=np.clongdouble) @ xl
    err = _maxabs(gram-np.eye(rank))
    if err > 10*(atol+rtol):
        raise CASPT2NumericalError(f'classwise metric orthogonalization failed: {err:.3e}')
    all_eigenvalues = (
        np.concatenate([eigenvalues for _, eigenvalues, _ in parts.values()])
        if parts else np.empty(0)
    )
    return X, out_slices, dict(metric_rank=rank, metric_cutoff=cutoff,
                               metric_minimum=(float(np.min(all_eigenvalues))
                                               if all_eigenvalues.size else None),
                               metric_maximum=(float(np.max(all_eigenvalues))
                                               if all_eigenvalues.size else None),
                               metric_orthogonalization_error=err,
                               metric_audit_accumulation_dtype=np.dtype(np.clongdouble).name,
                               discarded_metric_source_norm=null_source,
                               sector_metric_ranks=details)


def _orthogonal_problem(A, b, sectors, reference_energy=0.0):
    """A small already-orthonormal subsystem for the unchanged regularizer engine."""
    n = len(b)
    return CASPT2Matrices(np.eye(n, dtype=complex), A, b, np.empty((0,0)), 0.0,
                         reference_energy, sectors, (), (), {})


def _solve_partial_ccvv(matrices, model_A, *, controls, regularizer, level_shift,
                        regularization_basis):
    r"""Freeze CCVV's diagonal-model response, retain its coupling into all other rows.

    In an orthonormal class basis, C=ijrs and R=all remaining classes:
      t_C = -A_CC^{-1} b_C,
      A_RR t_R = -(b_R + A_RC t_C).
    The CCVV equation is not iterated. A_CR/A_RC are NOT both deleted.
    For regularized calculations the same existing regularizer is applied
    separately to the frozen CCVV problem and the driven remaining problem.
    This is an explicitly documented extension of the unshifted partial
    diagonal prescription. No claim about external-program shift defaults.
    """
    b = np.asarray(matrices.rhs)
    sectors = _checked_sector_slices(matrices)
    if 'ijrs' not in sectors:
        raise ValueError("partial_ccvv requires an explicit 'ijrs' sector")
    X, osl, metric_info = _sector_orthogonalizer(matrices, controls)
    ao, bo = _metric_congruence(X, model_A, b, metric_info)
    _hermitian(ao, 'partial CCVV orthogonal A', controls['atol'], controls['rtol'])
    ao = 0.5*(ao+ao.conj().T)
    csl = osl['ijrs']
    ci = np.arange(csl.start, csl.stop)
    remaining_parts = [
        np.arange(sl.start, sl.stop) for key, sl in osl.items() if key != 'ijrs'
    ]
    ri = (np.concatenate(remaining_parts) if remaining_parts
          else np.empty(0, dtype=np.int64))
    rsl, pos = {}, 0
    for key, sl in osl.items():
        if key != 'ijrs':
            rsl[key] = slice(pos, pos+sl.stop-sl.start)
            pos += sl.stop-sl.start
    common = dict(controls, regularizer=regularizer, level_shift=level_shift,
                  regularization_basis=regularization_basis, keep_matrices=False)
    # These systems have identity metrics. They must not acquire a second,
    # differently-scaled IC-null-space cutoff.
    common['metric_atol'], common['metric_rcond'] = 0.0, 0.0
    ac, ar = ao[np.ix_(ci, ci)], ao[np.ix_(ri, ri)]
    arc = ao[np.ix_(ri, ci)]
    bc, br = bo[ci], bo[ri]
    cres = _solve_caspt2_full(_orthogonal_problem(ac, bc, {'ijrs':slice(0,len(ci))}), **common)
    tc = cres.amplitudes
    effective_br = br + arc @ tc
    rres = _solve_caspt2_full(_orthogonal_problem(ar, effective_br, rsl), **common)
    tr = rres.amplitudes
    to = np.zeros(len(bo), dtype=complex)
    to[ci], to[ri] = tc, tr
    t = X @ to
    At = model_A @ t
    projected = np.vdot(b, t)
    hyll = np.vdot(t, At)+2*np.vdot(t,b).real
    limit = controls['atol']+controls['rtol']*max(1.0,abs(hyll.real),abs(projected.real))
    if not np.isfinite(hyll) or not np.isfinite(projected) or abs(hyll.imag)>limit:
        raise CASPT2NumericalError('partial CCVV Hylleraas energy is not finite and real')
    # b^dagger t need NOT be real for a complex constrained (not globally
    # stationary) response. Keep its imaginary part as a diagnostic, not an
    # erroneous reality gate on an intermediate quantity.
    constraint = float(np.vdot(tc, arc.conj().T @ tr).real)
    shift_correction = cres.e_shift_correction+rres.e_shift_correction
    if abs(hyll.real-projected.real-constraint-shift_correction)>10*limit:
        raise CASPT2NumericalError('partial CCVV energy/constraint/regularizer accounting failed')
    norm1 = float(np.vdot(to,to).real)
    sub = {key:float((np.vdot(t[sl],At[sl])+2*np.vdot(t[sl],b[sl]).real).real)
           for key,sl in sectors.items()}
    subp = {key:float(np.vdot(b[sl],t[sl]).real) for key,sl in sectors.items()}
    if abs(sum(sub.values())-hyll.real)>limit:
        raise CASPT2NumericalError('partial CCVV class allocations do not sum to Hylleraas energy')
    d = linalg.eigvalsh(ao) if len(bo) else np.empty(0)
    diagnostics = dict(matrices.diagnostics)
    diagnostics.update(metric_info)
    diagnostics.update(
        raw_dimension=len(b), approximation='partial_ccvv', ccvv_metric_rank=len(ci),
        remaining_metric_rank=len(ri), frozen_ccvv_amplitude_norm=float(linalg.norm(tc)),
        ccvv_to_remaining_coupling_norm=float(linalg.norm(arc)),
        ccvv_driving_term_norm=float(linalg.norm(arc@tc)),
        ccvv_full_equation_residual=float(linalg.norm((ao@to+bo)[ci])),
        remaining_unregularized_residual=float(linalg.norm((ao@to+bo)[ri])),
        unshifted_residual_norm=float(linalg.norm(ao@to+bo)),
        residual_norm=max(cres.diagnostics['residual_norm'],rres.diagnostics['residual_norm']),
        residual_definition='maximum of frozen-CCVV and driven-remainder defining-equation residuals',
        ccvv_solver_diagnostics=cres.diagnostics, remainder_solver_diagnostics=rres.diagnostics,
        first_order_norm=norm1, projected_energy_imaginary=float(projected.imag),
        energy_imaginary=float(hyll.imag),
        minimum_denominator=float(np.min(d)) if d.size else None,
        minimum_abs_denominator=float(np.min(abs(d))) if d.size else None,
        negative_denominator_count=int(np.count_nonzero(d < -controls['denominator_atol'])),
        energy_definition='full_model_Hylleraas_with_fixed_CCVV',
        regularizer=regularizer, effective_regularizer=('none' if level_shift==0 else regularizer),
        level_shift=level_shift, regularization_basis=regularization_basis,
        ccvv_diagonal_definition='exact_local_A_CC_spectrum_equivalent_to_canonical_core_virtual_MP2_denominators',
        partial_ccvv_memory_optimized=False,
        sub_energy_definition='Hylleraas_row_allocation_not_independent_class_energies')
    return CASPT2Result(matrices.reference_energy, matrices.fock_reference_energy,
                        float(hyll.real),float(hyll.real),t,1/(1+norm1),sub,diagnostics,
                        e_projected=float(projected.real), e_shift_correction=shift_correction,
                        regularizer=regularizer,level_shift=level_shift,
                        regularization_basis=regularization_basis,sub_projected_eners=subp,
                        approximation='partial_ccvv', e_constraint_correction=constraint)


def solve_caspt2(matrices, *, metric_atol=1e-12, metric_rcond=1e-10,
                 denominator_atol=1e-10, source_atol=1e-10,
                 atol=1e-10, rtol=1e-9, keep_matrices=False,
                 regularizer='none', level_shift=0.0,
                 regularization_basis='caspt2d', regularized_rcond=1e-12,
                 approximation='full', ipea_shift=0.0,
                 energy_functional='model',
                 ipea_definition='spinor_diagonal', ipea_basis='pseudocanonical'):
    r"""Solve full CASPT2, CASPT2-D, or partial-CCVV with optional IPEA/regularization.

    CASPT2-D uses A_D=sum_g P_g A P_g in the amplitude equation, not just
    as a regularization denominator. Its default variational energy uses
    A_D (OpenMolcas 24.10); energy_functional='full' uses full A (older
    BLOCK-enabled 48f9d80 build). Both diagnostics are always retained.
    partial_ccvv freezes diagonal-model ijrs amplitudes but keeps
    A_RC t_C in the remaining amplitude equation, and evaluates full-model
    Hylleraas energy. IPEA modifies H0 and is never shift-corrected away.
    See build_ipea_matrix for the explicitly chosen spinor IPEA convention.
    """
    approximation, lam, ipdef, ipbasis = _variant_options(
        approximation, ipea_shift, ipea_definition, ipea_basis)
    if energy_functional not in ('model', 'full'):
        raise ValueError("energy_functional must be 'model' or 'full'")
    reg, eps, regbasis = _regularization_options(regularizer, level_shift, regularization_basis)
    controls = dict(metric_atol=metric_atol,metric_rcond=metric_rcond,
                    denominator_atol=denominator_atol,source_atol=source_atol,
                    atol=atol,rtol=rtol,regularized_rcond=regularized_rcond)
    controls = {k:_positive_float(v,k) for k,v in controls.items()}
    if controls['metric_rcond']>=1 or controls['regularized_rcond']>=1:
        raise ValueError('metric_rcond and regularized_rcond must be smaller than one')
    kwargs = dict(controls,regularizer=reg,level_shift=eps,regularization_basis=regbasis)
    # Preserve the already-tested path bit for bit at default settings.
    if approximation=='full' and lam==0.0:
        result = _solve_caspt2_full(matrices,keep_matrices=keep_matrices,**kwargs)
        result.e_full_hylleraas=result.e_hylleraas
        result.e_model_hylleraas=result.e_hylleraas
        result.energy_functional=energy_functional
        result.e_no_ipea_hylleraas=result.e_hylleraas
        result.ipea_definition,result.ipea_basis=ipdef,ipbasis
        result.diagnostics.update(approximation='full',ipea_shift=0.0,
                                  ipea_definition=ipdef,ipea_basis=ipbasis)
        return result
    S, F, b = np.asarray(matrices.overlap),np.asarray(matrices.fock_matrix),np.asarray(matrices.rhs)
    if b.ndim!=1 or S.shape!=(len(b),len(b)) or F.shape!=S.shape:
        raise ValueError('CASPT2 matrices have incompatible shapes')
    for name,value in [('overlap',S),('projected Fock',F),('source',b)]:
        _finite(value,name)
    _hermitian(S,'overlap',atol,rtol); _hermitian(F,'projected Fock',atol,rtol)
    ef=complex(matrices.fock_reference_energy)
    if not np.isfinite(ef) or abs(ef.imag)>atol:
        raise ValueError('fock_reference_energy must be finite and real')
    A0=0.5*(F+F.conj().T)-ef.real*0.5*(S+S.conj().T)
    sectors=_checked_sector_slices(matrices)
    if lam:
        delta,ipinfo=build_ipea_matrix(matrices,lam,definition=ipdef,basis=ipbasis,atol=atol,rtol=rtol)
    else:
        delta=np.zeros_like(A0)
        ipinfo=dict(ipea_shift=0.0,ipea_definition=ipdef,ipea_basis=ipbasis)
    Afull=A0+delta
    model_A=(_caspt2d_operator(Afull,S,sectors,atol,rtol)
             if approximation=='caspt2d' else Afull)
    if approximation=='partial_ccvv':
        result=_solve_partial_ccvv(matrices,model_A,controls=controls,regularizer=reg,
                                  level_shift=eps,regularization_basis=regbasis)
    else:
        modified=replace(matrices,fock_matrix=model_A+ef.real*S,
                         diagnostics={**matrices.diagnostics,**ipinfo})
        result=_solve_caspt2_full(modified,keep_matrices=False,**kwargs)
    result.matrices=matrices if keep_matrices else None
    result.approximation,result.ipea_shift=approximation,lam
    result.ipea_definition,result.ipea_basis=ipdef,ipbasis
    t=result.amplitudes
    result.e_full_hylleraas=float((np.vdot(t,Afull@t)+2*np.vdot(t,b).real).real)
    result.e_model_hylleraas=result.e_hylleraas
    result.energy_functional=energy_functional
    if approximation=='caspt2d' and energy_functional=='full':
        result.e_constraint_correction=result.e_full_hylleraas-result.e_model_hylleraas
        result.e_corr=result.e_hylleraas=result.e_full_hylleraas
        At=Afull@t
        result.sub_eners={key:float((np.vdot(t[sl],At[sl])+2*np.vdot(t[sl],b[sl]).real).real)
                          for key,sl in sectors.items()}
        result.diagnostics['sub_energy_definition']='full_Hylleraas_row_allocation'
    noipea_A=(_caspt2d_operator(A0,S,sectors,atol,rtol)
               if approximation=='caspt2d' and energy_functional=='model' else A0)
    result.e_no_ipea_hylleraas=float((np.vdot(t,noipea_A@t)+2*np.vdot(t,b).real).real)
    result.diagnostics.update(ipinfo)
    result.diagnostics.update(
        approximation=approximation, ipea_shift=lam,ipea_definition=ipdef,ipea_basis=ipbasis,
        omitted_fock_coupling_norm=float(linalg.norm(Afull-model_A)),
        ipea_energy_functional_contribution=float(np.vdot(t,delta@t).real),
        ipea_included_in_hylleraas=True,
        e_full_hylleraas=result.e_full_hylleraas,e_no_ipea_hylleraas=result.e_no_ipea_hylleraas,
        e_model_hylleraas=result.e_model_hylleraas,
        energy_functional=energy_functional,
        e_constraint_correction=result.e_constraint_correction,
        caspt2d_is_energy_approximation=(approximation=='caspt2d'),
        regularization_denominator_basis_is_separate_option=True)
    if approximation!='partial_ccvv':
        result.diagnostics['energy_definition']=energy_functional+'_Hylleraas_including_IPEA_shift_corrected'
    return result


def _representation_options(approximation, ipea, definition, basis, representation):
    if representation != 'spinor':
        raise ValueError("Only native representation='spinor' is supported; scalar/paired backends were removed")
    definition = 'spinor_diagonal' if definition is None else definition
    approximation, ipea, definition, basis = _variant_options(
        approximation, ipea, definition, basis)
    return approximation, ipea, definition, basis


def caspt2_from_integrals(h1e, eri, ncore, ncas, nelecas, pdms, *,
                         fock_mo=None, constant=0.0, reference_energy=None,
                         wick_backend="block2", contraction_backend="numpy",
                         metric_atol=1e-12, metric_rcond=1e-10, denominator_atol=1e-10,
                         source_atol=1e-10, atol=1e-10, rtol=1e-9, check_rdms=True,
                         max_memory_mb=2048.0, tile_size=16, tile_memory_mb=32.0,
                         keep_matrices=False, ipea_shift=0.0, level_shift=0.0,
                         regularizer="none", regularization_basis="caspt2d",
                         regularized_rcond=1e-12, approximation="full",
                         energy_functional="model",
                         ipea_definition=None, ipea_basis="pseudocanonical",
                         representation="spinor", solver_backend="dense",
                         maxiter=100, conv_tol=1e-10):
    """Exact-RDM integral API; shifts are in Eh.

    ``representation='spinor'`` (default) uses spinor orbital counts and raw
    SGF RDMs. No spin tracing or spin-adapted scalar solver is used.
    ``solver_backend='dense'`` retains the small-system correctness oracle;
    ``'matrix_free'`` contracts the same spinor Wick equations without global
    IC matrices. Both return amplitudes in the native primitive spinor basis.
    Molcas-compatible nonzero IPEA is restricted to the matrix-free scalar
    limit and requires ``ipea_definition='molcas', ipea_basis='input'``.
    """
    approximation, ipea_shift, ipea_definition, ipea_basis = _representation_options(
        approximation, ipea_shift, ipea_definition, ipea_basis, representation)
    regularizer, level_shift, regularization_basis = _regularization_options(
        regularizer, level_shift, regularization_basis)
    if solver_backend == 'matrix_free':
        from ._caspt2_spinor import NativeCASPT2
        if keep_matrices:
            raise ValueError('keep_matrices requires solver_backend="dense"')
        model = NativeCASPT2(
            h1e, eri, ncore, ncas, nelecas, pdms, fock_mo=fock_mo,
            constant=constant, reference_energy=reference_energy,
            wick_backend=wick_backend, contraction_backend=contraction_backend,
            metric_atol=metric_atol, metric_rcond=metric_rcond,
            atol=atol, rtol=rtol, check_rdms=check_rdms,
            max_memory_mb=max_memory_mb)
        return model.solve(approximation=approximation, ipea_shift=ipea_shift,
                           regularizer=regularizer, level_shift=level_shift,
                           regularization_basis=regularization_basis,
                           energy_functional=energy_functional,
                           maxiter=maxiter, conv_tol=conv_tol,
                           denominator_atol=denominator_atol, source_atol=source_atol,
                           ipea_definition=ipea_definition, ipea_basis=ipea_basis)
    if solver_backend != 'dense':
        raise ValueError('solver_backend must be dense or matrix_free')
    matrices = build_caspt2_matrices(
        h1e, eri, ncore, ncas, nelecas, pdms, fock_mo=fock_mo, constant=constant,
        reference_energy=reference_energy, wick_backend=wick_backend,
        contraction_backend=contraction_backend, atol=atol, rtol=rtol,
        check_rdms=check_rdms, max_memory_mb=max_memory_mb, tile_size=tile_size,
        tile_memory_mb=tile_memory_mb)
    return solve_caspt2(matrices, metric_atol=metric_atol, metric_rcond=metric_rcond,
                       denominator_atol=denominator_atol, source_atol=source_atol,
                       atol=atol, rtol=rtol, keep_matrices=keep_matrices,
                       regularizer=regularizer, level_shift=level_shift,
                       regularization_basis=regularization_basis,
                       regularized_rcond=regularized_rcond, approximation=approximation,
                       energy_functional=energy_functional,
                       ipea_shift=ipea_shift, ipea_definition=ipea_definition,
                       ipea_basis=ipea_basis)


class WickX2CCASPT2(_pyscf_lib.StreamObject if _pyscf_lib is not None else object):
    """socutils/PySCF adapter. Orbitals, MPS and input integrals must share a basis.

    Input orbitals are kept. All inter-class Fock couplings are retained for
    approximation='full'. Supplied ERIs must
    already be in the ``mo_coeff`` basis. No orbital-energy vector is needed. Set
    ``fock_mo_input`` for a persistent matched-H0 regression input, or pass
    ``fock_mo=`` to one kernel/run call; ``fock_mo`` on the object records
    the matrix used by the most recent result.
    """
    def __init__(self, mc, frozen=0, *, representation='spinor'):
        if representation != 'spinor':
            raise ValueError('Only native spinor CASPT2 is supported')
        if _pyscf_lib is None:
            raise ImportError("the StreamObject adapter requires PySCF; use the low-level integral API otherwise")
        from . import nevpt2_utils as utils
        if utils._has_frozen_orbitals(frozen) or utils._has_frozen_orbitals(getattr(mc, "frozen", None)):
            raise NotImplementedError("nonzero frozen spinors are not supported")
        self._mc = mc
        self._scf = mc._scf
        self.mol = mc.mol
        self.verbose = getattr(mc, "verbose", self.mol.verbose)
        self.stdout = getattr(mc, "stdout", self.mol.stdout)
        self.root = 0
        self.frozen = 0
        self.mo_coeff = mc.mo_coeff
        self.wick_backend = "block2"
        self.contraction_backend = "numpy"
        self.representation = representation
        self.solver_backend = 'dense'
        self.maxiter = 100
        self.conv_tol = 1e-10
        self.ipea_shift = 0.0
        self.ipea_definition = "spinor_diagonal"
        self.ipea_basis = "pseudocanonical"
        self.approximation = "full"
        self.e_constraint_correction = 0.0
        self.e_full_hylleraas = None
        self.e_no_ipea_hylleraas = None
        self.e_model_hylleraas = None
        self.energy_functional = 'model'
        self.level_shift = 0.0
        self.regularizer = "none"
        self.regularization_basis = "caspt2d"
        self.regularized_rcond = 1e-12
        self.e_projected = None
        self.e_hylleraas = None
        self.e_shift_correction = None
        self.sub_projected_eners = {}
        self.metric_atol = 1e-12
        self.metric_rcond = 1e-10
        self.denominator_atol = 1e-10
        self.source_atol = 1e-10
        self.atol = 1e-10
        self.rtol = 1e-9
        self.max_memory_mb = float(getattr(mc, "max_memory", 2048))
        self.tile_size = 16
        self.tile_memory_mb = 32.0
        self.check_rdms = True
        self.keep_matrices = False
        self.result = None
        self.eris = None
        self.e_corr = None
        self.reference_energy = None
        self.fock_reference_energy = None
        # Optional user-supplied H0 for matched scalar/cross-code regressions.
        # Keep it separate from fock_mo, which records the matrix actually used
        # by the most recent calculation.
        self.fock_mo_input = None
        self.fock_mo = None
        self.sub_eners = {}
        self.diagnostics = {}
        self.amplitudes = None
        self.reference_weight = None
        self._keys = set(self.__dict__)

    @property
    def e_tot(self):
        return None if self.e_corr is None else self.reference_energy + self.e_corr

    def kernel(self, mc=None, mo_coeff=None, pdms=None, eris=None, root=None, *,
               fock_mo=None, constant=None, wick_backend=None, contraction_backend=None,
               regularizer=None, level_shift=None, regularization_basis=None,
               approximation=None, ipea_shift=None, ipea_definition=None, ipea_basis=None,
               energy_functional=None):
        from . import nevpt2_utils as utils
        from .spinor_helper import _SpinorERIs
        from pyscf.lib import logger
        mc = self._mc if mc is None else mc
        if utils._has_frozen_orbitals(self.frozen) or utils._has_frozen_orbitals(getattr(mc, "frozen", None)):
            raise NotImplementedError("nonzero frozen spinors are not supported")
        reg, eps, regbasis = _regularization_options(
            self.regularizer if regularizer is None else regularizer,
            self.level_shift if level_shift is None else level_shift,
            self.regularization_basis if regularization_basis is None else regularization_basis)
        approx, ipea, ipdef, ipbasis = _representation_options(
            self.approximation if approximation is None else approximation,
            self.ipea_shift if ipea_shift is None else ipea_shift,
            self.ipea_definition if ipea_definition is None else ipea_definition,
            self.ipea_basis if ipea_basis is None else ipea_basis, self.representation)
        root = _int(self.root if root is None else root, "root")
        mo = np.asarray(mc.mo_coeff if mo_coeff is None else mo_coeff)
        if pdms is None:
            # Only a converged, still-open SGF DMRG driver is implicitly read.
            # Other exact solvers can supply pdms explicitly in raw SGF order.
            pdms = utils.make_dm1234(mc.fcisolver, root=root)
        if eris is None:
            eris = utils._dense_eris_from_mc(mc, mo)
        if not isinstance(eris, _SpinorERIs):
            raise TypeError("CASPT2 requires full dense spinor_helper._SpinorERIs, not NEVPT2 compact blocks")
        if (eris.ncore, eris.ncas, eris.nmo) != (int(mc.ncore), int(mc.ncas), mo.shape[1]):
            raise ValueError("ERI partition and supplied MO basis disagree")
        if constant is None:
            energy_nuc = getattr(mc._scf, "energy_nuc", None)
            if not callable(energy_nuc):
                energy_nuc = getattr(mc.mol, "energy_nuc", None)
            if not callable(energy_nuc):
                raise ValueError("supply the physical scalar constant explicitly")
            constant = energy_nuc()
        selected_fock_mo = self.fock_mo_input if fock_mo is None else fock_mo
        result = caspt2_from_integrals(
            eris.h1e, eris.pppp, eris.ncore, eris.ncas,
            utils._total_nelec(mc.nelecas), pdms,
            fock_mo=selected_fock_mo, constant=constant,
            reference_energy=utils._reference_energy(mc, root),
            wick_backend=self.wick_backend if wick_backend is None else wick_backend,
            contraction_backend=self.contraction_backend if contraction_backend is None else contraction_backend,
            metric_atol=self.metric_atol, metric_rcond=self.metric_rcond,
            denominator_atol=self.denominator_atol, source_atol=self.source_atol,
            atol=self.atol, rtol=self.rtol, check_rdms=self.check_rdms,
            max_memory_mb=self.max_memory_mb, tile_size=self.tile_size,
            tile_memory_mb=self.tile_memory_mb, keep_matrices=self.keep_matrices,
            ipea_shift=ipea, level_shift=eps, regularizer=reg,
            regularization_basis=regbasis, regularized_rcond=self.regularized_rcond,
            approximation=approx, ipea_definition=ipdef, ipea_basis=ipbasis,
            energy_functional=self.energy_functional if energy_functional is None else energy_functional,
            representation=self.representation, solver_backend=self.solver_backend,
            maxiter=self.maxiter, conv_tol=self.conv_tol)
        self.regularizer, self.level_shift, self.regularization_basis = reg, eps, regbasis
        self.approximation, self.ipea_shift = approx, ipea
        self.ipea_definition, self.ipea_basis = ipdef, ipbasis
        self.root, self.mo_coeff, self.eris, self.result = root, mo, eris, result
        self.fock_mo = (result.matrices.fock_mo if result.matrices is not None else
                        build_generalized_fock(eris.h1e, eris.pppp, pdms[0], eris.ncore)[0]
                        if selected_fock_mo is None else np.asarray(selected_fock_mo))
        for attr in ("e_corr", "reference_energy", "fock_reference_energy", "sub_eners",
                     "diagnostics", "amplitudes", "reference_weight", "e_projected",
                     "e_hylleraas", "e_shift_correction", "sub_projected_eners",
                     "e_constraint_correction", "e_full_hylleraas", "e_no_ipea_hylleraas",
                     "e_model_hylleraas", "energy_functional"):
            setattr(self, attr, getattr(result, attr))
        logger.note(self, "root %d SS-X2C-CASPT2: E(ref)=%.14f E(F0)=%.14f E(2)=%.14f E(total)=%.14f",
                    root, self.reference_energy, self.fock_reference_energy, self.e_corr, self.e_tot)
        logger.note(self, "CASPT2 approximation=%s IPEA=%.8g Eh definition=%s basis=%s; "
                    "amplitude-constraint correction=%.14f", approx, ipea, ipdef, ipbasis,
                    self.e_constraint_correction)
        logger.note(self, "CASPT2 regularizer=%s epsilon=%.8g Eh denominator_basis=%s; "
                    "E(projected)=%.14f shift correction=%.14f",
                    reg, eps, regbasis, self.e_projected, self.e_shift_correction)
        logger.note(self, "CASPT2 metric rank %d/%d; residual %.3e; reference weight %.10f",
                    result.diagnostics["metric_rank"], result.diagnostics["raw_dimension"],
                    result.diagnostics["residual_norm"], result.reference_weight)
        for key, value in self.sub_eners.items():
            logger.info(self, "CASPT2 allocated corrected E(%s) = %.14f (classes are coupled)", key, value)
        return self.e_corr


X2CCASPT2 = WickX2CCASPT2
