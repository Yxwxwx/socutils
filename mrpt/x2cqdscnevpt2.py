#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
r"""Bloch and canonical Van Vleck QD extensions of X2C-SC-NEVPT2.

The stored matrix uses the source-row convention

.. math::

   H^{B}_{IJ} = \delta_{IJ} E_I^{(0)}
       - \sum_{\alpha\in\mathcal R_I}
       \frac{\langle\Psi_I|T_\alpha^{I\dagger}T_\alpha^I|\Psi_J\rangle}
            {\Delta_\alpha^I}.

Each row owns its state-specific CanonStep-1 semicanonical integrals,
strongly-contracted perturbers, retained mask, and positive strict-SI Dyall
gap.  This complete directed Bloch matrix is always retained.  The default
canonical Van Vleck (HQD) representation is formed only afterwards, class by
class, according to Lang, Sivalingam, and Neese, J. Chem. Phys. 152, 014109
(2020), Eqs. (45)--(46):

.. math::

   H^{\mathrm{VV}} = \frac12\left(H^B + H^{B\dagger}\right).

No additional SOC operator is introduced: SOC is already present in the
one-step X2C/X2CAMF complex-spinor reference Hamiltonian.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import gc
import time
from typing import Any

import numpy as np
from pyscf import lib
from pyscf.lib import logger
from scipy.linalg import eig, eigh

from . import nevpt2_utils as _utils
from . import spinor_helper
from . import x2cscnevpt2 as _ss


__all__ = [
    "QDSCNEVPT2Result",
    "QDBlochSCNEVPT2Result",
    "WickX2CQDSCNEVPT2",
    "X2CQDSCNEVPT2",
    "WickX2CQDBlochSCNEVPT2",
    "X2CQDBlochSCNEVPT2",
    "adjoint_transition_pdm",
    "make_transition_overlap",
    "make_transition_dm1234",
    "make_transition_rdm1",
    "make_transition_rdm2",
    "make_transition_rdm3",
    "make_transition_rdm4",
    "validate_transition_pdms",
]


_QD_TYPES = frozenset(("van_vleck", "bloch"))
_QD_KERNEL_KEYWORDS = frozenset(
    (
        "mc",
        "roots",
        "qd_type",
        "state_pdms",
        "transition_pdms",
        "model_overlap",
        "mo_coeff",
        "eris",
        "eris_basis",
        "denominator_mode",
        "contraction_backend",
        "validate_reverse_transition",
        "eigenvalue_imag_warn",
        "eigenvalue_imag_error",
    )
)


def _normalize_qd_type(value) -> str:
    """Return the canonical effective-Hamiltonian representation name."""

    if not isinstance(value, str):
        choices = ", ".join(sorted(_QD_TYPES))
        raise ValueError(f"qd_type must be one of: {choices}")
    normalized = value.strip().casefold().replace("-", "_")
    aliases = {
        "bloch": "bloch",
        "van_vleck": "van_vleck",
        "vanvleck": "van_vleck",
    }
    try:
        return aliases[normalized]
    except KeyError as error:
        choices = ", ".join(sorted(_QD_TYPES))
        raise ValueError(f"qd_type must be one of: {choices}") from error


def _retain_qd_row_arrays(subspace_arrays):
    """Detach only audited row tensors needed after diagonal preparation."""

    retained = {}
    for key in _utils.SUBSPACE_ORDER:
        arrays = subspace_arrays[key]
        retained[key] = {
            "norm": np.array(np.asarray(arrays["norm"]).real, copy=True),
            "denominator": np.array(
                np.asarray(arrays["denominator"]).real, copy=True
            ),
            "ordered": np.array(arrays["ordered"], dtype=bool, copy=True),
            "nonzero": np.array(arrays["nonzero"], dtype=bool, copy=True),
        }
    return retained


def _nested_array_nbytes(values):
    return int(
        sum(
            np.asarray(array).nbytes
            for arrays in values.values()
            for array in arrays.values()
        )
    )


# Transition-density operations are method-independent and re-exported here
# for compatibility with the original QD module API.
adjoint_transition_pdm = _utils.adjoint_transition_pdm
_transition_root_kets = _utils._transition_root_kets
_make_transition_rdm = _utils._make_transition_rdm
make_transition_rdm1 = _utils.make_transition_rdm1
make_transition_rdm2 = _utils.make_transition_rdm2
make_transition_rdm3 = _utils.make_transition_rdm3
make_transition_rdm4 = _utils.make_transition_rdm4
make_transition_dm1234 = _utils.make_transition_dm1234
_make_transition_dm123 = _utils._make_transition_dm123
_transition_contraction_pdms = _utils._transition_contraction_pdms
_complex_scalar = _utils._complex_scalar
_finite_nonnegative = _utils._finite_nonnegative
_make_transition_overlap = _utils._make_transition_overlap
make_transition_overlap = _utils.make_transition_overlap
_complex_pair = _utils._complex_pair
validate_transition_pdms = _utils.validate_transition_pdms


def _evaluate_transition_perturber_couplings(
    eris_row,
    transition_pdms,
    overlap,
    *,
    row_root,
    column_root,
    row_nonzero=None,
    row_norm=None,
    partner_norm=None,
    norm_tol=1.0e-14,
    zero_norm_atol=1.0e-10,
    zero_norm_rtol=1.0e-9,
    contraction_backend=_utils._DEFAULT_CONTRACTION_BACKEND,
    return_diagnostics=False,
):
    """Evaluate complex ``<I|T_I^dagger T_I|J>`` tensors for all classes."""

    norm_tol = _finite_nonnegative(norm_tol, name="norm_tol")
    zero_norm_atol = _finite_nonnegative(
        zero_norm_atol, name="zero_norm_atol"
    )
    zero_norm_rtol = _finite_nonnegative(
        zero_norm_rtol, name="zero_norm_rtol"
    )
    if row_nonzero is not None and (
        row_norm is None or partner_norm is None
    ):
        raise ValueError(
            "discarded row perturbers require row_norm and partner_norm "
            "for a Cauchy-bound audit"
        )
    contraction_backend = _utils._normalize_contraction_backend(
        contraction_backend
    )
    transition_pdms = _transition_contraction_pdms(transition_pdms)
    if contraction_backend == "pytblis":
        _utils._validate_tblis_operand_dtypes(
            [
                (f"h{key}", eris_row.get_h1eff(key))
                for key in _utils._H1_KEYS
            ]
            + [
                (f"w{key}", eris_row.get_phys(key))
                for key in _utils._W_KEYS
            ]
            + [
                (f"dm{rank}", density)
                for rank, density in enumerate(transition_pdms, 1)
            ]
        )
    overlap = _complex_scalar(overlap, name="transition overlap")
    equations = _ss._compile_wick_equations()
    wick_globals = {"np": _utils._wick_einsum_namespace(contraction_backend)}
    base_context = _utils._execution_context(
        eris_row, transition_pdms, overlap=overlap
    )
    couplings = {}
    diagnostics = {}
    for key in _utils.SUBSPACE_ORDER:
        shape = _utils._free_index_shape(key, eris_row)
        integral_dtypes = tuple(
            eris_row.get_h1eff(block).dtype for block in _utils._H1_KEYS
        ) + tuple(eris_row.get_phys(block).dtype for block in _utils._W_KEYS)
        dtype = np.result_type(
            np.complex128,
            *integral_dtypes,
            *(np.asarray(dm).dtype for dm in transition_pdms),
        )
        coupling = np.zeros(shape, dtype=dtype)
        local_context = dict(base_context)
        local_context["norm"] = coupling
        exec(
            equations.transition_norm_code[key],
            wick_globals,
            local_context,
        )
        ordered = _utils._strict_pair_mask(key, shape)
        selected = coupling[ordered]
        if not np.all(np.isfinite(selected)):
            raise FloatingPointError(
                f"row root {row_root}, column root {column_root}, subspace "
                f"{key}: transition coupling contains non-finite values"
            )

        if row_nonzero is None:
            nonzero = ordered
        else:
            try:
                nonzero = np.asarray(row_nonzero[key], dtype=bool)
            except (KeyError, TypeError) as error:
                raise ValueError(
                    f"row_nonzero has no mask for subspace {key}"
                ) from error
            if nonzero.shape != shape or np.any(nonzero & ~ordered):
                raise ValueError(
                    f"row_nonzero mask for subspace {key} is inconsistent"
                )
        zero_norm = ordered & ~nonzero
        if row_nonzero is None:
            cauchy_diagnostics = {
                "zero_norm_count": 0,
                "zero_norm_maximum_absolute_value": 0.0,
                "zero_norm_maximum_absolute_index": [],
                "zero_norm_row_norm_at_maximum": 0.0,
                "zero_norm_partner_norm_at_maximum": 0.0,
                "zero_norm_cauchy_bound_at_maximum": 0.0,
                "zero_norm_numerical_allowance_at_maximum": 0.0,
                "zero_norm_acceptance_limit_at_maximum": 0.0,
                "zero_norm_physical_excess_at_maximum": 0.0,
                "zero_norm_maximum_acceptance_excess": 0.0,
                "zero_norm_maximum_acceptance_excess_index": [],
                "zero_norm_maximum_cauchy_ratio": 0.0,
                "zero_norm_maximum_finite_cauchy_ratio": 0.0,
                "zero_norm_cauchy_ratio_unbounded_count": 0,
                "zero_norm_cauchy_gate_passed": True,
            }
        else:
            try:
                row_norm_key = np.asarray(row_norm[key])
            except (KeyError, TypeError) as error:
                raise ValueError(
                    f"row_norm has no tensor for subspace {key}"
                ) from error
            try:
                partner_norm_key = np.asarray(partner_norm[key])
            except (KeyError, TypeError) as error:
                raise ValueError(
                    f"partner_norm has no tensor for subspace {key}"
                ) from error
            cauchy_diagnostics = _validate_discarded_couplings_cauchy(
                coupling,
                zero_norm,
                row_norm_key,
                partner_norm_key,
                row_root=row_root,
                column_root=column_root,
                subspace=key,
                norm_tol=norm_tol,
                atol=zero_norm_atol,
                rtol=zero_norm_rtol,
            )
        couplings[key] = coupling
        diagnostics[key] = {
            "maximum_absolute_value": float(
                np.max(np.abs(selected), initial=0.0)
            ),
            "maximum_imaginary_part": float(
                np.max(np.abs(selected.imag), initial=0.0)
            ),
            **cauchy_diagnostics,
            "retained_dimension": int(np.count_nonzero(nonzero)),
            "contraction_backend": contraction_backend,
        }
    if return_diagnostics:
        return couplings, diagnostics
    return couplings


def _validate_discarded_couplings_cauchy(
    coupling,
    discarded,
    row_norm,
    partner_norm,
    *,
    row_root,
    column_root,
    subspace,
    norm_tol=1.0e-14,
    atol=1.0e-10,
    rtol=1.0e-9,
):
    """Gate discarded cross terms with their elementwise Cauchy bound.

    For a row-owned perturber ``T_alpha^I`` this checks

    ``|<I|T^dagger T|J>| <= sqrt(N_alpha(I) N_alpha(J)) + roundoff``,

    where *both* diagonal norms use the row-I operator and orbital basis.
    """

    coupling = np.asarray(coupling)
    discarded = np.asarray(discarded, dtype=bool)
    row_norm = np.asarray(row_norm)
    partner_norm = np.asarray(partner_norm)
    if not (
        coupling.shape
        == discarded.shape
        == row_norm.shape
        == partner_norm.shape
    ):
        raise ValueError(
            f"subspace {subspace}: Cauchy-audit arrays have inconsistent shapes"
        )
    norm_tol = _finite_nonnegative(norm_tol, name="norm_tol")
    atol = _finite_nonnegative(atol, name="zero_norm_atol")
    rtol = _finite_nonnegative(rtol, name="zero_norm_rtol")
    flat_indices = np.flatnonzero(discarded)
    if not flat_indices.size:
        return {
            "zero_norm_count": 0,
            "zero_norm_maximum_absolute_value": 0.0,
            "zero_norm_maximum_absolute_index": [],
            "zero_norm_row_norm_at_maximum": 0.0,
            "zero_norm_partner_norm_at_maximum": 0.0,
            "zero_norm_cauchy_bound_at_maximum": 0.0,
            "zero_norm_numerical_allowance_at_maximum": 0.0,
            "zero_norm_acceptance_limit_at_maximum": 0.0,
            "zero_norm_physical_excess_at_maximum": 0.0,
            "zero_norm_maximum_acceptance_excess": 0.0,
            "zero_norm_maximum_acceptance_excess_index": [],
            "zero_norm_maximum_cauchy_ratio": 0.0,
            "zero_norm_maximum_finite_cauchy_ratio": 0.0,
            "zero_norm_cauchy_ratio_unbounded_count": 0,
            "zero_norm_cauchy_gate_passed": True,
        }

    selected_coupling = coupling[discarded]
    selected_row_norm = row_norm[discarded]
    selected_partner_norm = partner_norm[discarded]
    if (
        not np.all(np.isfinite(selected_coupling))
        or not np.all(np.isfinite(selected_row_norm))
        or not np.all(np.isfinite(selected_partner_norm))
    ):
        raise FloatingPointError(
            f"row root {row_root}, column root {column_root}, subspace "
            f"{subspace}: Cauchy audit contains non-finite values"
        )
    if np.iscomplexobj(selected_row_norm):
        selected_row_norm = _utils._require_real(
            selected_row_norm,
            root=row_root,
            subspace=subspace,
            quantity="discarded row perturber norm",
            atol=atol,
            rtol=rtol,
        )
    if np.iscomplexobj(selected_partner_norm):
        selected_partner_norm = _utils._require_real(
            selected_partner_norm,
            root=column_root,
            subspace=subspace,
            quantity="row-basis partner perturber norm",
            atol=atol,
            rtol=rtol,
        )
    selected_row_norm = np.asarray(selected_row_norm, dtype=float)
    selected_partner_norm = np.asarray(selected_partner_norm, dtype=float)
    minimum_row = float(np.min(selected_row_norm))
    minimum_partner = float(np.min(selected_partner_norm))
    if minimum_row < -norm_tol or minimum_partner < -norm_tol:
        _utils._warn_numerical(
            f"row root {row_root}, column root {column_root}, subspace "
            f"{subspace}: negative diagonal norm in Cauchy audit "
            f"(row={minimum_row:.3e}, partner={minimum_partner:.3e})"
        )

    magnitude = np.abs(selected_coupling)
    cauchy_bound = np.sqrt(
        np.maximum(selected_row_norm, 0.0)
        * np.maximum(selected_partner_norm, 0.0)
    )
    numerical_allowance = atol + rtol * np.maximum.reduce(
        (np.ones_like(magnitude), magnitude, cauchy_bound)
    )
    acceptance_limit = cauchy_bound + numerical_allowance
    acceptance_excess = magnitude - acceptance_limit
    physical_excess = magnitude - cauchy_bound
    cauchy_ratio = np.divide(
        magnitude,
        cauchy_bound,
        out=np.zeros_like(magnitude, dtype=float),
        where=cauchy_bound > 0.0,
    )
    ratio_unbounded = (magnitude > 0.0) & (cauchy_bound == 0.0)
    ratio_unbounded_count = int(np.count_nonzero(ratio_unbounded))
    maximum_finite_cauchy_ratio = float(
        np.max(cauchy_ratio, initial=0.0)
    )

    maximum_local = int(np.argmax(magnitude))
    excess_local = int(np.argmax(acceptance_excess))

    def full_index(local_index):
        return [
            int(index)
            for index in np.unravel_index(
                int(flat_indices[int(local_index)]), coupling.shape
            )
        ]

    maximum_index = full_index(maximum_local)
    excess_index = full_index(excess_local)
    maximum_acceptance_excess = float(acceptance_excess[excess_local])
    if maximum_acceptance_excess > 0.0:
        _utils._warn_numerical(
            f"row root {row_root}, column root {column_root}, subspace "
            f"{subspace}: discarded near-null perturber violates its Cauchy "
            f"bound at {tuple(excess_index)}: |B|={magnitude[excess_local]:.3e}, "
            f"N_row={selected_row_norm[excess_local]:.3e}, "
            f"N_partner={selected_partner_norm[excess_local]:.3e}, "
            f"sqrt(N_row*N_partner)={cauchy_bound[excess_local]:.3e}, "
            f"roundoff={numerical_allowance[excess_local]:.3e}, "
            f"excess={maximum_acceptance_excess:.3e}"
        )

    return {
        "zero_norm_count": int(flat_indices.size),
        "zero_norm_maximum_absolute_value": float(magnitude[maximum_local]),
        "zero_norm_maximum_absolute_index": maximum_index,
        "zero_norm_row_norm_at_maximum": float(
            selected_row_norm[maximum_local]
        ),
        "zero_norm_partner_norm_at_maximum": float(
            selected_partner_norm[maximum_local]
        ),
        "zero_norm_cauchy_bound_at_maximum": float(
            cauchy_bound[maximum_local]
        ),
        "zero_norm_numerical_allowance_at_maximum": float(
            numerical_allowance[maximum_local]
        ),
        "zero_norm_acceptance_limit_at_maximum": float(
            acceptance_limit[maximum_local]
        ),
        "zero_norm_physical_excess_at_maximum": float(
            physical_excess[maximum_local]
        ),
        "zero_norm_maximum_acceptance_excess": maximum_acceptance_excess,
        "zero_norm_maximum_acceptance_excess_index": excess_index,
        "zero_norm_maximum_cauchy_ratio": (
            None
            if ratio_unbounded_count
            else maximum_finite_cauchy_ratio
        ),
        "zero_norm_maximum_finite_cauchy_ratio": (
            maximum_finite_cauchy_ratio
        ),
        "zero_norm_cauchy_ratio_unbounded_count": ratio_unbounded_count,
        "zero_norm_cauchy_gate_passed": bool(
            maximum_acceptance_excess <= 0.0
        ),
    }


def _evaluate_row_basis_partner_norms(
    eris_row,
    partner_pdms123,
    *,
    row_root,
    partner_root,
    scalar_atol=1.0e-10,
    scalar_rtol=1.0e-9,
    norm_tol=1.0e-14,
    contraction_backend=_utils._DEFAULT_CONTRACTION_BACKEND,
    return_diagnostics=False,
):
    """Evaluate ``<J|T_I^dagger T_I|J>`` in the row-I orbital basis."""

    raw_norms, contraction_diagnostics = (
        _evaluate_transition_perturber_couplings(
            eris_row,
            partner_pdms123,
            1.0,
            row_root=row_root,
            column_root=partner_root,
            contraction_backend=contraction_backend,
            return_diagnostics=True,
        )
    )
    norms = {}
    diagnostics = {}
    for key in _utils.SUBSPACE_ORDER:
        raw = raw_norms[key]
        ordered = _utils._strict_pair_mask(key, raw.shape)
        selected = _utils._require_real(
            raw[ordered],
            root=partner_root,
            subspace=key,
            quantity=f"row-{row_root} perturber norm on partner state",
            atol=scalar_atol,
            rtol=scalar_rtol,
        )
        if np.any(selected < -norm_tol):
            minimum = float(np.min(selected))
            _utils._warn_numerical(
                f"row root {row_root}, partner root {partner_root}, subspace "
                f"{key}: negative row-basis partner norm {minimum:.3e}"
            )
        norm = np.zeros(raw.shape, dtype=float)
        norm[ordered] = selected
        norms[key] = norm
        diagnostics[key] = {
            "minimum_ordered_norm": (
                float(np.min(selected)) if selected.size else 0.0
            ),
            "maximum_ordered_norm": (
                float(np.max(selected)) if selected.size else 0.0
            ),
            "nonnegative_norm_gate_passed": bool(
                not np.any(selected < -norm_tol)
            ),
            "ordered_dimension": int(np.count_nonzero(ordered)),
            "contraction_backend": contraction_backend,
            "raw_contraction": contraction_diagnostics[key],
        }
    if return_diagnostics:
        return norms, diagnostics
    return norms


@dataclass(frozen=True)
class QDSCNEVPT2Result:
    qd_type: str
    roots: tuple[int, ...]
    reference_energies: np.ndarray

    # Aliases for the representation selected by qd_type.
    h2_by_subspace: dict[str, np.ndarray]
    h2_effective: np.ndarray
    h_eff: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    left_eigenvectors: np.ndarray
    right_eigenvectors: np.ndarray

    # Both complete representations are retained for audit and regression.
    h2_bloch_by_subspace: dict[str, np.ndarray]
    h2_bloch: np.ndarray
    h_eff_bloch: np.ndarray
    h2_van_vleck_by_subspace: dict[str, np.ndarray]
    h2_van_vleck: np.ndarray
    h_eff_van_vleck: np.ndarray
    diagnostics: dict[str, Any]


# Historical result name retained as a direct compatibility alias.  There is
# one calculation and one result implementation, not a duplicated Bloch path.
QDBlochSCNEVPT2Result = QDSCNEVPT2Result


def _mapping_item(values, key, *, name):
    if values is None:
        return None
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping")
    try:
        return values[key]
    except KeyError as error:
        raise KeyError(f"{name} has no entry for {key!r}") from error


def _transition_mapping_item(values, bra_root, ket_root, *, max_rank=None):
    if values is None:
        return None, None
    if not isinstance(values, Mapping):
        raise TypeError("transition_pdms must be a mapping")
    direct_key = (bra_root, ket_root)
    reverse_key = (ket_root, bra_root)
    if direct_key in values:
        densities = values[direct_key]
        if not isinstance(densities, (tuple, list)):
            raise TypeError("an injected transition PDM entry must be a sequence")
        if max_rank is not None:
            densities = densities[: int(max_rank)]
        return tuple(densities), "injected_direct"
    if reverse_key in values:
        densities = values[reverse_key]
        if not isinstance(densities, (tuple, list)):
            raise TypeError("an injected transition PDM entry must be a sequence")
        if max_rank is not None:
            densities = densities[: int(max_rank)]
        return (
            tuple(adjoint_transition_pdm(dm) for dm in densities),
            "injected_adjoint",
        )
    raise KeyError(
        "transition_pdms has neither ordered root pair "
        f"{direct_key!r} nor {reverse_key!r}"
    )


def _injected_model_overlap(model_overlap, roots, *, atol, rtol):
    """Validate a complete injected overlap before any transition NPDM work."""

    if model_overlap is None:
        return None
    nmodel = len(roots)
    if isinstance(model_overlap, Mapping):
        matrix = np.eye(nmodel, dtype=complex)
        for irow, root_i in enumerate(roots):
            diagonal_key = (root_i, root_i)
            if diagonal_key in model_overlap:
                matrix[irow, irow] = _complex_scalar(
                    model_overlap[diagonal_key],
                    name="injected model overlap",
                )
            for jcol in range(irow + 1, nmodel):
                root_j = roots[jcol]
                direct_key = (root_i, root_j)
                reverse_key = (root_j, root_i)
                has_direct = direct_key in model_overlap
                has_reverse = reverse_key in model_overlap
                if not has_direct and not has_reverse:
                    raise KeyError(
                        "model_overlap has neither ordered root pair "
                        f"{direct_key!r} nor {reverse_key!r}"
                    )
                direct = (
                    _complex_scalar(
                        model_overlap[direct_key],
                        name="injected model overlap",
                    )
                    if has_direct
                    else None
                )
                reverse = (
                    _complex_scalar(
                        model_overlap[reverse_key],
                        name="injected model overlap",
                    )
                    if has_reverse
                    else None
                )
                if direct is not None and reverse is not None:
                    limit = atol + rtol * max(1.0, abs(direct), abs(reverse))
                    if abs(direct - reverse.conjugate()) > limit:
                        _utils._warn_numerical(
                            "injected forward/reverse model overlaps are not "
                            f"adjoints for roots ({root_i}, {root_j})"
                        )
                if direct is None:
                    direct = reverse.conjugate()
                if reverse is None:
                    reverse = direct.conjugate()
                matrix[irow, jcol] = direct
                matrix[jcol, irow] = reverse
    else:
        matrix = np.asarray(model_overlap, dtype=complex)
    expected = (nmodel, nmodel)
    if matrix.shape != expected:
        raise ValueError(
            f"model_overlap must have shape {expected}, got {matrix.shape}"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("model_overlap contains non-finite values")
    scale = max(1.0, float(np.max(np.abs(matrix), initial=0.0)))
    limit = atol + rtol * scale
    adjoint_error = float(
        np.max(np.abs(matrix - matrix.conj().T), initial=0.0)
    )
    if adjoint_error > limit:
        _utils._warn_numerical(
            "injected model_overlap is not Hermitian: maximum error="
            f"{adjoint_error:.3e}"
        )
    identity_error = float(
        np.max(np.abs(matrix - np.eye(nmodel)), initial=0.0)
    )
    if identity_error > atol + rtol:
        _utils._warn_numerical(
            "injected model states are not orthonormal: maximum overlap "
            f"error={identity_error:.3e}"
        )
    return np.array(matrix, dtype=complex, copy=True)


def _phase_gauge_eigenvectors(left, right):
    left = np.array(left, dtype=complex, copy=True)
    right = np.array(right, dtype=complex, copy=True)
    for column in range(right.shape[1]):
        pivot = int(np.argmax(np.abs(right[:, column])))
        value = right[pivot, column]
        if abs(value) == 0.0:
            continue
        phase = np.exp(-1j * np.angle(value))
        # SciPy stores left eigenvectors as columns.  Applying the same phase
        # to both columns preserves L^dagger R and both eigen-equations.
        right[:, column] *= phase
        left[:, column] *= phase
    return left, right


def _phase_gauge_orthonormal_eigenvectors(vectors):
    """Apply a deterministic column phase without changing orthonormality."""

    vectors = np.array(vectors, dtype=complex, copy=True)
    for column in range(vectors.shape[1]):
        pivot = int(np.argmax(np.abs(vectors[:, column])))
        value = vectors[pivot, column]
        magnitude = abs(value)
        if magnitude == 0.0:
            continue
        vectors[:, column] *= value.conjugate() / magnitude
        # Make the gauge convention exact rather than leaving a roundoff-sized
        # imaginary component at the pivot.
        vectors[pivot, column] = magnitude
    return vectors


def _square_numeric_matrix(matrix, *, name):
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    if not np.issubdtype(matrix.dtype, np.number):
        raise TypeError(f"{name} must be numeric")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite values")
    return matrix


def _van_vleck_hermitian_part(matrix):
    """Return ``(matrix + matrix**dagger) / 2`` at full complex precision."""

    matrix = _square_numeric_matrix(matrix, name="Bloch matrix")
    return 0.5 * (matrix + matrix.conj().T)


def _nonhermiticity_diagnostics(matrix):
    difference = matrix - matrix.conj().T
    return {
        "maximum": float(np.max(np.abs(difference), initial=0.0)),
        "frobenius": float(np.linalg.norm(difference)),
    }


def _build_van_vleck_matrices(
    h2_bloch_by_subspace,
    h_eff_bloch,
    reference_energies,
    *,
    audit_tol,
):
    """Hermitianize every SC class and audit Lang et al. Eqs. (45)--(46)."""

    audit_tol = _finite_nonnegative(audit_tol, name="van_vleck_audit_tol")
    if not isinstance(h2_bloch_by_subspace, Mapping):
        raise TypeError("h2_bloch_by_subspace must be a mapping")
    missing = [
        key for key in _utils.SUBSPACE_ORDER if key not in h2_bloch_by_subspace
    ]
    extras = [
        key for key in h2_bloch_by_subspace if key not in _utils.SUBSPACE_ORDER
    ]
    if missing or extras:
        raise ValueError(
            "h2_bloch_by_subspace must contain exactly SUBSPACE_ORDER; "
            f"missing={missing}, extras={extras}"
        )

    h_eff_bloch = _square_numeric_matrix(
        h_eff_bloch, name="h_eff_bloch"
    )
    nmodel = h_eff_bloch.shape[0]
    reference_energies = np.asarray(reference_energies)
    if reference_energies.shape != (nmodel,):
        raise ValueError(
            "reference_energies must match the model-space dimension"
        )
    if not np.issubdtype(reference_energies.dtype, np.number):
        raise TypeError("reference_energies must be numeric")
    if not np.all(np.isfinite(reference_energies)):
        raise ValueError("reference_energies contain non-finite values")
    reference_imaginary = float(
        np.max(np.abs(np.asarray(reference_energies).imag), initial=0.0)
    )
    if reference_imaginary > audit_tol:
        _utils._warn_numerical(
            "reference energies have a material imaginary component: "
            f"{reference_imaginary:.3e}"
        )
    reference_energies = np.asarray(reference_energies.real, dtype=float)

    h2_van_vleck_by_subspace = {}
    maximum_bloch_diagonal_imaginary = 0.0
    maximum_subspace_diagonal_change = 0.0
    subspace_diagnostics = {}
    for key in _utils.SUBSPACE_ORDER:
        h2_bloch_key = _square_numeric_matrix(
            h2_bloch_by_subspace[key],
            name=f"h2_bloch_by_subspace[{key!r}]",
        )
        if h2_bloch_key.shape != (nmodel, nmodel):
            raise ValueError(
                f"h2_bloch_by_subspace[{key!r}] has shape "
                f"{h2_bloch_key.shape}; expected {(nmodel, nmodel)}"
            )
        diagonal_imaginary = float(
            np.max(np.abs(np.diag(h2_bloch_key).imag), initial=0.0)
        )
        maximum_bloch_diagonal_imaginary = max(
            maximum_bloch_diagonal_imaginary, diagonal_imaginary
        )
        h2_van_vleck_key = _van_vleck_hermitian_part(h2_bloch_key)
        diagonal_change = float(
            np.max(
                np.abs(
                    np.diag(h2_van_vleck_key) - np.diag(h2_bloch_key)
                ),
                initial=0.0,
            )
        )
        maximum_subspace_diagonal_change = max(
            maximum_subspace_diagonal_change, diagonal_change
        )
        hermiticity = _nonhermiticity_diagnostics(h2_van_vleck_key)
        subspace_diagnostics[key] = {
            "bloch_diagonal_maximum_imaginary_part": diagonal_imaginary,
            "maximum_diagonal_change": diagonal_change,
            "maximum_nonhermiticity": hermiticity["maximum"],
            "frobenius_nonhermiticity": hermiticity["frobenius"],
        }
        h2_van_vleck_by_subspace[key] = h2_van_vleck_key

    if maximum_bloch_diagonal_imaginary > audit_tol:
        _utils._warn_numerical(
            "Bloch subspace diagonal has a material imaginary component: "
            f"{maximum_bloch_diagonal_imaginary:.3e}"
        )
    if maximum_subspace_diagonal_change > audit_tol:
        _utils._warn_numerical(
            "Van Vleck subspace Hermitianization changed a diagonal: "
            f"{maximum_subspace_diagonal_change:.3e}"
        )

    h2_van_vleck = np.zeros((nmodel, nmodel), dtype=np.complex128)
    for key in _utils.SUBSPACE_ORDER:
        h2_van_vleck += h2_van_vleck_by_subspace[key]
    reference_matrix = np.diag(reference_energies.astype(np.complex128))
    h_eff_van_vleck = reference_matrix + h2_van_vleck

    # Independent global oracles.  The first checks the sum of the eight
    # classwise Hermitianizations; the second is Lang et al. Eq. (46) applied
    # to the already complete source-row Bloch effective Hamiltonian.
    h2_bloch = h_eff_bloch - reference_matrix
    h2_van_vleck_direct = _van_vleck_hermitian_part(h2_bloch)
    h_eff_van_vleck_direct = _van_vleck_hermitian_part(h_eff_bloch)
    subspace_sum_error = float(
        np.max(
            np.abs(h2_van_vleck - h2_van_vleck_direct), initial=0.0
        )
    )
    global_formula_error = float(
        np.max(
            np.abs(h_eff_van_vleck - h_eff_van_vleck_direct),
            initial=0.0,
        )
    )
    maximum_diagonal_change = float(
        np.max(
            np.abs(
                np.diag(h_eff_van_vleck) - np.diag(h_eff_bloch)
            ),
            initial=0.0,
        )
    )
    if subspace_sum_error > audit_tol:
        _utils._warn_numerical(
            "classwise Van Vleck sum disagrees with the global H2 formula: "
            f"{subspace_sum_error:.3e}"
        )
    if global_formula_error > audit_tol:
        _utils._warn_numerical(
            "Van Vleck effective Hamiltonian disagrees with Eq. (46): "
            f"{global_formula_error:.3e}"
        )
    if maximum_diagonal_change > audit_tol:
        _utils._warn_numerical(
            "Van Vleck Hermitianization changed an effective-Hamiltonian "
            f"diagonal: {maximum_diagonal_change:.3e}"
        )

    bloch_nonhermiticity = _nonhermiticity_diagnostics(h_eff_bloch)
    van_vleck_nonhermiticity = _nonhermiticity_diagnostics(
        h_eff_van_vleck
    )
    if van_vleck_nonhermiticity["maximum"] > audit_tol:
        _utils._warn_numerical(
            "constructed Van Vleck effective Hamiltonian is not Hermitian: "
            f"{van_vleck_nonhermiticity['maximum']:.3e}"
        )
    hermitization_change = h_eff_van_vleck - h_eff_bloch
    diagnostics = {
        "van_vleck_audit_tolerance": float(audit_tol),
        "reference_energy_reality_gate_passed": bool(
            reference_imaginary <= audit_tol
        ),
        "bloch_diagonal_reality_gate_passed": bool(
            maximum_bloch_diagonal_imaginary <= audit_tol
        ),
        "subspace_diagonal_gate_passed": bool(
            maximum_subspace_diagonal_change <= audit_tol
        ),
        "subspace_sum_gate_passed": bool(subspace_sum_error <= audit_tol),
        "global_formula_gate_passed": bool(global_formula_error <= audit_tol),
        "effective_diagonal_gate_passed": bool(
            maximum_diagonal_change <= audit_tol
        ),
        "van_vleck_hermiticity_gate_passed": bool(
            van_vleck_nonhermiticity["maximum"] <= audit_tol
        ),
        "maximum_bloch_nonhermiticity": bloch_nonhermiticity["maximum"],
        "frobenius_bloch_nonhermiticity": bloch_nonhermiticity["frobenius"],
        "maximum_van_vleck_nonhermiticity": (
            van_vleck_nonhermiticity["maximum"]
        ),
        "frobenius_van_vleck_nonhermiticity": (
            van_vleck_nonhermiticity["frobenius"]
        ),
        "maximum_hermitization_change": float(
            np.max(np.abs(hermitization_change), initial=0.0)
        ),
        "frobenius_hermitization_change": float(
            np.linalg.norm(hermitization_change)
        ),
        "van_vleck_global_formula_error": global_formula_error,
        "van_vleck_subspace_sum_error": subspace_sum_error,
        "maximum_van_vleck_diagonal_change": maximum_diagonal_change,
        "maximum_subspace_diagonal_change": (
            maximum_subspace_diagonal_change
        ),
        "maximum_bloch_diagonal_imaginary_part": (
            maximum_bloch_diagonal_imaginary
        ),
        "van_vleck_subspace_diagnostics": subspace_diagnostics,
    }
    return (
        h2_van_vleck_by_subspace,
        h2_van_vleck,
        h_eff_van_vleck,
        diagnostics,
    )


def _solve_bloch_eigensystem(matrix):
    eigenvalues, left, right = eig(matrix, left=True, right=True)
    order = np.lexsort((eigenvalues.imag, eigenvalues.real))
    eigenvalues = eigenvalues[order]
    left = left[:, order]
    right = right[:, order]
    left, right = _phase_gauge_eigenvectors(left, right)
    right_residual = matrix @ right - right * eigenvalues[np.newaxis, :]
    left_rows = left.conj().T
    left_residual = left_rows @ matrix - eigenvalues[:, np.newaxis] * left_rows
    scale = max(1.0, float(np.linalg.norm(matrix)))
    diagnostics = {
        "right_residual_norm": float(np.linalg.norm(right_residual)),
        "left_residual_norm": float(np.linalg.norm(left_residual)),
        "right_relative_residual_norm": float(
            np.linalg.norm(right_residual) / scale
        ),
        "left_relative_residual_norm": float(
            np.linalg.norm(left_residual) / scale
        ),
        "biorthogonality": left.conj().T @ right,
        "right_eigenvector_condition_number": float(np.linalg.cond(right)),
        "maximum_nonhermiticity": float(
            np.max(np.abs(matrix - matrix.conj().T), initial=0.0)
        ),
        "maximum_eigenvalue_imaginary_part": float(
            np.max(np.abs(eigenvalues.imag), initial=0.0)
        ),
    }
    return eigenvalues, left, right, diagnostics


def _solve_van_vleck_eigensystem(
    matrix, *, hermiticity_tol, residual_tol
):
    """Solve one complex Hermitian HQD matrix with explicit audit gates."""

    hermiticity_tol = _finite_nonnegative(
        hermiticity_tol, name="van_vleck_hermiticity_tol"
    )
    residual_tol = _finite_nonnegative(
        residual_tol, name="eigen_residual_tol"
    )
    matrix = _square_numeric_matrix(matrix, name="h_eff_van_vleck")
    nonhermiticity = _nonhermiticity_diagnostics(matrix)
    if nonhermiticity["maximum"] > hermiticity_tol:
        _utils._warn_numerical(
            "Van Vleck matrix failed the pre-eigh Hermiticity check: "
            f"{nonhermiticity['maximum']:.3e}"
        )

    eigenvalues, vectors = eigh(matrix)
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    vectors = _phase_gauge_orthonormal_eigenvectors(vectors)
    residual = matrix @ vectors - vectors * eigenvalues[np.newaxis, :]
    scale = max(1.0, float(np.linalg.norm(matrix)))
    residual_norm = float(np.linalg.norm(residual))
    relative_residual_norm = float(residual_norm / scale)
    gram = vectors.conj().T @ vectors
    orthonormality_error = float(
        np.linalg.norm(gram - np.eye(matrix.shape[0]))
    )
    diagnostics = {
        "residual_norm": residual_norm,
        "relative_residual_norm": relative_residual_norm,
        "orthonormality_error": orthonormality_error,
        "maximum_orthonormality_error": float(
            np.max(
                np.abs(gram - np.eye(matrix.shape[0])), initial=0.0
            )
        ),
        "maximum_matrix_nonhermiticity": nonhermiticity["maximum"],
        "frobenius_matrix_nonhermiticity": nonhermiticity["frobenius"],
        "hermiticity_tolerance": float(hermiticity_tol),
        "hermiticity_gate_passed": bool(
            nonhermiticity["maximum"] <= hermiticity_tol
        ),
        "residual_tolerance": float(residual_tol),
        "residual_gate_passed": bool(
            relative_residual_norm <= residual_tol
        ),
        "orthonormality_gate_passed": bool(
            orthonormality_error <= residual_tol
        ),
        "biorthogonality": gram,
    }
    if relative_residual_norm > residual_tol:
        _utils._warn_numerical(
            "Hermitian Van Vleck eigensolver residual exceeds tolerance"
        )
    if orthonormality_error > residual_tol:
        _utils._warn_numerical(
            "Hermitian Van Vleck eigenvectors are not orthonormal within "
            "tolerance"
        )
    return eigenvalues, vectors, diagnostics


class WickX2CQDSCNEVPT2(lib.StreamObject):
    """Dense multipartitioning QD-SC-NEVPT2 driver.

    ``qd_type="van_vleck"`` (the default) selects the Hermitian canonical
    Van Vleck/HQD representation.  ``qd_type="bloch"`` retains the classical
    non-Hermitian source-row representation and its biorthogonal eigensystem.
    Both complete effective Hamiltonians are retained after every run.
    """

    def __init__(self, mc, frozen=0, qd_type="van_vleck"):
        if _utils._has_frozen_orbitals(frozen) or _utils._has_frozen_orbitals(
            getattr(mc, "frozen", None)
        ):
            raise NotImplementedError("nonzero frozen spinors are outside dense v1")
        self._mc = mc
        self._scf = mc._scf
        self.mol = self._scf.mol
        self.verbose = getattr(mc, "verbose", self.mol.verbose)
        self.stdout = getattr(mc, "stdout", self.mol.stdout)
        self.mo_coeff = getattr(mc, "mo_coeff", None)
        self.mo_energy = None
        self.canonicalized = False
        self.denominator_mode = "strict_si"
        self.contraction_backend = _utils._DEFAULT_CONTRACTION_BACKEND
        for name, value in _ss._SC_NUMERICAL_DEFAULTS:
            setattr(self, name, value)
        self.model_overlap_atol = 1.0e-8
        self.model_overlap_rtol = 1.0e-8
        self.zero_norm_coupling_atol = 1.0e-10
        self.zero_norm_coupling_rtol = 1.0e-9
        self.eigen_residual_tol = 1.0e-10
        self.eigenvalue_imag_warn = 1.0e-10
        self.eigenvalue_imag_error = None
        self.validate_reverse_transition = False
        self.van_vleck_audit_tol = 1.0e-10
        self.van_vleck_hermiticity_tol = 1.0e-12
        self.qd_type = _normalize_qd_type(qd_type)
        self.frozen = 0

        self._clear_results()
        self._keys = set(self.__dict__)

    @property
    def e_tot(self):
        """Alias for the complete QD eigenvalue array."""

        return self.e_qd

    def run(self, *args, **kwargs):
        """Run ``kernel`` while preserving PySCF's fluent object interface.

        ``StreamObject.run`` normally stores keyword arguments as attributes
        and then invokes ``kernel`` without forwarding them.  QD inputs such
        as an explicit root subset or injected transition PDMs are
        calculation-local, so forward the kernel keywords as well.  Existing
        public settings are still updated before the call, and the method
        continues to return ``self``.
        """

        kernel_kwargs = {
            name: value
            for name, value in kwargs.items()
            if name in _QD_KERNEL_KEYWORDS
        }
        setting_kwargs = {
            name: value
            for name, value in kwargs.items()
            if name not in _QD_KERNEL_KEYWORDS or name in self._keys
        }
        self.set(**setting_kwargs)
        self.kernel(*args, **kernel_kwargs)
        return self

    def _clear_results(self):
        """Drop a previous calculation before starting a transactional run."""

        self.roots = ()
        self.reference_energies = None
        self.row_data = {}
        self.h2_bloch_by_subspace = {}
        self.h2_van_vleck_by_subspace = {}
        self.h2_by_subspace = {}
        self.h2_effective = None
        self.h_eff = None
        self.h2_bloch = None
        self.h_eff_bloch = None
        self.h2_van_vleck = None
        self.h_eff_van_vleck = None
        self.e_qd = None
        self.eigenvectors = None
        self.left_eigenvectors = None
        self.right_eigenvectors = None
        self.model_overlap = None
        self.diagnostics = {}
        self.result = None

    def _new_ss_adapter(self, mc, root):
        adapter = _ss.WickX2CSCNEVPT2(mc)
        adapter.verbose = self.verbose
        adapter.stdout = self.stdout
        adapter.canonicalized = bool(self.canonicalized)
        if isinstance(self.mo_energy, Mapping):
            adapter.mo_energy = np.asarray(self.mo_energy[root])
        else:
            adapter.mo_energy = self.mo_energy
        for name, _default in _ss._SC_NUMERICAL_DEFAULTS:
            setattr(adapter, name, getattr(self, name))
        adapter.denominator_mode = "strict_si"
        adapter.contraction_backend = self.contraction_backend
        return adapter

    def _prepare_row(
        self,
        mc,
        root,
        *,
        mo_coeff,
        pdms,
        eris,
        eris_basis,
    ):
        adapter = self._new_ss_adapter(mc, root)
        return adapter._prepare_sc_root(
            mc,
            mo_coeff=mo_coeff,
            pdms=pdms,
            eris=eris,
            eris_basis=eris_basis,
            root=root,
            denominator_mode="strict_si",
            contraction_backend=self.contraction_backend,
            return_arrays=True,
            compact_eris=True,
            retain_pdms123=True,
        )

    def kernel(
        self,
        mc=None,
        roots=None,
        *,
        qd_type=None,
        state_pdms=None,
        transition_pdms=None,
        model_overlap=None,
        mo_coeff=None,
        eris=None,
        eris_basis="input_mo",
        denominator_mode=None,
        contraction_backend=None,
        validate_reverse_transition=None,
        eigenvalue_imag_warn=None,
        eigenvalue_imag_error=None,
    ):
        """Build both QD matrices and solve the selected representation."""

        total_start = time.perf_counter()
        self._clear_results()
        gc.collect()
        if qd_type is None:
            qd_type = self.qd_type
        qd_type = _normalize_qd_type(qd_type)
        self.qd_type = qd_type
        if mc is None:
            mc = self._mc
        if _utils._has_frozen_orbitals(getattr(mc, "frozen", None)):
            raise NotImplementedError("nonzero frozen spinors are outside dense v1")
        eris_basis = _utils._normalize_eris_basis(eris_basis)
        if denominator_mode is None:
            denominator_mode = self.denominator_mode
        if denominator_mode != "strict_si":
            raise ValueError(
                "QD-SC-NEVPT2 requires denominator_mode='strict_si'"
            )
        self.denominator_mode = "strict_si"
        if contraction_backend is None:
            contraction_backend = self.contraction_backend
        self.contraction_backend = _utils._normalize_contraction_backend(
            contraction_backend
        )
        for name in (
            "model_overlap_atol",
            "model_overlap_rtol",
            "zero_norm_coupling_atol",
            "zero_norm_coupling_rtol",
            "eigen_residual_tol",
            "van_vleck_audit_tol",
            "van_vleck_hermiticity_tol",
        ):
            setattr(
                self,
                name,
                _finite_nonnegative(getattr(self, name), name=name),
            )
        if validate_reverse_transition is None:
            validate_reverse_transition = self.validate_reverse_transition
        validate_reverse_transition = bool(validate_reverse_transition)
        if qd_type == "bloch":
            if eigenvalue_imag_warn is None:
                eigenvalue_imag_warn = self.eigenvalue_imag_warn
            if eigenvalue_imag_error is None:
                eigenvalue_imag_error = self.eigenvalue_imag_error
            if eigenvalue_imag_warn is not None:
                eigenvalue_imag_warn = _finite_nonnegative(
                    eigenvalue_imag_warn, name="eigenvalue_imag_warn"
                )
            if eigenvalue_imag_error is not None:
                eigenvalue_imag_error = _finite_nonnegative(
                    eigenvalue_imag_error, name="eigenvalue_imag_error"
                )

        if roots is None:
            nroots = int(getattr(mc.fcisolver, "nroots", 1))
            roots = tuple(range(nroots))
        else:
            roots = tuple(int(root) for root in roots)
        if not roots:
            raise ValueError("roots must contain at least one model state")
        if len(set(roots)) != len(roots) or min(roots) < 0:
            raise ValueError("roots must be unique non-negative integers")
        kets = getattr(mc.fcisolver, "kets", None)
        if kets is not None and max(roots) >= len(kets):
            raise IndexError("a requested QD root is unavailable in dmrgci.kets")
        if mo_coeff is None:
            mo_coeff = mc.mo_coeff
        input_mo = np.asarray(mo_coeff)
        nelec_source = getattr(mc.fcisolver, "nelecas", None)
        if nelec_source is None:
            nelec_source = mc.nelecas
        nelec = _utils._total_nelec(nelec_source)

        nmodel = len(roots)
        h2_by_subspace = {
            key: np.zeros((nmodel, nmodel), dtype=complex)
            for key in _utils.SUBSPACE_ORDER
        }
        reference_energies = np.empty(nmodel, dtype=float)
        row_data = {}
        row_diagnostics = {}
        shared_input_eris = eris
        shared_eris_basis = eris_basis
        shared_eris_generated = False
        shared_eris_time = 0.0
        if eris is None:
            shared_start = time.perf_counter()
            shared_input_eris = _utils._dense_eris_from_mc(
                mc,
                input_mo,
                roundoff_factor=self.integral_roundoff_factor,
            )
            shared_eris_time = time.perf_counter() - shared_start
            shared_eris_basis = "input_mo"
            shared_eris_generated = True
        shared_eris_bytes = (
            shared_input_eris.nbytes
            if isinstance(shared_input_eris, spinor_helper._SpinorERIs)
            else None
        )
        for irow, root in enumerate(roots):
            pdms_root = _mapping_item(state_pdms, root, name="state_pdms")
            if isinstance(shared_input_eris, Mapping):
                eris_root = _mapping_item(
                    shared_input_eris, root, name="eris"
                )
            else:
                eris_root = shared_input_eris
            row_start = time.perf_counter()
            row = self._prepare_row(
                mc,
                root,
                mo_coeff=input_mo,
                pdms=pdms_root,
                eris=eris_root,
                eris_basis=shared_eris_basis,
            )
            if row.subspace_arrays is None:
                raise RuntimeError("QD row preparation omitted subspace arrays")
            compact_eris = row.eris
            if not isinstance(compact_eris, _utils._WickERIBlocks):
                compact_eris = _utils._compact_wick_eris(compact_eris)
            retained_arrays = _retain_qd_row_arrays(row.subspace_arrays)
            row = replace(
                row,
                eris=compact_eris,
                subspace_arrays=retained_arrays,
            )
            source_eris_nbytes = (
                eris_root.nbytes
                if isinstance(eris_root, spinor_helper._SpinorERIs)
                else None
            )
            row_data[root] = row
            reference_energies[irow] = row.reference_energy
            for key in _utils.SUBSPACE_ORDER:
                h2_by_subspace[key][irow, irow] = row.subspace_energies[key]
            row_diagnostics[str(root)] = {
                "preparation_time": time.perf_counter() - row_start,
                "reference_energy": row.reference_energy,
                "e_corr_state_specific": row.e_corr,
                "strict_si_compatible": row.strict_si_compatible,
                "source_full_eris_bytes": source_eris_nbytes,
                "retained_wick_eris_bytes": compact_eris.nbytes,
                "retained_row_array_bytes": _nested_array_nbytes(
                    retained_arrays
                ),
                "temporary_state_pdm123_bytes": int(
                    sum(np.asarray(dm).nbytes for dm in row.pdms123)
                ),
                "subspace_energies": dict(row.subspace_energies),
                "subspace_gaps": {
                    key: list(row.subspace_gaps[key])
                    for key in _utils.SUBSPACE_ORDER
                },
            }
            logger.note(
                self,
                "QD_Bloch row root %d: E0=%.16g  E_SS^(2)=%.16g",
                root,
                row.reference_energy,
                row.e_corr,
            )
            del pdms_root, eris_root
            gc.collect()

        if shared_eris_generated:
            del shared_input_eris
            gc.collect()

        injected_overlap = _injected_model_overlap(
            model_overlap,
            roots,
            atol=self.model_overlap_atol,
            rtol=self.model_overlap_rtol,
        )
        cached_overlap = None
        if injected_overlap is None:
            raw_cached_overlap = getattr(mc.fcisolver, "root_overlap", None)
            if raw_cached_overlap is not None:
                raw_cached_overlap = np.asarray(raw_cached_overlap)
                if (
                    raw_cached_overlap.ndim != 2
                    or raw_cached_overlap.shape[0]
                    != raw_cached_overlap.shape[1]
                    or max(roots) >= raw_cached_overlap.shape[0]
                ):
                    raise ValueError(
                        "fcisolver.root_overlap does not cover the requested roots"
                    )
                cached_overlap = _injected_model_overlap(
                    raw_cached_overlap[np.ix_(roots, roots)],
                    roots,
                    atol=self.model_overlap_atol,
                    rtol=self.model_overlap_rtol,
                )
        available_overlap = (
            injected_overlap
            if injected_overlap is not None
            else cached_overlap
        )
        available_overlap_source = (
            "injected" if injected_overlap is not None else "solver_cached"
        )
        overlap_matrix = (
            np.eye(nmodel, dtype=complex)
            if available_overlap is None
            else available_overlap.copy()
        )
        overlap_sources = {}
        has_open_mps = (
            getattr(mc.fcisolver, "driver", None) is not None
            and getattr(mc.fcisolver, "kets", None) is not None
        )
        for irow, root in enumerate(roots):
            if available_overlap is not None:
                overlap_sources[f"{root},{root}"] = available_overlap_source
            elif has_open_mps:
                value, source = _make_transition_overlap(
                    mc.fcisolver, root, root
                )
                overlap_matrix[irow, irow] = value
                overlap_sources[f"{root},{root}"] = source

        pair_diagnostics = {}
        for irow in range(nmodel):
            root_i = roots[irow]
            row_i = row_data[root_i]
            if row_i.pdms123 is None:
                raise RuntimeError(
                    f"QD row root {root_i} omitted ordinary PDM ranks 1--3"
                )
            masks_i = {
                key: row_i.subspace_arrays[key]["nonzero"]
                for key in _utils.SUBSPACE_ORDER
            }
            norms_i = {
                key: row_i.subspace_arrays[key]["norm"]
                for key in _utils.SUBSPACE_ORDER
            }
            for jcol in range(irow + 1, nmodel):
                root_j = roots[jcol]
                row_j = row_data[root_j]
                if row_j.pdms123 is None:
                    raise RuntimeError(
                        f"QD row root {root_j} omitted ordinary PDM ranks 1--3"
                    )
                pair_start = time.perf_counter()
                if available_overlap is not None:
                    overlap_ij = available_overlap[irow, jcol]
                    overlap_source = available_overlap_source
                elif has_open_mps:
                    overlap_ij, overlap_source = _make_transition_overlap(
                        mc.fcisolver, root_i, root_j
                    )
                else:
                    overlap_ij = None
                    overlap_source = "dm1_trace_fallback"

                if overlap_ij is not None:
                    overlap_limit = self.model_overlap_atol + self.model_overlap_rtol
                    if abs(overlap_ij) > overlap_limit:
                        _utils._warn_numerical(
                            f"model roots {root_i} and {root_j} are not "
                            f"orthogonal: |overlap|={abs(overlap_ij):.3e}"
                        )

                pdms_ij, transition_source = _transition_mapping_item(
                    transition_pdms, root_i, root_j, max_rank=3
                )
                if pdms_ij is None:
                    pdms_ij = _make_transition_dm123(
                        mc.fcisolver, root_i, root_j
                    )
                    transition_source = "block2"
                if overlap_ij is None:
                    if nelec == 0:
                        raise RuntimeError(
                            "zero-electron injected transitions require model_overlap"
                        )
                    overlap_ij = _complex_scalar(
                        np.trace(np.asarray(pdms_ij[0])) / nelec,
                        name="dm1-derived overlap",
                    )
                    overlap_limit = self.model_overlap_atol + self.model_overlap_rtol
                    if abs(overlap_ij) > overlap_limit:
                        _utils._warn_numerical(
                            f"model roots {root_i} and {root_j} are not "
                            f"orthogonal: |overlap|={abs(overlap_ij):.3e}"
                        )

                reverse_debug = None
                has_injected_forward_and_reverse = bool(
                    isinstance(transition_pdms, Mapping)
                    and (root_i, root_j) in transition_pdms
                    and (root_j, root_i) in transition_pdms
                )
                if has_injected_forward_and_reverse:
                    reverse_debug = transition_pdms[(root_j, root_i)]
                elif validate_reverse_transition:
                    has_injected_reverse = bool(
                        isinstance(transition_pdms, Mapping)
                        and (root_j, root_i) in transition_pdms
                    )
                    if has_injected_reverse:
                        reverse_debug = transition_pdms[(root_j, root_i)]
                    elif has_open_mps:
                        reverse_debug = _make_transition_dm123(
                            mc.fcisolver, root_j, root_i
                        )
                    else:
                        raise RuntimeError(
                            "reverse transition validation needs reverse injected "
                            "PDMs or an open Block2 driver"
                        )
                if reverse_debug is not None and len(reverse_debug) == 4:
                    reverse_debug = tuple(reverse_debug[:3])
                pdms_ij, transition_validation = validate_transition_pdms(
                    pdms_ij,
                    int(mc.ncas),
                    nelec,
                    overlap_ij=overlap_ij,
                    pdms_ji=reverse_debug,
                    atol=self.rdm_atol,
                    rtol=self.rdm_rtol,
                    work_memory=self.rdm_work_memory,
                )
                pdms_ij = _transition_contraction_pdms(pdms_ij)
                pdms_ji = tuple(
                    adjoint_transition_pdm(density) for density in pdms_ij
                )
                overlap_ji = overlap_ij.conjugate()
                overlap_matrix[irow, jcol] = overlap_ij
                overlap_matrix[jcol, irow] = overlap_ji
                overlap_sources[f"{root_i},{root_j}"] = overlap_source
                overlap_sources[f"{root_j},{root_i}"] = "adjoint"

                partner_norms_ij, partner_norm_diagnostics_ij = (
                    _evaluate_row_basis_partner_norms(
                        row_i.eris,
                        row_j.pdms123,
                        row_root=root_i,
                        partner_root=root_j,
                        scalar_atol=self.scalar_atol,
                        scalar_rtol=self.scalar_rtol,
                        norm_tol=self.norm_tol,
                        contraction_backend=self.contraction_backend,
                        return_diagnostics=True,
                    )
                )
                couplings_ij, coupling_diagnostics_ij = (
                    _evaluate_transition_perturber_couplings(
                        row_i.eris,
                        pdms_ij,
                        overlap_ij,
                        row_root=root_i,
                        column_root=root_j,
                        row_nonzero=masks_i,
                        row_norm=norms_i,
                        partner_norm=partner_norms_ij,
                        norm_tol=self.norm_tol,
                        zero_norm_atol=self.zero_norm_coupling_atol,
                        zero_norm_rtol=self.zero_norm_coupling_rtol,
                        contraction_backend=self.contraction_backend,
                        return_diagnostics=True,
                    )
                )
                masks_j = {
                    key: row_j.subspace_arrays[key]["nonzero"]
                    for key in _utils.SUBSPACE_ORDER
                }
                norms_j = {
                    key: row_j.subspace_arrays[key]["norm"]
                    for key in _utils.SUBSPACE_ORDER
                }
                partner_norms_ji, partner_norm_diagnostics_ji = (
                    _evaluate_row_basis_partner_norms(
                        row_j.eris,
                        row_i.pdms123,
                        row_root=root_j,
                        partner_root=root_i,
                        scalar_atol=self.scalar_atol,
                        scalar_rtol=self.scalar_rtol,
                        norm_tol=self.norm_tol,
                        contraction_backend=self.contraction_backend,
                        return_diagnostics=True,
                    )
                )
                couplings_ji, coupling_diagnostics_ji = (
                    _evaluate_transition_perturber_couplings(
                        row_j.eris,
                        pdms_ji,
                        overlap_ji,
                        row_root=root_j,
                        column_root=root_i,
                        row_nonzero=masks_j,
                        row_norm=norms_j,
                        partner_norm=partner_norms_ji,
                        norm_tol=self.norm_tol,
                        zero_norm_atol=self.zero_norm_coupling_atol,
                        zero_norm_rtol=self.zero_norm_coupling_rtol,
                        contraction_backend=self.contraction_backend,
                        return_diagnostics=True,
                    )
                )
                class_corrections_ij = {}
                class_corrections_ji = {}
                for key in _utils.SUBSPACE_ORDER:
                    denominator_i = row_i.subspace_arrays[key]["denominator"]
                    denominator_j = row_j.subspace_arrays[key]["denominator"]
                    value_ij = -np.sum(
                        couplings_ij[key][masks_i[key]]
                        / denominator_i[masks_i[key]]
                    )
                    value_ji = -np.sum(
                        couplings_ji[key][masks_j[key]]
                        / denominator_j[masks_j[key]]
                    )
                    h2_by_subspace[key][irow, jcol] = value_ij
                    h2_by_subspace[key][jcol, irow] = value_ji
                    class_corrections_ij[key] = _complex_pair(value_ij)
                    class_corrections_ji[key] = _complex_pair(value_ji)
                pair_diagnostics[f"{root_i},{root_j}"] = {
                    "overlap": _complex_pair(overlap_ij),
                    "overlap_source": overlap_source,
                    "transition_source": transition_source,
                    "transition_validation": transition_validation,
                    "row_i_partner_norms": partner_norm_diagnostics_ij,
                    "row_j_partner_norms": partner_norm_diagnostics_ji,
                    "row_i_couplings": coupling_diagnostics_ij,
                    "row_j_couplings": coupling_diagnostics_ji,
                    "class_corrections_ij": class_corrections_ij,
                    "class_corrections_ji": class_corrections_ji,
                    "elapsed_time": time.perf_counter() - pair_start,
                }
                del reverse_debug
                del pdms_ij, pdms_ji
                del partner_norms_ij, partner_norms_ji
                del couplings_ij, couplings_ji
                gc.collect()

        # Ordinary state PDMs are required only for the pairwise Cauchy audit.
        # Do not retain them in the public result or duplicate their lifetime
        # with a subsequent calculation.
        for root in roots:
            row_data[root] = replace(row_data[root], pdms123=None)
        row = None
        row_i = None
        row_j = None
        gc.collect()

        identity = np.eye(nmodel, dtype=complex)
        overlap_error = float(
            np.max(np.abs(overlap_matrix - identity), initial=0.0)
        )
        overlap_limit = self.model_overlap_atol + self.model_overlap_rtol
        if overlap_error > overlap_limit:
            _utils._warn_numerical(
                "model-state overlap is not the identity within tolerance: "
                f"maximum error={overlap_error:.3e}"
            )
        overlap_adjoint_error = float(
            np.max(
                np.abs(overlap_matrix - overlap_matrix.conj().T), initial=0.0
            )
        )
        if overlap_adjoint_error > overlap_limit:
            _utils._warn_numerical(
                "model-state overlap is not Hermitian: maximum error="
                f"{overlap_adjoint_error:.3e}"
            )

        h2_bloch = np.zeros((nmodel, nmodel), dtype=complex)
        for key in _utils.SUBSPACE_ORDER:
            h2_bloch += h2_by_subspace[key]
        h_eff_bloch = np.diag(reference_energies.astype(complex)) + h2_bloch
        h2_bloch_by_subspace = h2_by_subspace
        (
            h2_van_vleck_by_subspace,
            h2_van_vleck,
            h_eff_van_vleck,
            van_vleck_diagnostics,
        ) = _build_van_vleck_matrices(
            h2_bloch_by_subspace,
            h_eff_bloch,
            reference_energies,
            audit_tol=self.van_vleck_audit_tol,
        )
        diagonal_reduction_error = 0.0
        for irow, root in enumerate(roots):
            row = row_data[root]
            for key in _utils.SUBSPACE_ORDER:
                diagonal_reduction_error = max(
                    diagonal_reduction_error,
                    abs(
                        h2_by_subspace[key][irow, irow]
                        - row.subspace_energies[key]
                    ),
                    abs(
                        h2_van_vleck_by_subspace[key][irow, irow]
                        - row.subspace_energies[key]
                    ),
                )
            diagonal_reduction_error = max(
                diagonal_reduction_error,
                abs(
                    h_eff_bloch[irow, irow]
                    - (row.reference_energy + row.e_corr)
                ),
                abs(
                    h_eff_van_vleck[irow, irow]
                    - (row.reference_energy + row.e_corr)
                ),
            )

        if qd_type == "bloch":
            eigenvalues, left, right, eig_diagnostics = (
                _solve_bloch_eigensystem(h_eff_bloch)
            )
            maximum_relative_residual = max(
                eig_diagnostics["right_relative_residual_norm"],
                eig_diagnostics["left_relative_residual_norm"],
            )
            eig_diagnostics["residual_tolerance"] = float(
                self.eigen_residual_tol
            )
            eig_diagnostics["residual_gate_passed"] = bool(
                maximum_relative_residual <= self.eigen_residual_tol
            )
            if not eig_diagnostics["residual_gate_passed"]:
                _utils._warn_numerical(
                    "general Bloch eigensolver residual exceeds tolerance"
                )
            maximum_eigen_imag = eig_diagnostics[
                "maximum_eigenvalue_imaginary_part"
            ]
            if (
                eigenvalue_imag_warn is not None
                and maximum_eigen_imag > eigenvalue_imag_warn
            ):
                logger.warn(
                    self,
                    "QD_Bloch eigenvalues have imaginary components up to %.3e",
                    maximum_eigen_imag,
                )
            if (
                eigenvalue_imag_error is not None
                and maximum_eigen_imag > eigenvalue_imag_error
            ):
                raise FloatingPointError(
                    "QD_Bloch eigenvalue imaginary component exceeds configured "
                    f"error threshold: {maximum_eigen_imag:.3e}"
                )
            eigenvectors = right
            selected_h2_by_subspace = h2_bloch_by_subspace
            selected_h2 = h2_bloch
            selected_h_eff = h_eff_bloch
            method = "classical_2004_multipartitioning_QD_Bloch_SC_NEVPT2"
        else:
            eigenvalues, vectors, eig_diagnostics = (
                _solve_van_vleck_eigensystem(
                    h_eff_van_vleck,
                    hermiticity_tol=self.van_vleck_hermiticity_tol,
                    residual_tol=self.eigen_residual_tol,
                )
            )
            eigenvectors = vectors
            # Hermitian left and right eigenvectors have the same column
            # representation.  Keep separate arrays so later mutation of one
            # public attribute cannot blur their semantics.
            left = np.array(vectors, copy=True)
            right = np.array(vectors, copy=True)
            selected_h2_by_subspace = h2_van_vleck_by_subspace
            selected_h2 = h2_van_vleck
            selected_h_eff = h_eff_van_vleck
            method = (
                "2020_multipartitioning_canonical_Van_Vleck_QD_SC_NEVPT2"
            )

        diagnostics = {
            "qd_type": qd_type,
            "method": method,
            "matrix_convention": (
                "H[I,J] uses bra I, ket J, row-I ERIs/perturbers/strict-SI gaps"
            ),
            "denominator_mode": "strict_si",
            "contraction_backend": self.contraction_backend,
            "shared_input_eris_generated": shared_eris_generated,
            "shared_input_eris_generation_time": shared_eris_time,
            "shared_input_eris_bytes": shared_eris_bytes,
            "row_diagnostics": row_diagnostics,
            "pair_diagnostics": pair_diagnostics,
            "model_overlap_sources": overlap_sources,
            "model_overlap_maximum_identity_error": overlap_error,
            "model_overlap_maximum_adjoint_error": overlap_adjoint_error,
            "model_overlap_tolerance": float(overlap_limit),
            "model_overlap_identity_gate_passed": bool(
                overlap_error <= overlap_limit
            ),
            "model_overlap_adjoint_gate_passed": bool(
                overlap_adjoint_error <= overlap_limit
            ),
            "diagonal_reduction_maximum_error": float(
                diagonal_reduction_error
            ),
            **van_vleck_diagnostics,
            "eigensystem": eig_diagnostics,
            "total_time": time.perf_counter() - total_start,
        }

        self.roots = roots
        self.reference_energies = reference_energies
        self.row_data = row_data
        self.h2_bloch_by_subspace = h2_bloch_by_subspace
        self.h2_van_vleck_by_subspace = h2_van_vleck_by_subspace
        self.h2_by_subspace = selected_h2_by_subspace
        self.h2_effective = selected_h2
        self.h_eff = selected_h_eff
        self.h2_bloch = h2_bloch
        self.h_eff_bloch = h_eff_bloch
        self.h2_van_vleck = h2_van_vleck
        self.h_eff_van_vleck = h_eff_van_vleck
        self.e_qd = eigenvalues
        self.eigenvectors = eigenvectors
        self.left_eigenvectors = left
        self.right_eigenvectors = right
        self.model_overlap = overlap_matrix
        self.diagnostics = diagnostics
        self.result = QDSCNEVPT2Result(
            qd_type=qd_type,
            roots=roots,
            reference_energies=reference_energies,
            h2_by_subspace=selected_h2_by_subspace,
            h2_effective=selected_h2,
            h_eff=selected_h_eff,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            left_eigenvectors=left,
            right_eigenvectors=right,
            h2_bloch_by_subspace=h2_bloch_by_subspace,
            h2_bloch=h2_bloch,
            h_eff_bloch=h_eff_bloch,
            h2_van_vleck_by_subspace=h2_van_vleck_by_subspace,
            h2_van_vleck=h2_van_vleck,
            h_eff_van_vleck=h_eff_van_vleck,
            diagnostics=diagnostics,
        )
        logger.note(self, "QD_Bloch effective Hamiltonian:\n%s", h_eff_bloch)
        logger.note(
            self,
            "QD_VanVleck effective Hamiltonian:\n%s",
            h_eff_van_vleck,
        )
        logger.note(self, "selected qd_type: %s", qd_type)
        if qd_type == "bloch":
            logger.info(
                self,
                "QD_Bloch non-Hermiticity=%.3e  right residual=%.3e  "
                "left residual=%.3e  cond(R)=%.3e",
                eig_diagnostics["maximum_nonhermiticity"],
                eig_diagnostics["right_residual_norm"],
                eig_diagnostics["left_residual_norm"],
                eig_diagnostics["right_eigenvector_condition_number"],
            )
        else:
            logger.info(
                self,
                "QD_VanVleck Hermiticity=%.3e  residual=%.3e  "
                "orthonormality=%.3e",
                eig_diagnostics["maximum_matrix_nonhermiticity"],
                eig_diagnostics["residual_norm"],
                eig_diagnostics["orthonormality_error"],
            )
        logger.note(
            self,
            "selected QD eigenvalues (%s): %s",
            qd_type,
            eigenvalues,
        )
        return self.e_qd


X2CQDSCNEVPT2 = WickX2CQDSCNEVPT2


class WickX2CQDBlochSCNEVPT2(WickX2CQDSCNEVPT2):
    """Backward-compatible wrapper that always selects ``qd_type='bloch'``."""

    def __init__(self, mc, frozen=0, qd_type="bloch"):
        qd_type = _normalize_qd_type(qd_type)
        if qd_type != "bloch":
            raise ValueError(
                "WickX2CQDBlochSCNEVPT2 only supports qd_type='bloch'; "
                "use WickX2CQDSCNEVPT2 for canonical Van Vleck/HQD"
            )
        super().__init__(mc, frozen=frozen, qd_type="bloch")

    def kernel(self, *args, qd_type=None, **kwargs):
        # The generic fluent ``run`` applies public settings before forwarding
        # calculation-local keywords.  Inspect both the stored and explicit
        # values so the historical class name can never silently select HQD.
        requested = self.qd_type if qd_type is None else qd_type
        try:
            requested = _normalize_qd_type(requested)
        except ValueError:
            self._clear_results()
            self.qd_type = "bloch"
            raise
        if requested != "bloch":
            # ``run`` follows PySCF convention and applies public settings
            # before entering ``kernel``.  Restore the compatibility
            # wrapper's invariant even when that requested setting is
            # rejected transactionally.
            self.qd_type = "bloch"
            self._clear_results()
            raise ValueError(
                "WickX2CQDBlochSCNEVPT2 only supports qd_type='bloch'; "
                "use WickX2CQDSCNEVPT2 for canonical Van Vleck/HQD"
            )
        return super().kernel(*args, qd_type="bloch", **kwargs)


X2CQDBlochSCNEVPT2 = WickX2CQDBlochSCNEVPT2
