"""Pulay extrapolation of unitary orbital transformations.

Full Super-CI uses anti-Hermitian logarithms relative to one fixed orthonormal
reference, so all gradient/error vectors share a tangent frame.  Super-CIPT
instead accumulates its local perturbative corrections and uses those
corrections as Pulay errors, matching the coordinates of the PT algorithm.
"""

from dataclasses import dataclass

import numpy as np
import scipy.linalg
from pyscf import lib


def _antihermitian(matrix):
    matrix = np.asarray(matrix, dtype=np.complex128)
    return 0.5 * (matrix - matrix.T.conj())


def _nearest_unitary(matrix):
    left, _, right = scipy.linalg.svd(
        np.asarray(matrix, dtype=np.complex128), full_matrices=False
    )
    return left.dot(right)


def _unitary_log(matrix):
    unitary = _nearest_unitary(matrix)
    return _antihermitian(scipy.linalg.logm(unitary))


@dataclass
class OrbitalDIISResult:
    """One orbital-DIIS proposal and its diagnostics."""

    mo_coeff: np.ndarray
    generator: np.ndarray
    diagnostics: dict


class OrbitalDIIS:
    """DIIS in a fixed unitary-orbital coordinate system.

    Parameters use Dutta's defaults: a 15-vector Pulay space starts after
    macroiteration 2 or once the orbital-gradient norm is below ``0.02``.
    """

    def __init__(
        self,
        reference_mo,
        overlap,
        *,
        space=15,
        start_cycle=3,
        start_gradient=0.02,
    ):
        self.reference_mo = np.array(
            reference_mo, dtype=np.complex128, copy=True
        )
        self.overlap = np.asarray(overlap, dtype=np.complex128)
        if self.overlap.shape != (self.reference_mo.shape[0],) * 2:
            raise ValueError("AO overlap and orbital dimensions disagree")
        metric = reduce_metric(self.reference_mo, self.overlap)
        if np.max(abs(metric - np.eye(metric.shape[0]))) > 1e-7:
            raise ValueError("DIIS reference orbitals are not orthonormal")
        self.space = int(space)
        self.start_cycle = int(start_cycle)
        self.start_gradient = (
            None if start_gradient is None else float(start_gradient)
        )
        if self.space < 2:
            raise ValueError("DIIS space must be at least 2")
        if self.start_cycle < 0:
            raise ValueError("DIIS start cycle must be nonnegative")
        if self.start_gradient is not None and self.start_gradient <= 0:
            raise ValueError("DIIS start gradient must be positive")
        self.started = False
        self._diis = self._new_diis()

    def _new_diis(self):
        extrapolator = lib.diis.DIIS()
        extrapolator.space = self.space
        extrapolator.min_space = 2
        return extrapolator

    def reset(self):
        """Clear the Pulay history while retaining all controls."""
        self.started = False
        self._diis = self._new_diis()

    def _relative_unitary(self, left, right):
        return _nearest_unitary(
            np.asarray(left).T.conj().dot(self.overlap).dot(right)
        )

    def update(
        self,
        current_mo,
        proposed_mo,
        gradient,
        *,
        cycle,
        gradient_norm=None,
        max_stepsize=None,
        step_metric="frobenius",
        projector=None,
    ):
        """Extrapolate an orbital proposal and return its current-frame step.

        ``gradient`` is the anti-Hermitian orbital gradient in the current MO
        frame. ``projector(mo, generator)`` may impose frozen-orbital,
        point-group, or Kramers constraints and should return either the
        projected matrix or ``(matrix, diagnostics)``.
        """
        current_mo = np.asarray(current_mo, dtype=np.complex128)
        proposed_mo = np.asarray(proposed_mo, dtype=np.complex128)
        gradient = _antihermitian(gradient)
        if current_mo.shape != self.reference_mo.shape:
            raise ValueError("current and reference orbital dimensions disagree")
        if proposed_mo.shape != current_mo.shape:
            raise ValueError("proposed and current orbital dimensions disagree")
        if gradient.shape != (current_mo.shape[1],) * 2:
            raise ValueError("orbital gradient has the wrong dimensions")
        if step_metric not in ("frobenius", "maximum"):
            raise ValueError("step_metric must be 'frobenius' or 'maximum'")

        start_by_gradient = (
            self.start_gradient is not None
            and gradient_norm is not None
            and float(gradient_norm) < self.start_gradient
        )
        if not self.started and (
            int(cycle) >= self.start_cycle or start_by_gradient
        ):
            self.started = True

        current_u = self._relative_unitary(self.reference_mo, current_mo)
        proposed_u = self._relative_unitary(self.reference_mo, proposed_mo)
        target_theta = _unitary_log(proposed_u)
        gradient_reference = _antihermitian(
            current_u.dot(gradient).dot(current_u.T.conj())
        )

        extrapolated = False
        reset_after_failure = False
        if self.started:
            try:
                target_vector = self._diis.update(
                    target_theta.ravel(),
                    xerr=gradient_reference.ravel(),
                )
                extrapolated = self._diis.get_num_vec() >= self._diis.min_space
                target_theta = _antihermitian(
                    np.asarray(target_vector).reshape(target_theta.shape)
                )
            except (np.linalg.LinAlgError, scipy.linalg.LinAlgError):
                # A singular Pulay system should lose acceleration, not the
                # underlying well-defined orbital step.
                self._diis = self._new_diis()
                self._diis.update(
                    target_theta.ravel(),
                    xerr=gradient_reference.ravel(),
                )
                reset_after_failure = True

        target_mo = self.reference_mo.dot(scipy.linalg.expm(target_theta))
        incremental = _unitary_log(
            self._relative_unitary(current_mo, target_mo)
        )
        projection_info = None
        if projector is not None:
            projected = projector(current_mo, incremental)
            if isinstance(projected, tuple):
                incremental, projection_info = projected
            else:
                incremental = projected
            incremental = _antihermitian(incremental)

        if step_metric == "maximum":
            proposed_size = float(np.max(abs(incremental)))
        else:
            proposed_size = float(np.linalg.norm(incremental))
        scale = 1.0
        if max_stepsize is not None:
            max_stepsize = float(max_stepsize)
            if max_stepsize <= 0:
                raise ValueError("maximum DIIS step must be positive")
            if proposed_size > max_stepsize:
                scale = max_stepsize / proposed_size
                incremental *= scale

        result_mo = current_mo.dot(scipy.linalg.expm(incremental))
        orthonormality_error = float(
            np.max(
                abs(
                    reduce_metric(result_mo, self.overlap)
                    - np.eye(result_mo.shape[1])
                )
            )
        )
        diagnostics = {
            "enabled": True,
            "started": bool(self.started),
            "extrapolated": bool(extrapolated),
            "vectors": int(self._diis.get_num_vec()) if self.started else 0,
            "space": self.space,
            "start_cycle": self.start_cycle,
            "start_gradient": self.start_gradient,
            "proposed_increment_norm": float(np.linalg.norm(incremental) / scale)
            if scale
            else 0.0,
            "applied_increment_norm": float(np.linalg.norm(incremental)),
            "step_scale": float(scale),
            "step_metric": step_metric,
            "pulay_reset": bool(reset_after_failure),
            "orthonormality_error": orthonormality_error,
            "projection": projection_info,
        }
        return OrbitalDIISResult(result_mo, incremental, diagnostics)


class IncrementalOrbitalDIIS:
    """DIIS for perturbative orbital corrections in accumulated coordinates.

    Super-CIPT supplies a local perturbative correction at every macrostep.
    Following the contributed Guo--Dutta implementation, this class adds that
    correction to an accumulated rotation parameter and uses the correction
    itself as the Pulay error.  This avoids mixing the redundant
    core--core/virtual--virtual semicanonical gauges into the fixed-reference
    logarithms used by :class:`OrbitalDIIS` for full Super-CI.

    The accumulated coordinates start when DIIS starts; the preceding plain
    PT steps define the local reference.  All proposed increments are still
    projected and trust-limited before they are applied.
    """

    def __init__(
        self,
        reference_mo,
        overlap,
        *,
        space=15,
        start_cycle=3,
        start_gradient=0.02,
    ):
        self.reference_mo = np.array(
            reference_mo, dtype=np.complex128, copy=True
        )
        self.overlap = np.asarray(overlap, dtype=np.complex128)
        if self.overlap.shape != (self.reference_mo.shape[0],) * 2:
            raise ValueError("AO overlap and orbital dimensions disagree")
        metric = reduce_metric(self.reference_mo, self.overlap)
        if np.max(abs(metric - np.eye(metric.shape[0]))) > 1e-7:
            raise ValueError("DIIS reference orbitals are not orthonormal")
        self.space = int(space)
        self.start_cycle = int(start_cycle)
        self.start_gradient = (
            None if start_gradient is None else float(start_gradient)
        )
        if self.space < 2:
            raise ValueError("DIIS space must be at least 2")
        if self.start_cycle < 0:
            raise ValueError("DIIS start cycle must be nonnegative")
        if self.start_gradient is not None and self.start_gradient <= 0:
            raise ValueError("DIIS start gradient must be positive")
        self.started = False
        self._theta = None
        self._diis = self._new_diis()

    def _new_diis(self):
        extrapolator = lib.diis.DIIS()
        extrapolator.space = self.space
        extrapolator.min_space = 2
        return extrapolator

    def reset(self):
        self.started = False
        self._theta = None
        self._diis = self._new_diis()

    def _should_start(self, cycle, gradient_norm):
        return int(cycle) >= self.start_cycle or (
            self.start_gradient is not None
            and gradient_norm is not None
            and float(gradient_norm) < self.start_gradient
        )

    def update(
        self,
        current_mo,
        proposed_mo,
        gradient,
        *,
        cycle,
        gradient_norm=None,
        max_stepsize=None,
        step_metric="maximum",
        projector=None,
    ):
        """Extrapolate one local PT correction and return the applied step."""
        del gradient  # The perturbative correction is the Pulay residual.
        current_mo = np.asarray(current_mo, dtype=np.complex128)
        proposed_mo = np.asarray(proposed_mo, dtype=np.complex128)
        if current_mo.shape != self.reference_mo.shape:
            raise ValueError("current and reference orbital dimensions disagree")
        if proposed_mo.shape != current_mo.shape:
            raise ValueError("proposed and current orbital dimensions disagree")
        if step_metric not in ("frobenius", "maximum"):
            raise ValueError("step_metric must be 'frobenius' or 'maximum'")

        relative = _nearest_unitary(
            current_mo.T.conj().dot(self.overlap).dot(proposed_mo)
        )
        correction = _unitary_log(relative)
        if not self.started and self._should_start(cycle, gradient_norm):
            self.started = True
            self._theta = np.zeros_like(correction)

        extrapolated = False
        reset_after_failure = False
        if self.started:
            trial_theta = self._theta + correction
            try:
                target_vector = self._diis.update(
                    trial_theta.ravel(),
                    xerr=correction.ravel(),
                )
                extrapolated = self._diis.get_num_vec() >= self._diis.min_space
                incremental = _antihermitian(
                    np.asarray(target_vector).reshape(correction.shape)
                    - self._theta
                )
            except (np.linalg.LinAlgError, scipy.linalg.LinAlgError):
                self._diis = self._new_diis()
                self._diis.update(
                    trial_theta.ravel(),
                    xerr=correction.ravel(),
                )
                incremental = correction
                reset_after_failure = True
        else:
            incremental = correction

        projection_info = None
        if projector is not None:
            projected = projector(current_mo, incremental)
            if isinstance(projected, tuple):
                incremental, projection_info = projected
            else:
                incremental = projected
            incremental = _antihermitian(incremental)

        if step_metric == "maximum":
            proposed_size = float(np.max(abs(incremental)))
        else:
            proposed_size = float(np.linalg.norm(incremental))
        scale = 1.0
        if max_stepsize is not None:
            max_stepsize = float(max_stepsize)
            if max_stepsize <= 0:
                raise ValueError("maximum DIIS step must be positive")
            if proposed_size > max_stepsize:
                scale = max_stepsize / proposed_size
                incremental *= scale

        if self.started:
            self._theta = _antihermitian(self._theta + incremental)
        result_mo = current_mo.dot(scipy.linalg.expm(incremental))
        orthonormality_error = float(
            np.max(
                abs(
                    reduce_metric(result_mo, self.overlap)
                    - np.eye(result_mo.shape[1])
                )
            )
        )
        diagnostics = {
            "enabled": True,
            "started": bool(self.started),
            "extrapolated": bool(extrapolated),
            "vectors": int(self._diis.get_num_vec()) if self.started else 0,
            "space": self.space,
            "start_cycle": self.start_cycle,
            "start_gradient": self.start_gradient,
            "proposed_increment_norm": float(np.linalg.norm(incremental) / scale)
            if scale
            else 0.0,
            "applied_increment_norm": float(np.linalg.norm(incremental)),
            "step_scale": float(scale),
            "step_metric": step_metric,
            "pulay_reset": bool(reset_after_failure),
            "orthonormality_error": orthonormality_error,
            "projection": projection_info,
            "coordinate_system": "accumulated-incremental",
            "error_vector": "perturbative-step",
        }
        return OrbitalDIISResult(result_mo, incremental, diagnostics)


def reduce_metric(mo_coeff, overlap):
    """Return ``C^H S C`` for an orbital coefficient matrix."""
    mo_coeff = np.asarray(mo_coeff)
    return mo_coeff.T.conj().dot(overlap).dot(mo_coeff)
