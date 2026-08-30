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


class _PulaySolveError(np.linalg.LinAlgError):
    """Internal Pulay rejection that retains the failed candidate details."""

    def __init__(self, reason, message, diagnostics):
        super().__init__(message)
        self.reason = str(reason)
        self.diagnostics = diagnostics


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


class AndersonOrbitalDIIS:
    """Gauge-consistent Anderson acceleration for Super-CIPT fixed points.

    The contributed prototype accumulated local generators as
    ``theta += kappa``.  That identity is false for finite, noncommuting
    unitary rotations, and projection or trust scaling makes the accumulated
    coordinates drift even farther from the actual orbitals.  Here every
    current and PT-trial orbital set is mapped afresh to logarithmic
    coordinates relative to one reference chosen when acceleration starts.
    The common-frame fixed-point residual is ``theta_trial-theta_current``.

    A relative-SVD Pulay solve, coefficient bound, true-gradient descent
    check, symmetry projection, and invariant Frobenius trust radius protect
    the expensive CI/DMRG evaluation from obviously pathological proposals.
    The historical class name is retained as an internal compatibility alias;
    its algorithm is no longer incremental accumulation.
    """

    def __init__(
        self,
        reference_mo,
        overlap,
        *,
        space=15,
        start_cycle=3,
        start_gradient=0.02,
        relative_cutoff=1e-8,
        coefficient_l1_max=50.0,
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
        self.relative_cutoff = float(relative_cutoff)
        self.coefficient_l1_max = float(coefficient_l1_max)
        if not 0.0 < self.relative_cutoff < 1.0:
            raise ValueError("DIIS relative cutoff must lie between zero and one")
        if self.coefficient_l1_max <= 1.0:
            raise ValueError("DIIS coefficient L1 bound must exceed one")
        self.started = False
        self._targets = []
        self._errors = []

    def reset(self):
        self.started = False
        self._targets = []
        self._errors = []

    def _should_start(self, cycle, gradient_norm):
        return int(cycle) >= self.start_cycle or (
            self.start_gradient is not None
            and gradient_norm is not None
            and float(gradient_norm) < self.start_gradient
        )

    def _start(self, current_mo):
        self.reference_mo = np.array(
            current_mo, dtype=np.complex128, copy=True
        )
        self.started = True
        self._targets = []
        self._errors = []

    def _relative_unitary(self, orbitals):
        return _nearest_unitary(
            self.reference_mo.T.conj().dot(self.overlap).dot(orbitals)
        )

    def _pulay_target(self):
        """Return extrapolated theta and bounded least-squares diagnostics."""
        count = len(self._errors)
        diagnostics = {
            "vectors": count,
            "solver": "constrained-residual-svd",
            "gram_condition": None,
            "gram_eigenvalues": [],
            "residual_singular_values": [],
            "residual_svd_rank": 0,
            "residual_condition": None,
            "coefficient_sum": None,
            "coefficients": [],
            "coefficient_l1_norm": None,
            "coefficient_maximum": None,
            "predicted_residual_norm": None,
            "best_residual_norm": None,
        }
        if count < 2:
            return self._targets[-1], False, diagnostics

        # Anti-Hermitian matrices form a *real* vector space.  Splitting the
        # residuals into real and imaginary components therefore gives the
        # same metric as Re(vdot(left, right)) while making the real-coefficient
        # Pulay problem explicit.  Solve the residual least-squares problem
        # directly instead of forming an augmented Gram KKT system: the latter
        # squares the condition number, and applying an SVD cutoff to the KKT
        # spectrum is not equivalent to the advertised residual cutoff.
        complex_residuals = np.stack(
            [np.asarray(error, dtype=np.complex128).ravel() for error in self._errors]
        )
        residuals = np.concatenate(
            (complex_residuals.real, complex_residuals.imag), axis=1
        )
        residual_scale = float(np.max(abs(residuals)))
        if not np.isfinite(residual_scale):
            raise _PulaySolveError(
                "nonfinite-pulay-residual",
                "nonfinite DIIS fixed-point residual",
                diagnostics,
            )
        if residual_scale:
            scaled_residuals = residuals / residual_scale
        else:
            scaled_residuals = residuals

        gram = scaled_residuals.dot(scaled_residuals.T)
        gram = 0.5 * (gram + gram.T)
        eigenvalues = np.linalg.eigvalsh(gram)
        with np.errstate(over="ignore", invalid="ignore"):
            residual_scale_squared = np.multiply(residual_scale, residual_scale)
            physical_eigenvalues = eigenvalues * residual_scale_squared
        diagnostics["gram_eigenvalues"] = physical_eigenvalues.tolist()

        residual_norms = np.linalg.norm(scaled_residuals, axis=1)
        # Prefer the newest vector on ties (notably when every residual is
        # exactly zero), so a rank-zero solve reduces to the current PT map.
        best_index = count - 1 - int(np.argmin(residual_norms[::-1]))
        baseline = np.zeros(count)
        baseline[best_index] = 1.0
        # c = baseline + Z y enforces 1^T c = 1 exactly because the columns
        # of Z span null(1^T).  Choosing the best stored residual as the
        # particular solution also gives a safe no-acceleration fallback when
        # the relative SVD truncates every useful correction.
        nullspace = scipy.linalg.null_space(np.ones((1, count)))
        design = scaled_residuals.T.dot(nullspace)
        rhs = -scaled_residuals.T.dot(baseline)
        try:
            correction, _, rank, singular_values = scipy.linalg.lstsq(
                design,
                rhs,
                cond=self.relative_cutoff,
                lapack_driver="gelsd",
            )
        except (np.linalg.LinAlgError, scipy.linalg.LinAlgError) as error:
            raise _PulaySolveError(
                "pulay-svd-failure",
                "residual-space Pulay SVD failed",
                diagnostics,
            ) from error
        coefficients = baseline + nullspace.dot(correction)

        singular_values = np.asarray(singular_values, dtype=float)
        diagnostics["residual_singular_values"] = (
            singular_values * residual_scale
        ).tolist()
        diagnostics["residual_svd_rank"] = int(rank)
        if rank:
            retained = singular_values[:rank]
            residual_condition = float(retained[0] / retained[-1])
            diagnostics["residual_condition"] = residual_condition
            diagnostics["gram_condition"] = residual_condition**2

        predicted_scaled = float(np.linalg.norm(coefficients.dot(scaled_residuals)))
        best_scaled = float(residual_norms[best_index])
        # A truncated solve should never be worse than doing no acceleration.
        # Roundoff can otherwise turn an old, very large residual into a poor
        # average of the small vectors that follow it.
        if predicted_scaled > best_scaled * (1.0 + 1e-12):
            coefficients = baseline
            predicted_scaled = best_scaled
            diagnostics["svd_fallback_to_best"] = True
        else:
            diagnostics["svd_fallback_to_best"] = False

        coefficient_l1 = float(np.sum(abs(coefficients)))
        coefficient_maximum = float(np.max(abs(coefficients)))
        diagnostics.update(
            {
                "coefficients": coefficients.tolist(),
                "coefficient_sum": float(np.sum(coefficients)),
                "coefficient_l1_norm": coefficient_l1,
                "coefficient_maximum": coefficient_maximum,
                "predicted_residual_norm": predicted_scaled * residual_scale,
                "best_residual_norm": best_scaled * residual_scale,
            }
        )
        if not np.all(np.isfinite(coefficients)):
            raise _PulaySolveError(
                "nonfinite-pulay-coefficients",
                "nonfinite DIIS coefficients",
                diagnostics,
            )
        if coefficient_l1 > self.coefficient_l1_max:
            raise _PulaySolveError(
                "coefficient-l1-limit",
                "DIIS coefficient L1 norm exceeds its limit",
                diagnostics,
            )
        target = np.zeros_like(self._targets[-1])
        for coefficient, theta in zip(coefficients, self._targets):
            target += coefficient * theta
        latest = np.zeros(count)
        latest[-1] = 1.0
        extrapolated = not np.allclose(
            coefficients,
            latest,
            rtol=1e-12,
            atol=1e-14,
        )
        return _antihermitian(target), extrapolated, diagnostics

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
        """Extrapolate one PT fixed-point map and return its safe local step."""
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

        plain_relative = _nearest_unitary(
            current_mo.T.conj().dot(self.overlap).dot(proposed_mo)
        )
        plain_incremental = _unitary_log(plain_relative)
        if not self.started and self._should_start(cycle, gradient_norm):
            self._start(current_mo)

        extrapolated = False
        reset_after_failure = False
        rejection_reason = None
        pulay_diagnostics = {
            "vectors": 0,
            "solver": "constrained-residual-svd",
            "gram_condition": None,
            "gram_eigenvalues": [],
            "residual_singular_values": [],
            "residual_svd_rank": 0,
            "residual_condition": None,
            "coefficient_sum": None,
            "coefficients": [],
            "coefficient_l1_norm": None,
            "coefficient_maximum": None,
            "predicted_residual_norm": None,
            "best_residual_norm": None,
        }
        if self.started:
            current_unitary = self._relative_unitary(current_mo)
            proposed_unitary = self._relative_unitary(proposed_mo)
            current_theta = _unitary_log(current_unitary)
            trial_theta = _unitary_log(proposed_unitary)
            residual = _antihermitian(trial_theta - current_theta)
            self._targets.append(trial_theta)
            self._errors.append(residual.ravel())
            if len(self._targets) > self.space:
                self._targets.pop(0)
                self._errors.pop(0)
            try:
                target_theta, extrapolated, pulay_diagnostics = (
                    self._pulay_target()
                )
                target_unitary = scipy.linalg.expm(target_theta)
                incremental = _unitary_log(
                    current_unitary.T.conj().dot(target_unitary)
                )
            except _PulaySolveError as error:
                pulay_diagnostics = error.diagnostics
                self._targets = [trial_theta]
                self._errors = [residual.ravel()]
                incremental = plain_incremental
                reset_after_failure = True
                rejection_reason = error.reason
            except (np.linalg.LinAlgError, scipy.linalg.LinAlgError):
                self._targets = [trial_theta]
                self._errors = [residual.ravel()]
                incremental = plain_incremental
                reset_after_failure = True
                rejection_reason = "pulay-coordinate-failure"
        else:
            incremental = plain_incremental

        projection_info = None
        def project(value):
            nonlocal projection_info
            if projector is None:
                return _antihermitian(value)
            projected = projector(current_mo, value)
            if isinstance(projected, tuple):
                value, projection_info = projected
                return _antihermitian(value)
            return _antihermitian(projected)

        plain_incremental = project(plain_incremental)
        incremental = project(incremental)
        plain_derivative = float(np.vdot(gradient, plain_incremental).real)
        proposed_derivative = float(np.vdot(gradient, incremental).real)
        descent_threshold = -1e-12 * max(
            1.0, np.linalg.norm(gradient) * np.linalg.norm(incremental)
        )
        if extrapolated and (
            not np.isfinite(proposed_derivative)
            or proposed_derivative >= descent_threshold
        ):
            incremental = plain_incremental
            proposed_derivative = plain_derivative
            extrapolated = False
            rejection_reason = "non-descent-direction"
            self._targets = self._targets[-1:]
            self._errors = self._errors[-1:]

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
            "extrapolation_rejection": rejection_reason,
            "vectors": int(len(self._targets)) if self.started else 0,
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
            "coordinate_system": "fixed-reference-unitary-log",
            "error_vector": "fixed-point-residual",
            "plain_directional_derivative": plain_derivative,
            "proposed_directional_derivative": proposed_derivative,
            "relative_svd_cutoff": self.relative_cutoff,
            "coefficient_l1_limit": self.coefficient_l1_max,
            "pulay": pulay_diagnostics,
        }
        return OrbitalDIISResult(result_mo, incremental, diagnostics)


# Internal compatibility for code that imported the prototype's historical
# name.  New code should use the algorithmically descriptive name above.
IncrementalOrbitalDIIS = AndersonOrbitalDIIS


def reduce_metric(mo_coeff, overlap):
    """Return ``C^H S C`` for an orbital coefficient matrix."""
    mo_coeff = np.asarray(mo_coeff)
    return mo_coeff.T.conj().dot(overlap).dot(mo_coeff)
