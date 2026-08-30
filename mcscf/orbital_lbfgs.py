"""PT-seeded limited-memory BFGS directions for unitary orbitals.

The orbital gradient, Super-CIPT preconditioned direction, and accepted
line-search secants are compared in one fixed reference frame.  If
``C = C_ref U``, a local anti-Hermitian tangent ``a`` is represented as
``U a U^H``.  This adjoint transport is isometric and avoids subtracting
gradients expressed in different moving orbital frames.

The initial inverse Hessian is not merely a scalar.  At every accepted
orbital point the current gradient ``g`` and uphill PT-preconditioned vector
``z = -kappa_PT`` define an inverse-BFGS rank-two action which is symmetric
positive definite and satisfies ``H0 g = z``.  A scalar action is used when
the PT vector is poorly aligned with the gradient.  Accepted line-search
secants then modify this action through the standard L-BFGS two-loop
recursion.

State changes are transactional: :meth:`propose` creates a pending search
direction, while only :meth:`accept_secant` may add the line search's
accepted fixed-frame secant.  :meth:`reject` discards the pending proposal,
so rejected CI/RDM gradients cannot enter the quasi-Newton history.
"""

from dataclasses import dataclass

import numpy as np
import scipy.linalg


def _antihermitian(matrix):
    matrix = np.asarray(matrix, dtype=np.complex128)
    return 0.5 * (matrix - matrix.T.conj())


def _nearest_unitary(matrix):
    left, _, right = scipy.linalg.svd(
        np.asarray(matrix, dtype=np.complex128), full_matrices=False
    )
    return left.dot(right)


def _inner(left, right):
    """Real Frobenius product on the anti-Hermitian tangent space."""
    return float(np.vdot(left, right).real)


def _metric(mo_coeff, overlap):
    mo_coeff = np.asarray(mo_coeff, dtype=np.complex128)
    return mo_coeff.T.conj().dot(overlap).dot(mo_coeff)


@dataclass(frozen=True)
class OrbitalLBFGSProposal:
    """One PT-seeded L-BFGS orbital proposal and its diagnostics."""

    mo_coeff: np.ndarray
    generator: np.ndarray
    diagnostics: dict


@dataclass(frozen=True)
class _SecantPair:
    step: np.ndarray
    gradient_difference: np.ndarray
    inverse_curvature: float


class PTSeededOrbitalLBFGS:
    """Fixed-reference Riemannian PT-seeded L-BFGS core.

    Parameters
    ----------
    reference_mo
        Orthonormal AO-to-MO coefficients defining the common tangent frame.
    overlap
        AO overlap matrix.
    memory
        Maximum number of accepted line-search secants.  The default is 7.
    scalar_min, scalar_max
        Bounds for ``h = <g,z>/<g,g>`` in the scalar part of ``H0``.
    pt_cosine_threshold
        The rank-two PT seed is used only if the cosine between ``g`` and
        ``z`` reaches this value.  A poorly aligned PT response instead uses
        the bounded scalar action ``h I``.
    descent_tolerance
        A direction is accepted only if its gradient cosine is smaller than
        ``-descent_tolerance``.  Otherwise the L-BFGS memory is cleared and
        the direction falls back to projected plain PT.
    accepted_point_tolerance
        Orthonormality tolerance used for orbital validation.
    secant_direction_tolerance
        Relative residual allowed when verifying that an accepted line-search
        step is a positive scalar multiple of the pending proposal direction.

    Notes
    -----
    ``preconditioned_direction`` passed to :meth:`propose` is the downhill
    Super-CIPT generator ``kappa_PT``; internally ``z = -kappa_PT``.

    :meth:`accept_secant` consumes the fixed-reference ``(s,y)`` returned by
    an accepted line search.  It applies the curvature floor

    ``max(1e-12, 1e-6*sqrt(<s,s><y,y>), <s,s>/1e3)``.

    Small curvature is minimally Levenberg-damped along ``s``.  Significantly
    negative curvature is skipped, and two consecutive such pairs reset the
    memory.
    """

    def __init__(
        self,
        reference_mo,
        overlap,
        *,
        memory=7,
        scalar_min=1e-3,
        scalar_max=50.0,
        pt_cosine_threshold=0.1,
        descent_tolerance=1e-2,
        accepted_point_tolerance=1e-7,
        secant_direction_tolerance=1e-7,
    ):
        self.reference_mo = np.array(
            reference_mo, dtype=np.complex128, copy=True
        )
        self.overlap = np.asarray(overlap, dtype=np.complex128)
        if self.reference_mo.ndim != 2:
            raise ValueError("reference orbitals must be a matrix")
        if self.overlap.shape != (self.reference_mo.shape[0],) * 2:
            raise ValueError("AO overlap and orbital dimensions disagree")
        identity = np.eye(self.reference_mo.shape[1])
        if np.max(abs(_metric(self.reference_mo, self.overlap) - identity)) > 1e-7:
            raise ValueError("L-BFGS reference orbitals are not orthonormal")

        if isinstance(memory, bool) or int(memory) != memory or memory <= 0:
            raise ValueError("L-BFGS memory must be a positive integer")
        self.memory = int(memory)
        self.scalar_min = float(scalar_min)
        self.scalar_max = float(scalar_max)
        self.pt_cosine_threshold = float(pt_cosine_threshold)
        self.descent_tolerance = float(descent_tolerance)
        self.accepted_point_tolerance = float(accepted_point_tolerance)
        self.secant_direction_tolerance = float(
            secant_direction_tolerance
        )
        if not 0.0 < self.scalar_min <= self.scalar_max:
            raise ValueError("invalid initial inverse-Hessian scalar bounds")
        if not 0.0 <= self.pt_cosine_threshold <= 1.0:
            raise ValueError("PT cosine threshold must lie in [0, 1]")
        if not 0.0 <= self.descent_tolerance < 1.0:
            raise ValueError("descent tolerance must lie in [0, 1)")
        if self.accepted_point_tolerance <= 0.0:
            raise ValueError("accepted-point tolerance must be positive")
        if self.secant_direction_tolerance <= 0.0:
            raise ValueError("secant direction tolerance must be positive")

        self._pairs = []
        self._pending = None
        self._negative_curvature_streak = 0

    @property
    def pair_count(self):
        """Number of accepted secant pairs currently retained."""
        return len(self._pairs)

    @property
    def has_pending(self):
        """Whether a proposal still needs acceptance or rejection."""
        return self._pending is not None

    @property
    def negative_curvature_streak(self):
        """Number of consecutive significantly negative accepted secants."""
        return self._negative_curvature_streak

    def _validate_mo(self, mo_coeff, label):
        mo_coeff = np.asarray(mo_coeff, dtype=np.complex128)
        if mo_coeff.shape != self.reference_mo.shape:
            raise ValueError(f"{label} and reference orbital dimensions disagree")
        if not np.all(np.isfinite(mo_coeff)):
            raise ValueError(f"{label} contains nonfinite values")
        identity = np.eye(mo_coeff.shape[1])
        error = float(np.max(abs(_metric(mo_coeff, self.overlap) - identity)))
        if error > self.accepted_point_tolerance:
            raise ValueError(f"{label} are not orthonormal")
        return mo_coeff

    def _validate_tangent(self, tangent, label):
        tangent = np.asarray(tangent, dtype=np.complex128)
        norb = self.reference_mo.shape[1]
        if tangent.shape != (norb, norb):
            raise ValueError(f"{label} has the wrong dimensions")
        if not np.all(np.isfinite(tangent)):
            raise ValueError(f"{label} contains nonfinite values")
        return _antihermitian(tangent)

    def _frame_unitary(self, mo_coeff):
        return _nearest_unitary(
            self.reference_mo.T.conj().dot(self.overlap).dot(mo_coeff)
        )

    def _to_reference(self, mo_coeff, tangent):
        unitary = self._frame_unitary(mo_coeff)
        return _antihermitian(unitary.dot(tangent).dot(unitary.T.conj()))

    def _from_reference(self, mo_coeff, tangent):
        unitary = self._frame_unitary(mo_coeff)
        return _antihermitian(
            unitary.T.conj().dot(tangent).dot(unitary)
        )

    def transport(self, tangent, from_mo, to_mo):
        """Isometrically transport a tangent through the fixed frame."""
        from_mo = self._validate_mo(from_mo, "transport source orbitals")
        to_mo = self._validate_mo(to_mo, "transport target orbitals")
        tangent = self._validate_tangent(tangent, "transport tangent")
        return self._from_reference(
            to_mo, self._to_reference(from_mo, tangent)
        )

    def _initial_inverse_action(self, gradient, preconditioned, vector):
        """Return ``H0 vector`` and diagnostics in the reference frame."""
        gg = _inner(gradient, gradient)
        zz = _inner(preconditioned, preconditioned)
        gz = _inner(gradient, preconditioned)
        scale = float(np.sqrt(max(0.0, gg * zz)))
        cosine = gz / scale if scale > 0.0 else None
        scalar_unclipped = gz / gg if gg > 0.0 else self.scalar_min
        if not np.isfinite(scalar_unclipped) or scalar_unclipped <= 0.0:
            scalar_unclipped = self.scalar_min
        scalar = float(
            np.clip(scalar_unclipped, self.scalar_min, self.scalar_max)
        )

        use_rank_two = bool(
            gg > np.finfo(float).tiny
            and zz > np.finfo(float).tiny
            and gz > np.finfo(float).tiny
            and cosine is not None
            and cosine >= self.pt_cosine_threshold
        )
        if use_rank_two:
            # Inverse-BFGS update of h*I with the synthetic secant
            # (s,y)=(z,g).  Written as an action so no dense fourth-rank
            # orbital Hessian is ever built:
            # H0=(I-rho*z*g') hI (I-rho*g*z') + rho*z*z'.
            rho = 1.0 / gz
            right = vector - rho * gradient * _inner(
                preconditioned, vector
            )
            middle = scalar * right
            result = (
                middle
                - rho * preconditioned * _inner(gradient, middle)
                + rho * preconditioned * _inner(preconditioned, vector)
            )
            mode = "PT-secant-rank-two"
        else:
            result = scalar * vector
            mode = "bounded-scalar"

        diagnostics = {
            "mode": mode,
            "rank_two": use_rank_two,
            "pt_curvature": gz,
            "pt_cosine": cosine,
            "pt_cosine_threshold": self.pt_cosine_threshold,
            "scalar": scalar,
            "scalar_unclipped": float(scalar_unclipped),
            "scalar_clipped": not np.isclose(
                scalar, scalar_unclipped, rtol=1e-14, atol=0.0
            ),
            "scalar_bounds": (self.scalar_min, self.scalar_max),
        }
        return _antihermitian(result), diagnostics

    def initial_inverse_action(self, gradient, preconditioned, vector):
        """Apply the current PT-seeded ``H0`` to a fixed-frame vector.

        ``gradient``, ``preconditioned`` (the uphill ``z``), and ``vector``
        must all be anti-Hermitian matrices in the fixed reference frame.
        This public action is useful for verifying the SPD/secant properties
        without constructing a dense orbital Hessian.
        """
        gradient = self._validate_tangent(gradient, "reference gradient")
        preconditioned = self._validate_tangent(
            preconditioned, "reference PT-preconditioned gradient"
        )
        vector = self._validate_tangent(vector, "reference action vector")
        result, _ = self._initial_inverse_action(
            gradient, preconditioned, vector
        )
        return result

    def _inverse_hessian_action(self, gradient, preconditioned, vector):
        q = np.array(vector, dtype=np.complex128, copy=True)
        reversed_alphas = []
        for pair in reversed(self._pairs):
            alpha = pair.inverse_curvature * _inner(pair.step, q)
            q -= alpha * pair.gradient_difference
            reversed_alphas.append(alpha)

        result, initial_diagnostics = self._initial_inverse_action(
            gradient, preconditioned, q
        )
        alphas = list(reversed(reversed_alphas))
        betas = []
        for pair, alpha in zip(self._pairs, alphas):
            beta = pair.inverse_curvature * _inner(
                pair.gradient_difference, result
            )
            result += pair.step * (alpha - beta)
            betas.append(beta)
        diagnostics = {
            "history_size": len(self._pairs),
            "two_loop_alphas": alphas,
            "two_loop_betas": betas,
            "initial_inverse": initial_diagnostics,
        }
        return _antihermitian(result), diagnostics

    def inverse_hessian_action(self, gradient, preconditioned, vector):
        """Apply the retained L-BFGS inverse Hessian in the reference frame."""
        gradient = self._validate_tangent(gradient, "reference gradient")
        preconditioned = self._validate_tangent(
            preconditioned, "reference PT-preconditioned gradient"
        )
        vector = self._validate_tangent(vector, "reference action vector")
        result, _ = self._inverse_hessian_action(
            gradient, preconditioned, vector
        )
        return result

    def _project(self, current_mo, generator, projector):
        generator = _antihermitian(generator)
        information = None
        if projector is not None:
            projected = projector(current_mo, generator)
            if isinstance(projected, tuple):
                generator, information = projected
            else:
                generator = projected
            generator = self._validate_tangent(
                generator, "projected orbital generator"
            )
        return generator, information

    def _descent(self, gradient, direction):
        slope = _inner(gradient, direction)
        scale = float(np.linalg.norm(gradient) * np.linalg.norm(direction))
        if not np.isfinite(slope) or not np.isfinite(scale) or scale == 0.0:
            return False, slope, None
        cosine = slope / scale
        return cosine < -self.descent_tolerance, slope, cosine

    def propose(
        self,
        current_mo,
        gradient,
        preconditioned_direction,
        *,
        projector=None,
    ):
        """Create a pending fixed-reference L-BFGS search direction.

        The returned generator is in the current orbital frame and may be
        handed directly to a geodesic line search.  The previous proposal
        must first be resolved with :meth:`accept_secant` or :meth:`reject`.
        """
        if self._pending is not None:
            raise RuntimeError(
                "resolve the pending L-BFGS proposal with accept_secant() "
                "or reject() before proposing another direction"
            )
        current_mo = self._validate_mo(current_mo, "current orbitals")
        gradient = self._validate_tangent(gradient, "orbital gradient")
        raw_direction = self._validate_tangent(
            preconditioned_direction, "PT-preconditioned direction"
        )
        gradient_reference = self._to_reference(current_mo, gradient)
        preconditioned_reference = self._to_reference(
            current_mo, -raw_direction
        )

        inverse_gradient, inverse_diagnostics = self._inverse_hessian_action(
            gradient_reference,
            preconditioned_reference,
            gradient_reference,
        )
        direction_reference = -inverse_gradient
        direction = self._from_reference(current_mo, direction_reference)
        direction, projection = self._project(
            current_mo, direction, projector
        )
        is_descent, slope, cosine = self._descent(gradient, direction)

        history_before = len(self._pairs)
        fallback = None
        restart_reason = None
        fallback_projection = None
        if not is_descent:
            self._pairs.clear()
            self._negative_curvature_streak = 0
            restart_reason = "non-descent-L-BFGS-direction"
            raw_direction, fallback_projection = self._project(
                current_mo, raw_direction, projector
            )
            raw_descent, raw_slope, raw_cosine = self._descent(
                gradient, raw_direction
            )
            if raw_descent:
                direction = raw_direction
                slope = raw_slope
                cosine = raw_cosine
                fallback = "plain-PT"
            else:
                steepest, fallback_projection = self._project(
                    current_mo, -gradient, projector
                )
                steepest_descent, steepest_slope, steepest_cosine = (
                    self._descent(gradient, steepest)
                )
                if steepest_descent:
                    direction = steepest
                    slope = steepest_slope
                    cosine = steepest_cosine
                    fallback = "negative-gradient"
                else:
                    direction = np.zeros_like(gradient)
                    slope = 0.0
                    cosine = None
                    fallback = "zero-direction"

        direction = _antihermitian(direction)
        result_mo = current_mo.dot(scipy.linalg.expm(direction))
        identity = np.eye(result_mo.shape[1])
        orthonormality_error = float(
            np.max(abs(_metric(result_mo, self.overlap) - identity))
        )
        direction_reference = self._to_reference(current_mo, direction)
        self._pending = {
            "base_mo": np.array(current_mo, copy=True),
            "gradient_reference": np.array(
                gradient_reference, copy=True
            ),
            "direction_reference": np.array(
                direction_reference, copy=True
            ),
        }

        diagnostics = {
            "enabled": True,
            "method": "fixed-reference-PT-seeded-L-BFGS",
            "coordinate_system": "fixed-reference-adjoint-transport",
            "memory": self.memory,
            "history_size_before": history_before,
            "history_size": len(self._pairs),
            "history_used": bool(history_before > 0 and fallback is None),
            "initial_inverse": inverse_diagnostics["initial_inverse"],
            "two_loop_alphas": inverse_diagnostics["two_loop_alphas"],
            "two_loop_betas": inverse_diagnostics["two_loop_betas"],
            "raw_PT_norm": float(np.linalg.norm(raw_direction)),
            "direction_norm": float(np.linalg.norm(direction)),
            "slope": slope,
            "packed_slope": 0.5 * slope,
            "descent_cosine": cosine,
            "descent_threshold": -self.descent_tolerance,
            "projection": {
                "L-BFGS": projection,
                "fallback": fallback_projection,
            },
            "fallback": fallback,
            "restart": restart_reason is not None,
            "restart_reason": restart_reason,
            "memory_cleared": bool(
                restart_reason is not None and history_before > 0
            ),
            "orthonormality_error": orthonormality_error,
            "pending": True,
        }
        return OrbitalLBFGSProposal(result_mo, direction, diagnostics)

    def accept_secant(
        self,
        step_reference,
        gradient_difference_reference,
        *,
        curvature=None,
        strong_wolfe=None,
    ):
        """Commit one accepted line-search secant to L-BFGS history.

        The two vectors must already be expressed in this optimizer's fixed
        reference frame.  The complete mapping returned by
        ``FixedReferenceStrongWolfeLineSearch.accepted_secant`` can be passed
        with ``accept_secant(**mapping)``; ``curvature`` and ``strong_wolfe``
        are retained only as consistency/provenance diagnostics.  A tiny or
        mildly negative curvature is minimally damped to the stated floor;
        significantly negative curvature is not stored.
        """
        if self._pending is None:
            raise RuntimeError("no L-BFGS proposal is pending")
        step = self._validate_tangent(
            step_reference, "accepted reference step"
        )
        gradient_difference = self._validate_tangent(
            gradient_difference_reference,
            "accepted reference gradient difference",
        )
        if curvature is not None:
            curvature = float(curvature)
            if not np.isfinite(curvature):
                raise ValueError("line-search curvature must be finite")
        if strong_wolfe is not None:
            strong_wolfe = bool(strong_wolfe)
        ss = _inner(step, step)
        yy = _inner(gradient_difference, gradient_difference)
        sy = _inner(step, gradient_difference)
        pending_direction = self._pending["direction_reference"]
        direction_norm_squared = _inner(
            pending_direction, pending_direction
        )
        line_search_alpha = (
            _inner(pending_direction, step) / direction_norm_squared
            if direction_norm_squared > np.finfo(float).tiny
            else None
        )
        if line_search_alpha is None or not np.isfinite(line_search_alpha):
            direction_relative_residual = np.inf
        else:
            direction_relative_residual = float(
                np.linalg.norm(
                    step - line_search_alpha * pending_direction
                )
                / max(np.linalg.norm(step), np.finfo(float).tiny)
            )
        if (
            line_search_alpha is None
            or line_search_alpha <= 0.0
            or direction_relative_residual
            > self.secant_direction_tolerance
        ):
            raise ValueError(
                "accepted secant step is not a positive scalar multiple "
                "of the pending L-BFGS direction"
            )
        if curvature is not None and not np.isclose(
            curvature,
            sy,
            rtol=1e-9,
            atol=1e-12,
        ):
            raise ValueError(
                "line-search curvature disagrees with its fixed-frame secant"
            )
        curvature_floor = max(
            1e-12,
            1e-6 * np.sqrt(max(0.0, ss * yy)),
            ss / 1e3,
        )
        history_before = len(self._pairs)
        damping = 0.0
        stored_curvature = None
        action = "accepted"
        reset_after_negative = False
        negative_streak_before_reset = None

        if ss <= np.finfo(float).tiny or yy <= np.finfo(float).tiny:
            action = "skipped-degenerate"
            self._negative_curvature_streak = 0
        elif sy < -curvature_floor:
            action = "skipped-negative-curvature"
            self._negative_curvature_streak += 1
            if self._negative_curvature_streak >= 2:
                negative_streak_before_reset = self._negative_curvature_streak
                self._pairs.clear()
                self._negative_curvature_streak = 0
                action = "reset-after-negative-curvature"
                reset_after_negative = True
        else:
            self._negative_curvature_streak = 0
            if sy < curvature_floor:
                damping = (curvature_floor - sy) / ss
                gradient_difference = _antihermitian(
                    gradient_difference + damping * step
                )
                action = "accepted-damped"
            stored_curvature = _inner(step, gradient_difference)
            # Roundoff after the minimal shift must not undershoot the floor.
            if stored_curvature < curvature_floor:
                correction = (curvature_floor - stored_curvature) / ss
                damping += correction
                gradient_difference = _antihermitian(
                    gradient_difference + correction * step
                )
                stored_curvature = _inner(step, gradient_difference)
                action = "accepted-damped"
            pair = _SecantPair(
                step=np.array(step, copy=True),
                gradient_difference=np.array(
                    gradient_difference, copy=True
                ),
                inverse_curvature=1.0 / stored_curvature,
            )
            self._pairs.append(pair)
            if len(self._pairs) > self.memory:
                self._pairs = self._pairs[-self.memory :]

        self._pending = None
        return {
            "accepted": action.startswith("accepted"),
            "action": action,
            "raw_curvature": sy,
            "stored_curvature": stored_curvature,
            "curvature_floor": float(curvature_floor),
            "damping": float(damping),
            "step_norm": float(np.sqrt(max(0.0, ss))),
            "gradient_difference_norm": float(np.sqrt(max(0.0, yy))),
            "history_size_before": history_before,
            "history_size": len(self._pairs),
            "memory": self.memory,
            "negative_curvature_streak": self._negative_curvature_streak,
            "negative_streak_before_reset": negative_streak_before_reset,
            "memory_reset": reset_after_negative,
            "line_search_curvature": curvature,
            "line_search_alpha": float(line_search_alpha),
            "direction_relative_residual": direction_relative_residual,
            "direction_tolerance": self.secant_direction_tolerance,
            "strong_wolfe": strong_wolfe,
            "pending": False,
        }

    def reject(self):
        """Discard a rejected line-search proposal without changing pairs."""
        if self._pending is None:
            raise RuntimeError("no L-BFGS proposal is pending")
        self._pending = None
        return {
            "accepted": False,
            "history_size": len(self._pairs),
            "negative_curvature_streak": self._negative_curvature_streak,
            "pending": False,
        }

    def reset(self):
        """Clear pending state, accepted pairs, and curvature streak."""
        history_before = len(self._pairs)
        had_pending = self._pending is not None
        self._pairs.clear()
        self._pending = None
        self._negative_curvature_streak = 0
        return {
            "history_size_before": history_before,
            "history_size": 0,
            "discarded_pending": had_pending,
        }


__all__ = ["OrbitalLBFGSProposal", "PTSeededOrbitalLBFGS"]
