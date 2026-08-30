"""Transactional strong-Wolfe searches on the unitary orbital manifold.

The orbital coefficient matrix is kept orthonormal by following one fixed
geodesic

``C(alpha) = C(base) exp(alpha K)``

with an anti-Hermitian generator ``K``.  Every trial is reconstructed from
the accepted base point.  A rejected trial is therefore never inverted or
used as the origin of a later trial, and its gradient can never enter the
accepted secant returned by this module.

The search uses a fixed reference orbital frame to transport gradients and
directions.  It combines the Armijo and strong-Wolfe tests with a finite
trust/step boundary.  A still strongly negative endpoint derivative expands
through a safeguarded derivative secant; an Armijo failure or a derivative
sign change brackets the line minimum and triggers safeguarded interpolation.
When an external CI/DMRG evaluation budget is nearly exhausted, the best
Armijo point can be replayed so that the caller's final CI/RDM state is
consistent with the orbitals it returns.
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
class LineSearchTrial:
    """One orbital point which requires an energy/gradient evaluation."""

    alpha: float
    mo_coeff: np.ndarray
    step_norm: float
    on_boundary: bool
    evaluation_index: int
    purpose: str


@dataclass(frozen=True)
class LineSearchDecision:
    """Atomic result of consuming one evaluated trial point.

    ``action`` is one of ``"accept"``, ``"continue"``,
    ``"reevaluate-best"``, ``"restore-base"``, or
    ``"budget-exhausted"``.  A non-``None`` ``trial`` is the only orbital
    point the caller should evaluate next.
    """

    action: str
    accepted: bool
    reason: str
    trial: LineSearchTrial | None
    diagnostics: dict


@dataclass(frozen=True)
class AcceptedLineSearchPoint:
    """The accepted endpoint and its fixed-reference gradient secant."""

    alpha: float
    mo_coeff: np.ndarray
    energy: float
    gradient: np.ndarray
    directional_derivative: float
    step_reference: np.ndarray
    gradient_difference_reference: np.ndarray
    diagnostics: dict


@dataclass
class _EvaluatedPoint:
    trial: LineSearchTrial
    energy: float
    gradient: np.ndarray
    gradient_reference: np.ndarray
    directional_derivative: float
    armijo: bool
    strong_wolfe: bool


@dataclass(frozen=True)
class _BracketPoint:
    alpha: float
    energy: float
    directional_derivative: float


class FixedReferenceStrongWolfeLineSearch:
    """A reusable fixed-reference Riemannian line-search state machine.

    Parameters
    ----------
    reference_mo
        Orthonormal AO-to-MO coefficients defining a common tangent frame.
        A search base may differ from this matrix.
    overlap
        AO overlap matrix.
    c1, c2
        Armijo and strong-Wolfe constants, with ``0 < c1 < c2 < 1``.
    energy_tolerance
        Additive allowance in the Armijo energy comparison.  Production
        DMRG callers can set this to their trusted energy noise floor.
    interpolation_guard
        Fraction excluded at both ends of a bracket interpolation.
    expansion_min, expansion_max
        Multiplicative safeguards around a derivative-secant expansion.
    boundary_tolerance
        Relative tolerance for recognizing the finite trust/step boundary.
    boundary_acceptance_ratio
        Minimum actual/full-linear reduction ratio for accepting an Armijo
        point at the finite boundary.  This variational-quality safeguard is
        also applied when the endpoint happens to satisfy strong Wolfe.
    point_tolerance
        Maximum unitary mismatch between an expected and supplied trial.

    Notes
    -----
    ``evaluate(..., evaluations_remaining=n)`` interprets ``n`` as the
    number of CI/DMRG evaluations available *after* the supplied trial.  A
    normal continuation is issued only when at least two evaluations remain:
    one for the continuation and one to replay the best Armijo/base point if
    that continuation fails.  With a smaller budget, the current best point
    is accepted or replayed immediately.
    """

    def __init__(
        self,
        reference_mo,
        overlap,
        *,
        c1=1e-4,
        c2=0.5,
        energy_tolerance=0.0,
        interpolation_guard=0.1,
        expansion_min=1.5,
        expansion_max=10.0,
        boundary_tolerance=1e-10,
        boundary_acceptance_ratio=0.1,
        point_tolerance=1e-7,
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
            raise ValueError("line-search reference orbitals are not orthonormal")

        self.c1 = float(c1)
        self.c2 = float(c2)
        self.energy_tolerance = float(energy_tolerance)
        self.interpolation_guard = float(interpolation_guard)
        self.expansion_min = float(expansion_min)
        self.expansion_max = float(expansion_max)
        self.boundary_tolerance = float(boundary_tolerance)
        self.boundary_acceptance_ratio = float(
            boundary_acceptance_ratio
        )
        self.point_tolerance = float(point_tolerance)
        if not 0.0 < self.c1 < self.c2 < 1.0:
            raise ValueError("Wolfe constants must satisfy 0 < c1 < c2 < 1")
        if self.energy_tolerance < 0.0:
            raise ValueError("energy tolerance must be nonnegative")
        if not 0.0 < self.interpolation_guard < 0.5:
            raise ValueError("interpolation guard must lie between zero and one half")
        if not 1.0 < self.expansion_min <= self.expansion_max:
            raise ValueError("invalid line-search expansion safeguards")
        if self.boundary_tolerance <= 0.0 or self.point_tolerance <= 0.0:
            raise ValueError("line-search tolerances must be positive")
        if not 0.0 <= self.boundary_acceptance_ratio < 1.0:
            raise ValueError(
                "boundary acceptance ratio must lie in [0, 1)"
            )

        self._clear_search()

    def _clear_search(self):
        self._base_mo = None
        self._base_energy = None
        self._base_gradient = None
        self._base_gradient_reference = None
        self._direction = None
        self._direction_reference = None
        self._direction_norm = None
        self._dphi0 = None
        self._alpha_max = None
        self._pending = None
        self._evaluated = []
        self._best_armijo = None
        self._bracket_low = None
        self._bracket_high = None
        self._accepted = None
        self._active = False
        self._budget_exhausted = False

    @property
    def active(self):
        return bool(self._active)

    @property
    def has_pending(self):
        return self._pending is not None

    @property
    def pending(self):
        return self._pending

    @property
    def phi0(self):
        return self._base_energy

    @property
    def dphi0(self):
        return self._dphi0

    @property
    def accepted(self):
        return self._accepted

    @property
    def accepted_secant(self):
        """Return copies of the accepted fixed-frame secant, if available."""
        if self._accepted is None:
            return None
        return {
            "step_reference": np.array(
                self._accepted.step_reference, copy=True
            ),
            "gradient_difference_reference": np.array(
                self._accepted.gradient_difference_reference, copy=True
            ),
            "curvature": float(
                _inner(
                    self._accepted.step_reference,
                    self._accepted.gradient_difference_reference,
                )
            ),
            "strong_wolfe": bool(
                self._accepted.diagnostics["strong_wolfe"]
            ),
        }

    @property
    def history(self):
        return [self._point_diagnostics(point) for point in self._evaluated]

    def reset(self):
        """Discard the current search while retaining reference/controls."""
        self._clear_search()

    def _validate_mo(self, mo_coeff, label):
        mo_coeff = np.asarray(mo_coeff, dtype=np.complex128)
        if mo_coeff.shape != self.reference_mo.shape:
            raise ValueError(f"{label} and reference orbital dimensions disagree")
        if not np.all(np.isfinite(mo_coeff)):
            raise ValueError(f"{label} contains nonfinite values")
        identity = np.eye(mo_coeff.shape[1])
        error = float(np.max(abs(_metric(mo_coeff, self.overlap) - identity)))
        if error > 1e-7:
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

    def _at_boundary(self, alpha):
        scale = max(1.0, self._alpha_max)
        return bool(
            self._alpha_max - float(alpha)
            <= self.boundary_tolerance * scale
        )

    def _make_trial(self, alpha, purpose):
        alpha = float(alpha)
        if not 0.0 <= alpha <= self._alpha_max * (
            1.0 + self.boundary_tolerance
        ):
            raise ValueError("line-search alpha lies outside its boundary")
        alpha = min(alpha, self._alpha_max)
        mo_coeff = self._base_mo.dot(
            scipy.linalg.expm(alpha * self._direction)
        )
        trial = LineSearchTrial(
            alpha=alpha,
            mo_coeff=mo_coeff,
            step_norm=float(alpha * self._direction_norm),
            on_boundary=self._at_boundary(alpha),
            evaluation_index=len(self._evaluated),
            purpose=str(purpose),
        )
        self._pending = trial
        return trial

    def begin(
        self,
        base_mo,
        direction,
        energy,
        gradient,
        *,
        alpha=1.0,
        trust_radius=None,
        max_stepsize=None,
    ):
        """Start a search and return its first unevaluated orbital trial."""
        if self._active:
            raise RuntimeError("reset the active line search before beginning another")
        base_mo = self._validate_mo(base_mo, "line-search base orbitals")
        direction = self._validate_tangent(direction, "line-search direction")
        gradient = self._validate_tangent(gradient, "base orbital gradient")
        energy = float(energy)
        alpha = float(alpha)
        if not np.isfinite(energy):
            raise ValueError("base energy must be finite")
        if not np.isfinite(alpha) or alpha <= 0.0:
            raise ValueError("initial line-search alpha must be positive")

        direction_norm = float(np.linalg.norm(direction))
        if direction_norm == 0.0:
            raise ValueError("line-search direction must be nonzero")
        dphi0 = _inner(gradient, direction)
        if not np.isfinite(dphi0) or dphi0 >= 0.0:
            raise ValueError("line-search direction must be a strict descent direction")

        bounds = []
        for label, value in (
            ("trust radius", trust_radius),
            ("maximum step", max_stepsize),
        ):
            if value is None:
                continue
            value = float(value)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{label} must be positive and finite")
            bounds.append(value / direction_norm)
        if not bounds:
            raise ValueError("a finite trust_radius or max_stepsize is required")
        alpha_max = min(bounds)

        # A completed instance is reusable.  Clear its previous accepted
        # endpoint only after all new inputs have passed validation.
        self._clear_search()
        self._base_mo = np.array(base_mo, copy=True)
        self._base_energy = energy
        self._base_gradient = np.array(gradient, copy=True)
        self._base_gradient_reference = self._to_reference(base_mo, gradient)
        self._direction = direction
        self._direction_reference = self._to_reference(base_mo, direction)
        self._direction_norm = direction_norm
        self._dphi0 = dphi0
        self._alpha_max = alpha_max
        self._bracket_low = _BracketPoint(0.0, energy, dphi0)
        self._active = True
        return self._make_trial(min(alpha, alpha_max), "initial")

    def _trial_point_error(self, trial_mo):
        relative = (
            self._pending.mo_coeff.T.conj()
            .dot(self.overlap)
            .dot(trial_mo)
        )
        return float(np.max(abs(relative - np.eye(relative.shape[0]))))

    def _point_diagnostics(self, point):
        prediction = float(point.trial.alpha * self._dphi0)
        linear_ratio = (
            float((point.energy - self._base_energy) / prediction)
            if prediction < 0.0
            else None
        )
        return {
            "alpha": float(point.trial.alpha),
            "energy": float(point.energy),
            "directional_derivative": float(point.directional_derivative),
            "step_norm": float(point.trial.step_norm),
            "on_boundary": bool(point.trial.on_boundary),
            "purpose": point.trial.purpose,
            "armijo": bool(point.armijo),
            "strong_wolfe": bool(point.strong_wolfe),
            "linear_prediction": prediction,
            "linear_ratio": linear_ratio,
        }

    def _commit(self, point, reason):
        step_reference = point.trial.alpha * self._direction_reference
        gradient_difference = _antihermitian(
            point.gradient_reference - self._base_gradient_reference
        )
        diagnostics = {
            **self._point_diagnostics(point),
            "reason": str(reason),
            "phi0": float(self._base_energy),
            "dphi0": float(self._dphi0),
            "evaluations": len(self._evaluated),
            "alpha_max": float(self._alpha_max),
            "accepted_secant_curvature": _inner(
                step_reference, gradient_difference
            ),
        }
        self._accepted = AcceptedLineSearchPoint(
            alpha=float(point.trial.alpha),
            mo_coeff=np.array(point.trial.mo_coeff, copy=True),
            energy=float(point.energy),
            gradient=np.array(point.gradient, copy=True),
            directional_derivative=float(point.directional_derivative),
            step_reference=np.array(step_reference, copy=True),
            gradient_difference_reference=np.array(
                gradient_difference, copy=True
            ),
            diagnostics=diagnostics,
        )
        self._pending = None
        self._active = False
        return LineSearchDecision(
            action="accept",
            accepted=True,
            reason=str(reason),
            trial=None,
            diagnostics=diagnostics,
        )

    def _set_best_armijo(self, point):
        if not point.armijo:
            return
        diagnostics = self._point_diagnostics(point)
        if (
            point.trial.on_boundary
            and diagnostics["linear_ratio"] is not None
            and diagnostics["linear_ratio"]
            < self.boundary_acceptance_ratio
        ):
            return
        if self._best_armijo is None or point.energy < (
            self._best_armijo.energy - self.energy_tolerance
        ):
            self._best_armijo = point
        elif (
            abs(point.energy - self._best_armijo.energy)
            <= self.energy_tolerance
            and point.trial.alpha > self._best_armijo.trial.alpha
        ):
            self._best_armijo = point

    def _zoom_alpha(self):
        low = self._bracket_low
        high = self._bracket_high
        if low is None or high is None:
            raise RuntimeError("line-search zoom requires a complete bracket")
        left, right = sorted((low.alpha, high.alpha))
        width = right - left
        if width <= self.boundary_tolerance * max(1.0, right):
            return 0.5 * (left + right)
        lower = left + self.interpolation_guard * width
        upper = right - self.interpolation_guard * width

        derivative_denominator = (
            high.directional_derivative - low.directional_derivative
        )
        candidate = None
        if abs(derivative_denominator) > np.finfo(float).eps:
            secant = low.alpha - low.directional_derivative * (
                high.alpha - low.alpha
            ) / derivative_denominator
            if np.isfinite(secant):
                candidate = secant

        displacement = high.alpha - low.alpha
        quadratic_denominator = 2.0 * (
            high.energy
            - low.energy
            - low.directional_derivative * displacement
        )
        if candidate is None and abs(quadratic_denominator) > np.finfo(float).eps:
            quadratic = low.alpha - (
                low.directional_derivative * displacement**2
                / quadratic_denominator
            )
            if np.isfinite(quadratic):
                candidate = quadratic
        if candidate is None:
            candidate = 0.5 * (left + right)
        return float(np.clip(candidate, lower, upper))

    def _expansion_alpha(self, point):
        alpha = point.trial.alpha
        denominator = point.directional_derivative - self._dphi0
        candidate = None
        if denominator > np.finfo(float).eps:
            secant = -self._dphi0 * alpha / denominator
            if np.isfinite(secant) and secant > alpha:
                candidate = secant
        if candidate is None:
            candidate = 2.0 * alpha
        lower = min(self.expansion_min * alpha, self._alpha_max)
        upper = min(self.expansion_max * alpha, self._alpha_max)
        if upper < lower:
            lower = upper
        return float(np.clip(candidate, lower, upper))

    def _fallback_for_budget(self, current, evaluations_remaining):
        if current.armijo and self._best_armijo is current:
            return self._commit(current, "budget-best-armijo")

        target = self._best_armijo
        if target is not None and evaluations_remaining >= 1:
            trial = self._make_trial(
                target.trial.alpha, "best-armijo-fallback"
            )
            return LineSearchDecision(
                action="reevaluate-best",
                accepted=False,
                reason="budget-best-armijo-reevaluation",
                trial=trial,
                diagnostics={
                    "best_alpha": float(target.trial.alpha),
                    "best_energy": float(target.energy),
                    "evaluations_remaining": int(evaluations_remaining),
                },
            )
        if target is None and evaluations_remaining >= 1:
            trial = self._make_trial(0.0, "base-fallback")
            return LineSearchDecision(
                action="restore-base",
                accepted=False,
                reason="budget-base-reevaluation",
                trial=trial,
                diagnostics={
                    "best_alpha": 0.0,
                    "best_energy": float(self._base_energy),
                    "evaluations_remaining": int(evaluations_remaining),
                },
            )

        self._pending = None
        self._budget_exhausted = True
        return LineSearchDecision(
            action="budget-exhausted",
            accepted=False,
            reason="no-evaluation-remains-to-restore-best-point",
            trial=None,
            diagnostics={
                "best_alpha": (
                    None if target is None else float(target.trial.alpha)
                ),
                "best_energy": (
                    float(self._base_energy)
                    if target is None
                    else float(target.energy)
                ),
                "requires_reevaluation": True,
                "evaluations_remaining": int(evaluations_remaining),
            },
        )

    def evaluate(
        self,
        trial_mo,
        energy,
        gradient,
        *,
        evaluations_remaining=None,
    ):
        """Consume one trial and atomically accept it or propose the next.

        The endpoint derivative is evaluated with the original search
        direction transported from the fixed base frame.  A caller must
        evaluate exactly the ``LineSearchTrial`` most recently returned by
        :meth:`begin` or this method.
        """
        if not self._active or self._pending is None:
            raise RuntimeError("no line-search trial is pending")
        trial_mo = self._validate_mo(trial_mo, "evaluated trial orbitals")
        mismatch = self._trial_point_error(trial_mo)
        if mismatch > self.point_tolerance:
            raise ValueError(
                "evaluated orbitals do not match the pending line-search trial"
            )
        energy = float(energy)
        if not np.isfinite(energy):
            raise ValueError("trial energy must be finite")
        gradient = self._validate_tangent(gradient, "trial orbital gradient")
        if evaluations_remaining is not None:
            evaluations_remaining = int(evaluations_remaining)
            if evaluations_remaining < 0:
                raise ValueError("remaining evaluation budget must be nonnegative")

        trial = self._pending
        self._pending = None
        gradient_reference = self._to_reference(trial_mo, gradient)
        transported_direction = self._from_reference(
            trial_mo, self._direction_reference
        )
        dphi = _inner(gradient, transported_direction)
        armijo_limit = (
            self._base_energy
            + self.c1 * trial.alpha * self._dphi0
            + self.energy_tolerance
        )
        armijo = bool(energy <= armijo_limit)
        strong_wolfe = bool(
            armijo and abs(dphi) <= self.c2 * abs(self._dphi0)
        )
        point = _EvaluatedPoint(
            trial=trial,
            energy=energy,
            gradient=np.array(gradient, copy=True),
            gradient_reference=gradient_reference,
            directional_derivative=dphi,
            armijo=armijo,
            strong_wolfe=strong_wolfe,
        )
        self._evaluated.append(point)
        self._set_best_armijo(point)

        if trial.purpose in (
            "best-armijo-fallback",
            "base-fallback",
        ):
            if trial.purpose == "best-armijo-fallback" and not armijo:
                # A noisy/non-deterministic reevaluation must satisfy the
                # energy safeguard again.  Restore the accepted base if one
                # more evaluation exists; otherwise report that restoration
                # authority/budget is required instead of silently accepting
                # an uphill point.
                if evaluations_remaining is not None and evaluations_remaining >= 1:
                    base_trial = self._make_trial(0.0, "base-fallback")
                    return LineSearchDecision(
                        action="restore-base",
                        accepted=False,
                        reason="best-armijo-reevaluation-failed",
                        trial=base_trial,
                        diagnostics={
                            **self._point_diagnostics(point),
                            "evaluations_remaining": int(evaluations_remaining),
                        },
                    )
                self._budget_exhausted = True
                return LineSearchDecision(
                    action="budget-exhausted",
                    accepted=False,
                    reason="reevaluated-best-point-failed-armijo",
                    trial=None,
                    diagnostics={
                        **self._point_diagnostics(point),
                        "requires_reevaluation": True,
                        "best_alpha": 0.0,
                        "best_energy": float(self._base_energy),
                    },
                )
            reason = (
                "budget-best-armijo"
                if trial.purpose == "best-armijo-fallback"
                else "budget-restored-base"
            )
            return self._commit(point, reason)

        prediction = float(trial.alpha * self._dphi0)
        linear_ratio = (
            float((energy - self._base_energy) / prediction)
            if prediction < 0.0
            else None
        )
        boundary_ratio_acceptable = bool(
            not trial.on_boundary
            or (
                linear_ratio is not None
                and linear_ratio >= self.boundary_acceptance_ratio
            )
        )
        if strong_wolfe and boundary_ratio_acceptable:
            return self._commit(point, "strong-wolfe")
        boundary_acceptable = bool(
            armijo
            and trial.on_boundary
            and linear_ratio is not None
            and linear_ratio >= self.boundary_acceptance_ratio
        )
        if boundary_acceptable:
            return self._commit(point, "boundary-armijo")

        # A normal continuation consumes one evaluation.  Keep one additional
        # evaluation in reserve to replay the best Armijo/base point if that
        # continuation is rejected by the variational energy.
        if evaluations_remaining is not None and evaluations_remaining <= 1:
            return self._fallback_for_budget(point, evaluations_remaining)

        base_bracket = _BracketPoint(
            0.0, self._base_energy, self._dphi0
        )
        current_bracket = _BracketPoint(
            trial.alpha, energy, dphi
        )
        poor_boundary = bool(
            armijo and trial.on_boundary and not boundary_acceptable
        )
        if armijo and dphi < -self.c2 * abs(self._dphi0) and not poor_boundary:
            self._bracket_low = current_bracket
            if self._bracket_high is None:
                next_alpha = self._expansion_alpha(point)
                reason = "expand-negative-endpoint-derivative"
            else:
                next_alpha = self._zoom_alpha()
                reason = "zoom-negative-endpoint-derivative"
        else:
            # An Armijo failure or a sufficiently positive derivative gives
            # an upper bracket.  Retain the latest lower Armijo point, or the
            # accepted base if no nonzero Armijo point exists yet.
            self._bracket_high = current_bracket
            if self._bracket_low is None:
                self._bracket_low = base_bracket
            next_alpha = self._zoom_alpha()
            reason = (
                "zoom-armijo-failure"
                if not armijo
                else (
                    "zoom-poor-boundary-ratio"
                    if poor_boundary
                    else "zoom-derivative-sign-change"
                )
            )

        if abs(next_alpha - trial.alpha) <= self.boundary_tolerance * max(
            1.0, trial.alpha
        ):
            if self._best_armijo is not None:
                return self._fallback_for_budget(
                    point,
                    1 if evaluations_remaining is None else evaluations_remaining,
                )
            next_alpha = 0.5 * trial.alpha
            reason += "-bisection"

        next_trial = self._make_trial(next_alpha, reason)
        return LineSearchDecision(
            action="continue",
            accepted=False,
            reason=reason,
            trial=next_trial,
            diagnostics={
                **self._point_diagnostics(point),
                "phi0": float(self._base_energy),
                "dphi0": float(self._dphi0),
                "armijo_limit": float(armijo_limit),
                "next_alpha": float(next_alpha),
                "alpha_max": float(self._alpha_max),
                "evaluations": len(self._evaluated),
            },
        )


__all__ = [
    "AcceptedLineSearchPoint",
    "FixedReferenceStrongWolfeLineSearch",
    "LineSearchDecision",
    "LineSearchTrial",
]
