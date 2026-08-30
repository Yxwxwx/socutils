"""Safeguarded nonlinear conjugate gradients for unitary orbitals.

The Super-CIPT correction is a useful, inexpensive approximation to a
preconditioned orbital-gradient step.  Repeated plain corrections can,
however, approach a stationary point only linearly.  This module accelerates
those corrections without accumulating generators in inconsistent moving
orbital frames.

All tangent vectors are compared in one fixed reference frame.  A vector
``a`` in the local frame of ``C = C_ref U`` is represented as ``U a U^H``;
this is also an isometric vector transport between orbital points.  Accepted
steps provide a Barzilai--Borwein spectral scale, while a flexible hybrid
PRP+/FR recurrence supplies conjugacy.  Projection, descent, and trust-region
guards are applied before a trial orbital set is returned.

The state transition is deliberately transactional.  ``propose`` creates a
pending trial, ``accept`` approves it, and only the *next* ``propose`` (where
the new gradient is available) forms a secant.  ``reject`` discards the
pending trial and restarts conjugacy, so a rejected CI/RDM can never enter the
history.
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


@dataclass
class OrbitalCGResult:
    """One spectral-CG orbital proposal and its diagnostics."""

    mo_coeff: np.ndarray
    generator: np.ndarray
    diagnostics: dict


@dataclass
class _PendingProposal:
    """Data which become secant history only after explicit acceptance."""

    base_mo: np.ndarray
    candidate_mo: np.ndarray
    gradient_reference: np.ndarray
    raw_preconditioned_reference: np.ndarray
    preconditioned_reference: np.ndarray
    search_reference: np.ndarray
    step_reference: np.ndarray
    energy: float | None
    slope: float
    applied_norm: float
    trust_radius: float
    on_boundary: bool
    gamma: float
    accepted: bool = False
    accepted_energy: float | None = None
    accepted_linear_ratio: float | None = None


class SpectralOrbitalCG:
    """Fixed-reference Riemannian spectral flexible PRP+ optimizer.

    Parameters
    ----------
    reference_mo
        An orthonormal AO-to-MO coefficient matrix.  It defines only the
        common tangent frame; it is never changed by ``reset``.
    overlap
        AO overlap matrix.
    gamma_initial, gamma_min, gamma_max
        Initial and bounded Barzilai--Borwein spectral scales.  ``gamma``
        multiplies the PT-preconditioned gradient, not the bare gradient.
    curvature_tolerance
        Relative threshold for accepting the secant curvature ``<s,y>``.
    descent_tolerance
        Required negative cosine (up to this tolerance) for a search
        direction.  A failed conjugate direction falls back to the spectral
        PT direction, then to plain PT, and finally to steepest descent.
    ratio_shrink_threshold, ratio_expand_threshold
        Linear-model ratio thresholds for adapting the trust radius.
    trust_shrink, trust_expand, reject_shrink
        Multiplicative trust-radius updates.  Expansion is permitted only
        after an accepted boundary step.  The very first radius is always at
        most the norm of the raw PT step, even if ``max_stepsize`` is larger.
    trust_restart_scale
        If trust truncation retains less than this fraction of a conjugate
        search direction, conjugacy is restarted before applying the step.

    Notes
    -----
    ``preconditioned_direction`` passed to :meth:`propose` is the *descent*
    Super-CIPT generator ``kappa_PT``.  Internally ``z = -kappa_PT`` is the
    uphill preconditioned gradient and ``w = gamma*z``.  The recurrence is

    ``d_k = -w_k + beta_k T(d_{k-1})``.

    The full matrix Frobenius slope reported in diagnostics is twice the
    corresponding packed-independent-variable slope used by PySCF.
    """

    def __init__(
        self,
        reference_mo,
        overlap,
        *,
        gamma_initial=1.0,
        gamma_min=1e-3,
        gamma_max=50.0,
        curvature_tolerance=1e-8,
        descent_tolerance=1e-2,
        ratio_shrink_threshold=0.1,
        ratio_expand_threshold=0.5,
        unresolved_boundary_ratio=0.8,
        unresolved_boundary_alignment=0.95,
        trust_shrink=0.5,
        trust_expand=2.0,
        reject_shrink=0.5,
        trust_restart_scale=0.1,
        accepted_point_tolerance=1e-7,
    ):
        self.reference_mo = np.array(
            reference_mo, dtype=np.complex128, copy=True
        )
        self.overlap = np.asarray(overlap, dtype=np.complex128)
        if self.reference_mo.ndim != 2:
            raise ValueError("reference orbitals must be a matrix")
        if self.overlap.shape != (self.reference_mo.shape[0],) * 2:
            raise ValueError("AO overlap and orbital dimensions disagree")
        reference_metric = _metric(self.reference_mo, self.overlap)
        identity = np.eye(self.reference_mo.shape[1])
        if np.max(abs(reference_metric - identity)) > 1e-7:
            raise ValueError("CG reference orbitals are not orthonormal")

        self.gamma_initial = float(gamma_initial)
        self.gamma_min = float(gamma_min)
        self.gamma_max = float(gamma_max)
        if not 0.0 < self.gamma_min <= self.gamma_initial <= self.gamma_max:
            raise ValueError(
                "spectral gamma bounds must satisfy "
                "0 < gamma_min <= gamma_initial <= gamma_max"
            )
        self.curvature_tolerance = float(curvature_tolerance)
        self.descent_tolerance = float(descent_tolerance)
        self.ratio_shrink_threshold = float(ratio_shrink_threshold)
        self.ratio_expand_threshold = float(ratio_expand_threshold)
        self.unresolved_boundary_ratio = float(unresolved_boundary_ratio)
        self.unresolved_boundary_alignment = float(
            unresolved_boundary_alignment
        )
        self.trust_shrink = float(trust_shrink)
        self.trust_expand = float(trust_expand)
        self.reject_shrink = float(reject_shrink)
        self.trust_restart_scale = float(trust_restart_scale)
        self.accepted_point_tolerance = float(accepted_point_tolerance)

        if self.curvature_tolerance < 0.0:
            raise ValueError("curvature tolerance must be nonnegative")
        if self.descent_tolerance < 0.0:
            raise ValueError("descent tolerance must be nonnegative")
        if not (
            0.0 <= self.ratio_shrink_threshold
            < self.ratio_expand_threshold
        ):
            raise ValueError("invalid trust-ratio thresholds")
        if self.unresolved_boundary_ratio < 0.0:
            raise ValueError("unresolved-curvature ratio threshold must be nonnegative")
        if not -1.0 <= self.unresolved_boundary_alignment <= 1.0:
            raise ValueError(
                "unresolved-curvature alignment threshold must lie in [-1, 1]"
            )
        if not 0.0 < self.trust_shrink < 1.0:
            raise ValueError("trust shrink factor must lie between zero and one")
        if self.trust_expand <= 1.0:
            raise ValueError("trust expansion factor must exceed one")
        if not 0.0 < self.reject_shrink < 1.0:
            raise ValueError("reject shrink factor must lie between zero and one")
        if not 0.0 < self.trust_restart_scale < 1.0:
            raise ValueError("trust restart scale must lie between zero and one")
        if self.accepted_point_tolerance <= 0.0:
            raise ValueError("accepted-point tolerance must be positive")

        self._pending = None
        self._trust_radius = None
        self._restart_next = None

    @property
    def trust_radius(self):
        """Current internal trust radius, or ``None`` before the first step."""
        return self._trust_radius

    @property
    def has_pending(self):
        """Whether a trial proposal is awaiting a driver decision."""
        return self._pending is not None

    @property
    def pending_accepted(self):
        """Whether the pending proposal has been explicitly accepted."""
        return bool(self._pending is not None and self._pending.accepted)

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
        """Isometrically transport a tangent vector through the fixed frame."""
        from_mo = self._validate_mo(from_mo, "transport source orbitals")
        to_mo = self._validate_mo(to_mo, "transport target orbitals")
        tangent = self._validate_tangent(tangent, "transport tangent")
        reference_tangent = self._to_reference(from_mo, tangent)
        return self._from_reference(to_mo, reference_tangent)

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

    def _is_descent(self, gradient, direction):
        slope = _inner(gradient, direction)
        scale = float(np.linalg.norm(gradient) * np.linalg.norm(direction))
        if not np.isfinite(slope) or not np.isfinite(scale) or scale == 0.0:
            return False, slope, None
        cosine = slope / scale
        return cosine < -self.descent_tolerance, slope, cosine

    def _accepted_point_error(self, pending, current_mo):
        relative = _nearest_unitary(
            pending.candidate_mo.T.conj()
            .dot(self.overlap)
            .dot(current_mo)
        )
        return float(np.max(abs(relative - np.eye(relative.shape[0]))))

    def _linear_ratio(
        self,
        pending,
        *,
        energy,
        previous_linear_ratio,
    ):
        if previous_linear_ratio is not None:
            ratio = float(previous_linear_ratio)
            return ratio if np.isfinite(ratio) else None, "argument"
        if pending.accepted_linear_ratio is not None:
            ratio = float(pending.accepted_linear_ratio)
            return ratio if np.isfinite(ratio) else None, "accept"

        accepted_energy = pending.accepted_energy
        if accepted_energy is None and energy is not None:
            accepted_energy = float(energy)
        if (
            pending.energy is None
            or accepted_energy is None
            or not np.isfinite(pending.slope)
            or pending.slope >= 0.0
        ):
            return None, None
        ratio = (accepted_energy - pending.energy) / pending.slope
        return (float(ratio), "energy") if np.isfinite(ratio) else (None, None)

    def _update_trust(self, ratio, pending, maximum):
        before = self._trust_radius
        action = "unchanged"
        if before is None:
            return before, action
        if ratio is not None:
            if ratio < self.ratio_shrink_threshold:
                self._trust_radius *= self.trust_shrink
                action = "shrink-poor-ratio"
            elif (
                ratio > self.ratio_expand_threshold
                and pending.on_boundary
            ):
                self._trust_radius *= self.trust_expand
                action = "expand-good-boundary-step"
        if maximum is not None and self._trust_radius > maximum:
            self._trust_radius = maximum
            if action == "unchanged":
                action = "clamp-maximum"
            else:
                action += "-clamped"
        return before, action

    def accept(self, *, energy=None, linear_ratio=None):
        """Approve the pending trial without yet adding a gradient secant.

        The next :meth:`propose` supplies the accepted point's gradient and PT
        direction.  Keeping that data out of ``accept`` makes it impossible
        for a rejected CI/RDM to enter the CG history accidentally.
        """
        if self._pending is None:
            raise RuntimeError("no spectral-CG proposal is pending")
        if self._pending.accepted:
            raise RuntimeError("the pending spectral-CG proposal is already accepted")
        if energy is not None:
            energy = float(energy)
            if not np.isfinite(energy):
                raise ValueError("accepted energy must be finite")
        if linear_ratio is not None:
            linear_ratio = float(linear_ratio)
            if not np.isfinite(linear_ratio):
                raise ValueError("accepted linear ratio must be finite")
        self._pending.accepted = True
        self._pending.accepted_energy = energy
        self._pending.accepted_linear_ratio = linear_ratio
        return {
            "accepted": True,
            "energy": energy,
            "linear_ratio": linear_ratio,
            "applied_norm": self._pending.applied_norm,
            "trust_radius": self._pending.trust_radius,
        }

    def reject(self, *, shrink=None):
        """Discard a rejected trial, restart CG, and shrink the trust radius."""
        if self._pending is None:
            raise RuntimeError("no spectral-CG proposal is pending")
        factor = self.reject_shrink if shrink is None else float(shrink)
        if not 0.0 < factor < 1.0:
            raise ValueError("reject shrink factor must lie between zero and one")
        before = self._trust_radius
        rejected_norm = self._pending.applied_norm
        self._pending = None
        if self._trust_radius is not None:
            self._trust_radius *= factor
        self._restart_next = "rejected-trial"
        return {
            "accepted": False,
            "rejected_applied_norm": rejected_norm,
            "trust_radius_before": before,
            "trust_radius_after": self._trust_radius,
            "restart": True,
        }

    def reset(self, *, keep_trust=False):
        """Clear all pending/conjugate history, optionally retaining trust."""
        retained = self._trust_radius if keep_trust else None
        self._pending = None
        self._trust_radius = retained
        self._restart_next = "manual-reset"

    def propose(
        self,
        current_mo,
        gradient,
        preconditioned_direction,
        *,
        energy=None,
        previous_linear_ratio=None,
        projector=None,
        max_stepsize=None,
    ):
        """Build one safeguarded orbital proposal from an accepted point.

        Before a second or later call, the previous trial must have been
        resolved explicitly with :meth:`accept` or :meth:`reject`.
        ``max_stepsize`` is a Frobenius-norm cap.  It is never used to enlarge
        the first raw PT correction.
        """
        current_mo = self._validate_mo(current_mo, "current orbitals")
        gradient = self._validate_tangent(gradient, "orbital gradient")
        raw_direction = self._validate_tangent(
            preconditioned_direction, "PT-preconditioned direction"
        )
        if energy is not None:
            energy = float(energy)
            if not np.isfinite(energy):
                raise ValueError("current energy must be finite")
        if max_stepsize is not None:
            maximum = float(max_stepsize)
            if not np.isfinite(maximum) or maximum <= 0.0:
                raise ValueError("maximum orbital step must be positive")
        else:
            maximum = None

        history = self._pending
        if history is not None and not history.accepted:
            raise RuntimeError(
                "resolve the pending spectral-CG proposal with accept() or "
                "reject() before proposing another step"
            )
        self._pending = None

        restart_reasons = []
        if self._restart_next is not None:
            restart_reasons.append(self._restart_next)
            self._restart_next = None

        accepted_point_error = None
        ratio = None
        ratio_source = None
        trust_action = "unchanged"
        trust_before_update = self._trust_radius
        if history is not None:
            accepted_point_error = self._accepted_point_error(
                history, current_mo
            )
            if accepted_point_error > self.accepted_point_tolerance:
                restart_reasons.append("accepted-point-mismatch")
                history = None
            else:
                ratio, ratio_source = self._linear_ratio(
                    history,
                    energy=energy,
                    previous_linear_ratio=previous_linear_ratio,
                )
                trust_before_update, trust_action = self._update_trust(
                    ratio, history, maximum
                )

        raw_direction, raw_projection = self._project(
            current_mo, raw_direction, projector
        )
        raw_is_descent, raw_slope, raw_cosine = self._is_descent(
            gradient, raw_direction
        )
        fallback = None
        if not raw_is_descent:
            steepest, steepest_projection = self._project(
                current_mo, -gradient, projector
            )
            steepest_is_descent, _, _ = self._is_descent(
                gradient, steepest
            )
            if steepest_is_descent:
                raw_direction = steepest
                raw_projection = steepest_projection
                fallback = "negative-gradient"
            else:
                raw_direction = np.zeros_like(gradient)
                fallback = "zero-direction"
            restart_reasons.append("raw-PT-non-descent")
            history = None
            raw_is_descent, raw_slope, raw_cosine = self._is_descent(
                gradient, raw_direction
            )

        raw_norm = float(np.linalg.norm(raw_direction))
        if self._trust_radius is None:
            # A larger first step has already been observed to turn a valid PT
            # descent correction uphill for the Cl CAS(7,16) debug case.
            # Initialise from the evidence-bearing raw correction itself.
            self._trust_radius = raw_norm
            if maximum is not None:
                self._trust_radius = min(self._trust_radius, maximum)
            trust_action = "initialize-from-raw-PT"
            trust_before_update = None
        elif maximum is not None and self._trust_radius > maximum:
            self._trust_radius = maximum
            trust_action = "clamp-maximum"

        current_unitary = self._frame_unitary(current_mo)
        gradient_reference = _antihermitian(
            current_unitary.dot(gradient).dot(current_unitary.T.conj())
        )
        raw_reference = _antihermitian(
            current_unitary.dot(raw_direction).dot(current_unitary.T.conj())
        )
        z_reference = -raw_reference

        gamma = self.gamma_initial
        gamma_unclipped = None
        gamma_clipped = False
        secant_curvature = None
        secant_curvature_cosine = None
        secant_step_norm = None
        secant_gradient_norm = None
        secant_preconditioner_norm = None
        preconditioned_curvature = None
        preconditioned_curvature_cosine = None
        raw_preconditioner_alignment = None
        unresolved_curvature_boundary = False
        y_reference = None
        dz_reference = None
        stable_secant = history is not None
        if history is not None:
            s_reference = history.step_reference
            y_reference = _antihermitian(
                gradient_reference - history.gradient_reference
            )
            dz_reference = _antihermitian(
                z_reference - history.raw_preconditioned_reference
            )
            ss = _inner(s_reference, s_reference)
            yy = _inner(y_reference, y_reference)
            dzdz = _inner(dz_reference, dz_reference)
            sy = _inner(s_reference, y_reference)
            ydz = _inner(y_reference, dz_reference)
            secant_curvature = sy
            preconditioned_curvature = ydz
            secant_step_norm = float(np.sqrt(max(0.0, ss)))
            secant_gradient_norm = float(np.sqrt(max(0.0, yy)))
            secant_preconditioner_norm = float(np.sqrt(max(0.0, dzdz)))
            denominator = np.sqrt(max(0.0, ss * yy))
            preconditioned_denominator = np.sqrt(max(0.0, yy * dzdz))
            secant_curvature_cosine = (
                sy / denominator if denominator > 0.0 else None
            )
            preconditioned_curvature_cosine = (
                ydz / preconditioned_denominator
                if preconditioned_denominator > 0.0
                else None
            )
            previous_z_norm = float(
                np.linalg.norm(history.raw_preconditioned_reference)
            )
            current_z_norm = float(np.linalg.norm(z_reference))
            if previous_z_norm > 0.0 and current_z_norm > 0.0:
                raw_preconditioner_alignment = _inner(
                    z_reference,
                    history.raw_preconditioned_reference,
                ) / (current_z_norm * previous_z_norm)

            degenerate = (
                ss <= np.finfo(float).tiny
                or yy <= np.finfo(float).tiny
                or dzdz <= np.finfo(float).tiny
            )
            physical_curvature_valid = (
                not degenerate
                and sy > self.curvature_tolerance * denominator
            )
            preconditioned_curvature_valid = (
                not degenerate
                and ydz
                > self.curvature_tolerance * preconditioned_denominator
            )
            if physical_curvature_valid and preconditioned_curvature_valid:
                # z=P g already contains the approximate inverse Hessian.
                # The dimensionless scalar below corrects that preconditioner
                # through gamma*P*y ~= s; ordinary BB1 (s.s/s.y) would apply
                # a second inverse-Hessian scale to the PT step.
                gamma_unclipped = sy / ydz
                if not np.isfinite(gamma_unclipped) or gamma_unclipped <= 0.0:
                    stable_secant = False
                    restart_reasons.append("invalid-spectral-gamma")
                else:
                    gamma = float(
                        np.clip(
                            gamma_unclipped,
                            self.gamma_min,
                            self.gamma_max,
                        )
                    )
                    gamma_clipped = not np.isclose(
                        gamma, gamma_unclipped, rtol=1e-14, atol=0.0
                    )
            else:
                # A varying PT preconditioner can make <y,dz> nonpositive
                # even while its raw direction remains a reliable local
                # descent direction.  At the Cl CAS(7,16) plateau the actual
                # reduction is essentially the full linear prediction and
                # consecutive PT directions are parallel.  In that narrowly
                # diagnosed case, truncated-CG logic moves along the stable
                # direction to the *current* trust boundary.  The next
                # variational energy evaluation remains authoritative and an
                # uphill point is rejected by the driver.
                unresolved_curvature_boundary = bool(
                    ratio is not None
                    and ratio >= self.unresolved_boundary_ratio
                    and raw_preconditioner_alignment is not None
                    and raw_preconditioner_alignment
                    >= self.unresolved_boundary_alignment
                    and raw_norm > 0.0
                    and self._trust_radius is not None
                    and self._trust_radius > 0.0
                )
                stable_secant = False
                if unresolved_curvature_boundary:
                    gamma_unclipped = self._trust_radius / raw_norm
                    gamma = float(
                        np.clip(
                            gamma_unclipped,
                            self.gamma_min,
                            self.gamma_max,
                        )
                    )
                    gamma_clipped = not np.isclose(
                        gamma, gamma_unclipped, rtol=1e-14, atol=0.0
                    )
                    restart_reasons.append(
                        "unresolved-curvature-trust-boundary"
                    )
                elif degenerate:
                    restart_reasons.append("degenerate-secant")
                elif not physical_curvature_valid:
                    restart_reasons.append("nonpositive-curvature")
                else:
                    restart_reasons.append(
                        "nonpositive-preconditioned-curvature"
                    )

        w_reference = gamma * z_reference
        beta_prp = None
        beta_fr = None
        beta = 0.0
        beta_clipped = False
        if history is not None and stable_secant:
            previous_denominator = _inner(
                history.gradient_reference,
                history.preconditioned_reference,
            )
            current_numerator = _inner(gradient_reference, w_reference)
            if (
                not np.isfinite(previous_denominator)
                or previous_denominator <= np.finfo(float).eps
                or not np.isfinite(current_numerator)
                or current_numerator <= 0.0
            ):
                restart_reasons.append("invalid-flexible-CG-metric")
            else:
                beta_prp = _inner(
                    gradient_reference,
                    w_reference - history.preconditioned_reference,
                ) / previous_denominator
                beta_fr = current_numerator / previous_denominator
                if not np.isfinite(beta_prp) or not np.isfinite(beta_fr):
                    restart_reasons.append("nonfinite-CG-beta")
                else:
                    beta = max(0.0, min(beta_prp, beta_fr))
                    beta_clipped = not np.isclose(
                        beta, beta_prp, rtol=1e-14, atol=1e-16
                    )

        if restart_reasons:
            beta = 0.0
        search_reference = -w_reference
        if beta > 0.0 and history is not None:
            # The standard recurrence uses the untruncated search direction;
            # the actually applied step is reserved for the spectral secant.
            search_reference = (
                search_reference + beta * history.search_reference
            )
        search_direction = _antihermitian(
            current_unitary.T.conj()
            .dot(search_reference)
            .dot(current_unitary)
        )
        search_direction, search_projection = self._project(
            current_mo, search_direction, projector
        )

        search_is_descent, search_slope, search_cosine = self._is_descent(
            gradient, search_direction
        )
        if not search_is_descent:
            spectral_seed = gamma * raw_direction
            spectral_seed, search_projection = self._project(
                current_mo, spectral_seed, projector
            )
            seed_is_descent, seed_slope, seed_cosine = self._is_descent(
                gradient, spectral_seed
            )
            if seed_is_descent:
                search_direction = spectral_seed
                search_slope = seed_slope
                search_cosine = seed_cosine
                fallback = fallback or "spectral-PT"
            elif raw_is_descent:
                search_direction = raw_direction
                search_slope = raw_slope
                search_cosine = raw_cosine
                fallback = fallback or "plain-PT"
            else:
                search_direction = np.zeros_like(gradient)
                search_slope = 0.0
                search_cosine = None
                fallback = fallback or "zero-direction"
            beta = 0.0
            restart_reasons.append("non-descent-CG-direction")

        search_norm = float(np.linalg.norm(search_direction))
        trust_radius = float(self._trust_radius)
        trust_scale = 1.0
        if search_norm > trust_radius and search_norm > 0.0:
            trust_scale = trust_radius / search_norm

        if (
            beta > 0.0
            and trust_scale < self.trust_restart_scale
        ):
            search_direction = gamma * raw_direction
            search_direction, search_projection = self._project(
                current_mo, search_direction, projector
            )
            search_norm = float(np.linalg.norm(search_direction))
            trust_scale = (
                min(1.0, trust_radius / search_norm)
                if search_norm > 0.0
                else 1.0
            )
            beta = 0.0
            fallback = fallback or "spectral-PT-after-trust-truncation"
            restart_reasons.append("severe-trust-truncation")

        applied_direction = trust_scale * search_direction
        applied_direction = _antihermitian(applied_direction)
        applied_norm = float(np.linalg.norm(applied_direction))
        applied_is_descent, applied_slope, applied_cosine = self._is_descent(
            gradient, applied_direction
        )
        if applied_norm > 0.0 and not applied_is_descent:
            # Scaling cannot change an exact slope sign.  This is therefore a
            # final numerical-safety fallback rather than part of normal flow.
            applied_direction = np.zeros_like(gradient)
            applied_norm = 0.0
            applied_slope = 0.0
            applied_cosine = None
            fallback = "zero-direction-after-final-descent-check"
            beta = 0.0
            restart_reasons.append("final-non-descent-direction")

        result_mo = current_mo.dot(scipy.linalg.expm(applied_direction))
        identity = np.eye(result_mo.shape[1])
        orthonormality_error = float(
            np.max(abs(_metric(result_mo, self.overlap) - identity))
        )
        search_reference = self._to_reference(
            current_mo, search_direction
        )
        step_reference = self._to_reference(
            current_mo, applied_direction
        )
        # Store the preconditioned vector corresponding to the seed actually
        # used by the flexible recurrence.  It remains uphill, with positive
        # <g,w>, even when beta is later rejected by a descent guard.
        preconditioned_reference = w_reference
        if fallback == "negative-gradient":
            preconditioned_reference = gradient_reference.copy()

        on_boundary = bool(
            trust_radius > 0.0
            and applied_norm >= 0.95 * trust_radius
        )
        self._pending = _PendingProposal(
            base_mo=np.array(current_mo, copy=True),
            candidate_mo=np.array(result_mo, copy=True),
            gradient_reference=gradient_reference,
            raw_preconditioned_reference=z_reference,
            preconditioned_reference=preconditioned_reference,
            search_reference=search_reference,
            step_reference=step_reference,
            energy=energy,
            slope=applied_slope,
            applied_norm=applied_norm,
            trust_radius=trust_radius,
            on_boundary=on_boundary,
            gamma=gamma,
        )

        diagnostics = {
            "enabled": True,
            "coordinate_system": "fixed-reference-adjoint-transport",
            "method": "spectral-flexible-hybrid-PRP+",
            "accelerated": bool(
                history is not None
                and stable_secant
                and (
                    not np.isclose(gamma, 1.0, rtol=1e-14, atol=1e-16)
                    or beta > 0.0
                )
            ),
            "guarded": bool(
                restart_reasons
                or fallback is not None
                or trust_scale < 1.0
            ),
            "plain_equivalent": bool(
                np.isclose(gamma, 1.0, rtol=1e-14, atol=1e-16)
                and beta == 0.0
                and np.isclose(trust_scale, 1.0, rtol=1e-14, atol=1e-16)
                and fallback is None
            ),
            "history_used": bool(history is not None and stable_secant),
            "accepted_point_error": accepted_point_error,
            "linear_ratio": ratio,
            "linear_ratio_source": ratio_source,
            "trust_radius_before_update": trust_before_update,
            "trust_radius": trust_radius,
            "trust_action": trust_action,
            "trust_scale": float(trust_scale),
            "on_trust_boundary": on_boundary,
            "maximum_stepsize": maximum,
            "raw_norm": raw_norm,
            "search_norm": search_norm,
            "applied_norm": applied_norm,
            "raw_slope": raw_slope,
            "slope": applied_slope,
            "packed_slope": 0.5 * applied_slope,
            "raw_descent_cosine": raw_cosine,
            "search_descent_cosine": search_cosine,
            "descent_cosine": applied_cosine,
            "gamma": float(gamma),
            "gamma_unclipped": gamma_unclipped,
            "gamma_clipped": bool(gamma_clipped),
            "gamma_bounds": (self.gamma_min, self.gamma_max),
            "secant_curvature": secant_curvature,
            "secant_curvature_cosine": secant_curvature_cosine,
            "secant_step_norm": secant_step_norm,
            "secant_gradient_norm": secant_gradient_norm,
            "secant_preconditioner_norm": secant_preconditioner_norm,
            "preconditioned_curvature": preconditioned_curvature,
            "preconditioned_curvature_cosine": (
                preconditioned_curvature_cosine
            ),
            "raw_preconditioner_alignment": raw_preconditioner_alignment,
            "unresolved_curvature_boundary": bool(
                unresolved_curvature_boundary
            ),
            "unresolved_boundary_ratio_threshold": (
                self.unresolved_boundary_ratio
            ),
            "unresolved_boundary_alignment_threshold": (
                self.unresolved_boundary_alignment
            ),
            "beta_prp": beta_prp,
            "beta_fr": beta_fr,
            "beta": float(beta),
            "beta_clipped": bool(beta_clipped),
            "restart": bool(restart_reasons),
            "restart_reasons": restart_reasons,
            "fallback": fallback,
            "projection": {
                "raw": raw_projection,
                "search": search_projection,
            },
            "orthonormality_error": orthonormality_error,
            "pending": True,
        }
        return OrbitalCGResult(result_mo, applied_direction, diagnostics)


__all__ = ["OrbitalCGResult", "SpectralOrbitalCG"]
