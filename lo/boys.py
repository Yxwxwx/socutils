"""Boys localization for real or complex molecular orbitals.

The optimizer maximizes the sum of squared orbital dipole centres on the
unitary manifold. Unlike the historical Jacobi implementation, all vectors
and transformations remain complex-valued, the requested orbital interval is
honoured, and localization never changes the selected orbital subspace.
"""

from dataclasses import dataclass

import numpy as np
import scipy.linalg
from pyscf.lib import logger


def _antihermitian(matrix):
    matrix = np.asarray(matrix, dtype=np.complex128)
    return 0.5 * (matrix - matrix.T.conj())


def _hermitian(matrix):
    matrix = np.asarray(matrix, dtype=np.complex128)
    return 0.5 * (matrix + matrix.T.conj())


def _assign_coefficients(target, columns, value):
    """Assign complex work arrays without silently truncating real inputs."""
    value = np.asarray(value)
    if not np.iscomplexobj(target):
        imaginary_error = float(np.max(abs(value.imag), initial=0.0))
        if imaginary_error > 1e-10:
            raise ValueError(
                "complex localized orbitals cannot be written in place to a "
                "real coefficient array"
            )
        value = value.real
    target[:, columns] = value


def boys_objective(dipole_matrices):
    """Return ``sum(i,xyz) <i|r_xyz|i>**2`` in atomic units."""
    matrices = np.asarray(dipole_matrices)
    if (
        matrices.ndim != 3
        or matrices.shape[0] != 3
        or matrices.shape[1] != matrices.shape[2]
    ):
        raise ValueError("dipole_matrices must have shape (3, nmo, nmo)")
    diagonals = np.diagonal(matrices, axis1=1, axis2=2).real
    return float(np.einsum("xi,xi->", diagonals, diagonals))


def _boys_gradient(dipole_matrices):
    gradient = np.zeros(dipole_matrices.shape[1:], dtype=np.complex128)
    for matrix in dipole_matrices:
        diagonal = np.diag(np.diag(matrix).real)
        gradient -= 2.0 * (diagonal.dot(matrix) - matrix.dot(diagonal))
    return _antihermitian(gradient)


def _project_time_reversal(generator, time_reversal):
    from socutils.dmrg.kramers import time_reverse_one_body

    projected = 0.5 * (
        generator + time_reverse_one_body(time_reversal, generator)
    )
    return _antihermitian(projected)


def _transform_sparse_block(dipoles, active, block_unitary):
    """Apply an identity-plus-small-block unitary in O(3*nmo^2*k)."""
    transformed = np.empty_like(dipoles)
    for component, matrix in enumerate(dipoles):
        right = np.array(matrix, copy=True)
        right[:, active] = matrix[:, active].dot(block_unitary)
        transformed[component] = right
        transformed[component, active, :] = (
            block_unitary.T.conj().dot(right[active, :])
        )
    return transformed


def _escape_stationary_point(dipoles, objective, time_reversal=None):
    """Find a deterministic Jacobi direction away from a zero-gradient saddle."""
    nmo = dipoles.shape[1]
    if nmo < 2:
        return None
    strengths = np.sum(abs(dipoles) ** 2, axis=0)
    pairs = [
        (float(strengths[i, j]), i, j)
        for i in range(nmo)
        for j in range(i + 1, nmo)
    ]
    pairs.sort(reverse=True)
    best = None
    best_objective = objective
    for _, first, second in pairs[: min(8, len(pairs))]:
        for value in (1.0, 1.0j):
            direction = np.zeros((nmo, nmo), dtype=np.complex128)
            direction[first, second] = value
            direction[second, first] = -value.conjugate()
            if time_reversal is not None:
                direction = _project_time_reversal(
                    direction,
                    time_reversal,
                )
            direction_norm = float(np.linalg.norm(direction))
            if direction_norm < 1e-12:
                continue
            direction /= direction_norm
            active = np.where(np.max(abs(direction), axis=0) > 1e-12)[0]
            block = direction[np.ix_(active, active)]
            for step in (
                -np.pi / 4,
                -np.pi / 8,
                np.pi / 8,
                np.pi / 4,
            ):
                block_unitary = scipy.linalg.expm(step * block)
                trial_dipoles = _transform_sparse_block(
                    dipoles,
                    active,
                    block_unitary,
                )
                trial_objective = boys_objective(trial_dipoles)
                if trial_objective > best_objective + 1e-12:
                    best_objective = trial_objective
                    best = (
                        active,
                        block_unitary,
                        trial_dipoles,
                        trial_objective,
                        step,
                    )
    return best


@dataclass(frozen=True)
class BoysLocalizationResult:
    """Localized orbitals and convergence information."""

    mo_coeff: np.ndarray
    rotation: np.ndarray
    converged: bool
    cycles: int
    initial_objective: float
    final_objective: float
    gradient_norm: float
    history: tuple
    symmetry_residual: float | None = None


def localize_dipoles(
    mo_coeff,
    dipole_matrices,
    *,
    conv_tol=1e-8,
    max_cycle=200,
    distance_threshold=5.0,
    time_reversal=None,
    verbose=logger.NOTE,
    log_object=None,
):
    """Localize one complete MO matrix from its dipole matrices.

    If ``time_reversal`` is supplied, every trial generator is projected onto
    the fermionic time-reversal-invariant (quaternionic) tangent space. The
    matrix is expressed in the supplied MO ordering, so Kramers partners need
    not be adjacent.
    """
    mo_coeff = np.asarray(mo_coeff, dtype=np.complex128)
    dipoles = np.asarray(dipole_matrices, dtype=np.complex128)
    if mo_coeff.ndim != 2:
        raise ValueError("mo_coeff must be a two-dimensional array")
    nmo = mo_coeff.shape[1]
    if dipoles.shape != (3, nmo, nmo):
        raise ValueError("dipole matrices and orbital dimensions disagree")
    if conv_tol <= 0:
        raise ValueError("conv_tol must be positive")
    if max_cycle <= 0:
        raise ValueError("max_cycle must be positive")
    if distance_threshold is not None and distance_threshold <= 0:
        raise ValueError("distance_threshold must be positive or None")

    dipoles = np.array([_hermitian(matrix) for matrix in dipoles])
    if time_reversal is not None:
        from socutils.dmrg.kramers import validate_time_reversal

        time_reversal = np.asarray(time_reversal, dtype=np.complex128)
        if time_reversal.shape != (nmo, nmo):
            raise ValueError("time-reversal and orbital dimensions disagree")
        validate_time_reversal(time_reversal, tolerance=1e-7)

    log = logger.new_logger(log_object, verbose)
    rotation = np.eye(nmo, dtype=np.complex128)
    objective = boys_objective(dipoles)
    initial_objective = objective
    history = []
    trial_step = 0.25
    converged = False
    gradient_norm = 0.0

    for cycle in range(max_cycle):
        gradient = _boys_gradient(dipoles)
        if distance_threshold is not None and nmo:
            centres = np.diagonal(dipoles, axis1=1, axis2=2).real.T
            distances = scipy.linalg.norm(
                centres[:, None, :] - centres[None, :, :],
                axis=2,
            )
            gradient[distances >= distance_threshold] = 0.0
            gradient = _antihermitian(gradient)
        if time_reversal is not None:
            gradient = _project_time_reversal(gradient, time_reversal)

        gradient_norm = float(np.linalg.norm(gradient))
        entry = {
            "cycle": int(cycle),
            "objective": float(objective),
            "gradient_norm": gradient_norm,
            "accepted_step": 0.0,
        }
        history.append(entry)
        if gradient_norm < conv_tol:
            escape = _escape_stationary_point(
                dipoles,
                objective,
                time_reversal=time_reversal,
            )
            if escape is None:
                converged = True
                break
            active, block_unitary, dipoles, trial_objective, step = escape
            rotation[:, active] = rotation[:, active].dot(block_unitary)
            entry["stationary_escape"] = True
            entry["accepted_step"] = float(step)
            entry["objective_change"] = float(trial_objective - objective)
            objective = trial_objective
            continue

        direction = gradient / gradient_norm
        accepted = False
        step = trial_step
        while step >= 1e-10:
            unitary = scipy.linalg.expm(step * direction)
            trial_dipoles = np.array(
                [unitary.T.conj().dot(matrix).dot(unitary) for matrix in dipoles]
            )
            trial_objective = boys_objective(trial_dipoles)
            if trial_objective > objective:
                accepted = True
                break
            step *= 0.5
        if not accepted:
            numerical_floor = np.sqrt(
                1000.0 * np.finfo(float).eps * max(1.0, abs(objective))
            )
            converged = gradient_norm <= max(conv_tol, numerical_floor)
            report = log.debug if converged else log.warn
            report(
                "Boys localization line search stalled at cycle %d; "
                "|g| = %.3e (numerical floor %.3e)",
                cycle,
                gradient_norm,
                numerical_floor,
            )
            break

        rotation = rotation.dot(unitary)
        dipoles = trial_dipoles
        entry["accepted_step"] = float(step)
        entry["objective_change"] = float(trial_objective - objective)
        objective = trial_objective
        trial_step = min(1.0, 1.5 * step)
        log.debug(
            "Boys cycle %d: objective %.12g, change %.3e, |g| %.3e, step %.3e",
            cycle,
            objective,
            entry["objective_change"],
            gradient_norm,
            step,
        )

    localized = mo_coeff.dot(rotation)
    symmetry_residual = None
    if time_reversal is not None:
        from socutils.dmrg.kramers import time_reverse_one_body

        symmetry_residual = float(
            np.max(
                abs(
                    rotation
                    - time_reverse_one_body(time_reversal, rotation)
                )
            )
        )
    log.info(
        "Boys localization %s after %d cycles: objective %.12g -> %.12g, |g| %.3e",
        "converged" if converged else "stopped",
        len(history),
        initial_objective,
        objective,
        gradient_norm,
    )
    return BoysLocalizationResult(
        mo_coeff=localized,
        rotation=rotation,
        converged=converged,
        cycles=len(history),
        initial_objective=initial_objective,
        final_objective=float(objective),
        gradient_norm=gradient_norm,
        history=tuple(history),
        symmetry_residual=symmetry_residual,
    )


def _dipole_integrals(mol, nao):
    if nao == mol.nao_nr():
        return mol.intor("int1e_r")
    if nao == len(mol.spinor_labels()):
        return mol.intor("int1e_r_spinor")
    raise ValueError(
        "MO row dimension is neither the scalar nor two-component AO dimension"
    )


def localize_boys(
    mol,
    mo_coeff,
    start=0,
    stop=None,
    *,
    conv_tol=1e-8,
    max_cycle=200,
    distance_threshold=5.0,
    inplace=False,
    return_info=False,
    verbose=None,
):
    """Boys-localize ``mo_coeff[:, start:stop]``.

    The returned full coefficient matrix is a copy unless ``inplace=True``.
    Set ``return_info=True`` to receive convergence details.
    """
    source = np.asarray(mo_coeff)
    if source.ndim != 2:
        raise ValueError("mo_coeff must be a two-dimensional array")
    nmo = source.shape[1]
    if stop is None:
        stop = nmo
    start, stop = int(start), int(stop)
    if not 0 <= start < stop <= nmo:
        raise ValueError("localization interval is outside the MO range")
    output = source if inplace else np.array(source, dtype=np.complex128, copy=True)
    selected = np.asarray(output[:, start:stop], dtype=np.complex128)
    dipole_ao = _dipole_integrals(mol, source.shape[0])
    dipole_mo = np.einsum(
        "pi,xpq,qj->xij",
        selected.conj(),
        dipole_ao,
        selected,
        optimize=True,
    )
    result = localize_dipoles(
        selected,
        dipole_mo,
        conv_tol=conv_tol,
        max_cycle=max_cycle,
        distance_threshold=distance_threshold,
        verbose=mol.verbose if verbose is None else verbose,
        log_object=mol,
    )
    _assign_coefficients(output, slice(start, stop), result.mo_coeff)
    if return_info:
        return output, result
    return output


def boys(
    smo,
    mdmx,
    mdmy,
    mdmz,
    ini0,
    ifi0,
    Dis_thr=5.0,
    maxcyc=2000,
):
    """Compatibility wrapper that localizes and updates arrays in place."""
    selected = np.asarray(smo)[:, ini0:ifi0]
    full_dipoles = [mdmx, mdmy, mdmz]
    selected_dipoles = np.array(
        [matrix[ini0:ifi0, ini0:ifi0] for matrix in full_dipoles]
    )
    result = localize_dipoles(
        selected,
        selected_dipoles,
        conv_tol=1e-5,
        max_cycle=maxcyc if maxcyc > 0 else 1000000,
        distance_threshold=Dis_thr,
    )
    _assign_coefficients(smo, slice(ini0, ifi0), result.mo_coeff)
    full_rotation = np.eye(smo.shape[1], dtype=np.complex128)
    full_rotation[ini0:ifi0, ini0:ifi0] = result.rotation
    for matrix in full_dipoles:
        transformed = full_rotation.T.conj().dot(matrix).dot(full_rotation)
        if not np.iscomplexobj(matrix):
            transformed = np.real_if_close(transformed, tol=1000).real
        matrix[:] = transformed
    return smo


def boys_driver(mol, smo, ini0, ifi0, Dis_thr=5.0, maxcyc=2000):
    """Historical in-place driver; prefer :func:`localize_boys`."""
    localized = localize_boys(
        mol,
        smo,
        ini0,
        ifi0,
        conv_tol=1e-5,
        max_cycle=maxcyc if maxcyc > 0 else 1000000,
        distance_threshold=Dis_thr,
        inplace=False,
    )
    smo[:] = localized
    return smo


__all__ = [
    "BoysLocalizationResult",
    "boys",
    "boys_driver",
    "boys_objective",
    "localize_boys",
    "localize_dipoles",
]
