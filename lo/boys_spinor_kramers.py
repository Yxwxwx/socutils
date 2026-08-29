"""Boys localization constrained to preserve fermionic time reversal."""

import numpy as np

from socutils.dmrg.kramers import ao_time_reverse, identify_kramers_orbitals
from socutils.lo.boys import _assign_coefficients, localize_dipoles


def apply_time_reversal_to_mo(mo_coeff, mol):
    """Apply the repository's validated AO time-reversal map."""
    return ao_time_reverse(mol, mo_coeff)


def validate_kramers_pair(
    mo_coeff,
    mo_energy,
    mol,
    energy_tol=1e-6,
    tol=1e-8,
):
    """Validate a Kramers-complete orbital space.

    ``mo_energy`` and ``energy_tol`` are accepted for compatibility. Partner
    identification is based on time reversal in the AO overlap metric and is
    therefore valid for reordered orbitals and higher degeneracies.
    """
    del mo_energy, energy_tol
    try:
        mapping = identify_kramers_orbitals(
            mol,
            mo_coeff,
            mol.intor("int1e_ovlp_spinor"),
            tolerance=tol,
        )
    except ValueError as error:
        return False, (str(error),), {}
    details = {
        "pairs": mapping.pairs,
        "phases": mapping.phases,
        "diagnostics": mapping.diagnostics,
    }
    return True, (), details


def _canonical_time_reversal(nmo):
    if nmo % 2:
        raise ValueError("a Kramers orbital space must have even dimension")
    matrix = np.zeros((nmo, nmo), dtype=np.complex128)
    matrix[1::2, 0::2] = np.eye(nmo // 2)
    matrix[0::2, 1::2] = -np.eye(nmo // 2)
    return matrix


def localize_boys_kramers(
    mol,
    mo_coeff,
    start=0,
    stop=None,
    *,
    conv_tol=1e-8,
    max_cycle=200,
    distance_threshold=5.0,
    pair_tolerance=1e-8,
    inplace=False,
    return_info=False,
    verbose=None,
):
    """Boys-localize a complete Kramers subspace without ordering assumptions."""
    source = np.asarray(mo_coeff)
    if source.ndim != 2:
        raise ValueError("mo_coeff must be a two-dimensional array")
    nmo = source.shape[1]
    if stop is None:
        stop = nmo
    start, stop = int(start), int(stop)
    if not 0 <= start < stop <= nmo:
        raise ValueError("localization interval is outside the MO range")
    selected = np.asarray(source[:, start:stop], dtype=np.complex128)
    overlap = mol.intor("int1e_ovlp_spinor")
    mapping = identify_kramers_orbitals(
        mol,
        selected,
        overlap,
        tolerance=pair_tolerance,
    )
    time_reversal = np.zeros_like(mapping.time_reversal)
    for (first, second), phase in zip(mapping.pairs, mapping.phases):
        phase = phase / abs(phase)
        time_reversal[second, first] = phase
        time_reversal[first, second] = -phase
    dipole_ao = mol.intor("int1e_r_spinor")
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
        time_reversal=time_reversal,
        verbose=mol.verbose if verbose is None else verbose,
        log_object=mol,
    )
    identify_kramers_orbitals(
        mol,
        result.mo_coeff,
        overlap,
        tolerance=max(pair_tolerance, 1e-7),
    )
    output = source if inplace else np.array(source, dtype=np.complex128, copy=True)
    _assign_coefficients(output, slice(start, stop), result.mo_coeff)
    if return_info:
        return output, result
    return output


def boys_spinor_kramers(
    smo,
    mdmx,
    mdmy,
    mdmz,
    ini0,
    ifi0,
    Dis_thr=5.0,
    maxcyc=1000,
    *,
    time_reversal=None,
):
    """Compatibility in-place optimizer for pretransformed dipole matrices.

    Pass ``time_reversal`` for arbitrary partner ordering. If omitted, the
    historical adjacent-pair convention is used for this low-level API;
    :func:`localize_boys_kramers` always derives the actual mapping.
    """
    selected = np.asarray(smo)[:, ini0:ifi0]
    selected_dipoles = np.array(
        [matrix[ini0:ifi0, ini0:ifi0] for matrix in (mdmx, mdmy, mdmz)]
    )
    if time_reversal is None:
        time_reversal = _canonical_time_reversal(ifi0 - ini0)
    result = localize_dipoles(
        selected,
        selected_dipoles,
        conv_tol=1e-5,
        max_cycle=maxcyc if maxcyc > 0 else 1000000,
        distance_threshold=Dis_thr,
        time_reversal=time_reversal,
    )
    _assign_coefficients(smo, slice(ini0, ifi0), result.mo_coeff)
    full_rotation = np.eye(smo.shape[1], dtype=np.complex128)
    full_rotation[ini0:ifi0, ini0:ifi0] = result.rotation
    for matrix in (mdmx, mdmy, mdmz):
        matrix[:] = full_rotation.T.conj().dot(matrix).dot(full_rotation)
    return smo


def boys_driver_kramers(mol, smo, ini0, ifi0, Dis_thr=5.0, maxcyc=1000):
    """Historical in-place driver; prefer :func:`localize_boys_kramers`."""
    localized = localize_boys_kramers(
        mol,
        smo,
        ini0,
        ifi0,
        conv_tol=1e-5,
        max_cycle=maxcyc if maxcyc > 0 else 1000000,
        distance_threshold=Dis_thr,
    )
    smo[:] = localized
    return smo


def check_spinor_orthonormality(mol, mo_coeff, s_mat=None, tol=1e-6):
    """Return ``(is_orthonormal, C^H S C, maximum_error)``."""
    if s_mat is None:
        s_mat = mol.intor("int1e_ovlp_spinor")
    metric = np.asarray(mo_coeff).T.conj().dot(s_mat).dot(mo_coeff)
    error = float(np.max(abs(metric - np.eye(metric.shape[0]))))
    return error < tol, metric, error


__all__ = [
    "apply_time_reversal_to_mo",
    "boys_driver_kramers",
    "boys_spinor_kramers",
    "check_spinor_orthonormality",
    "localize_boys_kramers",
    "validate_kramers_pair",
]
