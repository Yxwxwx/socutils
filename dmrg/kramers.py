"""Time-reversal adapter for full-spinor DMRG results.

The general :mod:`socutils.dmrg.dmrgci` solver uses one Block2 SGF site per
spinor.  Kramers restriction therefore does not mean discarding half of the
spinors.  This module instead identifies the time-reversal map in the actual
MO basis, pairs independently optimized roots, and assembles equal-weight
pair densities in the full active spinor space.

The RDM convention is the one used by ``zfci`` and ``DMRGCI``::

    dm1[p,q]       = <p^+ q>
    dm2[p,q,r,s]   = <p^+ r^+ s q>

If ``Theta |p> = sum_a |a> U[a,p]``, time reversal of these tensors is

``U* dm1* U^T`` and the corresponding four-index transformation with
``U*`` on the creation axes and ``U`` on the annihilation axes.  This is the
transpose of the more common AO density-matrix convention and is kept
explicit here to prevent a silent index error.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy


def _max_abs(value):
    value = numpy.asarray(value)
    return float(numpy.max(numpy.abs(value))) if value.size else 0.0


def ao_time_reverse(mol, coefficients):
    r"""Apply the repository/PySCF AO time-reversal map to MO columns.

    ``mol.time_reversal_map()`` is a signed, one-based *forward* map: a
    negative entry ``-j`` at position ``i`` means
    :math:`\Theta|i\rangle=-|j\rangle`.  Applying it consequently requires a
    scatter to row ``j`` rather than a gather from row ``j``.  Both two- and
    four-component coefficient arrays are supported.
    """
    coefficients = numpy.asarray(coefficients)
    if coefficients.ndim != 2:
        raise ValueError("MO coefficients must be a two-dimensional array")

    tao = numpy.asarray(mol.time_reversal_map(), dtype=int)
    n2c = tao.size
    if coefficients.shape[0] == n2c:
        nblocks = 1
    elif coefficients.shape[0] == 2 * n2c:
        nblocks = 2
    else:
        raise ValueError(
            "coefficient row count is incompatible with the molecular "
            "time-reversal map"
        )

    partner = numpy.abs(tao) - 1
    sign = numpy.where(tao > 0, 1.0, -1.0)
    transformed = numpy.empty_like(coefficients, dtype=numpy.complex128)
    for block in range(nblocks):
        start = block * n2c
        stop = start + n2c
        transformed[start + partner] = (
            sign[:, None] * coefficients[start:stop].conj()
        )
    return transformed


def _dominant_pairs(time_reversal):
    """Identify explicit orbital partners without assuming adjacency."""
    time_reversal = numpy.asarray(time_reversal)
    remaining = list(range(time_reversal.shape[0]))
    pairs = []
    phases = []
    while remaining:
        i = remaining[0]
        candidates = remaining[1:]
        if not candidates:
            raise ValueError("a Kramers orbital space must have even dimension")
        j = max(
            candidates,
            key=lambda candidate: (
                abs(time_reversal[candidate, i])
                + abs(time_reversal[i, candidate])
            ),
        )
        pairs.append((i, j))
        phases.append(complex(time_reversal[j, i]))
        remaining.remove(i)
        remaining.remove(j)
    return tuple(pairs), tuple(phases)


def validate_time_reversal(time_reversal, tolerance=1e-8):
    r"""Validate a one-particle antiunitary matrix with ``Theta**2 = -1``."""
    matrix = numpy.asarray(time_reversal, dtype=numpy.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("the time-reversal matrix must be square")
    if matrix.shape[0] % 2:
        raise ValueError("a Kramers orbital space must have even dimension")
    identity = numpy.eye(matrix.shape[0])
    diagnostics = {
        "unitarity_error": _max_abs(matrix.T.conj().dot(matrix) - identity),
        "time_reversal_square_error": _max_abs(
            matrix.dot(matrix.conj()) + identity
        ),
        "skew_symmetry_error": _max_abs(matrix + matrix.T),
    }
    if max(diagnostics.values(), default=0.0) > tolerance:
        raise ValueError(
            "invalid fermionic time-reversal matrix: "
            + ", ".join("%s=%.3e" % item for item in diagnostics.items())
        )
    return diagnostics


@dataclass(frozen=True)
class KramersOrbitalMap:
    """Validated time-reversal representation in one MO subspace."""

    time_reversal: numpy.ndarray
    pairs: tuple
    phases: tuple
    diagnostics: dict


def identify_kramers_orbitals(
    mol,
    mo_coeff,
    overlap,
    tolerance=1e-8,
):
    r"""Find and phase-check Kramers partners for the supplied MO columns.

    The partner matrix is obtained from ``C^H S Theta(C)``.  Partner indices
    are inferred from its dominant entries and then checked in the AO metric;
    no ``(2*i, 2*i+1)`` ordering is assumed.
    """
    mo_coeff = numpy.asarray(mo_coeff, dtype=numpy.complex128)
    overlap = numpy.asarray(overlap, dtype=numpy.complex128)
    if overlap.shape != (mo_coeff.shape[0],) * 2:
        raise ValueError("AO overlap and MO coefficient dimensions disagree")

    metric = mo_coeff.T.conj().dot(overlap).dot(mo_coeff)
    identity = numpy.eye(mo_coeff.shape[1])
    orthonormality_error = _max_abs(metric - identity)
    if orthonormality_error > tolerance:
        raise ValueError(
            "MO columns are not orthonormal in the supplied AO metric "
            "(error %.3e)" % orthonormality_error
        )

    transformed = ao_time_reverse(mol, mo_coeff)
    time_reversal = mo_coeff.T.conj().dot(overlap).dot(transformed)
    diagnostics = validate_time_reversal(time_reversal, tolerance=tolerance)
    closure = transformed - mo_coeff.dot(time_reversal)
    closure_metric = closure.T.conj().dot(overlap).dot(closure)
    closure_error = float(
        numpy.sqrt(max(0.0, numpy.max(numpy.diag(closure_metric).real)))
    ) if closure_metric.size else 0.0

    pairs, phases = _dominant_pairs(time_reversal)
    partner_errors = []
    phase_errors = []
    for (i, j), phase in zip(pairs, phases):
        difference = transformed[:, i] - phase * mo_coeff[:, j]
        error = numpy.vdot(difference, overlap.dot(difference)).real
        partner_errors.append(float(numpy.sqrt(max(0.0, error))))
        phase_errors.append(abs(time_reversal[i, j] + phase))

    diagnostics = dict(diagnostics)
    diagnostics.update(
        {
            "orthonormality_error": orthonormality_error,
            "subspace_closure_error": closure_error,
            "partner_orbital_error": max(partner_errors, default=0.0),
            "partner_phase_error": max(phase_errors, default=0.0),
            "pairs": pairs,
            "phases": phases,
        }
    )
    checked = (
        closure_error,
        diagnostics["partner_orbital_error"],
        diagnostics["partner_phase_error"],
    )
    if max(checked, default=0.0) > tolerance:
        raise ValueError(
            "the MO subspace is not composed of phase-resolved Kramers "
            "partners (closure %.3e, partner %.3e, phase %.3e)" % checked
        )
    return KramersOrbitalMap(
        numpy.array(time_reversal, copy=True), pairs, phases, diagnostics
    )


def time_reverse_rdm1(time_reversal, dm1):
    r"""Time reverse ``dm1[p,q] = <p^+ q>`` in the full spinor space."""
    matrix = numpy.asarray(time_reversal)
    dm1 = numpy.asarray(dm1)
    return numpy.einsum(
        "pa,qb,ab->pq", matrix.conj(), matrix, dm1.conj(), optimize=True
    )


def time_reverse_rdm2(time_reversal, dm2):
    r"""Time reverse ``dm2[p,q,r,s] = <p^+ r^+ s q>``."""
    matrix = numpy.asarray(time_reversal)
    dm2 = numpy.asarray(dm2)
    return numpy.einsum(
        "pa,qb,rc,sd,abcd->pqrs",
        matrix.conj(),
        matrix,
        matrix.conj(),
        matrix,
        dm2.conj(),
        optimize=True,
    )


def time_reverse_integrals(time_reversal, h1e, eri):
    r"""Time reverse spinor Hamiltonian coefficients.

    Unlike RDMs, Hamiltonian coefficients transform with ``U`` on creation
    axes and ``U*`` on annihilation axes.  This helper is primarily useful for
    constructing and checking exactly Kramers-symmetric test Hamiltonians.
    """
    matrix = numpy.asarray(time_reversal)
    h1e = numpy.asarray(h1e)
    eri = numpy.asarray(eri)
    h1e_tr = numpy.einsum(
        "ap,bq,pq->ab", matrix, matrix.conj(), h1e.conj(), optimize=True
    )
    eri_tr = numpy.einsum(
        "ap,bq,cr,ds,pqrs->abcd",
        matrix,
        matrix.conj(),
        matrix,
        matrix.conj(),
        eri.conj(),
        optimize=True,
    )
    return h1e_tr, eri_tr


def kramers_residual(time_reversal, dm1, dm2=None):
    """Return raw max-norm residuals against the time-reversal relation."""
    residual = {
        "dm1": _max_abs(dm1 - time_reverse_rdm1(time_reversal, dm1)),
    }
    if dm2 is not None:
        residual["dm2"] = _max_abs(
            dm2 - time_reverse_rdm2(time_reversal, dm2)
        )
    return residual


def align_transition_phase(candidate, reference=None, tolerance=1e-14):
    """Remove the arbitrary relative root phase from a transition 1-RDM.

    With a reference, the Frobenius overlap determines the phase.  Without a
    reference, the largest tensor entry is made positive real, providing a
    deterministic representation for reporting.
    """
    candidate = numpy.asarray(candidate)
    if reference is None:
        if not candidate.size:
            return numpy.array(candidate, copy=True), 1.0 + 0.0j
        value = candidate.ravel()[numpy.argmax(numpy.abs(candidate))]
        overlap = value.conjugate()
    else:
        overlap = numpy.vdot(candidate, numpy.asarray(reference))
    if abs(overlap) <= tolerance:
        raise ValueError("transition density has no stable phase anchor")
    phase = overlap / abs(overlap)
    return phase * candidate, complex(phase)


def canonicalize_root_space_rdm1(root_space, tolerance=1e-12):
    """Canonicalize unitary mixing and phases inside a root subspace.

    ``root_space[i,j,p,q]`` is ``<i|p^+ q|j>``.  A deterministic Hermitian
    one-body anchor with the largest projected eigenvalue separation fixes
    root mixing.  Remaining column phases are fixed from the largest
    transition-density entries.  The returned tensor can therefore be
    compared between exact CI and DMRG even when their degenerate roots use
    different bases.
    """
    root_space = numpy.asarray(root_space, dtype=numpy.complex128)
    if root_space.ndim != 4 or root_space.shape[0] != root_space.shape[1]:
        raise ValueError("root-space 1-RDM must have shape (nroot,nroot,norb,norb)")
    nroot, _, norb, norb2 = root_space.shape
    if norb != norb2:
        raise ValueError("orbital axes of the root-space 1-RDM must be square")

    anchors = []
    labels = []
    for p in range(norb):
        anchors.append(root_space[:, :, p, p])
        labels.append(("diagonal", p, p))
    for p in range(norb):
        for q in range(p):
            anchors.append(root_space[:, :, p, q] + root_space[:, :, q, p])
            labels.append(("real", p, q))
            anchors.append(
                1j * root_space[:, :, p, q]
                - 1j * root_space[:, :, q, p]
            )
            labels.append(("imaginary", p, q))

    best = None
    for label, anchor in zip(labels, anchors):
        anchor = (anchor + anchor.T.conj()) * 0.5
        eigenvalues, eigenvectors = numpy.linalg.eigh(anchor)
        if nroot <= 1:
            separation = numpy.inf
        else:
            separation = float(numpy.min(numpy.diff(eigenvalues)))
        candidate = (separation, label, eigenvalues, eigenvectors)
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is None or (nroot > 1 and best[0] <= tolerance):
        raise ValueError("no nondegenerate one-body anchor exists in the root subspace")

    separation, label, eigenvalues, rotation = best
    canonical = numpy.einsum(
        "ia,ijpq,jb->abpq",
        rotation.conj(),
        root_space,
        rotation,
        optimize=True,
    )
    phases = [1.0 + 0.0j]
    for root in range(1, nroot):
        transition = canonical[0, root]
        value = transition.ravel()[numpy.argmax(numpy.abs(transition))]
        if abs(value) <= tolerance:
            raise ValueError("root-space transition density has no phase anchor")
        phase = value.conjugate() / abs(value)
        rotation[:, root] *= phase
        phases.append(complex(phase))
    canonical = numpy.einsum(
        "ia,ijpq,jb->abpq",
        rotation.conj(),
        root_space,
        rotation,
        optimize=True,
    )
    diagnostics = {
        "anchor": label,
        "anchor_eigenvalues": eigenvalues,
        "anchor_separation": separation,
        "phases": tuple(phases),
        "root_hermiticity_error": _max_abs(
            canonical - canonical.transpose(1, 0, 3, 2).conj()
        ),
    }
    return canonical, rotation, diagnostics


@dataclass(frozen=True)
class KramersPairRDM:
    """Equal-weight full-spinor density of one identified root pair."""

    roots: tuple
    energies: tuple
    dm1: numpy.ndarray
    dm2: numpy.ndarray
    raw_dm1: numpy.ndarray
    raw_dm2: numpy.ndarray
    diagnostics: dict


class KramersResultAdapter:
    """Documented adapter for Kramers-paired full-spinor DMRG results.

    Projection is disabled by default.  When requested, the adapter first
    measures the raw pair-ensemble residual and refuses to project if it is
    larger than ``projection_tolerance``.  Thus projection can remove only
    roundoff/truncation noise and cannot hide a root, phase, or index error.
    """

    def __init__(
        self,
        time_reversal=None,
        *,
        energy_tolerance=1e-8,
        residual_tolerance=1e-8,
        orbital_tolerance=1e-8,
        project=False,
        projection_tolerance=None,
        root_projection_shift=100.0,
    ):
        self.energy_tolerance = float(energy_tolerance)
        self.residual_tolerance = float(residual_tolerance)
        self.orbital_tolerance = float(orbital_tolerance)
        self.project = bool(project)
        self.projection_tolerance = float(
            residual_tolerance
            if projection_tolerance is None
            else projection_tolerance
        )
        self.root_projection_shift = float(root_projection_shift)
        if min(
            self.energy_tolerance,
            self.residual_tolerance,
            self.orbital_tolerance,
            self.projection_tolerance,
            self.root_projection_shift,
        ) <= 0:
            raise ValueError("Kramers tolerances and projection shift must be positive")

        self.time_reversal = None
        self.orbital_pairs = ()
        self.orbital_phases = ()
        self.orbital_diagnostics = None
        self.orbital_history = []
        self.root_pairs = ()
        self.root_order = ()
        self.pair_results = ()
        self.diagnostics = {}
        if time_reversal is not None:
            self.set_time_reversal(time_reversal)

    def clear_results(self):
        self.root_pairs = ()
        self.root_order = ()
        self.pair_results = ()
        self.diagnostics = {}

    def set_time_reversal(self, time_reversal):
        matrix = numpy.asarray(time_reversal, dtype=numpy.complex128)
        diagnostics = validate_time_reversal(
            matrix, tolerance=self.orbital_tolerance
        )
        pairs, phases = _dominant_pairs(matrix)
        self.time_reversal = numpy.array(matrix, copy=True)
        self.orbital_pairs = pairs
        self.orbital_phases = phases
        self.orbital_diagnostics = dict(diagnostics)
        self.orbital_diagnostics.update({"pairs": pairs, "phases": phases})
        return self

    def set_orbitals(self, mol, mo_coeff, overlap):
        mapping = identify_kramers_orbitals(
            mol,
            mo_coeff,
            overlap,
            tolerance=self.orbital_tolerance,
        )
        self.time_reversal = mapping.time_reversal
        self.orbital_pairs = mapping.pairs
        self.orbital_phases = mapping.phases
        self.orbital_diagnostics = dict(mapping.diagnostics)
        self.orbital_history.append(dict(mapping.diagnostics))
        return mapping

    def validate_problem(self, norb, nelec, nroots):
        if self.time_reversal is None:
            raise RuntimeError(
                "Kramers mode needs an explicit time-reversal matrix or an "
                "orbital context from CASSCF"
            )
        if self.time_reversal.shape != (int(norb), int(norb)):
            raise ValueError("time-reversal matrix does not match the active space")
        if int(nelec) % 2 and int(nroots) % 2:
            raise ValueError(
                "an odd-electron Kramers calculation must target complete root pairs"
            )

    def align_transition(self, candidate, reference=None, tolerance=1e-14):
        """Phase-align one transition density through this adapter."""
        return align_transition_phase(
            candidate, reference=reference, tolerance=tolerance
        )

    def canonicalize_root_space(self, root_space, tolerance=1e-12):
        """Remove root mixing and phases from a transition-density space."""
        return canonicalize_root_space_rdm1(root_space, tolerance=tolerance)

    def _pair_roots(self, energies, dm1s, dm2s):
        nroots = len(energies)
        if nroots % 2:
            raise ValueError("Kramers root pairing requires an even number of roots")
        tr1 = [time_reverse_rdm1(self.time_reversal, dm) for dm in dm1s]
        tr2 = [time_reverse_rdm2(self.time_reversal, dm) for dm in dm2s]

        pair_data = {}
        for i in range(nroots):
            for j in range(i + 1, nroots):
                energy_error = abs(energies[i] - energies[j])
                dm1_error = max(
                    _max_abs(dm1s[j] - tr1[i]),
                    _max_abs(dm1s[i] - tr1[j]),
                )
                dm2_error = max(
                    _max_abs(dm2s[j] - tr2[i]),
                    _max_abs(dm2s[i] - tr2[j]),
                )
                cost = (
                    energy_error / self.energy_tolerance
                    + dm1_error / self.residual_tolerance
                    + dm2_error / self.residual_tolerance
                )
                pair_data[(i, j)] = (cost, energy_error, dm1_error, dm2_error)

        @lru_cache(None)
        def best_pairing(remaining):
            if not remaining:
                return 0.0, ()
            i = remaining[0]
            best = None
            for offset, j in enumerate(remaining[1:], start=1):
                rest = remaining[1:offset] + remaining[offset + 1 :]
                rest_cost, rest_pairs = best_pairing(rest)
                candidate = (
                    pair_data[(min(i, j), max(i, j))][0] + rest_cost,
                    ((min(i, j), max(i, j)),) + rest_pairs,
                )
                if best is None or candidate[0] < best[0]:
                    best = candidate
            return best

        _, pairs = best_pairing(tuple(range(nroots)))
        pairs = tuple(
            sorted(
                pairs,
                key=lambda pair: (energies[list(pair)].mean(), pair),
            )
        )
        return pairs, pair_data

    def analyze(
        self,
        energies,
        dm1s,
        dm2s,
        *,
        weights=None,
        overlap=None,
        projected_hamiltonian=None,
    ):
        """Pair roots and form validated equal-weight ensemble densities."""
        if self.time_reversal is None:
            raise RuntimeError("time-reversal matrix has not been configured")
        energies = numpy.asarray(energies, dtype=float)
        dm1s = [numpy.asarray(dm) for dm in dm1s]
        dm2s = [numpy.asarray(dm) for dm in dm2s]
        if not (len(energies) == len(dm1s) == len(dm2s)):
            raise ValueError("energies and root RDM lists have inconsistent lengths")

        root_pairs, pair_data = self._pair_roots(energies, dm1s, dm2s)
        results = []
        raw_max = 0.0
        projection_max = 0.0
        for pair in root_pairs:
            i, j = pair
            _, energy_error, partner_dm1, partner_dm2 = pair_data[pair]
            if energy_error > self.energy_tolerance:
                raise RuntimeError(
                    "Kramers-pair energy splitting %.3e exceeds %.3e"
                    % (energy_error, self.energy_tolerance)
                )
            if max(partner_dm1, partner_dm2) > self.residual_tolerance:
                raise RuntimeError(
                    "raw Kramers partner residual is too large "
                    "(dm1 %.3e, dm2 %.3e)" % (partner_dm1, partner_dm2)
                )
            if weights is not None and abs(weights[i] - weights[j]) > 1e-14:
                raise RuntimeError(
                    "members of a Kramers pair must have equal state-average weights"
                )

            raw_dm1 = (dm1s[i] + dm1s[j]) * 0.5
            raw_dm2 = (dm2s[i] + dm2s[j]) * 0.5
            residual = kramers_residual(
                self.time_reversal, raw_dm1, raw_dm2
            )
            raw_pair_max = max(residual.values())
            raw_max = max(raw_max, raw_pair_max)
            if raw_pair_max > self.residual_tolerance:
                raise RuntimeError(
                    "raw Kramers ensemble residual %.3e exceeds %.3e"
                    % (raw_pair_max, self.residual_tolerance)
                )

            if self.project:
                if raw_pair_max > self.projection_tolerance:
                    raise RuntimeError(
                        "refusing Kramers projection of a raw residual %.3e"
                        % raw_pair_max
                    )
                dm1 = (
                    raw_dm1
                    + time_reverse_rdm1(self.time_reversal, raw_dm1)
                ) * 0.5
                dm2 = (
                    raw_dm2
                    + time_reverse_rdm2(self.time_reversal, raw_dm2)
                ) * 0.5
            else:
                dm1 = numpy.array(raw_dm1, copy=True)
                dm2 = numpy.array(raw_dm2, copy=True)
            projection_change = max(
                _max_abs(dm1 - raw_dm1), _max_abs(dm2 - raw_dm2)
            )
            projection_max = max(projection_max, projection_change)
            diagnostics = {
                "energy_splitting": float(energy_error),
                "partner_dm1_residual": float(partner_dm1),
                "partner_dm2_residual": float(partner_dm2),
                "raw_ensemble_dm1_residual": residual["dm1"],
                "raw_ensemble_dm2_residual": residual["dm2"],
                "projection_applied": self.project,
                "projection_change": projection_change,
            }
            results.append(
                KramersPairRDM(
                    pair,
                    (float(energies[i]), float(energies[j])),
                    dm1,
                    dm2,
                    numpy.array(raw_dm1, copy=True),
                    numpy.array(raw_dm2, copy=True),
                    diagnostics,
                )
            )

        orthogonality_error = None
        projected_hamiltonian_error = None
        if overlap is not None:
            overlap = numpy.asarray(overlap)
            orthogonality_error = _max_abs(
                overlap - numpy.eye(overlap.shape[0])
            )
            if orthogonality_error > self.residual_tolerance:
                raise RuntimeError(
                    "Kramers root overlap residual %.3e exceeds %.3e"
                    % (orthogonality_error, self.residual_tolerance)
                )
        if projected_hamiltonian is not None:
            projected_hamiltonian = numpy.asarray(projected_hamiltonian)
            target = numpy.diag(energies)
            projected_hamiltonian_error = _max_abs(
                projected_hamiltonian - target
            )
            if projected_hamiltonian_error > self.energy_tolerance:
                raise RuntimeError(
                    "projected Kramers Hamiltonian residual %.3e exceeds %.3e"
                    % (projected_hamiltonian_error, self.energy_tolerance)
                )

        self.root_pairs = root_pairs
        self.root_order = tuple(root for pair in root_pairs for root in pair)
        self.pair_results = tuple(results)
        self.diagnostics = {
            "root_pairs": root_pairs,
            "root_order": self.root_order,
            "raw_ensemble_residual": raw_max,
            "projection_change": projection_max,
            "projection_applied": self.project,
            "root_orthogonality_error": orthogonality_error,
            "projected_hamiltonian_error": projected_hamiltonian_error,
            "pairs": [dict(result.diagnostics) for result in results],
        }
        return self.pair_results
