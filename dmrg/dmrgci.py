"""Block2 DMRG-CI solver for complex relativistic spinor Hamiltonians.

The boundary implemented here follows :mod:`pyscf.fci.fci_dhf_slow` and
:mod:`socutils.fci.zfci`, which are the exact-CI references used by the
spinor CASCI/CASSCF code in this project.  Block2 is used in complex SGF mode:
one Block2 site is one spinor orbital and the only conserved quantum number is
particle number.
"""

from dataclasses import dataclass
import gc
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile

import numpy
from pyscf import lib
from pyscf.lib import StreamObject, logger

from .kramers import KramersResultAdapter


@dataclass(frozen=True)
class DMRGSweepSchedule:
    """A pyblock2 sweep schedule expanded from PySCF DMRG anchor points."""

    anchor_sweeps: tuple
    anchor_bond_dims: tuple
    anchor_thrds: tuple
    anchor_noises: tuple
    bond_dims: tuple
    thrds: tuple
    noises: tuple
    n_sweeps: int
    twosite_to_onesite: int | None
    restart: bool = False

    def as_dict(self):
        """Return a JSON-friendly diagnostic representation."""
        return {
            "anchor_sweeps": list(self.anchor_sweeps),
            "anchor_bond_dims": list(self.anchor_bond_dims),
            "anchor_thrds": list(self.anchor_thrds),
            "anchor_noises": list(self.anchor_noises),
            "bond_dims": list(self.bond_dims),
            "thrds": list(self.thrds),
            "noises": list(self.noises),
            "n_sweeps": self.n_sweeps,
            "twosite_to_onesite": self.twosite_to_onesite,
            "restart": self.restart,
        }


def _expand_schedule(anchor_sweeps, values, n_sweeps):
    expanded = []
    anchor = 0
    for sweep in range(n_sweeps):
        while (
            anchor + 1 < len(anchor_sweeps)
            and anchor_sweeps[anchor + 1] <= sweep
        ):
            anchor += 1
        expanded.append(values[anchor])
    return tuple(expanded)


def pyscf_dmrg_schedule(
    max_bond_dimension=1000,
    tol=1e-7,
    start_bond_dimension=None,
    restart=False,
    restart_sweeps=8,
    noise_scale=1.0,
    max_davidson_threshold=None,
):
    """Translate PySCF's official Block schedule to direct pyblock2 arrays.

    The reference interface writes piecewise-constant anchor rows to a Block
    input file.  pyblock2 instead accepts one array indexed by sweep, so the
    anchor rows are expanded here without changing their ranges.  A restart
    follows the reference ``fullrestart`` path: maximum bond dimension,
    zero noise, one-site sweeps, and a local threshold of ``tol / 10``.
    """
    max_bond_dimension = int(max_bond_dimension)
    tol = float(tol)
    restart_sweeps = int(restart_sweeps)
    noise_scale = float(noise_scale)
    if max_davidson_threshold is not None:
        max_davidson_threshold = float(max_davidson_threshold)
    if max_bond_dimension <= 0:
        raise ValueError("max_bond_dimension must be positive")
    if tol <= 0.0 or not numpy.isfinite(tol):
        raise ValueError("tol must be finite and positive")
    if tol / 10.0 <= 0.0:
        raise ValueError("tol is too small to form a finite restart threshold")
    if restart_sweeps <= 1:
        raise ValueError("restart_sweeps must be at least two")
    if noise_scale < 0.0 or not numpy.isfinite(noise_scale):
        raise ValueError("noise_scale must be finite and nonnegative")
    if (
        max_davidson_threshold is not None
        and (
            max_davidson_threshold <= 0.0
            or not numpy.isfinite(max_davidson_threshold)
        )
    ):
        raise ValueError(
            "max_davidson_threshold must be finite and positive"
        )

    def local_threshold(value):
        if max_davidson_threshold is None:
            return value
        return min(value, max_davidson_threshold)

    if restart:
        anchor_sweeps = (0,)
        anchor_bond_dims = (max_bond_dimension,)
        anchor_thrds = (local_threshold(tol / 10.0),)
        anchor_noises = (0.0,)
        n_sweeps = restart_sweeps
        twosite_to_onesite = None
    else:
        if start_bond_dimension is None:
            start_bond_dimension = 50 if max_bond_dimension < 200 else 200
        start_bond_dimension = int(start_bond_dimension)
        if start_bond_dimension <= 0:
            raise ValueError("start_bond_dimension must be positive")

        sweeps = []
        bond_dims = []
        thrds = []
        noises = []
        sweep = 0
        bond_dimension = start_bond_dimension
        local_thrd = 1e-4
        while bond_dimension < max_bond_dimension:
            sweeps.append(sweep)
            bond_dims.append(bond_dimension)
            thrds.append(local_threshold(local_thrd))
            noises.append(noise_scale * local_thrd)
            sweep += 4
            bond_dimension *= 2
        while local_thrd > tol:
            sweeps.append(sweep)
            bond_dims.append(max_bond_dimension)
            thrds.append(local_threshold(local_thrd))
            noises.append(noise_scale * local_thrd)
            sweep += 2
            local_thrd /= 10.0
        sweeps.append(sweep)
        bond_dims.append(max_bond_dimension)
        thrds.append(local_threshold(tol / 10.0))
        noises.append(0.0)
        sweep += 2

        anchor_sweeps = tuple(sweeps)
        anchor_bond_dims = tuple(bond_dims)
        anchor_thrds = tuple(thrds)
        anchor_noises = tuple(noises)
        twosite_to_onesite = sweep + 2
        n_sweeps = twosite_to_onesite + 8

    return DMRGSweepSchedule(
        anchor_sweeps=anchor_sweeps,
        anchor_bond_dims=anchor_bond_dims,
        anchor_thrds=anchor_thrds,
        anchor_noises=anchor_noises,
        bond_dims=_expand_schedule(
            anchor_sweeps, anchor_bond_dims, n_sweeps
        ),
        thrds=_expand_schedule(anchor_sweeps, anchor_thrds, n_sweeps),
        noises=_expand_schedule(anchor_sweeps, anchor_noises, n_sweeps),
        n_sweeps=n_sweeps,
        twosite_to_onesite=twosite_to_onesite,
        restart=bool(restart),
    )


def block2_integrals(h1e, eri, norb):
    r"""Copy spinor-FCI integrals into the Block2 QC-MPO convention.

    Both interfaces use the chemists' axes ``eri[p,q,r,s] = (pq|rs)`` and

    .. math::

        H = \sum_{pq} h_{pq} a_p^\dagger a_q
          + \frac12 \sum_{pqrs} (pq|rs)
            a_p^\dagger a_r^\dagger a_s a_q.

    Consequently no transpose, antisymmetrization, or conjugation is applied.
    Copies are returned because ``DMRGDriver.get_qc_mpo`` may zero integrals
    below its configured cutoff.  The complex transformed-Hamiltonian tests in
    ``test_dmrgci.py`` compare this boundary independently with ``zfci``.
    """
    h1e = numpy.asarray(h1e)
    eri = numpy.asarray(eri)
    if h1e.shape != (norb, norb):
        raise ValueError("h1e must have shape (norb, norb)")
    if eri.size != norb**4:
        raise ValueError("eri must contain norb**4 elements")
    return (
        numpy.array(h1e, dtype=numpy.complex128, order="C", copy=True),
        numpy.array(
            eri.reshape((norb,) * 4),
            dtype=numpy.complex128,
            order="C",
            copy=True,
        ),
    )


def block2_rdm1(raw_rdm1):
    r"""Convert a Block2 SGF 1-RDM to the spinor-FCI convention.

    Block2 returns ``raw_rdm1[p,q] = <a_p^dagger a_q>``.  The relativistic
    ``fci_dhf_slow``/``zfci`` implementation used by this repository returns
    the same axis order, so this boundary is a copy with no transpose or
    conjugation.
    """
    raw_rdm1 = numpy.asarray(raw_rdm1)
    if raw_rdm1.ndim != 2 or raw_rdm1.shape[0] != raw_rdm1.shape[1]:
        raise ValueError("raw Block2 1-RDM must be square")
    return numpy.array(raw_rdm1, dtype=numpy.complex128, order="C", copy=True)


def block2_rdm2(raw_rdm2):
    r"""Convert a Block2 SGF 2-RDM to the spinor-FCI convention.

    Block2 returns

    ``raw[i,j,b,a] = <a_i^dagger a_j^dagger a_b a_a>``.

    The spinor-FCI tensor used in ``socutils`` is

    ``dm2[p,q,r,s] = <a_p^dagger a_r^dagger a_s a_q>``.

    Thus ``dm2 = raw.transpose(0, 3, 1, 2)``.  No symmetry projection is
    performed: the raw converted tensor is what the exact-CI tests compare.
    """
    raw_rdm2 = numpy.asarray(raw_rdm2)
    if raw_rdm2.ndim != 4 or len(set(raw_rdm2.shape)) != 1:
        raise ValueError("raw Block2 2-RDM must have shape (norb,)*4")
    return numpy.array(
        raw_rdm2.transpose(0, 3, 1, 2),
        dtype=numpy.complex128,
        order="C",
        copy=True,
    )


def block2_transition_rdm1(raw_rdm1):
    r"""Convert ``<bra|a_p^dagger a_q|ket>`` from Block2.

    The Block2 and spinor-FCI axes and complex-conjugation conventions are
    identical.  Transition tensors retain the unavoidable independent global
    phases of the two states.
    """
    return block2_rdm1(raw_rdm1)


def energy_from_rdms(h1e, eri, dm1, dm2, ecore=0.0):
    r"""Reconstruct an energy in the ``zfci`` spinor-RDM convention.

    Here ``dm1[p,q] = <a_p^dagger a_q>`` and
    ``dm2[p,q,r,s] = <a_p^dagger a_r^dagger a_s a_q>``.  This differs in the
    1-RDM axis orientation from PySCF's nonrelativistic McWeeney convention;
    the direct contractions below follow the actual ``fci_dhf_slow`` source.
    """
    return (
        numpy.einsum("pq,pq->", h1e, dm1)
        + 0.5 * numpy.einsum("pqrs,pqrs->", eri, dm2)
        + ecore
    )


def _electron_number(nelec):
    if isinstance(nelec, (tuple, list, numpy.ndarray)):
        return int(sum(nelec))
    return int(nelec)


def _real_energy(value, tolerance=1e-10):
    values = numpy.asarray(value)
    if numpy.iscomplexobj(values) and numpy.max(numpy.abs(values.imag)) > tolerance:
        raise RuntimeError("Block2 returned a non-real energy for a Hermitian problem")
    values = numpy.asarray(values.real, dtype=float)
    if values.ndim == 0:
        return float(values)
    return values


def _history_row(value):
    try:
        return numpy.asarray(list(value), dtype=float)
    except TypeError:
        return numpy.asarray([value], dtype=float)


class DMRGCI(StreamObject):
    """PySCF-style complex spinor DMRG solver backed by Block2.

    ``max_memory`` and ``stack_memory`` are expressed in MB at the public
    interface.  The selected value is converted to bytes exactly once when a
    :class:`pyblock2.driver.core.DMRGDriver` is created.  ``stack_memory`` is a
    cap because Block2's stack is only one part of its total memory use.

    An arbitrary external ``ci0`` is not trusted.  When the restart scheduler
    requests it, however, the solver can reuse its own structurally validated
    MPS as a warm guess for the next CASSCF Hamiltonian.  This is deliberately
    distinguished from an exact disk resume, whose Hamiltonian fingerprint is
    checked before the checkpoint is loaded.
    """

    spin_square = None
    states_spin_square = None

    def __init__(self, mol=None):
        self.mol = mol
        if mol is None:
            self.stdout = sys.stdout
            self.verbose = logger.NOTE
            self.max_memory = 2000.0
        else:
            self.stdout = mol.stdout
            self.verbose = mol.verbose
            self.max_memory = float(mol.max_memory)

        self.n_threads = int(lib.num_threads())
        self.stack_memory = min(self.max_memory, 1000.0)
        self.scratch = os.path.join(os.path.abspath(lib.param.TMPDIR), "dmrgci")
        self.keep_scratch = False

        self.ncas = None
        self.nelecas = None
        self.nroots = 1
        self.wfnsym = None
        self.schedule_mode = "pyscf"
        self.max_bond_dimension = 1000
        self.start_bond_dimension = None
        self.restart_sweeps = 8
        self.schedule_noise_scale = 1.0
        self.schedule_thrd_max = None
        self.tol = 1e-7
        self.schedule_sweeps = []
        self.schedule_bond_dims = []
        self.schedule_thrds = []
        self.schedule_noises = []
        self.bond_dims = []
        self.noises = []
        self.thrds = []
        self.n_sweeps = 0
        self.twosite_to_onesite = None
        self.generate_schedule()
        self.dmrg_switch_tol = 1e-3
        self.restart = False
        self._restart = False
        self.restart_diagnostics = {}
        self.checkpoint_dir = None
        self.resume = False
        self.checkpoint_per_sweep = False
        self.cutoff = 1e-20
        self.integral_cutoff = 1e-20
        self.dav_type = None
        self.dav_max_iter = 4000
        self.dav_def_max_size = 50
        self.dav_rel_conv_thrd = 0.0
        self.noise_type = None
        self.random_seed = 1234
        self.npdm_site_type = 2
        self.npdm_cutoff = 1e-24

        # Explicit opt-in: the general complex-spinor solver remains the
        # default and keeps its original state-averaged multi-root route.
        self.kramers_adapter = None
        self.kramers_diagnostics = {}
        self.root_overlap = None
        self.projected_hamiltonian = None

        self.driver = None
        self._active_mpo = None
        self.ci = None
        self.kets = None
        self._multi_mps = None
        self._scratch = None
        self._rdm_cache = {}
        self._mps_signature = None
        self.e_tot = None
        self.e_cas = None
        self.converged = False
        self.convergence_info = {}
        self.rdm_diagnostics = {}

        self._keys = set(self.__dict__)

    def init(
        self,
        ncas,
        nelecas,
        nroots=1,
        bond_dims=None,
        noises=None,
        thrds=None,
        n_sweeps=None,
        tol=None,
        schedule_mode=None,
        max_bond_dimension=None,
        start_bond_dimension=None,
        maxM=None,
        startM=None,
        restart=None,
        dmrg_switch_tol=None,
        restart_sweeps=None,
        schedule_noise_scale=None,
        schedule_thrd_max=None,
        checkpoint_dir=None,
        resume=None,
        checkpoint_per_sweep=None,
        scratch=None,
        keep_scratch=None,
        n_threads=None,
        stack_memory=None,
        cutoff=None,
        integral_cutoff=None,
        dav_type=None,
        dav_max_iter=None,
        dav_def_max_size=None,
        dav_rel_conv_thrd=None,
        noise_type=None,
        twosite_to_onesite=None,
        random_seed=None,
        npdm_site_type=None,
        npdm_cutoff=None,
    ):
        """Configure the active space, sweep schedule, and solver controls.

        ``schedule_mode="pyscf"`` expands the official PySCF DMRG schedule;
        ``maxM``/``startM`` are accepted as aliases for the descriptive bond
        dimension names. Supplying any legacy sweep arrays selects
        ``"explicit"`` mode unless a mode was stated explicitly.

        ``restart`` is a one-shot request to reuse this solver's compatible
        MPS. The callback returned by :meth:`restart_scheduler_` controls
        subsequent CASSCF warm starts. ``resume=True`` is distinct: it is a
        one-shot, exact-Hamiltonian disk reload from ``checkpoint_dir``.
        """
        self.ncas = int(ncas)
        self.nelecas = _electron_number(nelecas)
        self.nroots = int(nroots)
        if self.ncas <= 0 or not 0 <= self.nelecas <= self.ncas:
            raise ValueError("invalid spinor active space")
        if self.nroots <= 0:
            raise ValueError("nroots must be positive")

        if tol is not None:
            self.tol = float(tol)
        if maxM is not None:
            if (
                max_bond_dimension is not None
                and int(max_bond_dimension) != int(maxM)
            ):
                raise ValueError("maxM and max_bond_dimension disagree")
            max_bond_dimension = maxM
        if startM is not None:
            if (
                start_bond_dimension is not None
                and int(start_bond_dimension) != int(startM)
            ):
                raise ValueError("startM and start_bond_dimension disagree")
            start_bond_dimension = startM
        if max_bond_dimension is not None:
            self.max_bond_dimension = int(max_bond_dimension)
        if start_bond_dimension is not None:
            self.start_bond_dimension = int(start_bond_dimension)
        if restart_sweeps is not None:
            self.restart_sweeps = int(restart_sweeps)
        if schedule_noise_scale is not None:
            self.schedule_noise_scale = float(schedule_noise_scale)
        if schedule_thrd_max is not None:
            self.schedule_thrd_max = float(schedule_thrd_max)

        explicit_controls = any(
            value is not None
            for value in (bond_dims, noises, thrds, n_sweeps)
        )
        if schedule_mode is None:
            mode = "explicit" if explicit_controls else self.schedule_mode
        else:
            mode = str(schedule_mode).lower().replace("_", "-")
            mode = {
                "auto": "pyscf",
                "default": "pyscf",
                "manual": "explicit",
            }.get(mode, mode)
        if mode not in ("pyscf", "explicit"):
            raise ValueError("schedule_mode must be 'pyscf' or 'explicit'")
        if mode == "pyscf" and explicit_controls:
            raise ValueError(
                "explicit sweep arrays/n_sweeps cannot be combined with "
                "schedule_mode='pyscf'"
            )
        self.schedule_mode = mode
        if mode == "pyscf":
            self.generate_schedule()
        else:
            # Generated schedules carry their own two-site endpoint.  It must
            # not leak into a subsequently supplied explicit schedule.
            self.twosite_to_onesite = None
            if bond_dims is not None:
                self.bond_dims = [int(x) for x in bond_dims]
            if noises is not None:
                self.noises = [float(x) for x in noises]
            if thrds is not None:
                self.thrds = [float(x) for x in thrds]
            if n_sweeps is not None:
                self.n_sweeps = int(n_sweeps)
            self.max_bond_dimension = max(self.bond_dims)
            self.schedule_sweeps = []
            self.schedule_bond_dims = []
            self.schedule_thrds = []
            self.schedule_noises = []

        if not self.bond_dims or not self.noises or not self.thrds:
            raise ValueError("bond_dims, noises, and thrds must be nonempty")
        if (
            min(self.bond_dims) <= 0
            or min(self.noises) < 0
            or min(self.thrds) <= 0
        ):
            raise ValueError("invalid DMRG sweep schedule")

        if restart is not None:
            self.restart = bool(restart)
        if dmrg_switch_tol is not None:
            self.dmrg_switch_tol = float(dmrg_switch_tol)
        if checkpoint_dir is not None:
            self.checkpoint_dir = os.path.abspath(os.fspath(checkpoint_dir))
        if resume is not None:
            self.resume = bool(resume)
        if checkpoint_per_sweep is not None:
            self.checkpoint_per_sweep = bool(checkpoint_per_sweep)
        if scratch is not None:
            self.scratch = os.path.abspath(os.fspath(scratch))
        if keep_scratch is not None:
            self.keep_scratch = bool(keep_scratch)
        if n_threads is not None:
            self.n_threads = int(n_threads)
        if stack_memory is not None:
            self.stack_memory = float(stack_memory)
        if cutoff is not None:
            self.cutoff = float(cutoff)
        if integral_cutoff is not None:
            self.integral_cutoff = float(integral_cutoff)
        if dav_type is not None:
            self.dav_type = dav_type
        if dav_max_iter is not None:
            self.dav_max_iter = int(dav_max_iter)
        if dav_def_max_size is not None:
            self.dav_def_max_size = int(dav_def_max_size)
        if dav_rel_conv_thrd is not None:
            self.dav_rel_conv_thrd = float(dav_rel_conv_thrd)
        if noise_type is not None:
            self.noise_type = str(noise_type)
        if twosite_to_onesite is not None:
            self.twosite_to_onesite = int(twosite_to_onesite)
        if random_seed is not None:
            self.random_seed = int(random_seed)
        if npdm_site_type is not None:
            self.npdm_site_type = int(npdm_site_type)
        if npdm_cutoff is not None:
            self.npdm_cutoff = float(npdm_cutoff)

        if self.n_sweeps <= 0 or self.tol <= 0:
            raise ValueError("n_sweeps and tol must be positive")
        if self.dmrg_switch_tol <= 0 or self.restart_sweeps <= 1:
            raise ValueError(
                "dmrg_switch_tol must be positive and restart_sweeps >= 2"
            )
        if self.schedule_noise_scale < 0.0:
            raise ValueError("schedule_noise_scale must be nonnegative")
        if self.schedule_thrd_max is not None and self.schedule_thrd_max <= 0.0:
            raise ValueError("schedule_thrd_max must be positive")
        if self.resume and self.checkpoint_dir is None:
            raise ValueError("resume=True requires checkpoint_dir")
        if self.n_threads <= 0 or self.stack_memory <= 0:
            raise ValueError("n_threads and stack_memory must be positive")
        return self

    def generate_schedule(self):
        """Generate and install the official PySCF-style cold schedule."""
        schedule = pyscf_dmrg_schedule(
            max_bond_dimension=self.max_bond_dimension,
            start_bond_dimension=self.start_bond_dimension,
            tol=self.tol,
            restart=False,
            restart_sweeps=self.restart_sweeps,
            noise_scale=self.schedule_noise_scale,
            max_davidson_threshold=self.schedule_thrd_max,
        )
        self.schedule_sweeps = list(schedule.anchor_sweeps)
        self.schedule_bond_dims = list(schedule.anchor_bond_dims)
        self.schedule_thrds = list(schedule.anchor_thrds)
        self.schedule_noises = list(schedule.anchor_noises)
        self.bond_dims = list(schedule.bond_dims)
        self.thrds = list(schedule.thrds)
        self.noises = list(schedule.noises)
        self.n_sweeps = schedule.n_sweeps
        self.twosite_to_onesite = schedule.twosite_to_onesite
        return self

    def clearSchedule(self):
        """Clear generated anchor rows, matching the PySCF DMRGCI API."""
        self.schedule_sweeps = []
        self.schedule_bond_dims = []
        self.schedule_thrds = []
        self.schedule_noises = []
        return self

    @property
    def maxM(self):
        return self.max_bond_dimension

    @maxM.setter
    def maxM(self, value):
        self.max_bond_dimension = int(value)

    @property
    def startM(self):
        return self.start_bond_dimension

    @startM.setter
    def startM(self, value):
        self.start_bond_dimension = None if value is None else int(value)

    @property
    def maxIter(self):
        return self.n_sweeps

    @maxIter.setter
    def maxIter(self, value):
        self.n_sweeps = int(value)

    @property
    def scheduleSweeps(self):
        return self.schedule_sweeps

    @property
    def scheduleMaxMs(self):
        return self.schedule_bond_dims

    @property
    def scheduleTols(self):
        return self.schedule_thrds

    @property
    def scheduleNoises(self):
        return self.schedule_noises

    def restart_scheduler_step(self, environment):
        """Update the next-kernel warm-start flag from one macroiteration."""
        gradient = environment.get(
            "norm_gorb", environment.get("orbital_gradient_norm")
        )
        density_change = environment.get("norm_ddm")
        reasons = []
        if gradient is not None and numpy.isfinite(gradient):
            if float(gradient) < self.dmrg_switch_tol:
                reasons.append("orbital_gradient")
        if density_change is not None and numpy.isfinite(density_change):
            if float(density_change) < 10.0 * self.dmrg_switch_tol:
                reasons.append("density_change")
        self._restart = bool(reasons)
        self.restart_diagnostics = {
            "enabled_for_next_kernel": self._restart,
            "orbital_gradient_norm": (
                None if gradient is None else float(gradient)
            ),
            "density_change_norm": (
                None if density_change is None else float(density_change)
            ),
            "switch_tolerance": self.dmrg_switch_tol,
            "reasons": reasons,
        }
        if self._restart:
            logger.debug(
                self,
                "DMRG warm restart enabled for next macroiteration: %s",
                ", ".join(reasons),
            )
        return self._restart

    def restart_scheduler_(self):
        """Return a PySCF callback implementing the official restart gate."""
        return self.restart_scheduler_step

    @property
    def M(self):
        return max(self.bond_dims)

    def kramers_restricted(self, time_reversal=None, **kwargs):
        """Enable Kramers-pair root/result handling and return ``self``.

        ``time_reversal`` is the active-MO matrix ``U`` in
        ``Theta|p> = sum_q |q> U[q,p]``.  It may be omitted when this solver
        is attached to ``socutils`` CASSCF; the solver-to-CASSCF hook then
        derives and validates ``U`` from every current active MO subspace.

        Multi-root calculations use Block2's state-averaged ``MultiMPS`` with
        the same weights as the PySCF state-average wrapper.  They finish with
        one-site sweeps before the individual roots are split.  A pure
        two-site endpoint can leave the root vectors at the shared center
        inconsistent with the reported sweep energies, with the exact failure
        depending on the local preconditioner and energy degeneracies.
        """
        self.kramers_adapter = KramersResultAdapter(time_reversal, **kwargs)
        self.kramers_diagnostics = {}
        return self

    def set_orbital_context(self, mo_coeff, overlap, mol=None):
        """Receive and validate the current active orbitals from CASSCF."""
        if self.kramers_adapter is None:
            return None
        if mol is None:
            mol = self.mol
        if mol is None:
            raise ValueError("a molecule is required to identify Kramers orbitals")
        return self.kramers_adapter.set_orbitals(mol, mo_coeff, overlap)

    def _clear_results(self):
        self._rdm_cache.clear()
        self.rdm_diagnostics.clear()
        self.kramers_diagnostics.clear()
        self.root_overlap = None
        self.projected_hamiltonian = None
        if self.kramers_adapter is not None:
            self.kramers_adapter.clear_results()
        self.ci = None
        self.kets = None
        self._active_mpo = None
        self._multi_mps = None
        self._mps_signature = None
        gc.collect()

    def _release_run(self, remove_scratch=True):
        run_scratch = self._scratch
        self._clear_results()
        self.driver = None
        self._scratch = None
        gc.collect()
        if (
            remove_scratch
            and not self.keep_scratch
            and run_scratch is not None
            and os.path.isdir(run_scratch)
        ):
            shutil.rmtree(run_scratch)

    def cleanup(self):
        """Release the current MPS/driver and remove owned scratch data."""
        self._release_run(remove_scratch=True)
        return self

    close = cleanup

    def __del__(self):
        try:
            self._release_run(remove_scratch=True)
        except Exception:
            pass

    def _stack_bytes(self, max_memory):
        available_mb = self.max_memory if max_memory is None else float(max_memory)
        stack_mb = min(available_mb, self.stack_memory)
        if stack_mb <= 0:
            raise ValueError("no positive memory is available to Block2")
        return int(stack_mb * 1_000_000)

    @staticmethod
    def _expanded_values(values, n_sweeps):
        return tuple(values[min(sweep, len(values) - 1)] for sweep in range(n_sweeps))

    def _schedule_snapshot(self, restart=False):
        if restart:
            return pyscf_dmrg_schedule(
                max_bond_dimension=max(self.bond_dims),
                start_bond_dimension=self.start_bond_dimension,
                tol=self.tol,
                restart=True,
                restart_sweeps=self.restart_sweeps,
                noise_scale=self.schedule_noise_scale,
                max_davidson_threshold=self.schedule_thrd_max,
            )

        bond_dims = self._expanded_values(self.bond_dims, self.n_sweeps)
        thrds = self._expanded_values(self.thrds, self.n_sweeps)
        noises = self._expanded_values(self.noises, self.n_sweeps)
        if self.schedule_sweeps:
            anchor_sweeps = tuple(self.schedule_sweeps)
            anchor_bond_dims = tuple(self.schedule_bond_dims)
            anchor_thrds = tuple(self.schedule_thrds)
            anchor_noises = tuple(self.schedule_noises)
        else:
            anchor_sweeps = []
            anchor_bond_dims = []
            anchor_thrds = []
            anchor_noises = []
            previous = None
            for sweep, values in enumerate(zip(bond_dims, thrds, noises)):
                if values != previous:
                    anchor_sweeps.append(sweep)
                    anchor_bond_dims.append(values[0])
                    anchor_thrds.append(values[1])
                    anchor_noises.append(values[2])
                    previous = values
            anchor_sweeps = tuple(anchor_sweeps)
            anchor_bond_dims = tuple(anchor_bond_dims)
            anchor_thrds = tuple(anchor_thrds)
            anchor_noises = tuple(anchor_noises)
        return DMRGSweepSchedule(
            anchor_sweeps=anchor_sweeps,
            anchor_bond_dims=anchor_bond_dims,
            anchor_thrds=anchor_thrds,
            anchor_noises=anchor_noises,
            bond_dims=bond_dims,
            thrds=thrds,
            noises=noises,
            n_sweeps=self.n_sweeps,
            twosite_to_onesite=self.twosite_to_onesite,
            restart=False,
        )

    def _state_average_weights(self, nroots):
        if nroots == 1:
            return numpy.ones(1, dtype=float)
        weights = numpy.asarray(
            getattr(self, "weights", numpy.ones(nroots) / nroots),
            dtype=float,
        )
        if weights.shape != (nroots,) or not numpy.all(numpy.isfinite(weights)):
            raise ValueError(
                "state-average weights must be one finite value per root"
            )
        if numpy.any(weights <= 0.0) or weights.sum() <= 0.0:
            raise ValueError("state-average weights must be positive")
        return weights / weights.sum()

    @staticmethod
    def _wavefunction_problem(norb, nelec, nroots, weights):
        return (
            "SGFCPX",
            int(norb),
            int(nelec),
            int(nroots),
            tuple(float(value) for value in weights),
        )

    def _checkpoint_problem(
        self, h1e, eri, norb, nelec, nroots, weights, ecore
    ):
        structural = {
            "symmetry": "SGFCPX",
            "norb": int(norb),
            "nelec": int(nelec),
            "nroots": int(nroots),
            "weights": [float(value) for value in weights],
        }
        controls = {
            "ecore": float(ecore),
            "mpo_cutoff": float(self.cutoff),
            "integral_cutoff": float(self.integral_cutoff),
        }
        digest = hashlib.sha256()
        digest.update(
            json.dumps(
                {"structural": structural, "controls": controls},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        for name, values in (("h1e", h1e), ("eri", eri)):
            array = numpy.ascontiguousarray(values)
            digest.update(name.encode("ascii"))
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(repr(array.shape).encode("ascii"))
            digest.update(array.view(numpy.uint8))
        return {
            "structural": structural,
            "controls": controls,
            "hamiltonian_sha256": digest.hexdigest(),
        }

    def _checkpoint_paths(self):
        if self.checkpoint_dir is None:
            return None, None, None
        root = os.path.abspath(self.checkpoint_dir)
        return (
            os.path.join(root, "dmrgci-checkpoint.json"),
            os.path.join(root, "mps"),
            os.path.join(root, "sweeps"),
        )

    @staticmethod
    def _read_json(path):
        try:
            with open(path, encoding="utf-8") as handle:
                return json.load(handle)
        except FileNotFoundError:
            return None

    @staticmethod
    def _write_json_atomic(path, payload):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        temporary = "%s.tmp-%d" % (path, os.getpid())
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)

    def _load_checkpoint(self, problem, required=False):
        manifest_path, mps_path, _ = self._checkpoint_paths()
        if manifest_path is None:
            if required:
                raise ValueError("checkpoint resume requested without checkpoint_dir")
            return None
        manifest = self._read_json(manifest_path)
        if manifest is None:
            if required:
                raise FileNotFoundError(
                    "no DMRG checkpoint manifest at %s" % manifest_path
                )
            return None
        if (
            manifest.get("format") != "socutils.dmrgci.checkpoint"
            or manifest.get("version") != 1
        ):
            raise ValueError("unsupported DMRG checkpoint format")
        stored = manifest.get("problem", {})
        if stored.get("hamiltonian_sha256") != problem["hamiltonian_sha256"]:
            raise ValueError(
                "DMRG checkpoint Hamiltonian fingerprint does not match"
            )
        info_paths = (
            os.path.join(mps_path, "GS-mps_info.bin"),
            os.path.join(mps_path, "mps_info.bin"),
        )
        if not any(os.path.isfile(path) for path in info_paths):
            if required:
                raise FileNotFoundError("DMRG checkpoint contains no loadable MPS")
            return None
        return manifest

    def _begin_checkpoint(self, problem, run_mode):
        manifest_path, mps_path, sweeps_path = self._checkpoint_paths()
        if manifest_path is None:
            return None, None
        previous = self._read_json(manifest_path)
        previous_structure = (
            None
            if previous is None
            else previous.get("problem", {}).get("structural")
        )
        if previous_structure not in (None, problem["structural"]):
            if os.path.isdir(mps_path):
                shutil.rmtree(mps_path)
            if os.path.isdir(sweeps_path):
                shutil.rmtree(sweeps_path)
        os.makedirs(mps_path, exist_ok=True)
        per_sweep = None
        if self.checkpoint_per_sweep:
            per_sweep = os.path.join(
                sweeps_path, problem["hamiltonian_sha256"]
            )
            os.makedirs(per_sweep, exist_ok=True)
        manifest = {
            "format": "socutils.dmrgci.checkpoint",
            "version": 1,
            "status": "running",
            "mps_tag": "GS",
            "problem": problem,
            "run_mode": run_mode,
        }
        self._write_json_atomic(manifest_path, manifest)
        return mps_path, per_sweep

    def _complete_checkpoint(self, schedule, run_mode):
        manifest_path, _, _ = self._checkpoint_paths()
        if manifest_path is None:
            return
        manifest = self._read_json(manifest_path)
        if manifest is None:
            return
        manifest.update(
            {
                "status": "complete",
                "run_mode": run_mode,
                "converged": bool(self.converged),
                "energies": numpy.asarray(self.e_tot).tolist(),
                "schedule": schedule.as_dict(),
            }
        )
        self._write_json_atomic(manifest_path, manifest)

    @staticmethod
    def _copy_checkpoint_mps(source, destination):
        if not os.path.isdir(source):
            raise FileNotFoundError("DMRG checkpoint MPS directory is missing")
        shutil.copytree(source, destination, dirs_exist_ok=True)

    @staticmethod
    def _copy_internal_mps(source, destination, tag="GS"):
        """Copy one saved MPS image without carrying an obsolete MPO."""
        if not os.path.isdir(source):
            raise FileNotFoundError("internal DMRG MPS directory is missing")
        copied_info = False
        os.makedirs(destination, exist_ok=True)
        for name in os.listdir(source):
            path = os.path.join(source, name)
            if not os.path.isfile(path):
                continue
            if (
                name == "mps_info.bin"
                or name.startswith(tag + "-")
                or (".%s." % tag) in name
            ):
                shutil.copy2(path, os.path.join(destination, name))
                if name in ("mps_info.bin", "%s-mps_info.bin" % tag):
                    copied_info = True
        if not copied_info:
            raise FileNotFoundError("internal DMRG state contains no MPS info")

    def _validate_problem(self, norb, nelec, nroots):
        nelec = _electron_number(nelec)
        if self.ncas is not None and int(norb) != self.ncas:
            raise ValueError("kernel norb does not match configured ncas")
        if self.nelecas is not None and nelec != self.nelecas:
            raise ValueError("kernel nelec does not match configured nelecas")
        if not 0 <= nelec <= norb:
            raise ValueError("invalid electron number")
        dimension = math.comb(int(norb), nelec)
        if nroots > dimension:
            raise ValueError("nroots exceeds the determinant-space dimension")
        if self.kramers_adapter is not None:
            self.kramers_adapter.validate_problem(norb, nelec, nroots)
        return nelec

    def kernel(
        self,
        h1e,
        eri,
        norb,
        nelec,
        ci0=None,
        verbose=None,
        max_memory=None,
        ecore=0.0,
        nroots=None,
        **_kwargs,
    ):
        """Run complex SGF DMRG and return ``(energy, MPS)``.

        The scalar ``ecore`` is kept out of the Block2 MPO and added back to
        the public energies afterwards.  A constant cannot change an MPS, and
        excluding it keeps sweep-to-sweep convergence meaningful when a heavy
        atom's core energy is so large that its double-precision ULP exceeds
        the requested active-space energy tolerance.
        """
        from pyblock2.driver.core import DMRGDriver, SymmetryTypes

        if nroots is None:
            nroots = self.nroots
        nroots = int(nroots)
        nelec = self._validate_problem(norb, nelec, nroots)
        weights = self._state_average_weights(nroots)
        wavefunction_problem = self._wavefunction_problem(
            norb, nelec, nroots, weights
        )
        effective_dav_type = self.dav_type
        h1_block2, eri_block2 = block2_integrals(h1e, eri, norb)
        ecore_value = _real_energy(ecore)
        if numpy.asarray(ecore_value).ndim != 0:
            raise ValueError("ecore must be a scalar")
        ecore_value = float(ecore_value)
        checkpoint_problem = self._checkpoint_problem(
            h1_block2,
            eri_block2,
            norb,
            nelec,
            nroots,
            weights,
            ecore_value,
        )

        resume_checkpoint = bool(self.resume)
        if resume_checkpoint:
            self._load_checkpoint(checkpoint_problem, required=True)
        restart_requested = bool(self.restart or self._restart)
        in_memory_compatible = bool(
            self.driver is not None
            and self._multi_mps is not None
            and self._mps_signature == wavefunction_problem
        )
        use_internal_mps = (
            restart_requested
            and in_memory_compatible
            and not resume_checkpoint
        )
        if resume_checkpoint:
            run_mode = "checkpoint-resume"
        elif use_internal_mps:
            run_mode = "casscf-warm-start"
        else:
            run_mode = "cold-start"
        if restart_requested and run_mode == "cold-start":
            logger.warn(
                self,
                "DMRG restart was requested but no compatible internal MPS "
                "exists; falling back to the full cold schedule",
            )
        if ci0 is not None:
            logger.debug(
                self,
                "Ignoring external ci0; DMRG restart accepts only the solver's "
                "validated internal MPS or a fingerprinted checkpoint",
            )

        schedule = self._schedule_snapshot(
            restart=run_mode != "cold-start"
        )
        effective_twosite_to_onesite = schedule.twosite_to_onesite
        if (
            run_mode == "cold-start"
            and nroots > 1
            and int(norb) > 2
            and effective_twosite_to_onesite is None
        ):
            # Retain two two-site sweeps for initial optimization and leave at
            # least one one-site sweep even for deliberately short schedules.
            effective_twosite_to_onesite = (
                2 if schedule.n_sweeps >= 3 else 0
            )

        if run_mode == "cold-start":
            self._release_run(remove_scratch=True)
            os.makedirs(self.scratch, exist_ok=True)
            run_scratch = tempfile.mkdtemp(prefix="dmrgci_", dir=self.scratch)
            self._scratch = run_scratch
        else:
            os.makedirs(self.scratch, exist_ok=True)
            run_scratch = tempfile.mkdtemp(prefix="dmrgci_", dir=self.scratch)
            try:
                if resume_checkpoint:
                    _, checkpoint_mps, _ = self._checkpoint_paths()
                    self._copy_checkpoint_mps(checkpoint_mps, run_scratch)
                else:
                    self._copy_internal_mps(self._scratch, run_scratch)
            except Exception:
                shutil.rmtree(run_scratch)
                raise
            self._release_run(remove_scratch=True)
            self._scratch = run_scratch
        driver = None
        ket = None

        self.e_tot = None
        self.e_cas = None
        self.converged = False
        self.convergence_info = {}
        iprint = 1 if logger.new_logger(self, verbose).verbose >= logger.NOTE else 0
        stack_bytes = self._stack_bytes(max_memory)

        self.dump_flags(verbose=verbose)
        try:
            checkpoint_mps, checkpoint_sweeps = self._begin_checkpoint(
                checkpoint_problem, run_mode
            )
            driver = DMRGDriver(
                stack_mem=stack_bytes,
                scratch=run_scratch,
                clean_scratch=True,
                symm_type=SymmetryTypes.SGFCPX,
                n_threads=self.n_threads,
                restart_dir=checkpoint_mps,
                restart_dir_per_sweep=checkpoint_sweeps,
            )
            driver.bw.b.Random.rand_seed(self.random_seed)
            driver.initialize_system(
                n_sites=int(norb),
                n_elec=nelec,
                orb_sym=[0] * int(norb),
            )
            if run_mode != "cold-start":
                ket = driver.load_mps("GS", nroots=nroots)
            mpo = driver.get_qc_mpo(
                h1e=h1_block2,
                g2e=eri_block2,
                ecore=0.0,
                cutoff=self.cutoff,
                integral_cutoff=self.integral_cutoff,
                iprint=iprint,
            )
            dmrg_kwargs = {
                "n_sweeps": schedule.n_sweeps,
                "tol": self.tol,
                "bond_dims": list(schedule.bond_dims),
                "noises": list(schedule.noises),
                "thrds": list(schedule.thrds),
                "iprint": iprint,
                "dav_type": effective_dav_type,
                "dav_max_iter": self.dav_max_iter,
                "dav_def_max_size": self.dav_def_max_size,
                "dav_rel_conv_thrd": self.dav_rel_conv_thrd,
                "cutoff": self.cutoff,
                "twosite_to_onesite": effective_twosite_to_onesite,
                "real_density_matrix": False,
            }
            if self.noise_type is not None:
                dmrg_kwargs["noise_type"] = self.noise_type
            run_records = []
            if ket is None:
                ket = driver.get_random_mps(
                    tag="GS",
                    bond_dim=schedule.bond_dims[0],
                    nroots=nroots,
                )
            elif run_mode != "cold-start":
                ket, forward = driver.adjust_mps(ket, dot=1)
                dmrg_kwargs["forward"] = forward
            if nroots > 1:
                ket.weights = driver.bw.VectorFP(weights.tolist())
            energy = driver.dmrg(mpo, ket, **dmrg_kwargs)
            run_records.append(self._capture_dmrg_run(driver, schedule))
            if nroots > 1:
                kets = [
                    driver.split_mps(ket, root, tag="KET-%d" % root)
                    for root in range(nroots)
                ]
                ci = kets
            else:
                kets = [ket]
                ci = ket
            self._multi_mps = ket

            if nroots > 1:
                identity_mpo = driver.get_identity_mpo()
                self.root_overlap = numpy.empty(
                    (nroots, nroots), dtype=numpy.complex128
                )
                self.projected_hamiltonian = numpy.empty_like(
                    self.root_overlap
                )
                for i in range(nroots):
                    for j in range(i, nroots):
                        overlap_ij = driver.expectation(
                            kets[i], identity_mpo, kets[j]
                        )
                        active_hamiltonian_ij = driver.expectation(
                            kets[i], mpo, kets[j]
                        )
                        hamiltonian_ij = (
                            active_hamiltonian_ij + ecore_value * overlap_ij
                        )
                        self.root_overlap[i, j] = overlap_ij
                        self.projected_hamiltonian[i, j] = hamiltonian_ij
                        if i != j:
                            self.root_overlap[j, i] = numpy.conj(overlap_ij)
                            self.projected_hamiltonian[j, i] = numpy.conj(
                                hamiltonian_ij
                            )
                root_orthogonality_error = float(
                    numpy.max(abs(self.root_overlap - numpy.eye(nroots)))
                )
                root_eigen_equation_error = float(
                    numpy.max(
                        abs(
                            self.projected_hamiltonian
                            - self.root_overlap
                            * (numpy.asarray(energy) + ecore_value)[None, :]
                        )
                    )
                )
                root_validation_tolerance = max(
                    1e-7,
                    10.0 * math.sqrt(min(schedule.thrds)),
                    10.0 * self.tol,
                )
                if max(
                    root_orthogonality_error, root_eigen_equation_error
                ) > root_validation_tolerance:
                    raise RuntimeError(
                        "split state-averaged MultiMPS roots are inconsistent "
                        "with the reported energies (S-I %.3e, H-SE %.3e); "
                        "finish the multi-root calculation with one-site sweeps"
                        % (
                            root_orthogonality_error,
                            root_eigen_equation_error,
                        )
                    )

            self.driver = driver
            self._active_mpo = mpo
            self.kets = kets
            self.ci = ci
            self._mps_signature = wavefunction_problem
            self.nroots = nroots
            self.ncas = int(norb)
            self.nelecas = nelec
            self.e_cas = _real_energy(energy)
            self.e_tot = self.e_cas + ecore_value
            self._record_convergence(run_records)
            self.convergence_info.update(
                {
                    "constant_energy_shift": ecore_value,
                    "sweep_energy_origin": "active-space Hamiltonian without ecore",
                    "run_mode": run_mode,
                    "restart_transport": (
                        None
                        if run_mode == "cold-start"
                        else "fresh-driver-mps-reload"
                    ),
                    "restart_requested": restart_requested,
                    "schedule_mode": self.schedule_mode,
                    "schedule": schedule.as_dict(),
                    "checkpoint_dir": self.checkpoint_dir,
                    "checkpoint_fingerprint": checkpoint_problem[
                        "hamiltonian_sha256"
                    ],
                    "restart_scheduler": dict(self.restart_diagnostics),
                }
            )
            if nroots > 1:
                self.convergence_info["state_average_weights"] = weights
                self.convergence_info["effective_dav_type"] = (
                    "Normal" if effective_dav_type is None else effective_dav_type
                )
                self.convergence_info["effective_twosite_to_onesite"] = (
                    effective_twosite_to_onesite
                )
                self.convergence_info["root_orthogonality_error"] = (
                    root_orthogonality_error
                )
                self.convergence_info["root_eigen_equation_error"] = (
                    root_eigen_equation_error
                )
            self._complete_checkpoint(schedule, run_mode)
            if self.restart:
                self.restart = False
            if self.resume:
                self.resume = False
            return self.e_tot, self.ci
        except Exception:
            self._release_run(remove_scratch=True)
            raise

    def _capture_dmrg_run(self, driver, schedule=None):
        """Copy convergence data before a subsequent root overwrites it."""
        if schedule is None:
            schedule = self._schedule_snapshot(restart=False)
        history = [_history_row(row) for row in driver._dmrg.energies]
        nsweep = len(history)
        if nsweep >= 2:
            energy_change = float(numpy.max(numpy.abs(history[-1] - history[-2])))
        else:
            energy_change = numpy.inf
        schedule_index = max(nsweep - 1, 0)
        final_noise = schedule.noises[
            min(schedule_index, len(schedule.noises) - 1)
        ]
        final_thrd = schedule.thrds[
            min(schedule_index, len(schedule.thrds) - 1)
        ]
        discarded = numpy.asarray(
            list(driver._dmrg.discarded_weights), dtype=float
        )
        final_discarded = float(discarded[-1]) if discarded.size else numpy.nan
        converged = bool(
            nsweep >= 2
            and numpy.isfinite(energy_change)
            and energy_change <= self.tol
            and final_noise == 0.0
        )
        return {
            "converged": converged,
            "sweeps": nsweep,
            "sweep_energies": [row.copy() for row in history],
            "energy_change": energy_change,
            "discarded_weight": final_discarded,
            "max_discarded_weight": (
                float(numpy.max(numpy.abs(discarded)))
                if discarded.size else numpy.nan
            ),
            "local_squared_residual_threshold": final_thrd,
        }

    def _record_convergence(self, root_runs=None):
        if root_runs is None:
            root_runs = [self._capture_dmrg_run(self.driver)]
        self.converged = all(run["converged"] for run in root_runs)
        energy_change = max(run["energy_change"] for run in root_runs)
        final_discarded = max(
            (run["discarded_weight"] for run in root_runs),
            default=numpy.nan,
        )
        max_discarded = max(
            (run["max_discarded_weight"] for run in root_runs),
            default=numpy.nan,
        )
        final_thrd = max(
            run["local_squared_residual_threshold"] for run in root_runs
        )
        sweep_energies = (
            root_runs[0]["sweep_energies"]
            if len(root_runs) == 1
            else [run["sweep_energies"] for run in root_runs]
        )
        self.convergence_info = {
            "converged": self.converged,
            "sweeps": max(run["sweeps"] for run in root_runs),
            "sweep_energies": sweep_energies,
            "energy_change": energy_change,
            "discarded_weight": final_discarded,
            "max_discarded_weight": max_discarded,
            "local_squared_residual_threshold": final_thrd,
            "local_residual_bound": math.sqrt(final_thrd),
            "davidson_max_iter": self.dav_max_iter,
            "bond_dimension": max(ket.info.bond_dim for ket in self.kets),
            "canonical_forms": [ket.canonical_form for ket in self.kets],
            "npdm_site_type": self.npdm_site_type,
            "npdm_cutoff": self.npdm_cutoff,
            "scratch": self._scratch,
            "root_runs": root_runs,
            "root_strategy": (
                "state-averaged-multimps"
                if len(self.kets) > 1
                else "state-averaged"
            ),
        }

    def _require_run(self):
        if self.driver is None or self.kets is None:
            raise RuntimeError("DMRG must be run before requesting density matrices")

    def _resolve_state(self, state):
        self._require_run()
        if state is None:
            if isinstance(self.ci, list):
                raise ValueError("a root index or MPS is required for a multi-root result")
            return self.ci
        if isinstance(state, (int, numpy.integer)):
            root = int(state)
            if not 0 <= root < len(self.kets):
                raise IndexError("DMRG root index out of range")
            return self.kets[root]
        if isinstance(state, (list, tuple)):
            raise TypeError("pass one MPS/root; state averaging is handled by PySCF")
        return state

    def _check_rdm_problem(self, norb, nelec):
        if norb is not None and int(norb) != self.ncas:
            raise ValueError("RDM norb does not match the converged DMRG problem")
        if nelec is not None and _electron_number(nelec) != self.nelecas:
            raise ValueError("RDM nelec does not match the converged DMRG problem")

    def make_rdm1(self, state=None, norb=None, nelec=None, **kwargs):
        """Return ``dm1[p,q] = <a_p^dagger a_q>``."""
        return self.make_rdm12(state, norb, nelec, **kwargs)[0]

    def make_rdm2(self, state=None, norb=None, nelec=None, **kwargs):
        """Return ``dm2[p,q,r,s] = <a_p^dagger a_r^dagger a_s a_q>``."""
        return self.make_rdm12(state, norb, nelec, **kwargs)[1]

    def make_rdm12(self, state=None, norb=None, nelec=None, **_kwargs):
        """Return raw-converted Block2 1- and 2-RDMs without projection."""
        self._check_rdm_problem(norb, nelec)
        ket = self._resolve_state(state)
        key = id(ket)
        if key not in self._rdm_cache:
            raw1 = self.driver.get_1pdm(
                ket,
                site_type=self.npdm_site_type,
                cutoff=self.npdm_cutoff,
            )
            raw2 = self.driver.get_2pdm(
                ket,
                site_type=self.npdm_site_type,
                cutoff=self.npdm_cutoff,
            )
            dm1 = block2_rdm1(raw1)
            dm2 = block2_rdm2(raw2)
            contraction = numpy.einsum("pqrr->pq", dm2)
            self.rdm_diagnostics[key] = {
                "trace_error": float(abs(numpy.trace(dm1) - self.nelecas)),
                "contraction_error": float(
                    numpy.max(abs(contraction - (self.nelecas - 1) * dm1))
                ),
                "dm1_hermiticity": float(numpy.max(abs(dm1 - dm1.T.conj()))),
                "dm2_hermiticity": float(
                    numpy.max(abs(dm2.conj() - dm2.transpose(1, 0, 3, 2)))
                ),
                "creation_antisymmetry": float(
                    numpy.max(abs(dm2 + dm2.transpose(2, 1, 0, 3)))
                ),
                "annihilation_antisymmetry": float(
                    numpy.max(abs(dm2 + dm2.transpose(0, 3, 2, 1)))
                ),
                "projection_change": 0.0,
            }
            self._rdm_cache[key] = (dm1, dm2)
            self._refresh_kramers_results()
        return self._rdm_cache[key]

    def _refresh_kramers_results(self):
        """Analyze a complete cached root manifold without forcing NPDM I/O."""
        adapter = self.kramers_adapter
        if adapter is None or self.kets is None or len(self.kets) <= 1:
            return
        keys = [id(ket) for ket in self.kets]
        if not all(key in self._rdm_cache for key in keys):
            return
        dm1s = [self._rdm_cache[key][0] for key in keys]
        dm2s = [self._rdm_cache[key][1] for key in keys]
        weights = getattr(self, "weights", None)
        adapter.analyze(
            numpy.asarray(self.e_tot),
            dm1s,
            dm2s,
            weights=weights,
            overlap=self.root_overlap,
            projected_hamiltonian=self.projected_hamiltonian,
        )
        self.kramers_diagnostics = dict(adapter.diagnostics)

    def make_kramers_pair_rdm12(self, pair=0):
        """Return one validated equal-weight Kramers-pair 1-/2-RDM.

        Raw individual-root NPDMs are always computed and checked first.  The
        returned tensors are projected only if projection was explicitly
        enabled on :meth:`kramers_restricted` and the raw residual passed its
        configured gate.
        """
        if self.kramers_adapter is None:
            raise RuntimeError("Kramers mode is not enabled")
        self._require_run()
        for root in range(len(self.kets)):
            self.make_rdm12(root, self.ncas, self.nelecas)
        result = self.kramers_adapter.pair_results[int(pair)]
        return result.dm1, result.dm2

    def make_kramers_manifold_rdm12(self, manifold=0):
        """Return one validated equal-weight degenerate-manifold 1-/2-RDM.

        For an isolated doublet this is identical to
        :meth:`make_kramers_pair_rdm12`.  For a higher exact degeneracy it is
        the basis-invariant density of the complete manifold; arbitrary
        numerical roots inside that space are intentionally not assigned to
        non-unique Kramers pairs.
        """
        if self.kramers_adapter is None:
            raise RuntimeError("Kramers mode is not enabled")
        self._require_run()
        for root in range(len(self.kets)):
            self.make_rdm12(root, self.ncas, self.nelecas)
        result = self.kramers_adapter.manifold_results[int(manifold)]
        return result.dm1, result.dm2

    def kramers_root_space_rdm1(self, pair=0):
        """Return ``<i|p^+q|j>`` for both roots of one Kramers pair."""
        if self.kramers_adapter is None:
            raise RuntimeError("Kramers mode is not enabled")
        self.make_kramers_pair_rdm12(pair)
        roots = self.kramers_adapter.root_pairs[int(pair)]
        root_space = numpy.empty(
            (2, 2, self.ncas, self.ncas), dtype=numpy.complex128
        )
        for i, bra in enumerate(roots):
            for j, ket in enumerate(roots):
                if i == j:
                    root_space[i, j] = self.make_rdm1(
                        bra, self.ncas, self.nelecas
                    )
                else:
                    root_space[i, j] = self.trans_rdm1(
                        bra, ket, self.ncas, self.nelecas
                    )
        return root_space

    def canonical_kramers_root_space_rdm1(self, pair=0):
        """Remove arbitrary unitary root mixing and transition phases."""
        return self.kramers_adapter.canonicalize_root_space(
            self.kramers_root_space_rdm1(pair)
        )

    def trans_rdm1(self, cibra, ciket, norb=None, nelec=None, **_kwargs):
        """Return ``<bra|a_p^dagger a_q|ket>`` for two DMRG roots."""
        self._check_rdm_problem(norb, nelec)
        bra = self._resolve_state(cibra)
        ket = self._resolve_state(ciket)
        raw = self.driver.get_trans_1pdm(
            bra,
            ket,
            site_type=self.npdm_site_type,
            cutoff=self.npdm_cutoff,
        )
        return block2_transition_rdm1(raw)

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info("")
        log.info("******** DMRGCI flags ********")
        log.info("ncas                         = %s", self.ncas)
        log.info("nelecas                      = %s", self.nelecas)
        log.info("nroots                       = %d", self.nroots)
        log.info("schedule mode                = %s", self.schedule_mode)
        log.info("schedule anchor sweeps       = %s", self.schedule_sweeps)
        log.info("schedule anchor M            = %s", self.schedule_bond_dims)
        log.info("schedule anchor thresholds   = %s", self.schedule_thrds)
        log.info("schedule anchor noises       = %s", self.schedule_noises)
        log.info("bond dimensions              = %s", self.bond_dims)
        log.info("noises                       = %s", self.noises)
        log.info("Davidson squared residuals   = %s", self.thrds)
        log.info("two-site to one-site sweep   = %s", self.twosite_to_onesite)
        log.info("restart / scheduled restart  = %s / %s", self.restart, self._restart)
        log.info("restart switch tolerance     = %g", self.dmrg_switch_tol)
        log.info("restart sweeps               = %d", self.restart_sweeps)
        log.info("schedule noise scale         = %g", self.schedule_noise_scale)
        log.info("schedule Davidson thrd max   = %s", self.schedule_thrd_max)
        log.info("Davidson max iterations      = %d", self.dav_max_iter)
        log.info("noise type                   = %s", self.noise_type)
        log.info("n_threads                    = %d", self.n_threads)
        log.info("stack memory cap             = %.1f MB", self.stack_memory)
        log.info("scratch parent               = %s", self.scratch)
        log.info("keep scratch                 = %s", self.keep_scratch)
        log.info("checkpoint directory         = %s", self.checkpoint_dir)
        log.info("resume checkpoint            = %s", self.resume)
        log.info("checkpoint each sweep        = %s", self.checkpoint_per_sweep)
        log.info("energy tolerance             = %g", self.tol)
        log.info("maximum sweeps               = %d", self.n_sweeps)
        log.info("NPDM site type/cutoff        = %d / %g", self.npdm_site_type, self.npdm_cutoff)
        log.info("Kramers result adapter       = %s", self.kramers_adapter is not None)
        if self.kramers_adapter is not None:
            log.info(
                "Kramers E/RDM tolerances      = %g / %g",
                self.kramers_adapter.energy_tolerance,
                self.kramers_adapter.residual_tolerance,
            )
            log.info(
                "Kramers RDM projection        = %s",
                self.kramers_adapter.project,
            )
        log.info("")
        return self
