"""Block2 DMRG-CI solver for complex relativistic spinor Hamiltonians.

The boundary implemented here follows :mod:`pyscf.fci.fci_dhf_slow` and
:mod:`socutils.fci.zfci`, which are the exact-CI references used by the
spinor CASCI/CASSCF code in this project.  Block2 is used in complex SGF mode:
one Block2 site is one spinor orbital and the only conserved quantum number is
particle number.
"""

import gc
import math
import os
import shutil
import sys
import tempfile

import numpy
from pyscf import lib
from pyscf.lib import StreamObject, logger

from .kramers import KramersResultAdapter


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

    The solver deliberately does not reuse ``ci0``.  A CASSCF macroiteration
    changes the orbital basis, and a previous MPS is not valid in the new basis
    unless it is transformed consistently.
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
        self.bond_dims = [250]
        self.noises = [1e-5] * 4 + [0.0]
        self.thrds = [1e-8] * 4 + [1e-10]
        self.n_sweeps = 20
        self.tol = 1e-9
        self.cutoff = 1e-20
        self.integral_cutoff = 1e-20
        self.dav_type = None
        self.dav_max_iter = 4000
        self.dav_def_max_size = 50
        self.dav_rel_conv_thrd = 0.0
        self.twosite_to_onesite = None
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
        twosite_to_onesite=None,
        random_seed=None,
        npdm_site_type=None,
        npdm_cutoff=None,
    ):
        """Configure the active space, sweep schedule, and solver controls."""
        self.ncas = int(ncas)
        self.nelecas = _electron_number(nelecas)
        self.nroots = int(nroots)
        if self.ncas <= 0 or not 0 <= self.nelecas <= self.ncas:
            raise ValueError("invalid spinor active space")
        if self.nroots <= 0:
            raise ValueError("nroots must be positive")

        if bond_dims is not None:
            self.bond_dims = [int(x) for x in bond_dims]
        if noises is not None:
            self.noises = [float(x) for x in noises]
        if thrds is not None:
            self.thrds = [float(x) for x in thrds]
        if not self.bond_dims or not self.noises or not self.thrds:
            raise ValueError("bond_dims, noises, and thrds must be nonempty")
        if min(self.bond_dims) <= 0 or min(self.noises) < 0 or min(self.thrds) <= 0:
            raise ValueError("invalid DMRG sweep schedule")

        if n_sweeps is not None:
            self.n_sweeps = int(n_sweeps)
        if tol is not None:
            self.tol = float(tol)
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
        if self.n_threads <= 0 or self.stack_memory <= 0:
            raise ValueError("n_threads and stack_memory must be positive")
        return self

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

    def _release_run(self, remove_scratch=True):
        run_scratch = self._scratch
        self._rdm_cache.clear()
        self.rdm_diagnostics.clear()
        self.kramers_diagnostics.clear()
        self.root_overlap = None
        self.projected_hamiltonian = None
        if self.kramers_adapter is not None:
            self.kramers_adapter.clear_results()
        self.ci = None
        self.kets = None
        self._multi_mps = None
        self._active_mpo = None
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
        effective_dav_type = self.dav_type
        effective_twosite_to_onesite = self.twosite_to_onesite
        if (
            nroots > 1
            and int(norb) > 2
            and effective_twosite_to_onesite is None
        ):
            # Retain two two-site sweeps for initial optimization and leave at
            # least one one-site sweep even for deliberately short schedules.
            effective_twosite_to_onesite = 2 if self.n_sweeps >= 3 else 0
        h1_block2, eri_block2 = block2_integrals(h1e, eri, norb)
        ecore_value = _real_energy(ecore)
        if numpy.asarray(ecore_value).ndim != 0:
            raise ValueError("ecore must be a scalar")
        ecore_value = float(ecore_value)
        if ci0 is not None:
            logger.debug(
                self,
                "Ignoring ci0: an MPS is not reused without a validated orbital-basis transform",
            )

        self._release_run(remove_scratch=True)
        self.e_tot = None
        self.e_cas = None
        self.converged = False
        self.convergence_info = {}
        os.makedirs(self.scratch, exist_ok=True)
        run_scratch = tempfile.mkdtemp(prefix="dmrgci_", dir=self.scratch)
        self._scratch = run_scratch
        iprint = 1 if logger.new_logger(self, verbose).verbose >= logger.NOTE else 0
        stack_bytes = self._stack_bytes(max_memory)

        self.dump_flags(verbose=verbose)
        try:
            driver = DMRGDriver(
                stack_mem=stack_bytes,
                scratch=run_scratch,
                clean_scratch=True,
                symm_type=SymmetryTypes.SGFCPX,
                n_threads=self.n_threads,
            )
            driver.bw.b.Random.rand_seed(self.random_seed)
            driver.initialize_system(
                n_sites=int(norb),
                n_elec=nelec,
                orb_sym=[0] * int(norb),
            )
            mpo = driver.get_qc_mpo(
                h1e=h1_block2,
                g2e=eri_block2,
                ecore=0.0,
                cutoff=self.cutoff,
                integral_cutoff=self.integral_cutoff,
                iprint=iprint,
            )
            dmrg_kwargs = {
                "n_sweeps": self.n_sweeps,
                "tol": self.tol,
                "bond_dims": self.bond_dims,
                "noises": self.noises,
                "thrds": self.thrds,
                "iprint": iprint,
                "dav_type": effective_dav_type,
                "dav_max_iter": self.dav_max_iter,
                "dav_def_max_size": self.dav_def_max_size,
                "dav_rel_conv_thrd": self.dav_rel_conv_thrd,
                "cutoff": self.cutoff,
                "twosite_to_onesite": effective_twosite_to_onesite,
                "real_density_matrix": False,
            }
            run_records = []
            ket = driver.get_random_mps(
                tag="GS",
                bond_dim=self.bond_dims[0],
                nroots=nroots,
            )
            if nroots > 1:
                weights = numpy.asarray(
                    getattr(self, "weights", numpy.ones(nroots) / nroots),
                    dtype=float,
                )
                if weights.shape != (nroots,) or not numpy.all(
                    numpy.isfinite(weights)
                ):
                    raise ValueError(
                        "state-average weights must be one finite value per root"
                    )
                if numpy.any(weights <= 0.0) or weights.sum() <= 0.0:
                    raise ValueError("state-average weights must be positive")
                weights = weights / weights.sum()
                ket.weights = driver.bw.VectorFP(weights.tolist())
            energy = driver.dmrg(mpo, ket, **dmrg_kwargs)
            run_records.append(self._capture_dmrg_run(driver))
            if nroots > 1:
                kets = [
                    driver.split_mps(ket, root, tag="KET-%d" % root)
                    for root in range(nroots)
                ]
                ci = kets
                self._multi_mps = ket
            else:
                kets = [ket]
                ci = ket

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
                    10.0 * math.sqrt(min(self.thrds)),
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
            return self.e_tot, self.ci
        except Exception:
            self._release_run(remove_scratch=True)
            raise

    def _capture_dmrg_run(self, driver):
        """Copy convergence data before a subsequent root overwrites it."""
        history = [_history_row(row) for row in driver._dmrg.energies]
        nsweep = len(history)
        if nsweep >= 2:
            energy_change = float(numpy.max(numpy.abs(history[-1] - history[-2])))
        else:
            energy_change = numpy.inf
        schedule_index = max(nsweep - 1, 0)
        final_noise = self.noises[min(schedule_index, len(self.noises) - 1)]
        final_thrd = self.thrds[min(schedule_index, len(self.thrds) - 1)]
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
        log.info("bond dimensions              = %s", self.bond_dims)
        log.info("noises                       = %s", self.noises)
        log.info("Davidson squared residuals   = %s", self.thrds)
        log.info("Davidson max iterations      = %d", self.dav_max_iter)
        log.info("n_threads                    = %d", self.n_threads)
        log.info("stack memory cap             = %.1f MB", self.stack_memory)
        log.info("scratch parent               = %s", self.scratch)
        log.info("keep scratch                 = %s", self.keep_scratch)
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
