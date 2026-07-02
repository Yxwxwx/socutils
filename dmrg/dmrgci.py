"""Block2-based DMRG-CI solver for spinor (relativistic) CASCI.

Implements the FCI solver interface expected by
:class:`socutils.mcscf.zcasci.CASCI` and
:class:`socutils.mcscf.zmcscf.CASSCF`.

Uses the pyblock2 Python binding directly (no FCIDUMP / subprocess).

Standard PySCF usage::

    from socutils.dmrg import dmrgci
    from socutils.mcscf import zcasci

    mc = zcasci.CASCI(mf, ncas, nelec)
    mc.fcisolver = dmrgci.DMRGCI(mol)
    mc.fcisolver.init(ncas=mc.ncas, nelec=mc.nelecas,
                      nroots=2, bond_dims=[250]*4+[500]*4,
                      noises=[1e-4]*4+[1e-5]*4+[0],
                      thrds=[1e-10]*8)
    mc.kernel()
"""

import os
import sys
from pyscf import lib
from pyscf.lib import logger, StreamObject


class DMRGCI(StreamObject):
    """DMRG-CI solver backed by pyblock2.

    Constructor takes only the PySCF Mole object.  DMRG-specific settings
    are configured via :meth:`init`.

    Args:
        mol: PySCF :class:`Mole` object (also used for defaults).

    Saved attributes:
        driver: The Block2 :class:`DMRGDriver` instance (set after run).
        ci: The converged unsplit MPS (analogous to CI vector).
        kets: List of per-root split MPS.
        e_tot:  Total energy (including core), may be array for nroots>1.
    """

    def __init__(self, mol=None):
        if mol is None:
            self.stdout = sys.stdout
            self.verbose = logger.NOTE
            self.max_memory = 2 << 30
        else:
            self.stdout = mol.stdout
            self.verbose = mol.verbose
            self.max_memory = int(mol.max_memory / 1000) << 30

        # --- defaults derived from environment ---
        self.n_threads = int(lib.num_threads())
        self.scratch = os.path.abspath(lib.param.TMPDIR) + "/dmrgci"

        # --- DMRG parameters (populated by init()) ---
        self.ncas = None
        self.nelecas = None
        self.nroots = 1
        self.M = None
        self.bond_dims = None
        self.noises = None
        self.thrds = None
        self.n_sweeps = 30
        self.tol = 1e-8

        # --- results ---
        self.driver = None
        self.ci = None  # unsplit MPS
        self.kets = None  # per-root split MPS
        self.e_tot = None
        self.e_cas = None
        self.converged = False

        self._keys = set(self.__dict__.keys())

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

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
        n_threads=None,
    ):
        """Configure DMRG parameters.

        Args:
            ncas: number of active spinor orbitals.
            nelecas: number of active electrons.
            nroots: number of states.
            bond_dims: bond dimension schedule.
            noises: noise schedule.
            thrds: truncation threshold schedule.
            tol: DMRG convergence tolerance.
            scratch: Block2 scratch directory (default from env).
            n_threads: OpenMP threads (default from OMP_NUM_THREADS).
        """
        self.ncas = ncas
        self.nelecas = nelecas
        self.nroots = nroots
        if bond_dims is None or noises is None or thrds is None:
            raise ValueError("bond_dims, noises, and thrds must be specified")

        self.bond_dims = bond_dims
        self.noises = noises
        self.thrds = thrds

        if tol is not None:
            self.tol = tol
        if n_sweeps is not None:
            self.n_sweeps = n_sweeps
        self.M = max(bond_dims) if bond_dims is not None else None
        if scratch is not None:
            self.scratch = scratch
        if n_threads is not None:
            self.n_threads = n_threads
        return self

    # ------------------------------------------------------------------
    # Kernel
    # ------------------------------------------------------------------

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
        **_kwargs,
    ):
        """Run DMRG-CI."""
        import tempfile
        from pyblock2.driver.core import DMRGDriver, SymmetryTypes

        scratch = tempfile.mkdtemp(prefix="dmrgci_", dir=self.scratch)
        self._scratch = scratch
        bond_dims = self.bond_dims
        noises = self.noises
        thrds = self.thrds
        iprint = 1 if (verbose or 0) >= logger.NOTE else 0
        n_sweeps = self.n_sweeps
        tol = self.tol
        if max_memory is None:
            max_memory = self.max_memory
        max_memory = int(max_memory / 1000) << 30

        # C1 symmetry for relativistic (SOC) calculations
        orb_sym = [0] * norb

        self.dump_flags(verbose=verbose)
        driver = DMRGDriver(
            stack_mem=max_memory,
            scratch=scratch,
            symm_type=SymmetryTypes.SGFCPX,
            n_threads=self.n_threads,
        )
        driver.initialize_system(
            n_sites=norb,
            n_elec=nelec,
            orb_sym=orb_sym,
        )

        mpo = driver.get_qc_mpo(h1e=h1e, g2e=eri, ecore=ecore, iprint=iprint)
        ket = driver.get_random_mps(
            tag="GS",
            bond_dim=int(min(bond_dims)),
            nroots=self.nroots,
        )

        energy = driver.dmrg(
            mpo,
            ket,
            n_sweeps=n_sweeps,
            tol=tol,
            bond_dims=bond_dims,
            noises=noises,
            thrds=thrds,
            iprint=iprint,
        )

        # Split MPS for multi-root access
        if self.nroots > 1:
            self.kets = [
                driver.split_mps(ket, ir, tag="KET-%d" % ir) for ir in range(ket.nroots)
            ]
            self.ci = self.kets  # list of MPS (compatible with StateAverageFCISolver)
            fcivec = self.kets
        else:
            self.kets = [ket]
            self.ci = ket
            fcivec = ket

        self.driver = driver
        self.e_tot = energy
        self.e_cas = (
            (energy - ecore)
            if isinstance(energy, (float, int))
            else [e - ecore for e in energy]
        )
        self.converged = True

        return energy, fcivec

    # ------------------------------------------------------------------
    # Density matrices
    # ------------------------------------------------------------------

    def _require_run(self):
        if self.kets is None:
            raise RuntimeError("DMRG must be run before calling RDM methods.")

    def _resolve_state(self, state):
        """Return the split MPS for *state*.

        *state* may be an integer index (CASCI) or an MPS object
        passed directly by the CASSCF orbital optimizer.
        """
        if isinstance(state, int):
            return self.kets[state]
        return state  # already an MPS

    def make_rdm1(self, state, _norb, _nelec, **kwargs):
        """Active-space 1-RDM (complex)."""
        return self.make_rdm12(state)[0]

    def make_rdm2(self, state, _norb, _nelec, **kwargs):
        """Active-space 2-RDM (complex)."""
        return self.make_rdm12(state)[1]

    def make_rdm12(self, state, _norb=None, _nelec=None, **_kwargs):
        """Active-space 1- and 2-RDMs.

        Block2 NPDM convention: ``dm2[i, j, b, a]``
        (creation, creation, annihilation-reversed).

        PySCF convention: ``rdm2[p, q, r, s]``
        (creation, annihilation, creation, annihilation).

        Transpose (0,3,1,2) maps between the two.
        """
        self._require_run()
        ket_s = self._resolve_state(state)
        rdm1 = self.driver.get_1pdm(ket_s)
        rdm2 = self.driver.get_2pdm(ket_s).transpose(0, 3, 1, 2)
        dm1 = (rdm1 + rdm1.T.conj()) * 0.5
        Gamma = rdm2.transpose(0, 2, 1, 3)
        tmp = (
            Gamma
            - Gamma.transpose(1, 0, 2, 3)  # -Γ_QPRS
            - Gamma.transpose(0, 1, 3, 2)  # -Γ_PQSR
            + Gamma.transpose(1, 0, 3, 2)  # +Γ_QPSR
            + Gamma.transpose(2, 3, 0, 1).conj()  # +Γ_RSPQ^*
            - Gamma.transpose(3, 2, 0, 1).conj()  # -Γ_SRPQ^*
            - Gamma.transpose(2, 3, 1, 0).conj()  # -Γ_RSQP^*
            + Gamma.transpose(3, 2, 1, 0).conj()  # +Γ_SRQP^*
        ) * 0.125
        dm2 = tmp.transpose(0, 2, 1, 3)
        return dm1, dm2

    def trans_rdm1(self, state_i, state_j, _norb=None, _nelec=None, **_kwargs):
        """Transition 1-RDM between two roots."""
        self._require_run()
        return self.driver.get_trans_1pdm(
            self._resolve_state(state_i), self._resolve_state(state_j)
        )

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info("")
        log.info("******** DMRGCI flags ********")
        log.info("ncas      = %d", self.ncas)
        log.info("nelecas   = %d", self.nelecas)
        log.info("nroots    = %d", self.nroots)
        log.info("M         = %d", self.M)
        log.info("n_threads = %d", self.n_threads)
        log.info("stack_mem = %d GB", self.max_memory >> 30)
        log.info("scratch   = %s", self.scratch)
        log.info("tol       = %g", self.tol)
        log.info("n_sweeps  = %d", self.n_sweeps)
        log.info("")
        return self
