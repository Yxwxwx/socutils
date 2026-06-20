#
# Author Xubo Wang <wangxubo0201@outlook.com>
#
# TDA (Tamm-Dancoff) for a two-component (j-adapted spinor) reference.
#
# A spinor mean field is, structurally, a GHF/GKS reference with a single
# complex n2c x n2c MO coefficient matrix.  The linear-response sigma vector
# (pyscf.tdscf.ghf) is basis agnostic -- it only needs mf.gen_response, which
# returns (J - c_x K + f_xc) acting on the (complex) transition densities.  The
# spinor reference supplies that through the GHF response template (bound as
# SpinorSCF.gen_response): J/K from the spinor get_jk, f_xc from the
# 2-component SpinorNumInt2C (which also makes the collinear f_xc work with
# complex transition densities).
#
# Only TDA is exposed here.  TDA sets B=0, so its response matrix A is genuinely
# Hermitian and the Davidson solve (lr_eig.eigh) is robust.  The full RPA/TDDFT
# (Casida) problem is k-Hermitian (non-Hermitian) on a Krein space with an
# indefinite metric [Furche & Chen, J. Chem. Phys. 163, 174104 (2025)]; its
# iterative solution is delicate near eta-neutral roots and is left aside for
# now.
#
# Validated vs pyscf's 2-component GKS-TDA (non-relativistic integrals, no SOC)
# to machine precision (LDA / GGA / hybrid).  With SOC integrals the orbitals
# are genuinely complex; set mf._numint.collinear='mcol' for the spin-flip
# (non-collinear) response.
#

from pyscf import lib
from pyscf.tdscf import ghf as _ghf


class _KernelXCMixin:
    '''Allow the response (A) xc kernel to differ from the ground-state xc.

    Set ``xc_kernel`` to e.g. ``'LDA,VWN'`` to build the TDA response with an
    (A)LDA kernel on top of a GGA/hybrid ground state.  The whole response xc --
    the local f_xc *and* the exact-exchange fraction -- then comes from
    ``xc_kernel`` (so an 'LDA,VWN' kernel carries no HF exchange; keep the exact
    exchange with e.g. ``'0.2*HF + 0.8*LDA, VWN'``), evaluated on the
    ground-state density.  Leave it None to use the ground-state functional.
    '''
    xc_kernel = None

    def kernel(self, *args, **kwargs):
        # _gen_ghf_response returns a vind closure that reads mf.xc lazily (at
        # Davidson time), so the kernel xc must stay set for the whole solve,
        # not just while the response is generated.
        if self.xc_kernel is None:
            return super().kernel(*args, **kwargs)
        with lib.temporary_env(self._scf, xc=self.xc_kernel):
            return super().kernel(*args, **kwargs)

    def cvs(self, core):
        '''Core-valence separation for core-excitation spectra.

        Keep only the ``core`` occupied orbitals active as holes and **freeze
        every other occupied orbital** (all virtuals stay active).  This is the
        inverse selection of the usual frozen-core: it sets ``self.frozen`` to
        the complement of ``core`` within the occupied space and returns self,
        so it chains:

            td = mf.TDA().cvs([0, 1])      # holes restricted to the 1s pair
            es = td.kernel()
            td.xc_kernel = 'LDA,VWN'       # optional: ALDA response kernel
            es_alda = td.kernel()

        ``core`` is a list of occupied spinor-orbital indices (a deep core is a
        Kramers pair, e.g. ``[0, 1]``).  Builds on pyscf's ``frozen`` mechanism
        (get_frozen_mask), so it composes with everything else here.
        '''
        import numpy
        core = set(int(i) for i in numpy.atleast_1d(core))
        occ = numpy.where(self._scf.mo_occ == 1)[0]
        self.frozen = [int(i) for i in occ if i not in core]
        return self

    def _contract_multipole(self, ints, hermi=True, xy=None):
        '''Transition multipole (e.g. dipole) for a spinor (2-component)
        reference -- pyscf's GHF stub raises NotImplementedError.

        ``oscillator_strength`` calls this with the spin-free electric-dipole
        integrals built in the *spherical* AO basis; the spinor MOs live in the
        n2c basis, so rebuild the operator there (int1e_r_spinor) and contract
        with the X (and Y) amplitudes over the active occupied/virtual spinors.
        No factor of two: unlike closed-shell RHF the spinor basis carries no
        spin degeneracy to sum over.
        '''
        import numpy
        if xy is None:
            xy = self.xy
        nstates = len(xy)
        pol_shape = ints.shape[:-2]          # (3,) for the dipole
        mol = self._scf.mol
        n2c = mol.nao_2c()
        # spin-free electric dipole in the spinor AO basis (same (0,0,0) gauge
        # origin as pyscf's int1e_r)
        ints = mol.intor('int1e_r_spinor', comp=3).reshape(-1, n2c, n2c)

        mask = self.get_frozen_mask()
        mo = self._scf.mo_coeff[:, mask]
        occ = self._scf.mo_occ[mask]
        orbo = mo[:, occ == 1]
        orbv = mo[:, occ == 0]
        # MO transition dipole <i|r|a>  (active occ x active vir)
        ints = lib.einsum('xpq,pi,qj->xij', ints, orbo.conj(), orbv)
        pol = numpy.array([numpy.einsum('xij,ij->x', ints, x) for x, y in xy])
        if isinstance(xy[0][1], numpy.ndarray):    # TDHF/RPA has de-excitations
            ymo = numpy.array([numpy.einsum('xij,ij->x', ints, y) for x, y in xy])
            pol = pol + ymo if hermi else pol - ymo
        return pol.reshape((nstates,) + pol_shape)

    def oscillator_strength(self, e=None, xy=None, gauge='length', order=0):
        '''f = (2/3) * dE * |mu|^2.  The spinor transition dipole is complex, so
        use |mu|^2 = mu . mu* -- pyscf's base routine forms mu . mu (real-dipole
        assumption), which underestimates and can even go negative here.
        '''
        import numpy
        if e is None:
            e = self.e
        if gauge == 'length' and order == 0:
            mu = self.transition_dipole(xy)
            return (2. / 3. * numpy.einsum('s,sx,sx->s', e, mu, mu.conj())).real
        return super().oscillator_strength(e=e, xy=xy, gauge=gauge, order=order)


class TDA(_KernelXCMixin, _ghf.TDA):
    pass


CIS = TDA


class _PairedRPA(_ghf.TDHF):
    '''Full RPA/TDDFT solved with the paired-trial-vector Davidson
    (socutils.tdscf.lr_davidson), which represents the indefinite Krein
    metric correctly for complex orbitals.  Experimental -- TDA remains the
    recommended default.'''

    conv_tol = 1e-6

    def kernel(self, x0=None, nstates=None):
        import numpy as np
        from pyscf.lib import logger
        from socutils.tdscf import lr_davidson
        cpu0 = (logger.process_clock(), logger.perf_counter())
        self.check_sanity()
        self.dump_flags()
        if nstates is None:
            nstates = self.nstates
        else:
            self.nstates = nstates
        log = logger.Logger(self.stdout, self.verbose)

        vind, hdiag = self.gen_vind(self._scf)
        conv, e, zs, nmv = lr_davidson.paired_eig(
            vind, hdiag, nroots=nstates, x0=x0, conv_tol=self.conv_tol,
            max_cycle=self.max_cycle,
            pos_tol=getattr(self, 'positive_eig_threshold', 1e-3),
            verbose=self.verbose > 4)
        log.debug('TDRPA paired Davidson: %d matvecs', nmv)
        self.converged = conv
        self.e = e

        mask = self.get_frozen_mask()
        mo_occ = self._scf.mo_occ[mask]
        nocc = int(np.count_nonzero(mo_occ > 0))
        nvir = mo_occ.size - nocc
        # zs are normalized to |X|^2 - |Y|^2 = 1 already
        self.xy = [(z[:nocc*nvir].reshape(nocc, nvir),
                    z[nocc*nvir:].reshape(nocc, nvir)) for z in zs]
        log.timer('TDRPA (paired Davidson)', *cpu0)
        self._finalize()
        return self.e, self.xy


class TDRPA(_KernelXCMixin, _PairedRPA):
    pass
