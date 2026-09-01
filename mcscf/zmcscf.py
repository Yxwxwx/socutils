import scipy
import numpy
from functools import reduce
from pyscf import __config__
from pyscf.lib import logger

from socutils.scf import spinor_hf
from socutils.mcscf import zcasbase, zcasci
# mcscf_superci is imported lazily inside superci() to avoid a circular import
# (zmc_superci imports zmcscf).
try:
    from socutils.lib import zquatev
except ImportError:
    zquatev = None

def eig(h, irrep=None):
    if irrep is None:
        e, c = scipy.linalg.eigh(h)
    else:
        ir_set = numpy.unique(irrep)
        e = numpy.zeros(h.shape[1])
        c = numpy.zeros(h.shape, dtype=complex)
        for ir in ir_set:
            ir_idx = numpy.where(irrep == ir)[0]
            print(ir_idx)
            hi = h[numpy.ix_(ir_idx, ir_idx)]
            ei, ci = scipy.linalg.eigh(hi)
            e[ir_idx] = ei
            c[numpy.ix_(ir_idx,ir_idx)] = ci
    return e, c

def expmat(a):
    return scipy.linalg.expm(a)

def _fake_h_for_fast_casci(casscf, mo, eris):
    mc = casscf.view(zcasci.CASCI)
    mc.mo_coeff = mo

    if eris is None:
        return mc

    mc.get_h2eff = lambda *args: eris.aaaa

    # Precompute core JK from eris to avoid redundant get_jk in CASCI kernel
    if hasattr(eris, 'get_jk'):
        ncore = casscf.ncore
        ncas = casscf.ncas
        nocc = ncore + ncas
        mo_core = mo[:, :ncore]
        mo_cas = mo[:, ncore:nocc]
        core_occ = numpy.zeros(mo.shape[1])
        core_occ[:ncore] = 1
        dm_core_ao = numpy.dot(mo_core, mo_core.T.conj())
        vj_c, vk_c = eris.get_jk(dm_core_ao, mo_coeff=mo, mo_occ=core_occ)
        corevhf = vj_c - vk_c
        hcore = casscf.get_hcore()
        h1eff = reduce(numpy.dot, (mo_cas.T.conj(), hcore + corevhf, mo_cas))
        energy_core = casscf.energy_nuc()
        energy_core += numpy.einsum('ij,ji', dm_core_ao, hcore)
        energy_core += numpy.einsum('ij,ji', dm_core_ao, corevhf) * 0.5
        mc.get_h1eff = lambda *args, h1=h1eff, ec=energy_core: (h1, ec)

    return mc

def get_fock(mc, mo_coeff=None, ci=None, eris=None, casdm1=None, verbose=None):
    if ci is None: ci = mc.ci
    if mo_coeff is None: mo_coeff = mc.mo_coeff
    nmo = mo_coeff.shape[1]
    ncore = mc.ncore
    ncas = mc.ncas
    nocc = ncore + ncas
    nelecas = mc.nelecas

    if casdm1 is None:
        casdm1 = mc.fcisolver.make_rdm1(ci, ncas, nelecas)
    dm_core = numpy.dot(mo_coeff[:,:ncore], mo_coeff[:,:ncore].conj().T)
    mocas = mo_coeff[:,ncore:nocc]
    # Spinor FCI/Block2 use casdm1[p,q] = <a_p^+ a_q>, while PySCF's
    # get_jk consumes the covariant density with the annihilation index first.
    # The transpose is invisible for real orbitals but essential for a general
    # complex spinor density.  This is the same convention used by the
    # Super-CIPT generalized-Fock path in zmc_supercipt.py.
    dm = dm_core + reduce(numpy.dot, (mocas, casdm1.T, mocas.conj().T))
    vj, vk = mc._scf.get_jk(mc.mol, dm)
    fock = mc.get_hcore() + vj - vk
    return fock

def canonicalize(
    mc,
    mo_coeff=None,
    ci=None,
    eris=None,
    sort=False,
    cas_natorb=False,
    casdm1=None,
    verbose=logger.NOTE,
):
    """Semicanonicalize converged CASSCF core and virtual orbitals.

    This follows the post-CASSCF meaning of PySCF's ``canonicalization``
    option.  The generalized Fock matrix is built from the converged active
    one-particle density, then diagonalized independently in the inactive-core
    and external-virtual spaces.  The active orbitals and ``ci`` object are
    deliberately left unchanged, which keeps an exact-CI vector or DMRG MPS
    tied to the same active-orbital basis.

    ``cas_natorb=True`` is not supported here.  In socutils, active natural
    orbitals are generated transactionally by the Super-CI macroiterations,
    which rerun the active-space solver after changing that basis.
    """
    log = logger.new_logger(mc, verbose)
    if mo_coeff is None:
        mo_coeff = mc.mo_coeff
    if ci is None:
        ci = mc.ci
    if cas_natorb:
        raise NotImplementedError(
            "post-CASSCF active-orbital canonicalization is not supported; "
            "set mc.natorb=True before running Super-CI so the CI/MPS is "
            "reoptimized in every changed active basis"
        )
    if casdm1 is None:
        casdm1 = mc.fcisolver.make_rdm1(ci, mc.ncas, mc.nelecas)

    mo_input = numpy.asarray(mo_coeff)
    if mo_input.ndim != 2 or not numpy.issubdtype(mo_input.dtype, numpy.number):
        raise ValueError("mo_coeff must be a numeric two-dimensional array")
    if not numpy.all(numpy.isfinite(mo_input)):
        raise ValueError("mo_coeff contains non-finite values")
    casdm1 = numpy.asarray(casdm1)
    if casdm1.shape != (mc.ncas, mc.ncas):
        raise ValueError(
            "casdm1 must have shape (%d, %d), got %s"
            % (mc.ncas, mc.ncas, casdm1.shape)
        )
    if not numpy.all(numpy.isfinite(casdm1)):
        raise ValueError("casdm1 contains non-finite values")

    ncore = mc.ncore
    nocc = ncore + mc.ncas
    nmo = mo_input.shape[1]
    mo_coeff1 = numpy.array(mo_input, dtype=numpy.complex128, copy=True)
    core_density_before = mo_coeff1[:, :ncore].dot(
        mo_coeff1[:, :ncore].T.conj()
    )
    active_before = mo_coeff1[:, ncore:nocc].copy()
    virtual_projector_before = mo_coeff1[:, nocc:].dot(
        mo_coeff1[:, nocc:].T.conj()
    )

    fock_ao = numpy.asarray(
        mc.get_fock(mo_input, ci, eris, casdm1, verbose),
        dtype=numpy.complex128,
    )
    if fock_ao.shape != (mo_input.shape[0],) * 2:
        raise ValueError(
            "generalized AO Fock matrix has shape %s, expected (%d, %d)"
            % (fock_ao.shape, mo_input.shape[0], mo_input.shape[0])
        )
    if not numpy.all(numpy.isfinite(fock_ao)):
        raise ValueError("generalized AO Fock matrix contains non-finite values")
    fock_hermiticity_error = float(
        numpy.max(abs(fock_ao - fock_ao.T.conj()), initial=0.0)
    )
    fock_ao = (fock_ao + fock_ao.T.conj()) * 0.5
    fock_mo_before = reduce(
        numpy.dot, (mo_coeff1.T.conj(), fock_ao, mo_coeff1)
    )
    fock_mo_before = (fock_mo_before + fock_mo_before.T.conj()) * 0.5
    mo_energy = numpy.diag(fock_mo_before).real.copy()

    mask = numpy.ones(nmo, dtype=bool)
    frozen = getattr(mc, "frozen", None)
    if frozen is not None:
        if isinstance(frozen, (int, numpy.integer)):
            if frozen < 0 or frozen > nmo:
                raise ValueError("frozen integer must be between 0 and nmo")
            mask[:frozen] = False
        else:
            frozen_indices = numpy.asarray(frozen)
            if frozen_indices.ndim != 1 or not numpy.issubdtype(
                frozen_indices.dtype, numpy.integer
            ):
                raise ValueError("frozen must contain one-dimensional integer indices")
            if numpy.any(frozen_indices < -nmo) or numpy.any(frozen_indices >= nmo):
                raise IndexError("frozen contains an out-of-range orbital index")
            mask[frozen_indices] = False
    core_idx = numpy.where(mask[:ncore])[0]
    virtual_idx = numpy.where(mask[nocc:])[0] + nocc

    from socutils.mcscf.zmc_superci import (
        _kramers_subspace_eigh,
        _resolve_kramers_mode,
    )

    kramers = _resolve_kramers_mode(
        mc, getattr(mc, "orbital_symmetry", None)
    )

    def _subspace_eigh(fock, orbitals):
        mf = mc._scf
        if kramers:
            # Identify the actual partners rather than assuming adjacent pairs.
            return _kramers_subspace_eigh(mc, fock, orbitals)
        if isinstance(mf, spinor_hf.SymmSpinorSCF):
            return mf.eig(fock, mo=orbitals)
        return scipy.linalg.eigh(fock)

    def _diag_subfock(indices, label):
        indices = numpy.asarray(indices, dtype=int)
        if indices.size == 0:
            return
        orbitals = mo_coeff1[:, indices]
        fock = reduce(numpy.dot, (orbitals.T.conj(), fock_ao, orbitals))
        fock = (fock + fock.T.conj()) * 0.5
        if indices.size == 1:
            energies = numpy.array([fock[0, 0].real])
            rotation = numpy.ones((1, 1), dtype=numpy.complex128)
        else:
            energies, rotation = _subspace_eigh(fock, orbitals)
            energies = numpy.asarray(energies).real
            rotation = numpy.asarray(rotation, dtype=numpy.complex128)
        if sort and energies.size > 1:
            order = numpy.argsort(energies.round(9), kind="mergesort")
            energies = energies[order]
            rotation = rotation[:, order]
        unitary_error = float(
            numpy.max(
                abs(rotation.T.conj().dot(rotation) - numpy.eye(indices.size)),
                initial=0.0,
            )
        )
        if unitary_error > 1e-8:
            raise RuntimeError(
                "%s canonicalization returned a nonunitary rotation: %.3e"
                % (label, unitary_error)
            )
        mo_coeff1[:, indices] = orbitals.dot(rotation)
        mo_energy[indices] = energies

    _diag_subfock(core_idx, "core")
    _diag_subfock(virtual_idx, "virtual")

    fock_mo_after = reduce(
        numpy.dot, (mo_coeff1.T.conj(), fock_ao, mo_coeff1)
    )
    fock_mo_after = (fock_mo_after + fock_mo_after.T.conj()) * 0.5
    diagonal_after = numpy.diag(fock_mo_after).real
    mo_energy[ncore:nocc] = diagonal_after[ncore:nocc]

    def _offdiagonal_norm(matrix, indices):
        indices = numpy.asarray(indices, dtype=int)
        if indices.size < 2:
            return 0.0
        block = matrix[numpy.ix_(indices, indices)]
        return float(numpy.linalg.norm(block - numpy.diag(numpy.diag(block))))

    overlap = numpy.asarray(mc._scf.get_ovlp())
    orthonormality_error = float(
        numpy.max(
            abs(
                mo_coeff1.T.conj().dot(overlap).dot(mo_coeff1)
                - numpy.eye(nmo)
            ),
            initial=0.0,
        )
    )
    active_change = float(
        numpy.max(abs(mo_coeff1[:, ncore:nocc] - active_before), initial=0.0)
    )
    core_density_after = mo_coeff1[:, :ncore].dot(
        mo_coeff1[:, :ncore].T.conj()
    )
    virtual_projector_after = mo_coeff1[:, nocc:].dot(
        mo_coeff1[:, nocc:].T.conj()
    )
    core_density_change = float(
        numpy.max(abs(core_density_after - core_density_before), initial=0.0)
    )
    virtual_projector_change = float(
        numpy.max(
            abs(virtual_projector_after - virtual_projector_before),
            initial=0.0,
        )
    )
    energy_diagonal_error = float(
        numpy.max(abs(mo_energy - diagonal_after), initial=0.0)
    )
    diagnostics = {
        "enabled": True,
        "active_orbitals_changed": False,
        "active_orbital_change": active_change,
        "ci_object_preserved": True,
        "core_density_change": core_density_change,
        "virtual_projector_change": virtual_projector_change,
        "fock_hermiticity_error": fock_hermiticity_error,
        "orthonormality_error": orthonormality_error,
        "energy_diagonal_error": energy_diagonal_error,
        "core_indices": core_idx.tolist(),
        "virtual_indices": virtual_idx.tolist(),
        "frozen_indices": numpy.where(~mask)[0].tolist(),
        "core_offdiagonal_before": _offdiagonal_norm(fock_mo_before, core_idx),
        "core_offdiagonal_after": _offdiagonal_norm(fock_mo_after, core_idx),
        "virtual_offdiagonal_before": _offdiagonal_norm(
            fock_mo_before, virtual_idx
        ),
        "virtual_offdiagonal_after": _offdiagonal_norm(
            fock_mo_after, virtual_idx
        ),
        "sort": bool(sort),
        "cas_natorb": False,
    }
    mc.canonicalization_diagnostics = diagnostics
    maximum_post_offdiagonal = max(
        diagnostics["core_offdiagonal_after"],
        diagnostics["virtual_offdiagonal_after"],
    )
    if maximum_post_offdiagonal > 1e-8:
        log.warn(
            "CASSCF canonicalization left a %.3e core/virtual Fock "
            "off-diagonal norm",
            maximum_post_offdiagonal,
        )
    log.info(
        "CASSCF canonicalization | core offdiag %.3e -> %.3e | "
        "virtual offdiag %.3e -> %.3e | orthonormality %.3e",
        diagnostics["core_offdiagonal_before"],
        diagnostics["core_offdiagonal_after"],
        diagnostics["virtual_offdiagonal_before"],
        diagnostics["virtual_offdiagonal_after"],
        diagnostics["orthonormality_error"],
    )
    return mo_coeff1, ci, mo_energy


class CASSCF(zcasci.CASCI):

    max_cycle_macro = getattr(__config__, 'mcscf_mc1step_CASSCF_max_cycle_macro', 50)
    irrep=None
    _keys = zcasci.CASCI._keys.union({
        'max_cycle_macro', 'max_stepsize', 'conv_tol', 'conv_tol_grad',
        'freeze_pair', 'canonicalize_', 'superci_solver', 'superci_bfgs',
        'superci_davidson_tol', 'superci_davidson_max_space',
        'superci_davidson_strict', 'superci_diis', 'macro_history',
        'superci_diagnostics', 'superci_metric_diagnostics',
        'cholesky_diagnostics', 'canonicalization_diagnostics',
        'final_orbital_gradient_norm', 'supercipt_level_shift',
        'supercipt_metric_tol', 'supercipt_denominator_tol',
        'supercipt_diis',
        'orbital_symmetry', 'orbital_diis_space',
        'orbital_diis_start_cycle', 'orbital_diis_start_gradient',
        'supercipt_history', 'supercipt_diagnostics',
    })

    def __init__(self, mf_or_mol, ncas, nelecas, ncore=None, frozen=None, cholesky=True):
        zcasbase.CASBase.__init__(self, mf_or_mol, ncas, nelecas, ncore)
        self.frozen = frozen
        self.callback = None
        self._cderi = None
        self.max_stepsize = 0.2
        self.conv_tol = 1e-8
        self.conv_tol_grad = 1e-4
        self.freeze_pair = None
        self.canonicalize_ = False
        self.natorb = False
        self.superci_solver = 'davidson'
        self.superci_bfgs = False
        self.superci_davidson_tol = 1e-8
        self.superci_davidson_max_space = 200
        self.superci_davidson_strict = True
        self.superci_diis = False
        self.macro_history = []
        self.superci_diagnostics = None
        self.superci_metric_diagnostics = None
        self.cholesky_diagnostics = None
        self.canonicalization_diagnostics = None
        self.final_orbital_gradient_norm = None
        self.supercipt_level_shift = 0.0
        self.supercipt_metric_tol = 1e-6
        self.supercipt_denominator_tol = 1e-10
        self.supercipt_diis = False
        self.orbital_symmetry = None
        self.orbital_diis_space = 15
        self.orbital_diis_start_cycle = 3
        self.orbital_diis_start_gradient = 0.02
        self.supercipt_history = []
        self.supercipt_diagnostics = None

    def get_fock(self, mo_coeff=None, ci=None, eris=None, casdm1=None, verbose=None):
        return get_fock(self, mo_coeff, ci, eris, casdm1, verbose)

    canonicalize = canonicalize

    def uniq_var_indices(self, nmo, ncore, ncas, frozen):
        nocc = ncore + ncas
        mask = numpy.zeros((nmo,nmo),dtype=bool)
        mask[ncore:nocc,:ncore] = True
        mask[nocc:,:nocc] = True
        # if self.internal_rotation:
        #     mask[ncore:nocc,ncore:nocc][numpy.tril_indices(ncas,-1)] = True
        # if self.extrasym is not None:
        #     extrasym = numpy.asarray(self.extrasym)
        #     # Allow rotation only if extra symmetry labels are the same
        #     extrasym_allowed = extrasym.reshape(-1, 1) == extrasym
        #     mask = mask * extrasym_allowed
        if self.freeze_pair is not None:
            freeze_pair = self.freeze_pair
            set_i = freeze_pair[0]
            set_j = freeze_pair[1]
            for i in set_i:
                for j in set_j:
                    mask[i,j] = False
                    mask[j,i] = False
        if frozen is not None:
            if isinstance(frozen, (int, numpy.integer)):
                mask[:frozen] = mask[:,:frozen] = False
            else:
                frozen = numpy.asarray(frozen)
                mask[frozen] = mask[:,frozen] = False
        
        if self.irrep is not None: 
            irrep = self.irrep
            for i, iri in enumerate(irrep):
                for j, irj in enumerate(irrep):
                    if iri != irj:
                        mask[i,j] = False
        return mask

    def screen_irrep(self, mat):
        if self.irrep is not None:
            irrep = self.irrep
            for i, iri in enumerate(irrep):
                for j, irj in enumerate(irrep):
                    if iri != irj:
                        mat[i, j] = 0.
        return mat


    def pack_uniq_var(self, mat):
        nmo = self.mo_coeff.shape[1]
        idx = self.uniq_var_indices(nmo, self.ncore, self.ncas, self.frozen)
        ncore = self.ncore
        nocc = self.ncore + self.ncas
        #print(idx[nocc:, :ncore])
        #print(idx[:ncore,fnocc:])
        return self.screen_irrep(mat)[idx]

    # to anti symmetric matrix
    def unpack_uniq_var(self, v):
        nmo = self.mo_coeff.shape[1]
        idx = self.uniq_var_indices(nmo, self.ncore, self.ncas, self.frozen)
        mat = numpy.zeros((nmo,nmo), dtype=complex)
        mat[idx] = v
        all_indices = numpy.arange(nmo)
        if self.irrep is not None:
            irrep = self.irrep
            for i, iri in enumerate(irrep):
                for j, irj in enumerate(irrep):
                    if iri != irj:
                        mat[i, j] = 0.
        #frozen_pair = [[0,1,4,5,6,7,8,9],[10,11,12,13,14,15,16,17]]
        #set_i = frozen_pair[0]
        #set_j = frozen_pair[1]
        #for i in set_i:
        #    for j in set_j:
        #        mat[i,j] = 0.
        #        mat[j,i] = 0.
        
        return mat - mat.T.conj()

    def update_rotate_matrix(self, dx, u0=1):
        dr = self.unpack_uniq_var(dx)
        return numpy.dot(u0, expmat(dr))

    def casci(self, mo_coeff=None, ci0=None, verbose=None):
        mci = self.view(zcasci.CASCI)
        return mci.kernel(mo_coeff, ci0=ci0, verbose=verbose)

    def kernel(self, mo_coeff=None, ci0=None, callback=None):
        '''Optimize the CASSCF orbitals and CI vector.

        This drives the super-CI orbital optimizer directly (see
        :meth:`superci`); it is the CASSCF entry point analogous to PySCF's
        ``mc.kernel()``.
        '''
        return self.superci(mo_coeff, ci0=ci0, callback=callback)

    def superci(
        self,
        mo_coeff=None,
        ci0=None,
        callback=None,
        _kern=None,
        *,
        use_diis=None,
        symm=None,
        diis_space=None,
        diis_start_cycle=None,
        diis_start_gradient=None,
    ):
        '''Super-CI CASSCF orbital optimization.

        The integral representation follows the mean-field object.  An
        attached ``with_df`` (or legacy ``self._cderi``) selects factorized
        integrals; otherwise every macroiteration uses the full four-index
        spinor transformation.  When ``self.canonicalization`` is true, the
        converged core and virtual orbitals are semicanonicalized afterwards
        and the returned ``mo_energy`` contains generalized-Fock energies; the
        active orbitals and CI/MPS are not transformed by this final step.

        Returns:
            Five elements -- total energy, active-space CI energy, the
            active-space FCI coefficients, and the MCSCF canonical orbital
            coefficients and orbital energies.  They are also stored as the
            attributes ``.e_tot``, ``.e_cas``, ``.ci``, ``.mo_coeff`` and
            ``.mo_energy``.
        '''
        # Lazy import to break the zmcscf <-> zmc_superci circular import.
        from socutils.mcscf.zmc_superci import mcscf_superci
        if _kern is None:
            _kern = mcscf_superci
        if zquatev is None:
            raise RuntimeError('zquatev library is required for spinor CASSCF '
                               'orbital optimization')
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        else:  # overwrite self.mo_coeff; it is used by many methods of this class
            self.mo_coeff = mo_coeff
        if callback is None:
            callback = self.callback
        if use_diis is None:
            use_diis = self.superci_diis
        if symm is None:
            symm = self.orbital_symmetry
        if diis_space is None:
            diis_space = self.orbital_diis_space
        if diis_start_cycle is None:
            diis_start_cycle = self.orbital_diis_start_cycle
        if diis_start_gradient is None:
            diis_start_gradient = self.orbital_diis_start_gradient

        self.check_sanity()
        self.dump_flags()

        self.converged, self.e_tot, self.e_cas, self.ci, \
                self.mo_coeff, self.mo_energy = \
                _kern(self, mo_coeff, max_stepsize=self.max_stepsize,
                      conv_tol=self.conv_tol, conv_tol_grad=self.conv_tol_grad,
                      verbose=self.verbose, cderi=self._cderi,
                      bfgs=self.superci_bfgs, solver=self.superci_solver,
                      davidson_maxiter=self.superci_davidson_max_space,
                      davidson_tol=self.superci_davidson_tol,
                      davidson_strict=self.superci_davidson_strict,
                      use_diis=use_diis, symm=symm,
                      diis_space=diis_space,
                      diis_start_cycle=diis_start_cycle,
                      diis_start_gradient=diis_start_gradient,
                      callback=callback)
        logger.note(self, 'CASSCF energy = %#.15g', self.e_tot)
        self._finalize()
        return self.e_tot, self.e_cas, self.ci, self.mo_coeff, self.mo_energy

    def supercipt(
        self,
        mo_coeff=None,
        ci0=None,
        callback=None,
        _kern=None,
        *,
        use_diis=None,
        diis_space=None,
        diis_start_cycle=None,
        diis_start_gradient=None,
    ):
        '''Perturbative Super-CI CASSCF orbital optimization.

        This is the Guo--Dutta two-component Super-CIPT optimizer.  It is an
        explicit alternative to :meth:`superci`; calling :meth:`kernel` keeps
        using the validated full Super-CI/Davidson optimizer.

        General and Kramers-restricted complex spinors, exact CI, state
        averages and the common Block2 :class:`socutils.dmrg.DMRGCI` solver are
        supported.  Kramers symmetry and the full/factorized integral route
        are inferred from the SCF object and active-space solver.  Orbital
        DIIS is available as an explicit opt-in acceleration.
        '''
        del ci0  # Orbital changes invalidate untransformed CI/MPS guesses.
        from socutils.mcscf.zmc_supercipt import mcscf_supercipt
        if _kern is None:
            _kern = mcscf_supercipt
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        else:
            self.mo_coeff = mo_coeff
        if callback is None:
            callback = self.callback
        if use_diis is None:
            use_diis = self.supercipt_diis
        if diis_space is None:
            diis_space = self.orbital_diis_space
        if diis_start_cycle is None:
            diis_start_cycle = self.orbital_diis_start_cycle
        if diis_start_gradient is None:
            diis_start_gradient = self.orbital_diis_start_gradient

        self.check_sanity()
        self.dump_flags()
        self.converged, self.e_tot, self.e_cas, self.ci, \
                self.mo_coeff, self.mo_energy = _kern(
                    self, mo_coeff,
                    max_stepsize=self.max_stepsize,
                    conv_tol=self.conv_tol,
                    conv_tol_grad=self.conv_tol_grad,
                    max_cycle=self.max_cycle_macro,
                    level_shift=self.supercipt_level_shift,
                    metric_tol=self.supercipt_metric_tol,
                    denominator_tol=self.supercipt_denominator_tol,
                    verbose=self.verbose,
                    cderi=self._cderi,
                    use_diis=use_diis,
                    diis_space=diis_space,
                    diis_start_cycle=diis_start_cycle,
                    diis_start_gradient=diis_start_gradient,
                    callback=callback,
                )
        logger.note(self, 'Super-CIPT CASSCF energy = %#.15g', self.e_tot)
        self._finalize()
        return self.e_tot, self.e_cas, self.ci, self.mo_coeff, self.mo_energy
