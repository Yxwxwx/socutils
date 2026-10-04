"""UNF3-specific, AVAS-like preparation of CAS(16e,40 spinors).

Keep a 6e/20-spinor HF valence space, add the complete occupied U 5d
projection and ten external U 6d projections, and return orbitals ordered
as [inactive | 5d | original valence | 6d additions | external].  The
radial targets come from neutral-U spherical-average scalar-X2C HF in the
molecule's U basis, with [Rn] 5f^3 6d^1 7s^2 fractional occupations.

Usage (the molecular HF calculation must already be available)::

    from socutils.unf3_active_space import make_u_reference, make_unf3_cas

    reference = make_u_reference(mf.mol)  # reuse for all scan points
    ncas, nelecas, mo, info = make_unf3_cas(mf, u_reference=reference, localize=True)
    mc = zmcscf.CASSCF(mf, ncas, nelecas)
    # Configure a fresh DMRGCI solver on mc here.
    mc.second_order(mo_coeff=mo)

This prepares the initial orbitals only; it never rotates a live MPS or
changes the molecular Hamiltonian.  Counts refer to spinors, not spatial
orbitals.  The default valence window is the six highest occupied and
fourteen lowest virtual HF spinors.  This energy window is not chemical
state tracking across crossings: pass explicit zero-based ``base_active``
indices at such points.  SVD phases/degenerate directions are not aligned
between geometries; compare subspaces, not individual orbital columns.
Optional split-PM localizes the sixteen occupied and twenty-four empty
active spinors separately, once before the first DMRG solve.  After it,
5d/valence/6d character is tracked by target projections, not column ranges.

Run the real-HF checks without molecular SCF/DMRG sweeps::

    .venv/bin/python unf3_active_space.py point1/hf.chk point2/hf.chk
"""

from copy import deepcopy

import numpy as np
from pyscf import gto
from pyscf.scf import atom_hf


def _uranium_basis(mol):
    if (mol.cart or mol.has_ecp() or mol._pseudo
            or getattr(mol, 'so_contr', None) is not None
            or np.any(mol._bas[:, gto.KAPPA_OF])):
        raise ValueError('Use all-electron spherical AOs without spinor contractions')
    uranium = [i for i in range(mol.natm) if mol.atom_pure_symbol(i) == 'U']
    if len(uranium) != 1:
        raise ValueError('Exactly one U atom is required')
    iu = uranium[0]
    key = mol.atom_symbol(iu)
    basis = mol._basis[key if key in mol._basis else 'U']
    model = mol._atm[iu, gto.NUC_MOD_OF]
    if model not in (gto.NUC_POINT, gto.NUC_GAUSS):
        raise ValueError('Unsupported U nuclear model')
    zeta = float(mol._env[mol._atm[iu, gto.PTR_ZETA]])
    return iu, basis, zeta


def make_u_reference(mol):
    """Compute reusable spatial U 5d/6d targets in ``mol``'s U AO basis.

    The returned dictionary can be passed unchanged to ``make_unf3_cas``
    at other geometries with the same U basis and nuclear model.  No
    geometry, orbital, or DMRG checkpoint paths are embedded in it.
    """
    _, basis, zeta = _uranium_basis(mol)
    atom = gto.M(atom='U 0 0 0', basis={'U': deepcopy(basis)},
                 unit='Bohr', spin=0, verbose=0, max_memory=mol.max_memory)
    atom.set_nuc_mod(0, zeta)
    ahf = atom_hf.AtomSphAverageRHF(atom)
    ahf.atomic_configuration = deepcopy(ahf.atomic_configuration)
    ahf.atomic_configuration[92] = [14, 30, 31, 17]
    ahf = ahf.x2c()
    ahf.conv_tol, ahf.max_cycle = 1e-10, 200
    ahf.kernel()
    if not ahf.converged:
        raise RuntimeError('Neutral-U spherical-average scalar-X2C HF did not converge')

    # Resolve angular character with the AO metric, then select radial
    # roots 3 and 4 of l=2 (3d, 4d, 5d, 6d, ...).
    loc = atom.ao_loc_nr()
    drows = np.concatenate([np.arange(loc[i], loc[i + 1])
                            for i in range(atom.nbas) if atom.bas_angular(i) == 2])
    c = ahf.mo_coeff
    weight = np.einsum('pi,pi->i', c.conj(),
                       ahf.get_ovlp()[:, drows] @ c[drows]).real
    dmo = np.flatnonzero(weight > 1 - 1e-8)
    dmo = dmo[np.argsort(ahf.mo_energy[dmo], kind='stable')]
    if len(dmo) < 20 or len(dmo) % 5:
        raise RuntimeError('Could not resolve complete atomic d radial shells')
    shells = (dmo[10:15], dmo[15:20])
    for shell, occupation in zip(shells, (2.0, 0.2)):
        if (np.ptp(ahf.mo_energy[shell]) > 1e-8
                or not np.allclose(ahf.mo_occ[shell], occupation, atol=1e-8, rtol=0)):
            raise RuntimeError('Atomic 5d/6d degeneracy or occupation is inconsistent')
    return {'basis': deepcopy(basis), 'nuclear_zeta': zeta,
            '5d': c[:, shells[0]].copy(), '6d': c[:, shells[1]].copy(),
            'atomic_energy': float(ahf.e_tot)}


def localize_unf3_cas(mf, mo_coeff, *, iprint=0):
    """Apply one-time active split-PM to a prepared HF CAS(16e,40s) basis.

    Use the complex-spinor PM implementation in ``lo.uno_spinor``. Core
    and external columns are unchanged. Input must still separate HF
    occupied/empty directions; this is not for a live optimized MPS.
    Return a new coefficient array, preserving the HF density and CAS.
    """
    from socutils.lo.uno_spinor import pmloc

    c = np.asarray(mo_coeff)
    s = mf.get_ovlp()
    if (c.ndim != 2 or c.shape[0] != mf.mol.nao_2c() or c.shape[1] < 150
            or mf.mol.nelectron != 126 or not np.isfinite(c).all()
            or not np.allclose(c.conj().T @ s @ c, np.eye(c.shape[1]), atol=1e-7, rtol=0)):
        raise ValueError('Expected an orthonormal UNF3 CAS(16e,40s) coefficient matrix')
    t = mf.mo_coeff.conj().T @ s @ c
    occ = np.asarray(mf.mo_occ) @ abs(t)**2
    if (not np.allclose(occ, np.rint(occ), atol=1e-7, rtol=0)
            or not np.allclose(occ[:110], 1, atol=1e-7, rtol=0)
            or not np.allclose(occ[150:], 0, atol=1e-7, rtol=0)
            or abs(occ[110:150].sum() - 16) > 1e-7):
        raise ValueError('Split-PM requires separate HF occupied/empty columns and 16 active electrons')
    result = np.array(c, dtype=complex, copy=True)
    for occupied in (True, False):
        indices = 110 + np.flatnonzero((occ[110:150] > 0.5) == occupied)
        error, rotation = pmloc(mf.mol, c[:, indices], iprint=iprint)
        if error:
            raise RuntimeError(f'Active split-PM did not converge (occupied={occupied})')
        result[:, indices] = c[:, indices] @ rotation
    u = c[:, 110:150].conj().T @ s @ result[:, 110:150]
    if not np.allclose(u.conj().T @ u, np.eye(40), atol=1e-7, rtol=0):
        raise RuntimeError('Split-PM changed the active subspace or its orthonormality')
    return result


def make_unf3_cas(mf, *, base_active=None, u_reference=None, localize=False,
                  min_5d_overlap=0.9, min_6d_overlap=1e-6):
    """Return ``(40, 16, mo_coeff, info)`` for a neutral UNF3 spinor HF.

    ``base_active`` optionally gives 20 distinct zero-based HF MO indices
    containing six occupied spinors. Their columns are preserved exactly
    when ``localize=False``; ``localize=True`` applies active split-PM
    within the selected CAS. Otherwise use the frontier HF energy window.

    ``min_5d_overlap`` and ``min_6d_overlap`` bound the smallest singular
    value of the occupied/external target projection, respectively; these
    are amplitudes, not squared AVAS weights.  Failure raises an exception
    instead of changing the CAS size partway through a scan.  ``info``
    includes both spectra and the molecular-AO target spinors for further
    subspace/density analysis.  Neither ``mf`` nor ``u_reference`` is mutated.
    """
    mol = mf.mol
    if (sorted(mol.atom_pure_symbol(i) for i in range(mol.natm))
            != ['F', 'F', 'F', 'N', 'U'] or mol.charge != 0 or mol.nelectron != 126):
        raise ValueError('This prescription requires neutral, all-electron UNF3')
    iu, basis, zeta = _uranium_basis(mol)
    if not (0 < min_5d_overlap <= 1 and 0 < min_6d_overlap <= 1):
        raise ValueError('Projection thresholds must be in (0, 1]')
    c, occ, eps = (np.asarray(getattr(mf, key))
                   for key in ('mo_coeff', 'mo_occ', 'mo_energy'))
    s = np.asarray(mf.get_ovlp())
    nao = mol.nao_2c()
    if (c.ndim != 2 or c.shape[0] != nao or occ.shape != (c.shape[1],)
            or eps.shape != occ.shape or s.shape != (nao, nao)
            or not all(np.isfinite(x).all() for x in (c, occ, eps, s))):
        raise ValueError('Invalid spinor HF orbitals, energies, occupations, or overlap')
    if not np.allclose(s, mol.intor_symmetric('int1e_ovlp_spinor'), atol=1e-10, rtol=0):
        raise ValueError('HF must use the molecule\'s two-component spinor AO representation')
    if not np.allclose(c.conj().T @ s @ c, np.eye(c.shape[1]), atol=1e-7, rtol=0):
        raise ValueError('HF orbitals are not orthonormal')
    if (not np.all(np.isclose(occ, 0, atol=1e-8, rtol=0)
                   | np.isclose(occ, 1, atol=1e-8, rtol=0))
            or abs(occ.sum() - 126) > 1e-7):
        raise ValueError('Expected 126 electrons with integer spinor HF occupations')
    occupied, virtual = np.flatnonzero(occ > 0.5), np.flatnonzero(occ < 0.5)
    if base_active is None:
        base_active = np.concatenate((occupied[np.argsort(eps[occupied], kind='stable')[-6:]],
                                      virtual[np.argsort(eps[virtual], kind='stable')[:14]]))
    active = np.asarray(base_active)
    if (active.shape != (20,) or active.dtype.kind not in 'iu'
            or len(np.unique(active)) != 20 or np.any(active < 0)
            or np.any(active >= c.shape[1])):
        raise ValueError('base_active must contain 20 distinct valid integer MO indices')
    if np.count_nonzero(occ[active] > 0.5) != 6:
        raise ValueError('base_active must contain exactly six occupied spinors')
    core = np.setdiff1d(occupied, active)
    external = np.setdiff1d(virtual, active)
    if len(external) < 10:
        raise ValueError('Fewer than ten external virtual spinors remain')

    ref = make_u_reference(mol) if u_reference is None else u_reference
    if ref['basis'] != basis or ref['nuclear_zeta'] != zeta:
        raise ValueError('U reference basis/nuclear model differs from this molecule')
    _, _, start, stop = mol.aoslice_by_atom()[iu]
    lift = np.vstack(mol.sph2spinor_coeff())
    targets = {}
    for shell in ('5d', '6d'):
        radial = np.asarray(ref[shell])
        if radial.shape != (stop - start, 5) or not np.isfinite(radial).all():
            raise ValueError(f'Invalid U {shell} reference coefficients')
        spatial = np.zeros((2 * mol.nao_nr(), 10), dtype=complex)
        spatial[start:stop, :5] = radial
        spatial[mol.nao_nr() + start:mol.nao_nr() + stop, 5:] = radial
        targets[shell] = lift.conj().T @ spatial
    both = np.column_stack((targets['5d'], targets['6d']))
    if not np.allclose(both.conj().T @ s @ both, np.eye(20), atol=1e-7, rtol=0):
        raise ValueError('Embedded U radial targets are not mutually orthonormal')

    uc, sc, _ = np.linalg.svd(c[:, core].conj().T @ s @ targets['5d'], full_matrices=True)
    uv, sv, _ = np.linalg.svd(c[:, external].conj().T @ s @ targets['6d'], full_matrices=True)
    if sc[-1] < min_5d_overlap:
        raise ValueError(f'Incomplete 5d shell in the remaining occupied space: {sc}')
    if sv[-1] < min_6d_overlap:
        raise ValueError(f'Fewer than ten sufficiently independent external 6d directions: {sv}')
    core_rotated, virtual_rotated = c[:, core] @ uc, c[:, external] @ uv
    mo = np.column_stack((core_rotated[:, 10:], core_rotated[:, :10],
                          c[:, active], virtual_rotated[:, :10], virtual_rotated[:, 10:]))
    error = float(np.max(abs(mo.conj().T @ s @ mo - np.eye(mo.shape[1]))))
    if error > 1e-7:
        raise RuntimeError(f'Constructed orbitals lost orthonormality: {error:.3e}')
    info = {'ncore': 110, 'ncas': 40, 'nelecas': 16,
            'base_active_indices': active.tolist(),
            '5d_core_singular_values': sc.tolist(), '6d_external_singular_values': sv.tolist(),
            'blocks': {'5d': (110, 120), 'base': (120, 140), '6d': (140, 150)},
            'orthogonality_error': error, 'targets': targets}
    info['localization'] = None
    if localize:
        original = mo
        mo = localize_unf3_cas(mf, original)
        info['localization'] = 'active split-PM (occupied/empty)'
        info['initial_blocks'] = info['blocks']
        info['blocks'] = {'active': (110, 150)}
        info['active_rotation'] = original[:, 110:150].conj().T @ s @ mo[:, 110:150]
        info['orthogonality_error'] = float(np.max(abs(mo.conj().T @ s @ mo - np.eye(mo.shape[1]))))
    return 40, 16, mo, info


if __name__ == '__main__':
    import argparse
    from types import SimpleNamespace

    from pyscf import lib, scf

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('hf_chk', nargs='+', help='Real UNF3 spinor HF checkpoints')
    parser.add_argument('--compare', help='Compare first point with saved spaces_and_rdm.npz mo_C')
    parser.add_argument('--localize', action='store_true', help='Also check active split-PM invariants')
    args = parser.parse_args()
    lib.num_threads(4)
    reference = None
    for index, path in enumerate(args.hf_chk):
        mol, hf = scf.chkfile.load_scf(path)
        s = mol.intor_symmetric('int1e_ovlp_spinor')
        mf = SimpleNamespace(mol=mol, **hf, get_ovlp=lambda s=s: s)
        if reference is None:
            reference = make_u_reference(mol)
        ncas, nelec, mo, info = make_unf3_cas(mf, u_reference=reference)
        assert (ncas, nelec, info['ncore']) == (40, 16, 110)
        np.testing.assert_array_equal(mo[:, 120:140], mf.mo_coeff[:, info['base_active_indices']])
        # A change of occupied/virtual gauge must preserve the HF determinant.
        old_occ = mf.mo_coeff[:, mf.mo_occ > 0.5]
        np.testing.assert_allclose(np.linalg.svd(old_occ.conj().T @ s @ mo[:, :126],
                                                compute_uv=False), 1, atol=1e-7, rtol=0)
        if index == 0 and args.compare:
            old = np.load(args.compare)['mo_C'][:, 110:150]
            np.testing.assert_allclose(np.linalg.svd(old.conj().T @ s @ mo[:, 110:150],
                                                    compute_uv=False), 1, atol=1e-7, rtol=0)
        # Reordering input HF columns, with matching metadata, preserves the CAS.
        perm = np.random.default_rng(17).permutation(mo.shape[1])
        shuffled = SimpleNamespace(mol=mol, mo_coeff=mf.mo_coeff[:, perm],
                                   mo_occ=mf.mo_occ[perm], mo_energy=mf.mo_energy[perm],
                                   get_ovlp=lambda s=s: s)
        _, _, other, _ = make_unf3_cas(shuffled, u_reference=reference)
        np.testing.assert_allclose(np.linalg.svd(mo[:, 110:150].conj().T @ s @ other[:, 110:150],
                                                compute_uv=False), 1, atol=1e-7, rtol=0)
        if args.localize:
            nc, ne, localized, loc_info = make_unf3_cas(mf, u_reference=reference, localize=True)
            assert (nc, ne) == (40, 16) and loc_info['blocks'] == {'active': (110, 150)}
            np.testing.assert_array_equal(localized[:, :110], mo[:, :110])
            np.testing.assert_array_equal(localized[:, 150:], mo[:, 150:])
            np.testing.assert_allclose(np.linalg.svd(mo[:, 110:150].conj().T @ s @ localized[:, 110:150],
                                                    compute_uv=False), 1, atol=1e-7, rtol=0)
            np.testing.assert_allclose(localized[:, :126] @ localized[:, :126].conj().T,
                                       old_occ @ old_occ.conj().T, atol=1e-7, rtol=0)
        print(f'PASS {path}: CAS(16e,40 spinors), orthogonality={info["orthogonality_error"]:.2e}, '
              f'min 5d={min(info["5d_core_singular_values"]):.6f}, '
              f'min 6d={min(info["6d_external_singular_values"]):.6f}', flush=True)
