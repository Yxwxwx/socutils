# block2: Efficient MPO implementation of quantum chemistry DMRG
# Copyright (C) 2022 Huanchen Zhai <hczhai@caltech.edu>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# This module adapts block2's uno.py (lo/uno_b2.py, SHA256
# 613e77442c33a0feea57f8b90951356f8c0345af0701ffe306b8495b60ff4783).
# Original PM/UNO code: Zhendong Li and Qiming Sun; partition-list revision:
# Huanchen Zhai. Complex-spinor Jacobi and spinor metadata handling: socutils.
"""One-time, general-spinor orbital preparation for X2C DMRG-SCF.

This module does not preserve Kramers pairs. Start a new DMRG state in the
returned basis after localizing active orbitals.
"""

import numpy as np
from scipy.linalg import qr


def _hermitian_eigh(s):
    s = np.asarray(s)
    if s.ndim != 2 or s.shape[0] != s.shape[1] or not np.isfinite(s).all():
        raise ValueError("overlap must be a finite square matrix")
    if not np.allclose(s, s.conj().T, atol=1e-10, rtol=0):
        raise ValueError("overlap must be Hermitian")
    e, v = np.linalg.eigh(s)
    if e.size and e[0] <= max(1e-12 * e[-1], 0):
        raise ValueError("overlap must be positive definite")
    return e, v


def sqrtm(s):
    """Hermitian positive square root; reject unreliable overlap eigenvalues."""
    e, v = _hermitian_eigh(s)
    return (v * np.sqrt(e)) @ v.conj().T


def lowdin(s):
    """Hermitian inverse square root without dropping AO directions."""
    e, v = _hermitian_eigh(s)
    return (v / np.sqrt(e)) @ v.conj().T


def _partition(mol, nrow, iop):
    if isinstance(mol, list):
        if iop != 1:
            raise ValueError("a partition list requires iop=1")
        rows = []
        for group in mol:
            row = np.asarray(group)
            if row.ndim != 1 or (row.size and row.dtype.kind not in "iu"):
                raise ValueError("partition rows must be integer indices")
            rows.append(row.astype(int))
        s = None
    else:
        if getattr(mol, "so_contr", None) is not None:
            raise ValueError("so_contr changes the spinor AO metric; this representation is unsupported")
        if not hasattr(mol, "nao_2c") or nrow != mol.nao_2c():
            raise ValueError("coefficients must use the molecule's 2c spinor AO rows")
        s = np.asarray(mol.intor_symmetric("int1e_ovlp_spinor"))
        if s.shape != (nrow, nrow):
            raise ValueError("spinor overlap shape disagrees with coefficients")
        slices = np.asarray(mol.aoslice_2c_by_atom())
        if slices.shape != (mol.natm, 4):
            raise ValueError("2c atom slices have the wrong shape")
        rows = [np.arange(start, stop) for start, stop in slices[:, 2:4]]
    if any(group.ndim != 1 for group in rows):
        raise ValueError("each atom partition must be one-dimensional")
    joined = np.concatenate(rows) if rows else np.array([], dtype=int)
    if joined.size != nrow or not np.array_equal(np.sort(joined), np.arange(nrow)):
        raise ValueError("atom partitions must cover each spinor AO exactly once")
    return s, rows


def _population(mol, c, iop):
    s, rows = _partition(mol, c.shape[0], iop)
    if s is None:
        metric = np.eye(c.shape[0])
    else:
        _hermitian_eigh(s)
        metric = s
    if not np.allclose(c.conj().T @ metric @ c, np.eye(c.shape[1]), atol=1e-7, rtol=0):
        raise ValueError("input orbitals are not orthonormal in the spinor overlap")
    if iop == 0:
        cs = c.conj().T @ s
        q = []
        for row in rows:
            v = cs[:, row] @ c[row, :]
            q.append((v + v.conj().T) * 0.5)
    elif iop == 1:
        b = c if s is None else sqrtm(s) @ c
        q = [b[row, :].conj().T @ b[row, :] for row in rows]
    elif iop == 2:
        r = np.asarray(mol.intor_symmetric("int1e_r_spinor", comp=3))
        if r.shape != (3, c.shape[0], c.shape[0]):
            raise ValueError("spinor position integrals have the wrong shape")
        q = [c.conj().T @ component @ c for component in r]
    else:
        raise ValueError("iop must be 0, 1, or 2")
    q = np.asarray(q).reshape(len(q), c.shape[1], c.shape[1])
    if not np.allclose(q, q.swapaxes(1, 2).conj(), atol=1e-9, rtol=0):
        raise ValueError("localization matrices are not Hermitian")
    if iop != 2 and not np.allclose(q.sum(axis=0), np.eye(c.shape[1]), atol=1e-7, rtol=0):
        raise ValueError("atom populations do not sum to the active identity")
    return q


def _objective(q):
    return float(np.sum(np.diagonal(q, axis1=1, axis2=2).real ** 2))


def _pair_gram(q, i, j):
    z = (q[:, i, i].real - q[:, j, j].real) * 0.5
    v = np.stack((q[:, i, j].real, -q[:, i, j].imag, z), axis=1)
    return v.T @ v


def _complex_pair(q, i, j):
    """Best SU(2) rotation of one orbital pair."""
    g = _pair_gram(q, i, j)
    eig, vec = np.linalg.eigh(g)
    gain = 2 * (eig[-1] - g[2, 2])
    if gain <= 1e-13 * max(1.0, abs(eig[-1])):
        return None
    top = vec[:, eig[-1] - eig <= 1e-12 * max(1.0, abs(eig[-1]))]
    direction = top @ top[2, :]
    if np.linalg.norm(direction) <= 1e-12:
        direction = vec[:, -1]
    direction /= np.linalg.norm(direction)
    if direction[2] < 0:
        direction = -direction
    c = np.sqrt((1 + direction[2]) * 0.5)
    d = (direction[0] + 1j * direction[1]) / (2 * c)
    return np.array([[c, -d.conjugate()], [d, c]])


def _real_pair(q, i, j):
    """The original block2 real-angle Jacobi step, including its skips."""
    vij = q[:, i, i] - q[:, j, j]
    aij = np.dot(q[:, i, j], q[:, i, j]) - 0.25 * np.dot(vij, vij)
    bij = np.dot(q[:, i, j], vij)
    if abs(aij) < 1e-10 and abs(bij) < 1e-10:
        return None
    p1 = np.hypot(aij, bij)
    cos4a, sin4a = -aij / p1, bij / p1
    cos2a = np.sqrt((1 + cos4a) * 0.5)
    cosa = np.sqrt((1 + cos2a) * 0.5)
    sina = np.sqrt((1 - cos2a) * 0.5)
    if sin4a < 0:
        sina, cosa = cosa, sina
    if abs(cosa - 1) < 1e-10 or abs(sina - 1) < 1e-10:
        return None
    return np.array([[cosa, -sina], [sina, cosa]])


def pmloc(mol, mocoeff, tol=1e-6, maxcycle=1000, iop=0, iprint=1):
    """Return ``(ierr, U)`` for 2c PM (0/1) or Boys (2) Jacobi sweeps.

    ``ierr=0`` means the sweep objective increment fell below ``tol``.
    Kramers pairing and orbital-irrep labels are not preserved.
    """
    c = np.asarray(mocoeff)
    if c.ndim != 2 or not np.issubdtype(c.dtype, np.number) or not np.isfinite(c).all():
        raise ValueError("mocoeff must be a finite two-dimensional numeric array")
    c = np.asarray(c, dtype=np.complex128 if np.iscomplexobj(c) else np.float64)
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and positive")
    if isinstance(maxcycle, (bool, np.bool_)) or not isinstance(maxcycle, (int, np.integer)) or maxcycle < 1:
        raise ValueError("maxcycle must be a positive integer")
    if iop not in (0, 1, 2):
        raise ValueError("iop must be 0, 1, or 2")
    q = _population(mol, c, iop)
    n = c.shape[1]
    real_route = np.isrealobj(c) and np.max(abs(q.imag), initial=0) < 1e-13
    if real_route:
        q = q.real
    u = np.eye(n, dtype=float if real_route else complex)
    if iprint:
        print(f"[pm_loc_kernel] mocoeff.shape={c.shape} tol={tol} maxcycle={maxcycle} iop={iop}")
        print(f" initial funval = {_objective(q):.12g}")
    if n < 2:
        return 0, u
    for cycle in range(maxcycle):
        before = _objective(q)
        pairs = [(i, j, abs(np.sum(q[:, i, j] * (q[:, i, i] - q[:, j, j]))))
                 for i in range(n - 1) for j in range(i + 1, n)]
        for i, j, _ in sorted(pairs, key=lambda x: x[2], reverse=True):
            rotation = (_real_pair(q, i, j) if real_route else _complex_pair(q, i, j))
            if rotation is None:
                continue
            pair = [i, j]
            old = float(np.sum(q[:, pair, pair].real ** 2))
            q[:, pair, :] = np.einsum("ab,kbc->kac", rotation.conj().T, q[:, pair, :])
            q[:, :, pair] = np.einsum("kab,bc->kac", q[:, :, pair], rotation)
            new = float(np.sum(q[:, pair, pair].real ** 2))
            if new < old - 1e-10 * max(1, abs(old)):
                raise ArithmeticError("PM pair objective decreased")
            u[:, pair] = u[:, pair] @ rotation
        after = _objective(q)
        delta = after - before
        if delta < -1e-10 * max(1, abs(before)):
            raise ArithmeticError("PM sweep objective decreased")
        if iprint:
            print(f"icycle={cycle} delta={delta:.6g} fun={after:.12g}")
        if delta < tol:
            if iprint:
                print("CONG: PMloc converged!")
                gains = []
                for i, j, _ in pairs:
                    g = _pair_gram(q, i, j)
                    gains.append(2 * (np.linalg.eigvalsh(g)[-1] - g[2, 2]))
                print(f" maximum remaining pair gain = {max([0.0, *gains]):.6e}")
            return 0, u
    if iprint:
        print("WARNING: PMloc not converged")
    return 1, u


def sort_orbitals(mol, coeff, mo_occ, mo_energy, cas_list=None, nactorb=None,
                  nactelec=None, do_loc=False, split_low=0.0, split_high=0.0,
                  iprint=1):
    """Prepare selected active spinors once, before building a new DMRG state.

    Return coefficients, diagonal occupation/energy expectations, active
    spinor count and electron count. Use copies to preserve the input HF.
    Writable complex128/float64 input arrays receive active-slot write-back;
    other dtypes are left unchanged and receive promoted return arrays.
    """
    c0 = np.asarray(coeff)
    occ0 = np.asarray(mo_occ)
    energy0 = np.asarray(mo_energy)
    if c0.ndim != 2 or c0.shape[0] != mol.nao_2c():
        raise ValueError("coeff must use 2c spinor AO rows")
    nmo = c0.shape[1]
    if occ0.shape != (nmo,) or energy0.shape != (nmo,):
        raise ValueError("occupation and energy arrays must match coefficient columns")
    if not (np.isfinite(c0).all() and np.isfinite(occ0).all() and np.isfinite(energy0).all()):
        raise ValueError("orbital data must be finite")
    if np.iscomplexobj(occ0) or np.iscomplexobj(energy0):
        raise ValueError("occupation and orbital energy must be real")
    occ0 = np.array(occ0, dtype=np.float64, copy=True)
    energy0 = np.array(energy0, dtype=np.float64, copy=True)
    if np.any(occ0 < -1e-7) or np.any(occ0 > 1 + 1e-7):
        raise ValueError("individual spinor occupations must be in [0,1]")
    if abs(float(np.sum(occ0)) - mol.nelectron) > 1e-6:
        raise ValueError("spinor occupations do not sum to the molecular electron count")
    if not np.isfinite(split_low) or not np.isfinite(split_high):
        raise ValueError("split thresholds must be finite")
    split = split_low != 0 or split_high != 0
    if split and (not do_loc or split_high < split_low):
        raise ValueError("split localization requires do_loc=True and split_high>=split_low")
    for name, count in (("nactorb", nactorb), ("nactelec", nactelec)):
        if count is not None and (isinstance(count, (bool, np.bool_))
                                  or not isinstance(count, (int, np.integer))):
            raise ValueError(f"{name} must be an integer")
    if cas_list is None:
        if nactorb is None or nactelec is None:
            raise ValueError("nactorb and nactelec are required without cas_list")
        cas_list = list(range(mol.nelectron - nactelec, mol.nelectron - nactelec + nactorb))
    indices = np.asarray(cas_list)
    if indices.ndim != 1 or indices.dtype.kind not in "iu" or len(np.unique(indices)) != len(indices):
        raise ValueError("cas_list must contain distinct 0-based integer indices")
    if np.any(indices < 0) or np.any(indices >= nmo):
        raise ValueError("cas_list index outside orbital columns")
    if nactorb is not None and nactorb != len(indices):
        raise ValueError("nactorb conflicts with cas_list")
    trace = float(np.sum(occ0[indices]))
    electrons = int(np.rint(trace))
    if abs(trace - electrons) > 1e-6:
        raise ValueError(f"active occupation trace {trace:.10g} is not an integer")
    if nactelec is not None and nactelec != electrons:
        raise ValueError(f"nactelec={nactelec} conflicts with active trace {trace:.10g}")
    ncore = mol.nelectron - electrons
    if ncore < 0 or ncore > nmo - len(indices):
        raise ValueError("active electrons leave an invalid number of core spinors")
    selected = np.sort(indices)
    rest = np.setdiff1d(np.arange(nmo), selected)
    core_occ = occ0[rest[:ncore]]
    virtual_occ = occ0[rest[ncore:]]
    if core_occ.size and virtual_occ.size and core_occ.min() + 1e-6 < virtual_occ.max():
        raise ValueError("complement is not ordered from core to virtual occupations")
    s, _ = _partition(mol, c0.shape[0], 0)
    _hermitian_eigh(s)
    c0 = np.asarray(c0, dtype=np.complex128)
    if not np.allclose(c0.conj().T @ s @ c0, np.eye(nmo), atol=1e-7, rtol=0):
        raise ValueError("input orbitals are not orthonormal in the spinor overlap")
    if iprint:
        print("cas list =", indices.tolist())
        print("split localization" if split else "simple localization")

    def psort(x):
        t = c0.conj().T @ s @ x
        probabilities = abs(t) ** 2
        n = np.asarray(occ0 @ probabilities, dtype=float)
        e = np.asarray(energy0 @ probabilities, dtype=float)
        order = np.argsort(-n)
        return x[:, order], n[order], e[order]

    active = c0[:, indices].copy()
    if split:
        initial_occ = occ0[indices]
        masks = (initial_occ <= split_low,
                 (initial_occ > split_low) & (initial_occ <= split_high),
                 initial_occ > split_high)
        active_occ = initial_occ.copy()
        active_energy = energy0[indices].copy()
        for name, mask in zip(("low", "mid", "high"), masks):
            if not np.any(mask):
                continue
            ierr, u = pmloc(mol, active[:, mask], iprint=iprint)
            if ierr:
                raise RuntimeError(f"PM localization did not converge for {name} active group")
            x, n, e = psort(active[:, mask] @ u)
            active[:, mask], active_occ[mask], active_energy[mask] = x, n, e
    else:
        if do_loc:
            ierr, u = pmloc(mol, active, iprint=iprint)
            if ierr:
                raise RuntimeError("PM localization did not converge for active group")
            active = active @ u
        active, active_occ, active_energy = psort(active)
    if abs(float(np.sum(active_occ)) - electrons) > 1e-6:
        raise ArithmeticError("active occupation trace changed during localization")
    work = c0.copy()
    occ = np.asarray(occ0, dtype=float).copy()
    energy = np.asarray(energy0, dtype=float).copy()
    work[:, selected], occ[selected], energy[selected] = active, active_occ, active_energy
    # Real/low-precision inputs return complex128 results without partial write-back.
    writable = (isinstance(coeff, np.ndarray) and coeff.dtype == np.complex128
                and coeff.flags.writeable and isinstance(mo_occ, np.ndarray)
                and isinstance(mo_energy, np.ndarray)
                and mo_occ.dtype == np.float64
                and mo_energy.dtype == np.float64
                and mo_occ.flags.writeable and mo_energy.flags.writeable)
    if writable:
        coeff[:, selected] = active
        mo_occ[selected] = active_occ
        mo_energy[selected] = active_energy
    order = np.r_[rest[:ncore], selected, rest[ncore:]]
    if iprint:
        print(f"NACTORB = {len(indices)} NACTELEC = {electrons} NCORE = {ncore}")
    return work[:, order], occ[order], energy[order], len(indices), electrons


def _scdm_virtual(coeff, overlap):
    """SCDM in the virtual subspace using pivoted Löwdin-AO projections."""
    if coeff.shape[1] == 0:
        return coeff.copy()
    projection = coeff.conj().T @ sqrtm(overlap)
    _, _, pivots = qr(projection, pivoting=True, mode="economic")
    selected = projection[:, pivots[:coeff.shape[1]]]
    result = coeff @ selected @ lowdin(selected.conj().T @ selected)
    if not np.allclose(result.conj().T @ overlap @ result,
                       np.eye(coeff.shape[1]), atol=1e-7, rtol=0):
        raise ArithmeticError("virtual SCDM lost spinor orthonormality")
    return result


def localize_blocks(mol, coeff, mo_occ, mo_energy, cas_list=None, nactorb=None,
                    nactelec=None, split_low=0.0, split_high=0.0, iprint=1):
    """Localize a fixed CAS: core PM, active PM, virtual SCDM.

    This is the C/A/V localization stage of block2 ``get_uno()``, without
    its UHF-to-UNO transformation or active-space selection. The active block
    follows ``sort_orbitals()`` simple/split semantics. Return the same
    five-tuple in C/A/V order without modifying input arrays.
    """
    c, occ, energy, ncas, nelec = sort_orbitals(
        mol, np.array(coeff, copy=True), np.array(mo_occ, copy=True),
        np.array(mo_energy, copy=True), cas_list=cas_list, nactorb=nactorb,
        nactelec=nactelec, do_loc=True, split_low=split_low,
        split_high=split_high, iprint=iprint,
    )
    ncore = mol.nelectron - nelec
    if ncore:
        ierr, u = pmloc(mol, c[:, :ncore], iprint=iprint)
        if ierr:
            raise RuntimeError("PM localization did not converge for core group")
        c[:, :ncore] = c[:, :ncore] @ u
        weight = abs(u) ** 2
        occ[:ncore], energy[:ncore] = occ[:ncore] @ weight, energy[:ncore] @ weight
    start = ncore + ncas
    if start < c.shape[1]:
        overlap, _ = _partition(mol, c.shape[0], 0)
        old_virtual = c[:, start:].copy()
        c[:, start:] = _scdm_virtual(old_virtual, overlap)
        u = old_virtual.conj().T @ overlap @ c[:, start:]
        weight = abs(u) ** 2
        occ[start:], energy[start:] = occ[start:] @ weight, energy[start:] @ weight
    for begin, end in ((0, ncore), (start, c.shape[1])):
        order = np.argsort(-occ[begin:end])
        c[:, begin:end] = c[:, begin:end][:, order]
        occ[begin:end], energy[begin:end] = occ[begin:end][order], energy[begin:end][order]
    return c, occ, energy, ncas, nelec
