# SPDX-License-Identifier: GPL-3.0-or-later
"""Exact block AO2MO for spinor NEVPT2; no full MO four-index tensor.

The block-storage strategy follows pyblock2.icmr.eri_helper.  Transformations
and symmetry projection here use complex spinors and only the four complex
Coulomb identities, not the eight real spatial-orbital permutations.
"""

import numpy as np
from pyscf import lib
from pyscf.ao2mo import nrr_outcore
from pyscf.x2c.x2c import get_jk

from . import nevpt2_utils as u


# Chemists order: pair exchange, Hermitian exchange, and their composition.
_PERMS = ((0, 1, 2, 3), (2, 3, 0, 1), (1, 0, 3, 2), (3, 2, 1, 0))


def wick_eris_from_mc(mc, mo_coeff, *, max_memory=2000, ioblk_size=128,
                      roundoff_factor=u._DEFAULT_AO2MO_ROUNDOFF_FACTOR):
    """Transform just the ten Wick blocks with bounded AO2MO working memory.

    Memory parameters are MB, independent of MPS/RDM storage. Output blocks
    are in memory; HDF5 AO2MO intermediates are temporary and closed on error.
    Each block is projected using independently transformed symmetry partners
    before those partners are discarded. No DF or integral truncation is used.
    """
    for name, value in (("max_memory", max_memory), ("ioblk_size", ioblk_size)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    mo = np.asarray(mo_coeff)
    if mo.ndim != 2 or mo.shape[0] != mc.mol.nao_2c():
        raise ValueError("expected j-spinor MO coefficients")
    ncore, ncas = int(mc.ncore), int(mc.ncas)
    nmo = mo.shape[1]
    if min(ncore, ncas) < 0 or ncore + ncas > nmo:
        raise ValueError("invalid core/active/virtual partition")
    slices = {"I": slice(0, ncore), "A": slice(ncore, ncore + ncas),
              "E": slice(ncore + ncas, nmo)}
    coeffs = {key: mo[:, sl] for key, sl in slices.items()}

    h1e = mo.conj().T @ np.asarray(mc.get_hcore()) @ mo
    # Exact Coulomb J-K in the j-spinor AO basis. Each core spinor is occupied
    # once. Calling the SCF object's get_jk could silently introduce its DF.
    h1eff = h1e.copy()
    if ncore:
        core_dm = np.ascontiguousarray(coeffs["I"] @ coeffs["I"].conj().T)
        vj, vk = get_jk(mc.mol, core_dm, hermi=1)
        h1eff += mo.conj().T @ (vj - vk) @ mo
    h_error = u._maximum_abs(h1eff - h1eff.conj().T)
    h_tol, h_policy = u._ao2mo_roundoff_policy(
        h1eff, accumulation_length=mo.shape[0], contraction_stages=2,
        roundoff_factor=roundoff_factor)
    if h_error > h_tol:
        u._warn_numerical("core-dressed h1eff violates Hermiticity beyond roundoff")
    h1eff = 0.5 * h1eff + 0.5 * h1eff.conj().T
    phys_blocks, audits = {}, {}
    for key in u._W_KEYS:
        chem_key = key[0] + key[2] + key[1] + key[3]
        shape = tuple(coeffs[x].shape[1] for x in chem_key)
        if 0 in shape:
            phys_blocks[key] = np.zeros(tuple(coeffs[x].shape[1] for x in key),
                                        dtype=np.complex128)
            continue
        lib.logger.info(mc, "NEVPT2 AO2MO block %s: %.3f GiB", key,
                        np.prod(shape, dtype=float) * 16 / 2**30)
        # Never retain the four symmetry partners together in RAM.
        with lib.H5TmpFile() as disk:
            def transform(partition):
                if partition not in disk:
                    nrr_outcore.general(
                        mc.mol, tuple(coeffs[x] for x in partition), disk,
                        dataname=partition, motype="j-spinor", aosym="s1",
                        max_memory=max_memory, ioblk_size=ioblk_size,
                        verbose=getattr(mc, "verbose", 0))
                dims = tuple(coeffs[x].shape[1] for x in partition)
                return np.asarray(disk[partition]).reshape(dims)

            raw = transform(chem_key)
            tol, policy = u._ao2mo_roundoff_policy(
                raw, accumulation_length=mo.shape[0], contraction_stages=4,
                roundoff_factor=roundoff_factor)
            projected = 0.25 * raw
            residuals = []
            for iperm, perm in enumerate(_PERMS[1:], start=1):
                partner = transform("".join(chem_key[x] for x in perm)).transpose(perm)
                error = u._maximum_abs_relation(
                    raw, partner, sign=-1.0, conjugate_right=iperm >= 2,
                    work_memory=int(ioblk_size * 1e6))
                residuals.append(error)
                for sl in u._leading_chunks(raw.shape, raw.dtype,
                                             int(ioblk_size * 1e6)):
                    value = partner[sl].conj() if iperm >= 2 else partner[sl]
                    projected[sl] += 0.25 * value
                del partner, value
            if not u._all_finite_chunked(raw, int(ioblk_size * 1e6)):
                raise ValueError(f"non-finite AO2MO block {key}")
            if not u._all_finite_chunked(projected, int(ioblk_size * 1e6)):
                raise ValueError(f"non-finite projected AO2MO block {key}")
            if max(residuals) > tol:
                u._warn_numerical(f"AO2MO block {key} violates complex Coulomb "
                                 f"symmetry: error={max(residuals):.3e}, tolerance={tol:.3e}")
            audits[key] = {"raw_pair_exchange_error": residuals[0],
                           "raw_conjugate_exchange_error": residuals[1],
                           "raw_composed_exchange_error": residuals[2],
                           "roundoff_gate": tol,
                           "roundoff_gate_passed": bool(max(residuals) <= tol),
                           "roundoff_policy": policy}
            del raw
            # Transpose is a view, so no additional block copy is required.
            phys_blocks[key] = projected.transpose(0, 2, 1, 3)
            del projected
    diagnostics = {"storage": "direct_wick_blocks", "full_mo_eri_built": False,
                   "two_electron_operator": "full_coulomb",
                   "electronic_core_energy": float(np.real(
                       0.5 * np.trace((h1e + h1eff)[:ncore, :ncore]))),
                   "ao2mo_max_memory_mb": float(max_memory),
                   "h1eff": {"raw_hermiticity_error": h_error,
                              "roundoff_gate_passed": bool(h_error <= h_tol),
                              "roundoff_gate": h_tol, "roundoff_policy": h_policy},
                   "blocks": audits}
    return u._WickERIBlocks(ncore, ncas, nmo - ncore - ncas,
                            {k: h1eff[slices[k[0]], slices[k[1]]].copy()
                             for k in u._H1_KEYS}, phys_blocks, diagnostics)
