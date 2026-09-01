# SPDX-License-Identifier: GPL-3.0-or-later
"""Dense complex spinor-MO integral helper for X2C-SC-NEVPT2.

v1 only stores and partitions integrals that are already in the spinor-MO
basis.  It does not perform AO-to-MO transformation, depend on a CASSCF/DMRG
object, handle Kramers symmetry, store orbital energies, or generate RDMs.

Conventions
-----------

    h1e[p, q]       = <p | h | q>
    eri[p, q, r, s] = (p q | r s)

``eri`` is the unantisymmetrized chemists' integral tensor.  In the SI
notation used by the spinor Wick equations,

    <x y | x' y'> = eri[x, x', y, y'].

Orbitals are ordered as ``[core | active | virtual]``.  Every count refers to
individual spinors, not spatial orbitals or Kramers pairs.  All getters return
NumPy views.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "_SpinorERIs",
    "check_eri_symmetry",
    "get_chem_eri",
    "get_h1_eri",
    "get_h1eff_eri",
    "get_phys_eri",
    "init_eris",
]


_SPACE_LABELS = {
    "c": "c",
    "a": "a",
    "v": "v",
    "p": "p",
    "I": "c",
    "A": "a",
    "E": "v",
    "P": "p",
}


def _normalize_key(key: str, rank: int) -> str:
    if not isinstance(key, str) or len(key) != rank:
        raise ValueError(f"integral key must be a {rank}-character string")
    try:
        return "".join(_SPACE_LABELS[label] for label in key)
    except KeyError as exc:
        raise ValueError(
            f"unknown orbital-space label {exc.args[0]!r}; "
            "use c/a/v/p or I/A/E/P"
        ) from None


def _space_slices(eris: "_SpinorERIs") -> dict[str, slice]:
    return {
        "c": slice(0, eris.ncore),
        "a": slice(eris.ncore, eris.nocc),
        "v": slice(eris.nocc, eris.nmo),
        "p": slice(0, eris.nmo),
    }


def _key_slices(eris: "_SpinorERIs", key: str, rank: int) -> tuple[slice, ...]:
    key = _normalize_key(key, rank)
    spaces = _space_slices(eris)
    return tuple(spaces[label] for label in key)


def get_chem_eri(eris: "_SpinorERIs", key: str) -> np.ndarray:
    """Return a view of ``eri[p,q,r,s] = (pq|rs)`` for ``key``."""

    return eris.pppp[_key_slices(eris, key, 4)]


def get_phys_eri(eris: "_SpinorERIs", key: str) -> np.ndarray:
    """Return a view in SI/physicists order.

    ``result[p,q,r,s] = eri[p,r,q,s]``.  Therefore,

    ``get_phys("EAIA")[r,a,i,b] == eri[r,i,a,b]``.
    """

    key = _normalize_key(key, 4)
    chem_key = key[0] + key[2] + key[1] + key[3]
    return get_chem_eri(eris, chem_key).transpose(0, 2, 1, 3)


def get_h1_eri(eris: "_SpinorERIs", key: str) -> np.ndarray:
    """Return a view of the bare one-electron integral block."""

    return eris.h1e[_key_slices(eris, key, 2)]


def get_h1eff_eri(eris: "_SpinorERIs", key: str) -> np.ndarray:
    """Return a view of the core-dressed one-electron integral block."""

    return eris.h1eff[_key_slices(eris, key, 2)]


def _as_nonnegative_int(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _check_hermitian(name: str, matrix: np.ndarray) -> None:
    if not np.allclose(matrix, matrix.conj().T, rtol=1.0e-10, atol=1.0e-12):
        error = float(np.max(np.abs(matrix - matrix.conj().T)))
        raise ValueError(f"{name} is not Hermitian; max error = {error:.3e}")


def check_eri_symmetry(
    eri,
    *,
    rtol: float = 1.0e-10,
    atol: float = 1.0e-12,
) -> None:
    """Explicitly check the full complex chemists-ERI symmetries.

    This O(n^4) validation is intentionally separate from ``check=True`` in
    :func:`init_eris`:

        (pq|rs) = (rs|pq),
        (pq|rs) = (qp|sr)*.
    """

    eri = np.asarray(eri)
    if eri.ndim != 4 or len(set(eri.shape)) != 1:
        raise ValueError("eri must have shape (nmo, nmo, nmo, nmo)")
    if not np.allclose(eri, eri.transpose(2, 3, 0, 1), rtol=rtol, atol=atol):
        raise ValueError("eri violates (pq|rs) = (rs|pq)")
    if not np.allclose(
        eri,
        eri.transpose(1, 0, 3, 2).conj(),
        rtol=rtol,
        atol=atol,
    ):
        raise ValueError("eri violates (pq|rs) = (qp|sr)*")


def _build_h1eff(h1e: np.ndarray, eri: np.ndarray, ncore: int) -> np.ndarray:
    """Build ``h_eff[pq] = h[pq] + sum_i[(pq|ii) - (pi|iq)]``."""

    dtype = np.result_type(h1e.dtype, eri.dtype)
    h1eff = np.array(h1e, dtype=dtype, copy=True)
    if ncore:
        core = slice(0, ncore)
        h1eff += np.einsum("pqii->pq", eri[:, :, core, core], optimize=True)
        h1eff -= np.einsum("piiq->pq", eri[:, core, core, :], optimize=True)
    return h1eff


class _SpinorERIs:
    """Dense spinor-MO ERI container with ``[core|active|virtual]`` ordering."""

    known = ("pppp",)

    def __init__(
        self,
        h1e: np.ndarray,
        eri: np.ndarray,
        h1eff: np.ndarray,
        ncore: int,
        ncas: int,
    ) -> None:
        self.nmo = h1e.shape[0]
        self.ncore = ncore
        self.ncas = ncas
        self.nocc = ncore + ncas
        self.nvirt = self.nmo - self.nocc
        self.h1e = h1e
        self.h1eff = h1eff
        self.pppp = eri

    @property
    def nbytes(self) -> int:
        """Total storage owned or referenced by the dense integral arrays."""

        return int(self.h1e.nbytes + self.h1eff.nbytes + self.pppp.nbytes)

    get_chem = get_chem_eri
    get_phys = get_phys_eri
    get_h1 = get_h1_eri
    get_h1eff = get_h1eff_eri


def init_eris(
    h1e,
    eri,
    ncore,
    ncas,
    *,
    h1eff=None,
    frozen=0,
    copy=False,
    check=True,
) -> _SpinorERIs:
    """Initialize dense spinor-MO integrals for the Wick SC-NEVPT2 layer.

    ``frozen`` is retained only for call-site compatibility in v1.  Frozen
    spinors must be folded into the input integrals upstream.
    """

    frozen = _as_nonnegative_int(frozen, "frozen")
    if frozen:
        raise NotImplementedError(
            "spinor_helper v1 does not process frozen spinors; handle them upstream"
        )
    ncore = _as_nonnegative_int(ncore, "ncore")
    ncas = _as_nonnegative_int(ncas, "ncas")

    h1e = np.asarray(h1e)
    eri = np.asarray(eri)
    h1eff = None if h1eff is None else np.asarray(h1eff)

    if h1e.ndim != 2 or h1e.shape[0] != h1e.shape[1]:
        raise ValueError(f"h1e must be square; got shape {h1e.shape}")
    nmo = h1e.shape[0]
    if eri.shape != (nmo, nmo, nmo, nmo):
        raise ValueError(f"eri must have shape {(nmo,) * 4}; got {eri.shape}")
    if ncore + ncas > nmo:
        raise ValueError(
            f"invalid partition: nmo={nmo}, ncore={ncore}, ncas={ncas}"
        )
    if h1eff is not None and h1eff.shape != (nmo, nmo):
        raise ValueError(f"h1eff must have shape {(nmo, nmo)}; got {h1eff.shape}")

    if copy:
        h1e = h1e.copy()
        eri = eri.copy()
        if h1eff is not None:
            h1eff = h1eff.copy()
    if h1eff is None:
        h1eff = _build_h1eff(h1e, eri, ncore)

    if check:
        _check_hermitian("h1e", h1e)
        _check_hermitian("h1eff", h1eff)

    return _SpinorERIs(h1e, eri, h1eff, ncore, ncas)
