# SPDX-License-Identifier: GPL-3.0-or-later
"""Numerical IC-basis refinement shared by CASPT2 and MS-FIC-NEVPT2."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.linalg import eigh


def extended_dot(a, b, *, threads=1):
    """Extended accumulation, parallel by disjoint output rows (no BLAS)."""
    a = np.asarray(a, dtype=np.clongdouble, order="C")
    b = np.asarray(b, dtype=np.clongdouble, order="F")
    vector = b.ndim == 1
    if vector:
        b = b[:, None]
    workers = min(max(1, int(threads)), (len(a) + 127) // 128)
    if workers <= 1:
        result = a @ b
    else:
        result = np.empty((len(a), b.shape[1]), dtype=np.clongdouble)

        def rows(start):
            result[start : start + 128] = a[start : start + 128] @ b

        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(rows, range(0, len(a), 128)))
    return result[:, 0] if vector else result


def congruence(x, operator, y=None, *, extended=False, threads=1):
    """Transport an operator consistently with the refined IC basis."""
    if y is None:
        y = x
    if not extended:
        return x.conj().T @ operator @ y
    return np.asarray(
        extended_dot(
            extended_dot(x.conj().T, operator, threads=threads), y, threads=threads
        ),
        dtype=np.complex128,
    )


def refine_metric_basis(x, metric, *, tolerance, threads=1):
    """Re-whiten the actual Gram; never change the retained span or rank."""
    identity = np.eye(x.shape[1])
    before = float(np.max(np.abs(x.conj().T @ metric @ x - identity), initial=0))
    after, steps = before, 0
    if before > tolerance:
        for iteration in range(3):
            gram = congruence(x, metric, extended=True, threads=threads)
            after = float(np.max(np.abs(gram - identity), initial=0))
            if not np.isfinite(after):
                raise FloatingPointError("non-finite retained metric Gram")
            if after <= tolerance or iteration == 2:
                break
            values, vectors = eigh(0.5 * (gram + gram.conj().T))
            if values[0] <= 0.5 or values[-1] >= 1.5:
                raise FloatingPointError(
                    "retained metric Gram is too ill-conditioned to refine safely"
                )
            x = x @ ((vectors / np.sqrt(values)) @ vectors.conj().T)
            steps += 1
    return x, {
        "gram_error_before_refinement": before,
        "gram_refinement_steps": steps,
        "metric_orthogonalization_error": after,
        "metric_gram_accumulation_dtype": np.dtype(
            np.clongdouble if before > tolerance else np.complex128
        ).name,
    }
