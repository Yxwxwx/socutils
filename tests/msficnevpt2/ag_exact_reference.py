# SPDX-License-Identifier: GPL-3.0-or-later
"""Fixed-Hamiltonian, fixed-N exact Ag CASCI reference-accuracy control.

PySCF's complex spinor FCISolver supplies Davidson, diagonal preconditioning,
and convergence tests. Only its Python-loop Hamiltonian contraction is
vectorized, using the independently tested SGF measurement action. No
alpha/beta mapping, orbital optimization or symmetry projection is used.
"""
import argparse
from pathlib import Path
import os
import time

import numpy as np
from pyscf import lib
from pyscf.fci import fci_dhf_slow

from tests.nevpt2_mps_response.f_atom import active_action_batch, _active_action_links
from .ag_audit import write_json


class BatchedSGFFCI(fci_dhf_slow.FCISolver):
    def contract_2e(self, eri, fcivec, norb, nelec, link_index=None, **kwargs):
        return active_action_batch(eri, np.asarray(fcivec)[None, :], norb, nelec)[0]


def exact_reference(h1e, eri, ci0, *, nelec=11, nroots=2, residual_tol=1e-12, threads=1, max_memory=250000):
    solver = BatchedSGFFCI()
    solver.nroots = nroots
    # The residual sets accuracy; an energy gate below roundoff can destabilize degenerate roots.
    solver.conv_tol = 1e-12
    solver.conv_tol_residual = residual_tol
    solver.lindep = 1e-24
    solver.max_cycle, solver.max_space = 200, 128
    solver.max_memory, solver.threads, solver.verbose = max_memory, threads, 5
    # Keep the large inactive constant out of the Davidson energy subtraction.
    guess = None if ci0 is None else [v.copy() for v in np.asarray(ci0).T]
    if guess is None and residual_tol < 1e-9:
        # Restart native Davidson before roundoff-sized corrections pollute its basis.
        solver.conv_tol_residual = 1e-9
        _, guess = solver.kernel(h1e, eri, len(h1e), nelec, ci0=None, ecore=0)
        solver.conv_tol_residual = residual_tol
        print("EXACT_CI_STRICT_RESTART", "residual_target", residual_tol, flush=True)
    energies, vectors = solver.kernel(h1e, eri, len(h1e), nelec, ci0=guess, ecore=0)
    vectors = np.column_stack(vectors)
    absorbed = fci_dhf_slow.absorb_h1e(h1e, eri, len(h1e), nelec, 0.5)
    actions = active_action_batch(absorbed, vectors.T, len(h1e), nelec).T
    residuals = np.linalg.norm(actions - vectors * energies, axis=0)
    return solver.converged, energies, vectors, actions, residuals


def main(directory):
    with np.load(directory / "reference_vectors.npz", allow_pickle=False) as saved:
        original = saved["vectors"].copy()
        h1e, eri = saved["h1e"].copy(), saved["eri"].copy()
        fingerprint = str(saved["hamiltonian_sha256"].item())
    with np.load(directory.parent / "dmrg_state/reference.npz", allow_pickle=False) as saved:
        if fingerprint != str(saved["hamiltonian_sha256"].item()):
            raise ValueError("fixed-orbital exact-CI input fingerprint mismatch")
        np.testing.assert_array_equal(h1e, saved["h1e"])
        np.testing.assert_array_equal(eri, saved["eri"])
        ecore = float(saved["ecore"])
        old_energies = saved["e_states"][:2].copy()
    threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    lib.num_threads(threads)
    started = time.monotonic()
    converged, energies, vectors, actions, residuals = exact_reference(
        h1e, eri, original, threads=threads)
    gram = vectors.conj().T @ vectors
    overlap_singular_values = np.linalg.svd(vectors.conj().T @ original, compute_uv=False)
    np.savez(directory / "exact_reference_vectors.npz", vectors=vectors, actions=actions,
             active_energies=energies, hamiltonian_sha256=fingerprint)
    report = dict(hamiltonian_sha256=fingerprint, ncas=22, nelec=11, nroots=2,
                  requested_residual_tol=1e-12, converged=converged,
                  active_energies=energies, total_energies=energies + ecore,
                  original_saved_energies=old_energies,
                  residual_norms=residuals, gram=gram,
                  overlap_singular_values=overlap_singular_values,
                  spread_cm_inverse=float(np.ptp(energies) * 219474.6313705),
                  wall_seconds=time.monotonic() - started)
    write_json(directory / "exact_reference.json", report)
    _active_action_links.cache_clear()
    if not np.all(converged) or not np.all(np.isfinite(residuals)) or max(residuals) > 1e-10:
        raise RuntimeError("fixed-orbital exact CI did not meet the independent residual check")
    np.testing.assert_allclose(gram, np.eye(2), atol=1e-10, rtol=0)
    print("AG_EXACT_REFERENCE_DONE", report, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    main(parser.parse_args().directory.resolve())
