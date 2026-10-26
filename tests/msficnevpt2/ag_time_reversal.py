# SPDX-License-Identifier: GPL-3.0-or-later
"""Measure, never impose, time-reversal closure of the Ag two-root references."""
import argparse
from pathlib import Path
import os
import time

import numpy as np
from pyscf import lib, scf
from socutils.dmrg.kramers import ao_time_reverse, time_reverse_integrals
from tests.nevpt2_mps_response.uc_audit import time_reverse_coefficients
from .ag_audit import write_json


def model_closure(vectors, reversed_vectors):
    gram = vectors.conj().T @ vectors
    mapping = vectors.conj().T @ reversed_vectors
    leakage = reversed_vectors - vectors @ np.linalg.solve(gram, mapping)
    return dict(model_time_reverse=mapping, leakage_norms=np.linalg.norm(leakage, axis=0),
                closure_singular_values=np.linalg.svd(mapping, compute_uv=False))


def main(directory):
    lib.num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    source = directory.parent
    mol, _ = scf.chkfile.load_scf(str(source / "hf_kr/hf.chk"))
    if mol.atom_symbol(0) != "Ag":
        raise ValueError("this reference-symmetry audit is specific to Ag")
    mol.charge, mol.spin = 0, 1
    ncore = mol.nelectron - 11
    with np.load(source / "dmrg_state/reference.npz", allow_pickle=False) as data:
        mo = data["mo_coeff"][:, ncore:ncore + 22].copy()
        h1e, eri = data["h1e"].copy(), data["eri"].copy()
        fingerprint = str(data["hamiltonian_sha256"].item())
    columns = []
    for filename in ("reference_vectors.npz", "exact_reference_vectors.npz"):
        with np.load(directory / filename, allow_pickle=False) as data:
            if fingerprint != str(data["hamiltonian_sha256"].item()):
                raise ValueError("reference symmetry audit fingerprint mismatch")
            columns.append(data["vectors"].copy())
    overlap = mol.intor_symmetric("int1e_ovlp_spinor")
    reversed_mo = ao_time_reverse(mol, mo)
    t = mo.conj().T @ overlap @ reversed_mo
    orbital_leak = reversed_mo - mo @ t
    leak_norm = float(np.sqrt(max(0.0, np.trace(orbital_leak.conj().T @ overlap @ orbital_leak).real)))
    if np.max(abs(t.conj().T @ t - np.eye(22))) > 1e-10:
        raise ValueError("active space is not time-reversal closed; cannot use an active-only coefficient map")
    htr, gtr = time_reverse_integrals(t, h1e, eri)
    # Fermion monomials have operator norm <=1; this is a conservative bound.
    bound = float(abs(htr - h1e).sum() + 0.5 * abs(gtr - eri).sum())
    print("AG_TIME_REVERSAL_ACTION_BEGIN", "hamiltonian_bound", bound, flush=True)
    started = time.monotonic()
    reversed_states = time_reverse_coefficients(t, np.column_stack(columns), 11)
    original = model_closure(columns[0], reversed_states[:, :2])
    exact = model_closure(columns[1], reversed_states[:, 2:])
    report = dict(hamiltonian_sha256=fingerprint, active_time_reverse=t,
                  active_orbital_closure_norm=leak_norm,
                  orbital_orthonormality_defect=float(np.max(abs(mo.conj().T @ overlap @ mo - np.eye(22)))),
                  orbital_t_square_defect=float(np.max(abs(t @ t.conj() + np.eye(22)))),
                  h1e_time_reverse_defect=float(np.max(abs(htr - h1e))),
                  eri_time_reverse_defect=float(np.max(abs(gtr - eri))),
                  hamiltonian_time_reverse_operator_norm_bound=bound,
                  original_reference=original, exact_reference=exact,
                  wall_seconds=time.monotonic() - started)
    write_json(directory / "reference_time_reversal.json", report)
    print("AG_TIME_REVERSAL_DONE", "original_leak", original["leakage_norms"],
          "exact_leak", exact["leakage_norms"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    main(parser.parse_args().directory.resolve())
