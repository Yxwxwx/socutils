# SPDX-License-Identifier: GPL-3.0-or-later
"""Cu SA12 second-order FCI-CASSCF fallback, followed by exact MPS export."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from pyscf import lib
from pyscf.fci import fci_dhf_slow

from socutils.mcscf import zmcscf
from socutils.mcscf.zmc_utils import _resolve_kramers_mode
from .ag_audit import write_json
from .exact_mps import main as export_mps
from .exact_spectra import MANIFOLDS, WEIGHTS
from tests.nevpt2_mps_response.f_atom import active_action_batch, _active_action_links


def configure(mf, checkpoint, *, ncas=22, nelec=11, weights=WEIGHTS):
    mc = zmcscf.CASSCF(mf, ncas, nelec)
    mc.verbose = 5
    solver = mc.fcisolver
    solver.verbose = 5
    solver.spin_square = None
    solver.nroots = len(weights)
    solver.conv_tol = 1e-12
    solver.conv_tol_residual = 1e-11
    solver.lindep = 1e-24
    solver.max_cycle, solver.max_space = 200, 40
    solver.max_memory, solver.threads = mf.max_memory, lib.num_threads()
    mc.fcisolver = solver
    mc.state_average_(weights)
    mc.canonicalization = mc.canonicalize_ = mc.natorb = False
    mc.orbital_symmetry, mc.superci_bfgs = None, False
    mc.max_cycle_macro = 100
    mc.conv_tol, mc.conv_tol_grad = 1e-8, 1e-4
    mc.superci_davidson_tol, mc.superci_davidson_max_space = 1e-8, 500
    mc.superci_davidson_strict = True
    mc.second_order_max_rotation = 1.0
    mc.second_order_micro_step_tol = 1e-4
    mc.chkfile = str(checkpoint)
    return mc


def save_arrays(path, **arrays):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    temporary.replace(path)


def main(directory, source):
    directory, source = directory.resolve(), source.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    state = directory / "dmrg_state"
    state.mkdir(exist_ok=True)
    sys.path.insert(0, str(source.parents[1]))
    from kr_m2000_workflow import hf_reference, initial_orbitals

    lib.num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "16")))
    work = directory / "ms_nevpt2_exact"
    if (state / "READY.json").exists():
        ready = json.loads((state / "READY.json").read_text())
        if ready["source"] != str(source) or not ready["converged"]:
            raise ValueError("FCI-CASSCF checkpoint provenance mismatch")
        export_mps(work)
        return
    mol, mf = hf_reference("Cu", directory)
    mc = configure(mf, state / "mcscf.chk")
    if not _resolve_kramers_mode(mc):
        raise ValueError("expected the original KR orbital optimization")
    mo = initial_orbitals("Cu", mc, directory, from_hf=True)
    expected = json.loads((source / "orbital_initial_guess.json").read_text())
    selected = json.loads((directory / "orbital_initial_guess.json").read_text())
    if selected["active_indices_0based"] != expected["active_indices_0based"]:
        raise ValueError("active selection changed from original Cu HF-start calculation")
    mc.mo_coeff = mo
    settings = dict(source=str(source), symbol="Cu", basis="unc-ano-rcc",
                    ncas=22, nelec=11, nroots=12, weights=WEIGHTS,
                    optimizer="second_order", ci_solver="matrix-free full CI",
                    ci_residual_target=1e-11, final_residual_limit=1e-10,
                    orbital_kramers=True, cd_df=False, nuclear_model="point")
    write_json(directory / "settings.json", settings)

    def callback(row):
        write_json(directory / "macro_history.json", mc.macro_history)
        write_json(directory / "status.json", dict(stage="CASSCF", **row))

    mc.callback = callback
    write_json(directory / "status.json", dict(stage="CASSCF_INITIAL_CI"))
    mc.second_order()
    if not mc.converged or not np.all(mc.fcisolver.converged):
        raise RuntimeError("Cu FCI-CASSCF did not converge")
    h1e, ecore = mc.get_h1eff(mc.mo_coeff)
    eri = mc.get_h2eff(mc.mo_coeff).reshape((22,) * 4)
    vectors = np.asarray(mc.ci).T
    energies = np.asarray(mc.e_states) - float(np.real(ecore))
    absorbed = fci_dhf_slow.absorb_h1e(h1e, eri, 22, 11, 0.5)
    actions = active_action_batch(absorbed, vectors.T, 22, 11).T
    residuals = np.linalg.norm(actions - vectors * energies, axis=0)
    np.testing.assert_allclose(vectors.conj().T @ vectors, np.eye(12), atol=1e-10, rtol=0)
    if not np.isfinite(residuals).all() or max(residuals) > 1e-10:
        raise RuntimeError(f"final FCI-CASSCF residuals exceed 1e-10: {residuals}")
    digest = hashlib.sha256()
    for array in (h1e, eri, np.asarray(ecore)):
        digest.update(np.ascontiguousarray(array).tobytes())
    hsha = digest.hexdigest()
    save_arrays(state / "reference.npz", mo_coeff=mc.mo_coeff, ci=vectors,
                h1e=h1e, eri=eri, ecore=ecore, e_states=mc.e_states,
                weights=WEIGHTS, hcore_ao=mf.get_hcore(), overlap_ao=mf.get_ovlp(),
                hamiltonian_sha256=hsha)
    work.mkdir(exist_ok=True)
    save_arrays(work / "reference_vectors.npz", h1e=h1e, eri=eri, hamiltonian_sha256=hsha)
    save_arrays(work / "exact_reference_vectors.npz", vectors=vectors, actions=actions,
                active_energies=energies, hamiltonian_sha256=hsha)
    write_json(work / "exact_reference.json", dict(hamiltonian_sha256=hsha,
               ncas=22, nelec=11, nroots=12, converged=[True] * 12,
               residual_norms=residuals, active_energies=energies,
               total_energies=mc.e_states, no_alpha_beta_mapping=True))
    write_json(work / "settings.json", dict(**settings, hamiltonian_sha256=hsha,
               manifolds=MANIFOLDS))
    write_json(state / "READY.json", dict(**settings, converged=True,
               hamiltonian_sha256=hsha, e_states=mc.e_states, residual_norms=residuals))
    print("FCI_CASSCF_REFERENCE_READY", state / "reference.npz", residuals, flush=True)
    _active_action_links.cache_clear()
    export_mps(work)
    write_json(directory / "status.json", dict(stage="PREPARED", tensor_contraction=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    main(args.directory, args.source)
