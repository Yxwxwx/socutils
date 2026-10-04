"""Fixed-orbital UNF3 CASCI: original, simple PM, and split PM from one HF state.

This is the first-stage DMRG sweep comparison. It deliberately does not run
orbital optimization or reuse any old MPS/checkpoint.
"""

import argparse
import gc
import os
import tempfile
import time
from pathlib import Path

import numpy as np
from pyscf import lib, scf
from socutils.dmrg import DMRGCI
from socutils.lo.uno_spinor import sort_orbitals
from socutils.mcscf import zmcscf
from socutils.scf import spinor_hf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-checkpoint", required=True, type=Path)
    parser.add_argument("--result-dir", required=True, type=Path)
    args = parser.parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=False)
    lib.num_threads(32)
    mol, saved = scf.chkfile.load_scf(str(args.hf_checkpoint))
    mol.max_memory = 500_000
    c0 = np.asarray(saved["mo_coeff"])
    n0 = np.asarray(saved["mo_occ"])
    e0 = np.asarray(saved["mo_energy"])
    if mol.nelectron != 126 or c0.shape != (676, 676):
        raise ValueError("checkpoint is not the expected UNF3 2c HF state")
    mf = spinor_hf.SCF(mol).x2camf()
    mf.mo_coeff, mf.mo_occ, mf.mo_energy = c0, n0, e0
    mf.e_tot = float(saved["e_tot"])
    mf.converged = True
    mf.chkfile = str(args.result_dir / "hf.chk")
    if np.max(abs(c0.conj().T @ mf.get_ovlp() @ c0 - np.eye(676))) > 1e-7:
        raise ValueError("HF spinors are not orthonormal in this X2CAMF AO metric")
    print(f"HF source: {args.hf_checkpoint}; E={mf.e_tot:.15f}", flush=True)
    for variant, thresholds in (("original", None), ("simple", (0., 0.)),
                                ("split", (.05, .95))):
        folder = args.result_dir / variant
        folder.mkdir()
        c = c0.copy()
        if thresholds is not None:
            c, _, _, ncas, nelec = sort_orbitals(
                mol, c, n0.copy(), e0.copy(), cas_list=list(range(112, 142)),
                do_loc=True, split_low=thresholds[0], split_high=thresholds[1],
                iprint=0,
            )
            if (ncas, nelec) != (30, 14):
                raise ValueError("PM changed the requested CAS")
        print(f"BEGIN {variant}", flush=True)
        start = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix=f"unf3_{variant}_", dir=os.environ["TMPDIR"]) as scratch:
            solver = DMRGCI(mol).init(
                ncas=30, nelecas=14, nroots=1, max_bond_dimension=1000,
                tol=1e-8, scratch=Path(scratch) / "dmrg_scratch",
                schedule_thrd_max=1e-16,
                checkpoint_dir=folder / "dmrg_checkpoint",
                n_threads=32, stack_memory=200_000,
                orbital_ordering="original", final_one_site=False,
            )
            solver.maxIter = 60
            mc = zmcscf.CASSCF(mf, ncas=30, nelecas=14)
            mc.fcisolver = solver
            mc.mo_coeff = c
            mc.canonicalization = False
            mc.canonicalize_ = False
            mc.natorb = False
            mc.chkfile = str(folder / "casci.chk")
            try:
                energy = mc.casci(mo_coeff=c)[0]
                info = solver.convergence_info
                print(f"END {variant}: E={energy:.15f} converged={solver.converged} "
                      f"sweeps={info.get('sweeps')} energy_change={info.get('energy_change')} "
                      f"elapsed={time.perf_counter() - start:.2f}s", flush=True)
            finally:
                solver.close()
        del mc, solver
        gc.collect()


if __name__ == "__main__":
    main()
