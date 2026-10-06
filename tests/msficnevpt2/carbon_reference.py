# SPDX-License-Identifier: GPL-3.0-or-later
"""Isolated, restartable C SA15 DMRG-SCF reference (no CD, DF, or KR).

Run on gpu01 from the repository root with ``python -m
tests.msficnevpt2.carbon_reference``. Exact CAS is only a subsequent oracle.
"""

import os
from pathlib import Path

import numpy as np
from pyscf import gto, lib, scf
from socutils.dmrg import DMRGCI
from socutils.mcscf import zmcscf
from socutils.scf import spinor_hf

DIRECTORY = Path(__file__).resolve().parent / "carbon_tight_run"


def canonicalize_npdm_reference(mc):
    """QR-gauge owned MPS copies, with no truncation or new CI optimization.

    MultiMPS root splitting may leave a gauge that loses about 1e-10 in
    subsequent density-matrix decompositions. Native pyblock2 algebra QR
    preserves each state and its phase. Restore dot=2 before Expect: its
    one-site path gives invalid MKL calls for this SGF reference. This does
    not change the DMRG two-site schedule or the persistent checkpoint.
    """
    from pyblock2.algebra.io import MPSTools

    solver = mc.fcisolver
    original = solver.kets
    copies = []
    for root, ket in enumerate(original):
        working = solver.driver.copy_mps(ket, tag=f"C_MS_QR_IN_{root}")
        solver.driver.adjust_mps(working, dot=1)
        native = MPSTools.from_block2(working)
        copied = MPSTools.to_block2(
            native, solver.driver.basis, center=0, tag=f"C_MS_QR_NPDM_{root}"
        )
        solver.driver.adjust_mps(copied, dot=2)
        copies.append(copied)
    solver.kets = copies
    mc.ci = solver.ci = list(copies)
    return original


def reference(directory=DIRECTORY, bond_dim=64):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # PySCF's ANO library is ANO-RCC. Explicit VTZP contraction: 4s3p2d1f.
    basis = gto.basis.load("ano@4s3p2d1f", "C")
    mol = gto.M(
        atom="C 0 0 0",
        basis={"C": basis},
        charge=4,
        spin=0,
        cart=False,
        verbose=4,
        max_memory=120000,
    )
    mf = spinor_hf.SCF(mol).x2camf()
    mf.chkfile = str(directory / "hf.chk")
    saved_path = directory / "reference.npz"
    if saved_path.exists():
        _, hf = scf.chkfile.load_scf(mf.chkfile)
        mf.__dict__.update(hf)
        mf.converged = True
    else:
        mf.conv_tol, mf.max_cycle = 1e-12, 200
        mf.kernel()
        if not mf.converged:
            raise RuntimeError("C4+ X2CAMF HF failed")
    mol.charge, mol.spin = 0, 2
    solver = DMRGCI(mol).init(
        ncas=8,
        nelecas=4,
        nroots=15,
        bond_dims=[bond_dim],
        noises=[1e-5] * 2 + [1e-6] * 2 + [0.0] * 12,
        thrds=[1e-24],
        n_sweeps=16,
        tol=1e-13,
        dav_def_max_size=100,
        random_seed=1234,
        n_threads=int(os.environ.get("OMP_NUM_THREADS", "16")),
        stack_memory=8192,
        orbital_ordering="original",
        final_one_site=False,
        scratch=Path(lib.param.TMPDIR) / "carbon_msfic",
        checkpoint_dir=directory / "dmrg_checkpoint",
    )
    mc = zmcscf.CASSCF(mf, ncas=8, nelecas=4)
    mc.fcisolver = solver
    mc.canonicalization = mc.canonicalize_ = mc.natorb = False
    mc.orbital_symmetry = None
    mc.conv_tol, mc.conv_tol_grad = 1e-11, 1e-6
    mc.max_cycle_macro = 100
    mc.chkfile = str(directory / "mcscf.chk")
    if saved_path.exists():
        with np.load(saved_path) as saved:
            mf.with_x2c.hcore = saved["hcore_ao"].copy()
            solver.restore_checkpoint(
                saved["h1e"],
                saved["eri"],
                8,
                4,
                ecore=float(saved["ecore"]),
                nroots=15,
                max_memory=mol.max_memory,
            )
            mc.mo_coeff = saved["mo_coeff"].copy()
            mc.e_states = saved["e_states"].copy()
            mc.e_tot, mc.e_cas = float(np.mean(mc.e_states)), solver.e_cas
            mc.ci = list(solver.kets)
            mc.converged = True
    else:
        mc.state_average_(np.ones(15) / 15)
        solver = mc.fcisolver
        mc.callback = solver.restart_scheduler_()
        try:
            assert getattr(mf, "with_df", None) is None
            assert solver.kramers_adapter is None
            mc.second_order()
            if not mc.converged or not solver.converged:
                raise RuntimeError("C SA15 DMRG-SCF did not converge")
            snap = solver.checkpoint_hamiltonian
            np.savez(
                saved_path,
                mo_coeff=mc.mo_coeff,
                e_states=mc.e_states,
                h1e=snap["h1e"],
                eri=snap["eri"],
                ecore=snap["ecore"],
                hcore_ao=mf.get_hcore(),
                overlap_ao=mf.get_ovlp(),
                fingerprint=snap["hamiltonian_sha256"],
                gradient=mc.final_orbital_gradient_norm,
                bond_dim=bond_dim,
            )
        except BaseException:
            solver.close()
            raise
    print("C SA15 reference energies", np.asarray(mc.e_states), flush=True)
    return mc, solver


if __name__ == "__main__":
    mc, solver = reference()
    solver.close()
