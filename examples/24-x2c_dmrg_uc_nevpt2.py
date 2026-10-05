#!/usr/bin/env python
"""BH/STO-3G: full-ERI X2C-DMRG-SCF -> full eight-class UC-NEVPT2.

No Cholesky, Kramers restriction, high-order RDM, or imaginary-time method.
NCAS counts individual spinors. Run with ``ulimit -s unlimited``.
This is a small correctness example, not a large-system performance claim.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

from pyscf import gto, lib

from socutils.dmrg import DMRGCI
from socutils.mcscf import zmcscf
from socutils.mrpt import X2CUCNEVPT2
from socutils.scf import spinor_hf


lib.num_threads(1)
mol = gto.M(atom="B 0 0 0; H 0 0 1.232", basis="sto-3g",
            charge=0, spin=0, verbose=4, max_memory=2000)
mf = spinor_hf.SCF(mol).x2camf(with_gaunt=False, with_breit=False)
mf.init_guess = "1e"
mf.conv_tol, mf.max_cycle = 1e-11, 100
mf.kernel()
if not mf.converged:
    raise RuntimeError("X2CAMF HF did not converge")

with TemporaryDirectory(prefix="socutils-x2c-uc-nevpt2-") as scratch:
    solver = DMRGCI(mol).init(
        ncas=6, nelecas=4, nroots=1, max_bond_dimension=64,
        tol=1e-12, schedule_thrd_max=1e-24, final_one_site=False,
        orbital_ordering="original", random_seed=2468,
        n_threads=1, stack_memory=512, scratch=Path(scratch),
    )
    mc = zmcscf.CASSCF(mf, ncas=6, nelecas=4)
    mc.fcisolver = solver
    mc.canonicalization = mc.canonicalize_ = mc.natorb = False
    mc.max_cycle_macro = 50
    mc.conv_tol, mc.conv_tol_grad = 1e-9, 1e-5
    mc.callback = solver.restart_scheduler_()
    try:
        mc.second_order()
        if not mc.converged or not solver.converged:
            raise RuntimeError("X2C-DMRG-SCF did not converge")

        pt = X2CUCNEVPT2(mc)
        pt.response_mode = "full_chain"  # or "external_tuples": active-only occupation blocks
        # Native Linear.solve is the default; this small example also opts
        # into the expensive, independent post-solve global residual audit.
        pt.mps_response_options = dict(max_bond_dimension=128, n_sweeps=6, diagnostic=True)
        pt.run(root=0)
        print("E(MCSCF)   = %.15f" % pt.reference_energy)
        print("E(UC-PT2)  = %.15f" % pt.e_corr)
        print("E(total)   = %.15f" % pt.e_tot)
        print("Equation converged =", pt.converged)
        print("CAS constraint =", pt.diagnostics["cas_constraint"])
        if pt.response_mode == "full_chain":
            print("CAS purity certified =", pt.diagnostics["cas_purity_certified"])
            print("Reference zero-mode fraction =", pt.diagnostics["reference_zero_mode_leakage"])
        print("Global relative residual =", pt.diagnostics["global_relative_residual"])
        for name, energy in pt.sub_eners.items():
            print("  E(%4s) = % .15f" % (name, energy))
    finally:
        # The PT API needs the retained native reference MPS, so close last.
        solver.close()
