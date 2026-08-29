#!/usr/bin/env python
"""Block2-driven two-component CASSCF with the Super-CIPT optimizer.

``supercipt()`` is an explicit alternative to the default full Super-CI path;
calling ``kernel()`` still runs Super-CI.  The example uses a tiny active space
so that it is suitable as a smoke calculation, not as a production DMRG setup.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

from pyscf import gto

from socutils.dmrg import DMRGCI
from socutils.mcscf import zmcscf
from socutils.scf import spinor_hf


mol = gto.M(
    atom="H 0 0 0; F 0.35 0.27 0.8035",
    basis="sto-3g",
    verbose=4,
)
mf = spinor_hf.SCF(mol).x2camf(
    with_gaunt=False, with_breit=False
).cholesky(tau=1e-10)
mf.kernel()

with TemporaryDirectory(prefix="socutils-supercipt-") as scratch:
    solver = DMRGCI(mol).init(
        ncas=4,
        nelecas=2,
        nroots=1,
        bond_dims=[32] * 8,
        noises=[0.0] * 8,
        thrds=[1e-20] * 8,
        n_sweeps=8,
        tol=1e-12,
        scratch=Path(scratch),
        n_threads=1,
        stack_memory=256,
        random_seed=2468,
        npdm_site_type=2,
    )
    mc = zmcscf.CASSCF(mf, ncas=4, nelecas=2)
    mc.fcisolver = solver
    mc.conv_tol = 1e-9
    mc.conv_tol_grad = 1e-5
    mc.max_cycle_macro = 24
    mc.max_stepsize = 0.2
    mc.supercipt(use_diis=True, use_cderi=True)

    print("E(Super-CIPT DMRG-SCF) = %.12f" % mc.e_tot)
    print("final orbital gradient = %.6e" % mc.final_orbital_gradient_norm)
    print("macroiterations = %d" % len(mc.supercipt_history))
    solver.close()
