# SPDX-License-Identifier: GPL-3.0-or-later
"""Short C input; run on gpu01 with python -m tests.msficnevpt2.carbon_input.

For restartable references and the complete audit use the sibling carbon.py.
This example writes only to its own carbon_input_final_run directory.
"""

import numpy as np
from pyscf import gto, lib
from socutils.dmrg import DMRGCI
from socutils.mcscf import zmcscf
from socutils.mrpt import prepare_msfic, solve_msfic
from socutils.scf import spinor_hf

from .carbon import audit_reference
from .carbon_reference import DIRECTORY, canonicalize_npdm_reference

output = DIRECTORY.parent / "carbon_input_final_run"
output.mkdir(exist_ok=True)
mol = gto.M(
    atom="C 0 0 0",
    basis={"C": gto.basis.load("ano@4s3p2d1f", "C")},
    charge=4,
    spin=0,
    cart=False,
    verbose=4,
    max_memory=120000,
)
mf = spinor_hf.SCF(mol).x2camf()
mf.chkfile, mf.conv_tol, mf.max_cycle = str(output / "hf.chk"), 1e-12, 200
mf.kernel()
assert mf.converged
mol.charge, mol.spin = 0, 2
solver = DMRGCI(mol).init(
    ncas=8,
    nelecas=4,
    nroots=15,
    bond_dims=[64],
    noises=[1e-5] * 2 + [1e-6] * 2 + [0.0] * 12,
    thrds=[1e-24],
    n_sweeps=16,
    tol=1e-13,
    dav_def_max_size=100,
    n_threads=16,
    stack_memory=8192,
    random_seed=1234,
    orbital_ordering="original",
    final_one_site=False,
    scratch=lib.param.TMPDIR + "/carbon_msfic_input",
    checkpoint_dir=output / "dmrg_checkpoint",
)
mc = zmcscf.CASSCF(mf, ncas=8, nelecas=4)
mc.fcisolver = solver
mc.state_average_(np.ones(15) / 15)
mc.canonicalization = mc.canonicalize_ = mc.natorb = False
mc.orbital_symmetry = None
mc.conv_tol, mc.conv_tol_grad, mc.max_cycle_macro = 1e-11, 1e-6, 100
mc.chkfile = str(output / "mcscf.chk")
mc.callback = mc.fcisolver.restart_scheduler_()
try:
    mc.second_order()
    assert mc.converged and mc.fcisolver.converged
    roots, _, _ = audit_reference(mc, mc.fcisolver)
    canonicalize_npdm_reference(mc)
    common = prepare_msfic(
        mc, sa_roots=range(15), sa_weights=np.ones(15) / 15, model_roots=roots
    )
    for ansatz in ("ss_sr", "ms_mr"):
        result = solve_msfic(common, ansatz=ansatz, shift=0.2)
        print(ansatz, "energy for each state")
        for state, energy in enumerate(result.energies):
            print(f"  State {state} E = {energy:.14f}")
        print("spread/cm-1 =", result.spread * 219474.63137)
finally:
    mc.fcisolver.close()
