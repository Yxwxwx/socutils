#!/usr/bin/env python
"""Equal-weight Kramers-pair X2C-DMRG-SCF for a one-electron doublet."""

from pyscf import gto

from socutils.dmrg import DMRGCI
from socutils.mcscf import zmcscf
from socutils.scf import spinor_hf


mol = gto.M(
    atom="H 0 0 0",
    basis="6-31g",
    spin=1,
    verbose=4,
    max_memory=1000,
)
mf = spinor_hf.KRHF(mol).x2camf().cholesky(tau=1e-10)
mf.kernel()

# ncas counts spinors.  The odd-electron state must include the complete
# lowest Kramers doublet, not one arbitrarily oriented member of it.
solver = DMRGCI(mol).init(
    ncas=2,
    nelecas=1,
    nroots=2,
    schedule_mode="pyscf",
    max_bond_dimension=64,
    start_bond_dimension=16,
    schedule_thrd_max=1e-16,
    tol=1e-10,
    n_threads=1,
    stack_memory=512,
).kramers_restricted()

mc = zmcscf.CASSCF(mf, ncas=2, nelecas=1)
mc.fcisolver = solver
mc.state_average_([0.5, 0.5])
# state_average_ installs a solver view; bind continuation to that active
# solver rather than to the pre-state-average object.
solver = mc.fcisolver
# The current gradient controls whether the next CI solve uses the short,
# one-site max-M restart schedule.  initial_dmrg() installs this automatically.
mc.callback = solver.restart_scheduler_()
mc.canonicalize_ = False
mc.kernel()

print("E(KR-X2C-DMRG-SCF) = %.12f" % mc.e_tot)
print("Kramers diagnostics:", solver.kramers_diagnostics)
solver.close()
