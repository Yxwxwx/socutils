from pyscf import gto, lib
import numpy as np

from socutils.dmrg import DMRGCI
from socutils.mcscf import zmcscf
from socutils.scf import spinor_hf


mol = gto.M(
    atom="F 0 0 0",
    basis="dyallv3z",
    charge=-1,
    spin=0,
    verbose=4,
    max_memory=1000,
)

mf = spinor_hf.SCF(mol).x2camf().cholesky(tau=1e-8)
mf.conv_tol = 1e-12
mf.max_cycle = 200
mf.kernel()
if not mf.converged:
    raise RuntimeError("closed-shell F- X2CAMF reference did not converge")

initial_mo = np.array(mf.mo_coeff, copy=True)
mol.charge = 0
mol.spin = 1

ncas = 16
nelec = 7
nroots = 6
scratch = lib.param.TMPDIR + "/dmrg_scratch"

solver = DMRGCI(mol).init(
    ncas=ncas,
    nelecas=nelec,
    nroots=nroots,
    scratch=scratch,
    checkpoint_dir="dmrg_checkpoint",
    n_threads=16,
    stack_memory=mol.max_memory,  # MB
)

mc = zmcscf.CASSCF(mf, ncas=ncas, nelecas=nelec)
mc.fcisolver = solver
mc.state_average_(np.ones(nroots) / nroots)
solver = mc.fcisolver
mc.callback = solver.restart_scheduler_()

mc.kernel(mo_coeff=initial_mo)
solver.close()
