"""Fresh UNF3 X2CAMF-HF -> one PM pass -> original second-order DMRG-SCF.

Run with the project .venv and a new state directory, for example:
    .venv/bin/python examples/16-unf3-spinor-uno-pm.py --state-dir /tmp/unf3-pm-simple
    .venv/bin/python examples/16-unf3-spinor-uno-pm.py --variant split --state-dir /tmp/unf3-pm-split
    .venv/bin/python examples/16-unf3-spinor-uno-pm.py --full-blocks --state-dir /tmp/unf3-pm-cav

Both variants keep the R=1.750 ah200 input's geometry, basis, CAS, M,
DMRG schedule, orbital ordering, optimizer, and convergence settings.
The split variant changes only the explicit PM occupation thresholds.
--full-blocks additionally applies core PM and virtual SCDM once before DMRG.
"""

import argparse
from pathlib import Path

from pyscf import gto, lib
from socutils.dmrg import DMRGCI
from socutils.lo.uno_spinor import localize_blocks, sort_orbitals
from socutils.mcscf import zmcscf
from socutils.scf import spinor_hf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("simple", "split"), default="simple")
    parser.add_argument("--full-blocks", action="store_true")
    parser.add_argument("--state-dir", required=True, type=Path)
    args = parser.parse_args()
    args.state_dir.mkdir(parents=True, exist_ok=False)
    lib.num_threads(32)
    mol = gto.M(
        atom=[("U", (0, 0, 0)), ("N", (0, 0, 1.75)),
              ("F", (1.722815, 0, -1.101786)),
              ("F", (-0.861408, -1.492002, -1.101786)),
              ("F", (-0.861408, 1.492002, -1.101786))],
        basis={"U": "dyallv2z", "N": "cc-pvdz", "F": "cc-pvdz"},
        charge=0, spin=0, unit="Angstrom", verbose=4, max_memory=500_000,
    )
    mol.precision = 1e-15
    mf = spinor_hf.SCF(mol).x2camf()
    mf.chkfile = str(args.state_dir / "hf.chk")
    mf.conv_tol = 1e-10
    mf.max_cycle = 200
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("X2CAMF-HF did not converge")

    low, high = ((0.0, 0.0) if args.variant == "simple" else (0.05, 0.95))
    prepare = localize_blocks if args.full_blocks else sort_orbitals
    options = {} if args.full_blocks else {"do_loc": True}
    lo_coeff, lo_occ, lo_energy, nactorb, nactelec = prepare(
        mol, mf.mo_coeff.copy(), mf.mo_occ.copy(), mf.mo_energy.copy(),
        cas_list=list(range(112, 142)), split_low=low, split_high=high,
        **options,
    )
    assert (nactorb, nactelec) == (30, 14)
    print(f"PM variant={args.variant}; full_blocks={args.full_blocks}; "
          f"active occupation trace={lo_occ[112:142].sum():.12g}")
    print(f"PM active energy-proxy trace={lo_energy[112:142].sum():.12g}")

    solver = DMRGCI(mol).init(
        ncas=nactorb, nelecas=nactelec, nroots=1,
        max_bond_dimension=1000, tol=1e-8,
        scratch=args.state_dir / "dmrg_scratch",
        schedule_thrd_max=1e-16,
        checkpoint_dir=args.state_dir / "dmrg_checkpoint",
        n_threads=32, stack_memory=200_000,
        orbital_ordering="original", final_one_site=False,
    )
    solver.maxIter = 60
    mc = zmcscf.CASSCF(mf, ncas=nactorb, nelecas=nactelec)
    mc.fcisolver = solver
    mc.mo_coeff = lo_coeff
    mc.max_cycle_macro = 200
    mc.max_stepsize = 0.2
    mc.conv_tol = 1e-8
    mc.conv_tol_grad = 1e-4
    mc.superci_davidson_tol = 1e-8
    mc.superci_davidson_max_space = 500
    mc.superci_davidson_strict = True
    mc.canonicalization = False
    mc.canonicalize_ = False
    mc.natorb = False
    mc.chkfile = str(args.state_dir / "mcscf.chk")
    mc.callback = solver.restart_scheduler_()
    try:
        mc.second_order()
        print(f"MCSCF converged={mc.converged}; DMRG converged={solver.converged}; E={mc.e_tot:.15f}")
    finally:
        solver.close()


if __name__ == "__main__":
    main()
