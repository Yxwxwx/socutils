#!/usr/bin/env python
"""Small no-Kramers X2C-DMRG-SCF -> strict-SI Wick SC-NEVPT2 input.

The BH/STO-3G CAS(4e,6 spinors) calculation is intentionally small enough
to run as a smoke example while retaining nonzero active-space 1--4 RDMs.
``ncas`` counts individual spinors, not spatial orbitals or Kramers pairs.
The perturbation step uses dense spinor MO integrals and asks the live
Block2 ``DMRGCI`` object for the RDMs, so it must run before
``solver.close()``.

The dense complex128 4-RDM alone occupies ``16 * ncas**8`` bytes (about
6.41 GiB at ``ncas=12``).  Increase the active space only after budgeting
for the RDM plus Block2 work memory.  The current implementation is
no-Kramers and does not support frozen spinors.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

from pyscf import gto, lib

from socutils.dmrg import DMRGCI
from socutils.mcscf import zmcscf
from socutils.mrpt import WickX2CSCNEVPT2
from socutils.scf import spinor_hf


NCAS = 6
NELECAS = 4
# This controls both PySCF/BLAS and Block2.  Increase it for a larger input.
N_THREADS = 1


def main():
    lib.num_threads(N_THREADS)
    mol = gto.M(
        atom="B 0 0 0; H 0 0 1.232",
        basis="sto-3g",
        charge=0,
        spin=0,
        verbose=4,
        max_memory=2000,
    )

    # General complex-spinor SCF: do not replace this with KRHF or call
    # kramers_restricted().  Gaunt/Breit are disabled only to keep this smoke
    # input minimal; X2CAMF remains the two-component relativistic reference.
    mf = spinor_hf.SCF(mol).x2camf(with_gaunt=False, with_breit=False)
    mf.init_guess = "1e"
    mf.conv_tol = 1e-11
    mf.max_cycle = 100
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("X2CAMF-SCF did not converge")

    with TemporaryDirectory(prefix="socutils-x2c-sc-nevpt2-") as scratch:
        solver = DMRGCI(mol).init(
            ncas=NCAS,
            nelecas=NELECAS,
            nroots=1,
            bond_dims=[32] * 8,
            noises=[0.0] * 8,
            thrds=[1e-20] * 8,
            n_sweeps=8,
            tol=1e-12,
            scratch=Path(scratch),
            n_threads=N_THREADS,
            stack_memory=512,
            dav_max_iter=1000,
            random_seed=2468,
            npdm_site_type=0,
        )
        mc = zmcscf.CASSCF(mf, ncas=NCAS, nelecas=NELECAS)
        mc.fcisolver = solver
        mc.max_cycle_macro = 30
        mc.max_stepsize = 0.1
        mc.conv_tol = 1e-9
        mc.conv_tol_grad = 1e-5
        mc.superci_davidson_tol = 1e-10
        mc.superci_davidson_max_space = 100
        mc.superci_davidson_strict = True

        # Keep the optimized MPS and active orbitals in exactly the same
        # basis.  SC-NEVPT2 semicanonicalizes only core/virtual orbitals later.
        mc.canonicalization = False
        mc.canonicalize_ = False
        mc.natorb = False
        mc.callback = solver.restart_scheduler_()

        try:
            mc.superci()
            if not mc.converged or not solver.converged:
                raise RuntimeError("X2C-DMRG-SCF did not converge")

            pt = WickX2CSCNEVPT2(mc)
            pt.kernel(root=0, denominator_mode="strict_si")

            print("E(X2C-DMRG-SCF) = %.15f" % pt.reference_energy)
            print("E(SC-NEVPT2)    = %.15f" % pt.e_corr)
            print("E(total)        = %.15f" % pt.e_tot)
            print("strict-SI compatible =", pt.strict_si_compatible)
            for key, energy in pt.sub_eners.items():
                print("  E(%4s) = % .15f" % (key, energy))
        finally:
            # The PT call above needs solver.driver and solver.kets to form
            # the 1--4 RDMs.  Close Block2 only after SC-NEVPT2 is finished.
            solver.close()


if __name__ == "__main__":
    main()
