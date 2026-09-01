#!/usr/bin/env python
"""Cl CAS(5e,12 spinors) SA-X2C-DMRG-SCF -> QD-SC-NEVPT2.

A closed-shell Cl- X2CAMF calculation supplies the initial orbitals for the
neutral Cl target.  Six roots are optimized with equal state-average weights
and then coupled by the dense Wick QD-SC-NEVPT2 driver.  ``ncas`` counts
individual spinors, not spatial orbitals or Kramers pairs.

This is a production-scale example.  One dense complex128 4-RDM at
``ncas=12`` occupies about 6.41 GiB, before validation temporaries, dense MO
integrals, Block2 memory, and transition RDMs are included.  Set ``TMPDIR`` to
large node-local storage and adjust the memory/thread constants below for the
machine.  The perturbation calculation must finish before ``solver.close()``
because it uses all six live Block2 MPS roots.  Frozen spinors are not
currently supported.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytblis
from pyscf import gto, lib

from socutils.dmrg import DMRGCI
from socutils.mcscf import zmcscf
from socutils.mrpt import WickX2CQDSCNEVPT2
from socutils.scf import spinor_hf


NCAS = 12
NELECAS = 5
NROOTS = 6

# ``van_vleck`` selects the Hermitian canonical/HQD representation.  Use
# ``bloch`` for the non-Hermitian source-row representation instead; both
# complete effective Hamiltonians are retained by either run.
QD_TYPE = "van_vleck"
CONTRACTION_BACKEND = "pytblis"

N_THREADS = 16
PYSCF_MAX_MEMORY_MB = 120_000
BLOCK2_STACK_MEMORY_MB = 16_000
RDM_WORK_MEMORY_BYTES = 2 * 2**30


def main():
    lib.num_threads(N_THREADS)
    pytblis.set_num_threads(N_THREADS)
    mol = gto.M(
        atom="Cl 0 0 0",
        basis="cc-pvtz-dk",
        charge=-1,
        spin=0,
        verbose=4,
        max_memory=PYSCF_MAX_MEMORY_MB,
    )

    # Use the closed-shell anion only to obtain a stable initial orbital set.
    # The following DMRG-SCF and perturbation calculations target neutral Cl.
    mf = spinor_hf.SCF(mol).x2camf()
    mf.conv_tol = 1e-12
    mf.max_cycle = 200
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("closed-shell Cl- X2CAMF reference did not converge")
    initial_mo = np.array(mf.mo_coeff, copy=True)

    mol.charge = 0
    mol.spin = 1
    if mol.nelectron != 17:
        raise RuntimeError("neutral Cl target must contain 17 electrons")

    with TemporaryDirectory(
        prefix="socutils-cl-x2c-qd-sc-nevpt2-",
        dir=lib.param.TMPDIR,
    ) as temporary:
        scratch = Path(temporary)
        solver = DMRGCI(mol).init(
            ncas=NCAS,
            nelecas=NELECAS,
            nroots=NROOTS,
            max_bond_dimension=256,
            tol=1e-11,
            schedule_thrd_max=1e-14,
            scratch=scratch / "dmrg_scratch",
            checkpoint_dir=scratch / "dmrg_checkpoint",
            n_threads=N_THREADS,
            stack_memory=BLOCK2_STACK_MEMORY_MB,
            dav_max_iter=1000,
            random_seed=2468,
            npdm_site_type=2,
            npdm_cutoff=1e-24,
        )

        mc = zmcscf.CASSCF(mf, ncas=NCAS, nelecas=NELECAS)
        mc.fcisolver = solver
        mc.state_average_(np.full(NROOTS, 1.0 / NROOTS))
        mc.mo_coeff = initial_mo
        mc.natorb = False
        mc.canonicalization = False
        mc.canonicalize_ = False
        mc.max_cycle_macro = 50
        mc.max_stepsize = 0.2
        mc.conv_tol = 1e-8
        mc.conv_tol_grad = 1e-4
        mc.superci_davidson_tol = 1e-8
        mc.superci_davidson_max_space = 200
        mc.superci_davidson_strict = True

        # state_average_ installs the solver view that owns all model roots.
        solver = mc.fcisolver
        mc.callback = solver.restart_scheduler_()

        try:
            mc.superci()
            solver = mc.fcisolver
            if not mc.converged or not solver.converged:
                raise RuntimeError("six-root SA-X2C-DMRG-SCF did not converge")
            root_energies = np.asarray(mc.e_states, dtype=float)
            if root_energies.shape != (NROOTS,) or not np.all(
                np.isfinite(root_energies)
            ):
                raise RuntimeError("CASSCF did not retain six finite root energies")

            pt = WickX2CQDSCNEVPT2(mc, qd_type=QD_TYPE)
            pt.rdm_work_memory = RDM_WORK_MEMORY_BYTES
            pt.kernel(
                roots=range(NROOTS),
                denominator_mode="strict_si",
                contraction_backend=CONTRACTION_BACKEND,
            )

            print("E(SA-X2C-DMRG-SCF) = %.15f" % mc.e_tot)
            print("reference root energies =", pt.reference_energies)
            print("SS-corrected root energies =", np.diag(pt.h_eff_bloch).real)
            print("H_eff(Bloch) =\n", pt.h_eff_bloch)
            print("H_eff(Van Vleck) =\n", pt.h_eff_van_vleck)
            print("selected qd_type =", pt.qd_type)
            print("QD eigenvalues =", pt.e_qd)
            print("QD eigenvectors =\n", pt.eigenvectors)
            for key, matrix in pt.h2_by_subspace.items():
                print("  H2(%4s) =\n%s" % (key, matrix))
        finally:
            # State and transition RDMs require solver.driver and solver.kets.
            solver.close()


if __name__ == "__main__":
    main()
