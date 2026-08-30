#!/usr/bin/env python
"""Cl CAS(7,16), six-root DMRG-SCF with Super-CI or Super-CIPT DIIS.

The default reproduces the supplied unrestricted-spinor Cl input with the
new Super-CIPT optimizer.  Add ``--kramers`` to use a pair-resolved KRHF
reference and the Kramers-restricted DMRG/result path.  The two modes must not
be mixed: appending ``kramers_restricted()`` to orbitals from the general SCF
does not make arbitrary degenerate mixtures into phase-resolved pairs.
"""

import argparse
import os

import numpy as np
from pyscf import gto, lib

from socutils.dmrg import DMRGCI
from socutils.mcscf import zmcscf
from socutils.scf import spinor_hf


def main(*, optimizer="supercipt", kramers=False, max_cycle=50, n_threads=16):
    mol = gto.M(
        atom="Cl 0 0 0",
        basis="dyallv3z",
        charge=-1,
        spin=0,
        verbose=4,
        max_memory=1000,
    )

    scf_class = spinor_hf.KRHF if kramers else spinor_hf.SCF
    mf = scf_class(mol).x2camf().cholesky(tau=1e-8)
    mf.conv_tol = 1e-12
    mf.max_cycle = 200
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("closed-shell Cl- X2CAMF reference did not converge")

    initial_mo = np.array(mf.mo_coeff, copy=True)
    mol.charge = 0
    mol.spin = 1

    ncas, nelecas, nroots = 16, 7, 6
    scratch = os.path.join(lib.param.TMPDIR, "cl_cas16_dmrg_scratch")
    checkpoint = os.path.join(lib.param.TMPDIR, "cl_cas16_dmrg_checkpoint")
    solver = DMRGCI(mol).init(
        ncas=ncas,
        nelecas=nelecas,
        nroots=nroots,
        max_bond_dimension=1000,
        tol=1e-8,
        scratch=scratch,
        schedule_thrd_max=1e-16,
        checkpoint_dir=checkpoint,
        n_threads=n_threads,
        stack_memory=mol.max_memory,
    )
    if kramers:
        solver.kramers_restricted()

    mc = zmcscf.CASSCF(mf, ncas=ncas, nelecas=nelecas)
    mc.fcisolver = solver
    mc.state_average_(np.ones(nroots) / nroots)
    mc.mo_coeff = initial_mo
    mc.natorb = False
    mc.canonicalize_ = False
    mc.max_cycle_macro = max_cycle
    mc.max_stepsize = 0.2
    mc.conv_tol = 1e-8
    mc.conv_tol_grad = 1e-4
    mc.superci_davidson_tol = 1e-8
    mc.superci_davidson_max_space = 200
    mc.superci_davidson_strict = True
    mc.callback = mc.fcisolver.restart_scheduler_()

    try:
        if optimizer == "superci":
            mc.superci(use_diis=True)
        else:
            mc.supercipt(use_diis=True)
        print("converged =", mc.converged)
        print("E(%s) = %.15f" % (optimizer, mc.e_tot))
        print("final orbital gradient = %.6e" % mc.final_orbital_gradient_norm)
    finally:
        mc.fcisolver.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--optimizer",
        choices=("supercipt", "superci"),
        default="supercipt",
    )
    parser.add_argument("--kramers", action="store_true")
    parser.add_argument("--max-cycle", type=int, default=50)
    parser.add_argument("--n-threads", type=int, default=16)
    args = parser.parse_args()
    if args.max_cycle <= 0:
        parser.error("--max-cycle must be positive")
    if args.n_threads <= 0:
        parser.error("--n-threads must be positive")
    main(
        optimizer=args.optimizer,
        kramers=args.kramers,
        max_cycle=args.max_cycle,
        n_threads=args.n_threads,
    )
