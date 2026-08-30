#!/usr/bin/env python
"""Nd3+(H2O)8 CAS(3,14), 52-root Kramers Super-CIPT with DIIS.

This is the current-socutils form of the contributed
``v2z_svp_g_CASSCF-diis.py`` input.  It keeps the original one-cycle Nd6+
closed-shell reference construction and active-orbital swap, while using the
validated CASSCF/state-average interfaces instead of replacing CASCI internals.
"""

import argparse

import numpy as np
from pyscf import fci, gto

from socutils.mcscf import zmcscf
from socutils.scf import spinor_hf
from socutils.tools import analyze_casscf_spinors


ND_H2O_8 = """
Nd        0.000000    0.000000    0.000000
O         2.098371   -0.914223    0.831267
O         0.865772    1.745495    1.471619
O        -1.813898    0.514733    1.543559
O        -0.573834   -2.154727    0.992034
O         0.702164   -1.651828   -1.653826
O        -2.057324   -0.688404   -1.120258
O        -0.983865    2.067912   -0.840781
O         1.747102    1.104179   -1.303462
H         2.987817   -0.592919    0.624215
H         1.528647    1.644187    2.169537
H        -1.744228    0.962922    2.398816
H        -0.325886   -3.027904    0.655306
H         1.547985   -2.122836   -1.657496
H        -2.554447   -0.181919   -1.778565
H        -0.673617    2.586912   -1.596842
H         2.064968    0.841318   -2.179090
H        -2.738327    0.244814    1.445060
H         0.592885    2.674214    1.483354
H        -1.805246    2.481436   -0.539125
H        -2.472976   -1.561150   -1.072260
H        -1.046363   -2.303967    1.823718
H         2.200961   -1.594334    1.512335
H         2.209356    1.923599   -1.076355
H         0.234709   -1.917003   -2.459045
"""


def main(
    *,
    dry_run=False,
    max_cycle=40,
    init_guess="atom",
    max_memory=200000,
):
    basis = {
        "Nd": gto.load("dyallv2z", "Nd"),
        "O": gto.load("def2-svp", "O"),
        "H": gto.load("def2-svp", "H"),
    }
    mol = gto.M(
        atom=ND_H2O_8,
        basis=basis,
        charge=6,
        spin=0,
        verbose=5,
        nucmod="G",
        max_memory=int(max_memory),
    )
    mf = spinor_hf.KRHF(mol).x2camf(
        with_gaunt=True,
        with_breit=True,
    ).cholesky(tau=1e-8)
    if dry_run:
        mol.charge = 3
        mol.spin = 3
        print(
            "Nd input ready: %d atoms, %d scalar AOs, %d spinors, "
            "ncore=%d, active=%d:%d"
            % (
                mol.natm,
                mol.nao_nr(),
                len(mol.spinor_labels()),
                mol.nelectron - 3,
                mol.nelectron - 3,
                mol.nelectron - 3 + 14,
            )
        )
        return
    mf.max_cycle = 1
    mf.init_guess = init_guess
    mf.kernel()

    initial_mo = np.array(mf.mo_coeff, copy=True)
    initial_mo[:, 134:142], initial_mo[:, 148:156] = (
        initial_mo[:, 148:156].copy(),
        initial_mo[:, 134:142].copy(),
    )
    mol.charge = 3
    mol.spin = 3

    ncas, nelecas, nroots = 14, 3, 52
    mc = zmcscf.CASSCF(mf, ncas=ncas, nelecas=nelecas)
    solver = fci.fci_dhf_slow.FCI(mol)
    solver.pspace_size = 0
    solver.max_cycle = 2000
    solver.conv_tol = 1e-15
    mc.fcisolver = solver
    mc.state_average_(np.full(nroots, 1.0 / nroots))
    mc.mo_coeff = initial_mo
    mc.natorb = False
    mc.canonicalize_ = False
    mc.max_cycle_macro = int(max_cycle)
    mc.max_stepsize = 0.2
    mc.conv_tol = 1e-8
    mc.conv_tol_grad = 1e-3

    mc.supercipt(use_diis=True)
    print("converged =", mc.converged)
    print("E(Super-CIPT) = %.12f" % mc.e_tot)
    analyze_casscf_spinors(mc, threshold=0.05, mo_type="active")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="build the real Nd geometry/bases and validate CAS dimensions only",
    )
    parser.add_argument(
        "--max-cycle",
        type=int,
        default=40,
        help="maximum Super-CIPT macroiterations (default: 40)",
    )
    parser.add_argument(
        "--init-guess",
        default="atom",
        help="reference-SCF initial guess (default: atom; use 1e for a low-memory smoke test)",
    )
    parser.add_argument(
        "--max-memory",
        type=int,
        default=200000,
        help="PySCF memory limit in MB (default: 200000)",
    )
    args = parser.parse_args()
    if args.max_cycle <= 0:
        parser.error("--max-cycle must be positive")
    if args.max_memory <= 0:
        parser.error("--max-memory must be positive")
    main(
        dry_run=args.dry_run,
        max_cycle=args.max_cycle,
        init_guess=args.init_guess,
        max_memory=args.max_memory,
    )
