#!/usr/bin/env python
"""Kramers-preserving Boys localization without adjacent-pair assumptions."""

from pyscf import gto

from socutils.lo import localize_boys_kramers
from socutils.scf import spinor_hf


mol = gto.M(
    atom="H 0 0 -0.7; H 0 0 0.7",
    basis="sto-3g",
    verbose=4,
)
mf = spinor_hf.KRHF(mol).x2camf(
    with_gaunt=False,
    with_breit=False,
)
mf.kernel()

# Deliberately make the partners nonadjacent.  The localizer derives their
# indices and phases from AO time reversal instead of assuming (0,1), (2,3).
mo = mf.mo_coeff[:, [0, 2, 1, 3]]
localized, info = localize_boys_kramers(
    mol,
    mo,
    distance_threshold=None,
    return_info=True,
)
print("Boys objective: %.10f -> %.10f" % (
    info.initial_objective,
    info.final_objective,
))
print("Kramers rotation residual: %.3e" % info.symmetry_residual)
