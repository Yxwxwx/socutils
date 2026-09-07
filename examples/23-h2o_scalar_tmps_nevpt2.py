#!/usr/bin/env python
"""Minimal H2O fully uncontracted t-MPS-NEVPT2 input; stdout only."""

from _scalar_nevpt2 import run_scalar_nevpt2


run_scalar_nevpt2(
    atom="O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
    basis="ccpvdz",
    ncas=6,
    nelecas=8,
    method="tmps",
    threads=16,
    factorized_overlap_backend="dense_csf",
)
