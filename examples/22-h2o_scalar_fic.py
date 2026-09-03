#!/usr/bin/env python
"""Minimal H2O scalar FIC-NEVPT2 input; no SC grouping is involved."""

from _scalar_nevpt2 import run_scalar_nevpt2


run_scalar_nevpt2(
    atom="O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
    basis="ccpvdz",
    ncas=6,
    nelecas=8,
    method="fic",
    threads=16,
    max_memory_mb=480_000,
    stack_memory_mb=16_000,
)
