#!/usr/bin/env python
"""Shared machinery for the minimal scalar-to-spinor NEVPT2 examples.

This is example support code, not a molecule-specific implementation in
socutils.mrpt. It keeps the three user inputs short while performing only
the calculation needed for the comparison: exact spatial CASSCF, scalar
integral expansion, one fixed-integral spinor DMRG solve, 1--4 RDM
construction, and one NEVPT2 calculation. It writes no JSON or NPZ files.
"""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
from pyscf import ao2mo, gto, lib, mcscf, scf

from socutils.dmrg import DMRGCI
from socutils.mrpt import (
    WickX2CFICNEVPT2,
    WickX2CSCNEVPT2,
    init_spinor_eris,
)


SUBSPACES = ("ijrs", "rsi", "ijr", "rs", "ij", "ir", "r", "i")


def _spatial_reference(mol, ncas, nelecas, method):
    """Return an exact spatial CASSCF and its Block2 Wick correction."""

    mf = scf.RHF(mol)
    mf.conv_tol = 1.0e-13
    mf.conv_tol_grad = 1.0e-9
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("RHF did not converge")

    mc = mcscf.CASSCF(mf, ncas, nelecas)
    mc.fcisolver.conv_tol = 1.0e-14
    mc.conv_tol = 1.0e-11
    mc.canonicalization = True
    mc.kernel()
    if not mc.converged:
        raise RuntimeError("CASSCF did not converge")

    if method == "sc":
        from pyblock2.icmr.scnevpt2 import WickSCNEVPT2

        pt = WickSCNEVPT2(mc).run()
    else:
        from pyblock2.icmr.icnevpt2_full import WickICNEVPT2

        pt = WickICNEVPT2(mc).run()

    mo = np.asarray(pt.mo_coeff)
    nmo = mo.shape[1]
    h1e = mo.conj().T @ mf.get_hcore() @ mo
    eri = ao2mo.kernel(mol, mo, compact=False).reshape((nmo,) * 4)
    return SimpleNamespace(
        mc=mc,
        pt=pt,
        h1e=h1e,
        eri=eri,
        mo_energy=np.asarray(pt.mo_energy, dtype=float),
        ncore=int(mc.ncore),
        ncas=int(mc.ncas),
    )


def _expand_to_spinors(spatial):
    """Expand scalar MOs as p-alpha,p-beta in core/active/virtual order."""

    nmo = spatial.h1e.shape[0]
    nspinor = 2 * nmo
    h1e = np.kron(spatial.h1e, np.eye(2, dtype=np.complex128))
    eri = np.zeros((nspinor,) * 4, dtype=np.complex128)
    spin_blocks = (
        np.arange(0, nspinor, 2),
        np.arange(1, nspinor, 2),
    )
    for pq in spin_blocks:
        for rs in spin_blocks:
            eri[np.ix_(pq, pq, rs, rs)] = spatial.eri

    return SimpleNamespace(
        h1e=h1e,
        eri=eri,
        mo_energy=np.repeat(spatial.mo_energy, 2),
        spatial_index=np.repeat(np.arange(nmo, dtype=np.int64), 2),
        ncore=2 * spatial.ncore,
        ncas=2 * spatial.ncas,
    )


def _print_energies(label, reference_energy, spatial_pt, spinor_pt):
    print(f"\n{label}")
    print(f"E(reference)       = {reference_energy: .15f} Eh")
    print("space       spatial / Eh          spinor / Eh       difference / Eh")
    for key in SUBSPACES:
        spatial = float(spatial_pt.sub_eners[key])
        spinor = float(spinor_pt.sub_eners[key])
        print(f"{key:5s}  {spatial: .15f}  {spinor: .15f}  {spinor-spatial: .3e}")
    difference = float(spinor_pt.e_corr - spatial_pt.e_corr)
    print(
        f"TOTAL  {spatial_pt.e_corr: .15f}  {spinor_pt.e_corr: .15f}"
        f"  {difference: .3e}"
    )
    print(f"E(total, spatial) = {spatial_pt.e_tot: .15f} Eh")
    print(f"E(total, spinor)  = {spinor_pt.e_tot: .15f} Eh")


def run_scalar_nevpt2(
    *,
    atom,
    basis,
    ncas,
    nelecas,
    method,
    strong_contraction_basis=None,
    charge=0,
    spin=0,
    threads=1,
    max_memory_mb=480_000,
    stack_memory_mb=16_000,
    contraction_backend="pytblis",
    scratch_root=None,
):
    """Run one minimal closed-shell scalar/spinor SC or FIC comparison.

    ncas counts spatial orbitals. The explicit calculation contains twice
    that number of spin orbitals. For SC, strong_contraction_basis must be
    either "spatial" or "spinor". FIC has no strong-contraction grouping.
    """

    method = str(method).lower()
    if method not in ("sc", "fic"):
        raise ValueError("method must be 'sc' or 'fic'")
    if spin != 0:
        raise ValueError("this minimal scalar input is restricted to spin=0")
    if method == "sc":
        if strong_contraction_basis not in ("spatial", "spinor"):
            raise ValueError(
                "SC requires strong_contraction_basis='spatial' or 'spinor'"
            )
    elif strong_contraction_basis is not None:
        raise ValueError("FIC does not use strong_contraction_basis")

    threads = int(threads)
    lib.num_threads(threads)
    if contraction_backend == "pytblis":
        import pytblis

        pytblis.set_num_threads(threads)

    mol = gto.M(
        atom=atom,
        basis=basis,
        charge=charge,
        spin=spin,
        verbose=0,
        max_memory=max_memory_mb,
    )
    spatial = _spatial_reference(mol, int(ncas), int(nelecas), method)
    expansion = _expand_to_spinors(spatial)
    eris = init_spinor_eris(
        expansion.h1e,
        expansion.eri,
        expansion.ncore,
        expansion.ncas,
        frozen=0,
        copy=False,
        check=True,
    )

    scratch_parent = (
        scratch_root
        or os.environ.get("SLURM_TMPDIR")
        or os.environ.get("TMPDIR")
        or lib.param.TMPDIR
    )
    with TemporaryDirectory(
        prefix="scalar-spinor-nevpt2-",
        dir=str(Path(scratch_parent).resolve()),
    ) as scratch:
        solver = DMRGCI().init(
            ncas=expansion.ncas,
            nelecas=int(nelecas),
            nroots=1,
            max_bond_dimension=500,
            start_bond_dimension=250,
            tol=1.0e-12,
            schedule_thrd_max=1.0e-16,
            scratch=Path(scratch) / "dmrg",
            n_threads=threads,
            stack_memory=int(stack_memory_mb),
            cutoff=0.0,
            integral_cutoff=0.0,
            npdm_site_type=2,
            npdm_cutoff=0.0,
            random_seed=9173,
        )
        try:
            _active_energy, state = solver.kernel(
                eris.get_h1eff("AA"),
                eris.get_chem("AAAA"),
                expansion.ncas,
                int(nelecas),
                ecore=0.0,
                max_memory=max_memory_mb,
                verbose=0,
            )
            if not solver.converged:
                raise RuntimeError("explicit-spinor DMRG did not converge")

            mc_spinor = SimpleNamespace(
                _scf=SimpleNamespace(mol=mol),
                mol=mol,
                verbose=0,
                stdout=mol.stdout,
                ncore=expansion.ncore,
                ncas=expansion.ncas,
                nelecas=int(nelecas),
                frozen=0,
                fcisolver=solver,
                ci=state,
                mo_coeff=np.eye(
                    expansion.h1e.shape[0],
                    dtype=np.complex128,
                ),
                mo_energy=expansion.mo_energy,
                e_tot=float(spatial.mc.e_tot),
            )

            if method == "sc":
                spinor_pt = WickX2CSCNEVPT2(mc_spinor)
                spinor_pt.canonicalized = True
                spinor_pt.kernel(
                    eris=eris,
                    eris_basis="semicanonical",
                    denominator_mode="strict_si",
                    contraction_backend=contraction_backend,
                    strong_contraction_groups=(
                        expansion.spatial_index
                        if strong_contraction_basis == "spatial"
                        else None
                    ),
                    compact_eris=True,
                )
                label = (
                    "SC-NEVPT2: spatial-partner grouping"
                    if strong_contraction_basis == "spatial"
                    else "SC-NEVPT2: independent spinor channels"
                )
            else:
                spinor_pt = WickX2CFICNEVPT2(mc_spinor)
                spinor_pt.canonicalized = True
                spinor_pt.kernel(
                    eris=eris,
                    eris_basis="semicanonical",
                    contraction_backend=contraction_backend,
                    compact_eris=True,
                )
                label = "FIC-NEVPT2"

            _print_energies(
                label,
                float(spatial.mc.e_tot),
                spatial.pt,
                spinor_pt,
            )
        finally:
            solver.close()

