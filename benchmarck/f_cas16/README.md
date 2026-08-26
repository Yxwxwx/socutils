# F CAS(7e,16 spinors) Super-CI comparison

This benchmark compares complete-space iterative CI and Block2 DMRG-SCF for
the six lowest spin-orbit states of neutral fluorine. Both active solvers are
run with general-complex and explicitly Kramers-restricted orbital updates.
Super-CIPT is not part of this benchmark.

The common target is `2s2 2p5 2P-odd` in CAS(7e,16 spinors), starting from
closed-shell F- X2C orbitals in `cc-pVTZ-DK`. Six roots have equal weights.
The fourfold `J=3/2` ground manifold and twofold `J=1/2` excited manifold are
identified by degeneracy, and their mean separation is compared with
404.141(2) cm^-1.

Before any orbital optimization, each restriction has a fixed-orbital probe.
It compares sorted root energies and the basis-invariant six-state, quartet,
and doublet averaged 1-/2-RDMs. DMRG uses one state-averaged complex-SGF
MultiMPS, two-site-to-one-site sweeps, bond dimension 512, and the same six
weights as the Super-CI functional.

The raw per-root Kramers-partner RDM check uses a `1e-6` validation threshold.
This accommodates the numerical gauge of separately extracted degenerate
DMRG roots; the state-averaged RDM is neither projected nor repaired, and its
actual residual is recorded in the result.

Run from the repository root:

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv --cache-dir .cache/uv run python benchmarck/f_cas16_superci.py
```

Results and logs are written below this directory. Matching successful jobs
are restartable; pass `--force` to replace them.

The completed comparison and validation diagnostics are summarized in
[`RESULTS.md`](RESULTS.md).
