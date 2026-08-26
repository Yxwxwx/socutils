# Six-state halogen benchmark

This benchmark extends the historical F atom calculation to F, Cl, Br, I,
and At.  It deliberately uses the general complex-spinor path rather than the
Kramers-restricted adapter.

## Common protocol

- Closed-shell `X-` X2C SCF orbitals are used as the common initial orbitals
  for neutral `X`, where `X = F, Cl, Br, I, At`.
- `cc-pVTZ-DK` is used for every atom.  The original F example used
  `cc-pVTZ`, but that basis is unavailable for I and At in the locked PySCF
  installation; the Douglas--Kroll member is the closest uniform built-in
  triple-zeta family available for all five elements.
- AO Coulomb integrals are represented by pivoted Cholesky vectors with
  `tau = 1e-10`.  The same cached vectors and the same X2C mean-field object
  are used by all three methods.
- The neutral calculation is CAS(7 electrons, 8 spinors), with six equally
  weighted roots, matching the F reference setup.
- Active natural-orbital rotations are disabled.  Core/virtual
  canonicalization is enabled.
- Orbital convergence requires both `|dE| < 1e-8 Eh` and `|g| < 1e-4`.
  The maximum orbital step is 0.2 and at most 50 macroiterations are allowed.
- The full Super-CI Davidson residual tolerance is `1e-7`, three orders of
  magnitude below the orbital-gradient gate.  In the exactly degenerate F
  manifold the Block2 and exact state-averaged RDMs agree to `1.3e-15`, but
  arbitrary complex rotations inside degenerate core/virtual subspaces give
  the orbital Davidson a numerical residual floor up to `2.3e-8`.
- DMRG uses SGFCPX, bond dimension 32, eight zero-noise sweeps, energy
  tolerance `1e-12`, local squared residual threshold `1e-20`, Davidson
  maximum 1000, seed 2468, and NPDM site type 2.
- The large frozen-core constant is added to the returned DMRG energies but
  excluded from the MPO.  This leaves all states and total energies unchanged
  while making the sweep convergence test operate on the active energy rather
  than lose precision at the `-2.3e4 Eh` At total-energy scale.

The compared methods are:

1. exact relativistic CASSCF + full Super-CI;
2. Block2 DMRG-SCF + full Super-CI;
3. Block2 DMRG-SCF + Super-CIPT.

`macroiterations` in the summaries is the number of macroiteration
energy/gradient evaluations, including the initial evaluation and the final
converged evaluation.  `orbital_updates` is reported separately.

## Run and resume

From the repository root:

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv --cache-dir .cache/uv run python benchmarck/halogen_six_state.py
```

The driver runs one element/method per child process, skips matching successful
JSON results, and writes complete logs under `benchmarck/logs/`.  To rerun a
subset or force replacement of generated results:

```bash
uv --cache-dir .cache/uv run python benchmarck/halogen_six_state.py \
  --elements Br I At --methods dmrg-superci dmrg-supercipt

uv --cache-dir .cache/uv run python benchmarck/halogen_six_state.py \
  --elements F --methods casscf-superci --force
```

SCF orbitals and Cholesky factors are restart caches under
`benchmarck/.scratch/` and are not versioned.  Durable per-calculation JSON is
under `benchmarck/results/`.  Regenerate the CSV and Markdown tables with:

```bash
uv --cache-dir .cache/uv run python benchmarck/summarize_halogen.py
```

The completed overview is written to `RESULTS.md`; machine-readable outputs
are `summary.json`, `summary.csv`, `states.csv`, and `convergence.csv`.
