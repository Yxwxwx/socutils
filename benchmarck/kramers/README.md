# Kramers-restricted six-state halogen benchmark

This benchmark repeats the F, Cl, Br, I, and At six-state Super-CI comparison
with Kramers-restricted orbitals. Super-CIPT is deliberately excluded.

## Compared methods

1. exact full-spinor FCI with Kramers-restricted SCF orbital optimization;
2. Block2 SGFCPX DMRG with Kramers-restricted SCF orbital optimization and the
   explicit `KramersResultAdapter`.

The exact calculation does not reduce or reconstruct its CI space: it is the
full determinant-space reference. Its six-state equal-weight density and its
active orbital subspace are independently checked for time-reversal symmetry.
The DMRG calculation retains all eight active spinors and optimizes the six
roots together in one state-averaged MultiMPS, using the same equal weights as
the orbital functional. The calculation finishes with one-site sweeps, and
the split roots must pass overlap and projected eigen-equation checks. The
adapter then validates complete Kramers manifolds from the raw energies and
1-/2-RDMs. Raw diagnostics are reported; RDM projection is disabled.

## Common protocol

- The basis, Hamiltonian, initial charge, target charge, active space, root
  weights, Cholesky factors, and Super-CI tolerances match the unrestricted
  benchmark: `cc-pVTZ-DK`, one-electron spin-orbit X2C with Coulomb-only
  two-electron terms, closed-shell `X-` initial orbitals, neutral `X`,
  CAS(7 electrons, 8 spinors), and six equal weights.
- The standard PySCF one-electron X2C helper is attached to
  `socutils.scf.spinor_hf.KRHF`. This preserves the previous Hamiltonian while
  using the quaternion eigensolver for the AO SCF and MO-subspace
  diagonalizations.
- Active natural-orbital rotations and iterative core/virtual canonicalization
  are disabled. The latter is a physically redundant gauge rotation which is
  numerically non-unique inside the exactly degenerate atomic manifolds and
  can amplify roundoff differences between the exact and DMRG densities.
  Every Super-CI orbital generator is explicitly projected onto the
  time-reversal-invariant subspace before exponentiation, for both paths.
- Orbital convergence requires `|dE| < 1e-8 Eh` and `|g| < 1e-4`, with maximum
  step 0.2 and at most 50 macroiteration evaluations.
- Super-CI uses Davidson tolerance `1e-7`, maximum subspace 30, and strict
  failure if its full generalized residual does not converge. Its overlap
  metric uses the full active 1-RDM, is covariant to active-orbital rotations,
  and canonically removes only roundoff-level zero-metric directions.
- DMRG uses bond dimension 32, eight zero-noise sweeps, active-energy tolerance
  `1e-12`, local squared residual threshold `1e-20`, Davidson maximum 1000,
  seed 2468, one thread, and NPDM site type/cutoff 2/`1e-24`.
- Kramers energy, raw-RDM, and active-orbital validation tolerances are `1e-8`.

## Run and resume

From the repository root:

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv --cache-dir .cache/uv run python benchmarck/kramers_superci.py
```

The driver runs each element/method in a child process and skips a successful
result only when its complete protocol matches. Run a subset or force a rerun
with:

```bash
uv --cache-dir .cache/uv run python benchmarck/kramers_superci.py \
  --elements F Cl --methods exact-superci dmrg-superci

uv --cache-dir .cache/uv run python benchmarck/kramers_superci.py \
  --elements F --methods dmrg-superci --force
```

Runtime logs are under `logs/`; durable JSON is under `results/`. Regenerate
`RESULTS.md`, `summary.{json,csv}`, `states.csv`, and `convergence.csv` with:

```bash
uv --cache-dir .cache/uv run python benchmarck/summarize_kramers_superci.py
```
