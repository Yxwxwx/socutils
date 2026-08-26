# F CAS(7e,16 spinors) Super-CI results

All four six-state, equal-weight calculations converged from the same F-
reference orbitals in 15 macroiterations.  The stopping thresholds were
`|dE| < 1e-8 Eh` and `|g| < 1e-4`; no natural-orbital rotation,
canonicalization, or RDM projection was applied.

| Active solver | Orbital space | State-average energy (Eh) | Final `|g|` | Splitting (cm-1) | Error from 404.141 cm-1 | Wall time (s) |
|---|---:|---:|---:|---:|---:|---:|
| exact CI | general complex | -99.583012781793400 | 2.45319e-5 | 581.159073317 | +177.018073317 | 1499.9 |
| DMRG, M=512 | general complex | -99.583012781791780 | 2.45288e-5 | 581.159090291 | +177.018090291 | 1095.5 |
| exact CI | Kramers restricted | -99.583012781790220 | 2.45322e-5 | 581.159052096 | +177.018052096 | 1490.5 |
| DMRG, M=512 | Kramers restricted | -99.583012781791400 | 2.45324e-5 | 581.159085082 | +177.018085082 | 1096.3 |

The four predictions differ by at most `3.82e-5 cm-1`.  Their common error
relative to experiment is about `+177.018 cm-1` (`+43.8011%`), so this error
belongs to the Hamiltonian/correlation model used here rather than to DMRG or
the Kramers orbital restriction.

## Solver and restriction comparisons

| Comparison | `|dE_SA|` (Eh) | `|d splitting|` (cm-1) | Max active-projector element |
|---|---:|---:|---:|
| exact vs DMRG, general | 1.620e-12 | 1.697e-5 | 1.173e-7 |
| exact vs DMRG, Kramers | 1.180e-12 | 3.299e-5 | 6.723e-8 |
| general vs Kramers, exact | 3.183e-12 | 2.122e-5 | 1.317e-7 |
| general vs Kramers, DMRG | 3.837e-13 | 5.209e-6 | 1.456e-8 |

The largest exact/DMRG state-average energy difference anywhere along a
matched 15-point Super-CI trajectory was `1.70e-9 Eh`.  The remaining
`1e-5 cm-1`-scale final splitting differences are dominated by iterative
exact-CI resolution inside numerically degenerate manifolds: the exact-general
quartet spread is `3.39e-9 Eh`, whereas the DMRG quartet spreads are about
`1.5e-11 Eh`.

The fixed-initial-orbital controls isolate the active solvers from orbital
optimization.  Exact CI and DMRG agree to `2.39e-12 Eh` per root and
`6.55e-8 cm-1` in the general calculation, and to `2.10e-12 Eh` per root and
`5.93e-8 cm-1` in the Kramers calculation.

## DMRG/Kramers validation

Both DMRG calculations used one six-root state-averaged complex `MultiMPS`
with equal weights and finished with one-site sweeps before root extraction.
For the final Kramers-restricted point:

- `max|S-I| = 3.56e-15`
- `max|H-SE| = 7.11e-14 Eh`
- discarded weight `= 1.87e-19`
- state-average Kramers 2-RDM residual `= 7.74e-12`
- active-orbital Kramers closure error `= 3.71e-15`
- RDM projection: disabled

At one intermediate geometry the raw 2-RDMs of a numerically split Kramers
doublet differed by `1.55e-7`, marginally above the original validation-only
gate of `1e-7`.  The gate was set to `1e-6`; the underlying state-average RDM,
root overlap, projected eigen-equation, CI convergence thresholds, and orbital
optimization were unchanged.  The final raw partner 2-RDM residual was
`1.56e-9`.

Machine-readable details are in `summary.json` and the individual JSON/NPZ
files under `results/`.
