# F six-root NEVPT2 validation — 2026-10-04

## Setup and completion

F-/dyallv3z ordinary spinor X2CAMF HF followed by neutral F
CAS(7e,16 spinors), six equally weighted roots, reference M=1000,
original orbital ordering, and second-order DMRG-SCF. CD and Kramers
restriction are disabled in HF/MCSCF/PT. All comparisons restore the same
saved orbitals and root MPSs and use full-Coulomb PT integrals.

SA-DMRG-SCF converged in 10 orbital updates: SA energy
`-99.57285514115760 Eh`, gradient `3.29431e-5`, wall time 574.55 s,
peak process RSS 2.162 GiB. Reference checkpoint fingerprint:
`ebf54f2214c38450f5468d92316ca264934f3fa834501ca57d19d06f74c83d02`.
All six full-4-RDM and no-4-RDM energy calculations and response profiles
completed. No HF/MCSCF reoptimization was performed for subsequent roots.

## Six-root total energies

All energies are in Eh. The two hybrid columns below use the audited
M=2000 whole-class response, not the default response controls. They retain
the six contracted classes and replace only `i/r` by UC response; therefore
they are not alternate numerical evaluations of strict SC/FIC.

| Root | MCSCF | Strict SC | Strict FIC | SC+UC(i,r), M2000 | FIC+UC(i,r), M2000 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | -99.57346810017411 | -99.73562982209828 | -99.73678366198118 | -99.73618958916705 | -99.73678491363009 |
| 1 | -99.57346810016949 | -99.73550909655796 | -99.73669061618399 | -99.73608583592237 | -99.73669178661915 |
| 2 | -99.57346810016341 | -99.73547959518756 | -99.73666598891921 | -99.73605604313823 | -99.73666716160349 |
| 3 | -99.57346810015427 | -99.73556753595012 | -99.73673396956127 | -99.73612783321715 | -99.73673516022876 |
| 4 | -99.57162922314703 | -99.73379224497926 | -99.73486499568868 | -99.73429387125415 | -99.73486670117397 |
| 5 | -99.57162922313734 | -99.73379224497457 | -99.73486499569186 | -99.73429387125265 | -99.73486670117491 |

The six unchanged classes agree between full and no-4-RDM paths within
`6.835e-14 Eh` over all roots. The strict roots 0--3 have total-energy
spreads of `1.50227e-4 Eh` (SC) and `1.17673e-4 Eh` (FIC). These observed
splittings are retained as diagnostics, not treated as task failures.

## Response precision

Each root was tested with default and high-precision M=2000 controls.
Roots 0 and 4 additionally used 16 fixed sweeps, then tightened the linear
threshold to `1e-22`, set noise to zero, tightened the cutoff to `1e-24`,
and increased M to 1500 and 2000. RNG seed 1234 was reset for each profile.
Only response controls changed; Hamiltonian/source/reference stayed fixed.

The independent test-only SGF Hamiltonian action measures
`rho = ||Ax-b||/||b||`. At M=2000, the `i` residuals span
`6.391e-11` to `3.036e-10`, while the `r` values are:

| Root | aaav/r global relative residual |
| --- | ---: |
| 0 | 1.45992e-7 |
| 1 | 6.63812e-8 |
| 2 | 1.64275e-7 |
| 3 | 6.87055e-8 |
| 4 | 5.26209e-7 |
| 5 | 4.77057e-7 |

Thus **none of the six complete responses is certified at global `1e-8`**.
Finite excess residuals now produce warnings and retain the energies;
verification flags remain false. Nonfinite residuals still fail. The
default-control `r` residuals are approximately 0.0126--0.0164, so default
energies must not be described as globally residual-converged either.

Nevertheless, M=1500 to M=2000 changes the combined `i/r` energy by only
`1.143e-16 Eh` for root 0 and `-5.020e-16 Eh` for root 4. This is energy
stability in those measured profiles, not a residual certificate or a
universal error bound.

## Measured memory and root-0 default comparison

For the root-0 default-control comparison (separate processes):

| Path | SC E2 / Eh | FIC E2 / Eh | Peak RSS / GiB | Wall / s |
| --- | ---: | ---: | ---: | ---: |
| Strict, full 4-RDM | -0.162161721924162 | -0.163315561807068 | 135.666 | 2864.12 |
| No 4-RDM, SC/FIC+UC(i,r) | -0.162721321449537 | -0.163316709889051 | 5.674 | 1317.05 |

Peak process memory decreased by 23.91x (95.82%) for this example; the raw
complex 4-RDM alone is 64 GiB. These measurements are not memory estimates
for other active spaces or for the M=2000 audit. Default root-0 corrections
change by `-5.59599525e-4 Eh` (SC) and `-1.14808198e-6 Eh` (FIC), reflecting
both the different contraction and finite response accuracy.

## Regression and retained data

The current related regression set passed **90 tests in 122.92 s** on
gpu01 (CPU only). Its five uncaptured numerical warnings are deliberately
injected by existing validation tests. The new warning regression checks
that a finite excess global residual retains the energy without a false
precision flag, and that nonfinite residuals remain errors.

Production is the native pyblock2 whole-aaac/aaav route described in
[README.md](README.md); it does not use t-NEVPT2 classes, channel-wise
solves, alpha/beta conversion, CD or KR. Strict SC/FIC remain the default.
Historical prototype results are not evidence for the current workflow.

Raw JSON, logs, orbitals and MPS checkpoints are kept in a recoverable
archive outside the repository, rather than included in Git. Reproduction
uses the committed input scripts; resuming the archived six-root run
requires restoring its `f_sa6_ground_run` reference and `f_sa6_all_roots`
results first. The reference fingerprint above identifies that dataset.
