# Complex-spinor PM / UNO-style orbital preparation

`lo/uno_spinor.py` exposes `sqrtm`, `lowdin`, `pmloc`, and `sort_orbitals` with the
same signatures and return arity as the fixed block2 reference. It also exposes
`localize_blocks` for the subsequently requested fixed-CAS C/A/V preparation.
The untouched
reference is `lo/uno_b2.py`, SHA256
`613e77442c33a0feea57f8b90951356f8c0345af0701ffe306b8495b60ff4783`.
Its GPL-3.0-or-later notice and author/source attribution are retained in the
new module. Both entry points are called once after HF and CAS selection,
before constructing the initial DMRG MPS.
The reference's separate `get_uno()` route localizes core and active orbitals
with PM and virtual orbitals with SCDM after selecting a new CAS. That route
is outside this fixed-CAS `sort_orbitals()` transplant.

`localize_blocks(mol, coeff, mo_occ, mo_energy, cas_list=..., ...)` adds that
localization stage for an already selected CAS: active simple/split PM through
`sort_orbitals()`, separate core PM, and virtual SCDM. It returns the same
five-tuple in core/active/virtual order and leaves input arrays unchanged.
Virtual SCDM uses the reference's pivoted QR selection in the Löwdin AO basis;
for complex spinors its selected-column Gram matrix is `B.conj().T @ B`.
It does not construct UNO orbitals from UHF, reselect the CAS, or impose KR.

## Source behavior and deliberate differences

| Item | Fixed block2 `sort_orbitals` | Spinor implementation |
|---|---|---|
| PM selection | Only `cas_list` active columns; `do_loc=False` default | Same; no core/virtual PM or SCDM |
| Simple / split | Simple localizes all active together; explicit split localizes each occupied mask and writes it back to the original positions | Same masks, same `0.0/0.0` default, same group-local sorting |
| Default objective | Mulliken quadratic PM (`iop=0`) | Same objective with Hermitian complex populations and 2c AO atom slices |
| AO metric / electron count | Scalar AO, real rotation, spatial occupancy 0–2, `(N-nelec)//2` core | Spinor AO/overlap, complex SU(2) Jacobi, occupancy 0–1, `N-nelec` core |
| Empty split group | Tests AO row count, so an empty group may still call PM | Skips empty groups |
| Final metadata | Applies final C/A/V permutation to coefficients and occupations, but not energies | Applies the same permutation to all three |
| PM failure | Discards `ierr` | Raises before mutating caller arrays |
| Input checks | Several `assert` checks and implicit integer rounding | Explicit dimensions, metric, occupations, trace, and index checks |

`pmloc(mol=list, iop=1)` accepts an already orthonormal row partition;
`iop=1` on a molecule uses Löwdin populations, and `iop=2` uses spinor
position integrals for Boys. `sort_orbitals` always calls default `iop=0`.
For real, well-conditioned input, the Jacobi rotation agrees with the fixed
source; for complex input, each orbital pair maximizes the same objective
using a 3×3 real symmetric eigenproblem. No Kramers-pair constraint is
implemented or implied. Molecules using `so_contr` have a changed AO metric
and are explicitly rejected; the module expects the molecule's native 2c AO
representation returned by `mol.intor_symmetric("int1e_ovlp_spinor")`.

The returned `mo_occ` and `mo_energy` are diagonal expectations of the
original HF density and energy proxy. They are not new natural occupations
or canonical orbital eigenvalues. Simple PM can mix occupied and empty
active spinors, so the full original HF density is generally non-diagonal
in the returned basis. The caller must not round the returned occupations
or reuse an MPS built in the old basis. On success, writable complex128
coefficients and float64 metadata receive the source-compatible active-slot
write-back. Other input dtypes receive complex128/float64 return arrays
without partial write-back.

## Reproducible checks

Run with the project environment and its normal `PYTHONPATH`:

```bash
.venv/bin/python -m pytest tests/test_uno_spinor.py tests/test_localization.py -q -s
.venv/bin/python -m ruff check lo/uno_spinor.py lo/__init__.py tests/test_uno_spinor.py examples/16-unf3-spinor-uno-pm.py examples/17-unf3-spinor-uno-ci-benchmark.py
```

Raw output from the first command is in `lo/uno_spinor_test_output.txt`:
**22 passed in 35.81 s**. The project interpreter is
`/home/Yxwxwx/code/socutils/.venv/bin/python`; with its normal import path,
`x2camf` imports from `/home/Yxwxwx/code/x2camf/x2camf/__init__.py`.
The earlier import failure was caused by overriding `PYTHONPATH`, not by
a missing backend in this environment.

The source/API tests cover signatures, the real block2 Jacobi regression,
simple/split call widths (30 and 16+14), boundary masks, non-contiguous
and unsorted `cas_list`, odd spinor core count, transactional failure,
and synced occupation/energy metadata. Integer metadata are promoted to
float64 before split processing, avoiding truncated diagonal expectations;
real coefficient inputs retain complex results without partial write-back.
The C/A/V extension test checks separate active/core PM calls, complex virtual
SCDM, orthonormality and each block projector, metadata, and failure without
input mutation. It also compares the exact projected four-electron CAS
spectrum before and after all three block rotations. A fractional active
occupation no longer bypasses the core/virtual complement-order check.
Algebra tests verify 2c Mulliken populations, the independent 3x3 pair oracle,
random SU(2) and finite-difference stationarity checks, degenerate and purely
imaginary pair cases, phase covariance, and exact two-electron CAS Hamiltonian
similarity under the actual PM unitary. Independent density/energy-proxy
reconstruction also uses a nonidentity complex AO metric. The convergence
diagnostic reports the largest remaining pair gain, not global optimality.

Native molecular checks used `.venv/bin/python`:

| Check | Observed result | Scope |
|---|---|---|
| H₂/STO-3G general X2CAMF-HF → simple PM → `second_order()` DMRG-SCF | exact CASSCF `-1.137294866569286` Ha; DMRG-SCF `-1.137294866569289` Ha | CAS(2e,4s); both converged, final macro index 1 |
| H₂/STO-3G, four-root SA | exact CASSCF `-0.773437979550388` Ha; DMRG-SCF `-0.773437979550389` Ha | Weighted energies; weights `(0.4,0.2,0.2,0.2)` and individual root energies checked |
| H₂/STO-3G, CD with tau=1e-8 | exact CASSCF `-1.137294866569285` Ha; DMRG-SCF `-1.137294866569288` Ha | Same PM handoff, downstream CD integral route |
| H₂/6-31G, CAS(2e,4s) | exact CASSCF `-1.146246035086684` Ha; DMRG-SCF `-1.146246035086686` Ha | Nonzero initial orbital gradient, active/external rotations; converged at macro index 6 after six updates |
| UNF₃ R=1.750 X2CAMF-HF checkpoint, CAS(14e,30s), no PM | initial Mulliken objective `19.783699632` | Saved HF input to the separate DMRG-CI comparison below |
| Same checkpoint, simple PM | objective `21.974002643`; PM 1.15 s in this run | Active subspace, orthonormality, trace, occupations and energy proxy checked |
| Same checkpoint, split PM 0.05/0.95 | objective `21.095183467`; PM 1.28 s in this run | Same invariants; 16 empty and 14 occupied HF spinors localized separately |

The H₂ checks reconstruct the full transformed HF density and verify its
energy; they do not replace it with the diagonal returned occupations.
The 6-31G case tests actual orbital optimization beyond the full-active-space
STO-3G handoff. These molecular checks do not establish a UNF₃ speedup.

The initial UNF₃ active Mulliken matrices contain nonzero imaginary
off-diagonal populations (maximum `0.1487`), so this is a genuine complex
test. The saved checkpoint has 676 2c AO rows, 126 electrons, and input
orthonormality residual `1.41e-13`. Both variants preserve the active
occupation trace of 14 to numerical precision.
The optional C/A/V entry point was also run on this checkpoint with 16 threads:
112 core PM, 30 active PM, and 534 virtual SCDM completed in 8.85 s. The
result passed full spinor orthonormality, each block subspace, and occupation
and energy-proxy checks. The raw check is
`.tmp_uno_benchmark/check_blocks.out`; it does not include DMRG.
An independent real-input SCDM comparison agrees with the fixed Block2 formula
to `7.1e-16`; a complex eight-spinor projected-CAS spectrum agrees after C/A/V
localization to `2.5e-14` Ha. On the UNF₃ HF checkpoint, the final largest
two-orbital PM objective gain was `4.61e-8` for core and `1.55e-8` for active.

I read the existing no-PM input and log at
`/home/Yxwxwx/new-dmrgscf/unf3/R_1.750/large/ah200/`:
`dmrg.py` sets M=1000, 32 threads, 500000 MB, CAS(14e,30s), one root,
and `second_order`; `run.slurm` sets 200 macros and 60 cold DMRG sweeps;
the driver default orbital ordering is `original`. The existing log reaches
macro 22 and contains DMRG sweeps. A controlled fixed-orbital, three-variant
DMRG-CI comparison was submitted as Slurm job `249015` on `cpu32` using the
same HF checkpoint and M/schedule/ordering. The original group completed
on `cu08`. At the user's request, this job was cancelled while simple was
transforming integrals, to replace the sweep schedule in a standalone test.
Completed sweep records are copied verbatim to
`lo/uno_spinor_unf3_benchmark_output.txt`.

| Fixed-HF representation | Converged | Sweeps (count) | Total energy / Ha | Final energy change / Ha | Final discarded weight | DMRG sweep time / s | Total CASCI wall time / s |
|---|---|---:|---:|---:|---:|---:|---:|
| Original, no PM | yes | 23 | -28332.236658038516907 | 3.24022e-10 | 3.80616e-8 | 1371.058 | 1864.47 |

The original group's last sweep has zero noise and Davidson threshold
`1e-13`; sweep indices 0–22 mean 23 sweeps. Its total energy equals the
existing `ah200` macro-0 energy at the printed precision, independently
confirming reproduction of that starting calculation. Total CASCI wall time
includes integral transformation and solver setup; PM overhead is separate.
No entropy diagnostic is emitted by this run. The old schedule first reaches
zero noise at sweep index 22, so 23 sweeps is its minimum: this result cannot
show that PM reduces sweep count. No new UNF₃ DMRG-SCF run has completed, and
there is **no PM speedup or DMRG-SCF convergence claim** for UNF₃. The measured
1-second PM timings are preprocessing overhead only.

The sequential production input is `examples/16-unf3-spinor-uno-pm.py`.
It requires a new `--state-dir`, making scratch and checkpoints independent
of `ah200`. `--variant split` changes only the explicit PM thresholds;
the default `simple` localizes the entire 30-spinor active block. It
also accepts `--full-blocks` to use `localize_blocks()` for core PM and
virtual SCDM in addition to the selected active PM variant. The default
retains the original X2CAMF, basis, CAS, M, schedule, ordering, optimizer,
and convergence parameters. It performs no orbital re-canonicalization
and does not restart an old MPS.

`examples/17-unf3-spinor-uno-ci-benchmark.py` is the fixed-orbital comparison
driver used by job `249015`. It reads only the existing HF checkpoint and
keeps the three DMRG checkpoint directories separate.

## Revised standalone comparison requested by the user

Slurm job `249025` uses `.tmp_uno_benchmark/block2_ci.py` and
`.tmp_uno_benchmark/run.slurm`: 16 CPU cores, 500G, `cpu32`, project `.venv`.
It calls `pyblock2.driver.core.DMRGDriver` directly; the later optional C/A/V
code was not used in this active-only comparison. Each original/simple/split
representation gets its own AO-to-MO transformation and an independent cold
MPS; there is no orbital optimization or Fiedler reordering.

The arrays are passed exactly as requested, with `n_sweeps=60`, `tol=1e-8`:

```python
bond_dims = [500] * 4 + [750] * 4 + [1000] * 4
noises = [1e-4] * 4 + [1e-5] * 8 + [1e-6] * 4 + [0]
thrds = [1e-5] * 4 + [1e-6] * 8 + [5e-7]
```

Block2 handles entries beyond these array lengths natively; the script does
not pad or replace them. Zero noise starts at index 16. The native H₂ check
in `.tmp_uno_benchmark/check_block2_ci.py` verifies the received arrays and
compares all three representations with exact CI: original/split take 17
sweeps, simple takes 18, and all total-energy errors are below `1e-8` Ha.
Raw output is `.tmp_uno_benchmark/block2_smoke.out`. Its later local Davidson
thresholds are the values printed by Block2 (for example `1e-9` at index 16),
not an assumed repetition of the last explicitly supplied `5e-7`.

UNF₃ job `249025` completed. Its full log is
`.tmp_uno_benchmark/slurm-249025.out`; per-variant sweep energies and discarded
weights are in `.tmp_uno_benchmark/run_249025/` and copied to the tracked
`lo/uno_spinor_unf3_benchmark_output.txt`.

| Variant | Converged | Sweeps | Total energy / Ha | Final ΔE / Ha | Final discarded weight | AO→MO / s | DMRG / s | Variant wall / s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Original | yes | 19 | -28332.236658030673425 | 2.103214e-9 | 3.736249e-8 | 566.05 | 1262.037 | 1837.28 |
| Simple PM | yes (sweep ΔE only) | 21 | -28332.236652857700392 | 7.842196e-9 | 1.007171e-6 | 500.08 | 1540.954 | 2049.73 |
| Split PM | yes (sweep ΔE only) | 19 | -28332.236658066791279 | 1.261618e-9 | 5.161681e-8 | 548.61 | 1275.500 | 1832.80 |

The new no-PM energy differs from the old-schedule no-PM result by about
`7.84e-9` Ha. This validates the revised baseline at the requested `1e-8`
energy tolerance. Simple PM takes two more sweeps, 278.917 s more DMRG time,
and 212.45 s more per-variant wall time. Its final total energy is
`5.172973e-6` Ha higher than the new no-PM result, despite its consecutive
sweep energy change meeting `1e-8`; the final discarded weight is about 27
times larger. The comparison therefore does **not** show an equal-accuracy
speedup for simple PM at M=1000. The larger finite-M truncation is a plausible
explanation for the energy gap, not an independently proven attribution.
Simple PM is monotone across all 21 sweeps; no-PM has 5 energy rises in 19
sweeps, and split PM has 7 in 19. Smoothness therefore did not predict the
best final energy or sweep count in this run.

Split PM finishes in the same 19 sweeps as no-PM, with energy lower by
`3.611785e-8` Ha and discarded weight 1.38 times larger. Its DMRG time is
13.46 s longer; the total per-variant wall time is 4.48 s shorter because
its AO→MO stage happened to be faster. A single timing run does not establish
a speed advantage. The `converged=True` flags only test adjacent zero-noise
sweep changes against `1e-8`; they do not prove equal absolute finite-M
energies across representations. Simple and split PM preprocessing took 0.532
and 0.418 s, respectively, outside the per-variant wall times. No entropy
diagnostic was emitted by this standalone driver.
