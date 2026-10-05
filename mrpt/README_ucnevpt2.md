# X2C-UC-NEVPT2: native MPS-PT and independent validation

`X2CUCNEVPT2` solves a single, full-chain complex-SGF response using native
`MRCIMPSInfo(ci_order=2)` and `Linear.solve`. This follows the space/solver
organization of non-big-site `block2main nevpt2sd`, not eight independent
solves as the default. The RHS is contracted on demand from its MPO and
the embedded `CASCIMPSInfo` reference; no source MPS is generated before
the solve. `evaluate_uc(..., class_resolved=True)` retains eight
native `NEVPTMPSInfo` solves for cross-validation. Existing SC, FIC, and
SC/FIC+UC(i,r) interfaces remain unchanged.

`pt.response_mode = "external_tuples"` selects a second, active-only UC
organization (`UCNEVPT2` is an alias of `X2CUCNEVPT2`). The default is still
`"full_chain"`. Independent small-model validation and the completed real-F
benchmark are documented below, separately from the earlier full-chain runs.

## External-occupation-blocked option

After core/virtual semicanonicalization, each unique tuple
`mu=(holes, particles)` has `N_mu=N_A+len(holes)-len(particles)` and
`Delta_mu=sum(eps_particles)-sum(eps_holes)`. All eight classes are included,
including `(2,1)`, `(1,2)`, `(2,2)`; `(0,0)` is never enumerated. The equations are

```
A_mu = H_A^(N_mu) - E_A + Delta_mu
b_mu = U_mu† H U_0 Psi_A
A_mu x_mu = b_mu
E_projected,mu = -Re <b_mu|x_mu>
F_mu = Re <x_mu|A_mu|x_mu> - 2 Re <b_mu|x_mu>
```

One fixed active Hamiltonian supplies every tuple; a hole does not change
its active mean field. Sources use the tested core-normal-ordered blocks
with explicit full-chain fermion signs. All terms of a tuple are combined
coherently, then its nonzero source is normalized and its squared physical
amplitude restored. No weak-source screening, IC basis, active-reference
orthogonalization, level shift, 3/4-RDM, or full-chain response is used.
Block2's ordinary `MPSInfo` includes the whole active particle sector;
`Linear.solve` and FastBipartite still supply the numerical machinery.
When no `eris` is supplied, this mode reuses the existing exact
`wick_eris_from_mc` AO2MO blocks instead of constructing the full
`n_mo**4` MO tensor. This is full-Coulomb block storage, not CD or screening.
Explicit dense and compact inputs are both accepted; the full-chain default
still requires complete MO integrals.

In the semicanonical basis, these are orthogonal invariant Dyall blocks,
so their exact solutions sum to the same **pure external** UC resolvent.
This is not a proof that an unconstrained finite-M full-chain result must
match, nor a guarantee of restored multiplet degeneracy or faster execution.
The channel decomposition has direct precedent for i/r in
[Sharma et al. (2017)](https://arxiv.org/pdf/1609.03496);
the complete eight-class spinor generalization is verified here independently.

Nonzero source norms are always measured without SVD truncation for
normalization. `diagnostic=True` additionally measures response vector
residuals. The pyblock2 algebra MPO×MPS contractions are QR'd
as they are formed to avoid materializing both enlarged product bonds.
The original algebra product is an independent regression reference for
this storage-only reorganization. Global/class residuals sum actual squared
norms; the worst tuple is reported as well. Unmeasured residuals remain `None`.

For `ijrs`, `b=c*Psi_A` admits the candidate `x=c*Psi_A/Delta`. It is accepted
only when its **measured original-equation** relative residual
`|| (H_A-E_A)Psi_A ||/abs(Delta)` meets the requested tolerance and its MPS
fits the requested bond cap. Otherwise the unrestricted native solve is
used. Thus a finite-accuracy reference is not silently treated as exact.
The single reference-defect check remains enabled for eligible `ijrs`
blocks even with `diagnostic=False`; it is not a global response certificate.

```python
pt = UCNEVPT2(mc)
pt.response_mode = "external_tuples"
pt.mps_response_options = dict(max_bond_dimension=256, n_sweeps=8,
                               linear_threshold=1e-24, diagnostic=True)
pt.kernel(root=0)
```

Run the retained F reference without repeating HF/MCSCF:

```bash
ulimit -s unlimited
UC_REFERENCE=/path/to/f_sa6_ground_run UC_PREFLIGHT=0 UC_ROOT=0 \
UC_RESPONSE_MODE=external_tuples UC_M=256 UC_SWEEPS=8 UC_DIAGNOSTIC=1 \
PYTHONPATH=tests python -m nevpt2_mps_response.f_uc
```

### Completed F six-root benchmark (2026-10-05)

All six external-tuple runs completed on gpu01's CPUs (no Slurm), using
the same archived SA6 second-order full-ERI/no-CD/no-KR reference:
F/dyallv3z, 7 active electrons in 16 spinors, 2 correlated core spinors
and 92 virtual spinors. Its checkpoint fingerprint is
`ebf54f2214c38450f5468d92316ca264934f3fa834501ca57d19d06f74c83d02`.
All eight classes were evaluated for every root: `ijrs=4186`, `rsi=8372`,
`ijr=92`, `rs=4186`, `ij=1`, `ir=184`, `r=92`, `i=2`, totaling 17,115 tuples.
No roots, tuple energies, or orbital energies were averaged or reordered.

The initial all-native trial measured `rho=1.40117e-11` for its first ijrs
tuple. That trial was stopped with its log preserved after validating the
above ijrs candidate; the optimized complete run is
`tests/nevpt2_mps_response/uc_f_run/root0_external_M256_S8_analytic.out`.

Root 0 used 16 PT threads; roots 1--5 used three each, with independent
drivers and `mktemp` scratch directories under `/nvme`. All restored the
same checkpoint without reoptimizing or writing its persistent MPS image.
Their logs are `uc_f_run/root{1..5}_external_M256_S8_T3.out`. Every process
exited with status 0. Effective controls were M1=256, 8 sweeps, `tol=0`,
local residual-square threshold 1e-24, zero noise, cutoff 1e-24,
`linear_max_iter=4000`, and explicit vector audits with target rho=1e-8.

| Root | E2 / Eh | Total energy / Eh | Global rho | Worst tuple rho | Process wall time | Peak RSS / GiB |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | -0.163323512690385 | -99.73679161286449 | 3.97134e-9 | 6.39824e-7 | 12:56:40 | 5.909 |
| 1 | -0.163230281826984 | -99.73669838199648 | 9.79669e-10 | 2.24284e-7 | 16:20:01 | 5.728 |
| 2 | -0.163205669346891 | -99.73667376951030 | 1.07748e-9 | 2.00905e-7 | 16:44:28 | 5.719 |
| 3 | -0.163273667439474 | -99.73674176759374 | 2.28417e-9 | 3.33332e-7 | 16:37:23 | 5.805 |
| 4 | -0.163244326575076 | -99.73487354972211 | 4.07289e-9 | 5.43266e-7 | 16:07:46 | 5.695 |
| 5 | -0.163244326583290 | -99.73487354972063 | 1.04913e-8 | 1.68021e-6 | 15:45:10 | 5.696 |

All six results retain `converged=False`: passing a weighted global rho
does not clear failing individual tuples. Root 5 also fails the weighted
global target. These are finite-residual warnings, not discarded energies.
Class-level and individual-tuple residuals remain in the original JSON log
records; the following energies are their Hylleraas functionals:

| Class | Root 0 / Eh | Root 1 / Eh | Root 2 / Eh | Root 3 / Eh | Root 4 / Eh | Root 5 / Eh |
|---|---:|---:|---:|---:|---:|---:|
| ijrs | -0.028440770575 | -0.028440770613 | -0.028440770622 | -0.028440770595 | -0.028440745530 | -0.028440745530 |
| rsi | -0.009458413455 | -0.009458081119 | -0.009457993389 | -0.009458235707 | -0.009455288166 | -0.009455288165 |
| ijr | -0.005952278667 | -0.005952271444 | -0.005952269537 | -0.005952274806 | -0.005953294195 | -0.005953294195 |
| rs | -0.076592841349 | -0.076572951124 | -0.076567704390 | -0.076582188257 | -0.076593099231 | -0.076593099231 |
| ij | -0.000652041171 | -0.000652041171 | -0.000652041171 | -0.000652041171 | -0.000652424931 | -0.000652424931 |
| ir | -0.006402435128 | -0.006401765562 | -0.006401588813 | -0.006402077010 | -0.006400921975 | -0.006400921975 |
| r | -0.034810734551 | -0.034738395488 | -0.034719294136 | -0.034772078082 | -0.034735151881 | -0.034735151890 |
| i | -0.001013997794 | -0.001014005305 | -0.001014007288 | -0.001014001811 | -0.001013400667 | -0.001013400667 |

| Method | Roots 0--3 total-energy spread / Eh | Roots 4--5 spread / Eh |
|---|---:|---:|
| Reference | 1.98455e-11 | 9.69180e-12 |
| Full SC | 1.5022691072e-4 | 4.68958e-12 |
| Full FIC | 1.1767306196e-4 | 3.18323e-12 |
| Blocked full UC | 1.1784335419e-4 | 1.47793e-12 |

The quartet splitting is **not repaired**, although the directly paired
doublet stays degenerate. Thus these data do not support attributing the
quartet splitting solely to internal contraction. Its dominant class
ranges are `r=9.14404e-5` and `rs=2.51370e-5` Eh. The r class was already
uncontracted in the earlier hybrid, and the new i/r values agree with the
archived whole-class M2000 solves for all six roots: maximum differences
are 1.37e-17 Eh (i) and 2.01e-14 Eh (r). This localizes the dominant
splitting to the shared i/r reference/Dyall problem rather than its new
tuple decomposition; it does not uniquely identify an error in that
state-specific zero-order definition. The existing reference/covariance
audit below records T mixing within the quartet and the separate paired
root Dyall checks. No density averaging or symmetry-restoring energy
cleanup was substituted for those diagnostics.

The blocked totals are 7.77--8.55 micro-Eh below full FIC. The old hybrid
i/r M1500-to-M2000 differences are below 5.1e-16 Eh for representative
roots 0/4; that energy stability is not a residual certificate. M1=256
already has sufficient formal capacity for any 16-spinor active response
(maximum ranks for N=5,6,7,8,9 are 74,130,186,256,186). This does not prove
solver convergence. A complete blocked M/sweep/reference convergence scan
was not performed; no 1e-8 energy-stability claim is inferred from capacity.

The runner records both reference/PT thread counts. Root 0 loaded the earlier
code revision; the intervening edits only reuse a full-chain audit vector
and enable the ijrs reference check with diagnostics OFF. These do not
change the diagnostics-ON external-tuple algorithm used in these six runs.
Different thread counts are recorded, so these timings are not a
matched-thread speedup comparison. Peak RSS is substantially smaller than
the older combined full-chain M128/M256 audit process (51.072 GiB), but
those runs have different residuals and controls. No matched-certified-
accuracy timing or memory ratio is claimed.
The later compact-entry-point fix does not change these already-loaded
processes: their runner explicitly supplies the prepared dense integrals.
The six-root SC/FIC and hybrid baselines in
[the existing report](../tests/nevpt2_mps_response/report.md) were rechecked
against their raw archived root JSONs. All share this checkpoint fingerprint;
their hybrid residual flags remain false. They need not be recomputed for
this completed full-UC comparison.

The existing six-root summarizer can read those archived baselines and the
new external-tuple logs without modifying either input directory:

Raw logs and generated results are local artifacts, intentionally excluded
from Git. If they have been moved outside the repository during cleanup,
pass the archived `uc_f_run` directory to `--uc-logs`; the same command
regenerates the final tables with correct archived log paths. The benchmark
tables above and the validation source code remain in the repository.

```bash
PYTHONPATH=tests python -m nevpt2_mps_response.six_roots --summary-only \
  --reference /path/to/f_sa6_ground_run --baseline /path/to/f_sa6_all_roots \
  --uc-logs tests/nevpt2_mps_response/uc_f_run \
  --output tests/nevpt2_mps_response/uc_f_run/comparison
```

Only newline-complete `F_UC_RESULT` records enter the external-UC table;
pilots, partial class energies and partially written JSON are not totals.
Reference fingerprints, root IDs, class sums, energy constants and controls
are checked. Grouping uses the actual controls recorded by the eight
classes, so an omitted default `tol=0` and explicit `tol=0` group together;
requested options are also retained. Different effective controls get
separate tables; duplicate root/control
points are rejected. Raw root order is retained, with spreads over roots
0--3 and 4--5 rather than energy-sorted or averaged states. Per-class/worst-
tuple diagnostics, input/code hashes and process-level resources remain
in `summary.json`; complete tuple details stay in the original logs.
Missing energies remain pending, and finite excess residuals remain warnings.
The new summarizer check and the existing scratch/baseline-preservation
check passed **2 tests** (15.53 s); the actual archived-data invocation
correctly reports six SC/FIC/hybrid baselines and no complete external-UC
set at that earlier time. After completion, the default/explicit-option
grouping and preservation tests passed again (**2 passed**, 13.92 s).
The actual archived-data invocation now reports one complete six-root
external-UC energy set, with all residual certifications false. This
reporting check is not a new physics convergence result.

Initial verification: **37 passed** (155.81 s) across the blocked, full-chain,
hybrid and FIC tests in `uc_f_run/external_regression.out`; an additional
finite-reference contamination test passed (22.26 s). The latter forces
`ijrs` back to native Linear and matches the independent pure-external exact
resolvent, rather than accepting a scalar-denominator approximation.
Fiedler ordering, the new tests and unchanged SC regression subsequently
passed **53 tests** (62.20 s; one intentional invalid-RDM warning) in
`uc_f_run/external_order_sc_regression_v2.out`. Complex active rotation and
core/virtual rotation followed by semicanonicalization passed separately
in `uc_f_run/external_gauge_regression.out` (21.55 s), with full-capacity
energies within 1e-10 Eh and finite-M errors reported, not averaged away.
Empty/full active particle-sector boundaries also passed (2 tests,
22.28 s), including all legal nonzero classes and empty-sector reporting.

### Correctness and avoidable-overhead audit (2026-10-04)

This audit prioritizes correct equations and implementation over forcing
real-F residual convergence. Failure to certify remains a warning with
`converged=False`; it does not change thresholds, sources or energies.

The installed `block2main` was checked directly: space descriptors,
MovingEnvironment setup, signs, native Automatic Linear, reference bond
allowance and truncation machinery agree with the intended workflow.
The full-chain default remains unconstrained; it is not advertised as
guaranteeing a pure external response. The blocked alternative instead
excludes `(0,0)` by its exact external-occupation definition.

Independent tests now read each actual blocked response MPS into a tiny
determinant space and apply independently constructed H/Dyall matrices.
Both original and Fiedler ordering pass: each nonzero tuple residual is
below 1e-8 and agrees with the tensor-network audit within 1e-11. This
test-only CI readout is not part of the production algorithm.

Repeated-work fixes were made without replacing any solver:

- Full-chain auditing constructs `Lx-RPsi` once and reuses it for all eight
  sector checks instead of rebuilding the same direct-sum MPS eight times.
- The residual-certified `ijrs` shortcut no longer depends on enabling all
  response audits. One reference check can avoid redundant native sweeps;
  insufficient reference accuracy or bond capacity still selects Linear.
- Active-only automatic integral preparation now uses the existing exact
  block AO2MO helper. A reproduced compact-container `get_chem` interface
  error was fixed by using the shared `get_phys` interface and the existing
  chemists/physicists index transpose. No formula or truncation changed.

On gpu01 CPU, the pre-edit UC/blocked/hybrid/FIC suite passed **42 tests**
(177.19 s). After the edits, the two ordering/independent-residual tests
and native-default/strict-CAS cross-check passed **3 tests** (37.24 s),
and the complete blocked suite passed **12 tests** (51.15 s), including
diagnostics-off weak sources, exact zeros and the finite-reference fallback.

After the compact-entry-point fix, the combined blocked/full-chain/hybrid/
FIC/bounded-memory suite passed **62 tests** (216.43 s) on gpu01 CPU with
one thread. This covers explicit dense inputs, cached compact inputs,
automatic exact-block preparation, compact Fiedler ordering, and actual
AO2MO blocks versus dense integrals. Peak process RSS was 411504 KiB;
the log is `uc_f_run/external_compact_final_regression.out`. This is a
regression measurement, not the real-F resource benchmark.

An eight-selected-tuple F pilot with all response audits disabled took
18.95 s for the PT call. Explicit wall timers measured 13.76 s in seven
native Linear calls, 3.40 s in eight exact products (reference/source),
0.10 s building the shared active MPO and 0.043 s building seven source
MPOs. Its `ijrs` candidate used zero sweeps; global residual is correctly
`None` and the overall result remains uncertified. A separately profiled
audited pilot took 52.69 s and passed its selected-tuple checks (worst
rho=2.16e-10). These pilots demonstrate significant audit cost, not a
complete-F or matched-certified-accuracy speedup. Logs are
`uc_f_run/correctness_audit_pilot_no_audit.out` and
`uc_f_run/correctness_audit_pilot_profile.out`.
The selected-tuple energies in these two runs differ by at most 8.33e-15 Eh;
that agreement does not certify the unmeasured response residuals.

### Same-reference four-way and degenerate-external checks

The blocked test now explicitly runs the eight `NEVPTMPSInfo` classes and
the strict-CAS full-chain solver on the **same retained finite MPS**, RDMs,
integrals and Dyall definition as the active-only result. The independent
determinant solve also uses that finite reference, not the fixture's exact
eigenvector. All eight nonzero classes pass; measured class residuals are
below 1e-8 and the largest class-energy discrepancy against the independent
oracle is 2.09e-17 Eh.

| Representation | Total E2 / Eh |
|---|---:|
| Active-only external tuples | -0.016911416102501570 |
| Eight NEVPTMPSInfo classes | -0.016911416102501575 |
| Strict-CAS full chain | -0.016911416102501575 |
| Independent exact external solve | -0.016911416102501596 |

An additional public-API gauge test uses exactly degenerate inactive and
virtual blocks (`eps_I=(-9,-9)`, `eps_V=(8,8)`). Consistent complex active
rotations and legal core/virtual rotations leave the full-capacity blocked
energy spread at 2.09e-17 Eh; its largest global residual is 4.49e-14.
Low-M results remain uncertified with their nonzero representation errors;
no energies or orbital energies are averaged.

The ordering tests and all four public gauge configurations passed
**6 tests**, pytest 50.08 s, in
`uc_f_run/external_threeway_degenerate_regression.out` on gpu01 CPU.
These are small-model checks, not a completed F multiplet benchmark.

An additional finite-reference test uses deliberately nonphysical external
energies to make some blocks invertible but nonpositive. All 15 tuples then
use native Linear, including `ijrs`; all eight class energies match the
independent exact resolvent within 1e-10 Eh and worst tuple residuals pass
1e-8. Positive second-order contributions are retained, not clipped or
changed by absolute denominators. In this case the functional is stationary,
not a variational minimum. The positive/nonpositive pair passed **2 tests**
in 29.91 s in `uc_f_run/external_nonpositive_block_regression.out`.

### F root-0 short-chain pilot (not the complete energy)

The runner's explicit `UC_TUPLE_PILOT=1` selects **one tuple per class**
only for a solver diagnostic. Both runs kept M=256, local threshold 1e-24,
16 PT threads, and the same archived reference; no production screening is
introduced. In the following table each residual belongs to that selected
tuple, not to its entire class:

| Class | rho, 2 sweeps | rho, 8 sweeps |
|---|---:|---:|
| ijrs (residual-certified analytic candidate) | 2.66e-11 | 2.66e-11 |
| rsi | 4.57e-11 | 4.34e-11 |
| ijr | 8.88e-12 | 9.03e-12 |
| rs | 3.01e-4 | 2.05e-11 |
| ij | 1.67e-11 | 1.76e-11 |
| ir | 7.86e-11 | 8.57e-11 |
| r | 2.17e-10 | 2.19e-10 |
| i | 5.30e-11 | 5.26e-11 |

The 8-sweep pilot passes all eight selected-tuple residual checks:
combined rho=3.94431e-11, worst=2.18749e-10. The `rs` energy changes by only
3.86e-13 Eh between these runs, despite its 2-sweep residual failing badly.
This is why the complete run retains 8 sweeps and true vector audits.
Pilot point times are 41.23/52.43 s (2/8 sweeps); entire process times
including restore, RDMs and complete MO AO2MO are 63.32/74.88 s; respective
process peak RSS is 5.663/5.594 GiB. These are **partial-run resources**, not
the complete 17,115-tuple cost and not a matched-accuracy speedup benchmark.
Logs: `uc_f_run/root0_external_pilot_M256_S2.out` and
`uc_f_run/root0_external_pilot_M256_S8.out`.

This is an experimental implementation. Independent exact-oracle and opt-in
CAS-exclusion tests pass. The real-F root-0 M1=128/256 benchmark is complete,
but neither finite-M response is converged. Real-F 1e-8 convergence is not a
delivery requirement under the updated scope; warnings and the measured
residuals remain. Historical SC/FIC and hybrid F energies are not full-UC results.

Kramers restriction belongs to the MCSCF reference calculation, not to this
NEVPT2 implementation. PT treats individual complex spinors without a KR
switch, enforced partners, alpha/beta reduction, or averaging correlation
energies to restore degeneracies. All tests reported here disable KR and CD
in the reference as well.

## Relationship to block2main

This is an API extraction and general-complex-spinor extension of Block2's
Sharma--Chan MPS-PT workflow, not a new DMRG linear solver. The installed
`block2main` (Block2 0.5.4rc16, non-big-site `nevpt2sd` branch) is the primary
reference when resolving implementation differences.

- The right `MovingEnvironment` receives the source MPO and the embedded
  reference MPS directly. There is no source-MPS fitting/compression layer
  followed by an identity MPO. A scalar coefficient scale avoids uniformly
  tiny sources without assuming the RHS norm is one. Energies restore the
  square of this physical scale. The strict validator separately measures
  and normalizes the complete source norm.
- Default MPO construction is `DMRGDriver.get_qc_mpo(h1e, g2e, ecore=...,
  algo_type=MPOAlgorithmTypes.FastBipartite)` for both full H and spinor
  Dyall. Block2 owns the integral-to-expression-to-MPO conversion. All
  integral cutoffs, including `fast_cutoff`, are zero. No custom source
  expression builder or Conventional compatibility patch enters this path.
- `Linear` retains its native `Automatic` solver selection (MinRes in this
  version), rather than forcing GCROT. Its reference bond allowance is
  `reference.info.bond_dim + 400`, as in `block2main`; the left environment
  uses native delayed contraction and disables cached contraction.
- The native density-matrix decomposition is retained. The benchmark uses
  tighter, full-length local threshold schedules: local defaults are not a
  certificate of global accuracy.
- Default production sweeps use native `Linear.solve` without local
  projectors, environment rebuilding, or source QR. Only the independent
  `strict_cas=True` validator uses `Linear.blocking` and enforces the whole-CAS
  constraint after every update. The eight-sector cross-check also uses
  native `Linear.solve`.

The default now uses the upstream signs `L=E_D0-H_D`, `R=H-E_D0` and the
physical first-order wavefunction `Psi1`. The complete bare H integrals
are passed to Block2, not just core-dressed terms having nonzero action on
CAS kets. Dyall retains core-dressed active one-electron and AAAA two-electron
integrals plus semicanonical core/virtual one-body terms and its constant.
Spin-free `DyallFCIDUMP.initialize_from_1pdm_sz/su2` is not used for X2C spinors.

The shared full MO AO2MO helper is reused. Full-H UC requires `_SpinorERIs`;
compact Wick blocks alone do not determine the complete H. Omit `eris` from
`kernel()` to generate complete MO integrals. SC/FIC and the existing hybrid
paths retain their compact-block storage. Full MO integral storage is
O(n_mo^4), distinct from the avoided O(n_active^8) 4-RDM.

Conventional was checked in installed Block2 0.5.4rc16: its SGF middle
transform reused the annihilation-pair coefficient for the creation-pair
operator without complex conjugation. A tiny native matrix element differed
from independent fermion action by 1.20438e-3; an isolated PD-coefficient
conjugation diagnostic reduced that difference to 3.88e-17. The patched
tiny response had residual 2.44e-12, but **no patch was merged into production
or installed Block2**. FastBipartite bypasses that transform. This finding
is specific to the checked version/path, not a claim about all Block2 modes.

## Equation and constants

For one reference root, the default solves `L Psi1 = b` with `L = E_D0-H_D`
and `b = (H-E_D0)|Psi0>` in `MRCIMPSInfo(2)`, including its CAS sector.
The independent validator keeps the equivalent notation `A=-L`, `x=-Psi1`.
With `strict_cas=True`, it solves `Q A Q x = Q b` in the full
external space. The whole CAS block is **not** a zero operator: only
reference eigenstates at E_D0 are zero modes in the exact-reference limit.
The two separately measured energy estimators are

```
native default:
E_projected = Re <b|Psi1>
F[Psi1]     = -Re <Psi1|L|Psi1> + 2 Re <b|Psi1>

strict validator (x=-Psi1):
E_projected = -Re <b|x>
F[x]        = Re <x|A|x> - 2 Re <b|x>
```

The reported correction is the Hylleraas functional. No SC/FIC energies are
used to supply missing classes. Default total energy includes its measured
CAS contribution: the eight external contributions are not silently summed
to discard CAS contamination. `cas_hylleraas_energy` reports the difference.
The production path requests only raw SGF
1/2-RDMs, not 3/4-RDMs or determinant expansions; it has no t-NEVPT2 dependency.

The positive-sign operator A=-L on the full chain is
`H_active + sum_i eps_i n_i + sum_r eps_r n_r - E_active - sum_i eps_i`.
`E_D0 = E_active + E_core,electronic + E_nuclear` is checked against the
selected root energy, never the SA mean. Core/virtual Fock blocks are
semicanonicalized; supplying inconsistent `canonicalized=True` data raises
an error instead of dropping off-diagonal Fock elements.

For a state-averaged reference, call `kernel(root=k)` separately for each
retained root. Each call uses that root's MPS, raw 1/2-RDMs and reference
energy to prepare its own Fock/Dyall operator and overall response. It does
not use the SA density or copy another root's correction. Core/virtual
semicanonicalization can give different representations of the same orbital
spaces; the active orbitals and reference MPS are not reoptimized. These
are state-specific corrections, not a multistate effective Hamiltonian:
`E_tot[k] = E_reference[k] + E_corr[k]`. Convergence and CAS leakage must be
checked independently for every root; no averaging enforces degeneracy.

| Class | Core holes | Virtual electrons | Active electron change |
|---|---:|---:|---:|
| ijrs | 2 | 2 | 0 |
| rsi | 1 | 2 | -1 |
| ijr | 2 | 1 | +1 |
| rs | 0 | 2 | -2 |
| ij | 2 | 0 | +2 |
| ir | 1 | 1 | 0 |
| r | 0 | 1 | -1 |
| i | 1 | 0 | +1 |

In the independent eight-sector/strict validators, sources are coherent
sums of core-normal-ordered one-/two-body terms in
`nevpt2_mps_response._source_tensors`; pair factors are for full index sums,
not ordered tuples. The independent oracle projects the full electronic
Hamiltonian by fermion creation/annihilation, without reusing that table.
The Dyall operator preserves each external occupation pattern with these
semicanonical blocks, justifying the separate-sector comparison.

## Three diagnostics, separate from the native solve

- CAS residual: measured from the actual RHS only when `diagnostic=True`.
  The default leaves `cas_source_norm=None`; it does not add a squared-MPO
  variance contraction whose cancellation floor cannot resolve this residual.
- Zero-mode leakage: `abs(<reference|x>)/||x||` is measured independently of
  the whole-CAS response fraction. No final projection or energy averaging
  is performed. A zero-mode component can be appreciable while leaving the
  response equation and energy unchanged; neither implies pure Q space.
- `diagnostic=True` explicitly requests a potentially expensive post-solve
  audit: construct `L Psi1 - b` using the actual left/right MPOs passed to
  `Linear`, including CAS residual and their actual sign/scale. This also
  measures `||P_CAS b||/||b||` and the eight sector residuals. Default
  `diagnostic=False` reports unmeasured global residuals as `None`.

`converged` certifies the measured equation residual, not external-space
purity. The latter has its own `cas_purity_certified` flag. No local threshold
or sweep stopping flag substitutes for a true global residual.

## Optional full-CAS validator, not the production default

`MRCIMPSInfo` alone includes CAS. We impose
`P_CAS = product_i n_i product_r (1-n_r)`, identity on the entire active space,
and `Q = 1-P_CAS` within the MRCI occupation bounds **only when explicitly
requested with `strict_cas=True`**.

1. Measure the full projected source norm, scale its MPO, and project each
   local RHS into the allowed domain. No lossy source fit is performed.
   Keep the physical source norm separately.
2. For every two-site update, let `V` embed the current renormalized basis.
   The allowed local domain is `ker(P_CAS V)`, not merely orthogonality to
   the reference. QR/SVD of left/right restricted basis maps builds this
   kernel without enumerating CAS states. In particular, `V† P_CAS V` is
   not incorrectly treated as an orthogonal projector.
3. Unitary bond gauges turn the forbidden local directions into coordinate
   masks. Apply the mask to the initial guess, RHS, and both sides of every
   native matrix-vector action. The scalar Dyall shift is inside that action.
   Noise and a generic diagonal preconditioner are disabled here: neither
   is automatically domain-preserving. The native `Automatic` iteration remains
   Block2's solver; no custom linear solver is substituted.
4. After every native two-site truncation, constrain the truncated MPS before it
   becomes the next iterate. This is not a final energy/state cleanup.
   Exact `Q` application can increase stored bonds to at most twice the
   compression cap; actual retained dimensions and leakage are reported.

The numerical rank threshold is based on machine precision and normalized
boundary maps. Its chosen rank is retained through import/canonicalization,
and the resulting physical CAS-map error is checked. Projecting a dense
floating-point vector repeatedly was insufficient near singular directions;
coordinate masking keeps forbidden Krylov coordinates exactly zero.

Neither a CAS penalty/level shift, projection against one/six roots, nor
only an RHS projection replaces this constraint. The full Dyall MPO is
unchanged on both CAS and external sectors; no artificial CAS extension is
used to define the response energy. A native-default/strict-validator
comparison measures the consequence of relaxing this constraint; agreement
is tested, not presumed.

## Accuracy and present limits

When explicitly requested, actual residuals are formed as tensor-network MPS differences
`A x - b` using the original, uncompressed source, rather than subtracting
three nearly equal squared norms. Diagnostics include whole/per-class
residuals, both energy estimators, complex inner
products, actual sweeps/bonds, source/response CAS leakage, and every
truncated update's leakage. An unmeasured residual is `None`; it does not
certify convergence. Finite residual failures warn and set `converged=False`.
Source fitting error is identically zero because no fitted source is
used; this does not claim exact floating-point arithmetic. No general
positive-definiteness certificate is claimed for arbitrary roots.

Current implementation: threaded C1 SGFCPX, no nonzero frozen space. The
default uses native solve/noise/truncation machinery. Only the strict
validator requires fixed sweeps, zero noise and environment rebuilding.
Explicit residual audits can require large exact tensor products; absence
of 4-RDM does **not** imply negligible diagnostic memory.

Each public UC call creates a separate `DMRGDriver` and temporary scratch
directory. Reference tensors are copied before activating PT; MCSCF driver
attributes, tags, Hamiltonian and schedule are not reconfigured. Block2 has
a process-global frame, so these drivers are used **sequentially**, never
concurrently. Original frame/allocators/threading are restored on return or
failure. `pt.stack_memory` is PT stack memory in MB, `pt.n_threads` is its
thread count, and `pt.scratch` selects the temporary-directory parent.

Keep `M0` (reference) and `M1` (response) distinct. UC's default cap is 500,
not an inheritance of the MCSCF cap. Diagnostics report requested and actual
reference/response dimensions. Fix reference, orbitals and Dyall definition
while scanning M1; increasing M0 is a separate convergence study.

For a converged, live `mc`, before `mc.fcisolver.close()`:

```python
from socutils.mrpt import X2CUCNEVPT2

pt = X2CUCNEVPT2(mc)
pt.stack_memory = 4096  # MB, independent of the reference driver
pt.n_threads = 16
pt.mps_response_options = dict(max_bond_dimension=128, n_sweeps=12)
pt.run(root=0)  # core/virtual semicanonicalization; active MPS basis unchanged
print(pt.e_corr, pt.e_tot, pt.converged)
print(pt.sub_eners)
print(pt.diagnostics["global_relative_residual"])
# For an expensive, independent post-solve audit:
# pt.mps_response_options["diagnostic"] = True
# For the separate fully projected validation workflow:
# pt.mps_response_options.update(strict_cas=True, diagnostic=True)
```

## Measured validation, 2026-10-04

The first tables below record the strict-validator milestone, not a claim
that the current native default enforces its projectors. All measurements
use native `Automatic` and density-matrix decomposition. The combined suite
was rerun after the independent-driver, full-integral `get_qc_mpo`, and
coefficient-residual-audit revisions. Earlier numerical milestones are
distinguished below.

Tests ran directly on gpu01 CPU with unlimited stack and one BLAS/OpenMP
thread. Python 3.12.7, PySCF 2.14.0, Block2 0.5.4rc16, NumPy 2.5.2,
SciPy 1.18.1. Neither CD nor Kramers restriction is used.

```bash
ulimit -s unlimited
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /usr/bin/time -v .venv/bin/python -m pytest -p no:cacheprovider --tb=short -xq \
  tests/test_x2cucnevpt2.py tests/test_nevpt2_mps_response.py \
  tests/test_x2cficnevpt2.py
```

- Combined suite: **30 passed** (16 new cases and 14 existing response/FIC
  regressions), pytest 143.63 s, elapsed 144.28 s, peak RSS 425648 KiB
  (415.67 MiB). This includes oracle, boundary, gauge, five-method symmetry,
  and existing SC/FIC hybrid calls. The run shared gpu01 with the independent
  F benchmark; RSS refers to the test process only.
- The complete existing SC Wick suite also passed: **45 passed**, pytest
  22.92 s, elapsed 23.46 s, peak RSS 473788 KiB (462.68 MiB). Its one warning
  comes from deliberately violating a 4-to-3-RDM contraction in a validator
  test, not from a production calculation. The command is the same as above
  with `tests/test_x2cscnevpt2_wick.py` as the test target.
- The new full-integral `get_qc_mpo(FastBipartite)` test checks full H and
  Dyall actions, including non-CAS kets, against independent fermion rules.
  Its native response gives E2 = -0.016911416102501613 Eh and actual-equation
  relative residual 1.34675e-12.
- Tiny model: genuinely complex integrals, 2 core + 6 active + 2 virtual
  spinors, 5 electrons. All eight classes have nonzero sources.
- At the middle full-rank local problem, the projector is Hermitian and
  idempotent and removes **20** CAS dimensions from **252**, leaving **232**.
  Independent determinant amplitudes verify all 20 CAS components vanish.
- A CAS contamination vector orthogonal to the first **six** CAS eigenstates
  is injected into the actual native RHS MPO and initial response;
  energies/residuals pass. This is not a diagnostic-only RHS modification.
- An explicit SVD example fills an originally zero CAS block; the same
  constrained truncation used by the solver removes this leakage.
- Low-M response remains flagged unconverged. Phase/weak-source scaling and
  exact zero source pass; high-RDM entry points are forbidden in the API test.
- The retained reference MPS/MO and driver attributes are checked unchanged;
  reinitializing the reference driver is forbidden during PT. Both normal
  return and an injected PT exception restore its native frame and leave
  its 1-RDM usable; temporary PT scratch is removed after the exception.
  Inconsistent external Fock blocks are rejected.
- Independent fermion action on the retained DMRG reference verifies the
  eight measured residuals. Another guard rejects source-MPS compression
  inside the solver: only response truncations are permitted.

### Final correctness/performance review

The installed `block2main` was checked against the production space,
left/right signs, reference bond allowance, `MovingEnvironment` lifecycle,
native `Automatic` solver and density-matrix truncation. There is one whole
response solve and two MPOs, not eight production MPO/solver pairs. The eight
class measurements are post-processing of that same solved MPS. Strict CAS
constraints and explicit residual products remain opt-in; the default is
not claimed to guarantee a pure external response.

The final review removed the unconditional squared-MPO CAS variance
contraction: it was extra work and could not resolve the F CAS residual
above its cancellation floor. Unmeasured values remain `None`; explicit
auditing still measures the actual RHS. Coefficient scaling now reads one
class at a time rather than retaining all eight integral-block copies
throughout the solve. No physical formula or native sweep algorithm changed.
Removing structural zero blocks from the generic class projector was tried
and **reverted** after a low-M empty-sector contraction failed in Block2's
algebra interface; correctness takes precedence over that proposed shortcut.

Residual aggregation now uses measured squared residuals, avoiding `0*inf`
when a source-free sector has response leakage. Infinite relative residuals
remain visible, and a failing class supplies an explicit failure reason.
The public API test injects this boundary case and requires an unconverged
warning. No relaxation of the numerical certificate was made.

Final verification ran directly on gpu01 CPU with the command above:

- UC/hybrid/FIC: **31 passed**, pytest 140.32 s, elapsed 140.92 s,
  peak RSS 425016 KiB (415.05 MiB),
  `tests/nevpt2_mps_response/uc_f_run/final_review_regression.out`.
- SC Wick: **45 passed**, pytest 23.59 s, elapsed 24.21 s,
  peak RSS 474284 KiB (463.17 MiB), same deliberate validator warning,
  `tests/nevpt2_mps_response/uc_f_run/final_review_sc_regression.out`.
- After the last integral-lifetime simplification, the native-default/strict,
  public-API/zero-source-leakage, and full-H/Dyall-action tests were rerun:
  **3 passed**, pytest 35.69 s, elapsed 36.33 s, peak RSS 348404 KiB,
  `tests/nevpt2_mps_response/uc_f_run/final_review_native_regression.out`.

No final-version real-F speedup is claimed: the completed two-point timings
below precede these post-processing simplifications. The redundant M64/S2
explicit-product audit was stopped after the independent coefficient audit
completed; its partial log is not a convergence certificate. No reference
checkpoint or other calculation was removed or cancelled.

Strict-validator two-sweep, compression-cap 64 comparison (actual retained M = 32):

| Quantity | Measured value |
|---|---:|
| Overall MRCIMPSInfo E2 | -0.016911416102501565 Eh |
| Eight NEVPTMPSInfo E2 sum | -0.016911416102500760 Eh |
| Independent exact-UC E2 | -0.016911416096981737 Eh |
| Overall versus eight difference | 8.05e-16 Eh |
| Overall versus exact difference | 5.52e-12 Eh |
| True overall relative residual | 5.40e-13 |
| Maximum nonzero-class relative residual | 8.23e-12 |
| Source CAS norm | 3.22e-17 |
| Response CAS norm | 9.37e-18 |
| Maximum forbidden Krylov coordinate fraction | 0 |
| Maximum boundary CAS-map error | 7.32e-15 |

| Class | Overall-response E2 / Eh |
|---|---:|
| ijrs | -0.000000338557134418 |
| rsi | -0.000011754601199894 |
| ijr | -0.000006794864504672 |
| rs | -0.000018591147652996 |
| ij | -0.000001596254463736 |
| ir | -0.001981043583198334 |
| r | -0.008824772959453135 |
| i | -0.006066524134894388 |

The exact oracle uses an exact CAS eigenvector; the MPS calculations use the
independently converged DMRG reference. Their small reference difference is
not removed to force energy agreement. These tiny checks are separate from
the completed six-root F blocked benchmark above, which still has residual
warnings and does not certify restored quartet degeneracy.

Empty and fully occupied active spaces (0 and 6 electrons in 6 spinors)
were subsequently checked against exact UC. Two sweeps were insufficient
for the empty-CAS case; six sweeps at unchanged M=64 and solver thresholds
reduce the worst nonzero-class residual to 3.41e-11. The full-CAS case has
maximum 9.59e-12. Pauli-forbidden sectors and a model with no core/virtual
space return exact zeros without NaNs or false failures.

### Orbital representation check

The API was also tested on the same tiny Hamiltonian/reference expressed in
three bases: original, random complex active rotation, and additional random
complex core/virtual rotations. The reference CI is transformed by the
independent determinant exterior power and then imported as an SGF MPS;
only rotating integrals would be an invalid test. The external rotation
makes both Fock blocks non-diagonal, and the real MCSCF canonicalizer is
called to diagonalize them without changing the active basis.

| Response cap | Original E2 | Active-rotated E2 | All-block-rotated E2 |
|---|---:|---:|---:|
| M=4, 4 sweeps | -0.012049191165852 | -0.010195873185824 | -0.010160523896054 |
| M=64, 4 sweeps | -0.016911416096982 | -0.016911416096982 | -0.016911416096982 |

At M=4 the true relative residuals are 0.58--0.66 and every calculation is
correctly flagged unconverged. At M=64 they are 5.47e-13--6.34e-13, all
energies agree with exact UC, and the representation spread is below
3e-17 Eh. This is a gauge-covariance test, not a claim of restored physical
degeneracy or an imposed Kramers restriction.

The same three-basis test subsequently passed for the **native default**.
At M1=64 its energies were -0.016911416096981834, -0.016911416096981790,
and -0.016911416096981786 Eh; actual-equation residuals were
3.66e-13, 4.10e-13, and 4.02e-13. At M1=4 its residuals remained
0.59--0.66 and it was correctly uncertified. Both modes are now covered
by the parametrized regression and included in the combined run.

### Controlled unitary-symmetry doublet

A separate genuinely complex 10-spinor model has an ordinary unitary D3
symmetry, with 120-degree rotations and reflections. Its integrals are
group-invariant by construction. Two orthogonal CAS states in the same
two-dimensional E irrep have energy -0.2818801206635116 Eh. Both the full H
and the fixed Dyall operator preserve this symmetry. No antiunitary/Kramers
operation or symmetry restriction is passed to any PT solver: it runs C1
SGFCPX. The fixture, reference states, and complete eight-class energies are
reproducible in `test_unitary_d3_doublet_sc_fic_hybrids_and_full_uc`.

| Method | E2 of first state / Eh | Absolute doublet splitting / Eh |
|---|---:|---:|
| SC | -0.002656282874934479 | <5e-18 |
| FIC | -0.002656343665807906 | <5e-18 |
| SC+UC(i,r) | -0.002656341998466066 | <5e-17 |
| FIC+UC(i,r) | -0.002656343665807960 | <5e-17 |
| Full UC | -0.002656343666494659 | <5e-18 |

Full UC agrees class-by-class with the independent exact oracle within
1e-10 Eh and every nonzero-class residual is below 1e-8. Its i/r agree with
the existing hybrid solver at the same tolerance. This model has **no
resolved artificial SC/FIC splitting**. It validates covariance and the
five-method comparison, not a claim that UC repaired a splitting.

### Sequential physical X2C example

`examples/24-x2c_dmrg_uc_nevpt2.py` ran on gpu01 with one CPU thread:
BH/STO-3G, CAS(4e,6 spinors), full-ERI X2CAMF (Gaunt/Breit off), no CD/KR,
second-order DMRG-SCF followed by native full-UC M1=128, 6 sweeps and an
explicit global audit. MCSCF converged at macro 4. With separate reference
and PT drivers and full-integral `get_qc_mpo(FastBipartite)`, the complete
example took 3.18 s with peak RSS 296716 KiB (289.76 MiB), including
HF/MCSCF/PT. The energy matches the earlier compact-source run, but the
current residual certificate fails for the weak i class:

```
E(MCSCF) = -24.781706346266699 Eh
E2(UC)   =  -0.013275424698908 Eh
E(total) = -24.794981770965606 Eh
global relative residual = 5.62136781602423e-10
maximum class relative residual = 2.2673114498193202e-8 (i)
converged = False
reference zero-mode fraction = 0.1823559927525824
CAS purity certified = False
```

The i source norm squared is 4.1186557830506104e-13. Its relative residual
exceeds 1e-8 even though the overall residual passes; it is not dropped or
used to relax the convergence check. This run is not fully certified.

This is a physical X2C API smoke test; it does not replace the larger F
six-root benchmark. The F runner `tests/nevpt2_mps_response/f_uc.py` restores
the archived, fingerprint-checked reference without optimizing it again.
Its preflight estimates tensor-product source storage only, not peak RSS.
For an M1 scan, `UC_M=64,128,256` runs all three values in one process on
identical prepared RDMs, semicanonical orbitals and integral blocks. Their
fingerprints are printed as `F_UC_PREPARED_INPUT`. Set `UC_PREFLIGHT=0` to
solve and `UC_DIAGNOSTIC=1` to audit residuals with tensor-network products.
Alternatively, `UC_COEFFICIENT_AUDIT=1` selects the test-only fixed-N audit
described below; it requires `UC_DIAGNOSTIC=0` and `UC_STRICT_CAS=0`.
Without either measurement the run does not certify global convergence.
This avoids confounding an M1 scan
with independently regenerated semicanonical gauges. Per-point time is
reported separately; process peak RSS covers the whole multi-point run.

The full-H preflight completed on gpu01 in 2047.00 s (total process elapsed
2072.29 s), peak RSS 41.190 GiB. Its uncompressed RHS tensor-storage bound
was 65.097 GiB, with a largest single tensor bound of 21.798 GiB; these are
storage bounds, not measured residual-audit peaks. No response was solved
by this preflight.

### Test-only coefficient residual audit

`tests/nevpt2_mps_response/uc_audit.py` measures the **already solved native
MPS**. It contracts each core-hole/virtual-particle occupation pattern,
expands only the fixed-N active coefficient sector, and applies the Dyall
active Hamiltonian in small batches. It does not solve a determinant-space
response, change `Linear`, or enter the production API. CAS coefficients
and their residual are retained, not projected away. This diagnostic is
exponential in active-space size and restricted to small C1 SGF models
(at most 20 active spinors with nonempty core/virtual spaces).

The coherent source uses the tested shared source tensors. In
`test_qc_mpo_complex_spinor_actions_and_response`, every external pattern's
source is separately checked against independent full fermion-Hamiltonian
action; every extracted MPS coefficient is checked against native SGF
coefficients. The measured global residual and energy agree with the
independent full-H/Dyall coefficient calculation. This augmented targeted
test passed in 24.10 s (process elapsed 24.72 s), peak RSS 337568 KiB
(329.66 MiB), with E2=-0.016911416102501613 Eh and relative residual
1.34675e-12. This augmentation is included in the latest 30-case run above.

The subsequent audit revision also records the maximum relative residual
over **every nonzero occupation pattern**, its hole/particle tuple and
source/residual norms, plus absolute leakage into exactly zero-source
patterns. No weak-source cutoff is used. The tiny test checks these maxima
against independent full fermion-Hamiltonian action for both a random
unconverged MPS and the solved response. This targeted revision passed in
23.69 s (24.01 s process elapsed), peak RSS 342348 KiB (334.32 MiB).
These measurements are test-only; production `Linear` is unchanged.

For F, its independent certificate is stored as
`F_UC_RESULT.independent_coefficient_audit`; it does not overwrite the PT
API's unmeasured-residual `converged=False`. The fixed-reference root-0
M1=128/256, 12-sweep, local-threshold 1e-24 run completed both points with
the same prepared-input fingerprints. Requested and actual M1 agree; M0
was requested as 1000 and actually used 186. The combined process completed
in 2:49:07 with peak RSS **51.072 GiB** (53552980 KiB). This high-water mark
includes both points and the opt-in independent audits, not a per-point or
production-only peak.

| M1 | MPO construction / s | Native sweeps / s | Independent audit / s | Whole point / s |
|---:|---:|---:|---:|---:|
| 128 | 930.660 | 2264.115 | 1114.475 | 4538.849 |
| 256 | 938.655 | 2732.733 | 1180.202 | 5563.669 |

The first forward M1=128 sweep took 723.942 s. Whole-point timings additionally
include reference embedding, MPS conversions and post-solve energy/CAS
measurements; these runs predate removal of the default CAS variance
contraction. There is no measured speedup claim for that subsequent change.

The measured class contributions and actual-equation relative residuals are:

| Class | M1=128 energy / Eh | Residual | M1=256 energy / Eh | Residual |
|---|---:|---:|---:|---:|
| ijrs | -0.0279753328531864 | 0.127509488 | -0.0282809178408618 | 0.073726232 |
| rsi | -0.0091011095324388 | 0.166758401 | -0.0093031445805201 | 0.110443624 |
| ijr | -0.0057437602506943 | 0.195547106 | -0.0058503272277410 | 0.134531306 |
| rs | -0.0758565393933602 | 0.118610468 | -0.0763953477219967 | 0.064559956 |
| ij | -0.0006301101885551 | 0.186183809 | -0.0006422270457797 | 0.124693685 |
| ir | -0.0059446311412108 | 0.281350638 | -0.0062293580219547 | 0.175152426 |
| r | -0.0332926676178010 | 0.268638172 | -0.0345350085580439 | 0.117045902 |
| i | -0.0009273325588218 | 0.302412040 | -0.0009764275587131 | 0.200459388 |

The eight-class sum is -0.15947148353606835 Eh. The native unconstrained
full Hylleraas result includes a separately reported CAS contribution
+2.3597208917e-5 Eh, giving E2=-0.15944788632715143 Eh and
Etot=-99.73291598650127 Eh. The independent audit and native MPO energy
contractions agree within 6e-16 Eh in every external class, and 1e-16 Eh
overall. Their agreement validates these measurements, **not convergence**:
the true full response relative residual is 0.148744081, and the largest
external-class residual is 0.302412040. The independent certificate is false.
The physical full RHS and residual norms are 3.28154208 and 0.488109963 Eh.
External-class overlap imaginary defects reach 1.24489e-6 Eh even though
their summed imaginary part is near 4e-16 Eh; cancellation of class defects
is not taken as proof that each class has converged.

For M1=256, the eight-class sum is -0.16221275855561101 Eh and the CAS
contribution is +3.3945884844e-6 Eh. Full E2=-0.16220936396712660 Eh and
Etot=-99.73567746414123 Eh; E2 changes by -2.761477640 mEh from M1=128.
The true full residual decreases to 0.088516830 and the largest external
class residual to 0.200459388. Native and independent energy contractions
again agree within 1e-15 Eh. CAS response fraction is 0.0551142 and
reference zero-mode fraction 0.0533946. This is an unconverged finite-M
result, not a resolved UC contraction correction relative to SC/FIC.

The measured CAS source norm is only 1.45774e-9 Eh, but the CAS residual
norm is 0.0109303 Eh. Its source/whole-RHS norm ratio is 4.44224e-10,
not the response leakage. CAS response/total response norm is 0.0744081 and
reference zero-mode leakage is 0.0705495. The CAS contribution is not an
external dynamic-correlation class and is not silently removed to improve
the reported energy. At this finite M, the native unconstrained workflow
has appreciable CAS leakage and is not an externally pure response.

The full projected and Hylleraas energy estimates nevertheless agree to
2e-15 Eh, and backward sweep target differences are near 1e-16. Neither is
a convergence certificate: the printed native `F` is a selected local sweep
target, not a post-truncation global residual. No real-F energy or residual
convergence is claimed. That process imported the earlier
class-level coefficient audit; the per-pattern maxima added afterwards
will be measured by subsequent validation runs, not retroactively claimed
for this process.

An earlier compact-source F root-0 pilot (not the current full-H MPO path)
used M1=128, 6 sweeps and local threshold 1e-24. It returned
E2=-0.15914706296539805 Eh in 1329.47 s, peak RSS 6.316 GiB. Its global
residual was **not measured**, so it is explicitly unconverged. The
reference zero-mode fraction was 0.06321 and the CAS energy contribution
was +2.18723e-5 Eh. This is not a completed M-convergence or symmetry-root
benchmark, nor validation of the new full-H entry point.

The retained reference was also audited without modifying its orbitals or
MPS, using `PYTHONPATH=tests python -m nevpt2_mps_response.f_reference_symmetry`
with the same `UC_REFERENCE`. This uses the AO antiunitary map and raw
root 1/2-RDMs, not root numbers alone. The core/active TR mixing Frobenius
norm is 4.37e-11 and active/virtual mixing is 2.86e-9; active H1/H2 maximum
TR defects are 1.15e-10 / 4.26e-10 Eh. The low four roots do not have a
one-to-one TR pairing in their retained basis (all individual partner
1-RDM distances exceed 0.94). For roots 4/5 the partner 1/2-RDM distances
are 1.56e-8 / 7.80e-8. These density-level diagnostics alone are not many-body
overlaps or exact manifold-closure proofs. In particular, six equal SA
weights alone are not treated as a symmetry certificate. Averaging densities
is used only to inspect subspaces; no reference or correlation energy is
averaged or changed for production.

The same script now also reads native fixed-N SGF coefficients from copies
of all six retained MPSs and applies the many-body antiunitary map. It uses
second quantization of `log(T_active)` and SciPy sparse `expm_multiply`, not
an alpha/beta CI mapping, a new eigensolve, or a KR constraint. This test-only
exponential diagnostic is checked against an independent dense exterior-power
matrix on a complex six-spinor model, including coefficient conjugation and
T^2=-1. The targeted test passed in 13.92 s (elapsed 14.61 s), peak RSS
243044 KiB (237.35 MiB); it is an additional test beyond the 30-case run.

The actual F coefficient audit completed in 28.77 s, peak RSS 498788 KiB
(487.10 MiB). Its six-root Gram-matrix defect is 2.66e-15 and the many-body
T^2=-1 defect is 1.77e-14. The measured absolute overlaps
`|<Psi_i|T Psi_j>|` are:

| i / j | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|---:|
| 0 | <2e-15 | 0.086187 | 0.739011 | 0.668158 | <1e-8 | <1e-8 |
| 1 | 0.086187 | <2e-15 | 0.668158 | 0.739011 | <1e-8 | <1e-8 |
| 2 | 0.739011 | 0.668158 | <2e-15 | 0.086187 | <1e-8 | <1e-8 |
| 3 | 0.668158 | 0.739011 | 0.086187 | <2e-15 | <1e-8 | <1e-8 |
| 4 | <1e-8 | <1e-8 | <1e-8 | <1e-8 | <2e-15 | 0.999999999999998 |
| 5 | <1e-8 | <1e-8 | <1e-8 | <1e-8 | 0.999999999999998 | <2e-15 |

Roots 4/5 are therefore the appropriate directly paired retained states.
T mixes roots 0--3; comparing any chosen pair there would not be a valid
one-to-one time-reversal benchmark. Leakage out of each energy manifold
is 3.56e-9--1.05e-8 (quartet) and 1.12e-8 (doublet); leakage out of the full
six-root space is 7.90e-10--1.64e-9. These finite reference errors are reported
without projecting or averaging states/energies. The near-unit overlaps do
not certify full-UC response convergence or exact Hamiltonian symmetry.

The directly paired roots were subsequently checked with their **separate
state-specific** Fock matrices and the actual L=E0-HD definition. In the
common retained MO basis, time reversal gives one-body coefficient errors
3.55e-9 Eh overall, with core/active/virtual diagonal-block errors
4.82e-10/1.15e-10/1.48e-9 Eh. The scalar-shift difference is 1.32e-10 Eh;
the active two-body time-reversal error above is 4.26e-10 Eh. Transforming
between the two root-specific semicanonical representations gives a
3.90e-9 Eh one-body error. No density, partition, state or energy is averaged
or projected. These are finite covariance errors, not exact symmetry.

The actual semicanonical external-Fock checks for roots 4/5 give
7.28e-11/2.76e-10 Eh, both below the API's 1e-9 Eh input gate. Active plus
core reference energies agree with the retained root energies within
2.28e-13 Eh. The complete read-only check, including the preceding
coefficient tracking, finished in 28.54 s with peak RSS 498932 KiB
(487.24 MiB). This prepares the paired-root benchmark; it does not replace
actual UC solves for those roots.

### Native default versus strict validator

The tiny six-sweep M1=64 comparison gave native E2
`-0.016911416102501540` Eh versus strict `-0.016911416102501572` Eh
after the full-integral entry-point revision.
The native actual-equation residual was `3.8395e-13`, its CAS-source/RHS
norm ratio `2.0214e-9`, and its reference zero-mode fraction `0.23876`.
The CAS energy contribution was `-3.47e-18` Eh. Thus energy agreement
does not prove external-space purity. A regression explicitly forbids
source materialization and calls to the strict solver in the default path.
The same test independently extracts the native response coefficients on
the tiny determinant space and applies fermionic H and Dyall matrices.
Its actual-equation residual, CAS-source fraction and Hylleraas energy
agree with the tensor-network diagnostics within 1e-11. This includes the
CAS block; it is not a check only of the eight external sectors.

The controlled D3 doublet was also rerun with the unconstrained native
default. Its two **full** Hylleraas energies (CAS term included) were
`-0.002656343666494667` and `-0.0026563436664946588` Eh, a splitting below
1e-17 Eh. Each external class agrees with the exact oracle within 1e-10 Eh;
both overall and all nonzero-class residuals pass 1e-8. The strict validator
agrees at this accuracy. No correlation energies were averaged and no
reference- or response-state Kramers restriction was imposed.

### Antiunitary time-reversal check

The additional `test_antiunitary_time_reversal_without_kramers_restriction`
defines a genuinely complex, time-reversal-invariant tiny Hamiltonian and
the antiunitary map T=J K (including coefficient conjugation and fermionic
permutation signs). Its odd-electron CAS states satisfy T^2=-1 and are
orthogonal partners; full H, the partition, and the fixed Dyall operator
preserve this map. The native C1 SGF solver is not given J or a KR constraint.
Both members match independent exact UC in all eight classes within
1e-10 Eh and all class residuals pass 1e-8:

| Partner | Full native E2 / Eh | Global relative residual |
|---|---:|---:|
| Psi | -0.008406773372397410 | 3.39574e-13 |
| T Psi | -0.008406773372397442 | 2.81710e-13 |

The targeted run passed on gpu01 in 26.78 s (27.41 s process wall time),
peak RSS 322824 KiB (315.26 MiB). It is also included in the subsequent
30-case combined run above, not a completed time-reversal analysis of the
real F six-root data.

References: [upstream UC keywords](https://block2.readthedocs.io/en/latest/user/keywords.html#uncontracted-dynamic-correlation),
[block2main](https://github.com/block-hczhai/block2-preview/blob/master/pyblock2/driver/block2main),
[Sharma–Chan MPS-PT](https://arxiv.org/abs/1408.5092).
