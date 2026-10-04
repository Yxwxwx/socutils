# Block2-style no-4-RDM NEVPT2

The optional branch follows Block2's `E4=None` treatment in
[scnevpt2.py](https://github.com/block-hczhai/block2-preview/blob/master/pyblock2/icmr/scnevpt2.py)
and [icnevpt2_full.py](https://github.com/block-hczhai/block2-preview/blob/master/pyblock2/icmr/icnevpt2_full.py).
The six classes `ijrs/rsi/ijr/rs/ij/ir` retain their contracted equations;
only `i/r` become uncontracted aaac/aaav MPS responses. The resulting methods
are **SC+UC(i,r)** and **FIC+UC(i,r)**, not strict SC/FIC or full UC-NEVPT2.

```python
from socutils.mrpt import WickX2CSCNEVPT2, WickX2CFICNEVPT2

pt = WickX2CSCNEVPT2(mc)  # or WickX2CFICNEVPT2(mc)
e2 = pt.kernel(root=0, mps_response=True, contraction_backend="pytblis")
```

Default `mps_response=False` preserves strict SC/FIC with ranks 1--4.
Enabling it requests only raw SGF RDM ranks 1--3. Response controls can be
overridden through `pt.mps_response_options`; the actual controls and
native solver diagnostics are recorded in `pt.mps_response_diagnostics`.

## Implementation and defaults

Production uses the existing pyblock2 `DMRGDriver`, `ExprBuilder`, `get_mpo`
and `multiply(..., left_mpo=...)`. Native SGFCPX `NEVPTMPSInfo` restricts
one whole aaac MPS to a single core hole and one whole aaav MPS to a single
virtual electron, following
[block2main](https://github.com/block-hczhai/block2-preview/blob/master/pyblock2/driver/block2main).
The original reference tensors are embedded without CI expansion or refitting.
There is no channel-wise solve, source normalization, custom linear solver,
alpha/beta conversion, or dependency on `mrpt/x2ctnevpt2.py`.

This is MPS linear response / Hylleraas minimization, not imaginary-time
t-NEVPT2. With `A X = B Psi`, physical `Psi1 = -X`, the functional is
`<X|A|X> - 2 Re<X|B Psi>`, and the response energy is `-<B Psi|X>`.
Class energies use native `Linear.solve` returns; post-sweep overlap and
stationarity checks remain diagnostics, not global residual certificates.

Defaults are the reference bond dimension and sweep tolerance, eight
sweeps, noise `1e-5`, and truncation cutoff `1e-14`. The upstream input
Davidson threshold `1e-6` is divided by 50 for `Linear`, giving the actual
`multiply` residual-squared threshold `2e-8`. The high-level API uses native
SVD and ReducedPerturbative noise with nonzero noise; this corresponds to
block2main's SVD option, not its density-matrix-decomposition default.
There is no automatic precision retry or denominator shift.

Full-Coulomb PT integral blocks are read directly from dense or compact
storage; compact storage is not a Cholesky approximation. CD and Kramers
restriction are not required. PT core/virtual semicanonicalization explicitly
disables KR projection and leaves active orbitals/MPSs unchanged. Frozen
spinors, MPI and non-C1 orbital symmetry labels are currently rejected.

## Reproducible F validation

The F test follows `p-splittings/17/F`: F-/dyallv3z ordinary spinor X2CAMF HF,
then neutral F CAS(7e,16 spinors), six equal SA weights, reference M=1000,
original orbital order, and `mc.second_order()`. HF/MCSCF/PT all use full
ERIs, with CD and KR disabled. The completed results are summarized in
[report.md](report.md).

Run on gpu01 using CPU only:

```bash
# Creates/reuses the reference and compares root 0.
bash -l tests/nevpt2_mps_response/run.sh
# Reuses that reference; evaluates all six roots and response profiles.
bash -l tests/nevpt2_mps_response/run_six_roots.sh
```

Alternatively, `sbatch tests/nevpt2_mps_response/run.slurm` runs the root-0
comparison with 8 CPUs and 200 GB. The scripts set unlimited process stack
and `OMP_STACKSIZE=256M`. They use per-run NVMe scratch, preserve reference
orbitals/MPS checkpoints, and reuse completed results after fingerprint checks.
The six-root runner uses two independent 16-thread processes and atomically
writes `summary.json`, `energies.txt` and `status.json`.

All roots receive default and M=2000 response audits. Roots 0 and 4 also
receive a controlled sequence: 16 fixed sweeps, linear threshold `1e-22`,
zero noise, cutoff `1e-24`, then M=1500/2000. These are diagnostic overrides,
not new production defaults. The audit measures `||Ax-b||/||b||` by expanding
the existing SGF MPS and independently applying its Hamiltonian; it neither
diagonalizes CI nor changes the reference. This expensive small-CAS oracle
is test-only and is not run automatically by production.

A measured finite global residual above `1e-8` emits
`MRPTNumericalWarning`; energy evaluation continues, with
`residual_status="warning"` and the verification flag still false.
Nonfinite residuals remain errors. Unmeasured residuals are not certified.
Symmetry splittings are reported, not used as restart/failure gates.

Runtime outputs/checkpoints are ignored by Git. Historical outputs are
archived outside the repository; restore them before attempting to resume
an archived run, or run the reference stage again. `class_ab.py` and
`diagnose_r85.py` are historical test-only diagnostics and residual-oracle
helpers, not the production workflow.

## Regression checks

```bash
ulimit -s unlimited
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m pytest -p no:cacheprovider -q \
  tests/test_nevpt2_mps_response.py tests/test_x2cscnevpt2_wick.py \
  tests/test_x2cficnevpt2.py tests/test_nevpt2_memory.py \
  tests/test_x2cqdscnevpt2.py
```

Checks cover independently projected complex Hamiltonian sources/resolvents,
global residuals, original/Fiedler orbital order, dense/compact integral
blocks, exactly two native response solves, unchanged reference MPSs,
rank-3/rank-4 switching, warning behavior, strict SC/FIC and bounded-memory
integrals. Tight small-model regression controls are explicit and are not
claimed to be production defaults.
