# One-step X2C-MS-FIC-NEVPT2

`WickX2CMSFICNEVPT2` implements the SS-SR and MS-MR parametrizations of
Reynolds–Shiozaki, JCTC **15**, 1560–1571 (2019),
[DOI 10.1021/acs.jctc.8b00910](https://doi.org/10.1021/acs.jctc.8b00910).
SOC is already present in the complex-spinor reference and Hamiltonian;
this is not spin-free NEVPT2 followed by SOC state interaction.

The reference orbitals and common Dyall Fock are prepared once from the
specified SA ensemble. Only inactive/virtual orbitals are semicanonicalized.
The dynamic-correlation model roots are an explicitly selected subset of
the SA roots. Frozen orbitals are not yet supported (`frozen=0`).

```python
from socutils.mrpt import prepare_msfic, solve_msfic

prepared = prepare_msfic(mc, sa_roots=range(15), sa_weights=[1/15]*15,
                        model_roots=identified_1D2_roots)
ss = solve_msfic(prepared, ansatz="ss_sr", shift=0.2)
ms = solve_msfic(prepared, ansatz="ms_mr", shift=0.2)
print(ss.energies, ms.energies)
```

The IC basis is `O_alpha |N>`. SS-SR restricts each response column to its
own reference-generated span; MS-MR uses the union over all model roots.
Both keep cross-state source and overlap matrices and diagonalize a full
Hermitian effective Hamiltonian. True first-order amplitudes obey
`(A + eta*S)t = -V`; the full shift correction is
`Heff = Href + (V†t + t†V)/2 - eta*t†S*t`, including off-diagonal elements.
Non-diagonal model Hamiltonians are retained under reference rotations.

Raw transition ranks 0–4 enter the contractions unchanged. Production
preparation checks array shapes and numeric dtypes only, without scanning
RDM values for antisymmetry, reverse-adjoint or particle-number identities.
The standalone RDM validation utility remains available for regression tests.
Each ordered transition pair is written to `.npy` files, released from RAM,
and contracted through read-only memory maps before proceeding to the next
pair. By default only the current pair is staged in `lib.param.TMPDIR`, and
its files are deleted after contraction. Point `lib.param.TMPDIR` at the
job's NVMe scratch when available.

Configure `transition_rdm_fallback_dir="/path/to/large-filesystem/rdm_fallback"`
in `prepare_msfic`, or set `pt.transition_rdm_fallback_dir`, to spill to a
larger filesystem when the preferred directory cannot hold one pair.
The free-space check uses tensor sizes only, not RDM values. Capacity/quota
errors during writing also trigger fallback, reusing the already generated
pair instead of another DMRG call; incomplete staging files are cleaned.
Other errors propagate. This changes only transition-RDM storage, not
Block2's own scratch or PySCF's global temporary directory. Default staging
files on either filesystem are removed after their pair is contracted.

To retain all raw pairs, pass `transition_rdm_dir="/path/to/rdms"` to
`prepare_msfic`, or set `pt.transition_rdm_dir` on `WickX2CMSFICNEVPT2`.
Every preparation creates a fresh subdirectory, reported as
`prepared.diagnostics["transition_rdm_directory"]`, so earlier references
and their root phases are never implicitly reused or overwritten. Each pair
has `dm1.npy` through `dm4.npy` and a `READY.json` written last. Retained
files can be read with `np.load(path, mmap_mode="r", allow_pickle=False)` via
the existing `transition_pdms(bra, ket)` provider. Already read-only mmap
inputs are contracted directly without another disk copy in the default mode.
If retained pairs span both filesystems, their actual directories are in
`prepared.diagnostics["transition_rdm_locations"]`, indexed by `"bra,ket"`.
Six roots and 16 active
complex128 spinors require about 2.26 TiB to retain every rank-1--4 pair;
the default staging needs space for only one pair (about 64.25 GiB).

Finite Hermiticity defects warn before taking the Hermitian part, and the
retained IC residual is recorded with at most one warning per class.
Production solves do not repeat the Hylleraas identity audit for each tuple;
the independent IC-span regression tests validate the energy formula.
Non-finite data, invalid metric/null-source spaces and near-singular
denominators still raise. Invertible excited-state denominators need not
be positive and are never clipped. This does not require an uncontracted
external-space residual to vanish.

For SS-SR, `F - E_L*S` is formed before the metric congruence to avoid
subtracting two ill-conditioned projected products. This is an algebraic
reordering, not a changed threshold or ansatz. Residuals are evaluated
against the raw retained IC matrix, before numerical Hermitian averaging.

Canonical metric removal refines the actual retained Gram matrix when
needed, using the same algorithm as CASPT2. The cutoff, retained rank and
subspace are unchanged. Refined operator congruences use extended accumulation,
parallel over disjoint output rows within `lib.num_threads()`; ordinary blocks
retain BLAS products. MS-MR subtracts a common active-energy origin before the
congruence and restores it algebraically in the denominators. Production arrays
and amplitudes remain complex128. `solve_msfic(..., metric_refinement=False)`
reproduces the former path for comparisons. The completed F joint/4+2
benchmark keeps this former path in both runs. Gram refinement has unit-test
and individual F-block validation, but not yet a complete F energy benchmark.

Native MultiMPS root splitting can introduce an approximately `1e-10`
NPDM gauge error in the installed Block2 version. The C benchmark uses
official `pyblock2.algebra.io.MPSTools` QR on owned MPS copies, with no
truncation or reoptimization, and restores `dot=2` before native NPDM.
Actual determinant coefficients confirm preservation of every root and
phase. This does not force DMRG to switch to one-site optimization.

See [the C input](../tests/msficnevpt2/carbon_input.py),
[the independent tests](../tests/test_x2cmsficnevpt2.py), and
[the C result report](../tests/msficnevpt2/report.md).
The [F input](../tests/msficnevpt2/fluorine.py) and
[F joint/4+2 result report](../tests/msficnevpt2/fluorine_report.md) document
the six-state SA benchmark and separate-manifold comparison.
Existing SC, FIC, QD-SC and UC interfaces retain their semantics.
