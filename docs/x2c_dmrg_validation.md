# X2C DMRG validation

This document records the reproducible numerical gates for the relativistic
Block2 integration. Results are added only after the corresponding tests pass.

## Locked environment

The validated environment is Python 3.12.7, NumPy 2.5.2, SciPy 1.18.1,
PySCF 2.14.0, h5py 3.16.0, and Block2/pyblock2 0.5.4rc16. The lock uses the
official Block2 preview index because 0.5.4rc16 is the newest official build
available with the required CPython 3.12 complex-driver wheel and is newer
than the compatible stable 0.5.3 release.

From a clean checkout:

```bash
uv sync
make PYTHON=.venv/bin/python
uv run python -c "import pyscf; import block2; import pyblock2; import socutils"
```

The native build deliberately uses the BLAS shipped with the locked PySCF
wheel. This avoids loading the host MKL alongside the Block2 wheel. On Linux,
the native RPATH includes PySCF's library directory so hashed OpenBLAS runtime
dependencies are resolvable; the small bundled `zquatev` C++ library links its
GNU C++ runtime statically to remain loadable from the site's older Anaconda
runtime.

## Integral and density-matrix conventions

The `zfci` reference and its PySCF `fci_dhf_slow` cross-check use

\[
H = \sum_{pq}h_{pq}a_p^\dagger a_q
  + \frac12\sum_{pqrs}(pq|rs)a_p^\dagger a_r^\dagger a_s a_q,
\]

with raw, unsymmetrized chemist-order `eri[p,q,r,s] = (pq|rs)`. The density
matrices are

```text
dm1[p,q]       = <a_p† a_q>
dm2[p,q,r,s]   = <a_p† a_r† a_s a_q>
```

The installed Block2 QC MPO accepts the same raw chemist-order integrals; it
does not accept an antisymmetrized tensor at this boundary. The conversions
are centralized in `dmrg/dmrgci.py`:

| quantity | Block2 result | socutils result | operation |
| --- | --- | --- | --- |
| one-body integral | `h[p,q]` | `h[p,q]` | copy |
| two-body integral | `g[p,q,r,s]` | `eri[p,q,r,s]` | copy |
| 1-RDM | `raw[p,q]` | `dm1[p,q]` | copy |
| 2-RDM | `raw[i,j,b,a]` | `dm2[p,q,r,s]` | `raw[p,r,s,q]` |
| transition 1-RDM | `<bra|p†q|ket>` | same | copy; no conjugation |

No RDM symmetry projection or averaging is performed. The raw converted
tensors satisfy trace, contraction, Hermiticity, and creation/annihilation
antisymmetry identities before comparison with exact FCI.

For these definitions, the source-verified energy contraction is

```python
E = (
    np.einsum("pq,pq->", h1e, dm1)
    + 0.5 * np.einsum("pqrs,pqrs->", eri, dm2)
    + ecore
)
```

The alternative `h[p,q] * dm1[q,p]` expression sometimes used with the
McWeeney density convention is not equivalent here. A deterministic complex
unitary test gives the exact energy with the contraction above and fails
strongly with the transposed expression; this was also checked directly
against current `fci_dhf_slow` source and RDM output.

## Phase 1: DMRG-CI versus exact relativistic CASCI

Tiny exact-representation runs use one thread, seed 2468, bond dimension 32,
up to eight sweeps, zero noise, energy tolerance `1e-12`, MPO/integral cutoff
`1e-20`, NPDM cutoff `1e-24`, NPDM site type 2, Davidson maximum 1000, and a
per-local-solve squared residual threshold of `1e-14` (residual bound
`1e-7`). Block2 stopped after two converged sweeps in the molecular test; the
last maximum root-energy change was below `1e-14` Eh and the discarded weight
was zero. The public driver exposes the configured local residual threshold,
not a separately measured final local residual, so the validation record does
not mislabel the former as the latter.

| case | reference solver | DMRG solver | E_ref (Eh) | E_dmrg (Eh) | max \|ΔE\| | max \|Δdm1\| | max \|Δdm2\| | RDM energy error | gradient difference | final gradient norm |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| analytic 1e / determinant RDMs | analytic + zfci | Block2 SGFCPX | -0.763470103745470 | same within `1e-14` | `<1e-14` | `<1e-14` | `<1e-14` | `<1e-14` | n/a | n/a |
| complex-unitary 4 spinor, 2e, 2 roots | zfci + `fci_dhf_slow` | Block2 SGFCPX | -8.898063666144473, -6.811636031468529 | same | 2.132e-14 | 3.246e-15 | 3.164e-15 | `<1e-13` | n/a | n/a |
| tilted HF X2CAMF/STO-3G CAS(2e,4 spinors), weights 0.4/0.3/0.3 | zfci CASCI | Block2 CASCI | -98.37324729276855 | -98.37324729276854 | 1.421e-14 | 7.065e-15 | 1.037e-14 | `<1e-13` | n/a | n/a |

The complex-unitary transition 1-RDM error after global phase alignment is
2.979e-15. In the molecular case the first excited level is an exact
two-dimensional degeneracy. Its complete subspace is therefore included with
equal weights and compared through the invariant pair-averaged RDM; the
nondegenerate ground-state RDM is also compared root by root.

Commands for this milestone:

```bash
uv sync --all-groups
make PYTHON=.venv/bin/python test
uv run pytest -q tests/test_environment.py tests/test_dmrgci.py
uv run pytest -q tests/test_dmrgci_x2c.py
```

Milestone commit: `dmrg: validate block2 solver against relativistic CASCI`
(`61921a0da3f6d13281936de1b30aae237059165b`).

## Phase 2: Cholesky X2C-DMRG-SCF with Super-CI

The Cholesky boundary uses real spatial-AO factors

```text
(mu nu|lambda sigma) = sum_P L[P,mu,nu] L[P,lambda,sigma]
cd_pa[P,p,u] = sum_mu,nu C[mu,p]* L[P,mu,nu] C[nu,u]
cd_aa[P,t,u] = cd_pa[P,ncore+t,u]
(p u|v w) = sum_P cd_pa[P,p,u] cd_aa[P,v,w]
```

The last reconstruction is bilinear: neither factor is conjugated.  With
`dm2[t,u,v,w] = <t† v† w u>`, the two-particle orbital-gradient term is

```text
g_dm2[p,t] = sum_P,u,v,w cd_pa[P,p,u] cd_aa[P,v,w] dm2[t,u,v,w].
```

The four local pivoted-Cholesky implementations previously returned
`chol_vecs[:nchol]` even though `nchol` tracks the final valid vector index.
The variants that form a pending next vector now stop before doing so when the
residual has reached threshold, making the inclusive return contract
unambiguous.  On tilted HF/STO-3G the old default-route slice omitted a vector
and left a `1.145e-6` maximum AO ERI residual.  Returning through `nchol + 1`
gives 21 vectors and a maximum residual of `8.882e-16` at `tau=1e-10`.

The tilted HF test uses Coulomb-only X2CAMF, STO-3G, geometry
`H 0 0 0; F 0.35 0.27 0.8035`, and CAS(2 electrons, 4 spinors).  Its active
ERI has a maximum imaginary component of `5.431e-3`, so it independently
exercises the complex Cholesky transformation.  The fixed-orbital checks are:

| fixed-orbital quantity | maximum difference |
| --- | ---: |
| reconstructed AO ERI vs direct AO ERI | 8.882e-16 |
| Cholesky `aaaa` vs direct transform | 7.846e-16 |
| Cholesky `paaa` vs direct transform | 7.846e-16 |
| deliberately conjugated `aaaa` reconstruction | 2.795e-2 |
| core J / K | 1.776e-15 / 1.110e-15 |
| active J / K | 5.274e-16 / 2.220e-16 |
| two-RDM gradient contraction, max / norm | 2.220e-16 / 3.769e-16 |
| complete orbital gradient, max / norm | 1.776e-15 / 3.193e-15 |

At those identical orbitals, exact FCI and Block2 use byte-identical `h1eff`
and active ERIs and the same `ecore`.  With the Phase-2 local squared-residual
threshold, their energy, 1-RDM, 2-RDM, RDM energy, and orbital-gradient
differences are `2.422e-17`, `3.113e-15`, `3.103e-15`, `2.416e-17`, and
`7.684e-16` (maximum), respectively.

The full orbital-optimization reference is Be/STO-3G with Coulomb-only X2CAMF,
deterministic `1e` SCF orbitals, CAS(2 electrons, 4 spinors), 2 core spinors,
and 4 virtual spinors.  Both solvers start from the same copied MO array.  The
Cholesky threshold is `1e-10`; the energy and gradient convergence thresholds
are `1e-9` Eh and `1e-5`; the maximum orbital-step norm is `0.1`.  The Be
decomposition contains 15 Cholesky vectors.

The Super-CI Davidson solve uses tolerance `1e-10`, maximum subspace 20, and
strict failure on a nonconverged residual.  This is distinct from Block2's
local eigensolver.  The corrected complex Gram-Schmidt uses `vdot`, and the
reported convergence measure is the full generalized augmented-Hessian
residual after unit-reference normalization.  Every nonfinal Be
macroiteration converges in one expansion with residual between `2.86e-13`
and `4.59e-13`; the previous zero-step convergence test forced all ten old
iterations regardless of residual.

Block2 uses one thread, seed 2468, bond dimension 32, at most eight sweeps,
zero noise, energy tolerance `1e-12`, local squared-residual threshold
`1e-20` (configured residual bound `1e-10`), Davidson maximum 1000, NPDM site
type 2, and NPDM cutoff `1e-24`.  The tighter local threshold is material: a
single-root fixed-orbital run at `1e-14` had a `4.746e-8` 1-RDM error despite
an essentially exact energy, while `1e-20` reduced it to `3.113e-15`.

| mode | E_FCI-SCF (Eh) | E_DMRG-SCF (Eh) | \|Delta E\| | max \|Delta occ\| | final gradient difference | max projector difference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| active natural orbitals disabled | -14.376636105842476 | -14.376636105842476 | 0.000e+0 | 4.441e-16 | 1.505e-16 | 1.416e-15 |
| default natural orbitals + canonicalization | -14.376636105842474 | -14.376636105842477 | 3.553e-15 | 1.110e-15 | 5.715e-16 | 1.665e-15 |

Both paths converge in five accepted orbital steps.  The fixed-active exact
macroiteration record (the DMRG values agree within `1.8e-15` Eh) is:

| macro | total energy (Eh) | energy change (Eh) | CAS energy (Eh) | gradient norm | applied step | active occupations (each twice) |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | -14.376549573629577 | initial | -0.935340105706885 | 4.937539e-3 | 1.159350e-2 | 0.93534105 / 0.06465895 |
| 1 | -14.376611564102143 | -6.199047e-5 | -0.935373661719886 | 2.624906e-3 | 8.629190e-3 | 0.93534125 / 0.06465875 |
| 2 | -14.376633130381160 | -2.156628e-5 | -0.935764841052880 | 9.115018e-4 | 4.203322e-3 | 0.93546663 / 0.06453337 |
| 3 | -14.376636081593569 | -2.951212e-6 | -0.936068462239922 | 8.214274e-5 | 3.871218e-4 | 0.93556591 / 0.06443409 |
| 4 | -14.376636105714534 | -2.412096e-8 | -0.936100149099001 | 5.981439e-6 | 2.819383e-5 | 0.93557630 / 0.06442370 |
| 5 | -14.376636105842476 | -1.279421e-10 | -0.936102481364827 | 4.362332e-7 | converged | 0.93557706 / 0.06442294 |

The final active natural occupations are
`[0.93557706, 0.93557706, 0.06442294, 0.06442294]`.  Core, active, and
virtual spaces are compared through overlap-orthonormalized projectors rather
than raw MO coefficients.  Macroiteration histories, Cholesky source and
threshold, CI convergence data, natural occupations, applied steps, and the
Super-CI residual are available as `macro_history`, `cholesky_diagnostics`,
and `superci_diagnostics` on the CASSCF object.

Commands for this milestone:

```bash
uv run pytest -q tests/test_superci_cholesky.py
uv run pytest -q tests/test_x2c_dmrg_scf.py
uv run pytest -q
```

Milestone commit: `dmrg: align Cholesky X2C-DMRG-SCF with super-CI`
(`5bcea1bdab8a77f6620777a017502d3effaa93d6`).

## Phase 3: Kramers-restricted X2C-DMRG-SCF

Kramers support is explicit and solver-side.  Block2 continues to use the
full SGFCPX spinor space; no half-space RDM is inferred or reconstructed from
unvalidated blocks.  Enable the result adapter with
`DMRGCI(...).init(...).kramers_restricted()`.  Omitting that final call leaves
the general complex-spinor behavior from Phases 1--2 unchanged.

### Actual orbital and time-reversal convention

PySCF's `mol.time_reversal_map()` is a signed, one-based forward map.  If its
entry for AO spinor `i` is `-j`, then `Theta |i> = -|j>`; applying it to a
coefficient vector is consequently a scatter to `j`, not a gather from `j`.
For active MO coefficients `C` and AO overlap `S`, the adapter constructs

```text
Theta(C)[:,p] = time reversal of MO column p
U = C^H S Theta(C)
Theta |p> = sum_q |q> U[q,p]
```

and validates `U^H U = I`, `U U* = -I`, `U = -U^T`, closure of the active MO
subspace under time reversal, each partner orbital, and its phase.  Partners
are selected from the dominant entries of the measured `U`; adjacency is
never assumed.  For the repository's `zquatev` interleaved H2/STO-3G output,
the measured pairs are `(0,1)` and `(2,3)`, with
`Theta C_0 = +C_1` and `Theta C_1 = -C_0`.  The largest closure, partner, and
phase errors are `4.45e-16`, `4.05e-16`, and zero, respectively.

For the repository RDM axes, time reversal is

```text
dm1_TR[p,q] = sum_ab U[p,a]* U[q,b] dm1[a,b]*
dm2_TR[p,q,r,s]
    = sum_abcd U[p,a]* U[q,b] U[r,c]* U[s,d] dm2[a,b,c,d]*
```

where `dm2[p,q,r,s] = <p† r† s q>`.  The conjugations are intentionally the
transpose of the common AO density-matrix formula.  An exterior-power exact
CI check with complex pair phases verifies both transformations directly.

### Roots, ensembles, and projection

An individual odd-electron member of a Kramers doublet is not required to
have a time-reversal-symmetric RDM.  The adapter therefore:

1. targets a complete, even number of roots;
2. pairs roots using their energy splitting and the raw 1-/2-RDM relation;
3. checks the root overlap and projected Hamiltonian;
4. forms an equal-weight pair average in the full active spinor space; and
5. reports the raw ensemble time-reversal residual before any projection.

Energy-degeneracy, pair-RDM, and manifold-RDM tolerance failures are soft
result-quality checks: they emit ``RuntimeWarning`` and are retained in
``kramers_diagnostics['validation_warnings']`` with
``validation_passed=False``.  They do not discard a usable raw CASSCF result.
Malformed orbital/root input, inconsistent split MultiMPS roots, and a
requested projection whose raw residual exceeds ``projection_tolerance``
remain hard errors.

Multi-root calculations use one Block2 state-averaged `MultiMPS`, with its
weights copied from the PySCF state-average wrapper.  Four-spinor diagnostics
showed that splitting a pure two-site endpoint can produce MPSs inconsistent
with the reported energies: the default Olsen preconditioner fails at an exact
degeneracy, while Davidson preconditioning fails in a nondegenerate test.
Finishing with one-site sweeps removes both failures for Olsen, Davidson,
exact-local, and unpreconditioned solvers.  `DMRGCI` therefore inserts a
two-site-to-one-site transition for multi-root jobs and checks both `S - I`
and `H - S E` for every split root space before accepting any RDM.

RDM projection is off by default.  With `project=True`, the adapter replaces
an already validated pair density by `(D + Theta(D))/2`; it refuses to do so
when the raw residual exceeds `projection_tolerance`.  A test deliberately
damages an individual root by `1e-4` and verifies that this gate raises rather
than concealing the error.  The KR X2C-DMRG-SCF validation below consumes the
raw equal-weight pair density and applies no projection.

Degenerate transition densities have both independent root phases and an
arbitrary unitary rotation inside the doublet.  The test assembles the full
`<i|p†q|j>` tensor, selects the Hermitian one-body operator with the largest
projected root separation, diagonalizes it in root space, and fixes remaining
phases from the largest transition entries.  This canonical root-space tensor
is compared with exact CI; raw CI or MPS coefficients are not compared.

### Numerical gates

The exactly solvable active Hamiltonian has three electrons in four spinors,
nonadjacent pairs `(0,2)` and `(1,3)`, two different complex partner phases,
and genuinely complex one- and two-electron integrals.  Block2 uses the same
Phase-2 exact-representation settings: one thread, seed 2468, bond dimension
32, eight zero-noise sweeps, `tol=1e-12`, local squared-residual threshold
`1e-20`, Davidson maximum 1000, NPDM site type 2, and NPDM cutoff `1e-24`.

| active-space quantity | exact FCI versus DMRG difference/residual |
| --- | ---: |
| maximum root-energy difference | 1.055e-15 Eh |
| pair-averaged 1-RDM | 6.438e-16 |
| pair-averaged 2-RDM | 6.645e-16 |
| canonical root-space/transition 1-RDM | 1.022e-15 |
| raw ensemble time-reversal residual | 8.327e-16 |
| root-overlap residual | 4.442e-16 |
| projected-Hamiltonian residual | 1.712e-16 Eh |

The full molecular gate is one-electron H/6-31G with Coulomb-only KR-X2CAMF,
Cholesky threshold `1e-10`, CAS(1 electron, 2 active spinors), two virtual
spinors, and equal weights `(0.5,0.5)` over the lowest Kramers pair.  A
Kramers-preserving `0.18`-radian active/virtual tilt makes the orbital
optimization nontrivial.  Active natural-orbital rotations use the
repository's Kramers eigensolver at every macroiteration; core/virtual
canonicalization is immaterial for this zero-core gate and is disabled.
Super-CI uses energy/gradient thresholds `1e-10` and `1e-7`, maximum step
`0.1`, Davidson tolerance `1e-11`, maximum subspace 20, and strict residual
failure.

| KR X2C-SCF quantity | exact FCI | Block2 DMRG | difference/residual |
| --- | ---: | ---: | ---: |
| final total energy (Eh) | -0.498241138750104 | -0.498241138750104 | 0.000e+0 |
| final gradient norm | 3.468701e-8 | 3.468701e-8 | 0.000e+0 |
| state-averaged 1-RDM | -- | -- | 3.103e-17 |
| state-averaged 2-RDM | -- | -- | 0.000e+0 |
| active/virtual projector | -- | -- | 4.887e-23 |
| 13-macroiteration energy trajectory | -- | -- | 1.110e-16 Eh |
| raw ensemble time-reversal residual | -- | -- | 2.221e-16 |

Commands for this milestone:

```bash
uv run pytest -q tests/test_dmrgci_kramers.py
uv run pytest -q tests/test_dmrgci.py tests/test_dmrgci_x2c.py \
  tests/test_superci_cholesky.py tests/test_x2c_dmrg_scf.py
uv run pytest -q
```

Milestone commit: `dmrg: add validated Kramers-restricted X2C-DMRG-SCF`
(`e6a4ab4f2343fdce54d749633feb391d7e92d8e0`).

## Phase 5: direct-pyblock2 schedule and continuation

The schedule and restart policy now follow the current official
[PySCF `dmrgscf` implementation](https://github.com/pyscf/dmrgscf/blob/master/pyscf/dmrgscf/dmrgci.py).
The repository copy used during the audit and the fetched upstream source had
the same SHA-256 digest. PySCF writes piecewise-constant anchor rows for
`block2main`; `pyscf_dmrg_schedule` expands those rows into the per-sweep
`bond_dims`, `thrds`, and `noises` arrays accepted by `DMRGDriver.dmrg`.
Both interfaces define the Davidson threshold and noise in units of norm
squared, so the numeric values are passed directly rather than squared again.

The generated cold schedule retains the PySCF policy:

1. start at `M=50` below `maxM=200`, otherwise at `M=200`, unless `startM`
   is supplied;
2. double M every four sweeps until `maxM`;
3. reduce the local threshold and noise by decades every two sweeps;
4. finish the noise schedule at `tol/10` with zero noise; and
5. leave eight possible one-site sweeps after the generated two-site endpoint.

The legacy explicit-array interface remains available with
`schedule_mode="explicit"`. The generated path also permits an independent
noise scale and a final Davidson squared-residual threshold. The standalone
`pyscf_dmrg_schedule` generator leaves the official schedule unchanged when
that threshold is omitted. The relativistic `DMRGCI` sets
`schedule_thrd_max=1e-16` by default and stages its Davidson thresholds from
`1e-8` through `1e-16`, one decade at a time. The noise schedule continues to
follow PySCF independently and remains zero if the tighter Davidson schedule
requires additional sweeps.

PySCF's restart callback is reproduced by `restart_scheduler_()`. An orbital
gradient below `1e-3`, or a density change below `1e-2` when supplied, enables
the next CI warm start. The restart uses maximum M, zero noise, the final
`schedule_thrd_max` Davidson threshold, one-site DMRG, and at most eight
sweeps. An arbitrary external `ci0` remains untrusted. Instead, the
solver copies its own structurally compatible saved MPS into a new scratch
directory, constructs a fresh pyblock2 driver and MPO, and reloads that MPS.
This matches the lifecycle of `block2main fullrestart` and treats the old
coefficients only as a guess for the new-orbital Hamiltonian.

The restart call uses `tol=0` to complete all configured sweeps; convergence
is decided afterward from the maximum per-root energy change.  A two-site,
multi-root lattice is the only exception.  It has no interior one-site tensor,
and the locked Block2 build segfaults on repeated one-site direction changes
even with normal energy stopping, so this exact, negligible-cost CAS(1,2)
edge case is solved cold rather than loading a warm-start MultiMPS.

For process-level continuation, `checkpoint_dir` asks pyblock2 to copy a
loadable MPS after every completed sweep. A JSON manifest records orbital and
electron counts, root count and weights, MPO cutoffs, core shift, and a
SHA-256 hash over the complex one- and two-electron integrals. A new solver
with `resume=True` loads either a completed or interrupted checkpoint only
when that exact fingerprint matches. The checkpoint is not deleted by
`close()`. Optional per-sweep archives are available but disabled by default
because of their storage cost.

The final checkpoint now receives an explicit copy of the canonical one-site
MultiMPS after root-space validation.  This matters because
Block2's sweep-time `restart_dir` can otherwise retain two-site metadata from
an intermediate image even though the in-memory calculation finished with
one-site sweeps.  Directly changing that old image's `dot` flag to one-site
was found to corrupt a six-root Cl CAS(7,16) restart (`max|S-I| = 0.1845` and
`max|H-SE| = 2.006` Eh).  Legacy two-site checkpoints are therefore resumed
with two actual conversion sweeps followed by the configured eight one-site
sweeps.  The converted Cl restart recovered the reference energy and passed
the root-overlap/projected-eigenproblem validation; new checkpoints load
directly with `dot=1`.

The protocol-7 Kramers halogen matrix uses CAS(7,8), six equal roots,
`M=16 -> 32`, `tol=1e-10`, the official noise decay, a `1e-12` local-threshold
cap, and the `1e-3` restart gate. Every cold call converged in 23 sweeps; every
final warm call converged in two. All ten exact/DMRG workers completed:

| atom | DMRG - exact total E (Eh) | max root difference (Eh) | gradient difference | final S-I | final H-SE (Eh) |
| --- | ---: | ---: | ---: | ---: | ---: |
| F | 2.842e-14 | 8.527e-14 | 4.391e-15 | 1.475e-15 | 7.106e-14 |
| Cl | -4.547e-13 | 3.979e-13 | -3.440e-15 | 1.894e-15 | 5.686e-14 |
| Br | -1.364e-12 | 1.364e-12 | -3.896e-14 | 9.414e-16 | 8.415e-15 |
| I | 3.638e-12 | 3.638e-12 | 5.948e-14 | 1.555e-15 | 1.162e-14 |
| At | 1.091e-11 | 1.091e-11 | 6.470e-14 | 1.471e-15 | 1.198e-14 |

Unit tests cover literal schedule expansion, both callback vocabularies,
single-root SCF warm starts after NPDM generation, multi-root interrupted
checkpoint reload, and rejection of a one-integral fingerprint mismatch. The
complete repository suite passes 33 tests.

## Known limitations at this milestone

Both general and Kramers modes conserve particle number only. `ci0` is
explicitly ignored; the conditional internal MPS restart is an untransformed
warm guess for a changed orbital Hamiltonian, not an assertion that the two
wavefunctions are identical. Every restarted result must therefore satisfy
the normal energy, root-space, RDM, and orbital-convergence gates. A strict
disk resume covers one active Hamiltonian only; restarting an SCF process also
requires restoring its matching orbital iterate. The CASCI solver boundary
receives the reconstructed active four-index tensor; Block2 does not consume
Cholesky factors directly. Kramers mode retains the full spinor space and
requires a complete even root manifold for odd-electron systems; pair members
used by PySCF state averaging must have equal weights. Kramers-restricted
Super-CIPT is outside the later Phase-4 scope.

With the locked Block2 wheel, the six-root SGFCPX two-site NPDM calculation
prints MKL `ZGEMM` parameter-8 diagnostics for zero-sized symmetry blocks.
The same messages occur with the old flat schedule, cold starts, and no
checkpoint, so they are not caused by schedule generation or continuation.
They do not occur in exact CI or the smaller DMRG tests; all accepted halogen
NPDMs still pass Hermiticity, contraction, Kramers, energy, and exact-CI
comparisons. This noisy upstream path remains worth reporting to Block2.

The immutable old-code baseline, paper equation map, Pykylin replacement, and
Phase-4 exact/Block2 Super-CIPT results are recorded separately in
`supercipt_validation.md`.
