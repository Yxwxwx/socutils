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

Exactly degenerate complex Block2 `MultiMPS` roots cannot safely be handled
by blindly splitting the state-averaged object.  In the four-spinor test the
reported two energies were exact, but the split states had overlap magnitude
`2.74e-1`; their RDM-reconstructed energies were `0.0641` and `0.2889` Eh
instead of `-0.1715` Eh.  Kramers mode consequently optimizes each root as a
single MPS and projects all earlier roots with a configurable positive shift
(`root_projection_shift=100` Eh by default).  The resulting overlap and
projected-Hamiltonian residuals are both checked before the RDMs are accepted.
The ordinary general-complex multi-root route still uses `MultiMPS` and has a
dedicated unchanged-path regression.

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
(the concrete hash is reported in the next milestone and final run log).

## Known limitations at this milestone

Both general and Kramers modes conserve particle number only. `ci0` is
explicitly ignored because an MPS from the preceding CASSCF macroiteration is
not valid after an orbital rotation without a validated MPS orbital transform.
The CASCI solver boundary receives the reconstructed active four-index tensor;
Block2 does not consume Cholesky factors directly.  Kramers mode retains the
full spinor space and requires a complete even root manifold for odd-electron
systems; pair members used by PySCF state averaging must have equal weights.
The state-specific projection shift is configurable for unusually wide active
spectra.  Kramers-restricted Super-CIPT is outside the later Phase-4 scope.
