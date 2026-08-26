# Super-CIPT reference and Block2 validation

This record covers the Guo--Dutta two-component perturbative Super-CI
(Super-CIPT) optimizer and the replacement of the historical Pykylin CI
boundary by the common `socutils.dmrg.DMRGCI` Block2 solver.  The existing
full Super-CI/Davidson optimizer remains the default CASSCF path.

## Immutable historical baseline

The reference tree was kept byte-for-byte unchanged at commit
`5c00c55f3a7f2c8cda6fdc6da3d0b2a5c4c972b0`.  It lives in the nested Git
tree `note/socutils_gy` and was clean both before and after all validation.
Its `mcscf/zmc_supercipt_new.py` imports the old package as `socutils`, while
the directory itself is named `socutils_gy`; an isolated temporary directory
therefore supplied a `socutils` symlink.  Its `zdmrgci.py` calls
`pykylin.core.DMRGSolver` through a non-PySCF contract: `kernel` takes the
number of roots positionally and returns an energy list, and integer root IDs
stand in for CI vectors.

The runnable Pykylin extension is a CPython 3.11 binary.  The baseline used
Python 3.11, NumPy 1.26.4, SciPy 1.16, and PySCF 2.9.0 from the existing
`pyscf_311` environment.  The historical checkout omits the imported
`somf/settings.py`; an import-only two-line shim supplied its expected
`AMFIEXE = "amf_pyscf.exe"` setting, and the current pure-Python `x2camf`
package path supplied the otherwise missing import.  Neither workaround
changed the historical source.

The first untouched launch stopped at the missing `x2camf` import.  Its full
10-line output is preserved at
`/tmp/socutils-old-baseline.SteTtE/old_supercipt_full.log` with SHA-256
`4558578ad49e320ba9bb124822ccb42e4bfd248f5e35ff0ee3ba0cd4d2c93b93`.
The successful run's complete 45,529-line output, including every Pykylin
sweep, root energy, density trace, Koopmans eigenproblem, orbital step, and
canonicalization, is preserved at
`/tmp/socutils-old-baseline.SteTtE/old_supercipt_full_with_settings.log` with
SHA-256
`6c02690c47e4664283f591bca8c63c4cf947cdca29c4279dc236bca656ba80c8`.
These large generated logs are intentionally not committed.

### Historical calculation

The old example first converges an F- `cc-pVTZ` PySCF X2C mean field
(`E_SCF = -99.50566305876` Eh).  It then changes the molecular charge/spin to
neutral F, retains those initial orbitals, and uses CAS(7 electrons, 8
spinors), six equally weighted roots, and direct spinor Coulomb integrals.
The old object is also assigned an X2CAMF helper, although its
`CASBase.get_hcore()` delegates to the original X2C mean field.  Super-CIPT
uses a maximum matrix-element step of 0.2, `conv_etol = 1e-8`,
`conv_gtol = 1e-3`, at most 20 macroiterations, and Pykylin `MAX_M = 1000`.

Before Pykylin was used, old exact CI gave

```text
roots = [-99.37731710, -99.37731710, -99.37731710,
         -99.37731710, -99.37404055, -99.37404055]
average = -99.37622491560582 Eh
```

Pykylin reproduced the average as `-99.37622491560585` Eh.  The complete
numerically significant macroiteration record extracted from the raw log is:

| macro | six Pykylin root energies (Eh) | average (Eh) | gradient norm |
| ---: | --- | ---: | ---: |
| 0 | -99.377317098226, -99.377317098216, -99.377317098207, -99.377317098186, -99.374040550402, -99.374040550398 | -99.376224915606 | 7.2009e-1 |
| 1 | -99.477166714806, -99.477166714801, -99.477166714792, -99.477166714774, -99.474669126461, -99.474669126454 | -99.476334185348 | 2.2031e-1 |
| 2 | -99.481438723990, -99.481438723984, -99.481438723975, -99.481438723957, -99.478817201192, -99.478817201185 | -99.480564883047 | 3.7045e-2 |
| 3 | -99.481648833539, -99.481648833530, -99.479055937726, -99.479055937719, -98.596764766040, -98.596764766015 | -99.185823179095 | 3.7837e-2 |
| 4 | -99.508822109690, -99.482055204844, -99.481150476117, -99.479051897010, -99.478762595333, -99.454062384772 | -99.480650777961 | 2.7103e-2 |
| 5 | -99.495103637173, -99.481929928610, -99.481381731062, -99.479106784725, -99.478994002423, -99.468147028159 | -99.480777185359 | 8.1790e-3 |
| 6 | -99.488365611483, -99.482040228265, -99.481281927473, -99.479127082062, -99.478999469044, -99.474944303896 | -99.480793103704 | 2.9445e-3 |
| 7 | -99.485016382338, -99.481979785111, -99.481343795403, -99.479124217840, -99.479014773535, -99.478297638422 | -99.480796098775 | 1.4023e-3 |
| 8 | -99.483345359683, -99.481886803273, -99.481437491645, -99.479986572822, -99.479103446446, -99.479021021758 | -99.480796782605 | 6.8309e-4 |
| 9 | -99.482509052898, -99.481808186243, -99.481516534201, -99.480819217890, -99.479092446530, -99.479036266905 | -99.480796950778 | not evaluated |

The macro-3 discontinuity is a Pykylin multi-root collapse, not an orbital
equation feature.  The old loop nevertheless recovers.  It tests
`abs(dE) < etol OR gradient < gtol` and checks the previous step's gradient
before forming the next one; consequently it exits at macro 9 without
evaluating a final gradient.  The last available value is `6.8309e-4` at
macro 8.  Natural occupations and a final RDM are not printed by this old
example, so no values are invented for them.

### Historical tensor boundary

The old solver consumes `h1[p,q]` and physical chemist-order
`eri[p,q,r,s]`.  Pykylin returns

```text
raw_dm1[p,q]       = <p+ q>
raw_dm2[p,r,s,q]   = <p+ r+ s q>
```

so old `zdmrgci.make_rdm12()` uses `raw_dm2.transpose(0,3,1,2)`.  It then
applies an eight-term Hermitian/fermionic projection.  The replacement does
not need or apply that projection: the common Block2 solver's raw converted
RDMs already satisfy the required identities.  An arbitrary random complex
tensor was deliberately rejected as a Pykylin reference because its integral
loader assumes physical spinor ERI permutation structure.  All reported
Pykylin comparisons therefore use an actual X2C Hamiltonian.

## Fixed-orbital solver replacement

The small common Hamiltonian is tilted H--F,
`H 0 0 0; F 0.35 0.27 0.8035`, Coulomb-only X2CAMF/STO-3G,
CAS(2 electrons, 4 spinors), with `ecore = -96.76450580532774` Eh.  The exact
same saved `h1`, unsymmetrized `eri`, and `ecore` were loaded by old PySCF,
Pykylin, current exact CI, and Block2.  This independently crosses the Python
3.11/3.12 and PySCF 2.9/2.14 boundary.

| solver | energy (Eh) | max 1-RDM error | max 2-RDM error | RDM energy error |
| --- | ---: | ---: | ---: | ---: |
| old `fci_dhf_slow` | -98.63728554177987 | 2.04e-15 | 2.05e-15 | 2.0e-17 |
| old Pykylin | -98.63728554177987 | 2.14e-15 | 6.61e-14 | 2.0e-17 |
| current `zfci` | -98.63728554177987 | reference | reference | 4.8e-17 |
| current Block2 | -98.63728554177987 | 2.14e-15 | 2.09e-15 | 3.24e-17 |

The legacy Pykylin RDM passed through the current equations changes the
same-orbital gradient by `2.85e-15` and the first anti-Hermitian step by
`1.36e-15` versus current exact CI.  The complete Pykylin output is
`/tmp/socutils-old-baseline.SteTtE/old_physical_fixed_ci.log`, SHA-256
`74fda625ef258e1e0ded88905fb43e38c1af113731717fc8174112059d8cc0b9`.

## Equation map and port

The implementation follows Guo and Dutta, *J. Chem. Theory Comput.* **22**,
7154--7163 (2026), DOI `10.1021/acs.jctc.6c00400`, and retains the paper and
historical authorship attribution in `mcscf/zmc_supercipt.py`.

| paper | old source | current NumPy object |
| --- | --- | --- |
| eqs. 1--4, Hamiltonian/RDMs | `h1e_mo`, `eris.paaa`, `casdm1`, `casdm2` | `h1e[p,q]`, `eri[p,q,r,s]`, `dm1[p,q]=<p+q>`, `dm2[p,q,r,s]=<p+r+sq>` |
| eqs. 8--13, Dyall Hamiltonian | `fock`, `fock_eff` | `fock_core = h + f_occ`; `fock_effective = fock_core + f_act` |
| eqs. 15--18, independent blocks | three assignments to `kappa` | core--virtual, core--active, and active--virtual slices selected by `uniq_var_indices` |
| eq. 19, orbital gradient | `g - g.T.conj()` | `lagrangian - lagrangian.T.conj()` |
| eqs. 20--23, Koopmans problems | `compute_K1_K2`, `solve_Fc_eSc` | removal matrix in metric `D`; addition matrix in metric `I-D`, solved by canonical orthogonalization |
| eq. 24 | direct inactive--virtual denominator | `G[a,i] / (f[i,i]-f[a,a])` |
| eq. 25 | `compute_k_it` | hole-metric/addition eigenvectors and `f[i,i]-epsilon_add` |
| eq. 26 | `compute_k_ta` | density-metric/removal eigenvectors and `epsilon_remove-f[a,a]` |
| eq. 27 | `expm(kappa)` | anti-Hermitian `kappa`, maximum-element cap, then `C @ scipy.linalg.expm(kappa)` |

The two-RDM contribution is explicitly

```text
Q[p,t] = sum_u,v,w eris.paaa[p,u,v,w] * dm2[t,u,v,w].
```

The removal Koopmans matrix is the Hermitian part of
`-fock_active @ dm1.T - Q_active`; the addition matrix adds the active block
of `fock_effective`.  Density and hole-density metrics are positive
semidefinite, so eigenvectors below `supercipt_metric_tol` are removed instead
of being sent to an ill-conditioned generalized eigensolver.  A configurable,
sign-preserving denominator level shift and an explicit singular-denominator
guard replace silent division by a near-zero value.

State averaging is not a separate orbital algorithm.  PySCF's
`state_average_(weights)` supplies the weighted scalar energy and weighted
1-/2-RDM, and every quantity above is built from those same weights.  The
paper uses equal weights; arbitrary normalized PySCF weights work as well.
Convergence now requires both `abs(dE) < conv_tol` and the nonredundant orbital
gradient below `conv_tol_grad`, fixing the historical `OR`/stale-gradient
condition.

`CASSCF.supercipt()` is an explicit additional API.  `CASSCF.kernel()` and
`CASSCF.superci()` still invoke the pre-existing full Super-CI/Davidson
optimizer.  Super-CIPT reuses `_ERIS`/`_CDERIS`, current Cholesky JK and
two-RDM contractions, the common exact/DMRG solver contract, callbacks,
logging, convergence snapshots, and core/virtual subspace canonicalization.

## Orbital-optimization validation ladder

An all-block regression uses the same tilted H--F system with CAS(2e,3
spinors): eight core, three active, and one virtual spinor.  A deterministic
anti-Hermitian initial tilt has nonzero core--active, core--virtual, and
active--virtual elements.  Exact and Block2 runs use the same initial MO
array, maximum step 0.2, `conv_tol = 1e-10`, and
`conv_tol_grad = 1e-5`.  Block2 uses one thread, bond dimension 16, eight
sweeps, zero noise, energy tolerance `1e-12`, local squared residual
`1e-20`, Davidson maximum 1000, NPDM site type 2, and seed 2468.

The old equations were loaded directly from the immutable file and driven
first with exact CI and then with the common `DMRGCI`; no second Block2
wrapper was created.  Their complete output is
`/tmp/socutils-old-baseline.SteTtE/old_workflow_block2_validation.log`,
SHA-256
`cb5ddb7f0d91d132b857b205155d627a527c546013650a005438fc07b0ab7954`.

| path | initial energy (Eh) | final energy (Eh) | final CAS energy (Eh) | final gradient | iterations |
| --- | ---: | ---: | ---: | ---: | ---: |
| old equations + exact CI | -98.63628437827134 | -98.63650918755290 | -1.871857885573590 | 6.1854802e-6 | 17 |
| old equations + Block2 | -98.63628437827134 | -98.63650918755285 | -1.871857885573618 | 6.1854802e-6 | 17 |
| current port + exact CI | -98.63628437827134 | -98.63650918755262 | -1.871857885575267 | 6.1857722e-6 | 17 |
| current port + Block2 | -98.63628437827134 | -98.63650918755262 | -1.871857885575267 | 6.1857722e-6 | 17 |

| comparison | result |
| --- | ---: |
| old exact/Block2 maximum trajectory energy difference | 9.95e-14 Eh |
| old exact/Block2 maximum trajectory gradient difference | 1.23e-15 |
| old exact/Block2 final 1-/2-RDM differences | 2.29e-15 / 2.29e-15 |
| old/current exact final energy difference | 2.84e-13 Eh |
| old/current same-orbital full-gradient norm difference | 2.12e-16 |
| old/current first raw orbital update difference | 9.17e-8 |
| old/current final gradient difference | 2.92e-10 |
| old/current natural-occupation difference | 6.66e-16 |
| old/current core/active/virtual projector errors | 3.61e-10 / 1.27e-10 / 3.61e-10 |
| current exact/Block2 maximum trajectory energy difference | 7.11e-14 Eh |
| current exact/Block2 final 1-/2-RDM differences | 7.98e-16 / 7.98e-16 |
| current exact/Block2 core/active/virtual projector errors | 2.00e-15 / 9.99e-16 / 7.28e-16 |

The final occupations are `[1.0, 1.0, 0.0]` to displayed precision.  This
small determinant-valued endpoint is why the separate four-active-spinor
fixed-orbital case above is retained: it supplies a correlated, nontrivial
RDM comparison, while the three-active-spinor case exercises every orbital
rotation block and the complete macroiteration trajectory.

The historical diagnostic norm includes redundant active--active entries;
the current convergence norm screens to the independently optimizable blocks.
At the shared initial orbitals their displayed norms are therefore
`0.0142947293160142` and `0.0142808973596876`, respectively, even though the
full-gradient norms agree within `2.12e-16`.  This deliberate diagnostic
change does not alter any Super-CIPT block.

The six-root F/cc-pVTZ current exact run converges monotonically in eight
energy/gradient evaluations:

| macro | current exact average energy (Eh) | gradient norm |
| ---: | ---: | ---: |
| 0 | -99.37622476956264 | 7.2009121e-1 |
| 1 | -99.47633417861618 | 2.2030567e-1 |
| 2 | -99.48056488262486 | 3.7045103e-2 |
| 3 | -99.48078453488506 | 1.1159505e-2 |
| 4 | -99.48079630709262 | 1.7770127e-3 |
| 5 | -99.48079696668704 | 6.0042926e-4 |
| 6 | -99.48079700580055 | 9.4460819e-5 |
| 7 | -99.48079700825880 | 3.4976326e-5 |

The locked current PySCF release regenerates the initial X2C orbitals with a
`1.46e-7` Eh change in the first state-average relative to PySCF 2.9, so that
cross-version number is not treated as a same-Hamiltonian solver error.  The
stationary energy differs from the original Pykylin result by
`5.75e-8` Eh, within the `1e-7` target.  Unlike the old Pykylin run, the exact
six-root space has no macro-3 root collapse.

## Commands and scope boundary

The durable tests are:

```bash
env CUDA_VISIBLE_DEVICES=1 uv run pytest -q tests/test_supercipt.py
env CUDA_VISIBLE_DEVICES=1 uv run pytest -q
make PYTHON=.venv/bin/python test
```

`tests/test_supercipt.py` independently covers the metric eigensolver, fixed
energy/RDM/RDM-energy/gradient/first-step agreement, all-block exact/Block2
macroiterations, the six-root historical endpoint, projectors, occupations,
and the explicit Kramers restriction.

Kramers-restricted Super-CIPT orbital equations are intentionally not
implemented: the cited implementation and historical reference provide only
general complex-spinor Super-CIPT.  Kramers-restricted exact/DMRG CASSCF
remains available through the validated full Super-CI path.

Milestone commit message: `mcscf: port validated Super-CIPT optimizer to block2`.
