This repository adds utility for PySCF to include SOMF corrections within spinor style and GHF style calculations.

### Opt-in adaptive Super-CI

```python
mc.superci_adaptive = False  # default: unshifted, metric-conditioned Davidson
# mc.superci_adaptive = True # additionally bound the solved orbital step
mc.superci()
```

This replaces the job-local `from superci_metric_adaptive import install`
and `install()` calls; remove those calls when using the switch. Selection is
per CASSCF object, with no global overrides or `.tmp_superci_debug` dependency.
For full ERI, unrestricted, unscreened, unfrozen rotations and
`mc.canonicalize_ = False`, both modes use the corrected complex mixed-one-body
Super-CI operator and occupation-metric coordinates. Conditioning no longer
requires enabling adaptive shifts. Other orbital/integral routes retain their
existing operator implementation.

Adaptive mode adds an orbital level shift when the solved step exceeds
`mc.max_stepsize`; a well-conditioned metric no longer bypasses this bound.
It requires Davidson, full ERI (no CD/DF), no Kramers restriction or screened/frozen
rotations, `mc.canonicalize_ = False`, and `mc.natorb = False`. Unsupported
combinations raise an error rather than silently falling back. The corrected
full-ERI route uses the actual linear derivative `2 Re(g† step)` for its
outer energy prediction.
Convergence tolerances are unchanged; inner convergence does not guarantee
outer convergence. The enabled mode is logged and recorded
in `mc.superci_diagnostics['adaptive']`.

Adaptive mode reuses one Krylov space across the projected orbital-shift
search. Adaptive and second-order runs require orbital DIIS/BFGS disabled
and use accepted-point retries. The DMRG warm-start gate receives the actual
proposed step immediately before the corresponding CASCI calculation.

DMRG two-site results with inconsistent MPS expectation values and reported
root energies are marked unconverged. A failed internal warm restart is
retried once with the original full cold schedule and the same tolerances;
the failure is recorded in `fcisolver.convergence_info['restart_fallback']`.

### Second-order complex orbital optimization

```python
mc.orbital_trust_start = 0.2  # initial Frobenius radius
mc.max_stepsize = 0.4        # hard upper bound; default remains 0.2
mc.second_order_micro_step_tol = 1e-4  # 0 restores strict inner solves throughout
mc.second_order()  # keeps the configured active-space solver, including DMRG
```

This uses the fixed-1/2-RDM orbital Hessian (including Coulomb/exchange and
two-particle response) and BAGEL-style scaled augmented-Hessian iteration in
the real tangent space of complex orbital rotations. The step bound is solved
inside the projected AH problem. It requires full ERIs, no Kramers restriction
or frozen/screened rotations, `canonicalize_=False`, `natorb=False`, and no
orbital DIIS/BFGS. The three additional MO integral blocks use approximately
`48 * nmo**2 * ncas**2` bytes; the code checks this against available
`mc.max_memory`. Only 1/2-RDMs are needed.

Both bounded optimizers shrink the radius on rejected steps and may grow it
after reliable boundary steps, up to `max_stepsize`. Trials start from the
accepted orbitals; a rejected MPS is never used to initialize the retry.
After six failed trials the accepted point is recomputed cold, including its
live CI/MPS, RDMs and checkpoint, before raising an error if a trial CASCI was
attempted. Inner `maximum_space`/`linear_dependence` failures with finite
residuals also shrink the radius, sharing the six-attempt limit. They do not
run CASCI or change its restart state; failures before any trial CASCI leave
the accepted CI/MPS intact. Other errors propagate immediately. Noise-scale energy
rises are allowed only when the orbital gradient is already below tolerance;
the final energy/gradient tests remain unchanged.

Second-order microiterations use the BAGEL-style scaled-residual/step test
away from convergence and restore the strict unscaled Davidson tolerance
when the gradient is within ten times `conv_tol_grad`. Diagnostics distinguish
`converged_inexact` from `strict_converged`, with the actual residual and
effective tolerance. Natural/semicanonical coordinates affect only the
preconditioner, preserving the physical active orbitals and finite-M MPS.
Integral blocks share their first-pair transforms and use available memory
after accounting for resident data; the two response densities share a JK call.
Full-ERI CASCI and orbital gradients also share a core JK cache, keyed by the
exact tagged core density. Changed densities recompute JK; active and response
requests bypass this cache.

See [the BAGEL/CaOH comparison](docs/bagel_zmcscf_caoh.md) for the numerical
checks and the distinction between inner and outer convergence.
The [follow-up comparison](docs/bagel_zmcscf_followup.md) records the new
step control, integral reuse and F six-state regression.

The Python environment is locked with `uv` (Python 3.12):

```bash
uv sync
make PYTHON=.venv/bin/python
uv run python -c "import pyscf; import block2; import pyblock2; import socutils"
```

The lock selects the official Block2 preview index because its 0.5.4rc16
CPython 3.12 wheel is newer than the compatible stable 0.5.3 release and
provides the complex `pyblock2.driver` APIs used by the DMRG solver. See
[`docs/x2c_dmrg_validation.md`](docs/x2c_dmrg_validation.md) for the tested
tensor conventions and numerical results.
To do a spinor (j-adapted) style calculation, build a spinor SCF and attach an
X2CAMF spin-orbit Hamiltonian with the `.x2camf()` shortcut (the spinor analogue
of PySCF's `scf.RHF(mol).x2c()`):
```python
from pyscf import gto
from socutils.scf import spinor_hf

mol = gto.M(atom=[["O", (0., 0., 0.)],
                  ["H", (0., -0.757, 0.587)],
                  ["H", (0.,  0.757, 0.587)]],
            basis='ccpvdz', verbose=4)

mf = spinor_hf.SCF(mol).x2camf()          # Gaunt + Breit on by default
e_spinor = mf.kernel()

# turn the Gaunt/Breit two-electron SOC corrections off:
e_dc = spinor_hf.SCF(mol).x2camf(with_gaunt=False, with_breit=False).kernel()
```
Note: the C/C++ libraries these features need are **bundled** with socutils
(under `socutils/lib`) -- no external packages to install. They are compiled
once with a single `make` at the repo root (needs a BLAS/LAPACK):
```
make            # builds libx2camf_c (X2CAMF SOC integrals),
                #        libzquatev   (Kramers-restricted spinor SCF), and
                #        libccsdt_clib (spinor CCSDT kernels) into socutils/lib
```
- **x2camf** (the SOC integrals) ships as a pure-C reimplementation exposed as
  `import x2camf`; an external upstream
  [warlocat/x2camf](https://github.com/warlocat/x2camf) pybind11 build is
  *optional* (only for A/B comparison, selected via `X2CAMF_BACKEND` /
  `SOCUTILS_X2CAMF`).
- **zquatev** (Kramers-restricted spinor SCF) is the bundled quaternion
  eigensolver -- the former `xubwa/zquatev` pip package is no longer needed.

See `docs/source/install.rst` for details (BLAS/LAPACK vendor selection,
environment overrides).
To do a GHF (spin-orbital) style calculation, use `ghf.GHF` with the same
`.x2camf()` shortcut; it runs in a spin-orbital rather than a spinor basis and
agrees with the spinor result to numerical precision:
```python
from pyscf import gto
from socutils.scf import ghf

mol = gto.M(atom=[["O", (0., 0., 0.)],
                  ["H", (0., -0.757, 0.587)],
                  ["H", (0.,  0.757, 0.587)]],
            basis='ccpvdz', verbose=4)

gmf = ghf.GHF(mol).x2camf()               # Gaunt + Breit on by default
e_ghf = gmf.kernel()
```
