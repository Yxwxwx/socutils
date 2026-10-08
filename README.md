# socutils

Relativistic complex-spinor electronic-structure methods built on PySCF:
X2CAMF Hartree–Fock, second-order CASSCF/DMRG-SCF, FIC-NEVPT2 and
one-step MS-FIC-NEVPT2.

## Installation

Use Linux, Python 3.12, a C/C++ compiler, CMake, Make and
[uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone https://github.com/Yxwxwx/socutils.git
cd socutils
uv sync --frozen
make PYTHON=.venv/bin/python NPROC=8
source .venv/bin/activate
python -c "import socutils, pyscf, block2, pyblock2, pytblis; from socutils.scf import spinor_hf"
```

The locked environment includes PySCF 2.14.0, Block2 0.5.4rc16 and
pytblis 0.0.16. `make` builds the bundled X2CAMF, quaternion eigensolver
(`zquatev`) and integral kernels; separate X2CAMF/zquatev installations are
not required. Build and run with the same Python environment. By default,
the build uses the BLAS/LAPACK shipped with PySCF.

To explicitly select the bundled X2CAMF implementation, including when an
external `x2camf` package is installed:

```bash
export SOCUTILS_X2CAMF=bundled
export X2CAMF_BACKEND=c
```

Before running Block2, allow an unlimited process stack:

```bash
ulimit -s unlimited
python calculation.py
```

## HF: CD and Kramers restriction

`spinor_hf.SCF` uses unrestricted complex spinors; `spinor_hf.KRHF` enforces
Kramers restriction. Cholesky decomposition (CD) is independent of that choice.
For an existing PySCF molecule `mol`:

| HF configuration | Construction |
| --- | --- |
| Full ERIs, no KR | `spinor_hf.SCF(mol).x2camf()` |
| Full ERIs, KR | `spinor_hf.KRHF(mol).x2camf()` |
| CD, no KR | `spinor_hf.SCF(mol).x2camf().cholesky(tau=1e-8)` |
| CD, KR | `spinor_hf.KRHF(mol).x2camf().cholesky(tau=1e-8)` |

`.x2camf()` includes Gaunt and Breit corrections by default. Use
`.x2camf(with_gaunt=False, with_breit=False)` to disable those corrections,
not spin–orbit coupling itself. Omitting `.cholesky()` selects full ERIs.
An attached CD object is also used by second-order MCSCF.

KR requires time-reversal-compatible orbitals and complete Kramers pairs
within each core/active/virtual partition. For KR-DMRG-SCF, additionally
enable `solver.kramers_restricted()`; odd-electron state averages must include
complete Kramers manifolds with equal weights within each manifold.

## Second-order DMRG-SCF → FIC/MS-NEVPT2

The example below computes neutral F from closed-shell F⁻ starting orbitals,
with CAS(7 electrons, 16 spinors) and six equally weighted reference states.
`ncas` counts individual spinors, **not spatial orbitals**; `nelecas` is the
total active electron count. Active-orbital selection must be checked for
each new system.

Save as `calculation.py`. Set `USE_CD` and `USE_KR` independently to select
any of the four HF/MCSCF configurations above.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from pyscf import gto, lib

from socutils.dmrg import DMRGCI
from socutils.mcscf import zmcscf
from socutils.mrpt import WickX2CFICNEVPT2, prepare_msfic, solve_msfic
from socutils.scf import spinor_hf

USE_CD = False
USE_KR = False
THREADS = 16
lib.num_threads(THREADS)

mol = gto.M(
    atom="F 0 0 0", basis="dyallv3z", charge=-1, spin=0,
    verbose=4, max_memory=300000,  # MB; does not reserve physical RAM
)
hf_class = spinor_hf.KRHF if USE_KR else spinor_hf.SCF
mf = hf_class(mol).x2camf()
if USE_CD:
    mf = mf.cholesky(tau=1e-8)
mf.conv_tol = 1e-12
mf.max_cycle = 200
mf.kernel()
if not mf.converged:
    raise RuntimeError("F- HF did not converge")

# Keep the F- X2CAMF one-electron operator when changing the electron count.
mf.get_hcore()
initial_mo = np.array(mf.mo_coeff, copy=True)
mol.charge, mol.spin = 0, 1
ncas, nelec, nroots = 16, 7, 6
weights = np.ones(nroots) / nroots

with TemporaryDirectory(prefix="f_dmrg_", dir=lib.param.TMPDIR) as scratch:
    solver = DMRGCI(mol).init(
        ncas=ncas, nelecas=nelec, nroots=nroots,
        max_bond_dimension=1000, tol=1e-8,
        schedule_thrd_max=1e-16, orbital_ordering="original",
        final_one_site=USE_KR,
        n_threads=THREADS, stack_memory=8192,  # Block2 stack, MB
        scratch=scratch, checkpoint_dir="dmrg_checkpoint",
    )
    if USE_KR:
        solver.kramers_restricted()

    mc = zmcscf.CASSCF(mf, ncas=ncas, nelecas=nelec)
    mc.fcisolver = solver
    mc.state_average_(weights)
    solver = mc.fcisolver  # state_average_ installs a solver wrapper
    mc.callback = solver.restart_scheduler_()
    mc.canonicalization = mc.canonicalize_ = mc.natorb = False
    mc.superci_bfgs = False
    mc.max_cycle_macro = 100
    mc.conv_tol = 1e-9
    mc.conv_tol_grad = 1e-4

    try:
        mc.second_order(mo_coeff=initial_mo)
        if not mc.converged or not solver.converged:
            raise RuntimeError("DMRG-SCF did not converge")
        print("MCSCF energies:", mc.e_states)

        for root in range(nroots):
            pt = WickX2CFICNEVPT2(mc)
            e2 = pt.kernel(root=root, contraction_backend="pytblis")
            print(f"FIC root {root}: E0={pt.reference_energy:.12f} "
                  f"E2={e2:.12f} Etot={pt.e_tot:.12f}")
            del pt

        # Same six-state SA-Fock for both dynamic-correlation manifolds.
        for model_roots in ((0, 1, 2, 3), (4, 5)):
            prepared = prepare_msfic(
                mc, sa_roots=range(nroots), sa_weights=weights,
                model_roots=model_roots, contraction_backend="pytblis",
                transition_rdm_fallback_dir=str(Path("rdm_fallback").resolve()),
            )
            result = solve_msfic(prepared, ansatz="ms_mr", shift=0.0)
            print(f"MS-FIC manifold {model_roots}:", result.energies)
            del prepared, result
    finally:
        solver.close()  # PT needs the live reference MPS
```

`mc.second_order()` explicitly selects the fixed-1/2-RDM orbital Hessian
and scaled augmented-Hessian optimizer; `mc.kernel()` is not this entry
point. Full-ERI and CD integral routes, with or without KR, are supported.
Keep `natorb=False`, `canonicalize_=False` and orbital BFGS disabled.
`mc.canonicalization=False` also skips final semicanonicalization; PT handles
the required inactive/virtual semicanonicalization without rotating active
orbitals. For exact-CI CASSCF, leave the default `mc.fcisolver` in place
instead of assigning `DMRGCI`.

### FIC-NEVPT2

`WickX2CFICNEVPT2.kernel(root=...)` evaluates one state's correlation energy
from its raw complex-spinor 1–4 RDMs. It returns `E2`; `pt.reference_energy`
and `pt.e_tot` give `E0` and `E0 + E2`. Leave `pt.canonicalized=False`
(the default) unless the orbitals **and** orbital energies have already
been prepared consistently for the PT Hamiltonian.

### MS-FIC-NEVPT2

`prepare_msfic` constructs a common SA-Fock/Dyall partition from `sa_roots`
and `sa_weights`. `model_roots` independently selects the states mixed by
dynamic correlation and must be a subset of `sa_roots`. `solve_msfic`
diagonalizes the effective Hamiltonian; `result.energies` contains total
energies, not just corrections.

The F example uses a common six-root SA ensemble and separate four-root
and two-root manifolds. To mix all six jointly, prepare once with
`model_roots=tuple(range(6))`. These groupings are system-specific, not
automatic assignments. `ansatz="ms_mr"` selects MS-MR; `"ss_sr"` selects
SS-SR. The example explicitly uses zero level shift; the API default is
`shift=0.2` Hartree.

### Integrals, RDMs and scratch

FIC/MS-NEVPT2 use full Coulomb integrals even after CD-HF/CD-MCSCF. The PT
code reuses the orbitals and reference MPS, detaches CD on a private view
and transforms only the required exact integral blocks. It neither expands
a full all-MO four-index tensor by default nor reruns MCSCF. For a CD
reference, FIC's `reference_energy` is its full-Hamiltonian RDM expectation,
which can differ from the CD-MCSCF energy; use `pt.e_tot` for the PT total.
KR is a reference-stage restriction, not a NEVPT2-level constraint.

Both methods require full rank-4 densities. MS uses raw transition 1–4
RDMs, without averaging them to impose degeneracy. Transition pairs are
staged on disk one at a time, contracted through read-only memory maps
and deleted afterwards. Set `lib.param.TMPDIR` to job-local NVMe **before**
creating the solver. `transition_rdm_fallback_dir` provides a larger
filesystem when that staging directory is full; it does not relocate
Block2's own scratch. Use `transition_rdm_dir` only when retaining all raw
pairs is intentional.

Disk streaming does not remove the RAM needed to generate one dense
4-RDM/4-TRDM. At 16 active spinors, one complex128 rank-1–4 density set is
about 64.25 GiB, before contraction workspaces and MPS storage. PySCF
`max_memory` and Block2 `stack_memory` are in MB and are not hard limits on
total process memory. Both PT implementations currently require no frozen
orbitals (`frozen=0`). See the [MS-FIC API notes](mrpt/README_msficnevpt2.md)
for storage options and method details.
