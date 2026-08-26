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
(the concrete hash is recorded by the following milestone because a commit
cannot contain its own content-derived hash).

## Known limitations at this milestone

The general complex-spinor solver conserves particle number only. `ci0` is
explicitly ignored because an MPS from the preceding CASSCF macroiteration is
not valid after an orbital rotation without a validated MPS orbital transform.
Kramers-restricted adaptation and Super-CIPT are later validation milestones.
