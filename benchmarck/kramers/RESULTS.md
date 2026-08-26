# Kramers-restricted six-state halogen Super-CI results

Both methods use the same KR-X2C reference and Kramers-restricted
orbital optimization. `macroiterations` includes the initial and final
energy/gradient evaluations; `updates` counts applied orbital steps.
Only results from the newest available protocol version (7) are included.

| element | method | status | E (Eh) | ΔE vs exact (Eh) | max root Δ (Eh) | final |g| | macroiterations | updates | state-average TR residual | wall (s) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F | Exact-FCI KR-SCF + Super-CI | ok | -99.485998507156 | 0.000e+00 | 0.000e+00 | 6.586e-05 | 9 | 8 | 3.981e-12 | 1.1 |
| F | DMRG KR-SCF + Super-CI | ok | -99.485998507156 | 2.842e-14 | 8.527e-14 | 6.586e-05 | 9 | 8 | 5.773e-15 | 21.5 |
| Cl | Exact-FCI KR-SCF + Super-CI | ok | -460.887203718118 | 0.000e+00 | 0.000e+00 | 1.662e-05 | 8 | 7 | 5.019e-12 | 1.7 |
| Cl | DMRG KR-SCF + Super-CI | ok | -460.887203718118 | -4.547e-13 | 3.979e-13 | 1.662e-05 | 8 | 7 | 5.551e-15 | 19.2 |
| Br | Exact-FCI KR-SCF + Super-CI | ok | -2604.357953988483 | 0.000e+00 | 0.000e+00 | 2.394e-05 | 7 | 6 | 8.828e-11 | 4.0 |
| Br | DMRG KR-SCF + Super-CI | ok | -2604.357953988484 | -1.364e-12 | 1.364e-12 | 2.394e-05 | 7 | 6 | 3.775e-15 | 20.6 |
| I | Exact-FCI KR-SCF + Super-CI | ok | -7111.705653835465 | 0.000e+00 | 0.000e+00 | 1.549e-05 | 7 | 6 | 1.026e-11 | 9.9 |
| I | DMRG KR-SCF + Super-CI | ok | -7111.705653835461 | 3.638e-12 | 3.638e-12 | 1.549e-05 | 7 | 6 | 6.217e-15 | 25.6 |
| At | Exact-FCI KR-SCF + Super-CI | ok | -22855.620168719575 | 0.000e+00 | 0.000e+00 | 3.394e-05 | 8 | 7 | 1.359e-10 | 58.9 |
| At | DMRG KR-SCF + Super-CI | ok | -22855.620168719564 | 1.091e-11 | 1.091e-11 | 3.394e-05 | 8 | 7 | 5.656e-11 | 55.7 |

## DMRG Kramers adapter diagnostics

| element | root/manifold residual | root orthogonality | projected-H residual (Eh) | max pair splitting (Eh) | active-orbital closure |
| --- | ---: | ---: | ---: | ---: | ---: |
| F | 1.039e-09 | 1.475e-15 | 7.106e-14 | 1.440e-11 | 1.821e-15 |
| Cl | 1.005e-09 | 1.894e-15 | 5.686e-14 | 3.979e-12 | 2.278e-15 |
| Br | 7.978e-10 | 9.414e-16 | 8.415e-15 | 6.366e-12 | 2.061e-15 |
| I | 7.114e-10 | 1.555e-15 | 1.162e-14 | 1.819e-11 | 2.710e-15 |
| At | 1.054e-09 | 1.471e-15 | 1.198e-14 | 2.183e-10 | 2.779e-15 |

## Final six-state energies

These are total root energies at each method's final orbitals.

| element | method | state 1 | state 2 | state 3 | state 4 | state 5 | state 6 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| F | Exact-FCI KR-SCF + Super-CI | -99.486865749091 | -99.486865749088 | -99.486865749080 | -99.486865749077 | -99.484264023308 | -99.484264023294 |
| F | DMRG KR-SCF + Super-CI | -99.486865749091 | -99.486865749088 | -99.486865749080 | -99.486865749077 | -99.484264023308 | -99.484264023294 |
| Cl | Exact-FCI KR-SCF + Super-CI | -460.888772352677 | -460.888772352667 | -460.888772352664 | -460.888772352659 | -460.884066449022 | -460.884066449018 |
| Cl | DMRG KR-SCF + Super-CI | -460.888772352677 | -460.888772352668 | -460.888772352664 | -460.888772352660 | -460.884066449023 | -460.884066449019 |
| Br | Exact-FCI KR-SCF + Super-CI | -2604.363617372449 | -2604.363617372422 | -2604.363617372406 | -2604.363617372352 | -2604.346627220638 | -2604.346627220631 |
| Br | DMRG KR-SCF + Super-CI | -2604.363617372450 | -2604.363617372424 | -2604.363617372408 | -2604.363617372353 | -2604.346627220639 | -2604.346627220633 |
| I | Exact-FCI KR-SCF + Super-CI | -7111.716520647948 | -7111.716520647804 | -7111.716520647795 | -7111.716520647603 | -7111.683920210829 | -7111.683920210811 |
| I | DMRG KR-SCF + Super-CI | -7111.716520647944 | -7111.716520647800 | -7111.716520647791 | -7111.716520647599 | -7111.683920210826 | -7111.683920210808 |
| At | Exact-FCI KR-SCF + Super-CI | -22855.649222105709 | -22855.649222105643 | -22855.649222105585 | -22855.649222105443 | -22855.562061947639 | -22855.562061947421 |
| At | DMRG KR-SCF + Super-CI | -22855.649222105698 | -22855.649222105632 | -22855.649222105574 | -22855.649222105432 | -22855.562061947629 | -22855.562061947410 |
