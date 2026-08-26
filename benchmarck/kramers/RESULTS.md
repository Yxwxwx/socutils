# Kramers-restricted six-state halogen Super-CI results

Both methods use the same KR-X2C reference and Kramers-restricted
orbital optimization. `macroiterations` includes the initial and final
energy/gradient evaluations; `updates` counts applied orbital steps.
Only results from the newest available protocol version (5) are included.

| element | method | status | E (Eh) | ΔE vs exact (Eh) | max root Δ (Eh) | final |g| | macroiterations | updates | state-average TR residual | wall (s) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F | Exact-FCI KR-SCF + Super-CI | ok | -99.485998507156 | 0.000e+00 | 0.000e+00 | 6.586e-05 | 9 | 8 | 3.981e-12 | 2.0 |
| F | DMRG KR-SCF + Super-CI | ok | -99.485998507156 | 5.684e-14 | 4.420e-12 | 6.586e-05 | 9 | 8 | 2.098e-14 | 44.7 |
| Cl | Exact-FCI KR-SCF + Super-CI | ok | -460.887203718118 | 0.000e+00 | 0.000e+00 | 1.662e-05 | 8 | 7 | 5.019e-12 | 2.9 |
| Cl | DMRG KR-SCF + Super-CI | ok | -460.887203718118 | -1.137e-13 | 4.263e-12 | 1.662e-05 | 8 | 7 | 1.734e-14 | 37.0 |
| Br | Exact-FCI KR-SCF + Super-CI | ok | -2604.357953988483 | 0.000e+00 | 0.000e+00 | 2.394e-05 | 7 | 6 | 8.828e-11 | 5.5 |
| Br | DMRG KR-SCF + Super-CI | ok | -2604.357953988483 | 0.000e+00 | 4.957e-11 | 2.394e-05 | 7 | 6 | 8.691e-11 | 36.2 |
| I | Exact-FCI KR-SCF + Super-CI | ok | -7111.705653835465 | 0.000e+00 | 0.000e+00 | 1.549e-05 | 7 | 6 | 1.026e-11 | 11.3 |
| I | DMRG KR-SCF + Super-CI | ok | -7111.705653835459 | 5.457e-12 | 5.093e-11 | 1.549e-05 | 7 | 6 | 2.203e-13 | 40.4 |
| At | Exact-FCI KR-SCF + Super-CI | ok | -22855.620168719575 | 0.000e+00 | 0.000e+00 | 3.394e-05 | 8 | 7 | 1.359e-10 | 60.2 |
| At | DMRG KR-SCF + Super-CI | ok | -22855.620168719553 | 2.183e-11 | 2.910e-11 | 3.394e-05 | 8 | 7 | 1.340e-10 | 75.9 |

## DMRG Kramers adapter diagnostics

| element | root/manifold residual | root orthogonality | projected-H residual (Eh) | max pair splitting (Eh) | active-orbital closure |
| --- | ---: | ---: | ---: | ---: | ---: |
| F | 1.777e-09 | 1.072e-13 | 1.042e-11 | 1.333e-11 | 2.382e-15 |
| Cl | 1.142e-09 | 5.987e-14 | 5.996e-12 | 3.809e-12 | 2.061e-15 |
| Br | 2.212e-09 | 5.877e-13 | 5.877e-11 | 5.912e-12 | 2.046e-15 |
| I | 7.802e-10 | 5.747e-13 | 5.745e-11 | 9.095e-12 | 2.070e-15 |
| At | 1.055e-09 | 2.999e-13 | 2.455e-11 | 2.183e-10 | 2.310e-15 |

## Final six-state energies

These are total root energies at each method's final orbitals.

| element | method | state 1 | state 2 | state 3 | state 4 | state 5 | state 6 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| F | Exact-FCI KR-SCF + Super-CI | -99.486865749091 | -99.486865749088 | -99.486865749080 | -99.486865749077 | -99.484264023308 | -99.484264023294 |
| F | DMRG KR-SCF + Super-CI | -99.486865749086 | -99.486865749085 | -99.486865749084 | -99.486865749080 | -99.484264023308 | -99.484264023294 |
| Cl | Exact-FCI KR-SCF + Super-CI | -460.888772352677 | -460.888772352667 | -460.888772352664 | -460.888772352659 | -460.884066449022 | -460.884066449018 |
| Cl | DMRG KR-SCF + Super-CI | -460.888772352673 | -460.888772352667 | -460.888772352664 | -460.888772352664 | -460.884066449022 | -460.884066449018 |
| Br | Exact-FCI KR-SCF + Super-CI | -2604.363617372449 | -2604.363617372422 | -2604.363617372406 | -2604.363617372352 | -2604.346627220638 | -2604.346627220631 |
| Br | DMRG KR-SCF + Super-CI | -2604.363617372415 | -2604.363617372407 | -2604.363617372406 | -2604.363617372402 | -2604.346627220637 | -2604.346627220631 |
| I | Exact-FCI KR-SCF + Super-CI | -7111.716520647948 | -7111.716520647804 | -7111.716520647795 | -7111.716520647603 | -7111.683920210829 | -7111.683920210811 |
| I | DMRG KR-SCF + Super-CI | -7111.716520647897 | -7111.716520647815 | -7111.716520647810 | -7111.716520647609 | -7111.683920210820 | -7111.683920210811 |
| At | Exact-FCI KR-SCF + Super-CI | -22855.649222105709 | -22855.649222105643 | -22855.649222105585 | -22855.649222105443 | -22855.562061947639 | -22855.562061947421 |
| At | DMRG KR-SCF + Super-CI | -22855.649222105680 | -22855.649222105636 | -22855.649222105567 | -22855.649222105425 | -22855.562061947621 | -22855.562061947403 |
