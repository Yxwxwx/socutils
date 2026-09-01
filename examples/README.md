# socutils examples

Runnable examples matching the [documentation](https://xubwa.github.io/socutils/).
They use the canonical `.x2camf()` / `.x2cmp()` driver API.

| File | Topic |
| --- | --- |
| `00-spinor_x2camf.py` | spinor HF with X2CAMF (`spinor_hf.SCF(mol).x2camf()`) |
| `01-spinor_x2cmp.py` | the `x2cmp` flavor and toggling Gaunt/Breit |
| `02-kramers_restricted.py` | Kramers-restricted SCF (`spinor_hf.KRHF`, needs zquatev) |
| `03-symmetry_atom.py` | symmetry-adapted SCF, atom (`symmetry='sph'`) |
| `04-symmetry_linear.py` | symmetry-adapted SCF, linear molecule (must be on z) |
| `05-density_fitting.py` | density fitting via `.density_fit()` |
| `06-ghf_x2camf.py` | GHF (spin-orbital) driver (`ghf.GHF(mol).x2camf()`) |
| `07-somf_helper.py` | constructing the X2C SOC helper directly |
| `08-casci.py` | CASCI on a spinor reference |
| `09-four_component.py` | four-component Dirac-Hartree-Fock |
| `10-casscf.py` | Cholesky CASSCF orbital optimization (needs zquatev) |
| `11-kramers_dmrg_scf.py` | Kramers-pair X2C-DMRG-SCF (needs Block2 + zquatev) |
| `14-supercipt.py` | Block2 X2C-DMRG-SCF with the separate Super-CIPT optimizer |
| `15-boys-localization.py` | complex and Kramers-preserving Boys localization |
| `16-nd_h2o8_supercipt_diis.py` | Nd3+(H2O)8 CAS(3,14), 52-root Kramers Super-CIPT/DIIS input |
| `17-cl_cas16_diis.py` | Cl CAS(7,16), six-root DMRG Super-CIPT/Super-CI DIIS comparison, with optional Kramers restriction |
| `18-x2c_dmrg_sc_nevpt2.py` | BH CAS(4,6) no-Kramers dense X2C-DMRG-SCF to strict-SI Wick SC-NEVPT2 (needs Block2) |
| `19-x2c_dmrg_qd_sc_nevpt2.py` | neutral Cl CAS(5,12), six-root SA-X2C-DMRG-SCF to Bloch/canonical Van Vleck QD-SC-NEVPT2 (needs Block2) |

The `fci/` subfolder has examples for the spinor CI module (`socutils.fci`):

| File | Topic |
| --- | --- |
| `fci/00-spinor_fci_exact_diag.py` | exact full CI with `zfci.FCISolver` |
| `fci/01-selected_ci_determinant_list.py` | selected CI over a determinant list |
| `fci/02-rasci.py` | RASCI via `gen_ras_occslst` + `SelectedCI` |
| `fci/03-transition_dipole.py` | transition dipoles / oscillator strengths (`fci.addons`) |
| `fci/04-spin_composition.py` | spin / angular-momentum analysis of states |

Most examples need the optional `x2camf` package for the spin-orbit integrals;
the Kramers-restricted SCF examples additionally need `zquatev`, and the
DMRG-SCF, SC-NEVPT2, and QD-SC-NEVPT2 examples need Block2. See the
[installation guide](https://xubwa.github.io/socutils/install.html).
