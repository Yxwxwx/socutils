#!/usr/bin/env python
'''
Spinor (2-component) CIS on a Hartree-Fock reference, MO-driven.

CIS is just TDA with an HF reference (no f_xc, exact-exchange fraction = 1), so
`mf.TDA()` on a spinor SCF *is* spinor CIS.  Here we run it MO-driven:

  td.mo_driven = True

In that mode the Coulomb + exact-exchange block of the CIS matrix is built once
from the density-fitted (Cholesky) ERIs, half-transformed into the active
occ/vir space, and every Davidson matvec is then a small dense mat-vec instead
of a full AO Fock build.  That is the cheap path when the occupied (hole) space
is small but many roots / iterations are needed -- e.g. core excitations.

Requirements for mo_driven:
  * density fitting on the reference (.density_fit()) -- it consumes mf.with_df.
  * the orbital (Fock) term uses the full occ/vir Fock blocks, so it is correct
    for non-canonical references too (localized / rotated orbitals), not only
    canonical ones.

Swap the geometry/basis below for your molecule.  Add .x2camf() to the reference
(see example 00) to switch on spin-orbit coupling.
'''
from pyscf import gto
from socutils.scf import spinor_hf

au2ev = 27.211386245988

# ---- fill in your molecule here -------------------------------------------
mol = gto.M(
    atom='''
        H  0.0  0.0  0.0
        F  0.0  0.0  0.92
    ''',
    basis='ccpvdz',
    charge=0,
    spin=0,             # spinor/GHF: `spin` is ignored, the j-adapted basis is used
    verbose=4,
)
# ---------------------------------------------------------------------------

# spinor Hartree-Fock reference (add .x2camf() for spin-orbit coupling).
# density_fit() is required by the MO-driven path.
mf = spinor_hf.SpinorSCF(mol).density_fit()
mf.kernel()
print('E(spinor HF) = %.10f  converged = %s' % (mf.e_tot, mf.converged))

# spinor CIS, MO-driven
td = mf.TDA()                 # CIS == TDA on an HF reference
td.mo_driven = True
td.nstates = 10
td.kernel()

# transition properties (spinor dipole, |mu|^2 = mu . mu*)
osc = td.oscillator_strength()

print('\n  %-5s %14s %14s' % ('root', 'energy/eV', 'osc.str.'))
for k, (e, f) in enumerate(zip(td.e, osc)):
    print('  %-5d %14.4f %14.4e' % (k, e * au2ev, f))
