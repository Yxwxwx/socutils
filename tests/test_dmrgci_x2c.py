from pathlib import Path

import numpy as np
import pytest
from pyscf import gto

from socutils.dmrg.dmrgci import DMRGCI, energy_from_rdms
from socutils.fci import zfci
from socutils.mcscf import zcasci
from socutils.scf import spinor_hf


class RecordingExactFCI(zfci.FCISolver):
    def kernel(self, h1e, eri, norb, nelec, *args, ecore=0.0, **kwargs):
        self.last_h1e = np.array(h1e, copy=True)
        self.last_eri = np.array(eri, copy=True).reshape((norb,) * 4)
        self.last_ecore = ecore
        return super().kernel(
            h1e, eri, norb, nelec, *args, ecore=ecore, **kwargs
        )


class RecordingDMRGCI(DMRGCI):
    def kernel(self, h1e, eri, norb, nelec, *args, ecore=0.0, **kwargs):
        self.last_h1e = np.array(h1e, copy=True)
        self.last_eri = np.array(eri, copy=True).reshape((norb,) * 4)
        self.last_ecore = ecore
        return super().kernel(
            h1e, eri, norb, nelec, *args, ecore=ecore, **kwargs
        )


@pytest.mark.integration
def test_x2camf_casci_state_average_contract(tmp_path):
    mol = gto.M(
        atom="H 0 0 0; F 0.35 0.27 0.8035",
        basis="sto-3g",
        spin=0,
        charge=0,
        verbose=0,
        max_memory=1000,
    )
    mf = spinor_hf.SCF(mol).x2camf(with_gaunt=False, with_breit=False)
    mf.init_guess = "1e"
    mf.conv_tol = 1e-11
    mf.max_cycle = 100
    mf.kernel()
    assert mf.converged
    initial_mos = mf.mo_coeff.copy()

    # The first excited level is an exactly degenerate pair.  Include the
    # complete pair with equal weights so the molecular state average is
    # invariant to the arbitrary basis chosen inside that subspace.
    weights = np.array([0.4, 0.3, 0.3])
    exact_mc = zcasci.CASCI(mf, ncas=4, nelecas=2)
    exact_mc.fcisolver = RecordingExactFCI(mol)
    exact_mc.state_average_(weights)
    exact_energy, exact_active_energy, exact_ci = exact_mc.kernel(verbose=0)

    dmrg_mc = zcasci.CASCI(mf, ncas=4, nelecas=2)
    dmrg_mc.fcisolver = RecordingDMRGCI(mol).init(
        ncas=4,
        nelecas=2,
        nroots=3,
        bond_dims=[32] * 8,
        noises=[0.0] * 8,
        thrds=[1e-14] * 8,
        n_sweeps=8,
        tol=1e-12,
        scratch=tmp_path,
        n_threads=1,
        stack_memory=256,
        dav_max_iter=1000,
        random_seed=2468,
        npdm_site_type=2,
    )
    dmrg_mc.state_average_(weights)
    dmrg_energy, dmrg_active_energy, dmrg_ci = dmrg_mc.kernel(verbose=0)

    exact_solver = exact_mc.fcisolver
    dmrg_solver = dmrg_mc.fcisolver
    assert np.max(abs(exact_solver.last_h1e - dmrg_solver.last_h1e)) == 0.0
    assert np.max(abs(exact_solver.last_eri - dmrg_solver.last_eri)) == 0.0
    assert abs(exact_solver.last_ecore - dmrg_solver.last_ecore) == 0.0
    assert np.max(abs(dmrg_solver.last_eri.imag)) > 1e-4
    assert np.max(abs(mf.mo_coeff - initial_mos)) == 0.0

    assert abs(dmrg_energy - exact_energy) <= 1e-9
    assert abs(dmrg_active_energy - exact_active_energy) <= 1e-9
    assert np.max(abs(dmrg_solver.e_states - exact_solver.e_states)) <= 1e-9
    assert isinstance(exact_ci, list) and isinstance(dmrg_ci, list)
    assert len(exact_ci) == len(dmrg_ci) == 3

    exact_root_dm1, exact_root_dm2 = exact_solver.states_make_rdm12(
        exact_ci, 4, 2
    )
    dmrg_root_dm1, dmrg_root_dm2 = dmrg_solver.states_make_rdm12(
        dmrg_ci, 4, 2
    )
    assert np.max(abs(dmrg_root_dm1[0] - exact_root_dm1[0])) <= 1e-8
    assert np.max(abs(dmrg_root_dm2[0] - exact_root_dm2[0])) <= 1e-8
    assert np.max(
        abs(sum(dmrg_root_dm1[1:]) - sum(exact_root_dm1[1:]))
    ) <= 1e-8
    assert np.max(
        abs(sum(dmrg_root_dm2[1:]) - sum(exact_root_dm2[1:]))
    ) <= 1e-8

    exact_dm1, exact_dm2 = exact_solver.make_rdm12(exact_ci, 4, 2)
    dmrg_dm1, dmrg_dm2 = dmrg_solver.make_rdm12(dmrg_ci, 4, 2)
    assert np.max(abs(dmrg_dm1 - exact_dm1)) <= 1e-8
    assert np.max(abs(dmrg_dm2 - exact_dm2)) <= 1e-8
    print(
        "x2camf-casci",
        "Eref=%.14f" % exact_energy,
        "Edmrg=%.14f" % dmrg_energy,
        "dE=%.3e" % abs(dmrg_energy - exact_energy),
        "dm1=%.3e" % np.max(abs(dmrg_dm1 - exact_dm1)),
        "dm2=%.3e" % np.max(abs(dmrg_dm2 - exact_dm2)),
    )
    assert (
        abs(
            energy_from_rdms(
                dmrg_solver.last_h1e,
                dmrg_solver.last_eri,
                dmrg_dm1,
                dmrg_dm2,
                dmrg_solver.last_ecore,
            )
            - dmrg_energy
        )
        <= 1e-9
    )
    assert exact_mc.converged and dmrg_mc.converged and dmrg_solver.converged

    run_scratch = Path(dmrg_solver._scratch)
    dmrg_solver.close()
    assert not run_scratch.exists()
