from pathlib import Path

import numpy as np
import pytest
import scipy.linalg
from pyscf import gto, lib

from socutils.cd.cd import CD
from socutils.dmrg.dmrgci import (
    DMRGCI,
    block2_integrals,
    energy_from_rdms,
)
from socutils.fci import zfci
from socutils.mcscf import zmc_ao2mo, zmc_superci, zmcscf
from socutils.scf import spinor_hf


@pytest.fixture(scope='module')
def tilted_hf():
    mol = gto.M(
        atom='H 0 0 0; F 0.35 0.27 0.8035',
        basis='sto-3g',
        spin=0,
        charge=0,
        verbose=0,
        max_memory=1000,
    )
    mf_full = spinor_hf.SCF(mol).x2camf(
        with_gaunt=False, with_breit=False)
    mf_full.init_guess = '1e'
    mf_full.conv_tol = 1e-11
    mf_full.max_cycle = 100
    mf_full.kernel()
    assert mf_full.converged
    mo = mf_full.mo_coeff.copy()
    mf_cd = mf_full.cholesky(tau=1e-10)
    mf_cd.mo_coeff = mo.copy()
    return mol, mf_full, mf_cd, mo


def _casscf(mf, mo):
    mc = zmcscf.CASSCF(mf, ncas=4, nelecas=2)
    mc.mo_coeff = mo.copy()
    mc.natorb = False
    mc.canonicalize_ = False
    mc.verbose = 0
    return mc


@pytest.mark.parametrize(
    'factory_name',
    ['chunked_cholesky', 'chunked_cholesky_threeloop',
     'chunked_cholesky_twoloop', 'chunked_cholesky0'],
)
def test_cholesky_variants_return_all_vectors(factory_name):
    mol = gto.M(
        atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
    factory = getattr(zmc_ao2mo, factory_name)
    vectors = factory(mol, max_error=1e-10, verbose=False)
    vectors = vectors.reshape((-1, mol.nao_nr(), mol.nao_nr()))
    reconstructed = np.einsum('Pmn,Prs->mnrs', vectors, vectors)
    assert np.max(abs(reconstructed - mol.intor('int2e_sph'))) <= 1e-9


def test_cholesky_integrals_jk_and_full_gradient(tilted_hf):
    mol, mf_full, mf_cd, mo = tilted_hf
    mc_full = _casscf(mf_full, mo)
    mc_cd = _casscf(mf_cd, mo)
    eris_full = zmc_ao2mo._ERIS(mc_full, mo.copy(), level=2)
    eris_cd = zmc_ao2mo._CDERIS(mc_cd, mo.copy(), level=2)

    aaaa_error = np.max(abs(eris_cd.aaaa - eris_full.aaaa))
    paaa_error = np.max(abs(eris_cd.paaa - eris_full.paaa))
    wrong_aaaa = np.einsum(
        'Ptu,Pvw->tuvw', eris_cd.cd_aa.conj(), eris_cd.cd_aa)
    wrong_conjugation_error = np.max(abs(wrong_aaaa - eris_full.aaaa))

    nmo = mo.shape[1]
    ncore = mc_cd.ncore
    nocc = ncore + mc_cd.ncas
    dm_core_ao = mo[:, :ncore].dot(mo[:, :ncore].T.conj())
    core_occ = np.zeros(nmo)
    core_occ[:ncore] = 1
    vj_full, vk_full = eris_full.get_jk(
        dm_core_ao, mo_coeff=mo, mo_occ=core_occ)
    vj_cd, vk_cd = eris_cd.get_jk(
        dm_core_ao, mo_coeff=mo, mo_occ=core_occ)

    mci = zmcscf._fake_h_for_fast_casci(mc_cd, mo.copy(), eris_cd)
    _, _, ci = mci.kernel(mo.copy(), verbose=0)
    dm1, dm2 = mci.fcisolver.make_rdm12(ci, mc_cd.ncas, mc_cd.nelecas)
    vja_full, vka_full = eris_full.get_jk_active_mo(dm1)
    vja_cd, vka_cd = eris_cd.get_jk_active_mo(dm1)

    g2_full = zmc_superci._contract_dm2_gradient(eris_full, dm2)
    g2_cd = zmc_superci._contract_dm2_gradient(eris_cd, dm2)
    gradient_full = zmc_superci.gen_g_hop(
        mc_full, mo.copy(), dm1, dm2, eris_full)[0]
    gradient_cd = zmc_superci.gen_g_hop(
        mc_cd, mo.copy(), dm1, dm2, eris_cd)[0]

    cderi = np.vstack(list(mf_cd.with_df.loop()))
    chol_ao = lib.unpack_tril(cderi)
    eri_ao = mol.intor('int2e_sph')
    reconstructed_ao = np.einsum('Pmn,Prs->mnrs', chol_ao, chol_ao)
    ao_error = np.max(abs(reconstructed_ao - eri_ao))

    assert isinstance(mf_cd.with_df, CD)
    assert mf_cd.with_df.tau == 1e-10
    assert np.max(abs(eris_full.aaaa.imag)) > 1e-4
    assert ao_error <= 1e-9
    assert aaaa_error <= 1e-9
    assert paaa_error <= 1e-9
    assert wrong_conjugation_error >= 1e-3
    assert np.max(abs(vj_cd - vj_full)) <= 1e-9
    assert np.max(abs(vk_cd - vk_full)) <= 1e-9
    assert np.max(abs(vja_cd - vja_full)) <= 1e-9
    assert np.max(abs(vka_cd - vka_full)) <= 1e-9
    assert np.max(abs(g2_cd - g2_full)) <= 1e-9
    assert np.max(abs(gradient_cd - gradient_full)) <= 1e-7
    assert np.linalg.norm(gradient_cd - gradient_full) <= 1e-6
    print(
        'cholesky-fixed-orbital',
        'naux=%d' % chol_ao.shape[0],
        'ao=%.3e' % ao_error,
        'aaaa=%.3e' % aaaa_error,
        'paaa=%.3e' % paaa_error,
        'g2=%.3e' % np.max(abs(g2_cd - g2_full)),
        'gradient_max=%.3e' % np.max(abs(gradient_cd - gradient_full)),
        'gradient_norm=%.3e' % np.linalg.norm(gradient_cd - gradient_full),
    )


def test_superci_davidson_complex_residual():
    rng = np.random.default_rng(719)
    z = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    hessian = (z + z.T.conj()) * 0.5 + np.diag([1.0, 1.7, 2.4, 3.1])
    b = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    overlap = b.T.conj().dot(b) * 0.04 + np.eye(4)
    gradient = (rng.normal(size=4) + 1j * rng.normal(size=4)) * 0.03

    augmented = np.zeros((5, 5), dtype=complex)
    metric = np.zeros((5, 5), dtype=complex)
    augmented[0, 1:] = gradient.conj()
    augmented[1:, 0] = gradient
    augmented[1:, 1:] = hessian
    metric[0, 0] = 1
    metric[1:, 1:] = overlap
    eigvals, eigvecs = scipy.linalg.eigh(augmented, metric)
    root = next(i for i in range(5) if 0.1 < abs(eigvecs[0, i]) <= 1.1)
    expected_step = eigvecs[1:, root] / eigvecs[0, root]

    step, eigenvalue, info = zmc_superci.davidson(
        lambda x: hessian.dot(x),
        gradient,
        np.diag(hessian).real,
        sop=lambda x: overlap.dot(x),
        tol=1e-11,
        mmax=4,
    )
    residual = augmented.dot(np.r_[1.0, step]) - \
        eigenvalue * metric.dot(np.r_[1.0, step])
    assert info['converged']
    assert info['residual_norm'] <= 1e-11
    assert np.linalg.norm(residual) <= 1e-11
    assert abs(eigenvalue - eigvals[root]) <= 1e-11
    assert np.max(abs(step - expected_step)) <= 1e-10


@pytest.mark.integration
def test_fixed_orbital_fci_dmrg_gradient(tilted_hf, tmp_path):
    mol, _, mf_cd, mo = tilted_hf
    mc = _casscf(mf_cd, mo)
    eris = zmc_ao2mo._CDERIS(mc, mo.copy(), level=2)
    mci = zmcscf._fake_h_for_fast_casci(mc, mo.copy(), eris)
    h1eff, ecore = mci.get_h1eff(mo)
    eri_active = eris.aaaa.copy()

    h1_block2, eri_block2 = block2_integrals(h1eff, eri_active, mc.ncas)
    exact = zfci.FCISolver(mol)
    exact_energy, exact_ci = exact.kernel(
        h1eff, eri_active, mc.ncas, mc.nelecas,
        ecore=ecore, verbose=0)
    exact_dm1, exact_dm2 = exact.make_rdm12(
        exact_ci, mc.ncas, mc.nelecas)

    dmrg = DMRGCI(mol).init(
        ncas=mc.ncas,
        nelecas=mc.nelecas,
        nroots=1,
        bond_dims=[32] * 8,
        noises=[0.0] * 8,
        thrds=[1e-20] * 8,
        n_sweeps=8,
        tol=1e-12,
        scratch=tmp_path,
        n_threads=1,
        stack_memory=256,
        dav_max_iter=1000,
        random_seed=2468,
        npdm_site_type=2,
    )
    dmrg_energy, dmrg_ci = dmrg.kernel(
        h1eff, eri_active, mc.ncas, mc.nelecas,
        ecore=ecore, verbose=0)
    dmrg_dm1, dmrg_dm2 = dmrg.make_rdm12(
        dmrg_ci, mc.ncas, mc.nelecas)

    exact_gradient = zmc_superci.gen_g_hop(
        mc, mo.copy(), exact_dm1, exact_dm2, eris)[0]
    dmrg_gradient = zmc_superci.gen_g_hop(
        mc, mo.copy(), dmrg_dm1, dmrg_dm2, eris)[0]
    gradient_delta = dmrg_gradient - exact_gradient
    exact_rdm_energy_error = abs(
        energy_from_rdms(
            h1eff, eri_active, exact_dm1, exact_dm2, ecore)
        - exact_energy)
    dmrg_rdm_energy_error = abs(
        energy_from_rdms(
            h1eff, eri_active, dmrg_dm1, dmrg_dm2, ecore)
        - dmrg_energy)

    assert np.max(abs(h1_block2 - h1eff)) == 0
    assert np.max(abs(eri_block2 - eri_active)) == 0
    assert abs(dmrg_energy - exact_energy) <= 1e-9
    assert np.max(abs(dmrg_dm1 - exact_dm1)) <= 1e-8
    assert np.max(abs(dmrg_dm2 - exact_dm2)) <= 1e-8
    assert exact_rdm_energy_error <= 1e-9
    assert dmrg_rdm_energy_error <= 1e-9
    assert np.max(abs(gradient_delta)) <= 1e-7
    assert np.linalg.norm(gradient_delta) <= 1e-6
    assert dmrg.converged
    assert dmrg.convergence_info['energy_change'] <= dmrg.tol
    run_scratch = Path(dmrg._scratch)
    print(
        'fixed-orbital-differential',
        'dh1=0.000e+00',
        'deri=0.000e+00',
        'decore=0.000e+00',
        'dE=%.3e' % abs(dmrg_energy - exact_energy),
        'dm1=%.3e' % np.max(abs(dmrg_dm1 - exact_dm1)),
        'dm2=%.3e' % np.max(abs(dmrg_dm2 - exact_dm2)),
        'rdmE=%.3e' % dmrg_rdm_energy_error,
        'gradient_max=%.3e' % np.max(abs(gradient_delta)),
        'gradient_norm=%.3e' % np.linalg.norm(gradient_delta),
    )
    dmrg.close()
    assert not run_scratch.exists()
