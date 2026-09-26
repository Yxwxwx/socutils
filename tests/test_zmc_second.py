"""Independent energy derivatives and doubled-real references for orbital AH."""
from io import StringIO
from types import SimpleNamespace
import numpy as np
import pytest
from pyscf import gto
from pyscf.lib import logger
from scipy.linalg import eigh, expm, logm
from unittest.mock import patch

from socutils.dmrg.dmrgci import energy_from_rdms
from socutils.mcscf import zcasci, zmcscf, zmc_superci as sci, zmc_ah as second
from socutils.mcscf.zmc_utils import build_orbital_quantities
from socutils.scf import spinor_hf
from socutils.mcscf import zmc_ao2mo
from pyscf.ao2mo import nrr_outcore


def test_embedded_casci_omits_redundant_convergence_message():
    cas = zcasci.CASCI.__new__(zcasci.CASCI)
    cas.mo_coeff, cas.ci = np.eye(2), None
    cas.fcisolver = SimpleNamespace(converged=True)
    cas.stdout, cas.verbose = StringIO(), 0
    with patch.object(zcasci, 'kernel', return_value=(-1., -.5, None)):
        cas.kernel(verbose=logger.INFO)
        assert 'CASCI converged' in cas.stdout.getvalue()
        cas.stdout.seek(0)
        cas.stdout.truncate()
        embedded = zmcscf._fake_h_for_fast_casci(cas, cas.mo_coeff, None)
        embedded.kernel(verbose=logger.INFO)
        assert 'CASCI converged' not in cas.stdout.getvalue()


def test_core_jk_cache_shared_by_casci_and_both_orbital_operators():
    mol = gto.M(atom='H 0 0 0; H .2 .3 1.4', basis='6-31g', verbose=0)
    mf = spinor_hf.SCF(mol)
    rng = np.random.default_rng(128)
    n = mol.nao_2c()
    mo = np.linalg.qr(rng.normal(size=(n, n)) + 1j*rng.normal(size=(n, n)))[0]
    mc = zmcscf.CASSCF(mf, 4, 0)
    mc.mo_coeff = mo
    dm1, dm2 = np.zeros((4, 4)), np.zeros((4,)*4)
    eris = zmc_ao2mo._ERIS(mc, mo, level=2)
    dm = mo[:, :mc.ncore] @ mo[:, :mc.ncore].conj().T
    occ = np.arange(n) < mc.ncore
    reference = mf.get_jk(mol, dm)
    with patch.object(mf, 'get_jk', wraps=mf.get_jk) as jk:
        zmcscf._fake_h_for_fast_casci(mc, mo, eris)
        build_orbital_quantities(mc, mo, dm1, dm2, eris)
        sci._gen_g_hop_full(mc, mo, dm1, dm2, eris)
        assert jk.call_count == 3  # one core JK and two active JK calls
        actual = eris.get_jk(dm.copy(), mo_coeff=mo.copy(), mo_occ=occ)
        assert jk.call_count == 3
        np.testing.assert_allclose(actual, reference, atol=1e-12)
        # Responses must bypass the cache, even for the same numeric density.
        eris.get_jk(dm)
        assert jk.call_count == 4
        # Reusing an ERIS with a changed tagged density must not return stale JK.
        changed = dm.copy()
        changed[0, 0] += .01
        expected = mf.get_jk(mol, changed)
        actual = eris.get_jk(changed, mo_coeff=mo, mo_occ=occ)
        assert jk.call_count == 6
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        eris.get_jk(changed.copy(), mo_coeff=mo, mo_occ=occ)
        assert jk.call_count == 6
        # The cache owns a density copy, so in-place edits also invalidate it.
        changed[0, 0] += .01
        eris.get_jk(changed, mo_coeff=mo, mo_occ=occ)
        assert jk.call_count == 7


def test_shared_spinor_transforms_match_independent_pyscf_calls():
    mol = gto.M(atom='H 0 0 0; H .2 .3 1.4', basis='6-31g', verbose=0)
    mf = spinor_hf.SCF(mol)
    rng = np.random.default_rng(421)
    n = mol.nao_2c()
    mo = np.linalg.qr(rng.normal(size=(n, n)) + 1j*rng.normal(size=(n, n)))[0]
    mc = zmcscf.CASSCF(mf, 4, 2)
    with patch.object(nrr_outcore, 'half_e1', wraps=nrr_outcore.half_e1) as half:
        eris = zmc_ao2mo._ERIS(mc, mo, level=2)
        ppaa, papa, paap = eris.second_order_blocks()
        assert half.call_count == 2  # aa shared by aapa/aapp; pa by papa/paap
        assert all(0 < call.kwargs['max_memory'] <= mc.max_memory for call in half.call_args_list)
    a = mo[:, mc.ncore:mc.ncore+mc.ncas]
    with zmc_ao2mo.lib.H5TmpFile() as reference:
        for name, coeffs, actual in [('aapa', (a,a,mo,a), eris.paaa.transpose(2,3,0,1)),
                                     ('aapp', (a,a,mo,mo), ppaa.transpose(2,3,0,1)),
                                     ('papa', (mo,a,mo,a), papa), ('paap', (mo,a,a,mo), paap)]:
            nrr_outcore.general(mol, coeffs, reference, dataname=name, motype='j-spinor', verbose=0)
            np.testing.assert_allclose(actual.ravel(), reference[name][:].ravel(), atol=1e-12)


def test_inexact_ah_reports_tolerance_and_reuses_hessian_product():
    rng = np.random.default_rng(128)
    a = rng.normal(size=(24, 24))
    h = a.T @ a + np.eye(24)
    g = rng.normal(size=12) + 1j*rng.normal(size=12)
    calls = []
    def hop(x):
        calls.append(1)
        y = h @ np.r_[x.real, x.imag]
        return y[:12] + 1j*y[12:]
    strict, _, exact_info = second.davidson(hop, g, np.ones(12), tol=1e-10, mmax=24)
    calls.clear()
    step, _, info = second.davidson(hop, g, np.ones(12), tol=1e-10, mmax=24, micro_step_tol=1e-4)
    assert info['converged'] and not info['strict_converged']
    assert info['scaled_residual_norm'] <= info['effective_tolerance']
    assert len(calls) == info['iterations'] < exact_info['iterations']
    assert abs(info['quadratic_form'] - np.vdot(step, hop(step)).real) < 1e-12


def test_ah_rejects_degenerate_preconditioned_trial():
    def hop(vector):
        return vector
    hop.precondition = lambda vector, shift, floor: np.zeros_like(vector)
    with pytest.raises(second.AHNumericalError, match='trial vector'):
        second.davidson(hop, np.ones(2), np.ones(2))


@pytest.mark.parametrize('radius', [.2, 4.])
@pytest.mark.parametrize('curvature', [1., -5.])
def test_scaled_ah_matches_real_symmetric_reference(radius, curvature):
    rng = np.random.default_rng(834)
    a = rng.normal(size=(4, 4))
    h = a.T @ a + curvature*np.eye(4)
    gr = rng.normal(size=4)
    g = gr[:2] + 1j*gr[2:]
    def hop(x):
        y = h @ np.r_[x.real, x.imag]
        return y[:2] + 1j*y[2:]
    x, energy, info = second.davidson(hop, g, np.ones(2),
        max_stepsize=radius, tol=1e-10, mmax=4)
    assert info['converged']
    scale = info['ah_scale']
    ah = np.zeros((5, 5))
    ah[0, 1:] = ah[1:, 0] = gr
    ah[1:, 1:] = h/scale
    e, v = eigh(ah)
    root = np.flatnonzero(abs(v[0]) > .1)[0]
    expected = v[1:, root]/(scale*v[0, root])
    np.testing.assert_allclose(np.r_[x.real, x.imag], expected, atol=1e-10)
    assert abs(energy-e[root]) < 1e-10
    assert np.linalg.norm(hop(x) + g - scale*energy*x) < 1e-10
    # BAGEL AugHess stops its scale search within 1% of the requested step.
    assert np.linalg.norm(x) <= 1.01*radius/np.sqrt(2) + 1e-12
    if radius == .2:
        assert scale > 1


@pytest.mark.integration
def test_complex_orbital_hessian_against_fixed_rdm_energy(tmp_path):
    mol = gto.M(atom='H 0 0 0; F .35 .27 .8035', basis='6-31g', verbose=0)
    mf = spinor_hf.SCF(mol).x2camf(with_gaunt=False, with_breit=False)
    mf.init_guess = '1e'
    mf.conv_tol = 1e-11
    mf.kernel()
    assert mf.converged
    mc = zmcscf.CASSCF(mf, 4, 2)
    rng = np.random.default_rng(834)
    size = np.count_nonzero(mc.uniq_var_indices(len(mf.mo_coeff), mc.ncore, 4, None))
    v = rng.normal(size=size) + 1j*rng.normal(size=size)
    v /= np.linalg.norm(v)
    mo = mf.mo_coeff @ expm(.1*mc.unpack_uniq_var(v))
    mc.mo_coeff = mo.copy()
    eris, _ = sci._build_eris(mc, mo)
    cas = zmcscf._fake_h_for_fast_casci(mc, mo, eris)
    e0, _, ci = cas.kernel(mo, verbose=0)
    dm1, dm2 = mc.fcisolver.make_rdm12(ci, 4, 2)
    g, _, hop, *_ = second.gen_g_hop(mc, mo, dm1, dm2, eris)
    q = build_orbital_quantities(mc, mo, dm1, dm2, eris)
    u = rng.normal(size=size) + 1j*rng.normal(size=size)
    u /= np.linalg.norm(u)
    with patch.object(eris, 'get_jk', wraps=eris.get_jk) as jk:
        hv = hop(v)
        assert jk.call_count == 1 and jk.call_args.args[0].shape[0] == 2
    assert abs(np.vdot(u, hv).real - np.vdot(hop(u), v).real) < 1e-10
    assert np.linalg.norm(hop(1j*v)-1j*hop(v)) > 1e-5
    for direction in (v, 1j*v):
        x = mc.unpack_uniq_var(direction)
        energies, gradients = [], []
        eps = 2e-4
        for sign in (-1, 1):
            rotated = mo @ expm(sign*eps*x)
            eri, _ = sci._build_eris(mc, rotated)
            ci_model = zmcscf._fake_h_for_fast_casci(mc, rotated, eri)
            h1, ecore = ci_model.get_h1eff(rotated)
            energies.append(energy_from_rdms(h1, eri.aaaa, dm1, dm2, ecore).real)
            gradients.append(build_orbital_quantities(mc, rotated, dm1, dm2, eri).gradient)
        assert abs((energies[1]-energies[0])/(2*eps) - 2*np.vdot(g, direction).real) < 1e-7
        curvature = (energies[1]+energies[0]-2*e0)/eps**2
        assert abs(curvature - 2*np.vdot(direction, hop(direction)).real) < 5e-6
        expected = (gradients[1]-gradients[0])/(2*eps)
        expected += .5*(x@q.gradient-q.gradient@x)
        np.testing.assert_allclose(hop(direction), mc.pack_uniq_var(expected), atol=1e-7)
    # Core/virtual phase changes leave the active RDMs unchanged and must
    # transform both the gradient and Hessian action covariantly.
    phase = np.exp(1j*rng.uniform(-np.pi, np.pi, len(mo)))
    phase[mc.ncore:mc.ncore+mc.ncas] = 1
    phased = mo*phase
    peri, _ = sci._build_eris(mc, phased)
    pg, _, phop, *_ = second.gen_g_hop(mc, phased, dm1, dm2, peri)
    def transform(vector):
        return mc.pack_uniq_var(phase.conj()[:, None]*mc.unpack_uniq_var(vector)*phase)
    np.testing.assert_allclose(pg, transform(g), atol=1e-11)
    np.testing.assert_allclose(phop(transform(v)), transform(hop(v)), atol=1e-10)
    np.testing.assert_allclose(phop.precondition(transform(v), -.1, 1e-8),
                               transform(hop.precondition(v, -.1, 1e-8)), atol=1e-10)
    mc.max_cycle_macro = 1
    mc.canonicalization = False
    mc.chkfile = str(tmp_path / 'second.chk')
    with patch('scipy.linalg.expm', wraps=expm) as rotation:
        mc.second_order()
        assert rotation.call_count == 1
    row = mc.macro_history[0]
    assert row['orbital_method'] == 'second_order'
    assert row['linear_solver']['converged']
    assert row['prediction_model'] == 'quadratic_orbital_hessian'
    assert row['orbital_trials'][0]['radius'] == pytest.approx(np.sqrt(2))
    assert row['applied_orbital_step_norm'] <= np.sqrt(2)*mc.second_order_max_rotation + 1e-12
    step = mc.pack_uniq_var(logm(mo.conj().T @ mf.get_ovlp() @ mc.mo_coeff))
    assert np.linalg.norm(step) <= mc.second_order_max_rotation + 1e-12
    prediction = 2*np.vdot(g, step).real + np.vdot(step, hop(step)).real
    assert abs(row['predicted_energy_change']-prediction) < 1e-10
