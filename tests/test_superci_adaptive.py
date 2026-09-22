"""Adaptive Super-CI: independent complex projection, solves and opt-in routing."""
from itertools import combinations
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from pyscf import gto
from scipy.linalg import logm
from socutils.mcscf import zmc_superci as legacy
from socutils.mcscf import zmc_superci_adaptive as adaptive
from socutils.mcscf import zmcscf
from socutils.scf import spinor_hf


class Model:
    ncore, ncas = 1, 2
    frozen = freeze_pair = irrep = None
    canonicalize_ = natorb = False
    max_stepsize = .2
    verbose = 0
    mo_coeff = np.eye(4, dtype=complex)
    _scf = SimpleNamespace(with_df=None)
    fcisolver = SimpleNamespace()
    uniq_var_indices = zmcscf.CASSCF.uniq_var_indices
    screen_irrep = zmcscf.CASSCF.screen_irrep
    pack_uniq_var = zmcscf.CASSCF.pack_uniq_var
    unpack_uniq_var = zmcscf.CASSCF.unpack_uniq_var


def test_complex_determinant_projection():
    # Four spinors / two electrons, one core + one active electron. Explicit
    # fermion signs provide an independent H/S/source reference, including nulls.
    mc = Model()
    states = [sum(1 << p for p in occ) for occ in combinations(range(4), 2)]
    index = {state: i for i, state in enumerate(states)}
    ops = np.zeros((4, 4, 6, 6), complex)
    for p, q in np.ndindex(4, 4):
        for j, state in enumerate(states):
            if not state & (1 << q):
                continue
            sign = (-1) ** (state & ((1 << q) - 1)).bit_count()
            state ^= 1 << q
            if state & (1 << p):
                continue
            sign *= (-1) ** (state & ((1 << p) - 1)).bit_count()
            state |= 1 << p
            ops[p, q, index[state], j] = sign
    rng = np.random.default_rng(27)
    h = rng.normal(size=(4, 4)) + 1j*rng.normal(size=(4, 4))
    h = (h + h.conj().T) / 2
    _, vec = np.linalg.eigh(h[1:3, 1:3])
    psi = np.zeros(6, complex)
    psi[index[3]], psi[index[5]] = vec[:, 0]
    raw = np.einsum('i,pqij,j->pq', psi.conj(), ops, psi)[1:3, 1:3]
    lag = np.zeros_like(h)
    lag[:, :1] = h[:, :1]
    lag[:, 1:3] = h[:, 1:3] @ raw.T
    g, hd, hop, sop, _, _ = adaptive._operators(
        mc, mc.mo_coeff, raw, np.zeros((2,)*4), h, lag)
    rows, cols = np.where(mc.uniq_var_indices(4, 1, 2, None))
    exc = np.column_stack([ops[p, q] @ psi for p, q in zip(rows, cols)])
    mixed = h.copy()
    mixed[1:3, 1:3] = (lag[1:3, 1:3] + lag[1:3, 1:3].conj().T) / 2
    mixed[3:, 1:3] = lag[3:, 1:3]
    mixed[1:3, 3:] = lag[3:, 1:3].conj().T
    many_h = np.einsum('pq,pqij->ij', h, ops)
    many_mixed = np.einsum('pq,pqij->ij', mixed, ops)
    expected_h = exc.conj().T @ (many_mixed - np.vdot(psi, many_mixed@psi)*np.eye(6)) @ exc
    hh = np.column_stack([hop(x) for x in np.eye(len(g))])
    ss = np.column_stack([sop(x) for x in np.eye(len(g))])
    np.testing.assert_allclose(hh, expected_h, atol=1e-12)
    np.testing.assert_allclose(ss, exc.conj().T @ exc, atol=1e-12)
    np.testing.assert_allclose(g, exc.conj().T @ many_h @ psi, atol=1e-12)
    np.testing.assert_allclose(hh, hh.conj().T, atol=1e-12)
    np.testing.assert_allclose(hd, np.diag(hh), atol=1e-12)
    x = rng.normal(size=len(g)) + 1j*rng.normal(size=len(g))
    np.testing.assert_allclose(hop(1j*x), 1j*hop(x), atol=1e-12)
    expand, restrict, diagonal, shift_diagonal, _ = hop._superci_frame
    assert len(diagonal) < len(g)  # exact occupation nulls were removed
    for j, x in enumerate(np.eye(len(diagonal))):
        np.testing.assert_allclose(restrict(sop(expand(x))), x, atol=1e-12)
        assert abs(restrict(hop(expand(x)))[j] - diagonal[j]) < 1e-12
        assert abs(np.vdot(expand(x), expand(x)) - shift_diagonal[j]) < 1e-12


@pytest.mark.parametrize('occupation', [1., 1e-9, 1e-12])
def test_adaptive_residuals_and_zero_gradient(occupation):
    s = np.diag([1., occupation])
    h = np.diag([2., 3*occupation])
    b = np.diag([1., 1/np.sqrt(occupation)])
    def hop(x):
        return h @ x
    hop._superci_frame = (lambda x: b@x, lambda x: b.conj().T@x,
                          np.array([2., 3.]), np.array([1., 1/occupation]), .1)
    g = np.array([.02, 1e-4])
    step, energy, info = adaptive.davidson(hop, g, np.diag(h),
        sop=lambda x: s@x, tol=1e-9, mmax=20)
    assert info['converged'] and info['residual_norm'] <= 1e-9
    assert (info['orbital_shift'] == 0) == (occupation == 1)
    assert np.linalg.norm(h@step + info['orbital_shift']*step + g - energy*s@step) < 1e-9
    assert abs(np.vdot(g, step) - energy) < 1e-9
    if occupation < 1:
        assert np.linalg.norm(step) <= .1
    zero, _, info = adaptive.davidson(hop, g*0, np.diag(h),
        sop=lambda x: s@x, tol=1e-9, mmax=20)
    assert info['orbital_shift'] == 0 and np.linalg.norm(zero) == 0


def test_unshifted_solver_does_not_require_adaptive_regularization():
    # The full occupation metric spans twelve orders of magnitude. The
    # unshifted solution is large but must converge without an orbital shift.
    s = np.diag([1., 1e-12])
    h = np.diag([2., 3e-12])
    b = np.diag([1., 1e6])
    g = np.array([.02, 1e-6])
    def hop(x):
        return h @ x
    hop._superci_frame = (lambda x: b@x, lambda x: b.T@x,
                         np.array([2., 3.]), np.array([1., 1e12]), .1)
    step, energy, info = legacy.davidson(hop, g, np.diag(h),
        sop=lambda x: s@x, tol=1e-8, mmax=20)
    assert info['converged'] and info['orbital_shift'] == 0
    assert np.linalg.norm(step) > .1
    residual = h@step + g - energy*s@step
    assert np.linalg.norm(residual) < 1e-8
    assert np.linalg.norm(b.T@residual) < 1e-8
    assert abs(np.vdot(g, step) - energy) < 1e-8


def test_healthy_metric_still_requires_a_bounded_orbital_step():
    # The old condition-number-only policy accepted this oversized step,
    # then scaled every orbital direction down in the outer loop.
    def hop(x):
        return np.array([1., 2.])*x
    hop._superci_frame = (lambda x: x, lambda x: x,
                         np.array([1., 2.]), np.ones(2), .1)
    g = np.array([.4, .3])
    step, energy, info = adaptive.davidson(hop, g, np.array([1., 2.]),
        sop=lambda x: x, tol=1e-9, mmax=10)
    assert info['converged'] and info['orbital_shift'] > 0
    assert .0995 <= np.linalg.norm(step) <= .1
    assert np.linalg.norm(hop(step) + info['orbital_shift']*step + g - energy*step) < 1e-9


def test_shift_trials_share_complex_krylov_products():
    rng = np.random.default_rng(318)
    d = np.geomspace(1., 1e-4, 8)
    b = np.diag(1/np.sqrt(d))
    a = rng.normal(size=(8,8)) + 1j*rng.normal(size=(8,8))
    white = a.conj().T @ a + np.eye(8)
    h = np.sqrt(d)[:,None] * white * np.sqrt(d)[None,:]
    g = .1*(rng.normal(size=8) + 1j*rng.normal(size=8))
    calls = []
    def hop(x):
        calls.append(1)
        return h @ x
    hop._superci_frame = (lambda x:b@x, lambda x:b@x, np.diag(white).real, 1/d, .1)
    x, e, info = adaptive.davidson(hop, g, np.diag(h).real, sop=lambda x:d*x,
                                  tol=1e-8, mmax=8)
    assert info['converged'] and info['orbital_shift'] > 0
    assert len(calls) == info['total_davidson_iterations'] <= 8
    assert info['projected_shift_solves'] > len(calls)
    assert np.linalg.norm(x) <= .1*(1+1e-12)
    r = h@x + info['orbital_shift']*x + g - e*d*x
    assert max(np.linalg.norm(r), np.linalg.norm(b@r)) < 1e-8


@pytest.mark.parametrize('option,value,match', [
    ('canonicalize_', True, 'canonicalize_'),
    ('natorb', True, 'natorb'),
    ('frozen', 1, 'unfrozen'),
    ('freeze_pair', ([0], [1]), 'unscreened'),
    ('irrep', [0, 0, 1, 1], 'unscreened'),
    ('max_stepsize', 0, 'max_stepsize'),
])
def test_unsupported_orbital_options(option, value, match):
    mc = Model()
    setattr(mc, option, value)
    with pytest.raises(ValueError, match=match):
        adaptive.validate(mc, mc.mo_coeff)


@pytest.mark.parametrize('options,match', [
    ({'solver': 'gmres'}, 'davidson'),
    ({'cderi': object()}, 'full ERI'),
    ({'kramers': True}, 'Kramers'),
])
def test_unsupported_routes(options, match):
    mc = Model()
    with pytest.raises(ValueError, match=match):
        adaptive.validate(mc, mc.mo_coeff, **options)


@pytest.mark.integration
def test_instance_switch_separates_conditioning_from_step_control(tmp_path):
    # Full ERI, exact SGF CI, no CD or Kramers restriction. One macro update
    # exercises dispatch and checkpointing without claiming outer convergence.
    mol = gto.M(atom='H 0 0 0; F .35 .27 .8035', basis='sto-3g', verbose=0)
    mf = spinor_hf.SCF(mol).x2camf(with_gaunt=False, with_breit=False)
    mf.init_guess = '1e'
    mf.conv_tol = 1e-11
    mf.kernel()
    assert mf.converged
    original = (legacy.gen_g_hop, legacy.davidson)
    energies = []
    for i, enabled in enumerate([False, True, False]):
        mc = zmcscf.CASSCF(mf, 4, 2)
        assert mc.superci_adaptive is False
        assert 'superci_adaptive' in mc._keys
        mc.superci_adaptive = enabled
        mc.mo_coeff = mf.mo_coeff.copy()
        mc.canonicalization = False
        mc.max_cycle_macro = 1
        mc.conv_tol = mc.conv_tol_grad = 0.
        mc.superci_davidson_max_space = 200
        mc.chkfile = str(tmp_path / f'mc-{i}.chk')
        gradients = []
        def record(builder):
            def build(*args):
                result = builder(*args)
                gradients.append(result[0].copy())
                return result
            return build
        with patch.object(legacy, 'gen_g_hop', side_effect=record(original[0])) as old, \
             patch.object(adaptive, 'gen_g_hop', side_effect=record(adaptive.gen_g_hop)) as new:
            mc.superci()
        assert old.call_count == int(not enabled)
        assert new.call_count == int(enabled)
        assert mc.superci_diagnostics['adaptive'] == enabled
        assert mc.macro_history[0]['linear_solver']['converged']
        assert mc.macro_history[0]['adaptive'] == enabled
        assert mc.superci_diagnostics['integrals']['representation'] == 'full'
        assert not mc.superci_diagnostics['kramers_restricted']
        assert mc.macro_history[0]['prediction_model'] == 'linear_orbital_gradient'
        rotation = mf.mo_coeff.conj().T @ mf.get_ovlp() @ mc.mo_coeff
        applied = mc.pack_uniq_var(logm(rotation))
        expected_prediction = 2 * np.vdot(gradients[0], applied).real
        assert abs(mc.macro_history[0]['predicted_energy_change'] - expected_prediction) < 1e-10
        if not enabled:
            assert mc.macro_history[0]['linear_solver']['orbital_shift'] == 0
            assert mc.macro_history[0]['linear_solver']['reason'] == 'converged_unshifted_metric_frame'
        assert (legacy.gen_g_hop, legacy.davidson) == original
        energies.append(mc.e_tot)
    assert abs(energies[0] - energies[2]) < 1e-11
