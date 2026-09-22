"""Exercise actual DMRG state consistency across rejected orbital trials."""
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
from pyscf import gto, lib

from socutils.dmrg import DMRGCI
from socutils.dmrg.dmrgci import energy_from_rdms
from socutils.mcscf import zmcscf, zmc_superci as sci, zmc_second, zmc_superci_adaptive
from socutils.scf import spinor_hf


@pytest.mark.parametrize('second_order', [False, True])
def test_real_inner_solver_recovers_before_any_ci_call(second_order):
    rng = np.random.default_rng(128)
    if second_order:
        a = rng.normal(size=(24, 24))
        h = a.T @ a + np.eye(24)
        def hop(x):
            y = h @ np.r_[x.real, x.imag]
            return y[:12] + 1j*y[12:]
        solve, space = zmc_second.davidson, 4
    else:
        a = rng.normal(size=(12, 12)) + 1j*rng.normal(size=(12, 12))
        h = a.conj().T @ a + np.eye(12)
        def hop(x):
            return h @ x
        hop._superci_frame = (lambda x: x, lambda x: x, h.diagonal().real,
                             np.ones(12), .4/np.sqrt(2))
        solve, space = zmc_superci_adaptive.davidson, 8
    g = rng.normal(size=12) + 1j*rng.normal(size=12)
    def unpack(x):
        out = np.zeros((6, 6), complex)
        i, j = np.tril_indices(6, -1)
        out[i[:12], j[:12]] = x
        return out-out.conj().T
    mc = SimpleNamespace(second_order_micro_step_tol=1e-4,
        unpack_uniq_var=unpack, fcisolver=SimpleNamespace(converged=True))
    kernel = Mock(return_value=(-1., -1., None))
    with patch.object(sci, '_build_eris', return_value=(None, {})) as build, \
         patch.object(zmcscf, '_fake_h_for_fast_casci', return_value=SimpleNamespace(kernel=kernel)), \
         patch.object(sci, '_schedule_orbital_trial', wraps=sci._schedule_orbital_trial) as gate:
        result = sci._bounded_orbital_update(mc, np.eye(6), 0., g, np.ones(12),
            hop, lambda x: x, solve, .4, .4, 1e-8, space, 1e-8, 1e-4,
            second_order, 0, lib.logger.Logger(None, 0))
    trials = result[-1]
    assert [t['radius'] for t in trials] == [.4, .2, .1, .05, .025]
    assert all(t['stage'] == 'orbital_solve' and not t['accepted'] for t in trials[:-1])
    assert trials[-1]['accepted'] and trials[-1]['linear_solver']['converged']
    assert build.call_count == kernel.call_count == gate.call_count == 1
    assert gate.call_args.kwargs['accepted']  # failed inner solves did not replace the MPS
    assert np.linalg.norm(result[6]) <= .025


@pytest.mark.parametrize('reason,residual,attempts', [
    ('maximum_space', 1., 6), ('linear_dependence', 1., 6),
    ('invalid_operator', 1., 1), ('maximum_space', np.nan, 1),
])
def test_inner_failure_never_touches_ci_and_preserves_failure_history(reason, residual, attempts):
    mc = SimpleNamespace(converged=True, e_tot=-1., mo_coeff=np.eye(2), ci=object())
    saved_ci = mc.ci
    solve = Mock(return_value=(None, None,
        dict(converged=False, reason=reason, residual_norm=residual)))
    with patch.object(sci, '_build_eris') as build, \
         patch.object(sci, '_schedule_orbital_trial') as gate:
        with pytest.raises(RuntimeError, match='Six orbital trials|Orbital solve did not converge'):
            sci._bounded_orbital_update(mc, mc.mo_coeff, mc.e_tot, np.ones(1), np.ones(1),
                None, None, solve, .4, .4, 1e-8, 4, 1e-8, 1e-4,
                True, 0, lib.logger.Logger(None, 0))
    assert solve.call_count == len(mc.orbital_trial_history) == attempts
    assert [t['radius'] for t in mc.orbital_trial_history] == [.4/2**i for i in range(attempts)]
    build.assert_not_called()
    gate.assert_not_called()
    assert not mc.converged and mc.ci is saved_ci and mc.e_tot == -1.
    np.testing.assert_array_equal(mc.mo_coeff, np.eye(2))


@pytest.mark.parametrize('mode,exhaust', [('adaptive', False), ('second_order', False),
                                        ('second_order', True), ('second_order', 'inner')])
def test_rejected_trial_recomputes_matching_live_dmrg_state(tmp_path, mode, exhaust):
    mol = gto.M(atom='H 0 0 0; H .2 .3 1.4', basis='6-31g', verbose=0)
    mf = spinor_hf.SCF(mol).x2camf(with_gaunt=False, with_breit=False)
    mf.init_guess = '1e'
    mf.kernel()
    assert mf.converged
    solver = DMRGCI(mol).init(ncas=4, nelecas=2, nroots=1,
        bond_dims=[32]*8, noises=[0.]*8, thrds=[1e-14]*8, n_sweeps=8, tol=1e-10,
        scratch=tmp_path/'scratch', checkpoint_dir=tmp_path/'checkpoint',
        n_threads=1, stack_memory=128, random_seed=1234)
    mc = zmcscf.CASSCF(mf, 4, 2)
    mc.fcisolver = solver
    mc.canonicalization = False
    mc.max_cycle_macro = 1
    mc.max_stepsize = .4
    mc.chkfile = str(tmp_path/'mc.chk')
    mc.superci_adaptive = mode == 'adaptive'
    callbacks = []
    def callback(row):
        callbacks.append(row)
        solver.restart_scheduler_step(row)
    mc.callback = callback
    original = zmcscf._fake_h_for_fast_casci
    evaluated = []
    def inject(casscf, mo, eris):
        cas = original(casscf, mo, eris)
        kernel = cas.kernel
        def run(*args, **kwargs):
            result = kernel(*args, **kwargs)
            evaluated.append(np.array(mo))
            count = len(evaluated)
            # Disturb only the reported trial energy after real Block2 work.
            # The final base replay in the exhaustion case remains physical.
            if count == 2 or (exhaust is True and 2 <= count <= 7):
                return result[0]+1., result[1]+1., result[2]
            return result
        cas.kernel = run
        return cas
    original_solve = zmc_second.davidson
    def fail_after_rejected_ci(*args, **kwargs):
        x, e, info = original_solve(*args, **kwargs)
        if exhaust == 'inner' and len(evaluated) >= 2:
            info = dict(info, converged=False, reason='maximum_space')
        return x, e, info
    try:
        with patch.object(zmcscf, '_fake_h_for_fast_casci', side_effect=inject), \
             patch.object(zmc_second, 'davidson', side_effect=fail_after_rejected_ci):
            if exhaust:
                with pytest.raises(RuntimeError, match='Six orbital trials'):
                    mc.second_order()
                assert len(evaluated) == (3 if exhaust == 'inner' else 8)
                np.testing.assert_allclose(mc.mo_coeff, mf.mo_coeff, atol=1e-12)
                np.testing.assert_allclose(evaluated[-1], mf.mo_coeff, atol=1e-12)
                assert not mc.converged and not solver._restart
            else:
                (mc.superci if mode == 'adaptive' else mc.second_order)()
                assert len(evaluated) == 3
                row = callbacks[-1]
                trials = row['orbital_trials']
                assert row['accepted'] and row['rejected_trials'] == 1
                assert not trials[0]['accepted'] and trials[1]['accepted']
                assert trials[1]['radius'] == .5*trials[0]['radius']
                assert not trials[1]['restart']['enabled_for_next_kernel']
                assert trials[0]['restart']['structured_callback']
                assert trials[0]['restart']['orbital_step_norm'] == trials[0]['step_norm']
                np.testing.assert_allclose(mc.mo_coeff, evaluated[-1], atol=1e-12)
        assert solver.converged
        saved = solver.checkpoint_hamiltonian
        dm1, dm2 = solver.make_rdm12(mc.ci, 4, 2)
        energy = energy_from_rdms(saved['h1e'], saved['eri'], dm1, dm2, saved['ecore']).real
        assert abs(energy - mc.e_tot) < 1e-9
        eris, _ = sci._build_eris(mc, mc.mo_coeff)
        cas = original(mc, mc.mo_coeff, eris)
        h1, ecore = cas.get_h1eff(mc.mo_coeff)
        np.testing.assert_allclose(h1, saved['h1e'], atol=1e-11)
        np.testing.assert_allclose(eris.aaaa, saved['eri'], atol=1e-11)
        assert abs(ecore-saved['ecore']) < 1e-11
        # Terminal callback cannot enable another warm solve.
        solver.restart_scheduler_step(dict(accepted=True, ci_solver_converged=True,
                                          orbital_gradient_norm=1e-8, converged=True))
        assert not solver._restart
    finally:
        solver.close()
