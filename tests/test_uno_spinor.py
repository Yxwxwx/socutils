"""Numerical and interface checks for one-time spinor UNO/PM preparation."""

import hashlib
import importlib.util
import inspect
import os
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from pyscf.fci import cistring
from socutils.fci import zfci
from socutils.lo import uno_spinor as uno

REFERENCE = Path(__file__).resolve().parents[1] / "lo" / "uno_b2.py"


class FakeMol:
    def __init__(self, n, nelectron, groups=None):
        self.n = n
        self.nelectron = nelectron
        self.groups = groups if groups is not None else [list(range(n // 2)), list(range(n // 2, n))]
        self.natm = len(self.groups)

    def nao_2c(self):
        return self.n

    def aoslice_2c_by_atom(self):
        start = 0
        slices = []
        for group in self.groups:
            slices.append([0, 0, start, start + len(group)])
            start += len(group)
        return np.asarray(slices)

    def intor_symmetric(self, name, comp=None):
        if name == "int1e_ovlp_spinor":
            return np.eye(self.n)
        raise ValueError(name)


def _random_unitary(n, seed=11):
    rng = np.random.default_rng(seed)
    return np.linalg.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))[0]


def test_reference_and_public_signatures():
    assert hashlib.sha256(REFERENCE.read_bytes()).hexdigest() == (
        "613e77442c33a0feea57f8b90951356f8c0345af0701ffe306b8495b60ff4783"
    )
    spec = importlib.util.spec_from_file_location("uno_b2_reference", REFERENCE)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    for name in ("sqrtm", "lowdin", "pmloc", "sort_orbitals"):
        assert inspect.signature(getattr(uno, name)) == inspect.signature(getattr(upstream, name))
    assert len(uno.pmloc([[0], [1]], np.eye(2), iop=1, iprint=0)) == 2
    assert len(uno.sort_orbitals(FakeMol(2, 1), np.eye(2), np.array([1., 0.]),
                                 np.array([-1., 1.]), cas_list=[0], iprint=0)) == 5


def test_real_list_pmloc_matches_fixed_source():
    spec = importlib.util.spec_from_file_location("uno_b2_reference", REFERENCE)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    c = np.linalg.qr(np.random.default_rng(7).normal(size=(6, 6)))[0]
    rows = [[0, 1], [2, 3], [4, 5]]
    old_err, old_u = upstream.pmloc(rows, c, iop=1, iprint=0)
    err, u = uno.pmloc(rows, c, iop=1, iprint=0)
    assert err == old_err == 0
    np.testing.assert_allclose(u, old_u, atol=2e-11)
    assert uno.pmloc(rows, c[:, :0], iop=1, iprint=0)[0] == 0
    assert uno.pmloc(rows, c[:, :1], iop=1, iprint=0)[0] == 0
    assert uno.pmloc(rows, c, iop=1, tol=1e-14, maxcycle=1, iprint=0)[0] == 1


def test_complex_pair_exact_oracle_and_phase():
    rng = np.random.default_rng(21)
    q = rng.normal(size=(4, 2, 2)) + 1j * rng.normal(size=(4, 2, 2))
    q = 0.5 * (q + q.swapaxes(1, 2).conj())
    q[0, 0, 1] = 0.71j
    q[0, 1, 0] = -0.71j
    rotation = uno._complex_pair(q, 0, 1)
    np.testing.assert_allclose(rotation.conj().T @ rotation, np.eye(2), atol=1e-13)
    diag0 = np.diagonal(q, axis1=1, axis2=2).real
    old = float(np.sum(diag0 ** 2))
    rotated = rotation.conj().T @ q @ rotation
    new = uno._objective(rotated)
    v = np.stack((q[:, 0, 1].real, -q[:, 0, 1].imag,
                  (q[:, 0, 0].real - q[:, 1, 1].real) / 2), axis=1)
    g = v.T @ v
    np.testing.assert_allclose(new - old, 2 * (np.linalg.eigvalsh(g)[-1] - g[2, 2]), atol=1e-12)
    for random_rotation in (_random_unitary(2, seed) for seed in range(20)):
        assert new >= uno._objective(random_rotation.conj().T @ q @ random_rotation) - 1e-11
    # Independent finite differences in the real and imaginary mixing directions.
    eps = 1e-5
    for generator in (np.array([[0., -1.], [1., 0.]]),
                      np.array([[0., 1j], [1j, 0.]])):
        plus = np.cos(eps) * np.eye(2) + np.sin(eps) * generator
        minus = plus.conj().T
        fp = uno._objective(plus.conj().T @ rotated @ plus)
        fm = uno._objective(minus.conj().T @ rotated @ minus)
        assert abs((fp - fm) / (2 * eps)) < 1e-7
        assert fp + fm <= 2 * new + 1e-12
    c = _random_unitary(4)
    rows = [[0, 2], [1, 3]]
    err, u = uno.pmloc(rows, c, iop=1, iprint=0)
    assert err == 0
    np.testing.assert_allclose(u.conj().T @ u, np.eye(4), atol=1e-10)
    initial = uno._objective(uno._population(rows, c, 1))
    final = uno._objective(uno._population(rows, c @ u, 1))
    assert final >= initial - 1e-11
    phases = np.diag(np.exp(1j * np.array([.1, .4, -.2, .8])))
    err2, u2 = uno.pmloc(rows, c @ phases, iop=1, iprint=0)
    assert err2 == 0
    np.testing.assert_allclose(uno._objective(uno._population(rows, c @ phases @ u2, 1)), final, atol=1e-8)


def test_flat_degenerate_and_imaginary_pairs(capsys):
    assert uno.pmloc([], np.empty((0, 0)), iop=1, iprint=0)[0] == 0
    c = _random_unitary(2)
    err, u = uno.pmloc([[0, 1]], c, iop=1)
    assert err == 0
    np.testing.assert_array_equal(u, np.eye(2))
    assert "maximum remaining pair gain" in capsys.readouterr().out
    q = np.array([[[.5, .3j], [-.3j, .5]], [[.5, -.3j], [.3j, .5]]])
    rotation = uno._complex_pair(q, 0, 1)
    np.testing.assert_allclose(uno._objective(rotation.conj().T @ q @ rotation)
                               - uno._objective(q), .36, atol=1e-13)
    # A degenerate top eigenspace: select the optimal direction nearest z.
    axes = np.linalg.qr(np.random.default_rng(53).normal(size=(3, 3)))[0]
    vectors = np.diag(np.sqrt([2., 2., 1.])) @ axes.T
    q = np.zeros((3, 2, 2), dtype=complex)
    q[:, 0, 0], q[:, 1, 1] = vectors[:, 2], -vectors[:, 2]
    q[:, 0, 1] = vectors[:, 0] - 1j * vectors[:, 1]
    q[:, 1, 0] = q[:, 0, 1].conj()
    direction = axes[:, :2] @ axes[2, :2]
    direction /= np.linalg.norm(direction)
    expected = np.array([[direction[2], direction[0] - 1j * direction[1]],
                         [direction[0] + 1j * direction[1], -direction[2]]])
    rotation = uno._complex_pair(q, 0, 1)
    np.testing.assert_allclose(rotation @ np.diag([1, -1]) @ rotation.conj().T,
                               expected, atol=1e-13)


def test_spinor_population_and_overlap():
    rng = np.random.default_rng(31)
    a = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    s = a.conj().T @ a + np.eye(4)
    c = uno.lowdin(s) @ _random_unitary(4)
    mol = FakeMol(4, 2)
    mol.intor_symmetric = lambda name, comp=None: s
    q = uno._population(mol, c[:, :3], 0)
    np.testing.assert_allclose(q.sum(axis=0), np.eye(3), atol=1e-11)
    np.testing.assert_allclose(q, q.swapaxes(1, 2).conj(), atol=1e-11)
    for atom, rows in enumerate(([0, 1], [2, 3])):
        p = np.zeros((4, 4))
        p[rows, rows] = 1
        expected = c[:, :3].conj().T @ (s @ p + p @ s) @ c[:, :3] / 2
        np.testing.assert_allclose(q[atom], expected, atol=1e-11)
    np.testing.assert_allclose(uno.sqrtm(s) @ uno.sqrtm(s), s, atol=1e-11)
    np.testing.assert_allclose(uno.lowdin(s) @ s @ uno.lowdin(s), np.eye(4), atol=1e-11)
    with pytest.raises(ValueError, match="positive definite"):
        uno.lowdin(np.diag([1., -1.]))
    q_lowdin = uno._population(mol, c[:, :3], 1)
    assert np.linalg.eigvalsh(q_lowdin).min() > -1e-12
    # Mulliken populations need not be positive semidefinite.
    s = np.array([[1., .8], [.8, 1.]])
    mol = FakeMol(2, 1)
    mol.intor_symmetric = lambda name, comp=None: s
    c = np.linalg.solve(np.linalg.cholesky(s).T, np.eye(2))
    assert np.linalg.eigvalsh(uno._population(mol, c, 0)).min() < -.1
    assert uno.pmloc(mol, c, iprint=0)[0] == 0


def test_simple_split_and_noncontiguous_metadata():
    mol = FakeMol(32, 14)
    occ = np.r_[np.ones(14), np.zeros(18)]
    e = np.arange(32., dtype=float)
    c = np.eye(32, dtype=complex)
    widths = []

    def spy(_mol, x, **kwargs):
        widths.append(x.shape[1])
        assert kwargs == {"iprint": 0}
        return 0, np.eye(x.shape[1])

    with patch.object(uno, "pmloc", side_effect=spy):
        uno.sort_orbitals(mol, c.copy(), occ.copy(), e.copy(),
                          cas_list=list(range(30)), iprint=0)
        assert widths == []
        uno.sort_orbitals(mol, c.copy(), occ.copy(), e.copy(),
                          cas_list=list(range(30)), do_loc=True, iprint=0)
        assert widths == [30]
        widths.clear()
        uno.sort_orbitals(mol, c.copy(), occ.copy(), e.copy(),
                          cas_list=list(range(30)), do_loc=True,
                          split_low=.05, split_high=.95, iprint=0)
        assert widths == [16, 14]
        widths.clear()
        fractional = np.array([.2, .5, .3, 1.])
        uno.sort_orbitals(FakeMol(4, 2), np.eye(4, dtype=complex), fractional,
                          np.arange(4., dtype=float), cas_list=[0, 1, 2, 3],
                          do_loc=True, split_low=.2, split_high=.5, iprint=0)
        assert widths == [1, 2, 1]

    mol = FakeMol(6, 3)
    occ = np.array([1., 1., 1., 0., 0., 0.])
    energy = np.array([-6., -5., -4., 1., 2., 3.])
    c = np.eye(6, dtype=complex)
    result = uno.sort_orbitals(mol, c, occ, energy, cas_list=[3, 1], iprint=0)
    returned, n, en, na, ne = result
    assert (na, ne) == (2, 1)
    # Source extraction [3,1], sorted write-back [1,3], final C/A/V [0,2,1,3,4,5].
    np.testing.assert_allclose(returned, np.eye(6)[:, [0, 2, 1, 3, 4, 5]])
    np.testing.assert_array_equal(en, energy[[0, 2, 1, 3, 4, 5]])
    np.testing.assert_array_equal(n, occ[[0, 2, 1, 3, 4, 5]])


def test_full_block_pm_scdm_keeps_spaces_and_metadata():
    mol = FakeMol(8, 4)
    rng = np.random.default_rng(75)
    a = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    s = a.conj().T @ a + np.eye(8)
    mol.intor_symmetric = lambda name, comp=None: s
    c0 = np.linalg.solve(np.linalg.cholesky(s).conj().T, _random_unitary(8, 76))
    n0 = np.r_[np.ones(4), np.zeros(4)]
    e0 = np.arange(8., dtype=float) - 4
    seen = []
    original_pm = uno.pmloc

    def spy(_mol, block, **kwargs):
        seen.append(block.shape[1])
        return original_pm(_mol, block, **kwargs)

    with patch.object(uno, "pmloc", side_effect=spy):
        c, n, e, ncas, nelec = uno.localize_blocks(
            mol, c0, n0, e0, cas_list=list(range(2, 6)), iprint=0,
        )
    assert seen == [4, 2]  # active PM, core PM; virtual uses SCDM
    assert (ncas, nelec) == (4, 2)
    np.testing.assert_allclose(c.conj().T @ s @ c, np.eye(8), atol=1e-11)
    for begin, end in ((0, 2), (2, 6), (6, 8)):
        old = c0[:, begin:end]
        new = c[:, begin:end]
        np.testing.assert_allclose(new @ new.conj().T @ s,
                                   old @ old.conj().T @ s, atol=1e-11)
    t = c0.conj().T @ s @ c
    np.testing.assert_allclose(n, n0 @ abs(t) ** 2, atol=1e-11)
    np.testing.assert_allclose(e, e0 @ abs(t) ** 2, atol=1e-11)
    h = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    h = (h + h.conj().T) / 2
    factors = rng.normal(size=(3, 8, 8)) + 1j * rng.normal(size=(3, 8, 8))
    factors = (factors + factors.swapaxes(1, 2).conj()) / 2
    eri = np.einsum("kpq,krs->pqrs", factors, factors)
    hnew = t.conj().T @ h @ t
    erinew = np.einsum("pa,qb,rc,sd,pqrs->abcd", t.conj(), t, t.conj(), t, eri)
    determinants = list(cistring.gen_occslst(range(8), 4))
    cas = [i for i, det in enumerate(determinants)
           if 0 in det and 1 in det and all(orb < 6 for orb in det)]
    before = zfci.make_hmat(h, eri, 8, 4)[np.ix_(cas, cas)]
    after = zfci.make_hmat(hnew, erinew, 8, 4)[np.ix_(cas, cas)]
    np.testing.assert_allclose(np.linalg.eigvalsh(after),
                               np.linalg.eigvalsh(before), atol=1e-10)
    assert np.max(abs(t[6:8, 6:8].imag)) > 1e-3
    np.testing.assert_array_equal(c0, np.linalg.solve(np.linalg.cholesky(s).conj().T,
                                                       _random_unitary(8, 76)))
    np.testing.assert_array_equal(n0, np.r_[np.ones(4), np.zeros(4)])
    np.testing.assert_array_equal(e0, np.arange(8.) - 4)

    with (patch.object(uno, "pmloc", side_effect=[(0, np.eye(4)), (1, np.eye(2))]),
          pytest.raises(RuntimeError, match="core group")):
        uno.localize_blocks(mol, c0, n0, e0, cas_list=list(range(2, 6)), iprint=0)
    np.testing.assert_array_equal(n0, np.r_[np.ones(4), np.zeros(4)])


def test_failure_is_transactional():
    mol = FakeMol(4, 2)
    c = np.eye(4, dtype=complex)
    occ = np.array([1., 1., 0., 0.])
    e = np.arange(4., dtype=float)
    with (patch.object(uno, "pmloc", return_value=(1, np.eye(2))),
          pytest.raises(RuntimeError, match="did not converge")):
        uno.sort_orbitals(mol, c, occ, e, cas_list=[1, 2], do_loc=True, iprint=0)
    np.testing.assert_array_equal(c, np.eye(4))
    np.testing.assert_array_equal(occ, [1, 1, 0, 0])
    np.testing.assert_array_equal(e, np.arange(4.))
    with pytest.raises(ValueError, match="conflicts"):
        uno.sort_orbitals(mol, c, occ, e, cas_list=[1, 2], nactelec=2, iprint=0)
    mol.so_contr = np.eye(4)
    with pytest.raises(ValueError, match="so_contr"):
        uno.sort_orbitals(mol, c, occ, e, cas_list=[1, 2], iprint=0)


def test_invalid_inputs_fail_before_writeback():
    mol = FakeMol(4, 2)
    c = np.eye(4, dtype=complex)
    n = np.array([1., 1., 0., 0.])
    e = np.arange(4.)
    for kwargs in ({"cas_list": [1, 1]}, {"cas_list": [-1, 2]},
                   {"cas_list": [1, 4]}, {"cas_list": [1.5, 2.]},
                   {"cas_list": [1, 2], "nactorb": 3},
                   {"cas_list": [1, 2], "split_low": .2},
                   {"cas_list": [1, 2], "do_loc": True, "split_low": .9, "split_high": .1},
                   {"cas_list": [1, 2], "split_high": float("nan")}):
        with pytest.raises(ValueError):
            uno.sort_orbitals(mol, c, n, e, iprint=0, **kwargs)
        np.testing.assert_array_equal(c, np.eye(4))
        np.testing.assert_array_equal(n, [1., 1., 0., 0.])
        np.testing.assert_array_equal(e, np.arange(4.))
    with pytest.raises(ValueError, match="not an integer"):
        uno.sort_orbitals(mol, c, np.array([1., .2, .3, .5]), e, cas_list=[1, 2], iprint=0)
    # Fractional active occupations must not disable core/virtual ordering checks.
    with pytest.raises(ValueError, match="complement is not ordered"):
        uno.localize_blocks(FakeMol(6, 3), np.eye(6, dtype=complex),
                            np.array([1., .5, .5, 0., 1., 0.]), np.arange(6.),
                            cas_list=[1, 2], iprint=0)
    for controls in ({"tol": 0}, {"tol": float("nan")}, {"maxcycle": 0}, {"maxcycle": 1.5}):
        with pytest.raises(ValueError):
            uno.pmloc([[0, 1], [2, 3]], c, iop=1, iprint=0, **controls)
    with pytest.raises(ValueError, match="exactly once"):
        uno.pmloc([[0, 1], [1, 3]], c, iop=1, iprint=0)


def test_contiguous_window_and_odd_spinor_core():
    mol = FakeMol(5, 3, groups=[[0, 1], [2, 3, 4]])
    c = np.eye(5, dtype=complex)
    occ = np.array([1., 1., 1., 0., 0.])
    energy = np.arange(5., dtype=float)
    by_window = uno.sort_orbitals(mol, c.copy(), occ.copy(), energy.copy(),
                                  nactorb=2, nactelec=2, iprint=0)
    by_list = uno.sort_orbitals(mol, c.copy(), occ.copy(), energy.copy(),
                                cas_list=[1, 2], iprint=0)
    for a, b in zip(by_window, by_list):
        np.testing.assert_allclose(a, b)
    assert by_window[3:] == (2, 2)  # ncore=1: no Kramers/even-core restriction


def test_integer_metadata_and_real_input_keep_complex_results():
    mol = FakeMol(2, 1)
    c = np.eye(2)
    n = np.array([1, 0])
    e = np.array([-2, 1])
    rotation = np.array([[1., 1j], [1j, 1.]]) / np.sqrt(2)
    with patch.object(uno, "pmloc", return_value=(0, rotation)):
        out = uno.sort_orbitals(mol, c, n, e, cas_list=[0, 1], do_loc=True,
                                split_low=-.1, split_high=1.1, iprint=0)
    np.testing.assert_allclose(out[1], [.5, .5], atol=1e-14)
    np.testing.assert_allclose(out[2], [-.5, -.5], atol=1e-14)
    assert out[0].dtype == np.complex128 and abs(out[0].imag).max() > .7
    np.testing.assert_array_equal(c, np.eye(2))
    np.testing.assert_array_equal(n, [1, 0])
    np.testing.assert_array_equal(e, [-2, 1])


def test_split_masks_writeback_and_independent_ao_metadata():
    mol = FakeMol(10, 5)
    rng = np.random.default_rng(55)
    a = rng.normal(size=(10, 10)) + 1j * rng.normal(size=(10, 10))
    s = a.conj().T @ a + np.eye(10)
    mol.intor_symmetric = lambda name, comp=None: s
    c0 = np.linalg.solve(np.linalg.cholesky(s).conj().T, _random_unitary(10))
    n0 = np.array([1., 0., .5, 1., 1., 0., 0., 1., .5, 0.])
    e0 = np.arange(10.) - 4
    selected = [7, 1, 8, 2, 5, 4]
    masks = [n0[selected] <= .25, (n0[selected] > .25) & (n0[selected] <= .75),
             n0[selected] > .75]
    calls = []

    def group_rotation(_mol, block, **kwargs):
        calls.append(block.copy())
        return 0, _random_unitary(2)

    c_in, n_in, e_in = c0.copy(), n0.copy(), e0.copy()
    with patch.object(uno, "pmloc", side_effect=group_rotation):
        c, n, e, na, ne = uno.sort_orbitals(
            mol, c_in, n_in, e_in, cas_list=selected, do_loc=True,
            split_low=.25, split_high=.75, iprint=0,
        )
    assert (na, ne) == (6, 3)
    t = c0[:, selected].conj().T @ s @ c[:, 2:8]
    for mask, block in zip(masks, calls):
        np.testing.assert_allclose(block, c0[:, selected][:, mask], atol=1e-13)
        np.testing.assert_allclose(t[np.ix_(mask, ~mask)], 0, atol=1e-13)
    np.testing.assert_allclose(c[:, [0, 1, 8, 9]], c0[:, [0, 3, 6, 9]], atol=1e-13)
    np.testing.assert_allclose(c_in[:, sorted(selected)], c[:, 2:8], atol=1e-13)
    np.testing.assert_allclose(n_in[sorted(selected)], n[2:8], atol=1e-13)
    np.testing.assert_allclose(e_in[sorted(selected)], e[2:8], atol=1e-13)
    density = c0 @ np.diag(n0) @ c0.conj().T
    proxy = c0 @ np.diag(e0) @ c0.conj().T
    np.testing.assert_allclose(n, np.diag(c.conj().T @ s @ density @ s @ c).real, atol=1e-12)
    np.testing.assert_allclose(e, np.diag(c.conj().T @ s @ proxy @ s @ c).real, atol=1e-12)
    # A failure after one successful group must still leave all inputs intact.
    c_in, n_in, e_in = c0.copy(), n0.copy(), e0.copy()
    with (patch.object(uno, "pmloc", side_effect=[(0, _random_unitary(2)), (1, np.eye(2))]),
          pytest.raises(RuntimeError, match="mid active group")):
        uno.sort_orbitals(mol, c_in, n_in, e_in, cas_list=selected, do_loc=True,
                          split_low=.25, split_high=.75, iprint=0)
    np.testing.assert_array_equal(c_in, c0)
    np.testing.assert_array_equal(n_in, n0)
    np.testing.assert_array_equal(e_in, e0)


def test_psort_uses_full_original_density_and_energy_proxy():
    mol = FakeMol(4, 2)
    c0 = np.eye(4, dtype=complex)
    n0 = np.array([1., 1., 0., 0.])
    e0 = np.array([-3., -1., 2., 4.])
    theta = .37
    rotation = np.array([[np.cos(theta), -np.sin(theta) * 1j],
                         [-np.sin(theta) * 1j, np.cos(theta)]])
    with patch.object(uno, "pmloc", return_value=(0, rotation)):
        c, n, e, _, _ = uno.sort_orbitals(
            mol, c0.copy(), n0.copy(), e0.copy(), cas_list=[1, 2],
            do_loc=True, iprint=0,
        )
    density = c0 @ np.diag(n0) @ c0.conj().T
    proxy = c0 @ np.diag(e0) @ c0.conj().T
    np.testing.assert_allclose(n, np.diag(c.conj().T @ density @ c).real, atol=1e-13)
    np.testing.assert_allclose(e, np.diag(c.conj().T @ proxy @ c).real, atol=1e-13)
    assert abs((c.conj().T @ density @ c)[1, 2]) > .1
    np.testing.assert_allclose(c[:, 1:3] @ c[:, 1:3].conj().T,
                               c0[:, 1:3] @ c0[:, 1:3].conj().T, atol=1e-13)
    real_input = np.eye(4)
    result = uno.sort_orbitals(mol, real_input, n0.copy(), e0.copy(),
                               cas_list=[1, 2], do_loc=False, iprint=0)
    assert result[0].dtype == np.complex128
    np.testing.assert_array_equal(real_input, np.eye(4))


def test_complex_exact_casci_similarity():
    rng = np.random.default_rng(44)
    h = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    h = (h + h.conj().T) / 2
    factors = rng.normal(size=(3, 4, 4)) + 1j * rng.normal(size=(3, 4, 4))
    factors = (factors + factors.swapaxes(1, 2).conj()) / 2
    eri = np.einsum("kpq,krs->pqrs", factors, factors).astype(complex)
    ierr, u = uno.pmloc([[0, 1], [2, 3]], _random_unitary(4, 45), iop=1, iprint=0)
    assert ierr == 0
    hnew = u.conj().T @ h @ u
    enew = np.einsum("pa,qb,rc,sd,pqrs->abcd", u.conj(), u, u.conj(), u, eri)
    old = zfci.make_hmat(h, eri, 4, 2)
    new = zfci.make_hmat(hnew, enew, 4, 2)
    determinants = list(cistring.gen_occslst(range(4), 2))
    wedge = np.array([[np.linalg.det(u[np.ix_(old_pair, new_pair)])
                       for new_pair in determinants] for old_pair in determinants])
    np.testing.assert_allclose(new, wedge.conj().T @ old @ wedge, atol=1e-10, rtol=0)
    np.testing.assert_allclose(np.linalg.eigvalsh(new), np.linalg.eigvalsh(old), atol=1e-10, rtol=0)


@pytest.mark.parametrize("weights,use_cd,basis", [
    (None, False, "sto-3g"), ((.4, .2, .2, .2), False, "sto-3g"),
    (None, True, "sto-3g"), (None, False, "6-31g"),
])
def test_native_x2c_h2_pm_dmrg_scf(tmp_path, weights, use_cd, basis):
    from pyscf import gto, lib
    from socutils.dmrg import DMRGCI
    from socutils.mcscf import zmcscf
    from socutils.scf import spinor_hf

    lib.num_threads(2)
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis=basis, verbose=0,
                max_memory=2000)
    mf = spinor_hf.SCF(mol).x2camf(with_gaunt=False, with_breit=False)
    if use_cd:
        mf = mf.cholesky(tau=1e-8)
    mf.conv_tol = 1e-10
    mf.kernel()
    assert mf.converged
    np.testing.assert_allclose(mf.get_ovlp(), mol.intor_symmetric("int1e_ovlp_spinor"))
    c, _, _, ncas, nelec = uno.sort_orbitals(
        mol, mf.mo_coeff.copy(), mf.mo_occ.copy(), mf.mo_energy.copy(),
        cas_list=list(range(4)), do_loc=True, iprint=0,
    )
    t = mf.mo_coeff.conj().T @ mf.get_ovlp() @ c
    full_density = t.conj().T @ np.diag(mf.mo_occ) @ t
    rebuilt_dm = c @ full_density @ c.conj().T
    np.testing.assert_allclose(rebuilt_dm, mf.make_rdm1(), atol=1e-12)
    assert abs(mf.energy_tot(dm=rebuilt_dm) - mf.e_tot) < 1e-10
    assert uno.pmloc(mol, c[:, :2], iop=2, iprint=0)[0] == 0

    def configure(mc):
        mc.mo_coeff = c.copy()
        mc.max_cycle_macro = 30
        mc.conv_tol = 1e-9
        mc.conv_tol_grad = 1e-5
        mc.canonicalization = False
        mc.canonicalize_ = False
        mc.natorb = False
        return mc

    exact = configure(zmcscf.CASSCF(mf, ncas, nelec))
    if weights is not None:
        exact.state_average_(weights)
    exact.second_order()
    assert exact.converged
    exact_energy = exact.e_tot
    solver = DMRGCI(mol).init(
        ncas=ncas, nelecas=nelec, nroots=1 if weights is None else len(weights), bond_dims=[16] * 8,
        noises=[0.] * 8, thrds=[1e-14] * 8, n_sweeps=8, tol=1e-9,
        scratch=tmp_path / "scratch", checkpoint_dir=tmp_path / "checkpoint",
        n_threads=2, stack_memory=512, orbital_ordering="original",
    )
    mc = configure(zmcscf.CASSCF(mf, ncas, nelec))
    mc.fcisolver = solver
    if weights is not None:
        mc.state_average_(weights)
        solver = mc.fcisolver
    mc.callback = solver.restart_scheduler_()
    try:
        mc.second_order()
        assert mc.converged and solver.converged
        assert abs(mc.e_tot - exact_energy) < 1e-8
        if weights is not None:
            assert solver.nroots == len(weights)
            np.testing.assert_array_equal(solver.weights, weights)
            np.testing.assert_allclose(mc.e_states, exact.fcisolver.e_states, atol=1e-8, rtol=0)
        if basis == "6-31g":
            assert mc.macro_history[0]["orbital_gradient_norm"] > 1e-4
            assert len(mc.macro_history) > 1
        print(f"X2CAMF H2/{basis} PM roots={solver.nroots} CD={use_cd}: "
              f"exact CASSCF={exact_energy:.15f}, DMRG-SCF={mc.e_tot:.15f}, "
              f"macro={mc.macro_history[-1]['macro_iteration']}")
    finally:
        solver.close()


def test_native_unf3_x2camf_checkpoint_pm():
    from pyscf import scf

    path = Path(os.environ.get(
        "SOCUTILS_UNF3_HF_CHECKPOINT",
        "/home/Yxwxwx/new-dmrgscf/unf3/R_1.750/large/ah200/dmrg_state/hf.chk",
    ))
    if not path.is_file():
        pytest.skip("UNF3 X2CAMF-HF checkpoint unavailable")
    mol, saved = scf.chkfile.load_scf(path)
    c0 = np.asarray(saved["mo_coeff"])
    n0 = np.asarray(saved["mo_occ"])
    e0 = np.asarray(saved["mo_energy"])
    s = mol.intor_symmetric("int1e_ovlp_spinor")
    assert c0.shape == (mol.nao_2c(), mol.nao_2c())
    np.testing.assert_allclose(c0.conj().T @ s @ c0, np.eye(c0.shape[1]), atol=1e-7)
    start_q = uno._population(mol, c0[:, 112:142], 0)
    initial_objective = uno._objective(start_q)
    imaginary_q = np.max(abs(np.triu(start_q.imag, k=1)))
    assert imaginary_q > 1e-7
    for low, high in ((0., 0.), (.05, .95)):
        start = time.perf_counter()
        c, n, e, ncas, nelec = uno.sort_orbitals(
            mol, c0.copy(), n0.copy(), e0.copy(), cas_list=list(range(112, 142)),
            do_loc=True, split_low=low, split_high=high, iprint=0,
        )
        assert (ncas, nelec) == (30, 14)
        t = c0.conj().T @ s @ c
        np.testing.assert_allclose(abs(t) ** 2 @ np.ones(c.shape[1]),
                                   np.ones(c.shape[1]), atol=1e-7)
        np.testing.assert_allclose(n, n0 @ abs(t) ** 2, atol=1e-8)
        np.testing.assert_allclose(e, e0 @ abs(t) ** 2, atol=1e-8)
        np.testing.assert_allclose(t[112:142, 112:142].conj().T
                                   @ t[112:142, 112:142], np.eye(30), atol=1e-7)
        assert abs(n[112:142].sum() - 14) < 1e-7
        objective = uno._objective(uno._population(mol, c[:, 112:142], 0))
        assert objective >= initial_objective - 1e-8
        print(f"UNF3 {'simple' if low == 0 else 'split'} PM: "
              f"L={initial_objective:.9f}->{objective:.9f}, "
              f"max |Im Qij|={imaginary_q:.3e}, elapsed={time.perf_counter() - start:.2f}s")
