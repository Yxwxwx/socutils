# SPDX-License-Identifier: GPL-3.0-or-later
"""Exact complex-spinor regression for bounded-memory NEVPT2 paths."""

from types import SimpleNamespace

import numpy as np
import pytest
from pyscf import gto, lib
from pyscf.ao2mo import nrr_outcore

from socutils.scf import spinor_hf
from socutils.mrpt import nevpt2_utils as u, x2cscnevpt2 as sc, x2cficnevpt2 as fic
from socutils.mrpt.nevpt2_eris import wick_eris_from_mc
from test_x2cscnevpt2_wick import _active_ground_state, _raw_active_rdms


@pytest.fixture(scope="module")
def complex_case():
    lib.num_threads(1)
    mol = gto.M(atom="H 0 0 0; H 0.2 0 1.1; H 0 1.5 0; H 1.4 0.1 0.3",
                basis="sto-3g", verbose=0)
    mf = spinor_hf.SCF(mol)
    e, v = np.linalg.eigh(mf.get_ovlp())
    rng = np.random.default_rng(7151)
    q, _ = np.linalg.qr(rng.normal(size=v.shape) + 1j * rng.normal(size=v.shape))
    mo = (v / np.sqrt(e)) @ q
    mc = SimpleNamespace(mol=mol, _scf=mf, ncore=2, ncas=4, nelecas=2,
                         verbose=0, stdout=mol.stdout, get_hcore=mf.get_hcore,
                         mo_coeff=mo, frozen=0, e_tot=0.0,
                         fcisolver=SimpleNamespace(nelecas=2, nroots=1))
    dense = u._dense_eris_from_mc(mc, mo)
    a = slice(2, 6)
    _, state = _active_ground_state(dense.h1eff[a, a], dense.get_phys("AAAA"), 2)
    return mc, dense, _raw_active_rdms(state, 4)


def test_direct_complex_blocks_and_core_jk_match_dense(complex_case, monkeypatch):
    mc, dense, _ = complex_case
    original = nrr_outcore.general
    calls = []

    def block_only(mol, coeffs, *args, **kwargs):
        dims = tuple(c.shape[1] for c in coeffs)
        assert dims != (mc.mo_coeff.shape[1],) * 4
        assert kwargs["motype"] == "j-spinor"
        calls.append(dims)
        return original(mol, coeffs, *args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("full MO ERI construction was reached")

    monkeypatch.setattr(nrr_outcore, "general", block_only)
    monkeypatch.setattr(nrr_outcore, "full_iofree", forbidden)
    blocks = wick_eris_from_mc(mc, mc.mo_coeff, max_memory=100, ioblk_size=4)
    assert calls and not hasattr(blocks, "pppp")
    assert blocks.nbytes < dense.nbytes
    assert blocks.symmetry_diagnostics["full_mo_eri_built"] is False
    for key in u._W_KEYS:
        np.testing.assert_allclose(blocks.get_phys(key), dense.get_phys(key), atol=2e-12)
    for key in u._H1_KEYS:
        np.testing.assert_allclose(blocks.get_h1eff(key), dense.get_h1eff(key), atol=2e-12)


@pytest.mark.parametrize("ncore,ncas", [(0, 4), (4, 4)])
def test_block_transform_empty_partitions(complex_case, ncore, ncas):
    mc, _, _ = complex_case
    mc = SimpleNamespace(**{**vars(mc), "ncore": ncore, "ncas": ncas})
    dense = u._dense_eris_from_mc(mc, mc.mo_coeff)
    blocks = wick_eris_from_mc(mc, mc.mo_coeff, max_memory=100, ioblk_size=4)
    for key in u._W_KEYS:
        np.testing.assert_allclose(blocks.get_phys(key), dense.get_phys(key), atol=2e-12)
    for key in u._H1_KEYS:
        np.testing.assert_allclose(blocks.get_h1eff(key), dense.get_h1eff(key), atol=2e-12)


@pytest.mark.parametrize("backend", ["numpy", "pytblis"])
def test_sc_tiled_energy_norms_gaps_and_audits(complex_case, backend):
    _, dense, pdms = complex_case
    kwargs = dict(contraction_backend=backend, return_diagnostics=True)
    full = sc._evaluate_wick_subspaces(dense, pdms, [-30., -20.], [20., 30.], **kwargs)
    tiled = sc._evaluate_wick_subspaces(dense, pdms, [-30., -20.], [20., 30.],
                                        work_memory=16, **kwargs)
    for key in sc.SUBSPACE_ORDER:
        for a, b in zip(full[:3], tiled[:3]):
            np.testing.assert_allclose(a[key], b[key], atol=2e-12)
        for name in ("ordered_dimension", "retained_dimension", "discarded_zero_norm_dimension"):
            assert full[3][key][name] == tiled[3][key][name]
        for name in ("imaginary_energy_l1", "real_projection_shift_l1"):
            np.testing.assert_allclose(full[3][key]["one_sided_denominator"][name],
                                       tiled[3][key]["one_sided_denominator"][name], atol=2e-12)


@pytest.mark.parametrize("backend", ["numpy", "pytblis"])
def test_fic_shares_matrices_and_tiles_rhs(complex_case, backend, monkeypatch):
    _, dense, pdms = complex_case
    options = dict(contraction_backend=backend, return_diagnostics=True)
    full = fic._evaluate_fic_subspaces(dense, pdms, [-30., -20.], [20., 30.], **options)
    execute = fic._execute_tensor
    shapes = []

    def bounded(*args, **kwargs):
        result = execute(*args, **kwargs)
        shapes.append((args[1], result.shape))
        return result

    monkeypatch.setattr(fic, "_execute_tensor", bounded)
    tiled = fic._evaluate_fic_subspaces(dense, pdms, [-30., -20.], [20., 30.],
                                       work_memory=16, **options)
    for key in fic.SUBSPACE_ORDER:
        np.testing.assert_allclose(tiled[0][key], full[0][key], atol=2e-12)
        assert tiled[1][key]["active_matrices_shared"]
    assert all(shape[0] == 1 for name, shape in shapes)


@pytest.mark.parametrize("method", [sc.WickX2CSCNEVPT2, fic.WickX2CFICNEVPT2])
def test_default_driver_never_builds_full_eris(complex_case, monkeypatch, method):
    mc, dense, pdms = complex_case
    energies = np.array([-30., -20., 0., 0., 0., 0., 20., 30.])
    reference = method(mc)
    reference.canonicalized = True
    reference.mo_energy = energies
    e_dense = reference.kernel(pdms=pdms, eris=dense, compact_eris=False,
                                contraction_backend="numpy")

    def forbidden(*args, **kwargs):
        pytest.fail("default NEVPT2 called the dense AO2MO path")

    monkeypatch.setattr(nrr_outcore, "full_iofree", forbidden)
    pt = method(mc)
    pt.canonicalized = True
    pt.mo_energy = energies
    pt.integral_max_memory = 100
    e_blocks = pt.kernel(pdms=pdms, contraction_backend="numpy")
    np.testing.assert_allclose(e_blocks, e_dense, atol=2e-12)
    assert isinstance(pt.eris, u._WickERIBlocks)
    # Both public adapters must also accept cached compact ERIs.
    np.testing.assert_allclose(pt.kernel(pdms=pdms, eris=pt.eris,
                                        contraction_backend="numpy"), e_dense, atol=2e-12)


def test_compact_input_can_be_rotated_without_full_integrals(complex_case):
    _, dense, _ = complex_case
    compact = u._compact_wick_eris(dense)
    rng = np.random.default_rng(105)
    rotation = np.eye(dense.nmo, dtype=complex)
    for sl in (slice(0, dense.ncore), slice(dense.nocc, dense.nmo)):
        size = rotation[sl, sl].shape[0]
        q, _ = np.linalg.qr(rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size)))
        rotation[sl, sl] = q
    full = u._rotate_eris(dense, rotation)
    rotated = u._rotate_wick_eris(compact, rotation)
    assert not hasattr(rotated, "pppp")
    assert rotated.symmetry_diagnostics is compact.symmetry_diagnostics
    for key in u._H1_KEYS:
        np.testing.assert_allclose(rotated.get_h1eff(key), full.get_h1eff(key), atol=2e-12)
    for key in u._W_KEYS:
        np.testing.assert_allclose(rotated.get_phys(key), full.get_phys(key), atol=2e-12)
        assert not np.shares_memory(rotated.get_phys(key), compact.get_phys(key))


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_chunked_roundoff_audit_rejects_nonfinite_values(value):
    data = np.zeros((3, 3), dtype=complex)
    data[-1, -1] = value
    assert not np.isfinite(u._maximum_abs_chunked(data, 16))
    assert not np.isfinite(u._maximum_abs_relation(
        data, np.zeros_like(data), sign=-1, work_memory=16))
    with pytest.raises(ValueError, match="maximum absolute value"):
        u._ao2mo_roundoff_policy(data, accumulation_length=3,
                                 contraction_stages=4, roundoff_factor=1.0)


def test_sc_tile_audit_merges_global_indices_and_additive_gates():
    options = dict(root=0, subspace="i", gap_atol=1e-3, gap_rtol=0.,
                   numerator_noise_allowance=1e-3, energy_imag_l1_tol=1.5e-5,
                   projection_shift_l1_tol=1e-3, denominator_tol=1e-12,
                   enforce=False)
    norm = np.ones(1)
    commutator = np.array([4e-5j])
    gap = np.array([2. + 4e-5j])
    _, first = sc._audit_one_sided_si_denominator(
        norm, commutator, gap, np.ones(1, dtype=bool), **options)
    _, second = sc._audit_one_sided_si_denominator(
        norm, commutator, gap, np.ones(1, dtype=bool), **options)
    assert first["strict_si_compatible"] and second["strict_si_compatible"]
    merged = sc._merge_sc_diagnostics(first, second)
    assert not merged["strict_si_compatible"]
    assert not merged["imaginary_energy_l1_gate_passed"]

    empty = sc._scalar_reality_diagnostics(np.zeros(1), np.zeros(1, dtype=bool),
                                            atol=1e-12, rtol=0.)
    nonempty = sc._scalar_reality_diagnostics(np.ones(1), np.ones(1, dtype=bool),
                                               atol=1e-12, rtol=0.)
    sc._offset_sc_diagnostics(nonempty, 4)
    merged = sc._merge_sc_diagnostics(empty, nonempty)
    assert merged["maximum_imaginary_index"] == [4]
    assert merged["maximum_ratio_index"] == [4]
    assert merged["real_part_at_maximum"] == 1.
