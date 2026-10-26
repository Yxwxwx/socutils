# SPDX-License-Identifier: GPL-3.0-or-later
"""Small independent checks of exact-reference preparation and diagnostics."""
from types import SimpleNamespace

import numpy as np
import pytest
from pyscf.fci import cistring, fci_dhf_slow
from pyblock2.driver.core import DMRGDriver, SymmetryTypes
from msficnevpt2.ag_audit import fixed_n_coefficients, extract_root
from nevpt2_mps_response.f_atom import active_action_batch, _active_action_links
from test_x2cficnevpt2 import _fic_fixture, _fic_basis_vectors, _active_hamiltonian_matrix
from test_x2cscnevpt2_wick import _embed_active_state, _hamiltonian_action
from msficnevpt2.r_modes import RDirections, mode_contributions, doublet_decomposition


@pytest.mark.parametrize("nelec", [2, 3, 4])
@pytest.mark.parametrize("dot", [1, 2])
def test_sgf_tensor_contraction_matches_all_native_coefficients(tmp_path, nelec, dot):
    driver = DMRGDriver(scratch=str(tmp_path), symm_type=SymmetryTypes.SGFCPX,
                        stack_mem=128 * 2**20, n_threads=1)
    try:
        driver.initialize_system(n_sites=6, n_elec=nelec, orb_sym=[0] * 6)
        mps = driver.get_random_mps(tag="CHECK", bond_dim=16, dot=dot)
        vector = extract_root(driver, mps, 0, nelec)
        strings = cistring.make_strings(range(6), nelec)
        occupations = ((strings[:, None] >> np.arange(6)) & 1).astype(np.uint8)
        native = np.asarray(driver.get_csf_coefficients(
            mps, cutoff=0.0, given_dets=occupations,
            max_print=0, fci_conv=True, iprint=0,
        )[1])
        np.testing.assert_allclose(vector, native, atol=1e-12, rtol=0)
    finally:
        driver.finalize()


def test_coefficient_diagnostic_rejects_fused_or_oversized_input():
    with pytest.raises(ValueError, match="unfused"):
        fixed_n_coefficients(SimpleNamespace(tensors=[None] * 4), 2)
    with pytest.raises(ValueError, match="24 spinors"):
        fixed_n_coefficients(SimpleNamespace(tensors=[None] * 25), 2)


@pytest.mark.parametrize("compact", [False, True])
def test_sgf_residual_action_matches_pyscf_reference(compact):
    from socutils.mrpt.nevpt2_utils import _compact_wick_eris
    case = _fic_fixture(ncas=6, nelec=4)
    eris = _compact_wick_eris(case["eris"]) if compact else case["eris"]
    chem = eris.get_phys("AAAA").transpose(0, 2, 1, 3)
    np.testing.assert_allclose(chem, case["eris"].get_chem("AAAA"), atol=0, rtol=0)
    eri = fci_dhf_slow.absorb_h1e(eris.get_h1eff("AA"), chem, 6, 4, 0.5)
    rng = np.random.default_rng(613)
    vectors = rng.normal(size=(2, 15)) + 1j * rng.normal(size=(2, 15))
    actual = active_action_batch(eri, vectors, 6, 4)
    expected = np.array([fci_dhf_slow.contract_2e(eri, v, 6, 4) for v in vectors])
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
    _active_action_links.cache_clear()


@pytest.mark.parametrize("invalid", [None, "stale", "shape"])
def test_r_direct_restart_reuses_spectrum_only_for_unchanged_inputs(tmp_path, monkeypatch, invalid):
    import json
    import os
    from msficnevpt2 import r_modes
    for name in ("prepared.pkl", "S1_2.json"):
        (tmp_path / name).write_text("unused")
    np.savez(tmp_path / "reference_vectors.npz", vectors=np.eye(4, 2), actions=np.eye(4, 2))
    report = dict(selected_modes=[1, 2], selected_connected_eigenvalues=[3., 4.])
    (tmp_path / "r_modes.json").write_text(json.dumps(report))
    np.savez(tmp_path / "r_mode_coefficients.npz", coefficients=np.eye(3 if invalid == "shape" else 4, 2),
             modes=report["selected_modes"], eigenvalues=report["selected_connected_eigenvalues"])
    if invalid == "stale":
        latest = max(path.stat().st_mtime_ns for path in tmp_path.iterdir()) + 10**9
        os.utime(tmp_path / "prepared.pkl", ns=(latest, latest))
    prepared = SimpleNamespace(model_roots=(0, 1), active_reference=np.eye(2),
                               eris=SimpleNamespace(ncas=5),
                               classes={"r": SimpleNamespace(metric=np.eye(4))})
    monkeypatch.setattr(r_modes.pickle, "load", lambda handle: prepared)
    def unexpected_spectrum(*args):
        pytest.fail("must not recompute the cached spectrum")
    monkeypatch.setattr(r_modes, "spectral_audit", unexpected_spectrum)
    class ReachedDirect(Exception):
        pass
    def direct(*args):
        raise ReachedDirect
    monkeypatch.setattr(r_modes, "RDirections", direct)
    with pytest.raises(ValueError if invalid else ReachedDirect):
        r_modes.main(tmp_path, resume_direct=True)


def test_selected_r_directions_and_source_match_direct_full_fock_space():
    case = _fic_fixture(ncas=5, nelec=3)
    eris = case["eris"]
    incoming = cistring.make_strings(range(5), 3)
    outgoing = cistring.make_strings(range(5), 2)
    rng = np.random.default_rng(5061)
    vectors = np.linalg.qr(rng.normal(size=(10, 2)) + 1j*rng.normal(size=(10, 2)))[0]
    oracle = RDirections(vectors, 5, 3)
    c = rng.normal(size=(110, 3)) + 1j*rng.normal(size=(110, 3))
    for channel in range(2):
        full = []
        for v in vectors.T:
            active = np.zeros(1 << 5, complex)
            active[incoming] = v
            full.append(_embed_active_state(active, eris.ncore, 5, eris.nvirt))
        basis = np.concatenate([_fic_basis_vectors(v, eris, "r", (channel,)) for v in full], axis=1)
        target = ((1 << eris.ncore) - 1) | (outgoing << eris.ncore) | (1 << (eris.nocc + channel))
        np.testing.assert_allclose(oracle.apply(c), basis[target] @ c, atol=3e-14)
        h_vectors = np.column_stack([
            _hamiltonian_action(v, case["h1e"], case["w"]) for v in full])
        source = oracle.source(eris, channel)
        np.testing.assert_allclose(source, h_vectors[target], atol=2e-13)
        np.testing.assert_allclose(oracle.apply(c).conj().T @ source,
                                   c.conj().T @ basis.conj().T @ h_vectors, atol=2e-12)


@pytest.mark.parametrize("shift", [0.0, 0.2])
def test_spectral_mode_sum_equals_explicit_response(shift):
    rng = np.random.default_rng(917)
    g = rng.normal(size=(3, 9, 2)) + 1j*rng.normal(size=(3, 9, 2))
    d = rng.uniform(0.5, 5, size=g.shape) + shift
    y, n = mode_contributions(g, d, shift)
    t = -g / d
    expected_y = sum(source.conj().T @ response for source, response in zip(g, t))
    expected_n = sum(response.conj().T @ response for response in t)
    np.testing.assert_allclose(y.sum(axis=(0, 1)), 0.5 * (expected_y + expected_y.conj().T), atol=1e-13)
    np.testing.assert_allclose(n.sum(axis=(0, 1)), -shift * expected_n, atol=1e-13)


@pytest.mark.parametrize("nroots", [2, 4, 12])
@pytest.mark.parametrize("use_guess", [False, True])
def test_vectorized_pyscf_exact_ci_matches_independent_complex_hamiltonian(nroots, use_guess, monkeypatch):
    from msficnevpt2.ag_exact_reference import exact_reference, BatchedSGFFCI
    calls = []
    native_kernel = BatchedSGFFCI.kernel
    def kernel(solver, *args, **kwargs):
        calls.append(solver.conv_tol_residual)
        return native_kernel(solver, *args, **kwargs)
    monkeypatch.setattr(BatchedSGFFCI, "kernel", kernel)
    case = _fic_fixture(ncas=6, nelec=3)
    strings = cistring.make_strings(range(6), 3)
    matrix = _active_hamiltonian_matrix(case["active_h"], case["active_w"])[np.ix_(strings, strings)]
    expected, _ = np.linalg.eigh(matrix)
    rng = np.random.default_rng(712)
    guess = np.linalg.qr(rng.normal(size=(len(strings), nroots)) + 1j*rng.normal(size=(len(strings), nroots)))[0]
    converged, energies, vectors, actions, residuals = exact_reference(
        case["active_h"], case["eris"].get_chem("AAAA"), guess if use_guess else None,
        nroots=nroots, nelec=3, residual_tol=1e-11, threads=1, max_memory=512)
    assert np.all(converged)
    assert calls == ([1e-11] if use_guess else [1e-9, 1e-11])
    np.testing.assert_allclose(energies, expected[:nroots], atol=1e-12, rtol=0)
    np.testing.assert_allclose(actions, matrix @ vectors, atol=1e-12, rtol=0)
    assert max(residuals) < 1e-10
    _active_action_links.cache_clear()


def test_exact_spectra_preserves_common_partition_for_six_root_model():
    from msficnevpt2.exact_spectra import model_prepared, spectrum
    partition = dict(eris=object(), mo_coeff=np.eye(8), mo_energy=np.arange(8.),
                     sa_density=np.eye(4), ecore=-100., external_fock_defect=1e-12,
                     fingerprint="common")
    energies = np.repeat([-3., -2., -1.], [2, 6, 4])
    roots = tuple(range(2, 8))
    active = np.diag(energies[list(roots)])
    prepared = model_prepared(partition, roots, {}, active, np.eye(6), {}, energies, "ci")
    for name in ("eris", "mo_coeff", "mo_energy", "sa_density"):
        assert getattr(prepared, name) is partition[name]
    assert prepared.model_roots == roots
    np.testing.assert_allclose(prepared.reference, active - 100 * np.eye(6), atol=0, rtol=0)
    report = spectrum(energies - 100)
    assert report["d_splitting_cm_inverse"] == pytest.approx(219474.6313705)
    assert max(report["spreads_cm_inverse"].values()) == 0
    with pytest.raises(AssertionError):
        model_prepared(partition, roots, {}, active + 0.01, np.eye(6), {}, energies, "ci")


@pytest.mark.parametrize("symbol", ["Cu", "Au"])
def test_exact_spectra_loads_the_original_hf_checkpoint(tmp_path, monkeypatch, symbol):
    from msficnevpt2 import exact_spectra
    source = tmp_path / symbol / "kr_m2000_from_hf"
    (source / "dmrg_state").mkdir(parents=True)
    (source / "dmrg_state/reference.npz").write_bytes(b"snapshot")
    directory = source / "ms_nevpt2_exact"
    directory.mkdir()
    expected = (source.parent / "unc_ano_rcc_trials/hf_kr/hf.chk"
                if symbol == "Cu" else source / "hf_kr/hf.chk")
    class ReachedCheckpoint(Exception):
        pass
    def load(path):
        assert path == str(expected)
        raise ReachedCheckpoint
    monkeypatch.setattr(exact_spectra.scf.chkfile, "load_scf", load)
    with pytest.raises(ReachedCheckpoint):
        exact_spectra.common_partition(directory, {"hamiltonian_sha256": np.array("sha")}, 1)


def test_exact_spectra_prepare_only_never_builds_trdms(tmp_path, monkeypatch):
    import hashlib
    import json
    from msficnevpt2 import exact_spectra
    source = tmp_path / "Cu/kr_m2000_from_hf"
    state = source / "dmrg_state"
    state.mkdir(parents=True)
    settings = dict(symbol="Cu", ncas=22, nelec=11, nroots=12, M=2000,
                    orbital_kramers=True, cd_df=False, final_one_site=False, nuclear_model="point")
    (state / "READY.json").write_text(json.dumps(dict(settings=settings,
        hamiltonian_sha256="sha", converged=True, solver_converged=True)))
    np.savez(state / "reference.npz", hamiltonian_sha256="sha", weights=exact_spectra.WEIGHTS)
    directory = source / "ms_nevpt2_exact"
    directory.mkdir()
    (directory / "exact_reference_vectors.npz").write_bytes(b"exact")
    ci_hash = hashlib.sha256(b"exact").hexdigest()
    (directory / "exact_mps").mkdir()
    (directory / "exact_mps/READY.json").write_text(json.dumps(
        dict(source_ci_sha256=ci_hash, roots=list(range(12)))))
    monkeypatch.setattr(exact_spectra, "common_partition", lambda *args: {})
    monkeypatch.setattr(exact_spectra, "reference_cache", lambda *args: (np.eye(12), np.arange(12.)))
    monkeypatch.setattr(exact_spectra.exact_mps, "main", lambda *args: None)
    def unexpected(*args, **kwargs):
        pytest.fail("prepare-only must not allocate TRDMs or solve MS-NEVPT2")
    monkeypatch.setattr(exact_spectra.ms, "build_msfic_classes", unexpected)
    monkeypatch.setattr(exact_spectra.ms, "solve_msfic", unexpected)
    exact_spectra.main(directory, prepare_only=True)
    assert not (directory / "S1_2.json").exists()


@pytest.mark.parametrize("nranks", [3, 4])
def test_retained_s_trdms_are_independent_restartable_and_uncontracted(tmp_path, monkeypatch, nranks):
    import json
    from pathlib import Path
    from msficnevpt2 import save_s_rdms
    original_read = Path.read_text
    monkeypatch.setattr(Path, "read_text", lambda path, *args, **kwargs:
        "MemAvailable: 2000000000 kB\n" if str(path) == "/proc/meminfo"
        else original_read(path, *args, **kwargs))
    calls = []
    def provider(solver, bra, ket):
        calls.append((bra, ket))
        return tuple(np.full((2,) * (2 * rank), 10 * bra + ket + rank * 1j,
                             dtype=np.complex128) for rank in range(1, nranks + 1))
    def unexpected(*args, **kwargs):
        pytest.fail("retained TRDM preparation must not contract or solve PT2")
    monkeypatch.setattr(save_s_rdms, "make_transition_dm1234", provider)
    monkeypatch.setattr(save_s_rdms.u, "_make_transition_dm123", provider)
    monkeypatch.setattr(save_s_rdms.ms, "build_msfic_classes", unexpected)
    monkeypatch.setattr(save_s_rdms.ms, "solve_msfic", unexpected)
    inputs = dict(source_ci_sha256="ci", hamiltonian_sha256="h", ncas=2, nelec=1,
                  model_roots=[0, 1], raw_sgf=True)
    if nranks == 3:
        inputs.update(nranks=3, ordered_pairs=[[b, k] for b in (0, 1) for k in (0, 1)])
    directory = tmp_path / "S1_2_rdms"
    save_s_rdms.save_pairs(object(), directory, inputs)
    assert calls == [(0, 0), (0, 1), (1, 0), (1, 1)]
    for bra, ket in calls:
        arrays = save_s_rdms.load_saved_pair(directory, "ci", bra, ket)
        for rank, dm in enumerate(arrays, 1):
            assert not dm.flags.writeable
            np.testing.assert_array_equal(dm, 10 * bra + ket + rank * 1j)
            dm._mmap.close()
    assert json.loads((directory / "READY.json").read_text())["tensor_contraction"] is False
    before = (directory / f"bra-0_ket-1/dm{nranks}.npy").stat().st_mtime_ns
    monkeypatch.setattr(save_s_rdms, "make_transition_dm1234", unexpected)
    monkeypatch.setattr(save_s_rdms.u, "_make_transition_dm123", unexpected)
    save_s_rdms.save_pairs(object(), directory, inputs)
    assert (directory / f"bra-0_ket-1/dm{nranks}.npy").stat().st_mtime_ns == before
    if nranks == 3:
        fourth_calls = []
        def fourth(solver, bra, ket):
            fourth_calls.append((bra, ket))
            return np.full((2,) * 8, 10 * bra + ket + 4j, complex)
        monkeypatch.setattr(save_s_rdms.u, "make_transition_rdm4", fourth)
        full_inputs = {key: value for key, value in inputs.items() if key not in ("nranks", "ordered_pairs")}
        save_s_rdms.save_pairs(SimpleNamespace(lower_directory=directory), tmp_path / "full", full_inputs)
        assert fourth_calls == calls
        arrays = save_s_rdms.load_saved_pair(tmp_path / "full", "ci", 0, 1)
        for rank, dm in enumerate(arrays, 1):
            np.testing.assert_array_equal(dm, 1 + rank * 1j)
            dm._mmap.close()
    with pytest.raises(ValueError, match="reference mismatch"):
        save_s_rdms.load_saved_pair(directory, "another-ci", 0, 1)
    with pytest.raises(ValueError, match="another exact reference"):
        save_s_rdms.save_pairs(object(), directory, {**inputs, "source_ci_sha256": "other"})
    np.save(directory / "bra-0_ket-1/dm2.npy", np.zeros((2, 2), complex))
    with pytest.raises(ValueError, match="shape or dtype mismatch"):
        save_s_rdms.load_saved_pair(directory, "ci", 0, 1)


def test_lower_preparation_uses_all_56_model_pairs_without_four_rdms(tmp_path, monkeypatch):
    import hashlib
    import json
    from msficnevpt2 import save_s_rdms
    from msficnevpt2.exact_spectra import MANIFOLDS
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
    (tmp_path / "exact_mps").mkdir()
    (tmp_path / "exact_reference_vectors.npz").write_bytes(b"ci")
    metadata = dict(source_ci_sha256=hashlib.sha256(b"ci").hexdigest(),
        hamiltonian_sha256="h", ncas=22, nelec=11, residual_norms=[1e-12] * 12,
        roots=[dict(root=i, tag=f"ROOT_{i}", files=[]) for i in range(12)])
    (tmp_path / "exact_mps/READY.json").write_text(json.dumps(metadata))
    (tmp_path / "settings.json").write_text(json.dumps(dict(
        hamiltonian_sha256="h", manifolds=MANIFOLDS)))
    monkeypatch.setattr(save_s_rdms.lib.param, "TMPDIR", str(tmp_path))
    class Driver:
        def __init__(self, **kwargs):
            assert kwargs["stack_mem"] == 128000 * 1000000
        def initialize_system(self, **kwargs):
            assert kwargs["n_sites"] == 22 and kwargs["n_elec"] == 11
        def load_mps(self, tag):
            return tag
        def finalize(self):
            pass
    monkeypatch.setattr(save_s_rdms, "DMRGDriver", Driver)
    captured = []
    def save(solver, output, inputs):
        assert len(solver.kets) == 12
        assert output == tmp_path / "all_model_rdms123"
        assert inputs["nranks"] == 3
        pairs = inputs["ordered_pairs"]
        assert len(pairs) == len(set(map(tuple, pairs))) == 56
        assert [0, 2] not in pairs and [2, 0] not in pairs
        captured.append(inputs)
    monkeypatch.setattr(save_s_rdms, "save_pairs", save)
    save_s_rdms.main(tmp_path, lower_only=True)
    assert len(captured) == 1


def test_doublet_decomposition_preserves_cancellation_and_shift_terms():
    reference = np.eye(2) * -37.0
    one = np.array([[0.1, 0.2j], [-0.2j, -0.1]])
    norms = {key: np.eye(2) * 0.3 for key in ("r", "i")}
    result = doublet_decomposition(reference, {"r": one, "i": -one}, norms, 0.2)
    np.testing.assert_allclose(result["heff"], reference, atol=1e-14)
    assert result["total_spread_cm_inverse"] < 1e-8
    for key, correction in (("r", one), ("i", -one)):
        parts = result["classes"][key]
        np.testing.assert_allclose(parts["source_response"] + parts["shift_norm_correction"], correction)


@pytest.mark.parametrize("batch_size", [3, 7, 128])
def test_native_batched_exact_ci_import_and_persistent_reload(tmp_path, batch_size):
    from pyblock2.algebra.io import MPSTools
    from msficnevpt2.exact_mps import import_coefficients, copy_checkpoint
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    output = tmp_path / "checkpoint"
    output.mkdir()
    driver = DMRGDriver(scratch=str(scratch), symm_type=SymmetryTypes.SGFCPX,
                        stack_mem=128 * 2**20, n_threads=1)
    rng = np.random.default_rng(628)
    vector = rng.normal(size=20) + 1j*rng.normal(size=20)
    vector /= np.linalg.norm(vector)
    try:
        driver.initialize_system(n_sites=6, n_elec=3, orb_sym=[0] * 6)
        mps = import_coefficients(driver, vector, 3, tag="EXACT_TEST", batch_size=batch_size)
        actual = fixed_n_coefficients(MPSTools.from_block2(mps), 3)
        np.testing.assert_allclose(actual, vector, atol=1e-12, rtol=0)
        driver.adjust_mps(mps, dot=2)
        files = copy_checkpoint(driver, mps, output)
        assert "EXACT_TEST-mps_info.bin" in files
        assert not list(scratch.glob("*NEVPT2-*"))
        driver.scratch = str(output)
        loaded = driver.load_mps("EXACT_TEST")
        driver.adjust_mps(loaded, dot=1)
        roundtrip = fixed_n_coefficients(MPSTools.from_block2(loaded), 3)
        np.testing.assert_allclose(roundtrip, vector, atol=1e-12, rtol=0)
        from socutils.mrpt import nevpt2_utils as u
        driver.adjust_mps(loaded, dot=2)
        solver = SimpleNamespace(driver=driver, kets=[loaded], ncas=6, nelecas=3,
                                 npdm_site_type=2, npdm_cutoff=1e-24)
        native_dm1 = u.make_rdm1(solver, 0)
        np.testing.assert_allclose(native_dm1, fci_dhf_slow.make_rdm1(vector, 6, 3),
                                   atol=1e-12, rtol=0)
    finally:
        driver.finalize()


def test_time_reverse_diagnostic_supports_22_spinors_without_projecting_states():
    from nevpt2_mps_response.uc_audit import time_reverse_coefficients
    from msficnevpt2.ag_time_reversal import model_closure
    rng = np.random.default_rng(5402)
    t = np.kron(np.eye(11), np.array([[0.0, 1.0], [-1.0, 0.0]]))
    v = rng.normal(size=22) + 1j*rng.normal(size=22)
    v /= np.linalg.norm(v)
    tv = time_reverse_coefficients(t, v, 1)
    np.testing.assert_allclose(tv, t @ v.conj(), atol=1e-12, rtol=0)
    np.testing.assert_allclose(time_reverse_coefficients(t, tv, 1), -v, atol=1e-12, rtol=0)
    pair = np.column_stack([v, tv])
    reversed_pair = time_reverse_coefficients(t, pair, 1)
    audit = model_closure(pair, reversed_pair)
    np.testing.assert_allclose(audit["leakage_norms"], 0, atol=1e-12, rtol=0)
    broken = pair.copy()
    broken[:, 1] += 0.01 * rng.normal(size=22)
    broken_audit = model_closure(broken, time_reverse_coefficients(t, broken, 1))
    assert min(broken_audit["leakage_norms"]) > 1e-3


def test_frozen_partition_symmetry_diagnostic_does_not_repair_its_inputs():
    from msficnevpt2.r_modes import partition_symmetry
    from socutils.dmrg.kramers import time_reverse_rdm1
    rng = np.random.default_rng(430)
    j = np.array([[0, 1], [-1, 0]])
    u = np.linalg.qr(rng.normal(size=(4, 4)) + 1j*rng.normal(size=(4, 4)))[0]
    active_t = u.conj().T @ np.kron(np.eye(2), j) @ u.conj()
    t = np.zeros((8, 8), complex)
    t[:2, :2], t[2:6, 2:6], t[6:, 6:] = j, active_t, j
    d = rng.normal(size=(4, 4)) + 1j*rng.normal(size=(4, 4))
    d += d.conj().T
    d = 0.5 * (d + time_reverse_rdm1(active_t, d))
    p = SimpleNamespace(eris=SimpleNamespace(ncore=2, nocc=6), sa_density=d,
                        mo_energy=np.array([-1., -1., 0., 0., 0., 0., 0.3, 0.3]))
    report = partition_symmetry(p, t)
    assert report["sa_density_time_reverse_defect"] < 1e-13
    assert report["external"]["diagonal_fock_time_reverse_defect"] < 1e-13
    p.sa_density[0, 0] += 0.01
    p.mo_energy[-1] += 0.02
    before_d, before_e = p.sa_density.copy(), p.mo_energy.copy()
    report = partition_symmetry(p, t)
    assert report["sa_density_time_reverse_defect"] > 1e-3
    assert report["external"]["diagonal_fock_time_reverse_defect"] == pytest.approx(0.02)
    np.testing.assert_array_equal(p.sa_density, before_d)
    np.testing.assert_array_equal(p.mo_energy, before_e)


def test_exact_control_keeps_common_sa_partition_and_core_energy():
    from socutils.mrpt import x2cmsficnevpt2 as ms
    from msficnevpt2.ag_exact_msfic import frozen_control
    h = np.diag([-2., -1.])
    p = ms.MSFICPrepared(object(), np.eye(8), np.arange(8.), np.eye(4),
                         (0, 1), np.eye(2), h, h - 100 * np.eye(2), {"r": object()},
                         {"sa_roots": tuple(range(12)), "sa_weights": np.full(12, 1/12),
                          "external_fock_defect": 1e-12, "fingerprint": "original"})
    classes, new_h = {"r": object()}, np.eye(2) * -2.1
    actual = frozen_control(p, classes, new_h, np.eye(2), {})
    for name in ("eris", "mo_coeff", "mo_energy", "sa_density"):
        assert getattr(actual, name) is getattr(p, name)
    assert actual.classes is classes and p.classes is not classes
    np.testing.assert_allclose(actual.reference, new_h - 100 * np.eye(2))
    assert actual.diagnostics["baseline_fingerprint"] == "original"
    assert "fingerprint" not in actual.diagnostics
    p.reference[0, 1] = 0.1
    with pytest.raises(AssertionError):
        frozen_control(p, classes, new_h, np.eye(2), {})


def test_exact_native_cache_raw_transition_ranks_1_to_4_match_independent_fock_oracle(tmp_path):
    from msficnevpt2.exact_mps import import_coefficients
    from test_x2cmsficnevpt2 import transition_pdms
    from socutils.mrpt import make_transition_dm1234
    driver = DMRGDriver(scratch=str(tmp_path), symm_type=SymmetryTypes.SGFCPX,
                        stack_mem=128 * 2**20, n_threads=1)
    rng = np.random.default_rng(612)
    vectors = np.linalg.qr(rng.normal(size=(5, 2)) + 1j*rng.normal(size=(5, 2)))[0]
    try:
        driver.initialize_system(n_sites=5, n_elec=4, orb_sym=[0] * 5)
        kets = [import_coefficients(driver, v, 4, tag=f"EXACT_RDM_{i}", batch_size=3)
                for i, v in enumerate(vectors.T)]
        for ket in kets:
            driver.adjust_mps(ket, dot=2)
        solver = SimpleNamespace(driver=driver, kets=kets, ncas=5, nelecas=4,
                                 npdm_site_type=0, npdm_cutoff=1e-24)
        full = np.zeros((1 << 5, 2), complex)
        full[cistring.make_strings(range(5), 4)] = vectors
        for bra in range(2):
            for ket in range(2):
                expected = transition_pdms(full[:, bra], full[:, ket], 5)
                actual = make_transition_dm1234(solver, bra, ket)
                for rank in range(4):
                    np.testing.assert_allclose(actual[rank], expected[rank], atol=2e-12, rtol=0)
    finally:
        driver.finalize()
