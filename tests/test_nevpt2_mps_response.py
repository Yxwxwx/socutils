# SPDX-License-Identifier: GPL-3.0-or-later
"""The Block2-style optional branch leaves six contracted classes unchanged."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest
from pyscf import gto
from socutils.dmrg import DMRGCI
from socutils.mrpt import nevpt2_mps_response as response
from socutils.mrpt import nevpt2_utils as u
from socutils.mrpt import x2cficnevpt2 as fic
from socutils.mrpt import x2cscnevpt2 as sc
from test_x2cficnevpt2 import _active_hamiltonian_matrix, _fic_fixture
from test_x2cscnevpt2_wick import _apply_string


@pytest.mark.parametrize("key", ["i", "r"])
def test_whole_class_retains_orthogonal_external_labels(tmp_path, key):
    from nevpt2_mps_response.class_ab import channels, determinants, whole_class
    from nevpt2_mps_response.diagnose_r85 import (
        _source_coefficients,
        apply_source_vector,
        source_terms,
    )
    case = _fic_fixture(ncas=6, nelec=4)
    sources = [_source_coefficients(case["eris"], key, index)
               for index in range(2)]
    bits = np.sum(determinants(6, 4) * (1 << np.arange(6)), axis=1)
    v = case["active_state"][bits]
    gaps = np.array([10., 8.]) if key == "i" else np.array([7., 9.])
    delta = 1 if key == "i" else -1
    bs = np.array([apply_source_vector(key, s, v, 6, 4) for s in sources])
    target_bits = np.sum(determinants(6, 4 + delta)
                         * (1 << np.arange(6)), axis=1)
    h = _active_hamiltonian_matrix(case["active_h"], case["active_w"])
    expected = 0.
    for source, b, gap in zip(sources, bs, gaps):
        independent = sum((coefficient * _apply_string(case["active_state"], operators)
                           for coefficient, operators in source_terms(key, source)),
                          np.zeros(1 << 6, dtype=complex))
        np.testing.assert_allclose(b, independent[target_bits], atol=1e-14)
        a = h[np.ix_(target_bits, target_bits)] + (gap-case["active_energy"]) * np.eye(len(b))
        expected -= np.vdot(b, np.linalg.solve(a, b)).real
    solver = DMRGCI().init(ncas=6, nelecas=4, nroots=1, max_bond_dimension=64,
                          tol=1e-12, stack_memory=256, scratch=tmp_path / "reference",
                          n_threads=1, final_one_site=False, random_seed=1234)
    try:
        solver.kernel(case["active_h"], case["eris"].get_chem("AAAA"), 6, 4, verbose=0)
        mc = SimpleNamespace(fcisolver=solver, ncas=6, nelecas=4)
        eps = np.r_[[-10., -8.], np.zeros(6), [7., 9.]]
        _results, data = channels(mc, case["eris"], case["pdms"][:3], eps,
                                  key, 64, 8, whole_only=True)
    finally:
        solver.close()
    result = whole_class(data, 6, 4, key, 64, 8, tmp_path / "whole")
    assert result["e_corr"] == pytest.approx(expected, abs=1e-6)
    assert result["global_relative_residual"] < 1e-2


@pytest.mark.parametrize("orbital_ordering", ["original", "fiedler"])
def test_no_four_rdm_branch_matches_exact_complex_response(tmp_path, monkeypatch, orbital_ordering):
    # Neither the response implementation nor either public kernel may import
    # the legacy time-method module, even when that module is unavailable.
    monkeypatch.setitem(sys.modules, "socutils.mrpt.x2ctnevpt2", None)
    from nevpt2_mps_response import diagnose_r85 as diagnostic
    case = _fic_fixture(ncas=6, nelec=4)
    pdms = case["pdms"]
    # Cover both native PT container interfaces without expanding compact ERIs.
    eris = (u._compact_wick_eris(case["eris"])
            if orbital_ordering == "fiedler" else case["eris"])
    core, virtual = np.array([-10., -8.]), np.array([7., 9.])
    for evaluate in (sc._evaluate_wick_subspaces, fic._evaluate_fic_subspaces):
        strict = evaluate(eris, pdms, core, virtual, return_timings=True)[0]
        hybrid = evaluate(eris, pdms[:3], core, virtual,
                          return_timings=True, mps_response=True)[0]
        assert set(hybrid) == set(strict) - {"i", "r"}
        for key, value in hybrid.items():
            assert value == pytest.approx(strict[key], abs=1e-13)

    with pytest.raises(ValueError, match="dm4"):
        u.validate_pdms(pdms[:3], 6, 4)
    checked, _ = u.validate_pdms(pdms[:3], 6, 4, max_rank=3)
    assert len(checked) == 3
    ranks = []

    def make_rdm(_solver, _root, rank):
        assert rank < 4
        ranks.append(rank)
        return pdms[rank - 1]

    monkeypatch.setattr(u, "_make_rdm", make_rdm)
    assert len(response.prepare_pdms(None, None, 0)) == 3
    assert ranks == [1, 2, 3]

    solver = DMRGCI().init(
        ncas=6, nelecas=4, nroots=1, max_bond_dimension=64,
        tol=1e-12, stack_memory=256, scratch=tmp_path / "scratch",
        n_threads=1, final_one_site=False, random_seed=1234,
        orbital_ordering=orbital_ordering,
    )
    try:
        energy, _ci = solver.kernel(
            case["active_h"], case["eris"].get_chem("AAAA"), 6, 4, verbose=0,
        )
        assert energy == pytest.approx(case["active_energy"], abs=1e-11)
        # The residual diagnostic extracts the existing SGF MPS, not a new CI.
        from nevpt2_mps_response.diagnose_r85 import mps_vector
        from pyscf.fci import fci_dhf_slow
        driver, ket = u._root_ket(solver, 0)
        vector = mps_vector(driver, ket, 6, 4)
        h2e = fci_dhf_slow.absorb_h1e(case["active_h"], case["eris"].get_chem("AAAA"), 6, 4, .5)
        action = fci_dhf_slow.contract_2e(h2e, vector, 6, 4)
        assert np.vdot(vector, vector).real == pytest.approx(1., abs=1e-12)
        assert np.vdot(vector, action) == pytest.approx(energy, abs=1e-11)
        # The reference solver converges energy, not a 1e-9 vector residual.
        assert np.linalg.norm(action - energy * vector) < 1e-8
        mc = SimpleNamespace(
            fcisolver=solver, verbose=0, stdout=sys.stdout,
            ncore=2, ncas=6, nelecas=4, mo_coeff=np.eye(10),
            mo_energy=np.r_[core, np.zeros(6), virtual], e_tot=float(energy),
            _scf=SimpleNamespace(mol=SimpleNamespace(verbose=0, stdout=sys.stdout),
                                 get_ovlp=lambda: np.eye(10)),
        )
        native_multiply, calls = driver.multiply, []

        def whole_multiply(bra, mpo, ket, **options):
            assert isinstance(bra.info, driver.bw.brs.NEVPTMPSInfo)
            assert bra.n_sites == 8
            assert bra.info.n_ex_inactive + bra.info.n_ex_external == 1
            assert bra.info.n_inactive == 2 if bra.info.n_ex_inactive else bra.info.n_external == 2
            calls.append("i" if bra.info.n_ex_inactive else "r")
            return native_multiply(bra, mpo, ket, **options)

        def no_channel_solve(*args, **kwargs):
            raise AssertionError("production must not construct individual channel MPOs")

        monkeypatch.setattr(driver, "multiply", whole_multiply)
        monkeypatch.setattr(diagnostic, "_make_source_mpo", no_channel_solve)
        saved_system = driver.__dict__.copy()
        from nevpt2_mps_response.f_atom import audited_response
        # The validation-only fixed-N expansion measures the native whole
        # response; no CI solve or coefficient expansion enters production.
        actual, _norms, _gaps, diagnostics, _times = audited_response(
            response.evaluate_mps_response, mc, eris, pdms[:3], core, virtual,
            options=dict(max_bond_dimension=64, n_sweeps=10, tol=1e-12,
                         linear_threshold=1e-20, noise=0., cutoff=1e-20),
        )
        assert calls == ["i", "r"]
        assert driver.n_sites == 6
        assert driver.ghamil is saved_system["ghamil"]
        np.testing.assert_allclose(mps_vector(driver, ket, 6, 4), vector, atol=1e-12)
        h = _active_hamiltonian_matrix(case["active_h"], case["active_w"])
        for key in ("i", "r"):
            expected = 0.
            expected_norm = 0.
            for index in range(2):
                sector = [k for k in range(1 << 6) if k.bit_count() == 4 + (1 if key == "i" else -1)]
                # Project the independently applied full Hamiltonian, without
                # reusing the response source coefficients or operator strings.
                spectators = (1 << eris.ncore) - 1
                if key == "i":
                    spectators ^= 1 << index
                else:
                    spectators |= 1 << (eris.nocc + index)
                full_sector = (np.asarray(sector) << eris.ncore) | spectators
                b = case["h_reference"][full_sector]
                gap = -core[index] if key == "i" else virtual[index]
                a = h[np.ix_(sector, sector)] + (gap - case["active_energy"]) * np.eye(len(sector))
                expected -= np.vdot(b, np.linalg.solve(a, b)).real
                expected_norm += np.vdot(b, b).real
            assert actual[key] == pytest.approx(expected, abs=1e-8)
            assert diagnostics[key]["source_norm"] == pytest.approx(expected_norm, abs=1e-12)
            assert diagnostics[key]["uses_4rdm"] is False
            assert diagnostics[key]["representation"] == "whole_class"
            assert diagnostics[key]["space_restriction"] == "NEVPTMPSInfo"
            assert diagnostics[key]["global_residual_verified"]
            entry = diagnostics[key]["entries"][0]
            assert entry["hylleraas_energy"] == pytest.approx(actual[key], abs=1e-8)
        assert sc.WickX2CSCNEVPT2(mc).mps_response is False
        assert fic.WickX2CFICNEVPT2(mc).mps_response is False
        for cls in (sc.WickX2CSCNEVPT2, fic.WickX2CFICNEVPT2):
            pt = cls(mc)
            pt.canonicalized = True
            pt.mps_response_options = dict(n_sweeps=10, tol=1e-12,
                                          linear_threshold=1e-20, noise=0., cutoff=1e-20)
            ranks.clear()
            pt.kernel(eris=eris, mps_response=True)
            assert ranks == [1, 2, 3]
            for key in ("i", "r"):
                assert pt.sub_eners[key] == pytest.approx(actual[key], abs=1e-8)
            # Explicitly disabling the flag routes back to the strict 4-RDM request.
            ranks.clear()
            with pytest.raises(AssertionError):
                pt.kernel(eris=eris, mps_response=False)
            assert pt.approximation == "none"
            assert ranks == [1, 2, 3]
        # Even a native solver failure must restore the live reference driver
        # and remove only the temporary response MPSs.
        original_hamiltonian = driver.ghamil

        def fail_native(*args, **kwargs):
            raise RuntimeError("injected native response failure")

        with monkeypatch.context() as failed:
            failed.setattr(driver, "multiply", fail_native)
            with pytest.raises(RuntimeError, match="injected native response failure"):
                response.evaluate_mps_response(mc, eris, pdms[:3], core, virtual)
        assert driver.n_sites == 6
        assert driver.ghamil is original_hamiltonian
        np.testing.assert_allclose(mps_vector(driver, ket, 6, 4), vector, atol=1e-12)
        assert not list((tmp_path / "scratch").glob("*NEVPT2-*"))
    finally:
        solver.close()


def test_public_import_does_not_require_legacy_time_method():
    import subprocess
    code = """
import sys
sys.modules['socutils.mrpt.x2ctnevpt2'] = None
from socutils.mrpt import WickX2CSCNEVPT2, WickX2CFICNEVPT2
from socutils.mrpt import nevpt2_mps_response
assert sys.modules['socutils.mrpt.x2ctnevpt2'] is None
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_batched_sgf_residual_action_matches_pyscf():
    from nevpt2_mps_response.f_atom import active_action_batch
    from pyscf.fci import fci_dhf_slow
    rng = np.random.default_rng(2903)
    eri = rng.normal(size=(6,) * 4) + 1j * rng.normal(size=(6,) * 4)
    vectors = rng.normal(size=(5, 20)) + 1j * rng.normal(size=(5, 20))
    expected = np.array([fci_dhf_slow.contract_2e(eri, vector, 6, 3) for vector in vectors])
    np.testing.assert_allclose(active_action_batch(eri, vectors, 6, 3), expected,
                               atol=1e-12, rtol=1e-12)


def test_global_residual_above_target_warns_and_continues():
    from nevpt2_mps_response.f_atom import check_global_residual
    assert check_global_residual("i", 1e-10)
    with pytest.warns(u.MRPTNumericalWarning, match="retaining the energy and continuing"):
        assert not check_global_residual("r", 5.26e-7)
    # Invalid results are still errors; the warning policy is for finite residuals.
    with pytest.raises(ValueError, match="non-finite"):
        check_global_residual("r", float("nan"))


def test_six_root_runner_preserves_baseline_and_isolates_scratch(tmp_path, monkeypatch):
    import json
    from pathlib import Path

    from nevpt2_mps_response import six_roots
    reference, output = tmp_path / "reference", tmp_path / "results"
    reference.mkdir()
    (reference / "mcscf.json").write_text(json.dumps({"fingerprint": "same"}))
    def data(stage, root):
        result = {"reference_energy": -10., "e_tot": -10.1,
                  "sub_eners": {key: -.01 for key in ("ijrs", "rsi", "ijr", "rs", "ij", "ir", "i", "r")}}
        return dict(root=root, fingerprint="same", results={"SC": result, "FIC": result},
                    requested_rdm_ranks=[1, 2, 3, 4] if stage == "full" else [1, 2, 3])
    for stage in ("full", "hybrid"):
        (reference / f"{stage}.json").write_text(json.dumps(data(stage, 0)))
    original = (reference / "full.json").read_bytes()
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    calls = []
    def fake_run(cmd, **kwargs):
        scratch = Path(kwargs["env"]["TMPDIR"])
        assert scratch.exists() and scratch.parent == tmp_path
        assert kwargs["env"]["PYSCF_TMPDIR"] == str(scratch)
        assert scratch not in calls
        calls.append(scratch)
        stage = cmd[4]
        root = int(cmd[cmd.index("--root") + 1])
        directory = Path(cmd[cmd.index("--output-directory") + 1])
        (directory / f"{stage}.json").write_text(json.dumps(data(stage, root)))
    monkeypatch.setattr(six_roots.subprocess, "run", fake_run)
    six_roots.energies(reference, output, 0)
    six_roots.energies(reference, output, 1)
    six_roots.energies(reference, output, 1)
    assert len(calls) == 2 and all(not scratch.exists() for scratch in calls)
    assert (reference / "full.json").read_bytes() == original
    assert json.loads((output / "root_1/full.json").read_text())["root"] == 1


def test_external_uc_summary_requires_complete_matching_records(tmp_path):
    import json
    from nevpt2_mps_response import six_roots

    reference, baseline, logs, output = (tmp_path / name for name in
                                         ("reference", "baseline", "logs", "summary"))
    reference.mkdir()
    logs.mkdir()
    metadata = dict(fingerprint="same", root_energies=[-10.] * 4 + [-9.] * 2,
                    weights=[1 / 6] * 6)
    (reference / "mcscf.json").write_text(json.dumps(metadata))
    for root in range(6):
        directory = baseline / f"root_{root}"
        directory.mkdir(parents=True)
        values = dict(reference_energy=metadata["root_energies"][root],
                      e_tot=metadata["root_energies"][root] - .1)
        for stage in ("full", "hybrid"):
            (directory / f"{stage}.json").write_text(json.dumps(dict(
                root=root, fingerprint="same", results={"SC": values, "FIC": values})))
    original = (baseline / "root_0/full.json").read_bytes()
    controls = dict(max_bond_dimension=256, n_sweeps=8, diagnostic=True)
    header = "F_UC_POINT " + json.dumps(dict(options=controls)) + "\n"
    (logs / "root0_external_pilot.out").write_text(header + "F_UC_PILOT_RESULT {}\n")
    (logs / "root1_external_live.out").write_text(header + 'F_UC_RESULT {"root":1')
    summarize = lambda: six_roots.summarize(output, baseline=baseline,
                                           reference=reference, uc_logs=logs)
    summarize()
    data = json.loads((output / "summary.json").read_text())
    assert data["external_uc"]["energy_status"] == "pending"
    assert data["external_uc"]["groups"] == []
    assert "UC external_tuples: pending" in (output / "energies.txt").read_text()

    def record(root):
        e2 = -.01 - .001 * root
        return dict(root=root, fingerprint="same", response_mode="external_tuples",
            bond=256, sweeps=8, e_corr=e2, e_tot=metadata["root_energies"][root] + e2,
            converged=root != 5, classes={key: e2 / 8 for key in u.SUBSPACE_ORDER},
            diagnostics=dict(global_relative_residual=2e-8 if root == 5 else 1e-10,
                classes={key: dict(tuples=[dict(holes=[0])], tuple_count=1,
                                   controls=dict(controls, tol=0.))
                         for key in u.SUBSPACE_ORDER}))
    for root in range(6):
        requested = controls if root == 0 else dict(controls, tol=0.)
        root_header = "F_UC_POINT " + json.dumps(dict(options=requested)) + "\n"
        (logs / f"root{root}_external_live.out").write_text(root_header + "F_UC_RESULT "
            + json.dumps(record(root)) + "\nF_UC_RESOURCES "
            + json.dumps(dict(wall_seconds=1., peak_rss_gib=.1)) + "\n")
        if root == 0:
            summarize()
            group = json.loads((output / "summary.json").read_text())["external_uc"]["groups"][0]
            assert not group["all_six_energies_present"]
            assert group["multiplet_total_energy_spreads"]["roots_0_3"] is None
    summarize()
    data = json.loads((output / "summary.json").read_text())["external_uc"]
    assert data["energy_status"] == "complete"
    group = data["groups"][0]
    assert len(data["groups"]) == 1 and group["controls"]["tol"] == 0.
    assert "tol" not in group["roots"]["0"]["controls"]
    assert group["roots"]["1"]["controls"]["tol"] == 0.
    assert group["all_six_energies_present"] and not group["all_six_residuals_certified"]
    assert group["multiplet_total_energy_spreads"] == pytest.approx(dict(roots_0_3=.003, roots_4_5=.001))
    assert "tuples" not in group["roots"]["0"]["diagnostics"]["classes"]["i"]
    assert group["roots"]["0"]["process_resources"]["peak_rss_gib"] == .1
    assert "residual_status=warning" in (output / "energies.txt").read_text()
    assert (baseline / "root_0/full.json").read_bytes() == original
    different = record(5)
    for values in different["diagnostics"]["classes"].values():
        values["controls"]["tol"] = 1e-4
    (logs / "root5_external_live.out").write_text(header + "F_UC_RESULT "
        + json.dumps(different) + "\n")
    summarize()
    data = json.loads((output / "summary.json").read_text())["external_uc"]
    assert data["energy_status"] == "pending" and len(data["groups"]) == 2
    (logs / "root0_external_duplicate.out").write_text(header + "F_UC_RESULT " + json.dumps(record(0)) + "\n")
    with pytest.raises(ValueError, match="duplicate root/control"):
        summarize()
    wrong = dict(record(0), fingerprint="different")
    (logs / "root0_external_duplicate.out").write_text(header + "F_UC_RESULT " + json.dumps(wrong) + "\n")
    with pytest.raises(ValueError, match="incompatible external UC"):
        summarize()


def test_response_defaults_follow_upstream_same_m_schedule():
    controls, bond, tol = response._controls(
        SimpleNamespace(max_bond_dimension=1000, tol=1e-8), None)
    assert (bond, tol, controls.n_sweeps) == (1000, 1e-8, 8)
    # block2main converts the input schedule's Davidson threshold / 50.
    assert (controls.linear_threshold, controls.noise, controls.cutoff) == (2e-8, 1e-5, 1e-14)
    assert controls.linear_rel_conv_thrd == 0.


def test_pt_semicanonicalization_does_not_project_a_single_root_to_kramers(monkeypatch):
    from socutils.mcscf import zmc_superci, zmcscf
    from socutils.scf import spinor_hf

    mol = gto.M(atom="H 0 0 0; H 0.2 0 1.1; H 0 1.5 0; H 1.4 0.1 0.3",
                basis="sto-3g", verbose=0)
    mf = spinor_hf.SCF(mol)
    overlap = mf.get_ovlp()
    values, vectors = np.linalg.eigh(overlap)
    mo = vectors / np.sqrt(values)
    mc = zmcscf.CASSCF(mf, ncas=4, nelecas=2)
    mc.mo_coeff, mc.ci, mc.orbital_symmetry = mo, object(), "kramers"
    rng = np.random.default_rng(823)
    matrix = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    matrix += matrix.conj().T
    fock = overlap @ mo @ matrix @ mo.conj().T @ overlap
    mc.get_fock = lambda *args, **kwargs: fock
    dm1 = np.diag([1., 1., 0., 0.])

    def inherited_kr(*args, **kwargs):
        raise AssertionError("MC default still inherits KR")

    monkeypatch.setattr(zmc_superci, "_kramers_subspace_eigh", inherited_kr)
    with pytest.raises(AssertionError, match="inherits KR"):
        mc.canonicalize(mo, ci=mc.ci, casdm1=dm1)
    rotated, energies = u.semicanonicalize(mc, mo, dm1, 0)
    actual = rotated.conj().T @ fock @ rotated
    for indices in (slice(0, 2), slice(6, 8)):
        block = actual[indices, indices]
        np.testing.assert_allclose(block, np.diag(np.diag(block)), atol=1e-12)
    np.testing.assert_array_equal(rotated[:, 2:6], mo[:, 2:6])
    np.testing.assert_array_equal(mc.mo_coeff, mo)
    np.testing.assert_allclose(energies, actual.diagonal().real, atol=1e-12)
    assert mc.orbital_symmetry == "kramers"
