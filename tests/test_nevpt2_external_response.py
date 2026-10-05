# SPDX-License-Identifier: GPL-3.0-or-later
"""Active-only MPS response versus independent full determinant projections."""

from itertools import combinations
from dataclasses import replace
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from socutils.mrpt import nevpt2_external_response as blocked
from socutils.mrpt import nevpt2_utils as u, x2cucnevpt2 as uc
from test_x2cucnevpt2 import tiny_case, solve_reference, exact_uc
from test_x2cscnevpt2_wick import _apply_string
from test_x2cficnevpt2 import _fic_fixture


def test_tuple_sources_independent_full_hamiltonian():
    case = tiny_case()
    nc, na, nv, nelec = 2, 6, 2, 3
    core, virtual = np.array([-10., -8.]), np.array([7., 9.])
    expected = exact_uc(case, core, virtual)
    for key in u.SUBSPACE_ORDER:
        terms = uc.response._source_tensors(case["eris"], np.arange(na), key)
        nh, npart = sum(key.count(x) for x in "ij"), sum(key.count(x) for x in "rs")
        actual_norm2 = 0.
        for holes in combinations(range(nc), nh):
            for particles in combinations(range(nv), npart):
                coefficients = blocked.tuple_sources(terms, nc, nelec, holes, particles)
                b = np.zeros(1 << na, complex)
                for ops, tensor in coefficients.items():
                    for sites in np.ndindex(tensor.shape):
                        b += tensor[sites] * _apply_string(case["active_state"],
                                                         [(op, site) for op, site in zip(ops, sites)])
                active = np.array([a for a in range(1 << na) if a.bit_count() == nelec + nh - npart])
                indices = (3 ^ sum(1 << i for i in holes)) | (active << nc) | (
                    sum(1 << r for r in particles) << (nc + na))
                np.testing.assert_allclose(b[active], case["h_reference"][indices], atol=1e-13)
                actual_norm2 += np.vdot(b, b).real
        assert actual_norm2 == pytest.approx(expected[key]["norm2"], abs=1e-13)


@pytest.mark.parametrize("orbital_ordering", ["original", "fiedler"])
def test_active_only_native_response_all_eight_classes(tmp_path, monkeypatch, orbital_ordering):
    from nevpt2_mps_response.diagnose_r85 import mps_vector
    from pyscf.fci import cistring
    from test_x2cficnevpt2 import _hamiltonian_action
    case = dict(tiny_case())
    core, virtual = np.array([-10., -8.]), np.array([7., 9.])
    solver = solve_reference(case, tmp_path / "reference", orbital_ordering=orbital_ordering)
    # Compare the same finite reference, not the fixture's exact eigenvector.
    bits = np.sum(1 << cistring.gen_occslst(range(6), 3), axis=1)
    case["active_state"] = np.zeros(64, complex)
    case["active_state"][bits] = mps_vector(solver.driver, solver.kets[0], 6, 3)
    case["pdms"] = tuple(u._make_rdm(solver, 0, rank) for rank in (1, 2))
    case["active_energy"] = float((np.einsum("pq,pq", case["active_h"], case["pdms"][0])
        + .5 * np.einsum("pqrs,pqsr", case["active_w"], case["pdms"][1])).real)
    full_reference = np.zeros(1024, complex)
    full_reference[(bits << 2) | 3] = case["active_state"][bits]
    case["h_reference"] = _hamiltonian_action(full_reference, case["h1e"], case["w"])
    expected = exact_uc(case, core, virtual)
    native, calls, residuals = uc._linear_response, [], {}

    def tuple_key(pattern):
        determinant = int(pattern[0][0])
        return (tuple(i for i in range(2) if not (determinant >> i & 1)),
                tuple(r for r in range(2) if determinant >> (8 + r) & 1))

    patterns = iter((key, *tuple_key(pattern), pattern)
                    for key in u.SUBSPACE_ORDER if key != "ijrs"
                    for pattern in sorted(expected[key]["patterns"], key=tuple_key))

    def check(driver, state, reference, *args):
        assert state.n_sites == reference.n_sites == 6
        assert type(state.info) is driver.bw.brs.MPSInfo
        calls.append(int(state.info.target.n))
        print("BLOCKED_NATIVE", len(calls), state.info.target, args[0].const_e, flush=True)
        result = native(driver, state, reference, *args)
        key, holes, particles, (indices, matrix, b) = next(patterns)
        ntarget = int(state.info.target.n)
        # Test-only CI readout of the actual solved MPS. Restore the original
        # orbital order and fermion phase before using the independent H/HD.
        saved_order = driver.reorder_idx
        try:
            driver.reorder_idx = solver.driver.reorder_idx
            vector = mps_vector(driver, state, 6, ntarget)
        finally:
            driver.reorder_idx = saved_order
        bits = np.sum(1 << cistring.gen_occslst(range(6), ntarget), axis=1)
        x = np.zeros(64, complex)
        x[bits] = vector
        active_bits = (indices >> 2) & 63
        residuals[key, holes, particles] = np.linalg.norm(
            matrix @ x[active_bits] - b / np.linalg.norm(b))
        return result

    monkeypatch.setattr(uc, "_linear_response", check)
    try:
        mc = SimpleNamespace(fcisolver=solver, verbose=0, stdout=sys.stdout)
        eris = u._compact_wick_eris(case["eris"]) if orbital_ordering == "fiedler" else case["eris"]
        energies, data = uc.evaluate_uc(mc, eris, case["pdms"][:2], core, virtual,
            response_mode="external_tuples", options=dict(max_bond_dimension=64, n_sweeps=8, diagnostic=True))
        assert len(calls) == 14 and set(calls) == {1, 2, 3, 4, 5}
        for key in u.SUBSPACE_ORDER:
            assert energies[key] == pytest.approx(expected[key]["energy"], abs=1e-10), (key, data[key])
            assert data[key]["source_norm2"] == pytest.approx(expected[key]["norm2"], abs=1e-10)
            assert data[key]["converged"]
            assert data[key]["global_relative_residual"] < 1e-8
            assert data[key]["max_channel_relative_residual"] < 1e-8
            assert all(e["holes"] or e["particles"] for e in data[key]["tuples"])
            for entry in data[key]["tuples"]:
                if key != "ijrs":
                    independent = residuals[key, tuple(entry["holes"]), tuple(entry["particles"])]
                    assert independent < 1e-8
                    assert abs(entry["global_relative_residual"] - independent) < 1e-11
        assert sum(energies.values()) == pytest.approx(sum(e["energy"] for e in expected.values()), abs=1e-10)
        assert data["ijrs"]["tuples"][0]["actual_sweeps"] == 0
        if orbital_ordering == "original":
            # All representations use this SAME retained finite reference,
            # RDMs, integral tensors and Dyall energy, not separate fixtures.
            monkeypatch.setattr(uc, "_linear_response", native)
            eight, eight_data = uc.evaluate_uc(mc, case["eris"], case["pdms"][:2], core, virtual,
                class_resolved=True, options=dict(max_bond_dimension=64, n_sweeps=12, diagnostic=True))
            whole, whole_data = uc.evaluate_uc(mc, case["eris"], case["pdms"][:2], core, virtual,
                options=dict(max_bond_dimension=64, n_sweeps=2, strict_cas=True, diagnostic=True))
            for key in u.SUBSPACE_ORDER:
                for result, audit in ((eight, eight_data), (whole, whole_data)):
                    assert result[key] == pytest.approx(energies[key], abs=1e-10), key
                    assert audit[key]["converged"]
                    assert audit[key]["global_relative_residual"] < 1e-8
            print("UC_SAME_REFERENCE_COMPARISON", json.dumps(dict(
                blocked=energies, eight=eight, strict_whole=whole,
                exact={key: d["energy"] for key, d in expected.items()})))
    finally:
        solver.close()


def test_streamed_product_matches_native_algebra(tmp_path):
    case = tiny_case()
    solver = solve_reference(case, tmp_path / "reference")
    try:
        driver, ket = u._root_ket(solver, 0)
        active = uc.response._active_mps(driver, ket)
        ha = uc._qc_mpo(driver, case["active_h"], case["eris"].get_chem("AAAA"), -.317)
        for mpo in [ha, *[blocked._source_mpo(driver, blocked.tuple_sources(
                uc.response._source_tensors(case["eris"], np.arange(6), key), 2, 3,
                tuple(range(sum(key.count(x) for x in "ij"))),
                tuple(range(sum(key.count(x) for x in "rs")))), 1.) for key in u.SUBSPACE_ORDER]]:
            operator = uc._algebra_mpo(mpo)
            assert uc._norm(blocked.apply_mpo(operator, active) - operator @ active) < 1e-12
    finally:
        solver.close()


@pytest.mark.parametrize("eris_mode", ["dense", "compact", "auto_compact"])
def test_public_mode_and_reference_lifecycle(tmp_path, monkeypatch, eris_mode):
    from socutils.mrpt import UCNEVPT2, x2cficnevpt2, x2cscnevpt2
    from socutils.mrpt import nevpt2_eris
    case = tiny_case()
    solver = solve_reference(case, tmp_path / "reference")
    eps = np.r_[[-10., -8.], np.zeros(6), [7., 9.]]
    eris = case["eris"]
    ecore = .5 * np.trace((eris.h1e + eris.h1eff)[:2, :2]).real
    compact = u._compact_wick_eris(eris)
    compact = replace(compact, symmetry_diagnostics=dict(compact.symmetry_diagnostics or {},
                                                       electronic_core_energy=float(ecore)))
    mol = SimpleNamespace(energy_nuc=lambda: 2.)
    mf = SimpleNamespace(mol=mol, get_ovlp=lambda: np.eye(10))
    mc = SimpleNamespace(fcisolver=solver, verbose=0, stdout=sys.stdout, _scf=mf, mol=mol,
        ncore=2, ncas=6, nelecas=3, mo_coeff=np.eye(10), mo_energy=eps,
        e_tot=ecore + case["active_energy"] + 2., get_fock=lambda **kwargs: np.diag(eps))
    def forbidden(*args, **kwargs):
        raise AssertionError("blocked UC cannot use high RDMs, IC fallback or a full chain")
    for method in ("get_3pdm", "get_4pdm"):
        monkeypatch.setattr(solver.driver, method, forbidden)
    monkeypatch.setattr(uc, "_qc_problem", forbidden)
    monkeypatch.setattr(x2cscnevpt2, "_evaluate_wick_subspaces", forbidden)
    monkeypatch.setattr(x2cficnevpt2, "_evaluate_fic_subspaces", forbidden)
    monkeypatch.setattr(u, "_dense_eris_from_mc", forbidden)
    block_calls = []
    def exact_blocks(reference, mo):
        assert reference is mc
        np.testing.assert_array_equal(mo, mc.mo_coeff)
        block_calls.append(1)
        return compact
    monkeypatch.setattr(nevpt2_eris, "wick_eris_from_mc", exact_blocks)
    try:
        frame = solver.driver.frame
        pt = UCNEVPT2(mc)
        assert pt.response_mode == "full_chain"
        pt.response_mode, pt.canonicalized = "external_tuples", True
        pt.scratch, pt.stack_memory, pt.n_threads = str(tmp_path), 256, 1
        pt.mps_response_options = dict(max_bond_dimension=64, n_sweeps=6, diagnostic=True)
        pt.kernel(eris={"dense": eris, "compact": compact, "auto_compact": None}[eris_mode],
                  pdms=case["pdms"][:2])
        assert len(block_calls) == int(eris_mode == "auto_compact")
        assert pt.eris is (eris if eris_mode == "dense" else compact)
        expected = exact_uc(case, eps[:2], eps[8:])
        assert pt.converged
        assert pt.e_corr == pytest.approx(sum(e["energy"] for e in expected.values()), abs=1e-10)
        assert pt.diagnostics["max_channel_relative_residual"] < 1e-8
        assert pt.diagnostics["global_relative_residual"] < 1e-8
        assert abs(pt.diagnostics["reference_energy_difference"]) < 1e-12
        assert solver.driver.bw.b.Global.frame is frame
        np.testing.assert_array_equal(mc.mo_coeff, np.eye(10))
        bad = np.diag(eps)
        bad[0, 1] = bad[1, 0] = .01
        mc.get_fock = lambda **kwargs: bad
        with pytest.raises(ValueError, match="semicanonical"):
            pt.kernel(eris=eris, pdms=case["pdms"][:2])
    finally:
        solver.close()


@pytest.mark.parametrize("amplitude", [1e-11j, 0.])
@pytest.mark.parametrize("diagnostic", [False, True])
def test_weak_and_zero_sources_preserve_amplitude(tmp_path, monkeypatch, amplitude, diagnostic):
    case = tiny_case()
    solver = solve_reference(case, tmp_path / "reference")
    original = uc.response._source_tensors
    monkeypatch.setattr(uc.response, "_source_tensors", lambda *args: [
        (ops, amplitude * t, labels) for ops, t, labels in original(*args)])
    core, virtual = np.array([-10., -8.]), np.array([7., 9.])
    try:
        mc = SimpleNamespace(fcisolver=solver, verbose=0, stdout=sys.stdout)
        energies, data = uc.evaluate_uc(mc, case["eris"], case["pdms"][:2], core, virtual,
            response_mode="external_tuples", options=dict(max_bond_dimension=64, n_sweeps=6, diagnostic=diagnostic))
        exact = exact_uc(case, core, virtual)
        for key in u.SUBSPACE_ORDER:
            if amplitude:
                assert energies[key] / abs(amplitude)**2 == pytest.approx(exact[key]["energy"], abs=1e-10)
                if diagnostic or key == "ijrs":
                    assert data[key]["converged"]
                else:
                    assert not data[key]["converged"]
                    assert data[key]["global_relative_residual"] is None
            else:
                assert energies[key] == data[key]["source_norm2"] == 0.
                assert all(e["zero_source"] for e in data[key]["tuples"])
        if amplitude:
            assert data["ijrs"]["tuples"][0]["actual_sweeps"] == 0
    finally:
        solver.close()


@pytest.mark.parametrize("virtual", [[7., 9.], [-40., -30.]], ids=["positive", "nonpositive"])
def test_ijrs_finite_reference_requires_full_resolvent(tmp_path, monkeypatch, virtual):
    from test_x2cscnevpt2_wick import _raw_active_rdms
    from test_x2cficnevpt2 import _hamiltonian_action, _active_hamiltonian_matrix
    from pyscf.fci import cistring
    case = dict(tiny_case())
    bits = np.sum(1 << cistring.gen_occslst(range(6), 3), axis=1)
    approximate = case["active_state"].copy()
    approximate[bits[0]] += .01j
    approximate /= np.linalg.norm(approximate)
    ha = _active_hamiltonian_matrix(case["active_h"], case["active_w"])
    case["active_state"] = approximate
    case["active_energy"] = np.vdot(approximate, ha @ approximate).real
    case["pdms"] = _raw_active_rdms(approximate, 6)
    reference = np.zeros(1024, complex)
    reference[(bits << 2) | 3] = approximate[bits]
    case["h_reference"] = _hamiltonian_action(reference, case["h1e"], case["w"])
    solver = solve_reference(tiny_case(), tmp_path / "reference")
    old = solver.kets
    driver = solver.driver
    driver.reorder_idx = None
    determinants = np.zeros((len(bits), 6), dtype=np.uint8)
    for i, b in enumerate(bits):
        determinants[i] = [(b >> a) & 1 for a in range(6)]
    ket = driver.get_mps_from_csf_coefficients(determinants, approximate[bits],
        tag=uc.response._response_tag(), dot=2, iprint=0)
    solver.kets = [ket]
    native, calls = uc._linear_response, []
    def check(*args):
        calls.append(1)
        return native(*args)
    monkeypatch.setattr(uc, "_linear_response", check)
    core, virtual = np.array([-10., -8.]), np.asarray(virtual)
    try:
        mc = SimpleNamespace(fcisolver=solver, verbose=0, stdout=sys.stdout)
        energies, data = uc.evaluate_uc(mc, case["eris"], case["pdms"][:2], core, virtual,
            response_mode="external_tuples", options=dict(max_bond_dimension=64, n_sweeps=6, diagnostic=True))
        assert len(calls) == 15  # No eigen-reference shortcut for this perturbed reference.
        expected = exact_uc(case, core, virtual, require_positive=bool(np.all(virtual > 0)))
        for key in u.SUBSPACE_ORDER:
            assert energies[key] == pytest.approx(expected[key]["energy"], abs=1e-10)
            assert data[key]["max_channel_relative_residual"] < 1e-8
        assert data["ijrs"]["reference_active_residual_norm"] > 1e-3
        assert data["ijrs"]["tuples"][0]["actual_sweeps"] == 6
        if np.any(virtual < 0):
            # Deliberately nonphysical diagnostic partition: invertible but
            # nonpositive blocks have a stationary, not minimal, functional.
            assert expected["r"]["min_eigenvalue"] < 0 and energies["r"] > 0
            assert energies["ijrs"] > 0
    finally:
        solver.kets = old
        uc.response._release_response_mps(driver, ket)
        solver.close()


@pytest.mark.parametrize("nelec", [0, 6])
def test_particle_sector_boundaries(tmp_path, nelec):
    case = _fic_fixture(ncas=6, nelec=nelec)
    core, virtual = np.array([-10., -8.]), np.array([7., 9.])
    solver = solve_reference(case, tmp_path / "reference", nelec=nelec)
    try:
        mc = SimpleNamespace(fcisolver=solver, verbose=0, stdout=sys.stdout)
        energies, data = uc.evaluate_uc(mc, case["eris"], case["pdms"][:2], core, virtual,
            response_mode="external_tuples", options=dict(max_bond_dimension=64, n_sweeps=6, diagnostic=True))
        exact = exact_uc(case, core, virtual)
        for key in u.SUBSPACE_ORDER:
            assert energies[key] == pytest.approx(exact[key]["energy"], abs=1e-10)
            assert data[key]["converged"]
            assert data[key]["max_channel_relative_residual"] < 1e-8
    finally:
        solver.close()
