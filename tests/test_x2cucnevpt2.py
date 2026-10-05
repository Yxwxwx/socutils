# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent determinant projections for the full eight-class MPS response."""

import sys
import json
from contextlib import contextmanager
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import pytest

from socutils.dmrg import DMRGCI
from socutils.mrpt import nevpt2_utils as u
from socutils.mrpt import x2cucnevpt2 as uc
from test_x2cficnevpt2 import _active_hamiltonian_matrix, _fic_fixture


@lru_cache(maxsize=1)
def tiny_case():
    return _fic_fixture(ncas=6, nelec=3)


def solve_reference(case, scratch, nelec=3, orbital_ordering="original"):
    solver = DMRGCI().init(ncas=6, nelecas=nelec, nroots=1,
                          max_bond_dimension=64, tol=1e-13, schedule_thrd_max=1e-24,
                          stack_memory=256, n_threads=1, scratch=scratch,
                          final_one_site=False, random_seed=1234, orbital_ordering=orbital_ordering)
    solver.kernel(case["active_h"], case["eris"].get_chem("AAAA"), 6, nelec, verbose=0)
    return solver


def exact_uc(case, core, virtual, *, require_positive=True):
    """Project full H|Psi> using independent fermion action, never source code.

    Fixed external determinant patterns are orthogonal invariant Dyall blocks.
    Every active determinant of the required particle number is included,
    even when its source coefficient is zero.
    """
    eris = case["eris"]
    ncore, ncas, nvirt = eris.ncore, eris.ncas, eris.nvirt
    nelec = next(i.bit_count() for i, c in enumerate(case["active_state"]) if abs(c) > 1e-8)
    ha = _active_hamiltonian_matrix(case["active_h"], case["active_w"])
    assert np.max(abs(ha - ha.conj().T)) < 1e-12
    classes = {(2, 2): "ijrs", (1, 2): "rsi", (2, 1): "ijr",
               (0, 2): "rs", (2, 0): "ij", (1, 1): "ir", (0, 1): "r", (1, 0): "i"}
    results = {key: dict(energy=0., norm2=0., min_eigenvalue=np.inf, patterns=[])
               for key in classes.values()}
    seen = set()
    for ci in range(1 << ncore):
        holes = [i for i in range(ncore) if not (ci >> i & 1)]
        for vi in range(1 << nvirt):
            particles = [r for r in range(nvirt) if vi >> r & 1]
            key = classes.get((len(holes), len(particles)))
            if key is None:
                continue
            active = np.array([a for a in range(1 << ncas)
                               if a.bit_count() == nelec + len(holes) - len(particles)])
            if not len(active):
                continue
            indices = ci | (active << ncore) | (vi << (ncore + ncas))
            assert not seen.intersection(indices)
            seen.update(indices)
            b = case["h_reference"][indices]
            matrix = ha[np.ix_(active, active)] + np.eye(len(active)) * (
                sum(virtual[particles]) - sum(core[holes]) - case["active_energy"])
            eigenvalues = np.linalg.eigvalsh(matrix)
            mineig = eigenvalues[0]
            assert mineig > 0 if require_positive else np.min(abs(eigenvalues)) > 1e-10
            x = np.linalg.solve(matrix, b)
            assert np.linalg.norm(matrix @ x - b) < 1e-12
            result = results[key]
            result["energy"] -= np.vdot(b, x).real
            result["norm2"] += np.vdot(b, b).real
            result["min_eigenvalue"] = min(result["min_eigenvalue"], mineig)
            result["patterns"].append((indices, matrix, b))
    # All external H|Psi> lies in the eight projectors; pure CAS is excluded.
    outside = case["h_reference"].copy()
    outside[list(seen)] = 0
    outside[(1 << ncore) - 1 | (np.arange(1 << ncas) << ncore)] = 0
    assert np.linalg.norm(outside) < 1e-12
    return results


def test_whole_source_sector_projectors(tmp_path):
    """Projector diagnostics are checked independently of the response solve."""
    case = tiny_case()
    solver = solve_reference(case, tmp_path / "mps")
    core, virtual = np.array([-10., -8.]), np.array([7., 9.])
    expected = exact_uc(case, core, virtual)
    try:
        driver, ket = u._root_ket(solver, 0)
        active = uc.response._active_mps(driver, ket)
        with uc.response._class_problem(driver, active, case["eris"], np.arange(6),
                                       "all", core, virtual, case["active_energy"]) as problem:
            reference, source_mpo, _, nc, nv = problem
            from nevpt2_mps_response.f_uc import product_storage_bound
            mpo = uc._algebra_mpo(source_mpo)
            ref = uc.response._active_mps(driver, reference)
            bound = product_storage_bound(mpo, ref)
            b = mpo @ ref
            actual = sum(block.reduced.nbytes for tensor in b.tensors for block in tensor.blocks)
            assert bound["total_tensor_gib"] * 2**30 >= actual
            # Q must not double the large, redundant virtual-tail bonds of
            # an exact MPO product merely to subtract its rank-one CAS tail.
            qb = uc._external_part(b, nc, nv)
            for original, projected in zip(b.get_bond_dims()[-nv:], qb.get_bond_dims()[-nv:]):
                assert sum(projected.values()) <= sum(original.values()) + 1
            assert uc._norm(qb - b) < 1e-12
            for key in u.SUBSPACE_ORDER:
                part = uc._project_class(b, nc, nv, int(driver.target.n), key)
                assert uc._norm(part)**2 == pytest.approx(expected[key]["norm2"], abs=1e-10), (
                    key, [[str(block.q_labels) for block in b.tensors[j].blocks] for j in (1, 7)])
            assert uc._norm(uc._project_class(b, nc, nv, int(driver.target.n), "")) < 1e-12
    finally:
        solver.close()


def test_eight_complex_classes_against_independent_exact_uc(tmp_path, monkeypatch):
    case = tiny_case()
    assert np.max(abs(case["w"].imag)) > 1e-3
    core, virtual = np.array([-10., -8.]), np.array([7., 9.])
    expected = exact_uc(case, core, virtual)
    assert all(item["norm2"] > 1e-8 for item in expected.values())
    solver = solve_reference(case, tmp_path / "mps")
    try:
        mc = SimpleNamespace(fcisolver=solver, verbose=0, stdout=sys.stdout)
        native = uc._linear_response
        measured = {}
        def audit(driver, state, reference, dyall, source_mpo, controls, bond, iprint):
            from nevpt2_mps_response.diagnose_r85 import mps_vector
            from pyscf.fci import cistring
            from test_x2cficnevpt2 import _hamiltonian_action
            value = native(driver, state, reference, dyall, source_mpo, controls, bond, iprint)
            nc, nv = state.info.n_inactive, state.info.n_external
            nh, npart = state.info.n_ex_inactive, state.info.n_ex_external
            key = next(k for k in u.SUBSPACE_ORDER if sum(k.count(x) for x in "ij") == nh
                       and sum(k.count(x) for x in "rs") == npart)
            bits = np.sum(1 << cistring.gen_occslst(range(state.n_sites), int(driver.target.n)), axis=1)
            x = np.zeros(1 << state.n_sites, complex)
            x[bits] = mps_vector(driver, state, state.n_sites, int(driver.target.n))
            # Independent fermion action on the ACTUAL retained reference,
            # not the production source MPO or the fixture's exact CI state.
            full_reference = np.zeros(1024, complex)
            full_reference[bits if nc else (bits << 2) | 3] = mps_vector(
                driver, reference, state.n_sites, int(driver.target.n))
            h_reference = _hamiltonian_action(full_reference, case["h1e"], case["w"])
            norm = np.sqrt(sum(np.linalg.norm(h_reference[ix])**2
                               for ix, _, _ in expected[key]["patterns"]))
            rr = 0.
            for ix, matrix, _ in expected[key]["patterns"]:
                loc = ix if nc else ix >> 2
                if not nv:
                    loc = loc & ((1 << state.n_sites) - 1)
                rr += np.linalg.norm(matrix @ x[loc] - h_reference[ix] / norm)**2
            measured[key] = np.sqrt(rr)
            return value
        monkeypatch.setattr(uc, "_linear_response", audit)
        energies, diagnostics = uc.evaluate_uc(
            mc, case["eris"], case["pdms"][:2], core, virtual,
            options=dict(max_bond_dimension=128, n_sweeps=12, diagnostic=True), class_resolved=True)
        for key in u.SUBSPACE_ORDER:
            assert energies[key] == pytest.approx(expected[key]["energy"], abs=1e-10), key
            assert diagnostics[key]["source_norm2"] == pytest.approx(expected[key]["norm2"], abs=1e-10), key
            assert diagnostics[key]["global_relative_residual"] <= 1e-8, (key, diagnostics[key])
            assert abs(diagnostics[key]["global_relative_residual"] - measured[key]) < 1e-12
            assert diagnostics[key]["actual_sweeps"] == 12
        assert sum(energies.values()) == pytest.approx(sum(x["energy"] for x in expected.values()), abs=1e-10)
    finally:
        solver.close()


def test_constrained_whole_response_against_exact(tmp_path, monkeypatch):
    from pyblock2.algebra.core import MPS
    case = tiny_case()
    core, virtual = np.array([-10., -8.]), np.array([7., 9.])
    expected = exact_uc(case, core, virtual)
    solver = solve_reference(case, tmp_path / "mps")
    original_factors, original_projector = uc._cas_boundary_factors, uc._cas_local_projector
    captured = {}
    original_compress = MPS.compress

    def compress(mps, *args, **kwargs):
        # Only response truncations may compress here; no fitted RHS layer.
        assert kwargs.get("k") is not None
        return original_compress(mps, *args, **kwargs)

    monkeypatch.setattr(MPS, "compress", compress)

    def factors(mps, *args):
        captured["template"] = mps.deep_copy()
        return original_factors(mps, *args)

    def projector(driver, state, site, *args):
        from pyblock2.algebra.core import SubTensor, Tensor
        from pyblock2.algebra.io import TensorTools
        assert isinstance(state.info, driver.bw.brs.MRCIMPSInfo)
        assert state.info.ci_order == 2
        q = original_projector(driver, state, site, *args)
        if site != 4 or captured.get("checked"):
            return q
        captured["checked"] = True
        sp = state.tensors[site]
        size = sum(np.asarray(sp[j]).size for j in range(sp.info.n))
        # At the middle of this tiny full-rank chain V spans all C(10,5).
        # Removing all C(6,3)=20 CAS states leaves 232, not 251 or 246.
        assert size == 252
        dense_q = np.column_stack([q(e) for e in np.eye(size, dtype=complex)])
        np.testing.assert_allclose(dense_q, dense_q.conj().T, atol=1e-13)
        np.testing.assert_allclose(dense_q @ dense_q, dense_q, atol=1e-13)
        assert np.trace(dense_q).real == pytest.approx(232., abs=1e-10)
        rng = np.random.default_rng(813)
        vector = q(rng.normal(size=size) + 1j * rng.normal(size=size))
        info = state.info
        ld, rd = info.left_dims[site], info.right_dims[site + 2]
        a, b = info.basis[site], info.basis[site + 1]
        lm = a.__class__.tensor_product_ref(ld, a, info.left_dims_fci[site + 1])
        mr = a.__class__.tensor_product_ref(b, rd, info.right_dims_fci[site + 1])
        cl = a.__class__.get_connection_info(ld, a, lm)
        cr = a.__class__.get_connection_info(b, rd, mr)
        saved = [np.array(sp[j]) for j in range(sp.info.n)]
        try:
            offset = 0
            for j in range(sp.info.n):
                view = np.asarray(sp[j])
                view[:] = vector[offset:offset + view.size].reshape(view.shape)
                offset += view.size
            local = TensorTools.from_block2_left_and_right_fused(sp, ld, a, b, rd, lm, cl, mr, cr)
        finally:
            for j, value in enumerate(saved):
                np.asarray(sp[j])[:] = value
            lm.deallocate()
            mr.deallocate()
        trial = captured["template"].deep_copy()
        trial.tensors[site] = Tensor([SubTensor(
            x.q_labels[:-1] + (driver.target - x.q_labels[-1],), x.reduced) for x in local.blocks])
        trial.tensors[site + 1] = None
        # Independent determinant-amplitude test, no production CAS projector.
        for active_bits in (k for k in range(64) if k.bit_count() == 3):
            occupied = [1, 1] + [(active_bits >> j) & 1 for j in range(6)] + [0, 0]
            coefficient = np.ones(1, complex)
            electrons = 0
            for i, tensor in enumerate(trial.tensors):
                if tensor is None:
                    continue
                count = 2 if i == site else 1
                physical = 0 if i == 0 else 1
                matches = [block for block in tensor.blocks
                           if (i == 0 or block.q_labels[0].n == electrons)
                           and [label.n for label in block.q_labels[physical:physical + count]] == occupied[i:i + count]]
                assert len(matches) == 1
                array = matches[0].reduced
                rows = 1 if i == 0 else array.shape[0]
                cols = 1 if i + count == 10 else array.shape[-1]
                coefficient = coefficient @ array.reshape(rows, cols)
                electrons += sum(occupied[i:i + count])
            assert abs(coefficient.item()) < 1e-11
        return q

    monkeypatch.setattr(uc, "_cas_boundary_factors", factors)
    monkeypatch.setattr(uc, "_cas_local_projector", projector)
    try:
        driver, ket = u._root_ket(solver, 0)
        active = uc.response._active_mps(driver, ket)
        energy, diag = uc._solve_class(driver, active, case["eris"], np.arange(6), "all",
                                       core, virtual, case["active_energy"],
                                       uc.UCControls(n_sweeps=2, strict_cas=True, diagnostic=True), 64)
        assert diag["global_relative_residual"] < 1e-8, diag
        assert energy == pytest.approx(sum(d["energy"] for d in expected.values()), abs=1e-10)
        assert diag["cas_response_norm_over_source"] < 1e-12
        for key in u.SUBSPACE_ORDER:
            assert diag["classes"][key]["hylleraas_energy"] == pytest.approx(expected[key]["energy"], abs=1e-10)
        mc = SimpleNamespace(fcisolver=solver, verbose=0, stdout=sys.stdout)
        blocked, _ = uc.evaluate_uc(mc, case["eris"], case["pdms"][:2], core, virtual,
                                    options=dict(max_bond_dimension=64, n_sweeps=2, diagnostic=True), class_resolved=True)
        assert energy == pytest.approx(sum(blocked.values()), abs=1e-11)
        assert max(d["retained_cas_relative_norm"] for d in diag["constraint_history"]) < 1e-12
        assert max(d["local_krylov_cas_relative_norm"] for d in diag["constraint_history"]) < 1e-10
        assert captured["checked"]
        print("UC_TINY", json.dumps(dict(
            energy=energy, blocked_energy=sum(blocked.values()),
            exact_energy=sum(d["energy"] for d in expected.values()),
            global_residual=diag["global_relative_residual"],
            max_class_residual=max(d["global_relative_residual"] for d in diag["classes"].values()),
            source_cas_norm=diag["cas_source_norm"],
            response_cas_norm=diag["cas_response_norm"],
            max_krylov_cas=max(d["local_krylov_cas_relative_norm"] for d in diag["constraint_history"]),
            max_basis_map_error=max(d["cas_basis_mapping_error"] for d in diag["constraint_history"]),
            classes={k: d["hylleraas_energy"] for k, d in diag["classes"].items()},
        )))
    finally:
        solver.close()


@pytest.mark.parametrize("bond", [64, 2])
def test_entire_cas_contamination_and_truncation(tmp_path, monkeypatch, bond):
    """Inject a CAS vector orthogonal to SIX roots into both b and the guess."""
    from pyscf.fci import cistring
    from socutils.mrpt import nevpt2_mps_response as response
    case = tiny_case()
    ha = _active_hamiltonian_matrix(case["active_h"], case["active_w"])
    bits = np.array([i for i in range(64) if i.bit_count() == 3])
    _, vectors = np.linalg.eigh(ha[np.ix_(bits, bits)])
    rng = np.random.default_rng(921)
    ci = vectors[:, 6:] @ (rng.normal(size=14) + 1j * rng.normal(size=14))
    ci /= np.linalg.norm(ci)
    assert np.max(abs(vectors[:, :6].conj().T @ ci)) < 1e-13
    coefficients = dict(zip(bits, ci))
    solver = solve_reference(case, tmp_path / "mps")
    original_problem = response._class_problem
    original_state = response._nevpt_mps
    pending = {}

    @contextmanager
    def problem(*args, **kwargs):
        with original_problem(*args, **kwargs) as data:
            driver = args[0]
            active_occ = cistring.gen_occslst(range(6), 3)
            dets = np.zeros((20, 10), dtype=np.uint8)
            dets[:, :2] = 1
            dets[np.arange(20)[:, None], active_occ + 2] = 1
            values = np.array([coefficients[int(sum(1 << a))] for a in active_occ])
            native = driver.get_mps_from_csf_coefficients(
                dets, values, tag=response._response_tag(), dot=2, iprint=0)
            try:
                pending["cas"] = response._active_mps(driver, native)
            finally:
                response._release_response_mps(driver, native)
            # Pollute the ACTUAL native RHS MPO, not just a diagnostic copy.
            # |ci><det0| on the active N=3 sector, identity on spectators.
            builder = driver.expr_builder()
            sites = {"I": np.arange(2), "A": np.arange(6) + 2, "E": np.arange(2) + 8}
            for ops, tensor, labels in response._source_tensors(case["eris"], np.arange(6), "all"):
                response._add_tensor(builder, ops, kwargs.get("source_scale", 1.) * tensor,
                                     [sites[x] for x in labels])
            det0 = int(np.argmax(abs(case["active_state"])))
            annihilate = [i + 2 for i in range(6) if det0 >> i & 1][::-1]
            for occupied, coefficient in zip(active_occ, values):
                builder.add_term("CCCDDD", list(occupied + 2) + annihilate,
                                 (13. - 7.j) * coefficient / case["active_state"][det0])
            polluted = driver.get_mpo(builder.finalize(), cutoff=0., iprint=0, add_ident=False)
            yield data[0], polluted, *data[2:]

    def state(driver, nc, nv, key, m):
        native = original_state(driver, nc, nv, key, m)
        if key != "all":
            return native
        try:
            raw = response._active_mps(driver, native) + (11. + 9.j) * pending["cas"]
            return uc._mrci_from_algebra(driver, raw, nc, nv, 0)
        finally:
            response._release_response_mps(driver, native)

    monkeypatch.setattr(response, "_class_problem", problem)
    monkeypatch.setattr(response, "_nevpt_mps", state)
    core, virtual = np.array([-10., -8.]), np.array([7., 9.])
    try:
        driver, ket = u._root_ket(solver, 0)
        active = response._active_mps(driver, ket)
        energy, diag = uc._solve_class(driver, active, case["eris"], np.arange(6), "all",
                                       core, virtual, case["active_energy"],
                                       uc.UCControls(n_sweeps=2, strict_cas=True, diagnostic=True), bond)
        assert diag["cas_source_norm_before_projection"] > 1.
        assert diag["cas_source_norm"] < 1e-12
        assert diag["source_representation"] == "direct source MPO acting on reference MPS"
        assert diag["cas_response_norm_over_source"] < 1e-12
        assert max(d["retained_cas_relative_norm"] for d in diag["constraint_history"]) < 1e-12
        if bond == 64:
            expected = exact_uc(case, core, virtual)
            assert diag["global_relative_residual"] < 1e-8
            assert energy == pytest.approx(sum(d["energy"] for d in expected.values()), abs=1e-10)
        else:
            assert not diag["converged"]
            assert diag["global_relative_residual"] > 1e-3
    finally:
        solver.close()


def test_truncation_reintroduces_cas_unless_constrained(tmp_path):
    """Rank-one SVD of [[0,1],[1,1]] fills its forbidden CAS corner."""
    from socutils.mrpt import nevpt2_mps_response as response
    case = tiny_case()
    solver = solve_reference(case, tmp_path / "mps")
    try:
        driver, ket = u._root_ket(solver, 0)
        active = response._active_mps(driver, ket)
        with response._class_problem(driver, active, case["eris"], np.arange(6), "all",
                                      np.array([-10., -8.]), np.array([7., 9.]),
                                      case["active_energy"]):
            left = [3 | (1 << 2), 1 | (1 << 2) | (1 << 3)]
            right = [(1 << 5) | (1 << 6), (1 << 5) | (1 << 8)]
            determinants = [left[0] | right[1], left[1] | right[0], left[1] | right[1]]
            occ = np.array([[(d >> j) & 1 for j in range(10)] for d in determinants], dtype=np.uint8)
            native = driver.get_mps_from_csf_coefficients(
                occ, np.ones(3, complex) / np.sqrt(3), tag=response._response_tag(), iprint=0)
            try:
                raw = response._active_mps(driver, native)
            finally:
                response._release_response_mps(driver, native)
            assert uc._norm(uc._cas_part(raw, 2, 2)) < 1e-14
            raw.canonicalize(5)
            truncated, rotation, _ = raw.tensors[5].right_compress(k=1, cutoff=0.)
            raw.tensors[5] = truncated
            raw.tensors[4].right_multiply(rotation)
            assert uc._norm(uc._cas_part(raw, 2, 2)) > .1
            # This is the same constraint stage called immediately after
            # each native two-site SVD in the production update loop.
            kept, audit = uc._truncate_external(raw, 64, 0., 2, 2)
            assert audit["raw_truncation_cas_norm"] > .1
            assert audit["retained_cas_relative_norm"] < 1e-13
            assert uc._norm(uc._cas_part(kept, 2, 2)) < 1e-13
    finally:
        solver.close()


def test_source_phase_weak_source_and_zero_source(tmp_path, monkeypatch):
    from nevpt2_mps_response.diagnose_r85 import mps_vector
    from socutils.mrpt import nevpt2_mps_response as response
    case = tiny_case()
    solver = solve_reference(case, tmp_path / "mps")
    try:
        driver, ket = u._root_ket(solver, 0)
        active = response._active_mps(driver, ket)
        original_terms = response._source_tensors
        original_linear = uc._linear_response
        vectors = []
        def capture(driver, state, source, *args):
            result = original_linear(driver, state, source, *args)
            vectors.append(mps_vector(driver, state, state.n_sites, int(driver.target.n)))
            return result
        monkeypatch.setattr(uc, "_linear_response", capture)
        core, virtual = np.array([-10., -8.]), np.array([7., 9.])
        baseline = None
        for alpha in (1., np.exp(.71j), 2.8e-4 * np.exp(-.4j), 1e-12j, 0.):
            def terms(*args):
                return [(op, alpha * tensor, labels) for op, tensor, labels in original_terms(*args)]
            monkeypatch.setattr(response, "_source_tensors", terms)
            energy, d = uc._solve_class(driver, active, case["eris"], np.arange(6), "r",
                                        core, virtual, case["active_energy"], uc.UCControls(n_sweeps=6, diagnostic=True), 64)
            if alpha == 0:
                assert energy == 0 and d["zero_source"]
                continue
            x = np.sqrt(d["source_norm2"]) * vectors[-1]
            if baseline is None:
                baseline = energy, d["source_norm2"], x
            assert energy / abs(alpha)**2 == pytest.approx(baseline[0], abs=1e-11)
            assert d["source_norm2"] / abs(alpha)**2 == pytest.approx(baseline[1], abs=1e-12)
            np.testing.assert_allclose(x / alpha, baseline[2], atol=1e-11, rtol=1e-10)
            assert d["global_relative_residual"] < 1e-8
    finally:
        solver.close()


def test_native_default_no_source_mps_and_strict_crosscheck(tmp_path, monkeypatch):
    """Default calls native solve; full source materialization is opt-in only."""
    case = tiny_case()
    solver = solve_reference(case, tmp_path / "mps")
    core, virtual = np.array([-10., -8.]), np.array([7., 9.])
    exact = exact_uc(case, core, virtual)
    native_solve = uc._linear_response
    measured = []

    def independent_residual(driver, state, reference, dyall, source, controls, bond, iprint):
        from pyscf.fci import cistring
        from nevpt2_mps_response.diagnose_r85 import mps_vector
        from test_x2cficnevpt2 import _hamiltonian_action
        result = native_solve(driver, state, reference, dyall, source, controls, bond, iprint)
        bits = np.sum(1 << cistring.gen_occslst(range(10), 5), axis=1)
        ref, x = np.zeros(1024, complex), np.zeros(1024, complex)
        ref[bits] = mps_vector(driver, reference, 10, 5)
        ref /= np.linalg.norm(ref)
        # Only the solver's scalar amplitude convention is reused. H|ref>
        # and every Dyall block below use independent fermionic matrices.
        scale = max(np.max(abs(t)) for _, t, _ in uc.response._source_tensors(
            case["eris"], np.arange(6), "all"))
        # Native block2main sign: state is Psi1/scale, while this independent
        # oracle uses A x=b with x=-Psi1.
        x[bits] = -scale * mps_vector(driver, state, 10, 5)
        eris = case["eris"]
        ecore = .5 * np.trace((eris.h1e + eris.h1eff)[:2, :2]).real
        b = _hamiltonian_action(ref, case["h1e"], case["w"]) - (ecore + case["active_energy"]) * ref
        ax = np.zeros_like(x)
        for d in exact.values():
            for indices, matrix, _ in d["patterns"]:
                ax[indices] = matrix @ x[indices]
        active_bits = np.array([a for a in range(64) if a.bit_count() == 3])
        cas = 3 | (active_bits << 2)
        ha = _active_hamiltonian_matrix(case["active_h"], case["active_w"])
        ax[cas] = (ha[np.ix_(active_bits, active_bits)] - np.eye(len(cas)) * case["active_energy"]) @ x[cas]
        measured.append(dict(rho=np.linalg.norm(ax - b) / np.linalg.norm(b),
                             cas_ratio=np.linalg.norm(b[cas]) / np.linalg.norm(b),
                             energy=float((np.vdot(x, ax) - 2 * np.vdot(x, b)).real)))
        return result

    monkeypatch.setattr(uc, "_linear_response", independent_residual)
    try:
        driver, ket = u._root_ket(solver, 0)
        active = uc.response._active_mps(driver, ket)
        arguments = (driver, active, case["eris"], np.arange(6), "all",
                     core, virtual, case["active_energy"])
        def forbidden(*args, **kwargs):
            raise AssertionError("production must not materialize a source or impose Q")
        with monkeypatch.context() as patch:
            expectation = driver.expectation
            def no_variance(*args, **kwargs):
                assert kwargs.get("stacked_mpo") is None, "production must not compute a residual variance"
                return expectation(*args, **kwargs)
            patch.setattr(driver, "expectation", no_variance)
            patch.setattr(uc, "_algebra_mpo", forbidden)
            patch.setattr(uc, "_external_part", forbidden)
            patch.setattr(uc, "_constrained_linear_response", forbidden)
            patch.setattr(uc.response, "_class_problem", forbidden)
            energy, d = uc._solve_class(*arguments, uc.UCControls(n_sweeps=6), 64)
        assert energy == pytest.approx(sum(x["energy"] for x in exact.values()), abs=1e-10)
        assert d["solver"] == "pyblock2.Linear.solve/Automatic"
        assert d["global_relative_residual"] is None and not d["converged"]
        assert np.isfinite(d["reference_zero_mode_leakage"])
        assert d["cas_residual_squared_raw"] is None and d["cas_source_norm"] is None
        for key in u.SUBSPACE_ORDER:
            assert d["classes"][key]["hylleraas_energy"] == pytest.approx(exact[key]["energy"], abs=1e-10)
        audited, audit = uc._solve_class(*arguments, uc.UCControls(n_sweeps=6, diagnostic=True), 64)
        assert len(measured) == 2
        assert abs(audit["global_relative_residual"] - measured[-1]["rho"]) < 1e-11
        assert abs(audit["cas_source_relative_norm"] - measured[-1]["cas_ratio"]) < 1e-11
        assert audited == pytest.approx(measured[-1]["energy"], abs=1e-11)
        strict, sd = uc._solve_class(*arguments, uc.UCControls(
            n_sweeps=6, strict_cas=True, diagnostic=True), 64)
        print("UC_NATIVE_DEFAULT", json.dumps(dict(energy=energy, audited_energy=audited,
            strict_energy=strict, rho=audit["global_relative_residual"],
            cas_source_ratio=audit["cas_source_relative_norm"],
            zero_mode=audit["reference_zero_mode_leakage"],
            cas_energy=audit["cas_hylleraas_energy"])))
        assert audited == pytest.approx(strict, abs=1e-10)
        assert audit["global_relative_residual"] < 1e-8
        assert sd["global_relative_residual"] < 1e-8
    finally:
        solver.close()


def test_public_api_no_high_rdms_and_fock_guard(tmp_path, monkeypatch):
    from socutils.mrpt import X2CUCNEVPT2, x2cficnevpt2, x2cscnevpt2
    case = tiny_case()
    solver = solve_reference(case, tmp_path / "mps")
    eris = case["eris"]
    eps = np.r_[[-10., -8.], np.zeros(6), [7., 9.]]
    ecore = .5 * np.trace((eris.h1e + eris.h1eff)[:2, :2]).real
    eref = ecore + case["active_energy"] + 2.
    mol = SimpleNamespace(energy_nuc=lambda: 2.)
    mf = SimpleNamespace(mol=mol, get_ovlp=lambda: np.eye(10))
    mc = SimpleNamespace(fcisolver=solver, verbose=0, stdout=sys.stdout, _scf=mf, mol=mol,
                         ncore=2, ncas=6, nelecas=3, mo_coeff=np.eye(10), mo_energy=eps,
                         e_tot=eref + 1., e_states=[eref],
                         get_fock=lambda **kwargs: np.diag(eps))
    def forbidden(*args, **kwargs):
        raise AssertionError("full UC must not request high RDMs or use IC energies")
    for name in ("get_3pdm", "get_4pdm"):
        monkeypatch.setattr(solver.driver, name, forbidden)
    monkeypatch.setattr(x2cscnevpt2, "_evaluate_wick_subspaces", forbidden)
    monkeypatch.setattr(x2cficnevpt2, "_evaluate_fic_subspaces", forbidden)
    monkeypatch.setattr(u, "make_dm1234", forbidden)
    try:
        from nevpt2_mps_response.diagnose_r85 import mps_vector
        reference_driver = solver.driver
        reference_frame = reference_driver.frame
        reference_state = reference_driver.__dict__.copy()
        reference_rdm = u._make_rdm(solver, 0, 1)
        def forbidden_reconfiguration(*args, **kwargs):
            raise AssertionError("UC must not reinitialize the MCSCF driver")
        monkeypatch.setattr(reference_driver, "initialize_system", forbidden_reconfiguration)
        before = mps_vector(solver.driver, solver.kets[0], 6, 3)
        pt = X2CUCNEVPT2(mc)
        pt.canonicalized = True
        pt.scratch, pt.stack_memory, pt.n_threads = str(tmp_path), 256, 1
        pt.mps_response_options = dict(max_bond_dimension=64, n_sweeps=6, diagnostic=True)
        monkeypatch.setattr(u, "_dense_eris_from_mc", lambda *args, **kwargs: eris)
        pt.run(root=0)
        assert pt.converged
        assert pt.e_tot == pytest.approx(eref + pt.e_corr, abs=1e-13)
        assert set(pt.sub_eners) == set(u.SUBSPACE_ORDER)
        assert pt.diagnostics["global_relative_residual"] < 1e-8
        assert abs(pt.diagnostics["reference_energy_difference"]) < 1e-11
        np.testing.assert_allclose(mps_vector(solver.driver, solver.kets[0], 6, 3), before, atol=1e-13)
        np.testing.assert_array_equal(mc.mo_coeff, np.eye(10))
        assert solver.driver is reference_driver
        assert reference_driver.bw.b.Global.frame is reference_frame
        for name, value in reference_state.items():
            assert reference_driver.__dict__[name] is value, name
        np.testing.assert_allclose(u._make_rdm(solver, 0, 1), reference_rdm, atol=1e-13)
        with pytest.raises(TypeError, match="full _SpinorERIs"):
            uc.evaluate_uc(mc, u._compact_wick_eris(eris), case["pdms"][:2],
                           eps[:2], eps[8:])
        assert reference_driver.bw.b.Global.frame is reference_frame
        from socutils.mrpt.spinor_helper import init_eris
        zero_eris = init_eris(np.zeros_like(eris.h1e), np.zeros_like(eris.pppp), 2, 6)
        zero, zero_diagnostics = uc.evaluate_uc(mc, zero_eris, case["pdms"][:2], eps[:2], eps[8:])
        assert all(e == 0. for e in zero.values())
        assert all(d["zero_source"] and d["converged"] for d in zero_diagnostics.values())
        pt_directories = []
        def injected_failure(driver, *args, **kwargs):
            assert driver is not reference_driver and driver.frame is not reference_frame
            assert driver.scratch != reference_driver.scratch
            pt_directories.append(driver.scratch)
            raise RuntimeError("injected PT failure")
        with monkeypatch.context() as patch:
            patch.setattr(uc, "_solve_class", injected_failure)
            with pytest.raises(RuntimeError, match="injected PT failure"):
                pt.kernel(eris=eris)
        assert reference_driver.bw.b.Global.frame is reference_frame
        assert all(not uc.response.Path(path).exists() for path in pt_directories)
        np.testing.assert_allclose(u._make_rdm(solver, 0, 1), reference_rdm, atol=1e-13)
        # A source-free sector can have response leakage. Its infinite
        # relative residual must remain visible, not become 0*inf -> NaN.
        entries = {key: dict(source_norm2=0., residual_norm2=0.,
                             global_relative_residual=0., converged=True)
                   for key in u.SUBSPACE_ORDER}
        entries["rs"].update(source_norm2=1.)
        entries["i"].update(residual_norm2=.01, global_relative_residual=float("inf"), converged=False)
        with monkeypatch.context() as patch:
            patch.setattr(uc, "evaluate_uc", lambda *args, **kwargs: (
                {key: 0. for key in u.SUBSPACE_ORDER}, entries))
            with pytest.warns(u.MRPTNumericalWarning, match="not certified"):
                pt.kernel(eris=eris, pdms=case["pdms"][:2])
            assert pt.diagnostics["global_relative_residual"] == pytest.approx(.1)
            assert pt.diagnostics["max_channel_relative_residual"] == float("inf")
            assert not pt.converged
        bad_fock = np.diag(eps)
        bad_fock[0, 1] = bad_fock[1, 0] = .01
        mc.get_fock = lambda **kwargs: bad_fock
        with pytest.raises(ValueError, match="semicanonical"):
            pt.kernel(eris=eris)
        assert not pt.converged
    finally:
        solver.close()


def test_qc_mpo_complex_spinor_actions_and_response(tmp_path):
    """Full-H native FastBipartite input, with block2main's L/R signs."""
    from pyscf.fci import cistring
    from nevpt2_mps_response.diagnose_r85 import mps_vector
    from test_x2cficnevpt2 import _hamiltonian_action
    from pyblock2.algebra.io import MPSTools
    from itertools import combinations
    from nevpt2_mps_response.uc_audit import SectorCoefficients, Sources, active_bits, measure_response
    case = tiny_case()
    solver = solve_reference(case, tmp_path / "mps")
    eris = case["eris"]
    core, virtual = np.array([-10., -8.]), np.array([7., 9.])
    hd = np.diag(np.r_[core, np.zeros(6), virtual]).astype(complex)
    hd[2:8, 2:8] = case["active_h"]
    gd = np.zeros((10,) * 4, complex)
    gd[2:8, 2:8, 2:8, 2:8] = eris.get_chem("AAAA")
    shift = -case["active_energy"] - sum(core)
    ecore = .5 * np.trace((eris.h1e + eris.h1eff)[:2, :2]).real
    bits = np.sum(1 << cistring.gen_occslst(range(10), 5), axis=1)
    def vector(driver, state):
        x = np.zeros(1024, complex)
        x[bits] = mps_vector(driver, state, 10, 5)
        return x
    def check_pattern_residuals(measured, residual, source):
        maxima = {}
        for key in (*u.SUBSPACE_ORDER, "CAS"):
            nh = sum(key.count(x) for x in "ij") if key != "CAS" else 0
            npart = sum(key.count(x) for x in "rs") if key != "CAS" else 0
            ratios = {}
            zero_residual2 = 0.
            for holes in combinations(range(2), nh):
                for particles in combinations(range(2), npart):
                    ix = ((3 ^ sum(1 << i for i in holes))
                          | (active_bits(6, 3 + nh - npart) << 2)
                          | (sum(1 << r for r in particles) << 8))
                    norm = np.linalg.norm(source[ix])
                    if norm:
                        ratios[holes, particles] = np.linalg.norm(residual[ix]) / norm
                    else:
                        zero_residual2 += np.linalg.norm(residual[ix])**2
            entry = measured["classes"][key]
            maxima[key] = max(ratios.values(), default=0.)
            assert entry["nonzero_patterns"] == len(ratios)
            assert entry["zero_source_residual_norm2"] == pytest.approx(zero_residual2, abs=1e-11)
            # The tiny CAS source is cancellation-limited, not the external
            # channel oracle. Its overall residual is checked separately.
            if key != "CAS":
                assert entry["max_pattern_relative_residual"] == pytest.approx(maxima[key], abs=1e-11)
                worst = entry["worst_pattern"]
                if worst:
                    pattern = tuple(worst["holes"]), tuple(worst["particles"])
                    assert ratios[pattern] == pytest.approx(maxima[key], abs=1e-11)
        assert measured["max_external_pattern_relative_residual"] == pytest.approx(
            max(value for key, value in maxima.items() if key != "CAS"), abs=1e-11)
    try:
        driver, ket = u._root_ket(solver, 0)
        active = uc.response._active_mps(driver, ket)
        driver.initialize_system(n_sites=10, n_elec=5)
        driver.reorder_idx = None
        ref = uc.response._embed_reference(driver, active, 2, 2, cas_info=True)
        try:
            # Exactly block2main's signs: L = E0-HD, R = H-E0.
            qc_l = uc._qc_mpo(driver, -hd, -gd, -shift)
            qc_r = uc._qc_mpo(driver, eris.h1e, eris.pppp, -case["active_energy"] - ecore)
            # Wrap once, as in production. IdentityAddedMPO shares operator
            # tensors; repeatedly wrapping the same primitive is not safe.
            native_l = driver.bw.bs.IdentityAddedMPO(qc_l)
            native_r = driver.bw.bs.IdentityAddedMPO(qc_r)
            ref_vector = vector(driver, ref)
            exact_b = _hamiltonian_action(ref_vector, eris.h1e, case["w"]) - (
                case["active_energy"] + ecore) * ref_vector
            trial = uc.response._nevpt_mps(driver, 2, 2, "all", 64)
            try:
                trial_vector = vector(driver, trial)
                coefficients = SectorCoefficients(uc.response._active_mps(driver, trial), 2, 2, 5)
                ref_coefficients = SectorCoefficients(uc.response._active_mps(driver, ref), 2, 2, 5).values((), ())
                sources = Sources(eris, np.arange(6), ref_coefficients, 3)
                for key in (*u.SUBSPACE_ORDER, "CAS"):
                    nh = sum(key.count(x) for x in "ij") if key != "CAS" else 0
                    npart = sum(key.count(x) for x in "rs") if key != "CAS" else 0
                    for holes in combinations(range(2), nh):
                        for particles in combinations(range(2), npart):
                            ix = ((3 ^ sum(1 << i for i in holes))
                                  | (active_bits(6, 3 + nh - npart) << 2)
                                  | (sum(1 << r for r in particles) << 8))
                            np.testing.assert_allclose(coefficients.values(holes, particles), trial_vector[ix], atol=1e-13)
                            if key != "CAS":
                                np.testing.assert_allclose(sources.values(key, holes, particles), exact_b[ix], atol=1e-12)
                exact_a = _hamiltonian_action(trial_vector, hd, gd.transpose(0, 2, 1, 3)) + shift * trial_vector
                trial_measured = measure_response(driver, trial, ref, native_l, native_r, eris,
                                                  np.arange(6), core, virtual, case["active_energy"])
                check_pattern_residuals(trial_measured, -exact_a - exact_b, exact_b)
                exact_r_trial = _hamiltonian_action(trial_vector, eris.h1e, case["w"]) - (
                    case["active_energy"] + ecore) * trial_vector
                for name, mpo, native, ket, expected in (("QC-R-CAS", qc_r, native_r, ref, exact_b),
                                                        ("QC-R-full", qc_r, native_r, trial, exact_r_trial),
                                                        ("QC-L", qc_l, native_l, trial, -exact_a)):
                    # Exact tensor-network application, not a fitted source
                    # with a separate compression error. Also check native
                    # contraction, independently of the algebra exporter.
                    product = uc._algebra_mpo(mpo) @ uc.response._active_mps(driver, ket)
                    image = MPSTools.to_block2(product, driver.basis, center=0,
                                               tag=uc.response._response_tag())
                    try:
                        print("UC_QC_ACTION", name, flush=True)
                        np.testing.assert_allclose(vector(driver, image), expected, atol=1e-11, rtol=1e-11)
                        np.testing.assert_allclose(driver.expectation(
                            trial, native, ket),
                            np.vdot(trial_vector, expected), atol=1e-12, rtol=1e-12)
                    finally:
                        uc.response._release_response_mps(driver, image)
            finally:
                uc.response._release_response_mps(driver, trial)
            state = uc.response._nevpt_mps(driver, 2, 2, "all", 64)
            try:
                uc._linear_response(driver, state, ref, native_l, native_r, uc.UCControls(n_sweeps=6), 64, 0)
                measured = measure_response(driver, state, ref, native_l, native_r, eris,
                                            np.arange(6), core, virtual, case["active_energy"])
                psi1 = vector(driver, state)
                apsi1 = _hamiltonian_action(psi1, hd, gd.transpose(0, 2, 1, 3)) + shift * psi1
                check_pattern_residuals(measured, -apsi1 - exact_b, exact_b)
                rho = np.linalg.norm(-apsi1 - exact_b) / np.linalg.norm(exact_b)
                assert rho < 1e-8
                energy = float((np.vdot(psi1, apsi1) + 2 * np.vdot(psi1, exact_b)).real)
                assert measured["global_relative_residual"] == pytest.approx(rho, abs=1e-11)
                assert measured["hylleraas_energy"] == pytest.approx(energy, abs=1e-12)
                assert energy == pytest.approx(sum(d["energy"] for d in exact_uc(case, core, virtual).values()), abs=1e-10)
                print("UC_QC_MPO", json.dumps(dict(energy=energy, residual=rho)))
            finally:
                uc.response._release_response_mps(driver, state)
        finally:
            uc.response._release_response_mps(driver, ref)
    finally:
        solver.close()


@pytest.mark.parametrize("strict_cas,response_mode,degenerate_external", [
    (False, "full_chain", False), (True, "full_chain", False),
    (False, "external_tuples", False), (False, "external_tuples", True)])
def test_complex_orbital_gauge_with_fixed_reference_and_semicanonicalization(
        tmp_path, strict_cas, response_mode, degenerate_external):
    """Rotate integrals AND CI; external Fock mixing must be diagonalized.

    The determinant exterior power is a tiny independent test oracle only.
    Neither spin pairing nor an alpha/beta representation enters the PT API.
    """
    from itertools import combinations
    from types import MethodType
    from socutils.mcscf import zmcscf
    from test_x2cscnevpt2_wick import _random_unitary, _raw_active_rdms, _hamiltonian_action
    case = tiny_case()
    rng = np.random.default_rng(41551)
    occupations = np.array(list(combinations(range(6), 3)))
    bits = np.sum(1 << occupations, axis=1)
    dets = np.zeros((len(bits), 6), dtype=np.uint8)
    dets[np.arange(len(bits))[:, None], occupations] = 1
    active_rotation = _random_unitary(rng, 6)
    rotations = [np.eye(10, dtype=complex)]
    active_only = rotations[0].copy()
    active_only[2:8, 2:8] = active_rotation
    rotations.append(active_only)
    all_blocks = active_only.copy()
    all_blocks[:2, :2] = _random_unitary(rng, 2)
    all_blocks[8:, 8:] = _random_unitary(rng, 2)
    rotations.append(all_blocks)
    eps = (np.r_[[-9., -9.], np.zeros(6), [8., 8.]] if degenerate_external
           else np.r_[[-10., -8.], np.zeros(6), [7., 9.]])
    ecore = .5 * np.trace((case["eris"].h1e + case["eris"].h1eff)[:2, :2]).real
    eref = ecore + case["active_energy"]
    expected = sum(d["energy"] for d in exact_uc(case, eps[:2], eps[8:]).values())
    solver = solve_reference(case, tmp_path / "mps")
    original_kets = solver.kets
    original_order = solver.driver.reorder_idx
    np.testing.assert_array_equal(original_order, np.arange(6))
    # The native coefficient importer requires no pending reorder operation.
    solver.driver.reorder_idx = None
    measured = {4: [], 64: []}
    try:
        for rotation in rotations:
            ua = rotation[2:8, 2:8]
            exterior = np.array([[np.linalg.det(ua[np.ix_(old, new)])
                                  for new in occupations] for old in occupations])
            np.testing.assert_allclose(exterior.conj().T @ exterior, np.eye(20), atol=1e-13)
            vector = np.zeros(64, complex)
            vector[bits] = exterior.conj().T @ case["active_state"][bits]
            eris = u._rotate_eris(case["eris"], rotation)
            action = _hamiltonian_action(vector, eris.get_h1eff("AA"), eris.get_phys("AAAA"))
            assert np.linalg.norm(action - case["active_energy"] * vector) < 1e-12
            pdms = _raw_active_rdms(vector, 6)[:2]
            native = solver.driver.get_mps_from_csf_coefficients(
                dets, vector[bits], tag=uc.response._response_tag(), dot=2, iprint=0)
            solver.kets = [native]
            try:
                mol = SimpleNamespace(energy_nuc=lambda: 0.)
                mf = SimpleNamespace(mol=mol, get_ovlp=lambda: np.eye(10))
                mc = SimpleNamespace(
                    fcisolver=solver, verbose=0, stdout=sys.stdout, _scf=mf, mol=mol,
                    ncore=2, ncas=6, nelecas=3, mo_coeff=rotation.copy(), mo_energy=None,
                    e_tot=eref, e_states=[eref], ci=native,
                    get_fock=lambda *args, **kwargs: np.diag(eps))
                mc.canonicalize = MethodType(zmcscf.canonicalize, mc)
                for bond in measured:
                    pt = uc.X2CUCNEVPT2(mc)
                    pt.response_mode = response_mode
                    pt.mps_response_options = dict(max_bond_dimension=bond, n_sweeps=4,
                                                   strict_cas=strict_cas, diagnostic=True)
                    if bond == 4:
                        with pytest.warns(Warning, match="full UC not certified"):
                            pt.kernel(eris=eris, pdms=pdms)
                    else:
                        pt.kernel(eris=eris, pdms=pdms)
                    measured[bond].append(dict(
                        energy=pt.e_corr, residual=pt.diagnostics["global_relative_residual"]))
                    assert pt.converged == (bond == 64)
                    assert pt.diagnostics["external_fock_error"] < 1e-12
                    np.testing.assert_array_equal(pt.mo_coeff[:, 2:8], rotation[:, 2:8])
                    np.testing.assert_array_equal(mc.mo_coeff, rotation)
                    if bond == 64:
                        assert pt.e_corr == pytest.approx(expected, abs=1e-10)
                        assert pt.diagnostics["global_relative_residual"] < 1e-8
                if rotation is all_blocks:
                    d = mc.canonicalization_diagnostics
                    if degenerate_external:
                        # Legal rotations within exactly degenerate blocks
                        # leave the external Fock diagonal without averaging.
                        assert max(d["core_offdiagonal_before"], d["virtual_offdiagonal_before"]) < 1e-12
                    else:
                        assert d["core_offdiagonal_before"] > .01
                        assert d["virtual_offdiagonal_before"] > .01
                    assert max(d["core_offdiagonal_after"], d["virtual_offdiagonal_after"]) < 1e-12
            finally:
                solver.kets = original_kets
                uc.response._release_response_mps(solver.driver, native)
        spread = {bond: np.ptp([d["energy"] for d in rows]) for bond, rows in measured.items()}
        assert spread[64] < 1e-10
        assert spread[64] < spread[4]
        print("UC_ORBITAL_GAUGE", json.dumps(dict(strict_cas=strict_cas,
              response_mode=response_mode, degenerate_external=degenerate_external, results=measured)))
    finally:
        solver.kets = original_kets
        solver.driver.reorder_idx = original_order
        solver.close()


@pytest.mark.parametrize("nelec", [0, 6])
def test_boundary_active_particle_numbers_and_empty_external_space(tmp_path, nelec):
    from socutils.mrpt import spinor_helper
    from test_x2cscnevpt2_wick import _embed_active_state, _raw_active_rdms, _hamiltonian_action
    case = dict(tiny_case())
    state = np.zeros(64, complex)
    state[(1 << nelec) - 1] = 1.
    case["active_state"] = state
    case["active_energy"] = float(np.vdot(state, _hamiltonian_action(
        state, case["active_h"], case["active_w"])).real)
    case["reference"] = _embed_active_state(state, 2, 6, 2)
    case["h_reference"] = _hamiltonian_action(case["reference"], case["h1e"], case["w"])
    pdms = _raw_active_rdms(state, 6)[:2]
    core, virtual = np.array([-10., -8.]), np.array([7., 9.])
    expected = exact_uc(case, core, virtual)
    solver = solve_reference(case, tmp_path / "mps", nelec=nelec)
    mc = SimpleNamespace(fcisolver=solver, verbose=0, stdout=sys.stdout)
    try:
        energies, diagnostics = uc.evaluate_uc(
            mc, case["eris"], pdms, core, virtual,
            options=dict(max_bond_dimension=64, n_sweeps=6, strict_cas=True, diagnostic=True))
        print("UC_BOUNDARY", nelec, json.dumps({k: {name: d[name] for name in (
            "source_norm2", "hylleraas_energy", "global_relative_residual")}
            for k, d in diagnostics.items()}))
        for key in u.SUBSPACE_ORDER:
            assert energies[key] == pytest.approx(expected[key]["energy"], abs=1e-10)
            assert diagnostics[key]["global_relative_residual"] < 1e-8, key
        cas_only = spinor_helper.init_eris(case["active_h"], case["eris"].get_chem("AAAA"), 0, 6)
        for blocked in (False, True):
            energies, diagnostics = uc.evaluate_uc(mc, cas_only, pdms, [], [], class_resolved=blocked)
            assert all(value == 0. for value in energies.values())
            assert all(d["converged"] and d["source_norm2"] == 0. for d in diagnostics.values())
    finally:
        solver.close()


def test_unitary_d3_doublet_sc_fic_hybrids_and_full_uc(tmp_path):
    """Controlled ordinary unitary symmetry, not a Kramers/PT restriction.

    Project the model integrals onto D3 symmetry ONCE when defining the
    fixture. All methods then use those same integrals and two exact CAS
    states in its E irrep. No calculated energy is averaged or projected.
    The production solver runs C1 SGFCPX and receives no symmetry operators.
    """
    from itertools import combinations
    from socutils.mrpt import spinor_helper, x2cscnevpt2 as sc, x2cficnevpt2 as fic
    from test_x2cscnevpt2_wick import _embed_active_state, _raw_active_rdms, _hamiltonian_action
    angle = 2 * np.pi / 3
    r = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    s = np.diag([1., -1.])
    np.testing.assert_allclose(r @ r @ r, np.eye(2), atol=1e-14)
    np.testing.assert_allclose(s @ r @ s, r.T, atol=1e-14)
    group = [np.eye(2), r, r @ r, s, s @ r, s @ r @ r]
    images = [u._rotate_eris(tiny_case()["eris"], np.kron(np.eye(5), g)) for g in group]
    eris = spinor_helper.init_eris(sum(x.h1e for x in images) / 6,
                                  sum(x.pppp for x in images) / 6, 2, 6)
    assert np.max(abs(eris.h1e.imag)) > 1e-3
    for g in (r, s):
        rotated = u._rotate_eris(eris, np.kron(np.eye(5), g))
        np.testing.assert_allclose(rotated.h1e, eris.h1e, atol=1e-13)
        np.testing.assert_allclose(rotated.pppp, eris.pppp, atol=1e-13)
    occupations = np.array(list(combinations(range(6), 3)))
    bits = np.sum(1 << occupations, axis=1)
    def exterior(g):
        ua = np.kron(np.eye(3), g)
        return np.array([[np.linalg.det(ua[np.ix_(a, b)]) for b in occupations] for a in occupations])
    rotation, reflection = exterior(r), exterior(s)
    ham = _active_hamiltonian_matrix(eris.get_h1eff("AA"), eris.get_phys("AAAA"))[np.ix_(bits, bits)]
    np.testing.assert_allclose(rotation.conj().T @ ham @ rotation, ham, atol=1e-13)
    np.testing.assert_allclose(reflection.conj().T @ ham @ reflection, ham, atol=1e-13)
    eigenvalues, eigenvectors = np.linalg.eigh(ham)
    projector_e = (2 * np.eye(20) - rotation - rotation @ rotation) / 3
    root = next(j for j in range(20) if np.linalg.norm(projector_e @ eigenvectors[:, j]) > .9)
    doublet = eigenvectors[:, np.abs(eigenvalues - eigenvalues[root]) < 1e-10]
    assert doublet.shape[1] == 2
    _, parity = np.linalg.eigh(doublet.conj().T @ reflection @ doublet)
    first = doublet @ parity[:, 1]
    second = (np.eye(20) - reflection) @ rotation @ first
    second /= np.linalg.norm(second)
    assert abs(np.vdot(first, second)) < 1e-13
    assert np.linalg.norm(ham @ second - eigenvalues[root] * second) < 1e-13
    core, virtual = np.array([-9., -9.]), np.array([8., 8.])
    # Equal eigenvalues within these explicitly specified symmetry blocks
    # make the fixed Dyall Hamiltonian covariant as well as the full H.
    dets = np.zeros((20, 6), dtype=np.uint8)
    dets[np.arange(20)[:, None], occupations] = 1
    solver = solve_reference(tiny_case(), tmp_path / "mps")
    original_kets, original_order = solver.kets, solver.driver.reorder_idx
    np.testing.assert_array_equal(original_order, np.arange(6))
    solver.driver.reorder_idx = None
    results = []
    try:
        for coefficients in (first, second):
            state = np.zeros(64, complex)
            state[bits] = coefficients
            reference = _embed_active_state(state, 2, 6, 2)
            pdms = _raw_active_rdms(state, 6)
            case = dict(eris=eris, active_state=state, active_h=eris.get_h1eff("AA"),
                        active_w=eris.get_phys("AAAA"), active_energy=float(eigenvalues[root]),
                        h_reference=_hamiltonian_action(reference, eris.h1e, eris.get_phys("PPPP")))
            exact = exact_uc(case, core, virtual)
            native = solver.driver.get_mps_from_csf_coefficients(
                dets, coefficients, tag=uc.response._response_tag(), dot=2, iprint=0)
            solver.kets = [native]
            try:
                mc = SimpleNamespace(fcisolver=solver, verbose=0, stdout=sys.stdout,
                                     ncore=2, ncas=6, nelecas=3)
                full_uc, diagnostics = uc.evaluate_uc(mc, eris, pdms[:2], core, virtual,
                    options=dict(max_bond_dimension=64, n_sweeps=6, strict_cas=True, diagnostic=True))
                native_uc, native_diagnostics = uc.evaluate_uc(mc, eris, pdms[:2], core, virtual,
                    options=dict(max_bond_dimension=64, n_sweeps=6, diagnostic=True))
                joint = next(iter(native_diagnostics.values()))["joint_response"]
                assert joint["hylleraas_energy"] == pytest.approx(
                    sum(d["energy"] for d in exact.values()), abs=1e-10)
                assert joint["global_relative_residual"] < 1e-8
                full_sc = sc._evaluate_wick_subspaces(eris, pdms, core, virtual)[0]
                full_fic = fic._evaluate_fic_subspaces(eris, pdms, core, virtual)
                hybrid = uc.response.evaluate_mps_response(mc, eris, pdms[:3], core, virtual,
                    options=dict(max_bond_dimension=64, n_sweeps=8, tol=0.,
                                 linear_threshold=1e-24, noise=0., cutoff=1e-24))[0]
                for key in u.SUBSPACE_ORDER:
                    assert full_uc[key] == pytest.approx(exact[key]["energy"], abs=1e-10)
                    assert diagnostics[key]["global_relative_residual"] < 1e-8
                    assert native_uc[key] == pytest.approx(exact[key]["energy"], abs=1e-10)
                    assert native_diagnostics[key]["global_relative_residual"] < 1e-8
                for key in ("i", "r"):
                    assert hybrid[key] == pytest.approx(full_uc[key], abs=1e-10)
                results.append({"SC": full_sc, "FIC": full_fic,
                                "SC+UC(i,r)": {**full_sc, **hybrid},
                                "FIC+UC(i,r)": {**full_fic, **hybrid}, "UC": full_uc,
                                "UC-native": {**native_uc, "CAS": joint["cas_hylleraas_energy"]}})
            finally:
                solver.kets = original_kets
                uc.response._release_response_mps(solver.driver, native)
        for key in u.SUBSPACE_ORDER:
            assert results[0]["UC"][key] == pytest.approx(results[1]["UC"][key], abs=1e-10)
            assert results[0]["UC-native"][key] == pytest.approx(results[1]["UC-native"][key], abs=1e-10)
        assert sum(results[0]["UC-native"].values()) == pytest.approx(
            sum(results[1]["UC-native"].values()), abs=1e-10)
        print("UC_D3_DOUBLET", json.dumps(dict(
            reference_energy=float(eigenvalues[root]), classes=results,
            totals=[{k: sum(v.values()) for k, v in row.items()} for row in results])))
    finally:
        solver.kets, solver.driver.reorder_idx = original_kets, original_order
        solver.close()


def test_time_reversal_coefficients_match_independent_exterior_power():
    """Check the F-only overlap diagnostic on a dense complex one-body map."""
    from pyscf.fci import cistring
    from nevpt2_mps_response.uc_audit import time_reverse_coefficients
    rng = np.random.default_rng(76014)
    rotation, _ = np.linalg.qr(rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6)))
    j = np.kron(np.eye(3), np.array([[0., 1.], [-1., 0.]]))
    matrix = rotation.conj().T @ j @ rotation.conj()
    occupations = cistring.gen_occslst(range(6), 3)
    exterior = np.array([[np.linalg.det(matrix[np.ix_(a, b)])
                         for b in occupations] for a in occupations])
    states = rng.normal(size=(20, 2)) + 1j * rng.normal(size=(20, 2))
    reversed_states = time_reverse_coefficients(matrix, states, 3)
    np.testing.assert_allclose(reversed_states, exterior @ states.conj(), atol=1e-12, rtol=0.)
    np.testing.assert_allclose(time_reverse_coefficients(matrix, reversed_states, 3),
                               -states, atol=1e-12, rtol=0.)
    np.testing.assert_allclose(time_reverse_coefficients(matrix, states[:, 0], 3),
                               reversed_states[:, 0], atol=1e-12, rtol=0.)


def test_antiunitary_time_reversal_without_kramers_restriction(tmp_path):
    """T=J K defines the fixture, never a constraint supplied to the solver."""
    from itertools import combinations
    from socutils.mrpt import spinor_helper
    from test_x2cscnevpt2_wick import _embed_active_state, _raw_active_rdms, _hamiltonian_action

    def reverse(state):
        # Exterior power of J=[[0,1],[-1,0]], INCLUDING complex conjugation.
        result = np.zeros_like(state)
        for bits, coefficient in enumerate(state):
            occupied = [p for p in range(state.size.bit_length() - 1) if bits >> p & 1]
            mapped = [p ^ 1 for p in occupied]
            parity = sum(p % 2 == 0 for p in occupied) + sum(
                mapped[i] > mapped[j] for i in range(len(mapped)) for j in range(i + 1, len(mapped)))
            result[sum(1 << p for p in mapped)] = (-1)**parity * coefficient.conjugate()
        return result

    j = np.kron(np.eye(5), np.array([[0., 1.], [-1., 0.]]))
    original = tiny_case()["eris"]
    image = u._rotate_eris(original, j)
    eris = spinor_helper.init_eris((original.h1e + image.h1e.conj()) / 2,
                                  (original.pppp + image.pppp.conj()) / 2, 2, 6)
    image = u._rotate_eris(eris, j)
    np.testing.assert_allclose(image.h1e.conj(), eris.h1e, atol=1e-13)
    np.testing.assert_allclose(image.pppp.conj(), eris.pppp, atol=1e-13)
    assert np.max(abs(eris.pppp.imag)) > 1e-3
    occupations = np.array(list(combinations(range(6), 3)))
    bits = np.sum(1 << occupations, axis=1)
    ha = _active_hamiltonian_matrix(eris.get_h1eff("AA"), eris.get_phys("AAAA"))
    eig, vectors = np.linalg.eigh(ha[np.ix_(bits, bits)])
    first = np.zeros(64, complex)
    first[bits] = vectors[:, 0]
    second = reverse(first)
    np.testing.assert_allclose(reverse(second), -first, atol=1e-13)
    assert abs(np.vdot(first, second)) < 1e-13
    np.testing.assert_allclose(ha @ second, eig[0] * second, atol=1e-13)
    core, virtual = np.array([-9., -9.]), np.array([8., 8.])
    # Both members use the same T-invariant Dyall operator. The full core
    # pair and empty virtual pair are preserved; odd active N gives T^2=-1.
    dets = np.zeros((len(bits), 6), dtype=np.uint8)
    dets[np.arange(len(bits))[:, None], occupations] = 1
    solver = solve_reference(tiny_case(), tmp_path / "mps")
    original_kets, original_order = solver.kets, solver.driver.reorder_idx
    solver.driver.reorder_idx = None
    sources, results = [], []
    try:
        for state in (first, second):
            reference = _embed_active_state(state, 2, 6, 2)
            source = _hamiltonian_action(reference, eris.h1e, eris.get_phys("PPPP"))
            sources.append(source)
            case = dict(eris=eris, active_state=state, active_h=eris.get_h1eff("AA"),
                        active_w=eris.get_phys("AAAA"), active_energy=float(eig[0]), h_reference=source)
            exact = exact_uc(case, core, virtual)
            native = solver.driver.get_mps_from_csf_coefficients(
                dets, state[bits], tag=uc.response._response_tag(), dot=2, iprint=0)
            solver.kets = [native]
            try:
                mc = SimpleNamespace(fcisolver=solver, verbose=0, stdout=sys.stdout)
                energies, diagnostics = uc.evaluate_uc(mc, eris, _raw_active_rdms(state, 6)[:2],
                    core, virtual, options=dict(max_bond_dimension=64, n_sweeps=6, diagnostic=True))
                joint = next(iter(diagnostics.values()))["joint_response"]
                for key in u.SUBSPACE_ORDER:
                    assert energies[key] == pytest.approx(exact[key]["energy"], abs=1e-10)
                    assert diagnostics[key]["global_relative_residual"] < 1e-8
                assert joint["converged"]
                assert joint["hylleraas_energy"] == pytest.approx(
                    sum(d["energy"] for d in exact.values()), abs=1e-10)
                results.append(dict(energy=joint["hylleraas_energy"], classes=energies,
                                    residual=joint["global_relative_residual"]))
            finally:
                solver.kets = original_kets
                uc.response._release_response_mps(solver.driver, native)
        np.testing.assert_allclose(reverse(sources[0]), sources[1], atol=1e-13)
        for key in u.SUBSPACE_ORDER:
            assert results[0]["classes"][key] == pytest.approx(results[1]["classes"][key], abs=1e-10)
        assert results[0]["energy"] == pytest.approx(results[1]["energy"], abs=1e-10)
        print("UC_TIME_REVERSAL", json.dumps(results))
    finally:
        solver.kets, solver.driver.reorder_idx = original_kets, original_order
        solver.close()
