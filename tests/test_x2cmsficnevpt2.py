# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent complex fermionic IC-span oracle, not a UC comparison."""

import errno
import itertools
import json
import weakref
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from socutils.mrpt import spinor_helper
from socutils.mrpt import x2cficnevpt2 as fic
from socutils.mrpt import x2cmsficnevpt2 as ms
from socutils.mrpt._ic_metric import extended_dot
from test_x2cficnevpt2 import (
    _active_hamiltonian_matrix,
    _apply_active_hamiltonian,
    _fic_basis_vectors,
)
from test_x2cscnevpt2_wick import (
    _active_eigenstates,
    _apply_string,
    _embed_active_state,
    _hamiltonian_action,
    _random_physical_integrals,
)


def transition_pdms(bra, ket, ncas):
    result = []
    for rank in range(1, 5):
        tuples = tuple(itertools.product(range(ncas), repeat=rank))
        b = np.asarray([_apply_string(bra, tuple(("D", i) for i in t)) for t in tuples])
        k = np.asarray([_apply_string(ket, tuple(("D", i) for i in t)) for t in tuples])
        reversed_rows = (
            np.arange(ncas**rank)
            .reshape((ncas,) * rank)
            .transpose(tuple(reversed(range(rank))))
            .reshape(-1)
        )
        # Only the known output-N sector can be occupied (an oracle speedup,
        # not an approximate density or truncation).
        live = np.flatnonzero(np.any(b != 0, axis=0) | np.any(k != 0, axis=0))
        result.append(
            (b[reversed_rows][:, live].conj() @ k[:, live].T).reshape(
                (ncas,) * (2 * rank)
            )
        )
    return tuple(result)


@pytest.fixture(scope="module")
def data():
    nc, na, nv, ne, nmodel = 2, 5, 2, 4, 5
    h, g = _random_physical_integrals(nc + na + nv, seed=9721)
    eris = spinor_helper.init_eris(h, g, nc, na)
    energies, states = _active_eigenstates(
        eris.get_h1eff("AA"), eris.get_phys("AAAA"), ne
    )
    states = states[:nmodel]
    pdms = {
        (a, b): transition_pdms(states[a], states[b], na)
        for a in range(nmodel)
        for b in range(nmodel)
    }
    classes, eref, audit = ms.build_msfic_classes(
        eris,
        tuple(range(nmodel)),
        lambda a, b: pdms[a, b],
        np.eye(nmodel),
        contraction_backend="numpy",
    )
    np.testing.assert_allclose(eref, np.diag(energies), atol=1e-12)
    eps = np.r_[[-10.0, -8.0], np.zeros(na), [7.0, 9.0]]
    prepared = ms.MSFICPrepared(
        eris,
        np.eye(nc + na + nv),
        eps,
        np.mean([pdms[a, a][0] for a in range(nmodel)], axis=0),
        tuple(range(nmodel)),
        np.eye(nmodel),
        eref,
        eref,
        classes,
        audit,
    )
    refs = [_embed_active_state(s, nc, na, nv) for s in states]
    hrefs = np.column_stack(
        [_hamiltonian_action(r, h, eris.get_phys("PPPP")) for r in refs]
    )
    ha = _active_hamiltonian_matrix(eris.get_h1eff("AA"), eris.get_phys("AAAA"))
    return prepared, refs, hrefs, ha, pdms


def direct_basis(data, key, indices):
    p, refs, _hrefs, ha, _ = data
    columns = [_fic_basis_vectors(ref, p.eris, key, indices) for ref in refs]
    basis = np.concatenate(columns, axis=1)
    hb = _apply_active_hamiltonian(basis, ha, p.eris.ncore, p.eris.ncas, p.eris.nvirt)
    return columns, basis, hb


def test_all_eight_transition_matrices_non_degenerate_and_dm0(data):
    p, _refs, hrefs, _ha, pdms = data
    assert np.max(np.abs(pdms[0, 1][3])) > 1e-5
    assert np.linalg.norm(p.classes["r"].right - p.classes["r"].left) > 1e-3
    np.testing.assert_allclose(p.classes["ijrs"].metric, np.eye(5), atol=1e-12)
    for key, block in p.classes.items():
        for indices in fic._iter_free_tuples(key, p.eris):
            _, basis, hb = direct_basis(data, key, indices)
            s, f, v = (
                basis.conj().T @ basis,
                basis.conj().T @ hb,
                basis.conj().T @ hrefs,
            )
            np.testing.assert_allclose(block.metric, s, atol=2e-12)
            np.testing.assert_allclose(block.active, f, atol=2e-12)
            np.testing.assert_allclose(block.source[indices], v, atol=2e-12)
            e = np.repeat(np.diag(p.active_reference), block.dimension_per_reference)
            np.testing.assert_allclose(block.right, f - s * e[None, :], atol=2e-12)
            np.testing.assert_allclose(block.left, f - e[:, None] * s, atol=2e-12)


@pytest.mark.parametrize("ansatz", ["ss_sr", "ms_mr"])
@pytest.mark.parametrize("shift", [0.0, 0.2])
def test_heff_and_full_shift_norm_against_independent_ic_span(data, ansatz, shift):
    p, _refs, hrefs, ha, _pdms = data
    actual = ms.solve_msfic(p, ansatz=ansatz, shift=shift)
    y, norm = np.zeros((5, 5), complex), np.zeros((5, 5), complex)
    for key in p.classes:
        for indices in fic._iter_free_tuples(key, p.eris):
            columns, basis, _ = direct_basis(data, key, indices)
            delta = fic._orbital_gap_at(key, indices, p.mo_energy[:2], p.mo_energy[-2:])
            responses = []
            for root in range(5):
                span = basis if ansatz == "ms_mr" else columns[root]
                q, singular, _ = np.linalg.svd(span, full_matrices=False)
                q = q[:, singular > 1e-8]
                hq = _apply_active_hamiltonian(q, ha, 2, 5, 2)
                a = q.conj().T @ hq + (
                    delta + shift - p.active_reference[root, root]
                ) * np.eye(q.shape[1])
                responses.append(-q @ np.linalg.solve(a, q.conj().T @ hrefs[:, root]))
            t = np.column_stack(responses)
            y += hrefs.conj().T @ t
            norm += t.conj().T @ t
    oracle = p.reference + 0.5 * (y + y.conj().T) - shift * norm
    np.testing.assert_allclose(actual.heff, oracle, atol=3e-12)
    np.testing.assert_allclose(actual.shift_norm, norm, atol=3e-12)
    assert np.max(np.abs(norm - np.diag(np.diag(norm)))) > 1e-8
    assert all(
        b["maximum_ic_relative_residual"] < 1e-10
        for b in actual.diagnostics["classes"].values()
    )


@pytest.mark.parametrize("seed", [12, 91, 407])
def test_ms_mr_nondegenerate_u5_covariance(data, seed):
    p = data[0]
    rng = np.random.default_rng(seed)
    rotation = np.linalg.qr(rng.normal(size=(5, 5)) + 1j * rng.normal(size=(5, 5)))[0]
    a, b = ms.solve_msfic(p), ms.solve_msfic(p.rotated(rotation))
    np.testing.assert_allclose(
        b.heff, rotation.conj().T @ a.heff @ rotation, atol=3e-12
    )
    np.testing.assert_allclose(b.energies, a.energies, atol=3e-12)


@pytest.mark.parametrize("ansatz", ["ss_sr", "ms_mr"])
def test_negative_metric_and_null_source_are_errors(data, ansatz):
    p = data[0]
    block = p.classes["ijrs"]
    negative = block.metric.copy()
    negative[0, 0] = -0.01
    with pytest.raises(FloatingPointError, match="negative metric"):
        ms.solve_msfic(
            replace(p, classes={"ijrs": replace(block, metric=negative)}), ansatz=ansatz
        )
    zero = np.zeros_like(block.metric)
    with pytest.raises(FloatingPointError, match="discarded metric"):
        ms.solve_msfic(
            replace(p, classes={"ijrs": replace(block, metric=zero)}), ansatz=ansatz
        )


@pytest.mark.parametrize("ansatz", ["ss_sr", "ms_mr"])
def test_finite_ic_defects_warn_but_nonfinite_data_still_fail(data, ansatz):
    p = replace(data[0], classes={"ijrs": data[0].classes["ijrs"]})
    block = p.classes["ijrs"]
    expected = ms.solve_msfic(p, ansatz=ansatz)
    noisy = replace(block, active=block.active + 1e-7j * np.eye(len(block.active)))
    with pytest.warns(ms.u.MRPTNumericalWarning) as messages:
        result = ms.solve_msfic(replace(p, classes={"ijrs": noisy}), ansatz=ansatz)
    assert any("retained IC F" in str(w.message) for w in messages)
    assert any("IC relative residual" in str(w.message) for w in messages)
    assert result.diagnostics["classes"]["ijrs"]["maximum_ic_relative_residual"] > 1e-10
    np.testing.assert_allclose(result.energies, expected.energies, atol=1e-12, rtol=0)
    invalid = replace(block, active=np.full_like(block.active, np.nan))
    with pytest.raises(FloatingPointError, match="non-finite"):
        ms.solve_msfic(replace(p, classes={"ijrs": invalid}), ansatz=ansatz)


def test_metric_refinement_preserves_retained_span_and_rank():
    rng = np.random.default_rng(2)
    u = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))[0]
    s = (u * np.array([0.0, 1e-8, 0.4, 1.0])) @ u.conj().T
    s = 0.5 * (s + s.conj().T)
    args = {
        "metric_atol": 1e-12,
        "metric_rcond": 1e-11,
        "matrix_atol": 1e-10,
        "matrix_rtol": 1e-10,
    }
    old, old_null, before = ms._metric_basis(s, **args, metric_refinement=False)
    new, new_null, after = ms._metric_basis(s, **args)
    assert before["rank"] == after["rank"] == 3
    assert before["cutoff"] == after["cutoff"]
    assert after["gram_refinement_steps"] > 0
    np.testing.assert_allclose(
        new_null @ new_null.conj().T, old_null @ old_null.conj().T, atol=1e-14
    )
    old_gram = ms.congruence(old, s, extended=True)
    new_gram = ms.congruence(new, s, extended=True)
    assert np.max(np.abs(new_gram - np.eye(3))) < 1e-10
    assert (
        np.max(np.abs(new_gram - np.eye(3)))
        < np.max(np.abs(old_gram - np.eye(3))) / 100
    )


def test_parallel_extended_products_match_serial_accumulation():
    rng = np.random.default_rng(51)
    a = rng.normal(size=(260, 17)) + 1j * rng.normal(size=(260, 17))
    b = rng.normal(size=(17, 7)) + 1j * rng.normal(size=(17, 7))
    expected = extended_dot(a, b, threads=1)
    np.testing.assert_array_equal(extended_dot(a, b, threads=3), expected)
    np.testing.assert_array_equal(extended_dot(a, b[:, 0], threads=3), expected[:, 0])


def test_single_reference_matches_existing_fic(data):
    p, _, _, _, pdms = data
    classes, eref, _audit = ms.build_msfic_classes(
        p.eris,
        (0,),
        lambda a, b: pdms[a, b],
        np.eye(1),
        contraction_backend="numpy",
    )
    single = replace(
        p,
        model_roots=(0,),
        overlap=np.eye(1),
        active_reference=eref,
        reference=eref,
        classes=classes,
    )
    old = fic._evaluate_fic_subspaces(
        p.eris,
        pdms[0, 0],
        p.mo_energy[:2],
        p.mo_energy[-2:],
        contraction_backend="numpy",
    )
    for ansatz in ("ss_sr", "ms_mr"):
        result = ms.solve_msfic(single, ansatz=ansatz, shift=0)
        np.testing.assert_allclose(
            result.heff[0, 0] - eref[0, 0], sum(old.values()), atol=1e-12
        )


def test_manifold_restriction_matches_direct_transition_preparation(data):
    from tests.msficnevpt2.fluorine import _restrict_model

    p, _, _, _, pdms = data
    roots = (3, 1)  # Non-contiguous and reordered: test both source axes.
    restricted = _restrict_model(p, roots)
    assert restricted.eris is p.eris
    assert restricted.mo_coeff is p.mo_coeff
    assert restricted.mo_energy is p.mo_energy
    assert restricted.sa_density is p.sa_density
    classes, eref, audit = ms.build_msfic_classes(
        p.eris,
        roots,
        lambda a, b: pdms[a, b],
        np.eye(2),
        contraction_backend="numpy",
    )
    direct = replace(
        restricted,
        classes=classes,
        active_reference=eref,
        reference=eref,
        diagnostics=audit,
    )
    for key, block in classes.items():
        for name in ("metric", "right", "left", "active", "source"):
            np.testing.assert_allclose(
                getattr(restricted.classes[key], name),
                getattr(block, name),
                atol=2e-12,
            )
    for ansatz in ("ss_sr", "ms_mr"):
        a, b = (
            ms.solve_msfic(restricted, ansatz=ansatz),
            ms.solve_msfic(direct, ansatz=ansatz),
        )
        np.testing.assert_allclose(a.heff, b.heff, atol=3e-12)


def test_weak_sources_scale_quadratically_without_false_zero(data):
    p = data[0]
    factor = 1e-7
    weak = replace(
        p,
        classes={k: replace(b, source=b.source * factor) for k, b in p.classes.items()},
    )
    for ansatz in ("ss_sr", "ms_mr"):
        a, b = ms.solve_msfic(p, ansatz=ansatz), ms.solve_msfic(weak, ansatz=ansatz)
        np.testing.assert_allclose(
            b.source_response / factor**2, a.source_response, atol=1e-12
        )
        np.testing.assert_allclose(b.shift_norm / factor**2, a.shift_norm, atol=1e-12)


@pytest.mark.parametrize("ansatz", ["ss_sr", "ms_mr"])
def test_active_energy_zero_does_not_change_response(data, ansatz):
    p = data[0]
    constant = 50.0
    translated = replace(
        p,
        active_reference=p.active_reference + constant * np.eye(5),
        reference=p.reference + constant * np.eye(5),
        classes={
            key: replace(block, active=block.active + constant * block.metric)
            for key, block in p.classes.items()
        },
    )
    a = ms.solve_msfic(p, ansatz=ansatz)
    b = ms.solve_msfic(translated, ansatz=ansatz)
    np.testing.assert_allclose(
        b.heff - constant * np.eye(5), a.heff, atol=1e-12, rtol=0
    )
    np.testing.assert_allclose(b.source_response, a.source_response, atol=1e-12, rtol=0)
    np.testing.assert_allclose(b.shift_norm, a.shift_norm, atol=1e-12, rtol=0)


def test_indefinite_but_invertible_excited_state_block_is_not_clipped(data):
    p = data[0]
    block = p.classes["ijrs"]
    # A^L = (-100+orbital_gap-e_L) I is invertible and negative.
    negative = replace(block, active=-100 * np.eye(5))
    actual = ms.solve_msfic(replace(p, classes={"ijrs": negative}), shift=0)
    assert np.all(np.diag(actual.source_response).real > 0)
    assert actual.diagnostics["classes"]["ijrs"]["minimum_denominator"] < 0


@pytest.mark.parametrize("kind", ["phases", "permutation"])
def test_both_ansatzes_preserve_phase_permutation_covariance(data, kind):
    p = data[0]
    rotation = (
        np.diag(np.exp(1j * np.array([-0.7, 0.2, 0.9, -1.3, 0.4])))
        if kind == "phases"
        else np.eye(5)[:, [2, 4, 0, 1, 3]]
    )
    for ansatz in ("ss_sr", "ms_mr"):
        a = ms.solve_msfic(p, ansatz=ansatz)
        b = ms.solve_msfic(p.rotated(rotation), ansatz=ansatz)
        np.testing.assert_allclose(
            b.heff, rotation.conj().T @ a.heff @ rotation, atol=3e-12
        )


@pytest.mark.parametrize("ansatz", ["ss_sr", "ms_mr"])
def test_rotated_nondiagonal_reference_against_physical_sylvester_oracle(data, ansatz):
    """Independent physical IC spans, including cross-column Href terms."""
    p, refs, hrefs, ha, _ = data
    rng = np.random.default_rng(221)
    rotation = np.linalg.qr(rng.normal(size=(5, 5)) + 1j * rng.normal(size=(5, 5)))[0]
    changed = p.rotated(rotation)
    actual = ms.solve_msfic(changed, ansatz=ansatz, shift=0.2)
    hmodel = rotation.conj().T @ p.active_reference @ rotation
    assert np.linalg.norm(hmodel - np.diag(np.diag(hmodel))) > 1e-2
    rrefs, rhrefs = np.column_stack(refs) @ rotation, hrefs @ rotation
    y, norm = np.zeros((5, 5), complex), np.zeros((5, 5), complex)
    for key in p.classes:
        for indices in fic._iter_free_tuples(key, p.eris):
            columns = [
                _fic_basis_vectors(rrefs[:, root], p.eris, key, indices)
                for root in range(5)
            ]
            if ansatz == "ms_mr":
                columns = [np.concatenate(columns, axis=1)] * 5
            spans = []
            for column in columns:
                q, singular, _ = np.linalg.svd(column, full_matrices=False)
                spans.append(q[:, singular > 1e-8])
            offsets = np.cumsum([0] + [q.shape[1] for q in spans])
            delta = fic._orbital_gap_at(key, indices, p.mo_energy[:2], p.mo_energy[-2:])
            a = np.zeros((offsets[-1], offsets[-1]), complex)
            b = np.concatenate(
                [q.conj().T @ rhrefs[:, root] for root, q in enumerate(spans)]
            )
            for root, q in enumerate(spans):
                rows = slice(offsets[root], offsets[root + 1])
                hq = _apply_active_hamiltonian(q, ha, 2, 5, 2)
                a[rows, rows] += q.conj().T @ hq + (delta + 0.2) * np.eye(q.shape[1])
                for other, qo in enumerate(spans):
                    cols = slice(offsets[other], offsets[other + 1])
                    a[rows, cols] -= hmodel[other, root] * (q.conj().T @ qo)
            coefficients = -np.linalg.solve(a, b)
            response = np.column_stack(
                [
                    q @ coefficients[offsets[root] : offsets[root + 1]]
                    for root, q in enumerate(spans)
                ]
            )
            y += rhrefs.conj().T @ response
            norm += response.conj().T @ response
    expected = changed.reference + 0.5 * (y + y.conj().T) - 0.2 * norm
    np.testing.assert_allclose(actual.heff, expected, atol=3e-12, rtol=0)
    np.testing.assert_allclose(actual.shift_norm, norm, atol=3e-12, rtol=0)


def test_missing_rank4_and_invalid_density_metadata_are_rejected(data):
    p, _, _, _, pdms = data
    options = {"contraction_backend": "numpy"}
    with pytest.raises(ValueError, match="full transition ranks"):
        ms.build_msfic_classes(
            p.eris, (0,), lambda a, b: pdms[a, b][:3], np.eye(1), **options
        )
    for dm4, error, message in (
        (pdms[0, 0][2], ValueError, "dm4 has wrong shape"),
        (np.empty(pdms[0, 0][3].shape, dtype=object), TypeError, "dm4 must be numeric"),
    ):
        broken = (*pdms[0, 0][:3], dm4)
        with pytest.raises(error, match=message):
            ms.build_msfic_classes(
                p.eris,
                (0,),
                lambda a, b, densities=broken: densities,
                np.eye(1),
                **options,
            )


@pytest.mark.parametrize("keep", [False, True])
def test_production_streams_disk_rdms_without_audits(data, monkeypatch, tmp_path, keep):
    p, _, _, _, pdms = data
    monkeypatch.setattr(ms.lib.param, "TMPDIR", str(tmp_path))
    monkeypatch.setattr(
        ms.u,
        "validate_transition_pdms",
        lambda *args, **kwargs: pytest.fail("production must not audit raw RDMs"),
    )
    generated, maps, calls = [], [], []
    contract = ms._contract_pair

    def provider(bra, ket):
        assert all(ref() is None for ref in generated)
        assert all(m.closed for m in maps)
        arrays = tuple(dm.copy() for dm in pdms[bra, ket])
        generated[:] = [weakref.ref(dm) for dm in arrays]
        calls.append((bra, ket))
        return arrays

    def mapped_contract(eris, densities, *args):
        assert all(ref() is None for ref in generated)
        assert all(
            isinstance(dm, np.memmap) and not dm.flags.writeable for dm in densities
        )
        maps.extend(dm._mmap for dm in densities)
        return contract(eris, densities, *args)

    monkeypatch.setattr(ms, "_contract_pair", mapped_contract)
    classes, eref, audit = ms.build_msfic_classes(
        p.eris,
        (0, 1),
        provider,
        np.eye(2),
        contraction_backend="pytblis",
        transition_rdm_dir=tmp_path / "retained" if keep else None,
    )
    assert audit["transition_rdm_audited"] is False
    assert calls == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert all(ref() is None for ref in generated)
    assert all(m.closed for m in maps)
    if keep:
        directory = Path(audit["transition_rdm_directory"])
        for bra, ket in calls:
            pair = directory / f"bra-{bra}_ket-{ket}"
            ready = json.loads((pair / "READY.json").read_text())
            assert (ready["bra"], ready["ket"], ready["ncas"]) == (
                bra,
                ket,
                p.eris.ncas,
            )
            for rank in range(1, 5):
                dm = np.load(pair / f"dm{rank}.npy", mmap_mode="r", allow_pickle=False)
                try:
                    np.testing.assert_array_equal(dm, pdms[bra, ket][rank - 1])
                finally:
                    dm._mmap.close()
        monkeypatch.setattr(ms, "_contract_pair", contract)
        monkeypatch.setattr(
            ms.np,
            "save",
            lambda *a, **kw: pytest.fail("do not rewrite existing RDM maps"),
        )
        loaded, loaded_ref, _ = ms.build_msfic_classes(
            p.eris,
            (0, 1),
            lambda a, b: tuple(
                np.load(
                    directory / f"bra-{a}_ket-{b}" / f"dm{rank}.npy",
                    mmap_mode="r",
                    allow_pickle=False,
                )
                for rank in range(1, 5)
            ),
            np.eye(2),
            contraction_backend="pytblis",
        )
        np.testing.assert_array_equal(loaded_ref, eref)
        for key, block in loaded.items():
            for name in ("metric", "right", "left", "active", "source"):
                np.testing.assert_array_equal(
                    getattr(block, name), getattr(classes[key], name)
                )
    else:
        assert audit["transition_rdm_directory"] is None
        assert not list(tmp_path.iterdir())
    np.testing.assert_allclose(eref, p.active_reference[:2, :2], atol=1e-12)
    for key, block in classes.items():
        d = block.dimension_per_reference
        for name in ("metric", "right", "left", "active"):
            np.testing.assert_allclose(
                getattr(block, name),
                getattr(p.classes[key], name)[: 2 * d, : 2 * d],
                atol=2e-12,
            )
        np.testing.assert_allclose(
            block.source,
            p.classes[key].source[..., : 2 * d, :2],
            atol=2e-12,
        )


@pytest.mark.parametrize("keep", [False, True])
@pytest.mark.parametrize("mode", ["space", "enospc", "edquot", "short_write", "mkdir"])
def test_disk_capacity_falls_back_without_recomputing_rdms(
    data, monkeypatch, tmp_path, keep, mode
):
    p, _, _, _, pdms = data
    preferred, fallback = tmp_path / "nvme", tmp_path / "home"
    monkeypatch.setattr(ms.lib.param, "TMPDIR", str(preferred))
    calls, faulted = [], False
    real_save, real_usage = np.save, ms.shutil.disk_usage
    real_temporary, real_mkdtemp = ms.TemporaryDirectory, ms.mkdtemp

    def usage(path):
        if mode == "space" and Path(path).is_relative_to(preferred):
            return SimpleNamespace(free=0)
        return real_usage(path)

    def save(handle, array, **options):
        nonlocal faulted
        path = Path(handle.name)
        if (
            not faulted
            and mode in ("enospc", "edquot", "short_write")
            and path.is_relative_to(preferred)
            and path.name == "dm4.npy.tmp"
        ):
            faulted = True
            if mode == "short_write":
                raise OSError("100 requested and 12 written")
            raise OSError(errno.ENOSPC if mode == "enospc" else errno.EDQUOT, mode)
        return real_save(handle, array, **options)

    def allocate(original, *args, **kwargs):
        if mode == "mkdir" and Path(kwargs["dir"]) == preferred:
            raise OSError(errno.ENOSPC, "cannot create NVMe temporary directory")
        return original(*args, **kwargs)

    def provider(bra, ket):
        calls.append((bra, ket))
        return tuple(dm.copy() for dm in pdms[bra, ket])

    monkeypatch.setattr(ms.shutil, "disk_usage", usage)
    monkeypatch.setattr(ms.np, "save", save)
    monkeypatch.setattr(
        ms, "TemporaryDirectory", lambda *a, **kw: allocate(real_temporary, *a, **kw)
    )
    monkeypatch.setattr(
        ms, "mkdtemp", lambda *a, **kw: allocate(real_mkdtemp, *a, **kw)
    )
    classes, eref, audit = ms.build_msfic_classes(
        p.eris,
        (0, 1),
        provider,
        np.eye(2),
        contraction_backend="pytblis",
        transition_rdm_dir=preferred if keep else None,
        transition_rdm_fallback_dir=fallback,
    )
    assert calls == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert not list(preferred.rglob("*.tmp"))
    if keep:
        assert set(audit["transition_rdm_locations"]) == {f"{a},{b}" for a, b in calls}
        assert Path(audit["transition_rdm_locations"]["0,0"]).is_relative_to(fallback)
        for bra, ket in calls:
            path = Path(audit["transition_rdm_locations"][f"{bra},{ket}"])
            assert (path / "READY.json").is_file()
            for rank in range(1, 5):
                dm = np.load(path / f"dm{rank}.npy", mmap_mode="r", allow_pickle=False)
                try:
                    np.testing.assert_array_equal(dm, pdms[bra, ket][rank - 1])
                finally:
                    dm._mmap.close()
    else:
        assert not list(preferred.iterdir()) and not list(fallback.iterdir())
    np.testing.assert_allclose(eref, p.active_reference[:2, :2], atol=1e-12)
    for key, block in classes.items():
        d = block.dimension_per_reference
        for name in ("metric", "right", "left", "active"):
            np.testing.assert_allclose(
                getattr(block, name),
                getattr(p.classes[key], name)[: 2 * d, : 2 * d],
                atol=2e-12,
            )
        np.testing.assert_allclose(
            block.source, p.classes[key].source[..., : 2 * d, :2], atol=2e-12
        )


@pytest.mark.parametrize("error_code", [errno.EACCES, errno.EIO])
def test_noncapacity_disk_errors_are_not_hidden(
    data, monkeypatch, tmp_path, error_code
):
    p, _, _, _, pdms = data
    preferred, fallback = tmp_path / "nvme", tmp_path / "home"
    monkeypatch.setattr(ms.lib.param, "TMPDIR", str(preferred))
    calls = []

    def provider(bra, ket):
        calls.append((bra, ket))
        return pdms[bra, ket]

    def save(*args, **kwargs):
        raise OSError(error_code, "injected noncapacity error")

    monkeypatch.setattr(ms.np, "save", save)
    with pytest.raises(OSError) as raised:
        ms.build_msfic_classes(
            p.eris,
            (0,),
            provider,
            np.eye(1),
            contraction_backend="numpy",
            transition_rdm_fallback_dir=fallback,
        )
    assert raised.value.errno == error_code and calls == [(0, 0)]
    assert not list(preferred.iterdir()) and not list(fallback.iterdir())


@pytest.mark.parametrize("ansatz", ["ss_sr", "ms_mr"])
def test_singular_denominators_are_rejected_not_regularized(data, ansatz):
    p = data[0]
    block = p.classes["ijrs"]
    indices = next(fic._iter_free_tuples("ijrs", p.eris))
    delta = fic._orbital_gap_at("ijrs", indices, p.mo_energy[:2], p.mo_energy[-2:])
    singular = replace(block, active=p.active_reference - delta * block.metric)
    with pytest.raises(FloatingPointError, match="near-singular"):
        ms.solve_msfic(replace(p, classes={"ijrs": singular}), ansatz=ansatz, shift=0)
