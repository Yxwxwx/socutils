# SPDX-License-Identifier: GPL-3.0-or-later
"""Direct Fock-space checks of the eight relativistic SC-NEVPT2 classes."""

import itertools
import sys
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
from pyscf import ao2mo, gto, mcscf, scf
from pyscf.mrpt import nevpt2 as pyscf_nevpt2

from socutils.mcscf import zmcscf
from socutils.mrpt import spinor_helper, x2cscnevpt2


def test_adapter_input_normalizers_cover_multiroot_and_spin_resolved_cases():
    handles = np.array([object(), object(), object()], dtype=object)
    assert x2cscnevpt2._select_root_ci(range(3), 2, nroots=3) == 2
    assert x2cscnevpt2._select_root_ci(handles, 1, nroots=3) is handles[1]

    integer_handles = np.array([11, 12, 13])
    assert x2cscnevpt2._select_root_ci(
        integer_handles, 1, nroots=3
    ) == 12

    single_root_vector = np.arange(6.0)
    assert (
        x2cscnevpt2._select_root_ci(single_root_vector, 0, nroots=1)
        is single_root_vector
    )
    assert x2cscnevpt2._total_nelec(4) == 4
    assert x2cscnevpt2._total_nelec((3, 2)) == 5


def test_npdm_fallback_uses_block2_default_zero_site_algorithm():
    captured = {}

    class Driver:
        def get_1pdm(self, ket, **kwargs):
            captured.update(ket=ket, **kwargs)
            return np.eye(2, dtype=complex)

    ket = object()
    solver = SimpleNamespace(driver=Driver(), kets=(ket,), ncas=2)
    density = x2cscnevpt2.make_rdm1(solver)
    np.testing.assert_array_equal(density, np.eye(2))
    assert captured["ket"] is ket
    assert captured["site_type"] == 0
    assert captured["cutoff"] == pytest.approx(1.0e-24)


def test_nonzero_mc_frozen_is_rejected_at_construction_and_kernel():
    molecule = SimpleNamespace(verbose=0, stdout=sys.stdout)
    mean_field = SimpleNamespace(mol=molecule)
    mc = SimpleNamespace(_scf=mean_field, frozen=[0])
    with pytest.raises(NotImplementedError, match="frozen"):
        x2cscnevpt2.WickX2CSCNEVPT2(mc)

    mc.frozen = 0
    adapter = x2cscnevpt2.WickX2CSCNEVPT2(mc)
    mc.frozen = np.array([0])
    with pytest.raises(NotImplementedError, match="frozen"):
        adapter.kernel(mc=mc)


def test_explicit_partner_channels_are_grouped_before_sc_denominator():
    eris = SimpleNamespace(nmo=4, ncore=0, nocc=0)
    labels = x2cscnevpt2._normalize_strong_contraction_groups(
        [0, 0, 1, 1],
        eris,
        np.zeros(0),
        np.array([1.0, 1.0, 2.0, 2.0]),
        atol=1.0e-12,
        rtol=0.0,
    )
    ordered = x2cscnevpt2._strict_pair_mask("rs", (4, 4))
    norm = np.zeros((4, 4), dtype=complex)
    commutator = np.zeros_like(norm)
    norm[ordered] = np.arange(1.0, 7.0)
    commutator[ordered] = np.arange(1.0, 7.0) ** 2
    orbital_gap = np.add.outer(
        np.array([1.0, 1.0, 2.0, 2.0]),
        np.array([1.0, 1.0, 2.0, 2.0]),
    )

    grouped, grouped_gap, diagnostics = (
        x2cscnevpt2._group_strong_contraction_arrays(
            "rs",
            (norm, commutator),
            orbital_gap,
            ordered,
            labels,
            eris,
            atol=1.0e-12,
            rtol=0.0,
        )
    )
    grouped_norm, grouped_commutator = grouped
    np.testing.assert_array_equal(grouped_norm, [1.0, 14.0, 6.0])
    np.testing.assert_array_equal(grouped_commutator, [1.0, 54.0, 36.0])
    np.testing.assert_array_equal(grouped_gap, [2.0, 3.0, 4.0])
    assert diagnostics["raw_ordered_dimension"] == 6
    assert diagnostics["contracted_group_dimension"] == 3
    assert diagnostics["maximum_orbital_gap_spread"] == 0.0

    raw_norm = norm[ordered].real
    raw_commutator = commutator[ordered].real
    raw_gap = orbital_gap[ordered]
    ungrouped_energy = -np.sum(
        raw_norm / (raw_gap + raw_commutator / raw_norm)
    )
    grouped_energy = -np.sum(
        grouped_norm.real
        / (
            grouped_gap
            + grouped_commutator.real / grouped_norm.real
        )
    )
    assert abs(grouped_energy - ungrouped_energy) > 1.0e-2


def test_strong_contraction_group_rejects_nondegenerate_members():
    eris = SimpleNamespace(nmo=4, ncore=0, nocc=0)
    with pytest.raises(ValueError, match="nondegenerate semicanonical energies"):
        x2cscnevpt2._normalize_strong_contraction_groups(
            [0, 0, 1, 1],
            eris,
            np.zeros(0),
            np.array([1.0, 1.1, 2.0, 2.0]),
            atol=1.0e-12,
            rtol=0.0,
        )


def _apply_operator(vector, kind, orbital):
    result = np.zeros_like(vector)
    lower_mask = (1 << orbital) - 1
    orbital_mask = 1 << orbital
    for state, amplitude in enumerate(vector):
        if amplitude == 0:
            continue
        occupied = bool(state & orbital_mask)
        if kind == "D":
            if not occupied:
                continue
        elif kind == "C":
            if occupied:
                continue
        else:  # pragma: no cover - internal test helper misuse
            raise ValueError(kind)
        sign = -1 if (state & lower_mask).bit_count() % 2 else 1
        result[state ^ orbital_mask] += sign * amplitude
    return result


def _apply_string(vector, operators):
    result = vector
    for kind, orbital in reversed(operators):
        result = _apply_operator(result, kind, orbital)
    return result


def _hamiltonian_action(vector, h1e, w):
    result = np.zeros_like(vector)
    nmo = h1e.shape[0]
    for p, q in itertools.product(range(nmo), repeat=2):
        coefficient = h1e[p, q]
        if coefficient:
            result += coefficient * _apply_string(
                vector, (("C", p), ("D", q))
            )
    for p, q, r, s in itertools.product(range(nmo), repeat=4):
        coefficient = 0.5 * w[p, q, r, s]
        if coefficient:
            result += coefficient * _apply_string(
                vector,
                (("C", p), ("C", q), ("D", s), ("D", r)),
            )
    return result


def _fixed_particle_basis(norb, nelec):
    return [state for state in range(1 << norb) if state.bit_count() == nelec]


def _active_eigenstates(h1e, w, nelec):
    norb = h1e.shape[0]
    basis = _fixed_particle_basis(norb, nelec)
    hamiltonian = np.empty((len(basis), len(basis)), dtype=complex)
    for column, state in enumerate(basis):
        ket = np.zeros(1 << norb, dtype=complex)
        ket[state] = 1.0
        hket = _hamiltonian_action(ket, h1e, w)
        hamiltonian[:, column] = hket[basis]
    np.testing.assert_allclose(hamiltonian, hamiltonian.T.conj(), atol=2e-12)
    energies, vectors = np.linalg.eigh(hamiltonian)
    states = []
    for root in range(vectors.shape[1]):
        state = np.zeros(1 << norb, dtype=complex)
        state[basis] = vectors[:, root]
        states.append(state)
    return energies, tuple(states)


def _active_ground_state(h1e, w, nelec):
    energies, states = _active_eigenstates(h1e, w, nelec)
    return energies[0], states[0]


def _embed_active_state(active_state, ncore, ncas, nvirt):
    result = np.zeros(1 << (ncore + ncas + nvirt), dtype=complex)
    core_bits = (1 << ncore) - 1
    for active_bits, amplitude in enumerate(active_state):
        result[core_bits | (active_bits << ncore)] = amplitude
    return result


def _raw_active_rdms(active_state, ncas):
    """Build exact raw RDMs by Gram matrices of annihilated Fock vectors."""

    pdms = []
    for rank in range(1, 5):
        tuples = tuple(itertools.product(range(ncas), repeat=rank))
        annihilated = np.asarray(
            [
                _apply_string(
                    active_state,
                    tuple(("D", orbital) for orbital in indices),
                )
                for indices in tuples
            ]
        )
        rows = np.arange(ncas**rank).reshape((ncas,) * rank)
        reversed_rows = rows.transpose(tuple(reversed(range(rank)))).reshape(-1)
        # <C[p1]...C[pk]D[q1]...D[qk]> is the overlap between
        # D[pk]...D[p1]|Psi> and D[q1]...D[qk]|Psi>.
        density = np.einsum(
            "xi,yi->xy",
            annihilated[reversed_rows].conj(),
            annihilated,
            optimize=True,
        )
        pdms.append(density.reshape((ncas,) * (2 * rank)))
    return tuple(pdms)


def _random_physical_integrals(nmo, seed=8123):
    rng = np.random.default_rng(seed)
    trial = rng.normal(size=(nmo, nmo)) + 1j * rng.normal(size=(nmo, nmo))
    h1e = 0.04 * (trial + trial.T.conj())

    factors = rng.normal(size=(5, nmo, nmo))
    factors = factors + 1j * rng.normal(size=factors.shape)
    factors = 0.025 * (factors + factors.swapaxes(1, 2).conj())
    eri = np.einsum("Ppq,Prs->pqrs", factors, factors, optimize=True)
    spinor_helper.check_eri_symmetry(eri, atol=2e-13, rtol=2e-13)
    return h1e, eri


def _random_unitary(rng, size):
    trial = rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
    unitary, diagonal = np.linalg.qr(trial)
    phases = np.diag(diagonal)
    phases = np.where(abs(phases) == 0.0, 1.0, phases / abs(phases))
    return unitary * phases.conj()


def test_dense_mo_integral_projection_is_the_four_element_group_average():
    """Recover both physical s1 identities from production-scale noise."""

    h1e, eri = _random_physical_integrals(7, seed=1812)
    h1e *= 5.0e6
    eri *= 1.0e5
    raw_h1e = h1e.copy()
    raw_eri = eri.copy()
    raw_h1e[0, 1] += 7.0e-11 + 3.0e-11j
    raw_eri[0, 1, 2, 3] += 1.1e-10 - 0.4e-10j
    original_h1e = raw_h1e.copy()
    original_eri = raw_eri.copy()

    expected_h1e = 0.5 * (original_h1e + original_h1e.T.conj())
    pair = original_eri.transpose(2, 3, 0, 1)
    conjugate = original_eri.transpose(1, 0, 3, 2).conj()
    pair_conjugate = pair.transpose(1, 0, 3, 2).conj()
    expected_eri = 0.25 * (
        original_eri + pair + conjugate + pair_conjugate
    )

    projected_h1e, projected_eri, diagnostics = (
        x2cscnevpt2._project_dense_mo_integrals(raw_h1e, raw_eri)
    )
    np.testing.assert_allclose(
        projected_h1e, expected_h1e, atol=2.0e-11, rtol=0.0
    )
    np.testing.assert_allclose(
        projected_eri, expected_eri, atol=3.0e-13, rtol=0.0
    )
    np.testing.assert_array_equal(projected_h1e, projected_h1e.T.conj())
    np.testing.assert_array_equal(
        projected_eri, projected_eri.transpose(2, 3, 0, 1)
    )
    np.testing.assert_array_equal(
        projected_eri,
        projected_eri.transpose(1, 0, 3, 2).conj(),
    )

    assert 0.0 < diagnostics["h1e"]["raw_hermiticity_error"]
    assert (
        diagnostics["h1e"]["raw_hermiticity_error"]
        <= diagnostics["h1e"]["roundoff_gate"]
    )
    assert 0.0 < max(
        diagnostics["eri"]["raw_pair_exchange_error"],
        diagnostics["eri"]["raw_conjugate_exchange_error"],
    )
    assert (
        diagnostics["eri"]["raw_pair_exchange_error"]
        <= diagnostics["eri"]["roundoff_gate"]
    )
    assert (
        diagnostics["eri"]["raw_conjugate_exchange_error"]
        <= diagnostics["eri"]["roundoff_gate"]
    )
    assert diagnostics["h1e"]["post_projection_hermiticity_error"] == 0.0
    assert diagnostics["eri"]["post_projection_pair_exchange_error"] == 0.0
    assert (
        diagnostics["eri"]["post_projection_conjugate_exchange_error"]
        == 0.0
    )
    assert np.max(abs(projected_h1e - original_h1e)) <= diagnostics["h1e"][
        "projection_change_upper_bound"
    ]
    assert np.max(abs(projected_eri - original_eri)) <= diagnostics["eri"][
        "projection_change_upper_bound"
    ]

    twice_h1e, twice_eri, twice_diagnostics = (
        x2cscnevpt2._project_dense_mo_integrals(
            projected_h1e.copy(), projected_eri.copy()
        )
    )
    np.testing.assert_array_equal(twice_h1e, projected_h1e)
    np.testing.assert_array_equal(twice_eri, projected_eri)
    assert twice_diagnostics["h1e"]["raw_hermiticity_error"] == 0.0
    assert twice_diagnostics["eri"]["raw_pair_exchange_error"] == 0.0
    assert twice_diagnostics["eri"]["raw_conjugate_exchange_error"] == 0.0


@pytest.mark.parametrize(
    ("nmo", "dtype", "scale", "ao_dimension"),
    (
        (2, np.float32, 1.0e-3, 3),
        (3, np.complex64, 1.0e-2, 5),
        (5, np.float64, 1.0, 7),
        (9, np.complex128, 1.0e6, 13),
    ),
)
def test_dense_mo_integral_roundoff_policy_tracks_dtype_scale_and_dimension(
    nmo,
    dtype,
    scale,
    ao_dimension,
):
    h1e, eri = _random_physical_integrals(nmo, seed=1900 + nmo)
    if np.issubdtype(dtype, np.complexfloating):
        h1e = np.asarray(scale * h1e, dtype=dtype)
        eri = np.asarray(scale * eri, dtype=dtype)
    else:
        h1e = np.asarray(scale * h1e.real, dtype=dtype)
        eri = np.asarray(scale * eri.real, dtype=dtype)
    roundoff_factor = 1.5

    _projected_h1e, _projected_eri, diagnostics = (
        x2cscnevpt2._project_dense_mo_integrals(
            h1e.copy(),
            eri.copy(),
            roundoff_accumulation_length=ao_dimension,
            roundoff_factor=roundoff_factor,
        )
    )
    expected_real_dtype = np.asarray(h1e.real).dtype.name
    epsilon = float(np.finfo(np.asarray(h1e.real).dtype).eps)
    is_complex = bool(np.issubdtype(dtype, np.complexfloating))
    operations_per_multiply_add = 8 if is_complex else 2
    for name, stages in (("h1e", 2), ("eri", 4)):
        policy = diagnostics[name]["roundoff_policy"]
        operation_count = (
            operations_per_multiply_add * stages * ao_dimension
        )
        gamma = operation_count * epsilon / (1.0 - operation_count * epsilon)
        assert policy["input_dtype"] == np.dtype(dtype).name
        assert policy["is_complex"] is is_complex
        assert policy["real_dtype"] == expected_real_dtype
        assert policy["accumulation_length"] == ao_dimension
        assert policy["contraction_stages"] == stages
        assert (
            policy["real_operations_per_multiply_add"]
            == operations_per_multiply_add
        )
        assert policy["operation_count"] == operation_count
        assert policy["gamma"] == pytest.approx(gamma)
        assert policy["symmetry_comparison_paths"] == 2
        assert "independently rounded" in policy[
            "symmetry_comparison_path_provenance"
        ]
        assert policy["roundoff_factor"] == roundoff_factor
        assert policy["maximum_absolute_value"] == pytest.approx(
            diagnostics[name]["maximum_absolute_value"]
        )
        assert policy["effective_scale"] == max(
            1.0, policy["maximum_absolute_value"]
        )
        assert diagnostics[name]["roundoff_gate"] == pytest.approx(
            roundoff_factor
            * policy["symmetry_comparison_paths"]
            * gamma
            * policy["effective_scale"]
        )

    noisy_h1e = h1e.copy()
    noisy_eri = eri.copy()
    if is_complex:
        noisy_h1e[0, 1] += 0.2 * diagnostics["h1e"][
            "roundoff_gate"
        ] * (1.0 + 0.25j)
        noisy_eri[0, 1, 0, 1] += 0.2 * diagnostics["eri"][
            "roundoff_gate"
        ] * (1.0 - 0.25j)
    else:
        noisy_h1e[0, 1] += 0.2 * diagnostics["h1e"]["roundoff_gate"]
        noisy_eri[0, 1, 0, 1] += 0.2 * diagnostics["eri"][
            "roundoff_gate"
        ]
    projected_h1e, projected_eri, _diagnostics = (
        x2cscnevpt2._project_dense_mo_integrals(
            noisy_h1e,
            noisy_eri,
            roundoff_accumulation_length=ao_dimension,
            roundoff_factor=roundoff_factor,
        )
    )
    np.testing.assert_array_equal(projected_h1e, projected_h1e.T.conj())
    np.testing.assert_array_equal(
        projected_eri, projected_eri.transpose(2, 3, 0, 1)
    )
    np.testing.assert_array_equal(
        projected_eri,
        projected_eri.transpose(1, 0, 3, 2).conj(),
    )


def test_dense_mo_integral_roundoff_gate_covers_two_opposite_paths():
    nmo = 4
    h1e = np.zeros((nmo, nmo), dtype=np.complex128)
    eri = np.zeros((nmo,) * 4, dtype=np.complex128)
    _h1e, _eri, baseline = x2cscnevpt2._project_dense_mo_integrals(
        h1e.copy(), eri.copy()
    )

    h1e_single_path = 0.5 * baseline["h1e"]["roundoff_gate"]
    eri_single_path = 0.5 * baseline["eri"]["roundoff_gate"]
    h1e[0, 1] += 0.75 * h1e_single_path
    h1e[1, 0] -= 0.75 * h1e_single_path
    eri[0, 1, 2, 3] += 0.75 * eri_single_path
    eri[2, 3, 0, 1] -= 0.75 * eri_single_path

    projected_h1e, projected_eri, diagnostics = (
        x2cscnevpt2._project_dense_mo_integrals(h1e, eri)
    )
    assert (
        diagnostics["h1e"]["raw_hermiticity_error"] > h1e_single_path
    )
    assert (
        diagnostics["h1e"]["raw_hermiticity_error"]
        <= diagnostics["h1e"]["roundoff_gate"]
    )
    assert (
        diagnostics["eri"]["raw_pair_exchange_error"] > eri_single_path
    )
    assert (
        diagnostics["eri"]["raw_pair_exchange_error"]
        <= diagnostics["eri"]["roundoff_gate"]
    )
    np.testing.assert_array_equal(projected_h1e, projected_h1e.T.conj())
    np.testing.assert_array_equal(
        projected_eri, projected_eri.transpose(2, 3, 0, 1)
    )
    np.testing.assert_array_equal(
        projected_eri,
        projected_eri.transpose(1, 0, 3, 2).conj(),
    )


@pytest.mark.parametrize(
    "dtype", (np.float32, np.float64, np.complex64, np.complex128)
)
def test_dense_mo_integral_projection_is_stable_near_dtype_maximum(dtype):
    real_dtype = np.empty((), dtype=dtype).real.dtype
    magnitude = np.asarray(0.75 * np.finfo(real_dtype).max, dtype=real_dtype)
    value = np.asarray(magnitude, dtype=dtype)
    h1e = np.full((2, 2), value, dtype=dtype)
    eri = np.full((2,) * 4, value, dtype=dtype)
    original_h1e = h1e.copy()
    original_eri = eri.copy()

    with np.errstate(over="raise", invalid="raise"):
        projected_h1e, projected_eri, diagnostics = (
            x2cscnevpt2._project_dense_mo_integrals(h1e, eri)
        )

    assert np.all(np.isfinite(projected_h1e))
    assert np.all(np.isfinite(projected_eri))
    np.testing.assert_array_equal(projected_h1e, projected_h1e.T.conj())
    np.testing.assert_array_equal(
        projected_eri, projected_eri.transpose(2, 3, 0, 1)
    )
    np.testing.assert_array_equal(
        projected_eri,
        projected_eri.transpose(1, 0, 3, 2).conj(),
    )
    assert np.max(np.abs(projected_h1e - original_h1e)) <= diagnostics[
        "h1e"
    ]["projection_change_upper_bound"]
    assert np.max(np.abs(projected_eri - original_eri)) <= diagnostics[
        "eri"
    ]["projection_change_upper_bound"]


@pytest.mark.parametrize(
    ("nmo", "scale"),
    ((2, 1.0e-6), (6, 1.0), (10, 1.0e6)),
)
def test_dense_mo_integral_roundoff_policy_warns_at_all_scales(
    nmo,
    scale,
):
    h1e, eri = _random_physical_integrals(nmo, seed=2100 + nmo)
    h1e *= scale
    eri *= scale

    broken_h1e = h1e.copy()
    broken_h1e[0, 1] += 1.0e-3j * max(1.0, np.max(np.abs(h1e)))
    with pytest.warns(
        x2cscnevpt2.MRPTNumericalWarning,
        match="h1e violates Hermiticity",
    ):
        _h1e, _eri, diagnostics = x2cscnevpt2._project_dense_mo_integrals(
            broken_h1e, eri.copy()
        )
    assert not diagnostics["h1e"]["roundoff_gate_passed"]

    broken_eri = eri.copy()
    broken_eri[0, 1, 0, 1] += 1.0e-3 * max(1.0, np.max(np.abs(eri)))
    with pytest.warns(
        x2cscnevpt2.MRPTNumericalWarning,
        match="eri violates complex Coulomb",
    ):
        _h1e, _eri, diagnostics = x2cscnevpt2._project_dense_mo_integrals(
            h1e.copy(), broken_eri
        )
    assert not diagnostics["eri"]["roundoff_gate_passed"]


def test_dense_mo_integral_roundoff_policy_validates_configuration():
    h1e, eri = _random_physical_integrals(2, seed=2217)
    for invalid_factor in (0.0, -1.0, np.inf, np.nan, 10**10000):
        with pytest.raises(ValueError, match="roundoff_factor"):
            x2cscnevpt2._project_dense_mo_integrals(
                h1e.copy(), eri.copy(), roundoff_factor=invalid_factor
            )
    for invalid_length in (0, -1):
        with pytest.raises(ValueError, match="roundoff_accumulation_length"):
            x2cscnevpt2._project_dense_mo_integrals(
                h1e.copy(),
                eri.copy(),
                roundoff_accumulation_length=invalid_length,
            )
    for invalid_length in (True, 2.5):
        with pytest.raises(TypeError, match="roundoff_accumulation_length"):
            x2cscnevpt2._project_dense_mo_integrals(
                h1e.copy(),
                eri.copy(),
                roundoff_accumulation_length=invalid_length,
            )
    for invalid_stages in (0, -1):
        with pytest.raises(ValueError, match="contraction_stages"):
            x2cscnevpt2._ao2mo_roundoff_policy(
                h1e,
                accumulation_length=2,
                contraction_stages=invalid_stages,
                roundoff_factor=1.0,
            )
    for invalid_stages in (True, 2.5):
        with pytest.raises(TypeError, match="contraction_stages"):
            x2cscnevpt2._ao2mo_roundoff_policy(
                h1e,
                accumulation_length=2,
                contraction_stages=invalid_stages,
                roundoff_factor=1.0,
            )
    with pytest.raises(ValueError, match=r"operation_count \* eps"):
        x2cscnevpt2._ao2mo_roundoff_policy(
            h1e,
            accumulation_length=2,
            contraction_stages=10**400,
            roundoff_factor=1.0,
        )


@pytest.mark.parametrize(
    "dtype", (np.float16, np.longdouble, np.clongdouble)
)
def test_dense_mo_integral_projection_rejects_unsupported_dtype(dtype):
    if np.dtype(dtype) in x2cscnevpt2._SUPPORTED_AO2MO_DTYPES:
        pytest.skip("platform aliases this extended dtype to a supported dtype")
    h1e = np.zeros((2, 2), dtype=dtype)
    eri = np.zeros((2,) * 4, dtype=dtype)
    with pytest.raises(TypeError, match="unsupported dtype"):
        x2cscnevpt2._project_dense_mo_integrals(h1e, eri)


def test_dense_mo_integral_roundoff_policy_rejects_nonfinite_scale_or_gate():
    largest = np.finfo(np.float64).max
    excessive_complex_magnitude = np.array(
        [0.9 * largest + 0.9j * largest], dtype=np.complex128
    )
    with pytest.raises(ValueError, match="maximum absolute value"):
        x2cscnevpt2._ao2mo_roundoff_policy(
            excessive_complex_magnitude,
            accumulation_length=1,
            contraction_stages=1,
            roundoff_factor=1.0,
        )

    large_finite_values = np.array([0.75 * largest], dtype=np.float64)
    with pytest.raises(ValueError, match="roundoff gate is non-finite"):
        x2cscnevpt2._ao2mo_roundoff_policy(
            large_finite_values,
            accumulation_length=1,
            contraction_stages=1,
            roundoff_factor=largest,
        )


def test_dense_mo_integral_projection_warns_finite_and_rejects_nonfinite():
    h1e, eri = _random_physical_integrals(6, seed=927)

    broken_h1e = h1e.copy()
    broken_h1e[0, 1] += 1.0e-3j
    with pytest.warns(
        x2cscnevpt2.MRPTNumericalWarning,
        match="h1e violates Hermiticity",
    ):
        x2cscnevpt2._project_dense_mo_integrals(broken_h1e, eri.copy())

    broken_eri = eri.copy()
    broken_eri[0, 1, 2, 3] += 1.0e-3
    with pytest.warns(
        x2cscnevpt2.MRPTNumericalWarning,
        match="eri violates complex Coulomb",
    ):
        x2cscnevpt2._project_dense_mo_integrals(h1e.copy(), broken_eri)

    nonfinite_h1e = h1e.copy()
    nonfinite_h1e[0, 0] = np.inf
    with pytest.raises(ValueError, match="h1e contains non-finite"):
        x2cscnevpt2._project_dense_mo_integrals(nonfinite_h1e, eri.copy())

    nonfinite_eri = eri.copy()
    nonfinite_eri[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="eri contains non-finite"):
        x2cscnevpt2._project_dense_mo_integrals(h1e.copy(), nonfinite_eri)


def test_dense_eris_from_mc_projects_roundoff_and_records_diagnostics(
    monkeypatch,
):
    from pyscf.ao2mo import nrr_outcore

    nmo, ncore, ncas = 7, 2, 3
    h1e, eri = _random_physical_integrals(nmo, seed=713)
    hcore = 5.0e6 * h1e
    raw_eri = 1.0e5 * eri
    hcore[0, 1] += 7.0e-11j
    raw_eri[0, 1, 2, 3] += 1.0e-10 + 0.3e-10j
    monkeypatch.setattr(
        nrr_outcore,
        "full_iofree",
        lambda *args, **kwargs: raw_eri.copy(),
    )
    mc = SimpleNamespace(
        mol=object(),
        verbose=0,
        ncore=ncore,
        ncas=ncas,
        get_hcore=lambda: hcore,
    )

    result = x2cscnevpt2._dense_eris_from_mc(
        mc, np.eye(nmo, dtype=complex)
    )
    np.testing.assert_array_equal(result.h1e, result.h1e.T.conj())
    np.testing.assert_allclose(
        result.h1eff, result.h1eff.T.conj(), atol=1e-10, rtol=0.0
    )
    np.testing.assert_array_equal(
        result.pppp, result.pppp.transpose(2, 3, 0, 1)
    )
    np.testing.assert_array_equal(
        result.pppp, result.pppp.transpose(1, 0, 3, 2).conj()
    )
    assert result.symmetry_diagnostics["h1e"][
        "raw_hermiticity_error"
    ] > 0.0
    assert result.symmetry_diagnostics["eri"][
        "raw_pair_exchange_error"
    ] > 0.0
    assert result.symmetry_diagnostics["eri"]["roundoff_policy"][
        "accumulation_length"
    ] == nmo


def _spatial_to_spinor_integrals(h1e, eri):
    nmo = h1e.shape[0]
    h_spinor = np.zeros((2 * nmo, 2 * nmo), dtype=complex)
    eri_spinor = np.zeros((2 * nmo,) * 4, dtype=complex)
    for p, q in itertools.product(range(nmo), repeat=2):
        for spin in range(2):
            h_spinor[2 * p + spin, 2 * q + spin] = h1e[p, q]
    for p, q, r, s in itertools.product(range(nmo), repeat=4):
        for spin_1, spin_2 in itertools.product(range(2), repeat=2):
            eri_spinor[
                2 * p + spin_1,
                2 * q + spin_1,
                2 * r + spin_2,
                2 * s + spin_2,
            ] = eri[p, q, r, s]
    return h_spinor, eri_spinor


def _add_term(target, coefficient, reference, operators):
    if coefficient:
        target += coefficient * _apply_string(reference, operators)


def _direct_perturbers(reference, eris):
    ncore, ncas, nvirt = eris.ncore, eris.ncas, eris.nvirt
    nocc = eris.nocc
    active = range(ncore, nocc)
    core = range(ncore)
    virtual = range(nocc, eris.nmo)
    h = eris.h1eff
    w = eris.pppp.transpose(0, 2, 1, 3)
    dimension = reference.size
    result = {
        key: np.zeros(x2cscnevpt2._free_index_shape(key, eris) + (dimension,), complex)
        for key in x2cscnevpt2.SUBSPACE_ORDER
    }

    for i, j, rr, ss in itertools.product(core, core, virtual, virtual):
        vector = result["ijrs"][i, j, rr - nocc, ss - nocc]
        _add_term(
            vector,
            w[rr, ss, i, j] - w[rr, ss, j, i],
            reference,
            (("C", rr), ("C", ss), ("D", j), ("D", i)),
        )

    for rr, ss, i in itertools.product(virtual, virtual, core):
        vector = result["rsi"][rr - nocc, ss - nocc, i]
        for a in active:
            _add_term(
                vector,
                w[rr, ss, i, a] - w[ss, rr, i, a],
                reference,
                (("C", rr), ("C", ss), ("D", a), ("D", i)),
            )

    for i, j, rr in itertools.product(core, core, virtual):
        vector = result["ijr"][i, j, rr - nocc]
        for a in active:
            _add_term(
                vector,
                w[rr, a, i, j] - w[rr, a, j, i],
                reference,
                (("C", rr), ("C", a), ("D", j), ("D", i)),
            )

    for rr, ss in itertools.product(virtual, virtual):
        vector = result["rs"][rr - nocc, ss - nocc]
        for a, b in itertools.product(active, repeat=2):
            _add_term(
                vector,
                w[rr, ss, a, b],
                reference,
                (("C", rr), ("C", ss), ("D", b), ("D", a)),
            )

    for i, j in itertools.product(core, repeat=2):
        vector = result["ij"][i, j]
        for a, b in itertools.product(active, repeat=2):
            _add_term(
                vector,
                w[a, b, i, j],
                reference,
                (("C", a), ("C", b), ("D", j), ("D", i)),
            )

    for i, rr in itertools.product(core, virtual):
        vector = result["ir"][i, rr - nocc]
        _add_term(vector, h[rr, i], reference, (("C", rr), ("D", i)))
        for a, b in itertools.product(active, repeat=2):
            _add_term(
                vector,
                w[rr, a, i, b] - w[rr, a, b, i],
                reference,
                (("C", rr), ("C", a), ("D", b), ("D", i)),
            )

    for rr in virtual:
        vector = result["r"][rr - nocc]
        for a in active:
            _add_term(vector, h[rr, a], reference, (("C", rr), ("D", a)))
        for a, b, c in itertools.product(active, repeat=3):
            _add_term(
                vector,
                w[rr, a, b, c],
                reference,
                (("C", rr), ("C", a), ("D", c), ("D", b)),
            )

    for i in core:
        vector = result["i"][i]
        for a in active:
            _add_term(vector, h[a, i], reference, (("C", a), ("D", i)))
        for a, b, c in itertools.product(active, repeat=3):
            _add_term(
                vector,
                w[b, a, i, c],
                reference,
                (("C", b), ("C", a), ("D", c), ("D", i)),
            )
    return result


def _project_h_reference(h_reference, eris):
    ncore, nocc = eris.ncore, eris.nocc
    projected = {
        key: np.zeros(
            x2cscnevpt2._free_index_shape(key, eris) + (h_reference.size,),
            dtype=complex,
        )
        for key in x2cscnevpt2.SUBSPACE_ORDER
    }
    for state, amplitude in enumerate(h_reference):
        if amplitude == 0:
            continue
        holes = tuple(i for i in range(ncore) if not state & (1 << i))
        particles = tuple(
            r - nocc for r in range(nocc, eris.nmo) if state & (1 << r)
        )
        signature = (len(holes), len(particles))
        if signature == (2, 2):
            key, free = "ijrs", holes + particles
        elif signature == (1, 2):
            key, free = "rsi", particles + holes
        elif signature == (2, 1):
            key, free = "ijr", holes + particles
        elif signature == (0, 2):
            key, free = "rs", particles
        elif signature == (2, 0):
            key, free = "ij", holes
        elif signature == (1, 1):
            key, free = "ir", holes + particles
        elif signature == (0, 1):
            key, free = "r", particles
        elif signature == (1, 0):
            key, free = "i", holes
        else:
            continue
        projected[key][free + (state,)] = amplitude
    return projected


def _direct_norm_commutator(perturbers, h_active, reference_energy):
    norms = {}
    commutators = {}
    for key, vectors in perturbers.items():
        norms[key] = np.empty(vectors.shape[:-1], dtype=complex)
        commutators[key] = np.empty(vectors.shape[:-1], dtype=complex)
        for free in np.ndindex(vectors.shape[:-1]):
            vector = vectors[free]
            norm = np.vdot(vector, vector)
            norms[key][free] = norm
            commutators[key][free] = (
                np.vdot(vector, h_active(vector)) - reference_energy * norm
            )
    return norms, commutators


def _direct_commutator_branches(
    perturbers,
    h_reference_perturbers,
    h_active,
):
    """Direct right/left connected branches for a non-eigenstate reference."""

    right = {}
    left = {}
    for key, vectors in perturbers.items():
        right[key] = np.empty(vectors.shape[:-1], dtype=complex)
        left[key] = np.empty(vectors.shape[:-1], dtype=complex)
        for free in np.ndindex(vectors.shape[:-1]):
            vector = vectors[free]
            h_vector = h_active(vector)
            h_reference_vector = h_reference_perturbers[key][free]
            right[key][free] = np.vdot(vector, h_vector) - np.vdot(
                vector, h_reference_vector
            )
            left[key][free] = np.vdot(h_vector, vector) - np.vdot(
                h_reference_vector, vector
            )
    return right, left


def test_si_e_to_l_wick_arrays_against_direct_complex_fock_space():
    """SI I.E--I.L: compare every free-index norm, commutator, and energy."""

    ncore, ncas, nvirt, nelec = 2, 6, 2, 4
    nmo = ncore + ncas + nvirt
    h1e, eri = _random_physical_integrals(nmo)
    eris = spinor_helper.init_eris(h1e, eri, ncore, ncas)
    w = eris.get_phys("PPPP")

    active_slice = slice(ncore, ncore + ncas)
    active_h = eris.h1eff[active_slice, active_slice]
    active_w = w[active_slice, active_slice, active_slice, active_slice]
    active_energy, active_state = _active_ground_state(active_h, active_w, nelec)
    reference = _embed_active_state(active_state, ncore, ncas, nvirt)

    direct_perturbers = _direct_perturbers(reference, eris)
    full_h_reference = _hamiltonian_action(reference, h1e, w)
    projected = _project_h_reference(full_h_reference, eris)
    for key in x2cscnevpt2.SUBSPACE_ORDER:
        ordered = x2cscnevpt2._strict_pair_mask(
            key, direct_perturbers[key].shape[:-1]
        )
        np.testing.assert_allclose(
            direct_perturbers[key][ordered],
            projected[key][ordered],
            atol=2e-12,
            rtol=2e-12,
            err_msg=f"P_omega H|Phi> mismatch for {key}",
        )

    def active_action(full_vector):
        result = np.zeros_like(full_vector)
        for state, amplitude in enumerate(full_vector):
            if amplitude == 0:
                continue
            active_bits = (state >> ncore) & ((1 << ncas) - 1)
            local = np.zeros(1 << ncas, dtype=complex)
            local[active_bits] = amplitude
            local_result = _hamiltonian_action(local, active_h, active_w)
            spectator_bits = state & ~(((1 << ncas) - 1) << ncore)
            for output_bits, value in enumerate(local_result):
                if value:
                    result[spectator_bits | (output_bits << ncore)] += value
        return result

    direct_norm, direct_commutator = _direct_norm_commutator(
        direct_perturbers, active_action, active_energy
    )
    pdms = _raw_active_rdms(active_state, ncas)
    assert np.max(abs(pdms[2])) > 1.0e-6
    assert np.max(abs(pdms[3])) > 1.0e-6
    x2cscnevpt2.validate_pdms(pdms, ncas, nelec, atol=2e-12, rtol=2e-12)

    core_energy = np.array([-10.5, -9.0])
    virtual_energy = np.array([8.0, 11.0])
    sub_eners, sub_norms, _sub_gaps, arrays, diagnostics = (
        x2cscnevpt2._evaluate_wick_subspaces(
            eris,
            pdms,
            core_energy,
            virtual_energy,
            scalar_atol=2e-11,
            scalar_rtol=2e-11,
            return_arrays=True,
            return_diagnostics=True,
        )
    )
    for key in x2cscnevpt2.SUBSPACE_ORDER:
        np.testing.assert_allclose(
            arrays[key]["norm"], direct_norm[key], atol=2e-11, rtol=2e-11
        )
        np.testing.assert_allclose(
            arrays[key]["commutator"],
            direct_commutator[key],
            atol=2e-11,
            rtol=2e-11,
        )
        np.testing.assert_allclose(
            arrays[key]["right_commutator"],
            direct_commutator[key],
            atol=2e-11,
            rtol=2e-11,
        )
        np.testing.assert_allclose(
            arrays[key]["left_commutator"],
            direct_commutator[key],
            atol=2e-11,
            rtol=2e-11,
        )
        np.testing.assert_allclose(
            arrays[key]["si_right_commutator"],
            direct_commutator[key],
            atol=2e-11,
            rtol=2e-11,
        )
        assert diagnostics[key]["denominator_mode"] == "strict_si"
        assert diagnostics[key]["one_sided_denominator"][
            "strict_si_compatible"
        ]

        ordered = arrays[key]["ordered"]
        nonzero = ordered & (direct_norm[key].real > 1e-14)
        denominator = arrays[key]["orbital_gap"].astype(complex)
        denominator[nonzero] += (
            direct_commutator[key][nonzero] / direct_norm[key][nonzero]
        )
        expected_energy = -np.sum(
            direct_norm[key][nonzero] / denominator[nonzero]
        ).real
        np.testing.assert_allclose(sub_eners[key], expected_energy, atol=2e-12)
        np.testing.assert_allclose(
            sub_norms[key], direct_norm[key][ordered].sum().real, atol=2e-12
        )


def test_hermitian_commutator_extension_for_a_finite_residual_reference():
    """The two Wick branches remain auditable away from the eigenstate limit."""

    ncore, ncas, nvirt, nelec = 2, 4, 2, 2
    nmo = ncore + ncas + nvirt
    h1e, eri = _random_physical_integrals(nmo, seed=991)
    eris = spinor_helper.init_eris(h1e, eri, ncore, ncas)
    w = eris.get_phys("PPPP")
    active = slice(ncore, ncore + ncas)
    active_h = eris.h1eff[active, active]
    active_w = w[active, active, active, active]
    _energies, eigenstates = _active_eigenstates(active_h, active_w, nelec)
    active_state = eigenstates[0] + (0.02 - 0.03j) * eigenstates[-1]
    active_state /= np.linalg.norm(active_state)
    h_active_state = _hamiltonian_action(active_state, active_h, active_w)

    reference = _embed_active_state(active_state, ncore, ncas, nvirt)
    h_reference = _embed_active_state(
        h_active_state, ncore, ncas, nvirt
    )
    perturbers = _direct_perturbers(reference, eris)
    h_reference_perturbers = _direct_perturbers(h_reference, eris)

    def active_action(full_vector):
        result = np.zeros_like(full_vector)
        for state, amplitude in enumerate(full_vector):
            if amplitude == 0:
                continue
            active_bits = (state >> ncore) & ((1 << ncas) - 1)
            local = np.zeros(1 << ncas, dtype=complex)
            local[active_bits] = amplitude
            local_result = _hamiltonian_action(local, active_h, active_w)
            spectator_bits = state & ~(((1 << ncas) - 1) << ncore)
            for output_bits, value in enumerate(local_result):
                if value:
                    result[spectator_bits | (output_bits << ncore)] += value
        return result

    direct_right, direct_left = _direct_commutator_branches(
        perturbers, h_reference_perturbers, active_action
    )
    pdms = _raw_active_rdms(active_state, ncas)
    evaluation_kwargs = dict(
        scalar_atol=2.0e-11,
        scalar_rtol=2.0e-11,
        si_numerator_noise_allowance=1.0e-14,
        si_energy_imag_l1_tol=1.0e-12,
        si_projection_shift_l1_tol=1.0e-12,
    )
    with pytest.warns(
        x2cscnevpt2.MRPTNumericalWarning,
        match="one-sided SI denominator",
    ):
        strict_result = x2cscnevpt2._evaluate_wick_subspaces(
            eris,
            pdms,
            np.array([-9.0, -7.0]),
            np.array([6.0, 8.0]),
            denominator_mode="strict_si",
            return_diagnostics=True,
            **evaluation_kwargs,
        )
    strict_diagnostics = strict_result[-1]
    assert any(
        not strict_diagnostics[key]["one_sided_denominator"][
            "strict_si_compatible"
        ]
        for key in x2cscnevpt2.SUBSPACE_ORDER
    )

    result = x2cscnevpt2._evaluate_wick_subspaces(
        eris,
        pdms,
        np.array([-9.0, -7.0]),
        np.array([6.0, 8.0]),
        denominator_mode="hermitianized",
        return_arrays=True,
        return_diagnostics=True,
        **evaluation_kwargs,
    )
    _sub_eners, _sub_norms, _sub_gaps, arrays, diagnostics = result

    maximum_one_sided_imaginary_part = 0.0
    equations = x2cscnevpt2._compile_wick_equations()
    execution_context = x2cscnevpt2._execution_context(eris, pdms)
    for key in x2cscnevpt2.SUBSPACE_ORDER:
        np.testing.assert_allclose(
            arrays[key]["right_commutator"],
            direct_right[key],
            atol=3.0e-11,
            rtol=3.0e-11,
        )
        np.testing.assert_allclose(
            arrays[key]["left_commutator"],
            direct_left[key],
            atol=3.0e-11,
            rtol=3.0e-11,
        )
        expected_hermitian = 0.5 * (
            direct_right[key] + direct_left[key]
        )
        np.testing.assert_allclose(
            arrays[key]["commutator"],
            expected_hermitian,
            atol=3.0e-11,
            rtol=3.0e-11,
        )
        compiled_hermitian = np.zeros_like(arrays[key]["commutator"])
        local_context = dict(execution_context, commutator=compiled_hermitian)
        exec(
            equations.commutator_code[key],
            {"np": np},
            local_context,
        )
        np.testing.assert_allclose(
            compiled_hermitian,
            arrays[key]["commutator"],
            atol=3.0e-11,
            rtol=3.0e-11,
        )
        assert (
            diagnostics[key]["maximum_adjoint_error"]
            <= diagnostics[key]["reality_limit_at_maximum_error"]
        )
        assert diagnostics[key]["hermitian_commutator"][
            "maximum_imaginary_part"
        ] <= diagnostics[key]["hermitian_commutator"][
            "reality_limit_at_maximum"
        ]
        assert diagnostics[key]["zero_norm_hermitian_commutator"][
            "gate_passed"
        ]
        assert diagnostics[key]["zero_norm_right_commutator"]["gate_passed"]
        assert diagnostics[key]["zero_norm_left_commutator"]["gate_passed"]
        assert diagnostics[key]["denominator_mode"] == "hermitianized"
        assert diagnostics[key]["ordered_dimension"] == (
            diagnostics[key]["retained_dimension"]
            + diagnostics[key]["discarded_zero_norm_dimension"]
        )
        maximum_one_sided_imaginary_part = max(
            maximum_one_sided_imaginary_part,
            diagnostics[key]["one_sided_denominator"][
                "maximum_imaginary_part"
            ],
        )

    # The deliberately non-eigenstate reference makes the one-sided SI
    # commutator complex.  The independently generated Hermitian extension,
    # not a post-hoc ``.real``, is what permits the strict scalar checks.
    assert maximum_one_sided_imaginary_part > 1.0e-8
    assert any(
        not diagnostics[key]["one_sided_denominator"][
            "strict_si_compatible"
        ]
        for key in x2cscnevpt2.SUBSPACE_ORDER
    )


def test_independent_commutator_branch_residual_warns():
    mask = np.ones(2, dtype=bool)
    with pytest.warns(
        x2cscnevpt2.MRPTNumericalWarning, match="not adjoints"
    ):
        diagnostics = x2cscnevpt2._require_adjoint_pair(
            np.array([1.0 + 0.0j, 2.0 + 0.0j]),
            np.array([1.0 + 0.0j, 2.0 + 1.0e-4j]),
            mask,
            root=3,
            subspace="r",
            atol=1.0e-12,
            rtol=1.0e-12,
        )
    assert not diagnostics["adjoint_gate_passed"]


def test_zero_norm_perturber_with_nonzero_commutator_warns():
    values = np.array([0.0 + 0.0j, 0.0 + 2.0e-6j])
    with pytest.warns(
        x2cscnevpt2.MRPTNumericalWarning, match="zero-norm perturber"
    ):
        diagnostics = x2cscnevpt2._validate_zero_norm_commutator(
            values,
            np.ones(2, dtype=bool),
            root=4,
            subspace="r",
            atol=1.0e-12,
            rtol=1.0e-12,
        )
    assert not diagnostics["gate_passed"]


def test_strict_si_near_null_roundoff_is_explicitly_energy_bounded():
    norm = np.array([3.8288e-12 + 0.0j])
    gap = np.array([8.0 + 4.9534e-6j])
    numerator = np.array([0.0 + norm[0].real * gap[0].imag * 1j])
    real_gap, diagnostics = x2cscnevpt2._audit_one_sided_si_denominator(
        norm,
        numerator,
        gap,
        np.ones(1, dtype=bool),
        root=3,
        subspace="i",
        gap_atol=1.0e-10,
        gap_rtol=1.0e-9,
        numerator_noise_allowance=1.0e-12,
        energy_imag_l1_tol=1.0e-10,
        projection_shift_l1_tol=1.0e-12,
        denominator_tol=1.0e-12,
        enforce=True,
    )
    np.testing.assert_array_equal(real_gap, np.array([8.0]))
    assert not diagnostics["direct_gap_gate_passed"]
    assert diagnostics["roundoff_limited_near_null_count"] == 1
    assert diagnostics["strict_si_compatible"]
    assert diagnostics["imaginary_energy_l1"] < 1.0e-18
    assert diagnostics["real_projection_shift_l1"] < 1.0e-20


def test_empty_strict_si_class_has_a_complete_persistable_audit():
    _real_gap, diagnostics = x2cscnevpt2._audit_one_sided_si_denominator(
        np.zeros(2, dtype=complex),
        np.zeros(2, dtype=complex),
        np.ones(2, dtype=complex),
        np.zeros(2, dtype=bool),
        root=0,
        subspace="ij",
        gap_atol=1.0e-10,
        gap_rtol=1.0e-9,
        numerator_noise_allowance=1.0e-12,
        energy_imag_l1_tol=1.0e-10,
        projection_shift_l1_tol=1.0e-12,
        denominator_tol=1.0e-12,
        enforce=True,
    )
    assert diagnostics["strict_si_compatible"]
    assert diagnostics["imaginary_energy_l1_gate_passed"]
    assert diagnostics["real_projection_shift_l1_gate_passed"]
    assert diagnostics["maximum_numerator_imaginary_index"] == []


def test_strict_si_resolved_residual_and_projection_shift_warn():
    mask = np.ones(1, dtype=bool)
    with pytest.warns(
        x2cscnevpt2.MRPTNumericalWarning,
        match="one-sided SI denominator",
    ):
        _gap, diagnostics = x2cscnevpt2._audit_one_sided_si_denominator(
            np.array([1.0e-2 + 0.0j]),
            np.array([0.0 + 1.0e-5j]),
            np.array([8.0 + 1.0e-3j]),
            mask,
            root=1,
            subspace="r",
            gap_atol=1.0e-10,
            gap_rtol=1.0e-9,
            numerator_noise_allowance=1.0e-12,
            energy_imag_l1_tol=1.0e-10,
            projection_shift_l1_tol=1.0e-12,
            denominator_tol=1.0e-12,
            enforce=True,
        )
    assert not diagnostics["strict_si_compatible"]

    # A huge imaginary gap can make Im(-N/D) small while changing its real
    # part substantially; the projection-distance L1 gate catches that case.
    with pytest.warns(
        x2cscnevpt2.MRPTNumericalWarning, match="projection shift"
    ):
        _gap, diagnostics = x2cscnevpt2._audit_one_sided_si_denominator(
            np.array([1.0e-12 + 0.0j]),
            np.array([0.0 + 1.0e-4j]),
            np.array([1.0 + 1.0e8j]),
            mask,
            root=2,
            subspace="r",
            gap_atol=1.0e-10,
            gap_rtol=1.0e-9,
            numerator_noise_allowance=2.0e-4,
            energy_imag_l1_tol=1.0e-10,
            projection_shift_l1_tol=1.0e-14,
            denominator_tol=1.0e-12,
            enforce=True,
        )
    assert not diagnostics["real_projection_shift_l1_gate_passed"]


def test_invalid_denominator_mode_is_rejected_before_wick_evaluation():
    with pytest.raises(ValueError, match="denominator_mode"):
        x2cscnevpt2._normalize_denominator_mode("silent-real-part")


def test_wick_bra_uses_explicit_conjugate_coefficient_names():
    equations = x2cscnevpt2._compile_wick_equations()
    combined = "\n".join(equations.norm_code.values())
    assert "wc" in combined
    assert "hc" in "\n".join(equations.commutator_code.values())
    assert "dm5" not in x2cscnevpt2.dump_wick_equations()


def test_raw_rdm_validator_checks_every_chunk():
    """The bounded-memory path remains an exhaustive, not sampled, check."""

    ncas, nelec = 4, 2
    determinant = np.zeros(1 << ncas, dtype=complex)
    determinant[(1 << 0) | (1 << 1)] = 1.0
    pdms = _raw_active_rdms(determinant, ncas)
    checked, diagnostics = x2cscnevpt2.validate_pdms(
        pdms,
        ncas,
        nelec,
        atol=2.0e-12,
        rtol=2.0e-12,
        work_memory=64,
    )
    assert checked[3].shape == (ncas,) * 8
    assert diagnostics["dm4"]["shape"] == [ncas] * 8
    assert diagnostics["dm1"]["trace"] == pytest.approx(nelec)

    broken = list(pdms)
    broken[3] = broken[3].copy()
    broken[3][3, 2, 1, 0, 0, 1, 2, 3] = 1.0e-3
    with pytest.warns(
        x2cscnevpt2.MRPTNumericalWarning,
        match="dm4 violates raw SGF",
    ):
        _checked, diagnostics = x2cscnevpt2.validate_pdms(
            broken,
            ncas,
            nelec,
            atol=2.0e-12,
            rtol=2.0e-12,
            work_memory=64,
        )
    assert not diagnostics["dm4"]["symmetry_gate_passed"]


def test_complex_raw_dm1_is_transposed_at_the_pyscf_jk_boundary():
    """Raw <C[p]D[q]> and PySCF's covariant AO density have opposite axes."""

    rng = np.random.default_rng(7812)
    trial = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    mo, _ = np.linalg.qr(trial)
    raw_dm1 = np.array(
        [[0.8, 0.17 + 0.31j], [0.17 - 0.31j, 0.2]],
        dtype=complex,
    )
    captured = {}

    class MeanField:
        def get_jk(self, _mol, density):
            captured["density"] = np.array(density, copy=True)
            zeros = np.zeros_like(density)
            return zeros, zeros

    mc = SimpleNamespace(
        ci=None,
        mo_coeff=mo,
        ncore=1,
        ncas=2,
        nelecas=1,
        mol=SimpleNamespace(),
        _scf=MeanField(),
        get_hcore=lambda: np.zeros((4, 4), dtype=complex),
    )
    zmcscf.get_fock(mc, mo_coeff=mo, casdm1=raw_dm1)

    core = mo[:, :1]
    active = mo[:, 1:3]
    expected = core @ core.T.conj() + active @ raw_dm1.T @ active.T.conj()
    wrong = core @ core.T.conj() + active @ raw_dm1 @ active.T.conj()
    np.testing.assert_allclose(captured["density"], expected, atol=2.0e-15)
    assert np.max(abs(captured["density"] - wrong)) > 1.0e-2


def test_pyblock2_raw_rdm1234_and_multiroot_selection(tmp_path):
    """Block2 0.5.4rc16 SGF axes equal C...C D...D operator order."""

    from pyblock2.driver.core import DMRGDriver, SymmetryTypes

    ncore, ncas, nvirt, nelec = 2, 6, 2, 4
    h1e, eri = _random_physical_integrals(ncore + ncas + nvirt, seed=11902)
    eris = spinor_helper.init_eris(h1e, eri, ncore, ncas)
    active = slice(ncore, ncore + ncas)
    w = eris.get_phys("PPPP")
    active_energies, active_states = _active_eigenstates(
        eris.h1eff[active, active],
        w[active, active, active, active],
        nelec,
    )

    basis = _fixed_particle_basis(ncas, nelec)
    occupations = np.zeros((len(basis), ncas), dtype=np.uint8)
    for row, state in enumerate(basis):
        occupations[row] = [(state >> site) & 1 for site in range(ncas)]

    driver = DMRGDriver(
        stack_mem=300_000_000,
        scratch=str(tmp_path / "npdm"),
        clean_scratch=True,
        symm_type=SymmetryTypes.SGFCPX,
        n_threads=1,
    )
    driver.initialize_system(
        n_sites=ncas,
        n_elec=nelec,
        orb_sym=[0] * ncas,
    )
    kets = tuple(
        driver.get_mps_from_csf_coefficients(
            occupations,
            active_states[root][basis],
            f"ROOT-{root}",
            dot=1,
            iprint=0,
        )
        for root in range(2)
    )
    solver = SimpleNamespace(
        driver=driver,
        kets=kets,
        ncas=ncas,
        nelecas=nelec,
        npdm_site_type=2,
        npdm_cutoff=1.0e-24,
    )
    try:
        root_pdms = []
        for root in range(2):
            block2_pdms = x2cscnevpt2.make_dm1234(solver, root=root)
            direct_pdms = _raw_active_rdms(active_states[root], ncas)
            assert np.max(abs(direct_pdms[2])) > 1.0e-6
            assert np.max(abs(direct_pdms[3])) > 1.0e-6
            for rank, (actual, expected) in enumerate(
                zip(block2_pdms, direct_pdms, strict=True), start=1
            ):
                np.testing.assert_allclose(
                    actual,
                    expected,
                    atol=2e-11,
                    rtol=2e-11,
                    err_msg=f"root {root} raw dm{rank}",
                )
            x2cscnevpt2.validate_pdms(
                block2_pdms, ncas, nelec, atol=2e-11, rtol=2e-11
            )
            root_pdms.append(block2_pdms)

        # Both roots must be genuinely selected; returning root zero twice is
        # detectable already in the one-particle density.
        assert np.max(abs(root_pdms[0][0] - root_pdms[1][0])) > 1e-3

        core_energy = np.array([-10.0, -8.5])
        virtual_energy = np.array([7.5, 9.5])
        for root in range(2):
            from_block2 = x2cscnevpt2._evaluate_wick_subspaces(
                eris,
                root_pdms[root],
                core_energy,
                virtual_energy,
                root=root,
            )
            from_direct = x2cscnevpt2._evaluate_wick_subspaces(
                eris,
                _raw_active_rdms(active_states[root], ncas),
                core_energy,
                virtual_energy,
                root=root,
            )
            np.testing.assert_allclose(
                [from_block2[0][key] for key in x2cscnevpt2.SUBSPACE_ORDER],
                [from_direct[0][key] for key in x2cscnevpt2.SUBSPACE_ORDER],
                atol=2e-11,
                rtol=2e-11,
            )
            assert np.isfinite(active_energies[root])
    finally:
        driver.finalize()


def test_nonrelativistic_alpha_beta_embedding_matches_spatial_scnevpt2(
    monkeypatch,
):
    """Real spatial CASCI expanded to spin orbitals agrees class by class."""

    mol = gto.M(
        atom="Li 0 0 0; H 0 0 1.6",
        basis="sto-3g",
        spin=0,
        verbose=0,
    )
    mean_field = scf.RHF(mol).run(conv_tol=1.0e-13)
    assert mean_field.converged
    mc = mcscf.CASCI(mean_field, 2, 2)
    mc.fcisolver.conv_tol = 1.0e-14
    mc.kernel()
    assert mc.converged

    captured = {}
    spatial_names = {
        "Sijrs": "ijrs",
        "Srsi": "rsi",
        "Sijr": "ijr",
        "Srs": "rs",
        "Sij": "ij",
        "Sir": "ir",
        "Sr": "r",
        "Si": "i",
    }
    for function_name, key in spatial_names.items():
        original = getattr(pyscf_nevpt2, function_name)

        def recording(*args, _original=original, _key=key, **kwargs):
            result = _original(*args, **kwargs)
            captured[_key] = float(result[1])
            return result

        monkeypatch.setattr(pyscf_nevpt2, function_name, recording)

    spatial_pt = pyscf_nevpt2.NEVPT(mc, density_fit=False)
    spatial_pt.canonicalized = True
    spatial_energy = spatial_pt.kernel()
    assert set(captured) == set(x2cscnevpt2.SUBSPACE_ORDER)

    mo = mc.mo_coeff
    nmo = mo.shape[1]
    spatial_h1e = mo.T @ mean_field.get_hcore() @ mo
    spatial_eri = ao2mo.restore(1, ao2mo.kernel(mol, mo), nmo)
    h1e, eri = _spatial_to_spinor_integrals(spatial_h1e, spatial_eri)
    eris = spinor_helper.init_eris(
        h1e,
        eri,
        ncore=2 * mc.ncore,
        ncas=2 * mc.ncas,
    )
    active = slice(eris.ncore, eris.nocc)
    w = eris.get_phys("PPPP")
    _active_energy, active_state = _active_ground_state(
        eris.h1eff[active, active],
        w[active, active, active, active],
        nelec=2,
    )
    pdms = _raw_active_rdms(active_state, eris.ncas)
    spinor_subspaces = x2cscnevpt2._evaluate_wick_subspaces(
        eris,
        pdms,
        np.repeat(mc.mo_energy[: mc.ncore], 2),
        np.repeat(mc.mo_energy[mc.ncore + mc.ncas :], 2),
        strong_contraction_groups=np.repeat(np.arange(nmo), 2),
    )[0]
    spinor_energy = sum(spinor_subspaces.values())

    np.testing.assert_allclose(spinor_energy, spatial_energy, atol=1.0e-10)
    for key in x2cscnevpt2.SUBSPACE_ORDER:
        np.testing.assert_allclose(
            spinor_subspaces[key], captured[key], atol=1.0e-10
        )


def test_complex_active_rotation_and_core_virtual_resemicanonicalization():
    rng = np.random.default_rng(44191)
    ncore, ncas, nvirt, nelec = 2, 4, 2, 2
    h1e, eri = _random_physical_integrals(ncore + ncas + nvirt, seed=721)
    eris = spinor_helper.init_eris(h1e, eri, ncore, ncas)
    active = slice(eris.ncore, eris.nocc)
    w = eris.get_phys("PPPP")
    _energy, state = _active_ground_state(
        eris.h1eff[active, active],
        w[active, active, active, active],
        nelec,
    )
    pdms = _raw_active_rdms(state, ncas)
    core_energy = np.array([-9.0, -7.5])
    virtual_energy = np.array([6.5, 10.0])
    reference = x2cscnevpt2._evaluate_wick_subspaces(
        eris, pdms, core_energy, virtual_energy
    )[0]

    # A consistent active rotation changes both the integrals and reference
    # RDMs but no SC class.  Re-diagonalizing the tiny exact active Hamiltonian
    # supplies that same physical state in the rotated basis.
    active_rotation = np.eye(eris.nmo, dtype=complex)
    active_rotation[active, active] = _random_unitary(rng, ncas)
    active_eris = x2cscnevpt2._rotate_eris(eris, active_rotation)
    active_w = active_eris.get_phys("PPPP")
    _rotated_energy, rotated_state = _active_ground_state(
        active_eris.h1eff[active, active],
        active_w[active, active, active, active],
        nelec,
    )
    active_result = x2cscnevpt2._evaluate_wick_subspaces(
        active_eris,
        _raw_active_rdms(rotated_state, ncas),
        core_energy,
        virtual_energy,
    )[0]
    np.testing.assert_allclose(
        [active_result[key] for key in x2cscnevpt2.SUBSPACE_ORDER],
        [reference[key] for key in x2cscnevpt2.SUBSPACE_ORDER],
        atol=3.0e-11,
        rtol=3.0e-11,
    )

    # An arbitrary core/virtual rotation destroys diagonal orbital energies.
    # Transform the synthetic Fock blocks with it, diagonalize them again, and
    # apply the resulting local rotations to the integrals before comparison.
    cv_rotation = np.eye(eris.nmo, dtype=complex)
    cv_rotation[:ncore, :ncore] = _random_unitary(rng, ncore)
    cv_rotation[eris.nocc :, eris.nocc :] = _random_unitary(rng, nvirt)
    noncanonical_eris = x2cscnevpt2._rotate_eris(eris, cv_rotation)

    recanonicalize = np.eye(eris.nmo, dtype=complex)
    rotated_core_fock = (
        cv_rotation[:ncore, :ncore].T.conj()
        @ np.diag(core_energy)
        @ cv_rotation[:ncore, :ncore]
    )
    new_core_energy, recanonicalize[:ncore, :ncore] = np.linalg.eigh(
        rotated_core_fock
    )
    rotated_virtual_fock = (
        cv_rotation[eris.nocc :, eris.nocc :].T.conj()
        @ np.diag(virtual_energy)
        @ cv_rotation[eris.nocc :, eris.nocc :]
    )
    new_virtual_energy, recanonicalize[eris.nocc :, eris.nocc :] = np.linalg.eigh(
        rotated_virtual_fock
    )
    recanonical_eris = x2cscnevpt2._rotate_eris(
        noncanonical_eris, recanonicalize
    )
    cv_result = x2cscnevpt2._evaluate_wick_subspaces(
        recanonical_eris,
        pdms,
        new_core_energy,
        new_virtual_energy,
    )[0]
    np.testing.assert_allclose(
        [cv_result[key] for key in x2cscnevpt2.SUBSPACE_ORDER],
        [reference[key] for key in x2cscnevpt2.SUBSPACE_ORDER],
        atol=3.0e-11,
        rtol=3.0e-11,
    )


def test_rotate_eris_preserves_an_explicit_h1eff_override():
    rng = np.random.default_rng(9271)
    ncore, ncas, nvirt = 2, 4, 2
    h1e, eri = _random_physical_integrals(ncore + ncas + nvirt, seed=1982)
    automatic = spinor_helper.init_eris(h1e, eri, ncore, ncas)
    trial = rng.normal(size=h1e.shape) + 1j * rng.normal(size=h1e.shape)
    override = automatic.h1eff + 0.03 * (trial + trial.T.conj())
    eris = spinor_helper.init_eris(
        h1e, eri, ncore, ncas, h1eff=override
    )

    rotation = np.zeros(h1e.shape, dtype=complex)
    start = 0
    for size in (ncore, ncas, nvirt):
        rotation[start : start + size, start : start + size] = _random_unitary(
            rng, size
        )
        start += size
    rotated = x2cscnevpt2._rotate_eris(eris, rotation)
    expected = rotation.T.conj() @ override @ rotation
    np.testing.assert_allclose(rotated.h1eff, expected, atol=3.0e-13)

    rebuilt = spinor_helper.init_eris(
        rotated.h1e, rotated.pppp, ncore, ncas
    )
    assert np.max(abs(rotated.h1eff - rebuilt.h1eff)) > 1.0e-3


@pytest.mark.parametrize("compact", [False, True])
def test_external_eris_basis_contract_prevents_double_semicanonical_rotation(compact):
    rng = np.random.default_rng(41827)
    ncore, ncas, nvirt, nelec = 2, 4, 2, 2
    nmo = ncore + ncas + nvirt
    h1e, eri = _random_physical_integrals(nmo, seed=8812)
    semicanonical_eris = spinor_helper.init_eris(
        h1e, eri, ncore=ncore, ncas=ncas
    )

    rotation = np.eye(nmo, dtype=complex)
    rotation[:ncore, :ncore] = _random_unitary(rng, ncore)
    rotation[-nvirt:, -nvirt:] = _random_unitary(rng, nvirt)
    input_eris = x2cscnevpt2._rotate_eris(
        semicanonical_eris, rotation.T.conj()
    )

    active = slice(ncore, ncore + ncas)
    w = semicanonical_eris.get_phys("PPPP")
    _energy, state = _active_ground_state(
        semicanonical_eris.h1eff[active, active],
        w[active, active, active, active],
        nelec,
    )
    pdms = _raw_active_rdms(state, ncas)
    orbital_energy = np.array([-9.0, -7.0, 0.0, 0.0, 0.0, 0.0, 6.0, 8.0])

    molecule = SimpleNamespace(verbose=0, stdout=sys.stdout)

    class MeanField:
        mol = molecule

        def get_ovlp(self):
            return np.eye(nmo)

    solver = SimpleNamespace(nelecas=(1, 1), nroots=1)
    mc = SimpleNamespace(
        _scf=MeanField(),
        mol=molecule,
        verbose=0,
        stdout=sys.stdout,
        ncore=ncore,
        ncas=ncas,
        nelecas=(1, 1),
        frozen=0,
        fcisolver=solver,
        mo_coeff=np.eye(nmo, dtype=complex),
        e_tot=-42.0,
    )

    def prepared_semicanonicalization(_mc, _mo, _dm1, _root):
        return rotation, orbital_energy

    from_input = x2cscnevpt2.WickX2CSCNEVPT2(mc)
    from_input._semicanonicalize = prepared_semicanonicalization
    input_energy = from_input.kernel(
        pdms=pdms, eris=input_eris, eris_basis="input_mo", compact_eris=compact
    )

    already_semicanonical = x2cscnevpt2.WickX2CSCNEVPT2(mc)
    already_semicanonical._semicanonicalize = prepared_semicanonicalization
    semicanonical_energy = already_semicanonical.kernel(
        pdms=pdms,
        eris=semicanonical_eris,
        eris_basis="semicanonical",
        compact_eris=compact,
    )

    if compact:
        assert not hasattr(already_semicanonical.eris, "pppp")
        for key in x2cscnevpt2._W_KEYS:
            np.testing.assert_array_equal(already_semicanonical.eris.get_phys(key),
                                          semicanonical_eris.get_phys(key))
    else:
        assert already_semicanonical.eris is semicanonical_eris
    assert already_semicanonical.eris_basis == "semicanonical"
    np.testing.assert_allclose(input_energy, semicanonical_energy, atol=3.0e-12)
    np.testing.assert_allclose(
        [from_input.sub_eners[key] for key in x2cscnevpt2.SUBSPACE_ORDER],
        [
            already_semicanonical.sub_eners[key]
            for key in x2cscnevpt2.SUBSPACE_ORDER
        ],
        atol=3.0e-12,
    )

    with pytest.raises(ValueError, match="eris_basis"):
        already_semicanonical.kernel(
            pdms=pdms, eris=semicanonical_eris, eris_basis="unknown"
        )


def test_default_semicanonicalization_uses_complex_raw_dm1_convention():
    """Exercise the real adapter path, not only the isolated JK boundary."""

    nmo, ncore, ncas = 6, 2, 2
    molecule = SimpleNamespace(verbose=0, stdout=sys.stdout)
    captured = {}

    class MeanField:
        mol = molecule

        def get_ovlp(self):
            return np.eye(nmo)

        def get_jk(self, _mol, density):
            captured["density"] = np.array(density, copy=True)
            z = density[ncore, ncore + 1]
            vj = np.zeros((nmo, nmo), dtype=complex)
            vj[0, 1], vj[1, 0] = z, z.conjugate()
            vj[4, 5], vj[5, 4] = 0.7j * z, (0.7j * z).conjugate()
            return vj, np.zeros_like(vj)

    mean_field = MeanField()
    hcore = np.diag([-2.1, -1.2, -0.5, -0.2, 0.4, 1.3]).astype(complex)
    ci = object()
    mc = SimpleNamespace(
        _scf=mean_field,
        mol=molecule,
        verbose=0,
        stdout=sys.stdout,
        ncore=ncore,
        ncas=ncas,
        nelecas=1,
        frozen=None,
        orbital_symmetry=None,
        fcisolver=SimpleNamespace(),
        ci=ci,
        mo_coeff=np.eye(nmo, dtype=complex),
        mo_energy=None,
        e_tot=-1.0,
        get_hcore=lambda: hcore,
    )
    mc.get_fock = MethodType(zmcscf.get_fock, mc)
    mc.canonicalize = MethodType(zmcscf.canonicalize, mc)
    raw_dm1 = np.array(
        [[0.8, 0.17 + 0.31j], [0.17 - 0.31j, 0.2]], dtype=complex
    )

    adapter = x2cscnevpt2.WickX2CSCNEVPT2(mc)
    rotated, energies = adapter._semicanonicalize(
        mc, mc.mo_coeff, raw_dm1, root=0
    )

    expected_density = np.zeros((nmo, nmo), dtype=complex)
    expected_density[:ncore, :ncore] = np.eye(ncore)
    expected_density[ncore : ncore + ncas, ncore : ncore + ncas] = raw_dm1.T
    np.testing.assert_allclose(captured["density"], expected_density, atol=2.0e-15)
    np.testing.assert_array_equal(
        rotated[:, ncore : ncore + ncas],
        mc.mo_coeff[:, ncore : ncore + ncas],
    )

    correct_fock = hcore + mean_field.get_jk(None, expected_density)[0]
    correct_mo_fock = rotated.T.conj() @ correct_fock @ rotated
    np.testing.assert_allclose(
        correct_mo_fock[:ncore, :ncore],
        np.diag(np.diag(correct_mo_fock[:ncore, :ncore])),
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        correct_mo_fock[ncore + ncas :, ncore + ncas :],
        np.diag(np.diag(correct_mo_fock[ncore + ncas :, ncore + ncas :])),
        atol=2.0e-13,
    )
    np.testing.assert_allclose(energies, np.diag(correct_mo_fock).real, atol=2e-13)

    wrong_density = expected_density.copy()
    wrong_density[ncore : ncore + ncas, ncore : ncore + ncas] = raw_dm1
    wrong_fock = hcore + mean_field.get_jk(None, wrong_density)[0]
    wrong_mo_fock = rotated.T.conj() @ wrong_fock @ rotated
    wrong_offdiagonal = max(
        abs(wrong_mo_fock[0, 1]), abs(wrong_mo_fock[4, 5])
    )
    assert wrong_offdiagonal > 1.0e-2


@pytest.mark.parametrize("compact", [False, True])
def test_stream_object_injected_kernel_interface_and_result_fields(compact):
    ncore, ncas, nvirt, nelec = 2, 4, 2, 2
    h1e, eri = _random_physical_integrals(ncore + ncas + nvirt, seed=809)
    eris = spinor_helper.init_eris(h1e, eri, ncore, ncas)
    active = slice(eris.ncore, eris.nocc)
    w = eris.get_phys("PPPP")
    _energy, state = _active_ground_state(
        eris.h1eff[active, active],
        w[active, active, active, active],
        nelec,
    )
    pdms = _raw_active_rdms(state, ncas)
    mo_energy = np.array([-10.0, -8.0, 0.0, 0.0, 0.0, 0.0, 7.0, 9.0])
    molecule = SimpleNamespace(verbose=0, stdout=sys.stdout)
    mean_field = SimpleNamespace(mol=molecule)
    solver = SimpleNamespace(nelecas=nelec, nroots=1)
    mc = SimpleNamespace(
        _scf=mean_field,
        mol=molecule,
        verbose=0,
        stdout=sys.stdout,
        ncore=ncore,
        ncas=ncas,
        nelecas=nelec,
        fcisolver=solver,
        mo_coeff=np.eye(eris.nmo, dtype=complex),
        mo_energy=mo_energy,
        e_tot=-75.125,
    )

    result = x2cscnevpt2.WickX2CSCNEVPT2(mc)
    assert result.rdm_work_memory == 512 * 2**20
    result.canonicalized = True
    returned = result.kernel(root=0, pdms=pdms, eris=eris, compact_eris=compact)
    np.testing.assert_allclose(returned, result.e_corr)
    assert result.root == 0
    if compact:
        assert not hasattr(result.eris, "pppp")
        for key in x2cscnevpt2._W_KEYS:
            np.testing.assert_array_equal(result.eris.get_phys(key), eris.get_phys(key))
    else:
        assert result.eris is eris
    assert result.mo_coeff is mc.mo_coeff
    assert set(result.sub_eners) == set(x2cscnevpt2.SUBSPACE_ORDER)
    assert set(result.sub_norms) == set(x2cscnevpt2.SUBSPACE_ORDER)
    assert set(result.sub_times) == {
        "pdms",
        "eris",
        *x2cscnevpt2.SUBSPACE_ORDER,
    }
    np.testing.assert_allclose(result.e_corr, sum(result.sub_eners.values()))
    np.testing.assert_allclose(result.e_tot, mc.e_tot + result.e_corr)
