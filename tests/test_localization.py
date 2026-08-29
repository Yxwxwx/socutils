import numpy as np
import scipy.linalg
from pyscf import gto, scf

from socutils.dmrg.kramers import identify_kramers_orbitals
from socutils.lo import localize_boys, localize_boys_kramers
from socutils.lo.boys import _boys_gradient, boys_objective
from socutils.scf import spinor_hf


def _projector(mo_coeff, overlap):
    return mo_coeff.dot(mo_coeff.T.conj()).dot(overlap)


def test_complex_boys_gradient_matches_finite_difference():
    rng = np.random.default_rng(917)
    raw = rng.normal(size=(3, 3, 3)) + 1j * rng.normal(size=(3, 3, 3))
    dipoles = 0.5 * (raw + raw.transpose(0, 2, 1).conj())
    raw_direction = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    direction = 0.5 * (raw_direction - raw_direction.T.conj())
    epsilon = 1e-6

    def rotated_objective(step):
        unitary = scipy.linalg.expm(step * direction)
        transformed = np.array(
            [unitary.T.conj().dot(matrix).dot(unitary) for matrix in dipoles]
        )
        return boys_objective(transformed)

    finite_difference = (
        rotated_objective(epsilon) - rotated_objective(-epsilon)
    ) / (2.0 * epsilon)
    analytic = np.vdot(_boys_gradient(dipoles), direction).real
    assert abs(finite_difference - analytic) <= 1e-8


def test_complex_boys_localization_is_unitary_and_monotone():
    mol = gto.M(
        atom="H 0 0 -0.7; H 0 0 0.7",
        basis="sto-3g",
        spin=0,
        verbose=0,
    )
    mf = scf.RHF(mol).run()
    phase_rotation = np.array(
        [
            [np.cos(0.31), np.sin(0.31) * np.exp(0.23j)],
            [-np.sin(0.31) * np.exp(-0.23j), np.cos(0.31)],
        ]
    )
    initial = mf.mo_coeff.astype(complex).dot(phase_rotation)
    localized, result = localize_boys(
        mol,
        initial,
        distance_threshold=None,
        conv_tol=1e-6,
        max_cycle=200,
        return_info=True,
        verbose=0,
    )
    overlap = mf.get_ovlp()

    assert result.converged
    assert result.final_objective >= result.initial_objective - 1e-12
    assert all(
        later["objective"] >= earlier["objective"] - 1e-12
        for earlier, later in zip(result.history, result.history[1:])
    )
    assert np.max(abs(localized.T.conj().dot(overlap).dot(localized) - np.eye(2))) <= 1e-10
    assert np.max(
        abs(_projector(localized, overlap) - _projector(initial, overlap))
    ) <= 1e-10


def test_kramers_boys_handles_nonadjacent_partner_order():
    mol = gto.M(
        atom="H 0 0 -0.7; H 0 0 0.7",
        basis="sto-3g",
        spin=0,
        verbose=0,
    )
    mf = spinor_hf.KRHF(mol).x2camf(
        with_gaunt=False,
        with_breit=False,
    )
    mf.init_guess = "1e"
    mf.conv_tol = 1e-12
    mf.kernel()
    assert mf.converged

    initial = mf.mo_coeff[:, [0, 2, 1, 3]]
    initial_mapping = identify_kramers_orbitals(
        mol,
        initial,
        mf.get_ovlp(),
    )
    assert any(abs(first - second) != 1 for first, second in initial_mapping.pairs)

    localized, result = localize_boys_kramers(
        mol,
        initial,
        distance_threshold=None,
        conv_tol=1e-6,
        max_cycle=200,
        return_info=True,
        verbose=0,
    )
    final_mapping = identify_kramers_orbitals(
        mol,
        localized,
        mf.get_ovlp(),
        tolerance=1e-7,
    )
    overlap = mf.get_ovlp()

    assert result.converged
    assert result.final_objective > result.initial_objective + 1.0
    assert any(row.get("stationary_escape", False) for row in result.history)
    assert result.symmetry_residual <= 1e-10
    assert final_mapping.diagnostics["subspace_closure_error"] <= 1e-7
    assert final_mapping.diagnostics["partner_orbital_error"] <= 1e-7
    assert np.max(abs(localized.T.conj().dot(overlap).dot(localized) - np.eye(4))) <= 1e-9
    assert np.max(
        abs(_projector(localized, overlap) - _projector(initial, overlap))
    ) <= 1e-9
