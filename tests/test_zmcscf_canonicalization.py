import sys
from types import SimpleNamespace

import numpy as np

from socutils.mcscf import zmcscf


class _FakeSCF:
    def __init__(self, overlap):
        self._overlap = overlap

    def get_ovlp(self):
        return self._overlap


class _FakeCASSCF:
    def __init__(self, fock_ao, overlap, ncore, ncas):
        self._fock_ao = fock_ao
        self._scf = _FakeSCF(overlap)
        self.ncore = ncore
        self.ncas = ncas
        self.nelecas = 2
        self.frozen = None
        self.orbital_symmetry = None
        self.fcisolver = SimpleNamespace()
        self.verbose = 0
        self.stdout = sys.stdout
        self.canonicalization_diagnostics = None

    def get_fock(
        self,
        mo_coeff=None,
        ci=None,
        eris=None,
        casdm1=None,
        verbose=None,
    ):
        return self._fock_ao


def test_canonicalize_preserves_active_basis_and_ci_object():
    rng = np.random.default_rng(9127)
    trial = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    mo_coeff, _ = np.linalg.qr(trial)

    fock_mo = np.zeros((8, 8), dtype=complex)
    fock_mo[:2, :2] = np.array(
        [[-2.1, 0.18 + 0.07j], [0.18 - 0.07j, -1.2]]
    )
    fock_mo[2:5, 2:5] = np.array(
        [
            [-0.7, 0.11j, -0.03],
            [-0.11j, -0.4, 0.06 + 0.02j],
            [-0.03, 0.06 - 0.02j, -0.1],
        ]
    )
    fock_mo[5:, 5:] = np.array(
        [
            [0.4, 0.13 - 0.05j, 0.02],
            [0.13 + 0.05j, 0.9, -0.08j],
            [0.02, 0.08j, 1.5],
        ]
    )
    # Interspace elements do not enter the redundant-space eigensolves, but
    # make this a realistic generalized Fock matrix rather than block diagonal.
    coupling = rng.normal(scale=0.03, size=(5, 3))
    coupling = coupling + 1j * rng.normal(scale=0.03, size=(5, 3))
    fock_mo[:5, 5:] = coupling
    fock_mo[5:, :5] = coupling.T.conj()
    fock_ao = mo_coeff.dot(fock_mo).dot(mo_coeff.T.conj())

    mc = _FakeCASSCF(fock_ao, np.eye(8), ncore=2, ncas=3)
    ci_or_mps = object()
    active_before = mo_coeff[:, 2:5].copy()
    core_projector_before = mo_coeff[:, :2].dot(mo_coeff[:, :2].T.conj())
    virtual_projector_before = mo_coeff[:, 5:].dot(mo_coeff[:, 5:].T.conj())

    canonical_mo, returned_ci, mo_energy = zmcscf.canonicalize(
        mc,
        mo_coeff,
        ci_or_mps,
        casdm1=np.diag([0.9, 0.7, 0.4]),
        verbose=0,
    )

    final_fock = canonical_mo.T.conj().dot(fock_ao).dot(canonical_mo)
    np.testing.assert_array_equal(canonical_mo[:, 2:5], active_before)
    assert returned_ci is ci_or_mps
    np.testing.assert_allclose(mo_energy, np.diag(final_fock).real, atol=1e-13)
    np.testing.assert_allclose(
        canonical_mo[:, :2].dot(canonical_mo[:, :2].T.conj()),
        core_projector_before,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        canonical_mo[:, 5:].dot(canonical_mo[:, 5:].T.conj()),
        virtual_projector_before,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        final_fock[:2, :2], np.diag(np.diag(final_fock[:2, :2])), atol=1e-13
    )
    np.testing.assert_allclose(
        final_fock[5:, 5:], np.diag(np.diag(final_fock[5:, 5:])), atol=1e-13
    )
    assert mc.canonicalization_diagnostics["active_orbital_change"] == 0.0
    assert mc.canonicalization_diagnostics["ci_object_preserved"]
