# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent oracle for original/QR native NPDM of the same actual MPS."""

import json

import numpy as np

from .carbon import (
    annihilated_coefficients,
    audit_reference,
    density_from_annihilated,
    json_value,
)
from .carbon_reference import DIRECTORY, canonicalize_npdm_reference, reference


def audit():
    mc, solver = reference()
    results = []
    try:
        roots, vectors, _ = audit_reference(mc, solver)
        original = canonicalize_npdm_reference(mc)
        _, qr_vectors, _ = audit_reference(mc, solver)
        np.testing.assert_allclose(qr_vectors, vectors, atol=1e-12, rtol=0)
        for bra, ket in ((roots[0], roots[0]), (roots[0], roots[1])):
            for rank in (3, 4):
                expected = density_from_annihilated(
                    annihilated_coefficients(vectors[:, bra], 8, 4, rank),
                    annihilated_coefficients(vectors[:, ket], 8, 4, rank),
                    8,
                    rank,
                )
                for name, states in (("original", original), ("qr", solver.kets)):
                    for cutoff in (1e-24, 1e-30):
                        dm = solver.driver.get_npdm(
                            states[ket],
                            bra=states[bra],
                            pdm_type=rank,
                            site_type=2,
                            cutoff=cutoff,
                        )
                        error = float(np.max(np.abs(dm - expected)))
                        if name == "qr":
                            assert error < 1e-12
                        result = {
                            "pair": (bra, ket),
                            "rank": rank,
                            "gauge": name,
                            "cutoff": cutoff,
                            "error": error,
                        }
                        print("NPDM ORACLE", result, flush=True)
                        results.append(result)
        with (DIRECTORY / "rdm_precision_verified.json").open("w") as handle:
            json.dump(results, handle, default=json_value, indent=2)
    finally:
        solver.close()
    return results


if __name__ == "__main__":
    audit()
