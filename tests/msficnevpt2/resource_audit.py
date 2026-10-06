# SPDX-License-Identifier: GPL-3.0-or-later
"""Record actual native bond sizes, input protocol and installed versions."""

import importlib.metadata
import json
import platform

from .carbon import json_value, mps_bonds
from .carbon_reference import DIRECTORY, canonicalize_npdm_reference, reference


def audit():
    evidence = {
        "host": platform.node(),
        "python": platform.python_version(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "pyscf", "block2", "pytblis")
        },
        "references": {},
    }
    for bond_dim in (32, 64, 128):
        directory = (
            DIRECTORY
            if bond_dim == 64
            else DIRECTORY.parent / f"carbon_M{bond_dim}_run"
        )
        mc, solver = reference(directory, bond_dim=bond_dim)
        try:
            original = mps_bonds(solver)
            canonicalize_npdm_reference(mc)
            evidence["references"][str(bond_dim)] = {
                "requested": bond_dim,
                "original_actual_bonds": original,
                "qr_actual_bonds": mps_bonds(solver),
                "hamiltonian_sha256": solver.checkpoint_hamiltonian[
                    "hamiltonian_sha256"
                ],
                "nuclear_model": mc.mol.nucmod,
                "basis": mc.mol._basis,
                "spatial_ao_count": mc.mol.nao_nr(),
                "spinor_ao_count": mc.mol.nao_2c(),
                "no_cd_df": getattr(mc._scf, "with_df", None) is None,
                "no_kr": solver.kramers_adapter is None,
            }
        finally:
            solver.close()
    with (DIRECTORY / "resource_audit.json").open("w") as handle:
        json.dump(evidence, handle, default=json_value, indent=2)
    print(
        json.dumps(
            {
                **evidence,
                "references": {
                    m: {k: v for k, v in r.items() if k != "basis"}
                    for m, r in evidence["references"].items()
                },
            },
            default=json_value,
            indent=2,
        ),
        flush=True,
    )
    return evidence


if __name__ == "__main__":
    audit()
