# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent M32/M128 SA15 DMRG-SCF/RDM/PT runs, compared with M64."""

import json

import numpy as np

from .carbon import json_value, run
from .carbon_reference import DIRECTORY


def scan():
    with (DIRECTORY / "results.json").open() as handle:
        baseline = json.load(handle)
    results = {}
    for bond_dim in (32, 128):
        directory = DIRECTORY.parent / f"carbon_M{bond_dim}_run"
        data = run(directory, bond_dim=bond_dim)
        comparisons = {}
        for key, result in data["results"].items():
            energies = np.asarray(result["energies"])
            reference = np.asarray(baseline["results"][key]["energies"])
            change = float(np.max(np.abs(energies - reference)))
            spread_change = (
                abs(float(np.ptp(energies) - np.ptp(reference))) * 219474.63137
            )
            comparisons[key] = {
                "spectrum_change": change,
                "spread_change_cm_inverse": spread_change,
            }
            if key.startswith("ms_mr"):
                assert (
                    spread_change <= 0.002 and np.ptp(energies) * 219474.63137 <= 0.02
                )
            print("M CONVERGENCE", bond_dim, key, comparisons[key], flush=True)
        results[str(bond_dim)] = {"data": data, "comparison": comparisons}
        with (DIRECTORY / "convergence.json").open("w") as handle:
            json.dump(results, handle, default=json_value, indent=2)
    return results


if __name__ == "__main__":
    scan()
