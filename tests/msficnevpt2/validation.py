# SPDX-License-Identifier: GPL-3.0-or-later
"""Fixed reference rotations and metric scan on the completed native C data."""

import json
import pickle
import resource
import time

import numpy as np
from socutils.mrpt import solve_msfic

from .carbon import json_value
from .carbon_reference import DIRECTORY

CONVERSION = 219474.63137
SEEDS = (12, 91, 407)


def validate(directory=DIRECTORY):
    start = time.perf_counter()
    with (directory / "prepared.pkl").open("rb") as handle:
        prepared, reference_audit = pickle.load(handle)
    rotations = {
        "native": np.eye(5),
        "phases": np.diag(np.exp(1j * np.array([-0.7, 0.2, 0.9, -1.3, 0.4]))),
        "permutation": np.eye(5)[:, [2, 4, 0, 1, 3]],
    }
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        rotations[f"U5_seed_{seed}"] = np.linalg.qr(
            rng.normal(size=(5, 5)) + 1j * rng.normal(size=(5, 5))
        )[0]
    results, baseline = {}, {}
    for name, rotation in rotations.items():
        changed = prepared if name == "native" else prepared.rotated(rotation)
        for shift in (0.2, 0.0):
            for ansatz in ("ss_sr", "ms_mr"):
                result = solve_msfic(changed, ansatz=ansatz, shift=shift)
                key = f"{name}/{ansatz}/eta_{shift}"
                if name == "native":
                    baseline[ansatz, shift] = result
                base = baseline[ansatz, shift]
                defect = float(
                    np.max(
                        np.abs(result.heff - rotation.conj().T @ base.heff @ rotation)
                    )
                )
                spectrum_defect = float(np.max(np.abs(result.energies - base.energies)))
                if ansatz == "ms_mr" or name in ("native", "phases", "permutation"):
                    assert defect < 1e-10 and spectrum_defect < 1e-10
                assert result.spread * CONVERSION <= 0.02 or ansatz == "ss_sr"
                results[key] = {
                    **result.__dict__,
                    "rotation": rotation,
                    "covariance_defect": defect,
                    "spectrum_defect": spectrum_defect,
                    "spread_cm_inverse": result.spread * CONVERSION,
                }
                print(
                    key,
                    "spread/cm-1",
                    result.spread * CONVERSION,
                    "covariance/Eh",
                    defect,
                    flush=True,
                )
    metrics = {}
    for cutoff in (1e-10, 1e-11, 1e-12, 1e-13):
        for ansatz in ("ss_sr", "ms_mr"):
            key = f"{ansatz}/{cutoff}"
            try:
                result = solve_msfic(
                    prepared,
                    ansatz=ansatz,
                    shift=0.2,
                    metric_atol=cutoff,
                    metric_rcond=cutoff,
                )
            except FloatingPointError as error:
                metrics[key] = {"status": "rejected", "error": str(error)}
                print("METRIC", key, "REJECTED", error, flush=True)
                continue
            base = baseline[ansatz, 0.2]
            change = float(np.max(np.abs(result.energies - base.energies)))
            metrics[key] = {
                **result.__dict__,
                "status": "passed",
                "spectrum_change": change,
                "spread_cm_inverse": result.spread * CONVERSION,
            }
            if ansatz == "ms_mr":
                assert abs(result.spread - base.spread) * CONVERSION <= 0.002
            print(
                "METRIC",
                key,
                "spread/cm-1",
                result.spread * CONVERSION,
                "spectrum change/Eh",
                change,
                flush=True,
            )
    report = {
        "fingerprint": prepared.diagnostics["fingerprint"],
        "reference_audit": reference_audit,
        "rotations": results,
        "metrics": metrics,
        "wall_seconds": time.perf_counter() - start,
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
    }
    with (directory / "validation.json").open("w") as handle:
        json.dump(report, handle, default=json_value, indent=2)
    return report


if __name__ == "__main__":
    validate()
