"""Check the small native MO audit report using only Python's standard library."""
import json
import math
import sys
from pathlib import Path


def validate_report(report, tolerance=1e-8):
    names = ("h1_hermiticity", "g2_pair_exchange", "g2_hermiticity",
             "active_g2_error", "folded_h1_hermiticity", "active_h1_error")
    errors = {name: report[name] for name in names}
    errors["core_energy_error"] = abs(report["reconstructed_core"] - report["native_core"])
    sample = report["independent_sample"]
    for name in ("h1_error", "g2_error", "g2_pair_exchange", "g2_hermiticity"):
        errors[f"independent_sample.{name}"] = sample[name]
    raw = report["raw_active_block_errors"]
    if not raw:
        raise ValueError("Missing independent native active-block comparisons")
    errors.update({f"raw_active.{key}": value for key, value in raw.items()})
    for name, value in errors.items():
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0 or value >= tolerance):
            raise ValueError(f"Native MO integral audit failed: {name}={value}, limit={tolerance}")
    return max(errors.values())


if __name__ == "__main__":
    report = json.loads(Path(sys.argv[1]).read_text())
    print("BAGEL_MO_AUDIT_PASSED", validate_report(report), flush=True)
