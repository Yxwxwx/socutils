#!/usr/bin/env python
"""Summarize structured results from the six-state halogen benchmark."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ELEMENTS = ("F", "Cl", "Br", "I", "At")
METHODS = ("casscf-superci", "dmrg-superci", "dmrg-supercipt")
METHOD_LABELS = {
    "casscf-superci": "CASSCF + Super-CI",
    "dmrg-superci": "DMRG-SCF + Super-CI",
    "dmrg-supercipt": "DMRG-SCF + Super-CIPT",
}


def _load(path):
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def _format(value, kind="float"):
    if value is None:
        return "--"
    if kind == "energy":
        return "%.12f" % value
    if kind == "scientific":
        return "%.3e" % value
    return str(value)


def summarize(results_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    state_rows = []
    convergence_rows = []
    loaded = {}
    for element in ELEMENTS:
        loaded[element] = {}
        for method in METHODS:
            data = _load(results_dir / element / (method + ".json"))
            loaded[element][method] = data
            result = (data or {}).get("result", {})
            scf = (data or {}).get("scf", {})
            roots = result.get("root_energies") or []
            row = {
                "element": element,
                "method": method,
                "status": (data or {}).get("status", "missing"),
                "converged": result.get("converged"),
                "total_energy_eh": result.get("total_energy"),
                "cas_energy_eh": result.get("cas_energy"),
                "final_gradient_norm": result.get("final_gradient_norm"),
                "macroiterations": result.get("macroiterations"),
                "orbital_updates": result.get("orbital_updates"),
                "wall_seconds": result.get("wall_seconds"),
                "scf_energy_eh": scf.get("energy"),
                "scf_iterations": scf.get("iterations"),
                "naux": scf.get("naux"),
                "nspinor": scf.get("nspinor"),
            }
            rows.append(row)
            for state, energy in enumerate(roots):
                state_rows.append(
                    {
                        "element": element,
                        "method": method,
                        "state": state,
                        "energy_eh": energy,
                    }
                )
            for entry in (data or {}).get("macro_history", []):
                convergence_rows.append(
                    {
                        "element": element,
                        "method": method,
                        "macro_iteration": entry.get("macro_iteration"),
                        "total_energy_eh": entry.get("total_energy"),
                        "energy_change_eh": entry.get("energy_change"),
                        "orbital_gradient_norm": entry.get("orbital_gradient_norm"),
                        "converged": entry.get("converged"),
                    }
                )

    reference = {
        row["element"]: row["total_energy_eh"]
        for row in rows
        if row["method"] == "casscf-superci" and row["status"] == "ok"
    }
    for row in rows:
        ref = reference.get(row["element"])
        energy = row["total_energy_eh"]
        row["delta_vs_casscf_superci_eh"] = (
            None if ref is None or energy is None else energy - ref
        )

    def write_csv(name, records):
        path = output_dir / name
        if not records:
            path.write_text("", encoding="utf-8")
            return
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)

    write_csv("summary.csv", rows)
    write_csv("states.csv", state_rows)
    write_csv("convergence.csv", convergence_rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"summary": rows, "states": state_rows},
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    lines = [
        "# Six-state halogen benchmark results",
        "",
        "`macroiterations` counts energy/gradient evaluations, including the "
        "initial and converged evaluations; `updates` counts applied orbital steps.",
        "",
        "| element | method | status | E (Eh) | ΔE vs exact Super-CI (Eh) | final |g| | macroiterations | updates | wall (s) |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {element} | {method} | {status} | {energy} | {delta} | {gradient} | {macro} | {updates} | {wall} |".format(
                element=row["element"],
                method=METHOD_LABELS[row["method"]],
                status=row["status"],
                energy=_format(row["total_energy_eh"], "energy"),
                delta=_format(row["delta_vs_casscf_superci_eh"], "scientific"),
                gradient=_format(row["final_gradient_norm"], "scientific"),
                macro=_format(row["macroiterations"]),
                updates=_format(row["orbital_updates"]),
                wall=_format(
                    None if row["wall_seconds"] is None else round(row["wall_seconds"], 1)
                ),
            )
        )

    lines.extend(
        [
            "",
            "## Final six-state energies",
            "",
            "These are total root energies at each method's final orbitals.",
            "",
            "| element | method | state 1 | state 2 | state 3 | state 4 | state 5 | state 6 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for element in ELEMENTS:
        for method in METHODS:
            data = loaded[element][method]
            roots = ((data or {}).get("result") or {}).get("root_energies") or []
            formatted = [_format(value, "energy") for value in roots]
            formatted.extend(["--"] * (6 - len(formatted)))
            lines.append(
                "| %s | %s | %s |" % (
                    element,
                    METHOD_LABELS[method],
                    " | ".join(formatted[:6]),
                )
            )
    lines.append("")
    (output_dir / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote %s" % (output_dir / "RESULTS.md"))


def main(argv=None):
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=root / "results")
    parser.add_argument("--output-dir", type=Path, default=root)
    args = parser.parse_args(argv)
    summarize(args.results_dir.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
