#!/usr/bin/env python
"""Summarize the Kramers-restricted six-state halogen Super-CI results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ELEMENTS = ("F", "Cl", "Br", "I", "At")
METHODS = ("exact-superci", "dmrg-superci")
METHOD_LABELS = {
    "exact-superci": "Exact-FCI KR-SCF + Super-CI",
    "dmrg-superci": "DMRG KR-SCF + Super-CI",
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


def _maximum(values):
    values = [value for value in values if value is not None]
    return max(values) if values else None


def _trajectory_error(exact, dmrg, key):
    exact_history = (exact or {}).get("macro_history", [])
    dmrg_history = (dmrg or {}).get("macro_history", [])
    if not exact_history or len(exact_history) != len(dmrg_history):
        return None
    return max(
        abs(drow[key] - erow[key])
        for erow, drow in zip(exact_history, dmrg_history)
    )


def summarize(results_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    state_rows = []
    convergence_rows = []
    raw_loaded = {
        element: {
            method: _load(results_dir / element / (method + ".json"))
            for method in METHODS
        }
        for element in ELEMENTS
    }
    protocol_versions = [
        data.get("protocol", {}).get("version")
        for methods in raw_loaded.values()
        for data in methods.values()
        if data is not None
        and isinstance(data.get("protocol", {}).get("version"), int)
    ]
    current_protocol_version = (
        max(protocol_versions) if protocol_versions else None
    )
    loaded = {}
    for element in ELEMENTS:
        loaded[element] = {}
        for method in METHODS:
            raw_data = raw_loaded[element][method]
            protocol_version = (raw_data or {}).get("protocol", {}).get(
                "version"
            )
            stale = (
                raw_data is not None
                and current_protocol_version is not None
                and protocol_version != current_protocol_version
            )
            data = None if stale else raw_data
            loaded[element][method] = data
            result = (data or {}).get("result", {})
            scf = (data or {}).get("scf", {})
            kramers = (data or {}).get("kramers", {})
            solver = (data or {}).get("solver", {})
            solver_kr = solver.get("kramers_diagnostics") or {}
            solver_orbitals = solver.get("kramers_orbital_diagnostics") or {}
            state_residual = kramers.get("state_average_rdm_residual") or {}
            pair_diagnostics = solver_kr.get("pairs") or []
            roots = result.get("root_energies") or []
            row = {
                "element": element,
                "method": method,
                "protocol_version": protocol_version,
                "status": (
                    "stale"
                    if stale
                    else (data or {}).get("status", "missing")
                ),
                "converged": result.get("converged"),
                "total_energy_eh": result.get("total_energy"),
                "cas_energy_eh": result.get("cas_energy"),
                "final_gradient_norm": result.get("final_gradient_norm"),
                "macroiterations": result.get("macroiterations"),
                "orbital_updates": result.get("orbital_updates"),
                "wall_seconds": result.get("wall_seconds"),
                "state_average_tr_residual": _maximum(
                    [state_residual.get("dm1"), state_residual.get("dm2")]
                ),
                "active_orbital_closure_error": (
                    (kramers.get("active_orbital_diagnostics") or {}).get(
                        "subspace_closure_error"
                    )
                ),
                "root_pair_residual": solver_kr.get("raw_ensemble_residual"),
                "root_orthogonality_error": solver_kr.get(
                    "root_orthogonality_error"
                ),
                "projected_hamiltonian_error_eh": solver_kr.get(
                    "projected_hamiltonian_error"
                ),
                "maximum_pair_splitting_eh": _maximum(
                    [pair.get("energy_splitting") for pair in pair_diagnostics]
                ),
                "solver_orbital_closure_error": solver_orbitals.get(
                    "subspace_closure_error"
                ),
                "scf_energy_eh": scf.get("energy"),
                "scf_iterations": scf.get("iterations"),
                "naux": scf.get("naux"),
                "nspinor": scf.get("nspinor"),
            }
            rows.append(row)
            for state, energy in enumerate(roots, start=1):
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
                        "orbital_gradient_norm": entry.get(
                            "orbital_gradient_norm"
                        ),
                        "converged": entry.get("converged"),
                    }
                )

    by_key = {(row["element"], row["method"]): row for row in rows}
    for element in ELEMENTS:
        exact_data = loaded[element]["exact-superci"]
        dmrg_data = loaded[element]["dmrg-superci"]
        exact_result = (exact_data or {}).get("result", {})
        dmrg_result = (dmrg_data or {}).get("result", {})
        exact_energy = exact_result.get("total_energy")
        exact_roots = exact_result.get("root_energies") or []
        dmrg_roots = dmrg_result.get("root_energies") or []
        exact_occ = exact_result.get("natural_occupations") or []
        dmrg_occ = dmrg_result.get("natural_occupations") or []
        for method in METHODS:
            row = by_key[(element, method)]
            energy = row["total_energy_eh"]
            roots = (
                exact_roots
                if method == "exact-superci"
                else dmrg_roots
            )
            occupations = (
                exact_occ
                if method == "exact-superci"
                else dmrg_occ
            )
            row["delta_vs_exact_eh"] = (
                None
                if exact_energy is None or energy is None
                else energy - exact_energy
            )
            row["max_root_delta_vs_exact_eh"] = (
                None
                if len(roots) != len(exact_roots) or not roots
                else max(abs(value - ref) for value, ref in zip(roots, exact_roots))
            )
            row["max_occupation_delta_vs_exact"] = (
                None
                if len(occupations) != len(exact_occ) or not occupations
                else max(
                    abs(value - ref)
                    for value, ref in zip(occupations, exact_occ)
                )
            )
            row["trajectory_energy_error_eh"] = (
                0.0
                if method == "exact-superci" and exact_data is not None
                else _trajectory_error(exact_data, dmrg_data, "total_energy")
            )
            row["trajectory_gradient_error"] = (
                0.0
                if method == "exact-superci" and exact_data is not None
                else _trajectory_error(
                    exact_data, dmrg_data, "orbital_gradient_norm"
                )
            )

    def write_csv(name, records):
        path = output_dir / name
        if not records:
            path.write_text("", encoding="utf-8")
            return
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(records[0]),
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(records)

    write_csv("summary.csv", rows)
    write_csv("states.csv", state_rows)
    write_csv("convergence.csv", convergence_rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "protocol_version": current_protocol_version,
                "summary": rows,
                "states": state_rows,
                "convergence": convergence_rows,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    lines = [
        "# Kramers-restricted six-state halogen Super-CI results",
        "",
        "Both methods use the same KR-X2C reference and Kramers-restricted",
        "orbital optimization. `macroiterations` includes the initial and final",
        "energy/gradient evaluations; `updates` counts applied orbital steps.",
        "Only results from the newest available protocol version (%s) are included."
        % current_protocol_version,
        "",
        "| element | method | status | E (Eh) | ΔE vs exact (Eh) | max root Δ (Eh) | final |g| | macroiterations | updates | state-average TR residual | wall (s) |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {element} | {method} | {status} | {energy} | {delta} | {root_delta} | {gradient} | {macro} | {updates} | {tr} | {wall} |".format(
                element=row["element"],
                method=METHOD_LABELS[row["method"]],
                status=row["status"],
                energy=_format(row["total_energy_eh"], "energy"),
                delta=_format(row["delta_vs_exact_eh"], "scientific"),
                root_delta=_format(
                    row["max_root_delta_vs_exact_eh"], "scientific"
                ),
                gradient=_format(row["final_gradient_norm"], "scientific"),
                macro=_format(row["macroiterations"]),
                updates=_format(row["orbital_updates"]),
                tr=_format(row["state_average_tr_residual"], "scientific"),
                wall=_format(
                    None
                    if row["wall_seconds"] is None
                    else round(row["wall_seconds"], 1)
                ),
            )
        )

    lines.extend(
        [
            "",
            "## DMRG Kramers adapter diagnostics",
            "",
            "| element | root/manifold residual | root orthogonality | projected-H residual (Eh) | max pair splitting (Eh) | active-orbital closure |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for element in ELEMENTS:
        row = by_key[(element, "dmrg-superci")]
        lines.append(
            "| {element} | {pair} | {overlap} | {hamiltonian} | {splitting} | {closure} |".format(
                element=element,
                pair=_format(row["root_pair_residual"], "scientific"),
                overlap=_format(
                    row["root_orthogonality_error"], "scientific"
                ),
                hamiltonian=_format(
                    row["projected_hamiltonian_error_eh"], "scientific"
                ),
                splitting=_format(
                    row["maximum_pair_splitting_eh"], "scientific"
                ),
                closure=_format(
                    row["solver_orbital_closure_error"], "scientific"
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
                "| %s | %s | %s |"
                % (
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
    parser.add_argument(
        "--results-dir", type=Path, default=root / "kramers" / "results"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=root / "kramers"
    )
    args = parser.parse_args(argv)
    summarize(args.results_dir.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
