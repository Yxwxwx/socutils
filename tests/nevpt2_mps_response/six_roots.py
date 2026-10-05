# SPDX-License-Identifier: GPL-3.0-or-later
"""Six-root F energies and measurement-only whole-class response convergence.

Reuse the existing no-CD/no-KR SA6 reference. Two independent processes use
16 threads each; never keep two live Block2 frames in the same process.
"""
import argparse
import json
import math
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pyscf import lib
from socutils.mrpt import nevpt2_mps_response as response
from socutils.mrpt import nevpt2_utils as u
from socutils.mrpt.nevpt2_eris import wick_eris_from_mc

from .f_atom import NROOTS, THREADS, audited_response, restore, save_json

HERE = Path(__file__).resolve().parent
PROFILES = {
    "default": {},
    "sweeps16": dict(n_sweeps=16, tol=0.),
    "linear_tight": dict(n_sweeps=16, tol=0., linear_threshold=1e-22),
    "noise_zero": dict(n_sweeps=16, tol=0., linear_threshold=1e-22, noise=0.),
    "cutoff_tight": dict(n_sweeps=16, tol=0., linear_threshold=1e-22, noise=0., cutoff=1e-24),
}
for bond in (1500, 2000):
    PROFILES[f"M{bond}"] = dict(PROFILES["cutoff_tight"], max_bond_dimension=bond)


def run_process(cmd, logfile, root, stage):
    # A shared TMPDIR would make separate Block2 frames overwrite each other's
    # MPS/NPDM scratch. The persistent checkpoint is read-only and shared.
    with tempfile.TemporaryDirectory(prefix=f"F_root{root}_{stage}_",
                                     dir=os.environ["TMPDIR"]) as scratch:
        env = dict(os.environ, TMPDIR=scratch, PYSCF_TMPDIR=scratch)
        with logfile.open("a") as stream:
            subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT,
                           env=env, check=True)


def external_results(logs, metadata):
    """Only complete full-UC records, never pilots or partial class energies."""
    points = []
    for path in sorted(logs.glob("root*_external_*.out")):
        controls, implementation, inputs, resources, records = None, None, None, None, []
        with path.open() as stream:
            for line in stream:
                if not line.endswith("\n"):
                    continue  # The live process may still be writing this record.
                if line.startswith("F_UC_POINT "):
                    controls = json.loads(line.split(" ", 1)[1])["options"]
                elif line.startswith("F_UC_IMPLEMENTATION "):
                    implementation = json.loads(line.split(" ", 1)[1])
                elif line.startswith("F_UC_PREPARED_INPUT "):
                    inputs = json.loads(line.split(" ", 1)[1])
                elif line.startswith("F_UC_RESOURCES "):
                    resources = json.loads(line.split(" ", 1)[1])
                elif line.startswith("F_UC_RESULT "):
                    result = json.loads(line.split(" ", 1)[1])
                    root = result["root"]
                    if (type(root) is not int or not 0 <= root < NROOTS
                            or not path.name.startswith(f"root{root}_")
                            or result["fingerprint"] != metadata["fingerprint"]
                            or result["response_mode"] != "external_tuples"):
                        raise ValueError(f"incompatible external UC result: {path}")
                    if (controls is None or controls["max_bond_dimension"] != result["bond"]
                            or controls["n_sweeps"] != result["sweeps"]
                            or set(result["classes"]) != set(u.SUBSPACE_ORDER)
                            or not all(math.isfinite(e) for e in
                                       [result["e_corr"], result["e_tot"], *result["classes"].values()])
                            or abs(sum(result["classes"].values()) - result["e_corr"]) > 1e-10
                            or abs(result["e_tot"] - result["e_corr"]
                                   - metadata["root_energies"][root]) > 1e-8):
                        raise ValueError(f"inconsistent external UC energy/controls: {path}")
                    # Keep per-class/worst-tuple certificates. All tuple details
                    # remain in the original log rather than duplicated here.
                    effective = [data.get("controls", controls) for data in
                                 result["diagnostics"]["classes"].values()]
                    if any(value != effective[0] for value in effective):
                        raise ValueError(f"inconsistent effective UC controls: {path}")
                    for data in result["diagnostics"]["classes"].values():
                        data.pop("tuples", None)
                    records.append(dict(result, controls=controls.copy(), log=str(path.resolve()),
                                        effective_controls=effective[0].copy(),
                                        implementation=implementation, prepared_inputs=inputs))
        for record in records:
            record["process_resources"] = resources  # Whole process, not per-point peak RSS.
        points.extend(records)
    return points


def summarize(output, *, baseline=None, reference=None, uc_logs=None):
    metadata = json.loads((reference / "mcscf.json").read_text()) if uc_logs is not None else None
    baseline = output if baseline is None else baseline
    roots = {}
    fingerprint = metadata["fingerprint"] if metadata is not None else None
    for root in range(NROOTS):
        directory = baseline / f"root_{root}"
        if not all((directory / f"{stage}.json").exists() for stage in ("full", "hybrid")):
            continue
        full, hybrid = (json.loads((directory / f"{stage}.json").read_text())
                        for stage in ("full", "hybrid"))
        fingerprint = full["fingerprint"] if fingerprint is None else fingerprint
        if any(data["fingerprint"] != fingerprint or data["root"] != root for data in (full, hybrid)):
            raise ValueError(f"baseline reference/root mismatch: {directory}")
        results = {f"{stage}_{method}": values for stage, data in (("full", full), ("hybrid", hybrid))
                   for method, values in data["results"].items()}
        measured = {path.stem.removeprefix("response_"): json.loads(path.read_text())
                    for path in directory.glob("response_*.json")}
        if any(data["fingerprint"] != fingerprint or data["root"] != root for data in measured.values()):
            raise ValueError(f"response baseline reference/root mismatch: {directory}")
        roots[str(root)] = dict(reference_energy=full["results"]["SC"]["reference_energy"],
                                results=results, convergence=measured)
    verified = len(roots) == NROOTS and all(
        r["convergence"].get("M2000", {}).get("global_residual_verified", False)
        for r in roots.values())
    measured_all = len(roots) == NROOTS and all("M2000" in r["convergence"] for r in roots.values())
    residual_status = "ok" if verified else "warning" if measured_all else "pending"
    bond_stability = {}
    for root in ("0", "4"):
        cases = roots.get(root, {}).get("convergence", {})
        if "M1500" in cases and "M2000" in cases:
            differences = {key: cases["M2000"]["response_energies"][key]
                           - cases["M1500"]["response_energies"][key] for key in ("i", "r")}
            bond_stability[root] = dict(class_energy_differences=differences,
                                         total_energy_difference=sum(differences.values()))
    summary = dict(roots=roots, fingerprint=fingerprint,
              all_six_root_energies_present=len(roots) == NROOTS,
              all_six_final_global_residuals_verified=verified,
              response_residual_status=residual_status,
              representative_M1500_to_M2000_stability=bond_stability)
    if uc_logs is not None:
        points = external_results(uc_logs, metadata)
        groups = {}
        for point in points:
            signature = json.dumps(point["effective_controls"], sort_keys=True)
            group = groups.setdefault(signature, dict(controls=point["effective_controls"], roots={}))
            if str(point["root"]) in group["roots"]:
                raise ValueError(f"duplicate root/control UC point: {point['log']}")
            group["roots"][str(point["root"])] = point
        for group in groups.values():
            group["multiplet_total_energy_spreads"] = {}
            for label, members in (("roots_0_3", range(4)), ("roots_4_5", range(4, 6))):
                values = [group["roots"][str(root)]["e_tot"] for root in members
                          if str(root) in group["roots"]]
                group["multiplet_total_energy_spreads"][label] = (
                    max(values) - min(values) if len(values) == len(members) else None)
            group["all_six_energies_present"] = len(group["roots"]) == NROOTS
            group["all_six_residuals_certified"] = (group["all_six_energies_present"]
                and all(point["converged"] for point in group["roots"].values()))
        summary["external_uc"] = dict(groups=list(groups.values()),
            energy_status="complete" if any(g["all_six_energies_present"] for g in groups.values()) else "pending",
            note="No process-liveness inference from logs; missing complete records are pending.")
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "summary.json", summary)
    # Same State / weight / E layout as the production inputs; totals, not E2.
    lines = []
    for method in ("MCSCF", "full_SC", "full_FIC", "hybrid_SC", "hybrid_FIC"):
        lines.append(f"{method} energy for each state")
        for root, data in roots.items():
            energy = data["reference_energy"] if method == "MCSCF" else data["results"][method]["e_tot"]
            lines.append(f"  State {root} weight {1/NROOTS:.7g}  E = {energy:.14f}")
    for method in ("SC", "FIC"):
        lines.append(f"{method}+UC(i,r) M2000 validation energy for each state")
        for root, data in roots.items():
            precision = data["convergence"].get("M2000")
            if precision is not None:
                energy = precision["hybrid_totals"][method]["e_tot"]
                verified = precision["global_residual_verified"]
                lines.append(f"  State {root} weight {1/NROOTS:.7g}  E = {energy:.14f}  "
                             f"residual_status={'ok' if verified else 'warning'}")
    for group in summary.get("external_uc", {}).get("groups", []):
        controls = group["controls"]
        lines.append(f"UC external_tuples M{controls['max_bond_dimension']} "
                     f"S{controls['n_sweeps']} energy for each state")
        for root, point in sorted(group["roots"].items(), key=lambda item: int(item[0])):
            lines.append(f"  State {root} weight {metadata['weights'][int(root)]:.7g}  "
                         f"E = {point['e_tot']:.14f}  E2 = {point['e_corr']:.14f}  "
                         f"residual_status={'ok' if point['converged'] else 'warning'}")
        lines.append("  Multiplet spreads / Eh: "
                     + json.dumps(group["multiplet_total_energy_spreads"]))
    if uc_logs is not None and summary["external_uc"]["energy_status"] == "pending":
        lines.append("UC external_tuples: pending; no complete six-root energy set yet.")
    (output / "energies.txt").write_text("\n".join(lines) + "\n")


def validation(reference, output, root):
    """The native solver runs each complete class; only measurement uses SGF CI coefficients."""
    directory = output / f"root_{root}"
    full = json.loads((directory / "full.json").read_text())
    mc, solver, fingerprint = restore(reference)
    try:
        assert fingerprint == full["fingerprint"] and full["root"] == root
        pdms = tuple(u._make_rdm(solver, root, rank) for rank in (1, 2, 3))
        full_mc = u._full_integral_mc(mc)
        mo, eps = u.semicanonicalize(full_mc, mc.mo_coeff, pdms[0], root)
        eris = wick_eris_from_mc(full_mc, mo)
        # Each multiplet's first root receives the one-variable-at-a-time
        # sequence. Every root receives default and the final high-M check.
        profiles = PROFILES if root in (0, 4) else {
            name: PROFILES[name] for name in ("default", "M2000")}
        for name, options in profiles.items():
            path = directory / f"response_{name}.json"
            if path.exists():
                saved = json.loads(path.read_text())
                assert saved["fingerprint"] == fingerprint and saved["root"] == root
                assert saved["requested_options"] == options
                continue
            solver.driver.bw.b.Random.rand_seed(1234)
            start = time.perf_counter()
            print(f"F_CONVERGENCE_START root={root} case={name} options={options}", flush=True)
            energies, norms, gaps, diagnostics, timings = audited_response(
                response.evaluate_mps_response, mc, eris, pdms,
                eps[:eris.ncore], eps[eris.nocc:], root=root, options=options,
                contraction_backend="pytblis")
            totals = {}
            for method, values in full["results"].items():
                e2 = sum(value for key, value in values["sub_eners"].items()
                         if key not in ("i", "r")) + sum(energies.values())
                totals[method] = dict(e_corr=e2, e_tot=values["reference_energy"] + e2,
                                      delta_from_strict=e2 - values["e_corr"])
            data = dict(root=root, fingerprint=fingerprint, case=name,
                        requested_options=options, requested_rdm_ranks=[1, 2, 3], random_seed=1234,
                        reference_energy=float(mc.e_states[root]),
                        response_energies=energies, response_norms=norms,
                        diagnostics=diagnostics, hybrid_totals=totals, timings=timings,
                        global_residual_verified=all(d["global_residual_verified"]
                                                     for d in diagnostics.values()),
                        wall_seconds=time.perf_counter() - start,
                        peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20)
            save_json(path, data)
            print(f"F_CONVERGENCE_DONE root={root} case={name} "
                  f"global_verified={data['global_residual_verified']}", flush=True)
    finally:
        solver.close()


def energies(reference, output, root):
    directory = output / f"root_{root}"
    directory.mkdir(parents=True, exist_ok=True)
    fingerprint = json.loads((reference / "mcscf.json").read_text())["fingerprint"]
    for stage in ("full", "hybrid"):
        path = directory / f"{stage}.json"
        # Keep original root-0 evidence intact and avoid recomputing its 4-RDM.
        if root == 0 and not path.exists():
            previous = json.loads((reference / f"{stage}.json").read_text())
            assert previous["root"] == root and previous["fingerprint"] == fingerprint
            shutil.copy2(reference / f"{stage}.json", path)
        if path.exists():
            saved = json.loads(path.read_text())
            assert saved["root"] == root and saved["fingerprint"] == fingerprint
            assert saved["requested_rdm_ranks"] == ([1, 2, 3, 4] if stage == "full" else [1, 2, 3])
            continue
        cmd = [sys.executable, "-u", "-m", "tests.nevpt2_mps_response.f_atom", stage,
               "--directory", str(reference), "--output-directory", str(directory), "--root", str(root)]
        if stage == "hybrid":
            cmd.append("--skip-residual-audit")
        print(f"F_SIX_ROOT_START root={root} stage={stage}", flush=True)
        run_process(cmd, directory / f"{stage}.out", root, stage)
        print(f"F_SIX_ROOT_DONE root={root} stage={stage}", flush=True)
    a, b = (json.loads((directory / f"{stage}.json").read_text()) for stage in ("full", "hybrid"))
    for method in ("SC", "FIC"):
        av, bv = a["results"][method], b["results"][method]
        assert abs(av["reference_energy"] - bv["reference_energy"]) < 1e-10
        assert max(abs(value - bv["sub_eners"][key]) for key, value in av["sub_eners"].items()
                   if key not in ("i", "r")) < 1e-10


def run(reference, output):
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((reference / "mcscf.json").read_text())
    assert metadata["converged"] and metadata["nroots"] == NROOTS
    assert not metadata["kr"] and not metadata["cd"]
    assert metadata["optimizer"] == "second_order"
    failures = {}
    for stage in ("energies", "validation"):
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {}
            for root in range(NROOTS):
                if stage == "energies":
                    task = pool.submit(energies, reference, output, root)
                elif str(root) not in failures:
                    directory = output / f"root_{root}"
                    def validate_task(root=root, directory=directory):
                        cmd = [sys.executable, "-u", "-m", "tests.nevpt2_mps_response.six_roots",
                               "--reference", str(reference), "--output", str(output),
                               "--validate-root", str(root)]
                        run_process(cmd, directory / "convergence.out", root, "validation")
                    task = pool.submit(validate_task)
                else:
                    continue
                futures[task] = root
            for task in as_completed(futures):
                root = futures[task]
                try:
                    task.result()
                except Exception as error:
                    failures[str(root)] = f"{stage}: {error}"
                    print(f"F_SIX_ROOT_FAILED root={root} {error}", flush=True)
                save_json(output / "status.json", dict(stage=stage, failures=failures,
                          fingerprint=metadata["fingerprint"]))
                summarize(output)
    if failures:
        raise RuntimeError(f"six-root calculations failed: {failures}")
    print("F_SIX_ROOT_FINISHED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=HERE / "f_sa6_ground_run")
    parser.add_argument("--output", type=Path, default=HERE / "f_sa6_all_roots")
    parser.add_argument("--validate-root", type=int, choices=range(NROOTS))
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--baseline", type=Path, help="Read existing root JSONs here; write only to --output")
    parser.add_argument("--uc-logs", type=Path, help="Include complete external-tuple F_UC_RESULT records")
    args = parser.parse_args()
    if not args.summary_only and (args.baseline is not None or args.uc_logs is not None):
        parser.error("--baseline and --uc-logs require --summary-only")
    lib.num_threads(THREADS)
    if args.summary_only:
        summarize(args.output, baseline=args.baseline, reference=args.reference, uc_logs=args.uc_logs)
    elif args.validate_root is not None:
        validation(args.reference, args.output, args.validate_root)
    else:
        run(args.reference, args.output)
