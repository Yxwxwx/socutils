# SPDX-License-Identifier: GPL-3.0-or-later
"""Six-root F energies and measurement-only whole-class response convergence.

Reuse the existing no-CD/no-KR SA6 reference. Two independent processes use
16 threads each; never keep two live Block2 frames in the same process.
"""
import argparse
import json
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


def summarize(output):
    roots = {}
    for root in range(NROOTS):
        directory = output / f"root_{root}"
        if not all((directory / f"{stage}.json").exists() for stage in ("full", "hybrid")):
            continue
        full, hybrid = (json.loads((directory / f"{stage}.json").read_text())
                        for stage in ("full", "hybrid"))
        results = {f"{stage}_{method}": values for stage, data in (("full", full), ("hybrid", hybrid))
                   for method, values in data["results"].items()}
        measured = {path.stem.removeprefix("response_"): json.loads(path.read_text())
                    for path in directory.glob("response_*.json")}
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
    save_json(output / "summary.json", dict(roots=roots,
              all_six_root_energies_present=len(roots) == NROOTS,
              all_six_final_global_residuals_verified=verified,
              response_residual_status=residual_status,
              representative_M1500_to_M2000_stability=bond_stability))
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
    args = parser.parse_args()
    lib.num_threads(THREADS)
    if args.summary_only:
        summarize(args.output)
    elif args.validate_root is not None:
        validation(args.reference, args.output, args.validate_root)
    else:
        run(args.reference, args.output)
