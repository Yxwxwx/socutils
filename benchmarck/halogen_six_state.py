#!/usr/bin/env python
"""Run the six-state halogen X2C CASSCF/DMRG-SCF benchmark.

The directory name follows the spelling requested for this benchmark.  Each
worker handles one element/method pair and writes its result atomically, while
the parent mode runs the requested matrix sequentially and keeps a separate
log for every calculation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
from pyscf import __version__ as pyscf_version
from pyscf import gto, scf

from socutils.cd.cd import CD
from socutils.dmrg import DMRGCI
from socutils.mcscf import zmcscf
from socutils.scf.spinor_hf import density_fit as spinor_density_fit


ELEMENTS = ("F", "Cl", "Br", "I", "At")
METHODS = ("casscf-superci", "dmrg-superci", "dmrg-supercipt")
PROTOCOL_VERSION = 1
NROOTS = 6
NCAS = 8
NELECAS = 7
WEIGHTS = np.ones(NROOTS) / NROOTS


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, complex):
        if abs(value.imag) <= 1e-12:
            return float(value.real)
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _write_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temporary, path)


def _basis_slug(basis):
    return "".join(char if char.isalnum() else "-" for char in basis.lower())


def _protocol(args, element, method=None):
    result = {
        "version": PROTOCOL_VERSION,
        "element": element,
        "basis": args.basis,
        "hamiltonian": "spin-orbit X2C with Coulomb-only two-electron terms",
        "initial_reference": "%s- closed-shell X2C SCF" % element,
        "target": "neutral %s" % element,
        "initial_charge": -1,
        "initial_spin": 0,
        "target_charge": 0,
        "target_spin": 1,
        "symmetry": True,
        "ncas": NCAS,
        "nelecas": NELECAS,
        "nroots": NROOTS,
        "weights": WEIGHTS.tolist(),
        "cholesky_tau": args.cholesky_tau,
        "conv_tol": args.conv_tol,
        "conv_tol_grad": args.conv_tol_grad,
        "max_cycle_macro": args.max_cycle_macro,
        "max_stepsize": args.max_stepsize,
        "natorb": False,
        "canonicalize": True,
        "max_memory_mb": args.max_memory,
        "superci_davidson_tol": args.superci_davidson_tol,
        "superci_davidson_max_space": args.superci_davidson_max_space,
        "dmrg": {
            "bond_dimension": args.bond_dimension,
            "ecore_in_mpo": False,
            "n_sweeps": args.dmrg_sweeps,
            "energy_tolerance": args.dmrg_tol,
            "local_squared_residual_threshold": args.dmrg_thrd,
            "davidson_max_iter": args.dmrg_davidson_max_iter,
            "noise": 0.0,
            "random_seed": args.random_seed,
            "n_threads": args.threads,
            "stack_memory_mb": args.dmrg_stack_memory,
            "npdm_site_type": 2,
            "npdm_cutoff": 1e-24,
        },
    }
    if method is not None:
        result["method"] = method
    return result


def _valid_cderi(path, mol):
    if not path.is_file():
        return False
    try:
        naux, npair = CD.cderi_shape(path)
    except Exception:
        return False
    return naux > 0 and npair == mol.nao_nr() * (mol.nao_nr() + 1) // 2


def _make_mean_field(mol, cderi_path, cholesky_tau):
    with_df = CD(mol, tau=cholesky_tau)
    if _valid_cderi(cderi_path, mol):
        with_df._cderi = str(cderi_path)
    else:
        if cderi_path.exists():
            cderi_path.unlink()
        cderi_path.parent.mkdir(parents=True, exist_ok=True)
        with_df._cderi_to_save = str(cderi_path)

    # PySCF's generic DF wrapper does not split a 2c density into spherical
    # alpha/beta blocks.  The socutils wrapper does, while retaining the
    # original pyscf.scf.X2C one-electron Hamiltonian used by the F reference.
    return spinor_density_fit(scf.X2C(mol), with_df=with_df)


def _load_scf_cache(path, protocol, mf):
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            cached_protocol = json.loads(str(data["protocol"].item()))
            if cached_protocol != protocol:
                return None
            mo_coeff = np.array(data["mo_coeff"], copy=True)
            mo_energy = np.array(data["mo_energy"], copy=True)
            mo_occ = np.array(data["mo_occ"], copy=True)
            e_tot = float(data["e_tot"].item())
            history = json.loads(str(data["history"].item()))
            wall_seconds = float(data["wall_seconds"].item())
    except Exception:
        return None
    if mo_coeff.shape != (mf.mol.nao_2c(), mf.mol.nao_2c()):
        return None
    mf.mo_coeff = mo_coeff
    mf.mo_energy = mo_energy
    mf.mo_occ = mo_occ
    mf.e_tot = e_tot
    mf.converged = True
    return {
        "converged": True,
        "energy": e_tot,
        "iterations": len(history),
        "history": history,
        "wall_seconds": wall_seconds,
        "cache_reused": True,
    }


def _run_or_load_scf(args, element, scratch_root):
    mol = gto.M(
        atom="%s 0 0 0" % element,
        basis=args.basis,
        charge=-1,
        spin=0,
        symmetry=True,
        verbose=args.verbose,
        max_memory=args.max_memory,
    )
    element_scratch = scratch_root / _basis_slug(args.basis) / element
    cderi_path = element_scratch / "cderi.h5"
    cache_path = element_scratch / "anion_scf.npz"
    mf = _make_mean_field(mol, cderi_path, args.cholesky_tau)
    mf.conv_tol = args.scf_conv_tol
    mf.max_cycle = args.scf_max_cycle
    scf_protocol = {
        "version": PROTOCOL_VERSION,
        "element": element,
        "basis": args.basis,
        "charge": -1,
        "spin": 0,
        "symmetry": True,
        "cholesky_tau": args.cholesky_tau,
        "scf_conv_tol": args.scf_conv_tol,
        "pyscf_version": pyscf_version,
    }
    cached = _load_scf_cache(cache_path, scf_protocol, mf)
    if cached is not None and _valid_cderi(cderi_path, mol):
        cached.update(
            {
                "nao_nr": mol.nao_nr(),
                "nspinor": mol.nao_2c(),
                "naux": int(mf.with_df.get_naoaux()),
                "cderi": str(cderi_path),
            }
        )
        return mol, mf, cached

    history = []

    def scf_callback(environment):
        if "cycle" not in environment:
            return
        energy = float(environment["e_tot"])
        energy_change = (
            None if not history else energy - history[-1]["energy"]
        )
        history.append(
            {
                "cycle": int(environment["cycle"]),
                "energy": energy,
                "energy_change": energy_change,
                "gradient_norm": (
                    None
                    if environment.get("norm_gorb") is None
                    else float(environment["norm_gorb"])
                ),
                "density_change_norm": (
                    None
                    if environment.get("norm_ddm") is None
                    else float(environment["norm_ddm"])
                ),
            }
        )

    mf.callback = scf_callback
    started = time.perf_counter()
    mf.kernel()
    wall_seconds = time.perf_counter() - started
    if not mf.converged:
        raise RuntimeError("%s- X2C SCF did not converge" % element)
    if not _valid_cderi(cderi_path, mol):
        raise RuntimeError("Cholesky cache was not written correctly")
    _write_npz(
        cache_path,
        protocol=json.dumps(scf_protocol, sort_keys=True),
        mo_coeff=mf.mo_coeff,
        mo_energy=mf.mo_energy,
        mo_occ=mf.mo_occ,
        e_tot=mf.e_tot,
        history=json.dumps(history, sort_keys=True),
        wall_seconds=wall_seconds,
    )
    return mol, mf, {
        "converged": True,
        "energy": float(mf.e_tot),
        "iterations": len(history),
        "history": history,
        "wall_seconds": wall_seconds,
        "cache_reused": False,
        "nao_nr": mol.nao_nr(),
        "nspinor": mol.nao_2c(),
        "naux": int(mf.with_df.get_naoaux()),
        "cderi": str(cderi_path),
    }


def _configure_casscf(args, mol, mf, method, dmrg_scratch):
    mc = zmcscf.CASSCF(mf, ncas=NCAS, nelecas=NELECAS)
    if method.startswith("dmrg-"):
        solver = DMRGCI(mol).init(
            ncas=NCAS,
            nelecas=NELECAS,
            nroots=NROOTS,
            bond_dims=[args.bond_dimension] * args.dmrg_sweeps,
            noises=[0.0] * args.dmrg_sweeps,
            thrds=[args.dmrg_thrd] * args.dmrg_sweeps,
            n_sweeps=args.dmrg_sweeps,
            tol=args.dmrg_tol,
            scratch=dmrg_scratch,
            n_threads=args.threads,
            stack_memory=args.dmrg_stack_memory,
            random_seed=args.random_seed,
            dav_max_iter=args.dmrg_davidson_max_iter,
            npdm_site_type=2,
            npdm_cutoff=1e-24,
        )
        mc.fcisolver = solver
    else:
        mc.fcisolver.conv_tol = args.fci_conv_tol
        mc.fcisolver.max_cycle = args.fci_max_cycle
    mc.state_average_(WEIGHTS)
    mc.natorb = False
    mc.canonicalize_ = True
    mc.max_stepsize = args.max_stepsize
    mc.max_cycle_macro = args.max_cycle_macro
    mc.conv_tol = args.conv_tol
    mc.conv_tol_grad = args.conv_tol_grad
    mc.superci_davidson_tol = args.superci_davidson_tol
    mc.superci_davidson_max_space = args.superci_davidson_max_space
    mc.superci_davidson_strict = True
    mc.verbose = args.verbose
    return mc


def _solver_snapshot(solver):
    return {
        "class": type(solver).__name__,
        "root_energies": getattr(solver, "e_states", None),
        "converged": getattr(solver, "converged", None),
        "dmrg_convergence": getattr(solver, "convergence_info", None),
        "rdm_diagnostics": getattr(solver, "rdm_diagnostics", None),
    }


def _result_matches(path, protocol):
    if not path.is_file():
        return False
    try:
        with path.open(encoding="utf-8") as handle:
            result = json.load(handle)
    except Exception:
        return False
    return result.get("status") == "ok" and result.get("protocol") == protocol


def run_worker(args, element, method):
    result_path = args.results_dir / element / (method + ".json")
    progress_path = args.results_dir / element / (method + ".progress.json")
    protocol = _protocol(args, element, method)
    if not args.force and _result_matches(result_path, protocol):
        print("SKIP %s %s: matching converged result exists" % (element, method))
        return 0

    started = time.perf_counter()
    payload = {
        "status": "running",
        "protocol": protocol,
        "started_unix": time.time(),
    }
    _write_json(progress_path, payload)
    mc = None
    try:
        mol, mf, scf_result = _run_or_load_scf(args, element, args.scratch_dir)
        initial_mo = np.array(mf.mo_coeff, copy=True)
        mol.charge = 0
        mol.spin = 1
        dmrg_scratch = args.scratch_dir / "dmrg" / element / method
        dmrg_scratch.mkdir(parents=True, exist_ok=True)
        mc = _configure_casscf(args, mol, mf, method, dmrg_scratch)

        latest = {}

        def macro_callback(environment):
            latest.clear()
            latest.update(_jsonable(environment))
            _write_json(
                progress_path,
                {
                    "status": "running",
                    "protocol": protocol,
                    "scf": scf_result,
                    "latest_macroiteration": latest,
                    "elapsed_seconds": time.perf_counter() - started,
                },
            )

        if method.endswith("supercipt"):
            mc.supercipt(mo_coeff=initial_mo, callback=macro_callback)
        else:
            mc.superci(mo_coeff=initial_mo, callback=macro_callback)

        history = list(mc.macro_history)
        solver_data = _solver_snapshot(mc.fcisolver)
        macroiterations = len(history)
        orbital_updates = sum(
            "applied_orbital_step_norm" in entry for entry in history
        )
        payload = {
            "status": "ok" if mc.converged else "not_converged",
            "protocol": protocol,
            "scf": scf_result,
            "result": {
                "converged": bool(mc.converged),
                "total_energy": float(np.real(mc.e_tot)),
                "cas_energy": float(np.real(mc.e_cas)),
                "final_gradient_norm": float(mc.final_orbital_gradient_norm),
                "macroiterations": macroiterations,
                "orbital_updates": orbital_updates,
                "root_energies": solver_data["root_energies"],
                "natural_occupations": (
                    history[-1].get("natural_occupations") if history else None
                ),
                "wall_seconds": time.perf_counter() - started,
            },
            "solver": solver_data,
            "macro_history": history,
        }
        _write_json(result_path, payload)
        if progress_path.exists():
            progress_path.unlink()
        print(
            "%s %s: status=%s E=%.15f |g|=%.6e macros=%d updates=%d wall=%.1fs"
            % (
                element,
                method,
                payload["status"],
                payload["result"]["total_energy"],
                payload["result"]["final_gradient_norm"],
                macroiterations,
                orbital_updates,
                payload["result"]["wall_seconds"],
            ),
            flush=True,
        )
        return 0 if mc.converged else 2
    except Exception as error:
        payload = {
            "status": "error",
            "protocol": protocol,
            "elapsed_seconds": time.perf_counter() - started,
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        }
        _write_json(result_path, payload)
        _write_json(progress_path, payload)
        traceback.print_exc()
        return 1
    finally:
        if mc is not None and method.startswith("dmrg-"):
            close = getattr(mc.fcisolver, "close", None)
            if close is not None:
                close()


def run_matrix(args):
    args.logs_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    script = Path(__file__).resolve()
    for element in args.elements:
        for method in args.methods:
            result_path = args.results_dir / element / (method + ".json")
            protocol = _protocol(args, element, method)
            if not args.force and _result_matches(result_path, protocol):
                print("SKIP %s %s" % (element, method), flush=True)
                continue
            command = [
                sys.executable,
                str(script),
                "--worker",
                "--element",
                element,
                "--method",
                method,
                "--basis",
                args.basis,
                "--results-dir",
                str(args.results_dir),
                "--scratch-dir",
                str(args.scratch_dir),
                "--logs-dir",
                str(args.logs_dir),
                "--verbose",
                str(args.verbose),
                "--threads",
                str(args.threads),
                "--max-memory",
                str(args.max_memory),
                "--max-cycle-macro",
                str(args.max_cycle_macro),
                "--conv-tol",
                str(args.conv_tol),
                "--conv-tol-grad",
                str(args.conv_tol_grad),
                "--max-stepsize",
                str(args.max_stepsize),
                "--cholesky-tau",
                str(args.cholesky_tau),
                "--superci-davidson-tol",
                str(args.superci_davidson_tol),
                "--superci-davidson-max-space",
                str(args.superci_davidson_max_space),
                "--bond-dimension",
                str(args.bond_dimension),
                "--dmrg-sweeps",
                str(args.dmrg_sweeps),
                "--dmrg-tol",
                str(args.dmrg_tol),
                "--dmrg-thrd",
                str(args.dmrg_thrd),
                "--dmrg-davidson-max-iter",
                str(args.dmrg_davidson_max_iter),
                "--dmrg-stack-memory",
                str(args.dmrg_stack_memory),
                "--random-seed",
                str(args.random_seed),
                "--scf-conv-tol",
                str(args.scf_conv_tol),
                "--scf-max-cycle",
                str(args.scf_max_cycle),
                "--fci-conv-tol",
                str(args.fci_conv_tol),
                "--fci-max-cycle",
                str(args.fci_max_cycle),
            ]
            if args.force:
                command.append("--force")
            log_path = args.logs_dir / ("%s-%s.log" % (element, method))
            print("RUN  %s %s -> %s" % (element, method, log_path), flush=True)
            environment = dict(os.environ)
            environment.update(
                {
                    "OMP_NUM_THREADS": str(args.threads),
                    "OPENBLAS_NUM_THREADS": str(args.threads),
                    "MKL_NUM_THREADS": str(args.threads),
                }
            )
            with log_path.open("w", encoding="utf-8") as log_handle:
                completed = subprocess.run(
                    command,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    env=environment,
                    check=False,
                )
            if completed.returncode:
                failed.append((element, method, completed.returncode))
                print(
                    "FAIL %s %s (exit %d); see %s"
                    % (element, method, completed.returncode, log_path),
                    flush=True,
                )
            else:
                print("DONE %s %s" % (element, method), flush=True)

    summary_command = [
        sys.executable,
        str(Path(__file__).with_name("summarize_halogen.py")),
        "--results-dir",
        str(args.results_dir),
        "--output-dir",
        str(Path(__file__).resolve().parent),
    ]
    subprocess.run(summary_command, check=False)
    if failed:
        print("Incomplete jobs: %s" % failed, flush=True)
        return 1
    return 0


def parse_args(argv=None):
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--element", choices=ELEMENTS)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--elements", nargs="+", choices=ELEMENTS, default=list(ELEMENTS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--basis", default="cc-pvtz-dk")
    parser.add_argument("--results-dir", type=Path, default=root / "results")
    parser.add_argument("--scratch-dir", type=Path, default=root / ".scratch")
    parser.add_argument("--logs-dir", type=Path, default=root / "logs")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", type=int, default=4)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-memory", type=float, default=32000.0)
    parser.add_argument("--scf-conv-tol", type=float, default=1e-12)
    parser.add_argument("--scf-max-cycle", type=int, default=200)
    parser.add_argument("--fci-conv-tol", type=float, default=1e-12)
    parser.add_argument("--fci-max-cycle", type=int, default=1000)
    parser.add_argument("--cholesky-tau", type=float, default=1e-10)
    parser.add_argument("--max-cycle-macro", type=int, default=50)
    parser.add_argument("--conv-tol", type=float, default=1e-8)
    parser.add_argument("--conv-tol-grad", type=float, default=1e-4)
    parser.add_argument("--max-stepsize", type=float, default=0.2)
    parser.add_argument("--superci-davidson-tol", type=float, default=1e-7)
    parser.add_argument("--superci-davidson-max-space", type=int, default=30)
    parser.add_argument("--bond-dimension", type=int, default=32)
    parser.add_argument("--dmrg-sweeps", type=int, default=8)
    parser.add_argument("--dmrg-tol", type=float, default=1e-12)
    parser.add_argument("--dmrg-thrd", type=float, default=1e-20)
    parser.add_argument("--dmrg-davidson-max-iter", type=int, default=1000)
    parser.add_argument("--dmrg-stack-memory", type=float, default=512.0)
    parser.add_argument("--random-seed", type=int, default=2468)
    args = parser.parse_args(argv)
    args.results_dir = args.results_dir.resolve()
    args.scratch_dir = args.scratch_dir.resolve()
    args.logs_dir = args.logs_dir.resolve()
    if args.worker and (args.element is None or args.method is None):
        parser.error("--worker requires --element and --method")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.worker:
        return run_worker(args, args.element, args.method)
    return run_matrix(args)


if __name__ == "__main__":
    raise SystemExit(main())
