#!/usr/bin/env python
"""Run the six-state Kramers-restricted halogen Super-CI benchmark.

The exact-FCI and Block2 workers share one Kramers-restricted mean-field
reference and one orbital-optimization protocol.  Block2 uses one genuine
state-averaged MultiMPS with the same six weights as the orbital functional.
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
from socutils.dmrg.kramers import identify_kramers_orbitals, kramers_residual
from socutils.mcscf import zmcscf
from socutils.scf import spinor_hf
from socutils.scf.spinor_hf import density_fit as spinor_density_fit

try:
    from halogen_six_state import (
        ELEMENTS,
        NCAS,
        NELECAS,
        NROOTS,
        WEIGHTS,
        _basis_slug,
        _jsonable,
        _valid_cderi,
        _write_json,
        _write_npz,
    )
except ModuleNotFoundError:  # Supports ``python -m benchmarck...`` as well.
    from benchmarck.halogen_six_state import (
        ELEMENTS,
        NCAS,
        NELECAS,
        NROOTS,
        WEIGHTS,
        _basis_slug,
        _jsonable,
        _valid_cderi,
        _write_json,
        _write_npz,
    )


METHODS = ("exact-superci", "dmrg-superci")
PROTOCOL_VERSION = 6


def _protocol(args, element, method=None):
    protocol = {
        "version": PROTOCOL_VERSION,
        "element": element,
        "basis": args.basis,
        "hamiltonian": "PySCF one-electron spin-orbit X2C with Coulomb-only two-electron terms",
        "mean_field_driver": "socutils.scf.spinor_hf.KRHF",
        "orbital_constraint": "Kramers-restricted",
        "orbital_step_projection": "explicit time-reversal projection",
        "initial_reference": "%s- closed-shell KR-X2C SCF" % element,
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
        "complete_kramers_pairs": True,
        "cholesky_tau": args.cholesky_tau,
        "conv_tol": args.conv_tol,
        "conv_tol_grad": args.conv_tol_grad,
        "max_cycle_macro": args.max_cycle_macro,
        "max_stepsize": args.max_stepsize,
        "natorb": False,
        "canonicalize": False,
        "canonicalization_reason": (
            "disabled inside macroiterations to avoid arbitrary rotations "
            "within exactly degenerate atomic core/virtual manifolds"
        ),
        "max_memory_mb": args.max_memory,
        "superci_davidson_tol": args.superci_davidson_tol,
        "superci_davidson_max_space": args.superci_davidson_max_space,
        "superci_overlap_metric": (
            "basis-covariant full active 1-RDM with canonical null-space "
            "deflation"
        ),
        "kramers": {
            "projection": False,
            "energy_tolerance": args.kramers_energy_tol,
            "residual_tolerance": args.kramers_residual_tol,
            "orbital_tolerance": args.kramers_orbital_tol,
            "projected_hamiltonian_residual": "H_projected - S_projected E",
        },
        "dmrg": {
            "bond_dimension": args.bond_dimension,
            "ecore_in_mpo": False,
            "root_strategy": "state-averaged MultiMPS",
            "local_eigensolver": "Block2 Normal (Olsen)",
            "twosite_to_onesite": 2,
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
        protocol["method"] = method
        protocol["active_solver"] = (
            "exact full-spinor FCI"
            if method == "exact-superci"
            else "Block2 SGFCPX DMRG with Kramers result adapter"
        )
    return protocol


def _make_mean_field(mol, cderi_path, cholesky_tau):
    with_df = CD(mol, tau=cholesky_tau)
    if _valid_cderi(cderi_path, mol):
        with_df._cderi = str(cderi_path)
    else:
        if cderi_path.exists():
            cderi_path.unlink()
        cderi_path.parent.mkdir(parents=True, exist_ok=True)
        with_df._cderi_to_save = str(cderi_path)

    # Keep precisely the one-electron X2C Hamiltonian used by the unrestricted
    # benchmark, but use the repository's quaternion eigensolver for every AO
    # and MO-subspace diagonalization.  The helper and overlap use the same 2c
    # spinor AO ordering.
    x2c_reference = scf.X2C(mol)
    mean_field = spinor_hf.KRHF(mol)
    mean_field.with_x2c = x2c_reference.with_x2c
    mean_field._keys = set(mean_field._keys).union({"with_x2c"})
    return spinor_density_fit(mean_field, with_df=with_df)


def _orbital_mapping(mol, mean_field, tolerance):
    mapping = identify_kramers_orbitals(
        mol,
        mean_field.mo_coeff,
        mean_field.get_ovlp(),
        tolerance=tolerance,
    )
    return {
        "pairs": mapping.pairs,
        "phases": mapping.phases,
        "diagnostics": mapping.diagnostics,
    }


def _load_scf_cache(path, protocol, mean_field, orbital_tolerance):
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            cached_protocol = json.loads(str(data["protocol"].item()))
            if cached_protocol != protocol:
                return None
            mean_field.mo_coeff = np.array(data["mo_coeff"], copy=True)
            mean_field.mo_energy = np.array(data["mo_energy"], copy=True)
            mean_field.mo_occ = np.array(data["mo_occ"], copy=True)
            mean_field.e_tot = float(data["e_tot"].item())
            history = json.loads(str(data["history"].item()))
            wall_seconds = float(data["wall_seconds"].item())
        if mean_field.mo_coeff.shape != (
            mean_field.mol.nao_2c(),
            mean_field.mol.nao_2c(),
        ):
            return None
        mapping = _orbital_mapping(
            mean_field.mol, mean_field, orbital_tolerance
        )
    except Exception:
        return None
    mean_field.converged = True
    return {
        "converged": True,
        "energy": mean_field.e_tot,
        "iterations": len(history),
        "history": history,
        "wall_seconds": wall_seconds,
        "cache_reused": True,
        "kramers_orbitals": mapping,
    }


def _run_or_load_scf(args, element):
    mol = gto.M(
        atom="%s 0 0 0" % element,
        basis=args.basis,
        charge=-1,
        spin=0,
        symmetry=True,
        verbose=args.verbose,
        max_memory=args.max_memory,
    )
    element_scratch = args.scratch_dir / _basis_slug(args.basis) / element
    cderi_path = element_scratch / "cderi.h5"
    cache_path = element_scratch / "anion_kr_x2c_scf.npz"
    mean_field = _make_mean_field(mol, cderi_path, args.cholesky_tau)
    mean_field.conv_tol = args.scf_conv_tol
    mean_field.max_cycle = args.scf_max_cycle
    scf_protocol = {
        "version": PROTOCOL_VERSION,
        "element": element,
        "basis": args.basis,
        "charge": -1,
        "spin": 0,
        "symmetry": True,
        "hamiltonian": "PySCF one-electron spin-orbit X2C",
        "driver": "socutils.scf.spinor_hf.KRHF",
        "cholesky_tau": args.cholesky_tau,
        "scf_conv_tol": args.scf_conv_tol,
        "kramers_orbital_tolerance": args.kramers_orbital_tol,
        "pyscf_version": pyscf_version,
    }
    cached = _load_scf_cache(
        cache_path, scf_protocol, mean_field, args.kramers_orbital_tol
    )
    if cached is not None and _valid_cderi(cderi_path, mol):
        cached.update(
            {
                "nao_nr": mol.nao_nr(),
                "nspinor": mol.nao_2c(),
                "naux": int(mean_field.with_df.get_naoaux()),
                "cderi": str(cderi_path),
            }
        )
        return mol, mean_field, cached

    history = []

    def scf_callback(environment):
        if "cycle" not in environment:
            return
        energy = float(environment["e_tot"])
        history.append(
            {
                "cycle": int(environment["cycle"]),
                "energy": energy,
                "energy_change": (
                    None if not history else energy - history[-1]["energy"]
                ),
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

    mean_field.callback = scf_callback
    started = time.perf_counter()
    mean_field.kernel()
    wall_seconds = time.perf_counter() - started
    if not mean_field.converged:
        raise RuntimeError("%s- KR-X2C SCF did not converge" % element)
    if not _valid_cderi(cderi_path, mol):
        raise RuntimeError("Cholesky cache was not written correctly")
    mapping = _orbital_mapping(mol, mean_field, args.kramers_orbital_tol)
    _write_npz(
        cache_path,
        protocol=json.dumps(scf_protocol, sort_keys=True),
        mo_coeff=mean_field.mo_coeff,
        mo_energy=mean_field.mo_energy,
        mo_occ=mean_field.mo_occ,
        e_tot=mean_field.e_tot,
        history=json.dumps(history, sort_keys=True),
        wall_seconds=wall_seconds,
    )
    return mol, mean_field, {
        "converged": True,
        "energy": float(mean_field.e_tot),
        "iterations": len(history),
        "history": history,
        "wall_seconds": wall_seconds,
        "cache_reused": False,
        "nao_nr": mol.nao_nr(),
        "nspinor": mol.nao_2c(),
        "naux": int(mean_field.with_df.get_naoaux()),
        "cderi": str(cderi_path),
        "kramers_orbitals": mapping,
    }


def _configure_casscf(args, mol, mean_field, method, dmrg_scratch):
    mc = zmcscf.CASSCF(mean_field, ncas=NCAS, nelecas=NELECAS)
    if method == "dmrg-superci":
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
        ).kramers_restricted(
            energy_tolerance=args.kramers_energy_tol,
            residual_tolerance=args.kramers_residual_tol,
            orbital_tolerance=args.kramers_orbital_tol,
            project=False,
        )
        mc.fcisolver = solver
    else:
        mc.fcisolver.conv_tol = args.fci_conv_tol
        mc.fcisolver.max_cycle = args.fci_max_cycle
    mc.state_average_(WEIGHTS)
    mc.natorb = False
    mc.canonicalize_ = False
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
    adapter = getattr(solver, "kramers_adapter", None)
    return {
        "class": type(solver).__name__,
        "root_energies": getattr(solver, "e_states", None),
        "converged": getattr(solver, "converged", None),
        "dmrg_convergence": getattr(solver, "convergence_info", None),
        "rdm_diagnostics": getattr(solver, "rdm_diagnostics", None),
        "kramers_diagnostics": getattr(solver, "kramers_diagnostics", None),
        "kramers_orbital_diagnostics": (
            None if adapter is None else adapter.orbital_diagnostics
        ),
        "kramers_orbital_history": (
            None if adapter is None else adapter.orbital_history
        ),
        "root_overlap": getattr(solver, "root_overlap", None),
        "projected_hamiltonian": getattr(
            solver, "projected_hamiltonian", None
        ),
    }


def _final_kramers_snapshot(mc, mol, tolerance):
    ncore = mc.ncore
    active_mo = mc.mo_coeff[:, ncore : ncore + NCAS]
    mapping = identify_kramers_orbitals(
        mol,
        active_mo,
        mc._scf.get_ovlp(),
        tolerance=tolerance,
    )
    dm1, dm2 = mc.fcisolver.make_rdm12(mc.ci, NCAS, NELECAS)
    residual = kramers_residual(mapping.time_reversal, dm1, dm2)
    residual_max = max(residual.values())
    if residual_max > tolerance:
        raise RuntimeError(
            "final state-average Kramers residual %.3e exceeds %.3e"
            % (residual_max, tolerance)
        )
    return {
        "active_orbital_pairs": mapping.pairs,
        "active_orbital_phases": mapping.phases,
        "active_orbital_diagnostics": mapping.diagnostics,
        "state_average_rdm_residual": residual,
        "particle_number": np.trace(dm1),
        "dm1_hermiticity": np.max(abs(dm1 - dm1.T.conj())),
        "dm2_hermiticity": np.max(
            abs(dm2.conj() - dm2.transpose(1, 0, 3, 2))
        ),
        "natural_occupations": np.linalg.eigvalsh(
            (dm1 + dm1.T.conj()) * 0.5
        )[::-1],
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
    _write_json(
        progress_path,
        {"status": "running", "protocol": protocol, "started_unix": time.time()},
    )
    mc = None
    try:
        mol, mean_field, scf_result = _run_or_load_scf(args, element)
        initial_mo = np.array(mean_field.mo_coeff, copy=True)
        mol.charge = 0
        mol.spin = 1
        dmrg_scratch = args.scratch_dir / "dmrg-kramers" / element / method
        dmrg_scratch.mkdir(parents=True, exist_ok=True)
        mc = _configure_casscf(
            args, mol, mean_field, method, dmrg_scratch
        )

        def macro_callback(environment):
            _write_json(
                progress_path,
                {
                    "status": "running",
                    "protocol": protocol,
                    "scf": scf_result,
                    "latest_macroiteration": environment,
                    "elapsed_seconds": time.perf_counter() - started,
                },
            )

        mc.superci(mo_coeff=initial_mo, callback=macro_callback)
        history = list(mc.macro_history)
        solver_data = _solver_snapshot(mc.fcisolver)
        kramers_data = _final_kramers_snapshot(
            mc, mol, args.kramers_orbital_tol
        )
        macroiterations = len(history)
        orbital_updates = sum(
            "applied_orbital_step_norm" in entry for entry in history
        )
        fully_converged = bool(mc.converged) and bool(
            np.all(getattr(mc.fcisolver, "converged", True))
        )
        payload = {
            "status": "ok" if fully_converged else "not_converged",
            "protocol": protocol,
            "scf": scf_result,
            "result": {
                "converged": fully_converged,
                "total_energy": float(np.real(mc.e_tot)),
                "cas_energy": float(np.real(mc.e_cas)),
                "final_gradient_norm": float(mc.final_orbital_gradient_norm),
                "macroiterations": macroiterations,
                "orbital_updates": orbital_updates,
                "root_energies": solver_data["root_energies"],
                "natural_occupations": kramers_data["natural_occupations"],
                "wall_seconds": time.perf_counter() - started,
            },
            "kramers": kramers_data,
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
        return 0 if fully_converged else 2
    except Exception as error:
        payload = {
            "status": "error",
            "protocol": protocol,
            "elapsed_seconds": time.perf_counter() - started,
            "superci_diagnostics": (
                None if mc is None else mc.superci_diagnostics
            ),
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
        if mc is not None and method == "dmrg-superci":
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
            ]
            forwarded = {
                "basis": args.basis,
                "results-dir": args.results_dir,
                "scratch-dir": args.scratch_dir,
                "logs-dir": args.logs_dir,
                "verbose": args.verbose,
                "threads": args.threads,
                "max-memory": args.max_memory,
                "max-cycle-macro": args.max_cycle_macro,
                "conv-tol": args.conv_tol,
                "conv-tol-grad": args.conv_tol_grad,
                "max-stepsize": args.max_stepsize,
                "cholesky-tau": args.cholesky_tau,
                "superci-davidson-tol": args.superci_davidson_tol,
                "superci-davidson-max-space": args.superci_davidson_max_space,
                "bond-dimension": args.bond_dimension,
                "dmrg-sweeps": args.dmrg_sweeps,
                "dmrg-tol": args.dmrg_tol,
                "dmrg-thrd": args.dmrg_thrd,
                "dmrg-davidson-max-iter": args.dmrg_davidson_max_iter,
                "dmrg-stack-memory": args.dmrg_stack_memory,
                "random-seed": args.random_seed,
                "scf-conv-tol": args.scf_conv_tol,
                "scf-max-cycle": args.scf_max_cycle,
                "fci-conv-tol": args.fci_conv_tol,
                "fci-max-cycle": args.fci_max_cycle,
                "kramers-energy-tol": args.kramers_energy_tol,
                "kramers-residual-tol": args.kramers_residual_tol,
                "kramers-orbital-tol": args.kramers_orbital_tol,
            }
            for name, value in forwarded.items():
                command.extend(("--" + name, str(value)))
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
        str(Path(__file__).with_name("summarize_kramers_superci.py")),
        "--results-dir",
        str(args.results_dir),
        "--output-dir",
        str(Path(__file__).resolve().parent / "kramers"),
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
    parser.add_argument(
        "--results-dir", type=Path, default=root / "kramers" / "results"
    )
    parser.add_argument("--scratch-dir", type=Path, default=root / ".scratch")
    parser.add_argument(
        "--logs-dir", type=Path, default=root / "kramers" / "logs"
    )
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
    parser.add_argument("--kramers-energy-tol", type=float, default=1e-8)
    parser.add_argument("--kramers-residual-tol", type=float, default=1e-8)
    parser.add_argument("--kramers-orbital-tol", type=float, default=1e-8)
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
