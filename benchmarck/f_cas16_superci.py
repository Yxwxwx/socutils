#!/usr/bin/env python
"""F CAS(7e,16 spinors) six-state exact-CI/DMRG Super-CI benchmark.

Four routes are compared: full determinant-space iterative CI and Block2
state-averaged DMRG, each with general-complex or Kramers-restricted orbital
optimization.  A fixed-initial-orbital probe is run separately so that active
solver errors cannot be confused with orbital-optimization errors.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import scipy.linalg
from pyscf import __version__ as pyscf_version
from pyscf import gto, scf
from pyscf.data.nist import HARTREE2WAVENUMBER
from pyscf.fci import fci_dhf_slow

from socutils.cd.cd import CD
from socutils.dmrg import DMRGCI
from socutils.dmrg.dmrgci import energy_from_rdms
from socutils.dmrg.kramers import identify_kramers_orbitals, kramers_residual
from socutils.mcscf import zmcscf
from socutils.scf import spinor_hf
from socutils.scf.spinor_hf import density_fit as spinor_density_fit

try:
    from halogen_six_state import (
        _basis_slug,
        _jsonable,
        _valid_cderi,
        _write_json,
        _write_npz,
    )
except ModuleNotFoundError:
    from benchmarck.halogen_six_state import (
        _basis_slug,
        _jsonable,
        _valid_cderi,
        _write_json,
        _write_npz,
    )


ELEMENT = "F"
NCAS = 16
NELECAS = 7
NROOTS = 6
WEIGHTS = np.ones(NROOTS) / NROOTS
METHODS = (
    "exact-general",
    "dmrg-general",
    "exact-kramers",
    "dmrg-kramers",
)
RESTRICTIONS = ("general", "kramers")
EXPERIMENTAL_SPLITTING_CM = 404.141
EXPERIMENTAL_UNCERTAINTY_CM = 0.002
PROTOCOL_VERSION = 1


def _method_parts(method):
    solver, restriction = method.split("-", 1)
    return solver, restriction


def _protocol(args, *, method=None, restriction=None, task="superci"):
    if method is not None:
        solver, restriction = _method_parts(method)
    else:
        solver = "exact+dmrg"
    result = {
        "version": PROTOCOL_VERSION,
        "task": task,
        "element": ELEMENT,
        "basis": args.basis,
        "hamiltonian": (
            "PySCF one-electron spin-orbit X2C with Coulomb-only "
            "two-electron terms"
        ),
        "initial_reference": "F- closed-shell X2C SCF",
        "target": "neutral F 2s2 2p5 2P-odd",
        "active_space": "CAS(7 electrons, 16 spinors)",
        "ncas": NCAS,
        "nelecas": NELECAS,
        "nroots": NROOTS,
        "weights": WEIGHTS.tolist(),
        "restriction": restriction,
        "active_solver": solver,
        "determinant_dimension": math.comb(NCAS, NELECAS),
        "state_average": True,
        "cholesky_tau": args.cholesky_tau,
        "natorb": False,
        "canonicalize": False,
        "superci": {
            "conv_tol": args.conv_tol,
            "conv_tol_grad": args.conv_tol_grad,
            "max_cycle_macro": args.max_cycle_macro,
            "max_stepsize": args.max_stepsize,
            "davidson_tol": args.superci_davidson_tol,
            "davidson_max_space": args.superci_davidson_max_space,
            "davidson_strict": True,
        },
        "exact_ci": {
            "space": "complete determinant space",
            "algorithm": "PySCF fci_dhf_slow Davidson",
            "conv_tol": args.fci_conv_tol,
            "max_cycle": args.fci_max_cycle,
            "max_space": args.fci_max_space,
            "davidson_only": True,
        },
        "dmrg": {
            "algorithm": "Block2 SGFCPX state-averaged MultiMPS",
            "local_eigensolver": "Block2 Normal (Olsen)",
            "bond_dimension": args.bond_dimension,
            "n_sweeps": args.dmrg_sweeps,
            "twosite_to_onesite": 2,
            "noise": 0.0,
            "energy_tolerance": args.dmrg_tol,
            "local_squared_residual_threshold": args.dmrg_thrd,
            "davidson_max_iter": args.dmrg_davidson_max_iter,
            "random_seed": args.random_seed,
            "n_threads": args.threads,
            "stack_memory_mb": args.dmrg_stack_memory,
            "ecore_in_mpo": False,
            "npdm_site_type": 2,
            "npdm_cutoff": 1e-24,
        },
        "kramers": {
            "orbital_step_projection": restriction == "kramers",
            "rdm_projection": False,
            "energy_tolerance": args.kramers_energy_tol,
            "residual_tolerance": args.kramers_residual_tol,
            "orbital_tolerance": args.kramers_orbital_tol,
        },
        "experimental_reference": {
            "transition": "F I 2s2 2p5 2P3/2 -> 2P1/2",
            "splitting_cm-1": EXPERIMENTAL_SPLITTING_CM,
            "uncertainty_cm-1": EXPERIMENTAL_UNCERTAINTY_CM,
        },
    }
    if method is not None:
        result["method"] = method
    return result


def _make_mean_field(mol, cderi_path, cholesky_tau, restriction):
    with_df = CD(mol, tau=cholesky_tau)
    if _valid_cderi(cderi_path, mol):
        with_df._cderi = str(cderi_path)
    else:
        if cderi_path.exists():
            cderi_path.unlink()
        cderi_path.parent.mkdir(parents=True, exist_ok=True)
        with_df._cderi_to_save = str(cderi_path)

    if restriction == "general":
        mean_field = scf.X2C(mol)
    else:
        x2c_reference = scf.X2C(mol)
        mean_field = spinor_hf.KRHF(mol)
        mean_field.with_x2c = x2c_reference.with_x2c
        mean_field._keys = set(mean_field._keys).union({"with_x2c"})
    return spinor_density_fit(mean_field, with_df=with_df)


def _load_scf_cache(path, protocol, mean_field, orbital_tolerance, restriction):
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if json.loads(str(data["protocol"].item())) != protocol:
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
        kramers = None
        if restriction == "kramers":
            kramers = identify_kramers_orbitals(
                mean_field.mol,
                mean_field.mo_coeff,
                mean_field.get_ovlp(),
                tolerance=orbital_tolerance,
            ).diagnostics
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
        "kramers_orbitals": kramers,
    }


def _run_or_load_scf(args, restriction):
    mol = gto.M(
        atom="F 0 0 0",
        basis=args.basis,
        charge=-1,
        spin=0,
        symmetry=True,
        verbose=args.verbose,
        max_memory=args.max_memory,
    )
    scf_root = args.scratch_dir / _basis_slug(args.basis) / ELEMENT
    cderi_path = scf_root / "cderi.h5"
    cache_path = scf_root / ("anion_%s_x2c_scf.npz" % restriction)
    mean_field = _make_mean_field(
        mol, cderi_path, args.cholesky_tau, restriction
    )
    mean_field.conv_tol = args.scf_conv_tol
    mean_field.max_cycle = args.scf_max_cycle
    protocol = {
        "version": PROTOCOL_VERSION,
        "element": ELEMENT,
        "basis": args.basis,
        "charge": -1,
        "spin": 0,
        "symmetry": True,
        "restriction": restriction,
        "driver": type(mean_field).__name__,
        "hamiltonian": "PySCF one-electron spin-orbit X2C",
        "cholesky_tau": args.cholesky_tau,
        "scf_conv_tol": args.scf_conv_tol,
        "orbital_tolerance": args.kramers_orbital_tol,
        "pyscf_version": pyscf_version,
    }
    cached = _load_scf_cache(
        cache_path,
        protocol,
        mean_field,
        args.kramers_orbital_tol,
        restriction,
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

    def callback(environment):
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

    mean_field.callback = callback
    started = time.perf_counter()
    mean_field.kernel()
    wall_seconds = time.perf_counter() - started
    if not mean_field.converged:
        raise RuntimeError("F- %s X2C SCF did not converge" % restriction)
    if not _valid_cderi(cderi_path, mol):
        raise RuntimeError("Cholesky cache was not written correctly")
    kramers = None
    if restriction == "kramers":
        kramers = identify_kramers_orbitals(
            mol,
            mean_field.mo_coeff,
            mean_field.get_ovlp(),
            tolerance=args.kramers_orbital_tol,
        ).diagnostics
    _write_npz(
        cache_path,
        protocol=json.dumps(protocol, sort_keys=True),
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
        "kramers_orbitals": kramers,
    }


def _exact_solver(args, mol):
    solver = fci_dhf_slow.FCISolver(mol)
    # The relativistic solver inherits a nonrelativistic spin diagnostic that
    # returns ``NotImplemented``.  PySCF's state-average finalizer otherwise
    # mistakes that callable for a usable per-root S^2 implementation.
    solver.spin_square = None
    solver.states_spin_square = None
    solver.nroots = NROOTS
    solver.conv_tol = args.fci_conv_tol
    solver.max_cycle = args.fci_max_cycle
    solver.max_space = args.fci_max_space
    solver.davidson_only = True
    return solver


def _dmrg_solver(args, mol, scratch, restriction):
    solver = DMRGCI(mol).init(
        ncas=NCAS,
        nelecas=NELECAS,
        nroots=NROOTS,
        bond_dims=[args.bond_dimension] * args.dmrg_sweeps,
        noises=[0.0] * args.dmrg_sweeps,
        thrds=[args.dmrg_thrd] * args.dmrg_sweeps,
        n_sweeps=args.dmrg_sweeps,
        tol=args.dmrg_tol,
        scratch=scratch,
        n_threads=args.threads,
        stack_memory=args.dmrg_stack_memory,
        random_seed=args.random_seed,
        dav_max_iter=args.dmrg_davidson_max_iter,
        npdm_site_type=2,
        npdm_cutoff=1e-24,
    )
    if restriction == "kramers":
        solver.kramers_restricted(
            energy_tolerance=args.kramers_energy_tol,
            residual_tolerance=args.kramers_residual_tol,
            orbital_tolerance=args.kramers_orbital_tol,
            project=False,
        )
    return solver


def _configure_casscf(args, mol, mean_field, method, scratch):
    solver_kind, restriction = _method_parts(method)
    mc = zmcscf.CASSCF(mean_field, ncas=NCAS, nelecas=NELECAS)
    if solver_kind == "exact":
        mc.fcisolver = _exact_solver(args, mol)
    else:
        mc.fcisolver = _dmrg_solver(args, mol, scratch, restriction)
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


def _level_statistics(root_energies):
    values = np.asarray(root_energies)
    if np.max(abs(values.imag)) > 1e-9:
        raise RuntimeError("active solver returned non-real root energies")
    roots = np.sort(np.asarray(values.real, dtype=float))
    if roots.shape != (NROOTS,):
        raise RuntimeError("expected six final root energies")
    lower = roots[:4]
    upper = roots[4:]
    splitting_eh = float(np.mean(upper) - np.mean(lower))
    splitting_cm = splitting_eh * HARTREE2WAVENUMBER
    return {
        "ordered_root_energies": roots,
        "ground": {
            "configuration": "2s2 2p5",
            "term": "2P-odd",
            "J": "3/2",
            "degeneracy": 4,
            "mean_energy": float(np.mean(lower)),
            "energy_spread": float(np.ptp(lower)),
            "relative_cm-1": 0.0,
        },
        "excited": {
            "configuration": "2s2 2p5",
            "term": "2P-odd",
            "J": "1/2",
            "degeneracy": 2,
            "mean_energy": float(np.mean(upper)),
            "energy_spread": float(np.ptp(upper)),
            "relative_cm-1": splitting_cm,
        },
        "splitting_eh": splitting_eh,
        "splitting_cm-1": splitting_cm,
        "experiment_cm-1": EXPERIMENTAL_SPLITTING_CM,
        "experiment_uncertainty_cm-1": EXPERIMENTAL_UNCERTAINTY_CM,
        "error_cm-1": splitting_cm - EXPERIMENTAL_SPLITTING_CM,
        "absolute_error_cm-1": abs(splitting_cm - EXPERIMENTAL_SPLITTING_CM),
        "relative_error_percent": (
            100.0
            * (splitting_cm - EXPERIMENTAL_SPLITTING_CM)
            / EXPERIMENTAL_SPLITTING_CM
        ),
    }


def _orthogonal_projector(mo, overlap, start, size):
    eigenvalues, eigenvectors = scipy.linalg.eigh(overlap)
    overlap_half = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T.conj()
    orbitals = overlap_half @ mo[:, start : start + size]
    return orbitals @ orbitals.T.conj()


def _final_density_snapshot(mc, restriction, tolerance):
    solver = mc.fcisolver
    dm1, dm2 = solver.make_rdm12(mc.ci, NCAS, NELECAS)
    result = {
        "particle_number": np.trace(dm1),
        "dm1_hermiticity": np.max(abs(dm1 - dm1.T.conj())),
        "dm2_hermiticity": np.max(
            abs(dm2.conj() - dm2.transpose(1, 0, 3, 2))
        ),
        "natural_occupations": np.linalg.eigvalsh(
            (dm1 + dm1.T.conj()) * 0.5
        )[::-1],
    }
    if restriction == "kramers":
        active = mc.mo_coeff[:, mc.ncore : mc.ncore + NCAS]
        mapping = identify_kramers_orbitals(
            mc.mol,
            active,
            mc._scf.get_ovlp(),
            tolerance=tolerance,
        )
        residual = kramers_residual(mapping.time_reversal, dm1, dm2)
        result["kramers"] = {
            "active_orbital_pairs": mapping.pairs,
            "active_orbital_phases": mapping.phases,
            "active_orbital_diagnostics": mapping.diagnostics,
            "state_average_rdm_residual": residual,
        }
        if max(residual.values()) > tolerance:
            raise RuntimeError(
                "final Kramers RDM residual %.3e exceeds %.3e"
                % (max(residual.values()), tolerance)
            )
    return result


def _result_matches(path, protocol):
    if not path.is_file():
        return False
    try:
        with path.open(encoding="utf-8") as handle:
            result = json.load(handle)
    except Exception:
        return False
    return result.get("status") == "ok" and result.get("protocol") == protocol


def run_worker(args, method):
    result_path = args.results_dir / (method + ".json")
    progress_path = args.results_dir / (method + ".progress.json")
    orbital_path = args.results_dir / (method + "-orbitals.npz")
    protocol = _protocol(args, method=method)
    if not args.force and _result_matches(result_path, protocol):
        print("SKIP %s: matching converged result exists" % method, flush=True)
        return 0

    solver_kind, restriction = _method_parts(method)
    started = time.perf_counter()
    _write_json(
        progress_path,
        {"status": "running", "protocol": protocol, "started_unix": time.time()},
    )
    mc = None
    try:
        mol, mean_field, scf_result = _run_or_load_scf(args, restriction)
        initial_mo = np.array(mean_field.mo_coeff, copy=True)
        mol.charge = 0
        mol.spin = 1
        scratch = args.scratch_dir / "dmrg" / method
        scratch.mkdir(parents=True, exist_ok=True)
        mc = _configure_casscf(args, mol, mean_field, method, scratch)

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
        snapshot = _solver_snapshot(mc.fcisolver)
        roots = np.asarray(snapshot["root_energies"]).real.astype(float)
        density = _final_density_snapshot(
            mc, restriction, args.kramers_orbital_tol
        )
        fully_converged = bool(mc.converged) and bool(
            np.all(getattr(mc.fcisolver, "converged", True))
        )
        level_data = _level_statistics(roots)
        state_average_consistency = abs(float(np.dot(WEIGHTS, roots)) - mc.e_tot)
        payload = {
            "status": "ok" if fully_converged else "not_converged",
            "protocol": protocol,
            "scf": scf_result,
            "result": {
                "converged": fully_converged,
                "total_energy": float(np.real(mc.e_tot)),
                "cas_energy": float(np.real(mc.e_cas)),
                "final_gradient_norm": float(mc.final_orbital_gradient_norm),
                "macroiterations": len(history),
                "orbital_updates": sum(
                    "applied_orbital_step_norm" in row for row in history
                ),
                "root_energies": roots,
                "state_average_energy_consistency": state_average_consistency,
                "wall_seconds": time.perf_counter() - started,
            },
            "levels": level_data,
            "density": density,
            "solver": snapshot,
            "macro_history": history,
            "orbitals_file": str(orbital_path),
        }
        _write_npz(
            orbital_path,
            mo_coeff=mc.mo_coeff,
            overlap=mean_field.get_ovlp(),
            active_projector=_orthogonal_projector(
                mc.mo_coeff, mean_field.get_ovlp(), mc.ncore, NCAS
            ),
        )
        _write_json(result_path, payload)
        if progress_path.exists():
            progress_path.unlink()
        print(
            "%s: status=%s E=%.15f split=%.6f cm-1 |g|=%.3e "
            "macros=%d wall=%.1fs"
            % (
                method,
                payload["status"],
                payload["result"]["total_energy"],
                level_data["splitting_cm-1"],
                payload["result"]["final_gradient_norm"],
                payload["result"]["macroiterations"],
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
                None if mc is None else getattr(mc, "superci_diagnostics", None)
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
        if mc is not None and solver_kind == "dmrg":
            close = getattr(mc.fcisolver, "close", None)
            if close is not None:
                close()


def _ordered_rdms(solver, ci, energies):
    dm1s, dm2s = solver.states_make_rdm12(ci, NCAS, NELECAS)
    order = np.argsort(np.asarray(energies, dtype=float))
    return (
        [dm1s[int(root)] for root in order],
        [dm2s[int(root)] for root in order],
    )


def _mean_tensors(tensors, roots):
    return sum(tensors[root] for root in roots) / len(roots)


def run_probe(args, restriction):
    result_path = args.results_dir / ("fixed-%s.json" % restriction)
    protocol = _protocol(args, restriction=restriction, task="fixed-orbital-probe")
    if not args.force and _result_matches(result_path, protocol):
        print("SKIP fixed-%s" % restriction, flush=True)
        return 0

    started = time.perf_counter()
    dmrg_mc = None
    try:
        mol, mean_field, scf_result = _run_or_load_scf(args, restriction)
        initial_mo = np.array(mean_field.mo_coeff, copy=True)
        mol.charge = 0
        mol.spin = 1
        integral_mc = zmcscf.CASSCF(mean_field, ncas=NCAS, nelecas=NELECAS)
        integral_mc.natorb = False
        integral_mc.canonicalize_ = False
        h1e, ecore = integral_mc.get_h1eff(initial_mo)
        eri = np.asarray(integral_mc.get_h2eff(initial_mo)).reshape((NCAS,) * 4)

        exact_mc = _configure_casscf(
            args,
            mol,
            mean_field,
            "exact-%s" % restriction,
            args.scratch_dir / "unused-exact",
        )
        exact_energy, exact_ci = exact_mc.fcisolver.kernel(
            h1e, eri, NCAS, NELECAS, ecore=ecore, verbose=args.verbose
        )
        exact_roots = np.asarray(exact_mc.fcisolver.e_states).real.astype(float)
        exact_dm1s, exact_dm2s = _ordered_rdms(
            exact_mc.fcisolver, exact_ci, exact_roots
        )

        dmrg_mc = _configure_casscf(
            args,
            mol,
            mean_field,
            "dmrg-%s" % restriction,
            args.scratch_dir / "dmrg" / ("fixed-%s" % restriction),
        )
        if restriction == "kramers":
            active = initial_mo[
                :, integral_mc.ncore : integral_mc.ncore + NCAS
            ]
            dmrg_mc.fcisolver.set_orbital_context(
                active, mean_field.get_ovlp(), mol=mol
            )
        dmrg_energy, dmrg_ci = dmrg_mc.fcisolver.kernel(
            h1e, eri, NCAS, NELECAS, ecore=ecore, verbose=args.verbose
        )
        dmrg_roots = np.asarray(dmrg_mc.fcisolver.e_states).real.astype(float)
        dmrg_dm1s, dmrg_dm2s = _ordered_rdms(
            dmrg_mc.fcisolver, dmrg_ci, dmrg_roots
        )

        exact_sorted = np.sort(exact_roots)
        dmrg_sorted = np.sort(dmrg_roots)
        groups = {"all_six": range(6), "lower_quartet": range(4), "upper_doublet": range(4, 6)}
        rdm_errors = {}
        for name, roots in groups.items():
            exact_dm1 = _mean_tensors(exact_dm1s, roots)
            exact_dm2 = _mean_tensors(exact_dm2s, roots)
            dmrg_dm1 = _mean_tensors(dmrg_dm1s, roots)
            dmrg_dm2 = _mean_tensors(dmrg_dm2s, roots)
            rdm_errors[name] = {
                "dm1_max_abs": float(np.max(abs(dmrg_dm1 - exact_dm1))),
                "dm2_max_abs": float(np.max(abs(dmrg_dm2 - exact_dm2))),
            }

        solver = dmrg_mc.fcisolver
        overlap = np.asarray(solver.root_overlap)
        hamiltonian = np.asarray(solver.projected_hamiltonian)
        rdm_energy_errors = []
        dmrg_order = np.argsort(dmrg_roots)
        for ordered_root, source_root in enumerate(dmrg_order):
            reconstructed = energy_from_rdms(
                h1e,
                eri,
                dmrg_dm1s[ordered_root],
                dmrg_dm2s[ordered_root],
                ecore=ecore,
            )
            rdm_energy_errors.append(
                abs(reconstructed - dmrg_roots[int(source_root)])
            )
        hse = hamiltonian - overlap * dmrg_roots[np.newaxis, :]

        kramers = None
        if restriction == "kramers":
            mapping = identify_kramers_orbitals(
                mol,
                initial_mo[
                    :, integral_mc.ncore : integral_mc.ncore + NCAS
                ],
                mean_field.get_ovlp(),
                tolerance=args.kramers_orbital_tol,
            )
            exact_ensemble1 = _mean_tensors(exact_dm1s, range(6))
            exact_ensemble2 = _mean_tensors(exact_dm2s, range(6))
            dmrg_ensemble1 = _mean_tensors(dmrg_dm1s, range(6))
            dmrg_ensemble2 = _mean_tensors(dmrg_dm2s, range(6))
            kramers = {
                "exact_ensemble_residual": kramers_residual(
                    mapping.time_reversal, exact_ensemble1, exact_ensemble2
                ),
                "dmrg_ensemble_residual": kramers_residual(
                    mapping.time_reversal, dmrg_ensemble1, dmrg_ensemble2
                ),
                "adapter": solver.kramers_diagnostics,
            }

        fully_converged = bool(
            np.all(getattr(exact_mc.fcisolver, "converged", True))
        ) and bool(np.all(getattr(solver, "converged", True)))
        payload = {
            "status": "ok" if fully_converged else "not_converged",
            "protocol": protocol,
            "scf": scf_result,
            "integrals": {
                "ecore": ecore,
                "h1_hermiticity": np.max(abs(h1e - h1e.T.conj())),
                "eri_shape": eri.shape,
            },
            "exact": {
                "state_average_energy": exact_energy,
                "root_energies": exact_sorted,
                "levels": _level_statistics(exact_sorted),
                "converged": getattr(exact_mc.fcisolver, "converged", None),
            },
            "dmrg": {
                "state_average_energy": dmrg_energy,
                "root_energies": dmrg_sorted,
                "levels": _level_statistics(dmrg_sorted),
                "convergence": solver.convergence_info,
                "root_overlap_error": np.max(abs(overlap - np.eye(NROOTS))),
                "projected_eigen_equation_error": np.max(abs(hse)),
                "rdm_energy_errors": rdm_energy_errors,
            },
            "comparison": {
                "root_energy_max_abs_eh": np.max(abs(dmrg_sorted - exact_sorted)),
                "state_average_energy_abs_eh": abs(dmrg_energy - exact_energy),
                "splitting_error_cm-1": (
                    _level_statistics(dmrg_sorted)["splitting_cm-1"]
                    - _level_statistics(exact_sorted)["splitting_cm-1"]
                ),
                "rdm_errors": rdm_errors,
            },
            "kramers": kramers,
            "wall_seconds": time.perf_counter() - started,
        }
        _write_json(result_path, payload)
        print(
            "fixed-%s: status=%s max|dE|=%.3e Eh split error=%.6g cm-1 "
            "S-I=%.3e wall=%.1fs"
            % (
                restriction,
                payload["status"],
                payload["comparison"]["root_energy_max_abs_eh"],
                payload["comparison"]["splitting_error_cm-1"],
                payload["dmrg"]["root_overlap_error"],
                payload["wall_seconds"],
            ),
            flush=True,
        )
        return 0 if fully_converged else 2
    except Exception as error:
        _write_json(
            result_path,
            {
                "status": "error",
                "protocol": protocol,
                "elapsed_seconds": time.perf_counter() - started,
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                },
            },
        )
        traceback.print_exc()
        return 1
    finally:
        if dmrg_mc is not None:
            close = getattr(dmrg_mc.fcisolver, "close", None)
            if close is not None:
                close()


def _forwarded_arguments(args):
    names = (
        "basis",
        "results_dir",
        "scratch_dir",
        "logs_dir",
        "verbose",
        "threads",
        "max_memory",
        "scf_conv_tol",
        "scf_max_cycle",
        "fci_conv_tol",
        "fci_max_cycle",
        "fci_max_space",
        "cholesky_tau",
        "max_cycle_macro",
        "conv_tol",
        "conv_tol_grad",
        "max_stepsize",
        "superci_davidson_tol",
        "superci_davidson_max_space",
        "bond_dimension",
        "dmrg_sweeps",
        "dmrg_tol",
        "dmrg_thrd",
        "dmrg_davidson_max_iter",
        "dmrg_stack_memory",
        "random_seed",
        "kramers_energy_tol",
        "kramers_residual_tol",
        "kramers_orbital_tol",
    )
    forwarded = []
    for name in names:
        forwarded.extend(("--" + name.replace("_", "-"), str(getattr(args, name))))
    if args.force:
        forwarded.append("--force")
    return forwarded


def _run_child(args, command, log_path):
    environment = dict(os.environ)
    environment.update(
        {
            "OMP_NUM_THREADS": str(args.threads),
            "OPENBLAS_NUM_THREADS": str(args.threads),
            "MKL_NUM_THREADS": str(args.threads),
        }
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        return subprocess.run(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=environment,
            check=False,
        ).returncode


def run_matrix(args):
    script = Path(__file__).resolve()
    forwarded = _forwarded_arguments(args)
    failed = []
    if not args.skip_probes:
        for restriction in RESTRICTIONS:
            command = [
                sys.executable,
                str(script),
                "--task",
                "probe",
                "--restriction",
                restriction,
                *forwarded,
            ]
            log_path = args.logs_dir / ("fixed-%s.log" % restriction)
            print("RUN  fixed-%s -> %s" % (restriction, log_path), flush=True)
            code = _run_child(args, command, log_path)
            if code:
                failed.append(("fixed-%s" % restriction, code))
                print("FAIL fixed-%s (exit %d)" % (restriction, code), flush=True)
                if not args.continue_after_probe_failure:
                    return 1
            else:
                print("DONE fixed-%s" % restriction, flush=True)

    for method in args.methods:
        command = [
            sys.executable,
            str(script),
            "--task",
            "worker",
            "--method",
            method,
            *forwarded,
        ]
        log_path = args.logs_dir / (method + ".log")
        print("RUN  %s -> %s" % (method, log_path), flush=True)
        code = _run_child(args, command, log_path)
        if code:
            failed.append((method, code))
            print("FAIL %s (exit %d)" % (method, code), flush=True)
        else:
            print("DONE %s" % method, flush=True)
    if failed:
        print("Incomplete jobs: %s" % failed, flush=True)
        return 1
    return 0


def parse_args(argv=None):
    root = Path(__file__).resolve().parent / "f_cas16"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("matrix", "worker", "probe"), default="matrix")
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--restriction", choices=RESTRICTIONS)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--skip-probes", action="store_true")
    parser.add_argument("--continue-after-probe-failure", action="store_true")
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
    parser.add_argument("--fci-conv-tol", type=float, default=1e-10)
    parser.add_argument("--fci-max-cycle", type=int, default=1000)
    parser.add_argument("--fci-max-space", type=int, default=80)
    parser.add_argument("--cholesky-tau", type=float, default=1e-10)
    parser.add_argument("--max-cycle-macro", type=int, default=50)
    parser.add_argument("--conv-tol", type=float, default=1e-8)
    parser.add_argument("--conv-tol-grad", type=float, default=1e-4)
    parser.add_argument("--max-stepsize", type=float, default=0.2)
    parser.add_argument("--superci-davidson-tol", type=float, default=1e-7)
    parser.add_argument("--superci-davidson-max-space", type=int, default=30)
    parser.add_argument("--bond-dimension", type=int, default=512)
    parser.add_argument("--dmrg-sweeps", type=int, default=12)
    parser.add_argument("--dmrg-tol", type=float, default=1e-10)
    parser.add_argument("--dmrg-thrd", type=float, default=1e-20)
    parser.add_argument("--dmrg-davidson-max-iter", type=int, default=2000)
    parser.add_argument("--dmrg-stack-memory", type=float, default=2048.0)
    parser.add_argument("--random-seed", type=int, default=2468)
    parser.add_argument("--kramers-energy-tol", type=float, default=1e-7)
    # Individual 2-RDMs extracted from numerically split, exactly degenerate
    # DMRG roots fluctuate at roughly 1e-7 even when the state-averaged RDM,
    # root overlap, and projected eigen-equation are much more accurate.  This
    # remains a validation-only gate: no RDM projection is enabled.
    parser.add_argument("--kramers-residual-tol", type=float, default=1e-6)
    parser.add_argument("--kramers-orbital-tol", type=float, default=1e-8)
    args = parser.parse_args(argv)
    args.results_dir = args.results_dir.resolve()
    args.scratch_dir = args.scratch_dir.resolve()
    args.logs_dir = args.logs_dir.resolve()
    if args.task == "worker" and args.method is None:
        parser.error("--task worker requires --method")
    if args.task == "probe" and args.restriction is None:
        parser.error("--task probe requires --restriction")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.task == "worker":
        return run_worker(args, args.method)
    if args.task == "probe":
        return run_probe(args, args.restriction)
    return run_matrix(args)


if __name__ == "__main__":
    raise SystemExit(main())
