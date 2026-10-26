# SPDX-License-Identifier: GPL-3.0-or-later
"""Fixed-orbital exact-reference Cu/Ag/Au 2+6+4 MS-FIC spectra.

Reuse the original twelve-root SA partition, not a new exact-CI SA density.
Only model references change. All expensive stages have persistent caches;
raw transition densities use the production one-pair temporary disk workflow.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import sys
from types import SimpleNamespace

import numpy as np
from pyscf import lib, scf
from pyblock2.driver.core import DMRGDriver, SymmetryTypes

from socutils.dmrg import DMRGCI
from socutils.mcscf import zmcscf
from socutils.mrpt import make_transition_dm1234
from socutils.mrpt import x2cmsficnevpt2 as ms, nevpt2_utils as u
from socutils.scf import spinor_hf
from .ag_audit import block2_stack_bytes, write_json
from .ag_exact_reference import exact_reference
from . import exact_mps
from tests.nevpt2_mps_response.f_atom import _active_action_links

MANIFOLDS = {"S1_2": (0, 1), "D5_2": tuple(range(2, 8)), "D3_2": tuple(range(8, 12))}
WEIGHTS = np.full(12, 1 / 12)
CONVERSION = 219474.6313705


def write_pickle(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def model_prepared(partition, roots, classes, active, overlap, audits, energies, ci_hash):
    """Insert exact model data without rebuilding the common Dyall partition."""
    roots = tuple(roots)
    np.testing.assert_allclose(active, np.diag(energies[list(roots)]), atol=1e-9, rtol=0)
    np.testing.assert_allclose(overlap, np.eye(len(roots)), atol=1e-10, rtol=0)
    fingerprint = hashlib.sha256((partition["fingerprint"] + ci_hash + repr(roots)).encode()).hexdigest()
    return ms.MSFICPrepared(
        partition["eris"], partition["mo_coeff"], partition["mo_energy"], partition["sa_density"],
        roots, overlap, active, active + partition["ecore"] * overlap, classes,
        {**audits, "sa_roots": tuple(range(12)), "sa_weights": WEIGHTS.copy(),
         "external_fock_defect": partition["external_fock_defect"], "fingerprint": fingerprint,
         "partition_fingerprint": partition["fingerprint"], "reference_ci_sha256": ci_hash,
         "frozen_common_sa_partition": True})


def common_partition(directory, reference, threads):
    path = directory / "partition.pkl"
    source = directory.parent
    symbol = source.parent.name
    hsha = str(reference["hamiltonian_sha256"].item())
    with (source / "dmrg_state/reference.npz").open("rb") as handle:
        snapshot_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    if path.exists():
        with path.open("rb") as handle:
            partition = pickle.load(handle)
        if partition["hamiltonian_sha256"] != hsha or partition["source_snapshot_sha256"] != snapshot_hash:
            raise ValueError("partition cache belongs to another Hamiltonian")
        return partition
    ag_cache = source / "ms_nevpt2_audit/prepared.pkl"
    if symbol == "Ag":
        with ag_cache.open("rb") as handle:
            old = pickle.load(handle)
        settings = json.loads((ag_cache.parent / "audit_settings.json").read_text())
        if settings["hamiltonian_sha256"] != hsha:
            raise ValueError("validated Ag partition belongs to another Hamiltonian")
        np.testing.assert_array_equal(old.diagnostics["sa_weights"], WEIGHTS)
        core = old.reference - old.active_reference
        np.testing.assert_allclose(core, float(reference["ecore"]) * np.eye(2), atol=1e-8, rtol=0)
        partition = dict(eris=old.eris, mo_coeff=old.mo_coeff, mo_energy=old.mo_energy,
                         sa_density=old.sa_density, ecore=float(reference["ecore"]),
                         external_fock_defect=old.diagnostics["external_fock_defect"])
        del old
    else:
        hf_checkpoint = (source.parent / "unc_ano_rcc_trials/hf_kr/hf.chk"
                         if symbol == "Cu" else source / "hf_kr/hf.chk")
        mol, hf = scf.chkfile.load_scf(str(hf_checkpoint))
        if mol.atom_symbol(0) != symbol:
            raise ValueError("HF checkpoint does not match the requested atom")
        mol.stdout, mol.verbose, mol.max_memory = sys.stdout, 4, int(
            os.environ.get("PYSCF_MAX_MEMORY", "1800000"))
        mf = spinor_hf.SCF(mol).x2camf()
        mf.__dict__.update(hf)
        mf.converged, mf.max_memory = True, mol.max_memory
        mf.with_x2c.hcore = reference["hcore_ao"]
        mf.chkfile = str(directory / "unused_hf.chk")
        mol.charge, mol.spin = 0, 1
        mc = zmcscf.CASSCF(mf, 22, 11)
        mc.fcisolver = DMRGCI(mol).init(
            ncas=22, nelecas=11, nroots=12, max_bond_dimension=2000,
            tol=1e-8, schedule_thrd_max=1e-16, n_threads=threads,
            stack_memory=block2_stack_bytes(16384 * 1000000) / 1000000,
            random_seed=1234, orbital_ordering="original", final_one_site=False,
            scratch=Path(lib.param.TMPDIR) / "original_sa_mps",
            checkpoint_dir=source / "dmrg_state/dmrg_checkpoint")
        mc.state_average_(WEIGHTS)
        try:
            mc.fcisolver.restore_checkpoint(reference["h1e"], reference["eri"], 22, 11,
                ecore=float(reference["ecore"]), nroots=12, max_memory=mol.max_memory)
            mc.fcisolver.driver.reorder_idx = None
            assert mc.fcisolver.kramers_adapter is None and not mc.fcisolver.final_one_site
            mc.mo_coeff, mc.mo_energy = reference["mo_coeff"], None
            mc.ci, mc.converged = list(mc.fcisolver.kets), True
            mc.e_tot, mc.e_cas = float(WEIGHTS @ reference["e_states"]), mc.fcisolver.e_cas
            density = sum(w * u.make_rdm1(mc.fcisolver, r) for r, w in enumerate(WEIGHTS))
            mo, eps = u.semicanonicalize(mc, mc.mo_coeff, density, 0)
            fock = mo.conj().T @ mc.get_fock(mo_coeff=mo, casdm1=density) @ mo
            nc, no = int(mc.ncore), int(mc.ncore + 22)
            defect = max(u._maximum_abs(fock[:nc, :nc] - np.diag(eps[:nc])),
                         u._maximum_abs(fock[no:, no:] - np.diag(eps[no:])))
            ms._gate(defect, u._maximum_abs(fock), 1e-10, 1e-10, "common external SA Fock")
            eris = ms.wick_eris_from_mc(mc, mo, max_memory=2000)
            core = eris.symmetry_diagnostics["electronic_core_energy"] + mol.energy_nuc()
            np.testing.assert_allclose(core, reference["ecore"], atol=1e-8, rtol=0)
            partition = dict(eris=eris, mo_coeff=mo, mo_energy=eps, sa_density=density,
                             ecore=core, external_fock_defect=defect)
        finally:
            mc.fcisolver.close()
    np.testing.assert_allclose(partition["eris"].get_h1eff("AA"), reference["h1e"], atol=1e-10, rtol=0)
    np.testing.assert_allclose(partition["eris"].get_phys("AAAA").transpose(0, 2, 1, 3),
                               reference["eri"], atol=1e-10, rtol=0)
    digest = hashlib.sha256((hsha + snapshot_hash).encode())
    for name in ("mo_coeff", "mo_energy", "sa_density"):
        digest.update(np.ascontiguousarray(partition[name]).tobytes())
    partition.update(fingerprint=digest.hexdigest(), hamiltonian_sha256=hsha,
                     source_snapshot_sha256=snapshot_hash,
                     sa_roots=list(range(12)), sa_weights=WEIGHTS.copy())
    write_pickle(path, partition)
    print("EXACT_SPECTRA_PARTITION_SAVED", path, flush=True)
    gc.collect()
    return partition


def reference_cache(directory, reference, threads):
    path = directory / "exact_reference_vectors.npz"
    hsha = str(reference["hamiltonian_sha256"].item())
    np.savez(directory / "reference_vectors.npz", h1e=reference["h1e"], eri=reference["eri"],
             hamiltonian_sha256=hsha)
    if path.exists() and (directory / "exact_reference.json").exists():
        report = json.loads((directory / "exact_reference.json").read_text())
        if report["hamiltonian_sha256"] != hsha or report["nroots"] != 12:
            raise ValueError("exact-CI cache does not match the requested reference")
        if not all(report["converged"]) or max(report["residual_norms"]) > 1e-10:
            raise ValueError("cached exact CI is not converged")
    else:
        converged, energies, vectors, actions, residuals = exact_reference(
            reference["h1e"], reference["eri"], None, nroots=12, threads=threads)
        if not np.all(converged) or not np.isfinite(residuals).all() or max(residuals) > 1e-10:
            raise RuntimeError("twelve-root exact CI did not converge")
        np.testing.assert_allclose(vectors.conj().T @ vectors, np.eye(12), atol=1e-10, rtol=0)
        temporary = path.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez(handle, vectors=vectors, actions=actions, active_energies=energies,
                     hamiltonian_sha256=hsha)
        temporary.replace(path)
        write_json(directory / "exact_reference.json", dict(
            hamiltonian_sha256=hsha, ncas=22, nelec=11, nroots=12, converged=converged,
            active_energies=energies, total_energies=energies + reference["ecore"],
            residual_norms=residuals, no_alpha_beta_mapping=True))
        del vectors, actions
        _active_action_links.cache_clear()
        gc.collect()
    with np.load(path, allow_pickle=False) as saved:
        if str(saved["hamiltonian_sha256"].item()) != hsha:
            raise ValueError("exact-CI array Hamiltonian mismatch")
        return saved["vectors"].copy(), saved["active_energies"].copy()


def spectrum(energies):
    values = np.asarray(energies, float)
    if values.shape != (12,) or not np.isfinite(values).all():
        raise ValueError("expected twelve finite state energies")
    centers = {name: float(values[list(roots)].mean()) for name, roots in MANIFOLDS.items()}
    return dict(energies=values, manifold_centroids_Eh=centers,
                spreads_cm_inverse={name: float(np.ptp(values[list(roots)]) * CONVERSION)
                                    for name, roots in MANIFOLDS.items()},
                excitation_cm_inverse={name: (value - centers["S1_2"]) * CONVERSION
                                       for name, value in centers.items()},
                d_splitting_cm_inverse=(centers["D3_2"] - centers["D5_2"]) * CONVERSION)


def main(directory, *, prepare_only=False):
    directory.mkdir(parents=True, exist_ok=True)
    source, symbol = directory.parent, directory.parent.parent.name
    if symbol not in ("Cu", "Ag", "Au"):
        raise ValueError("expected Cu/Ag/Au orbital checkpoint directory")
    threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "16"))
    lib.num_threads(threads)
    with np.load(source / "dmrg_state/reference.npz", allow_pickle=False) as saved:
        reference = {key: saved[key] for key in saved.files}
    ready = json.loads((source / "dmrg_state/READY.json").read_text())
    hsha = str(reference["hamiltonian_sha256"].item())
    expected = dict(symbol=symbol, ncas=22, nelec=11, nroots=12, M=2000,
                    orbital_kramers=True, cd_df=False, final_one_site=False, nuclear_model="point")
    if (ready["hamiltonian_sha256"] != hsha or not ready["converged"] or
            not ready["solver_converged"] or
            any(ready["settings"].get(key) != value for key, value in expected.items())):
        raise ValueError("original orbital checkpoint mismatch")
    np.testing.assert_array_equal(reference["weights"], WEIGHTS)
    settings = dict(symbol=symbol, source=str(source), hamiltonian_sha256=hsha,
                    ncas=22, nelec=11, nroots=12, sa_weights=WEIGHTS,
                    manifolds=MANIFOLDS, ansatz="ms_mr", shift=0.0,
                    metric_atol=1e-12, metric_rcond=1e-11, metric_refinement=True,
                    cd_df=False, nevpt2_kramers_reduction=False, frozen_common_sa_partition=True)
    write_json(directory / "settings.json", settings)
    partition = common_partition(directory, reference, threads)
    vectors, energies = reference_cache(directory, reference, threads)
    with (directory / "exact_reference_vectors.npz").open("rb") as handle:
        ci_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    if symbol == "Ag" and not (directory / "S1_2.json").exists():
        validated = source / "ms_nevpt2_audit"
        with np.load(validated / "exact_reference_vectors.npz", allow_pickle=False) as old:
            np.testing.assert_allclose(np.linalg.svd(old["vectors"].conj().T @ vectors[:, :2],
                                                    compute_uv=False), 1, atol=1e-10, rtol=0)
            np.testing.assert_allclose(old["active_energies"], energies[:2], atol=1e-9, rtol=0)
        result = json.loads((validated / "S1_2_exact.json").read_text())
        write_json(directory / "S1_2.json", dict(**result, reused_from=str(validated),
            reference_ci_sha256=ci_hash, model_roots=MANIFOLDS["S1_2"],
            partition_fingerprint=partition["fingerprint"]))
        print("EXACT_SPECTRA_AG_S_REUSED", flush=True)
    exact_mps.main(directory)
    metadata = json.loads((directory / "exact_mps/READY.json").read_text())
    if metadata["source_ci_sha256"] != ci_hash or len(metadata["roots"]) != 12:
        raise ValueError("native exact MPS cache mismatch")
    if prepare_only:
        print("EXACT_SPECTRA_REFERENCE_PREPARED", symbol, flush=True)
        return
    partial = dict(settings=settings, saved_DMRG_SCF=spectrum(reference["e_states"]),
                   exact_CASCI=spectrum(energies + partition["ecore"]), manifolds={})
    values = solve_manifolds(directory, partition, vectors, energies, metadata, ci_hash, partial, threads)
    for label, values in (("Saved-DMRG-SCF", reference["e_states"]),
                          ("Exact-CASCI", energies + partition["ecore"]), ("MS-FIC-NEVPT2", values)):
        print(f"{label} energy for each state", flush=True)
        for root, energy in enumerate(values):
            print(f"  State {root} weight {WEIGHTS[root]:.7g}  E = {energy:.14f}", flush=True)
    print("EXACT_SPECTRA_DONE", symbol, flush=True)


def solve_manifolds(directory, partition, vectors, energies, metadata, ci_hash, partial, threads):
    """Shared ordered-TRDM and MS-FIC workflow for supplied spinor integrals."""
    # Finish the smaller D manifold first; the physical grouping stays 2+6+4.
    for name in ("S1_2", "D3_2", "D5_2"):
        roots = MANIFOLDS[name]
        output = directory / f"{name}.json"
        if output.exists():
            result = json.loads(output.read_text())
            if (tuple(result["model_roots"]) != roots or
                    result["partition_fingerprint"] != partition["fingerprint"] or
                    result["reference_ci_sha256"] != ci_hash):
                raise ValueError("completed manifold belongs to another partition")
        else:
            work = directory / name
            work.mkdir(exist_ok=True)
            cache = work / "prepared.pkl"
            if cache.exists():
                with cache.open("rb") as handle:
                    prepared = pickle.load(handle)
                if (prepared.model_roots != roots or prepared.diagnostics["reference_ci_sha256"] != ci_hash or
                        prepared.diagnostics["partition_fingerprint"] != partition["fingerprint"]):
                    raise ValueError("prepared manifold inputs changed")
            else:
                scratch = Path(lib.param.TMPDIR) / f"exact_{name}_mps"
                scratch.mkdir(exist_ok=True)
                for root in metadata["roots"]:
                    for filename in root["files"]:
                        if Path(filename).name != filename:
                            raise ValueError("invalid native MPS checkpoint filename")
                        shutil.copy2(directory / "exact_mps" / filename, scratch / filename)
                driver = DMRGDriver(scratch=str(scratch), symm_type=SymmetryTypes.SGFCPX,
                                    stack_mem=block2_stack_bytes(500000 * 1000000), n_threads=threads)
                try:
                    driver.initialize_system(n_sites=22, n_elec=11, orb_sym=[0] * 22)
                    solver = SimpleNamespace(driver=driver,
                        kets=[driver.load_mps(root["tag"]) for root in metadata["roots"]],
                        ncas=22, nelecas=11, npdm_site_type=0, npdm_cutoff=1e-24)
                    def pair(bra, ket):
                        raw = directory / "S1_2_rdms"
                        if name == "S1_2" and (raw / "READY.json").exists():
                            from .save_s_rdms import load_saved_pair
                            return load_saved_pair(raw, ci_hash, bra, ket)
                        available = next(int(line.split()[1]) * 1024
                            for line in Path("/proc/meminfo").read_text().splitlines()
                            if line.startswith("MemAvailable:"))
                        if available < sum(22**(2*r) * 16 for r in range(1, 5)) + 128 * 2**30:
                            raise MemoryError("insufficient available RAM for ordered 4-TRDM")
                        print("EXACT_SPECTRA_TRDM_BEGIN", name, bra, ket, flush=True)
                        lower = directory / "all_model_rdms123"
                        if (lower / "READY.json").exists():
                            from .save_s_rdms import load_saved_pair
                            return (*load_saved_pair(lower, ci_hash, bra, ket),
                                    u.make_transition_rdm4(solver, bra, ket))
                        return make_transition_dm1234(solver, bra, ket)
                    overlap = vectors[:, roots].conj().T @ vectors[:, roots]
                    classes, active, audits = ms.build_msfic_classes(
                        partition["eris"], roots, pair, overlap, contraction_backend="pytblis",
                        transition_rdm_fallback_dir=directory / "temporary_rdms")
                    prepared = model_prepared(partition, roots, classes, active, overlap,
                                              audits, energies, ci_hash)
                    write_pickle(cache, prepared)
                    print("EXACT_SPECTRA_PREPARED", name, cache, flush=True)
                    del classes, active, audits
                finally:
                    driver.finalize()
            result_obj = ms.solve_msfic(prepared, ansatz="ms_mr", shift=0.0, metric_refinement=True)
            result = dict(**result_obj.__dict__, model_roots=roots,
                          reference_ci_sha256=ci_hash, partition_fingerprint=partition["fingerprint"])
            write_json(output, result)
            del result_obj, prepared
            gc.collect()
        partial["manifolds"][name] = result
        write_json(directory / "partial_results.json", partial)
        print("EXACT_SPECTRA_MANIFOLD_COMPLETE", name, result["energies"], flush=True)
    values = np.concatenate([partial["manifolds"][name]["energies"] for name in MANIFOLDS])
    partial.update(status="DONE", MS_FIC_NEVPT2=spectrum(values))
    write_json(directory / "results.json", partial)
    return values


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true",
                        help="save integrals and exact reference MPSs, without building TRDMs")
    args = parser.parse_args()
    main(args.directory.resolve(), prepare_only=args.prepare_only)
