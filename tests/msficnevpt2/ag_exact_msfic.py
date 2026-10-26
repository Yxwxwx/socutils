# SPDX-License-Identifier: GPL-3.0-or-later
"""Repeat Ag MS-FIC on exact reference MPSs with the original SA partition.

Only model-reference accuracy changes. Production contraction and solve APIs
are reused unchanged; control TRDMs occupy one temporary ordered pair while
the original-reference raw TRDMs remain untouched in their persistent cache.
"""
import argparse
from dataclasses import replace
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
from types import SimpleNamespace

import numpy as np
from pyscf import lib
from pyblock2.driver.core import DMRGDriver, SymmetryTypes

from socutils.mrpt import make_transition_dm1234
from socutils.mrpt import x2cmsficnevpt2 as ms
from .ag_audit import write_json


def frozen_control(baseline, classes, active_reference, overlap, audits):
    """Keep orbitals, ERIs, SA density and energies; replace only model data."""
    n = len(baseline.model_roots)
    core_matrix = baseline.reference - baseline.active_reference
    core = float(np.trace(core_matrix).real / n)
    np.testing.assert_allclose(core_matrix, core * np.eye(n), atol=1e-8, rtol=0)
    diagnostics = {
        **audits,
        "sa_roots": baseline.diagnostics["sa_roots"],
        "sa_weights": baseline.diagnostics["sa_weights"],
        "external_fock_defect": baseline.diagnostics["external_fock_defect"],
        "baseline_fingerprint": baseline.diagnostics["fingerprint"],
        "reference_accuracy_control": "fixed-Hamiltonian exact-CI native MPS",
        "frozen_common_sa_partition": True,
    }
    return replace(baseline, classes=classes, active_reference=active_reference,
                   overlap=overlap, reference=active_reference + core * overlap,
                   diagnostics=diagnostics)


def correction_spread(correction):
    return float(np.ptp(np.linalg.eigvalsh(correction)) * 219474.6313705)


def main(directory):
    threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    lib.num_threads(threads)
    with (directory / "prepared.pkl").open("rb") as handle:
        baseline = pickle.load(handle)
    if tuple(baseline.model_roots) != (0, 1) or tuple(baseline.diagnostics["sa_roots"]) != tuple(range(12)):
        raise ValueError("exact-reference control requires the original two-model/twelve-SA preparation")
    native_directory = directory / "exact_mps"
    metadata = json.loads((native_directory / "READY.json").read_text())
    ci_path = directory / "exact_reference_vectors.npz"
    with ci_path.open("rb") as handle:
        ci_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    if ci_hash != metadata["source_ci_sha256"] or (metadata["ncas"], metadata["nelec"]) != (22, 11):
        raise ValueError("native MPS cache does not match the exact-CI control")
    with np.load(ci_path, allow_pickle=False) as saved:
        vectors, energies = saved["vectors"].copy(), saved["active_energies"].copy()
        fingerprint = str(saved["hamiltonian_sha256"].item())
    with np.load(directory / "reference_vectors.npz", allow_pickle=False) as saved:
        if fingerprint != str(saved["hamiltonian_sha256"].item()) or fingerprint != metadata["hamiltonian_sha256"]:
            raise ValueError("exact/baseline Hamiltonian mismatch")
        np.testing.assert_allclose(baseline.eris.get_h1eff("AA"), saved["h1e"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(baseline.eris.get_phys("AAAA").transpose(0, 2, 1, 3), saved["eri"], atol=1e-10, rtol=0)
    if max(metadata["residual_norms"]) > 1e-10:
        raise ValueError("exact native MPS residual check failed")
    overlap = vectors.conj().T @ vectors
    np.testing.assert_allclose(overlap, np.eye(2), atol=1e-10, rtol=0)
    cache = directory / "prepared_exact.pkl"
    if cache.exists():
        with cache.open("rb") as handle:
            prepared = pickle.load(handle)
        if prepared.diagnostics["baseline_fingerprint"] != baseline.diagnostics["fingerprint"]:
            raise ValueError("prepared control belongs to another baseline partition")
        if prepared.diagnostics["reference_ci_sha256"] != ci_hash:
            raise ValueError("prepared control belongs to another exact reference")
    else:
        scratch = Path(lib.param.TMPDIR) / "exact_control_mps"
        scratch.mkdir(exist_ok=True)
        for root in metadata["roots"]:
            for filename in root["files"]:
                if Path(filename).name != filename:
                    raise ValueError("invalid native checkpoint filename")
                shutil.copy2(native_directory / filename, scratch / filename)
        driver = DMRGDriver(scratch=str(scratch), symm_type=SymmetryTypes.SGFCPX,
                            stack_mem=500000 * 1000000, n_threads=threads)
        try:
            driver.initialize_system(n_sites=22, n_elec=11, orb_sym=[0] * 22)
            kets = [driver.load_mps(root["tag"]) for root in metadata["roots"]]
            solver = SimpleNamespace(driver=driver, kets=kets, ncas=22, nelecas=11,
                                     npdm_site_type=0, npdm_cutoff=1e-24)

            def pair(bra, ket):
                available = next(int(line.split()[1]) * 1024
                                 for line in Path("/proc/meminfo").read_text().splitlines()
                                 if line.startswith("MemAvailable:"))
                if available < sum(22**(2*r) * 16 for r in range(1, 5)) + 128 * 2**30:
                    raise MemoryError("insufficient available RAM for exact-control ordered 4-TRDM")
                print("AG_EXACT_CONTROL_TRDM_BEGIN", bra, ket, flush=True)
                return make_transition_dm1234(solver, bra, ket)

            classes, active_reference, audits = ms.build_msfic_classes(
                baseline.eris, (0, 1), pair, overlap,
                contraction_backend="pytblis",
                transition_rdm_fallback_dir=directory / "exact_control_temporary_rdms")
            np.testing.assert_allclose(active_reference, np.diag(energies), atol=1e-9, rtol=0)
            prepared = frozen_control(baseline, classes, active_reference, overlap, audits)
            prepared.diagnostics["reference_ci_sha256"] = ci_hash
            temporary = cache.with_suffix(".pkl.tmp")
            with temporary.open("wb") as handle:
                pickle.dump(prepared, handle, protocol=pickle.HIGHEST_PROTOCOL)
            temporary.replace(cache)
            print("AG_EXACT_CONTROL_PREPARED", cache, flush=True)
        finally:
            driver.finalize()
    del vectors, baseline
    gc.collect()
    result = ms.solve_msfic(prepared, ansatz="ms_mr", shift=0.0, metric_refinement=True)
    write_json(directory / "S1_2_exact.json", result.__dict__)
    original = json.loads((directory / "S1_2.json").read_text())
    original_r = np.asarray(original["class_corrections"]["r"])
    original_r = original_r[..., 0] + 1j * original_r[..., 1]
    write_json(directory / "reference_accuracy_comparison.json", dict(
        frozen_common_sa_partition=True, shift=0.0,
        original_r_spread_cm_inverse=correction_spread(original_r),
        exact_r_spread_cm_inverse=correction_spread(result.class_corrections["r"]),
        original_total_spread_cm_inverse=float(np.ptp(original["energies"]) * 219474.6313705),
        exact_total_spread_cm_inverse=result.spread * 219474.6313705,
        original_r_diagnostics=original["diagnostics"]["classes"]["r"],
        exact_r_diagnostics=result.diagnostics["classes"]["r"],
        exact_reference_residual_norms=metadata["residual_norms"]))
    print("AG_EXACT_CONTROL_COMPLETE", "energies", result.energies, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    main(parser.parse_args().directory.resolve())
