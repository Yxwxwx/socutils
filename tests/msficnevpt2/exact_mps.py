# SPDX-License-Identifier: GPL-3.0-or-later
"""Import the exact Ag CI control with native Block2 construction/algebra.

The native determinant importer builds dense prefix tensors before any
canonicalization. Batches bound that intermediate storage. Balanced sums
and QR use existing pyblock2 algebra, without a singular-value cutoff, bond
cap, eigenvalue solve, one-site DMRG sweep, or symmetry projection.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
from pyscf import lib
from pyscf.fci import cistring, fci_dhf_slow
from pyblock2.algebra.io import MPSTools
from pyblock2.driver.core import DMRGDriver, SymmetryTypes

from socutils.mrpt.nevpt2_mps_response import _release_response_mps, _response_tag
from tests.nevpt2_mps_response.f_atom import active_action_batch, _active_action_links
from .ag_audit import block2_stack_bytes, fixed_n_coefficients, write_json


def import_coefficients(driver, vector, nelec, *, tag, batch_size=2048):
    """Exact batched use of get_mps_from_csf_coefficients, in fixed-N order."""
    ncas = driver.n_sites
    if not 0 <= nelec <= ncas <= 24 or batch_size < 1:
        raise ValueError("CI import needs <=24 spinors and a positive batch size")
    strings = cistring.make_strings(range(ncas), nelec)
    vector = np.asarray(vector, dtype=np.complex128)
    if vector.shape != strings.shape or not np.isfinite(vector).all() or not np.any(vector):
        raise ValueError("invalid fixed-N CI coefficients")
    if driver.reorder_idx is not None:
        raise ValueError("coefficient import requires original orbital order")
    levels = []
    for start in range(0, len(strings), batch_size):
        values = vector[start:start + batch_size]
        if not np.any(values):
            continue
        dets = ((strings[start:start + batch_size, None] >> np.arange(ncas)) & 1).astype(np.uint8)
        native = driver.get_mps_from_csf_coefficients(
            dets, values, tag=_response_tag(), dot=1, full_fci=False, iprint=0)
        try:
            driver.align_mps_center(native, ref=0)
            part = MPSTools.from_block2(native)
        finally:
            _release_response_mps(driver, native)
        # Native/Python QR changes representation only; no Schmidt values drop.
        level = 0
        while level < len(levels) and levels[level] is not None:
            part = levels[level] + part
            levels[level] = None
            part.canonicalize(ncas - 1)
            part.canonicalize(0)
            level += 1
        if level == len(levels):
            levels.append(part)
        else:
            levels[level] = part
        if start % (batch_size * 32) == 0:
            print("EXACT_MPS_BATCH", tag, start, len(strings), flush=True)
    result = None
    for part in levels:
        if part is not None:
            result = part if result is None else result + part
            result.canonicalize(ncas - 1)
            result.canonicalize(0)
    return MPSTools.to_block2(result, driver.basis, center=0, tag=tag)


def copy_checkpoint(driver, mps, target):
    """Copy only this newly constructed tag; never touch the original MPS."""
    tag = mps.info.tag
    mps.info.save_data(str(Path(driver.scratch) / f"{tag}-mps_info.bin"))
    mps.save_data()
    patterns = (f"{tag}-mps_info.bin", f"F.MPS.{tag}.*",
                f"F.MPS.INFO.{tag}.LEFT.*", f"F.MPS.INFO.{tag}.RIGHT.*")
    copied = []
    for pattern in patterns:
        for source in Path(driver.scratch).glob(pattern):
            if source.is_symlink() or not source.is_file():
                raise ValueError("unexpected native MPS checkpoint path")
            shutil.copy2(source, target / source.name)
            copied.append(source.name)
    if f"{tag}-mps_info.bin" not in copied:
        raise RuntimeError("native MPS info was not saved")
    return copied


def main(directory, *, residual_limit=1e-10):
    residual_limit = float(residual_limit)
    if not np.isfinite(residual_limit) or residual_limit <= 0:
        raise ValueError("positive finite CI residual limit required")
    source = directory / "exact_reference_vectors.npz"
    with source.open("rb") as handle:
        source_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    metadata = json.loads((directory / "exact_reference.json").read_text())
    if not all(metadata["converged"]) or max(metadata["residual_norms"]) > residual_limit:
        raise ValueError("the independent exact-CI control is not converged")
    with np.load(source, allow_pickle=False) as data:
        vectors, energies = data["vectors"].copy(), data["active_energies"].copy()
        fingerprint = str(data["hamiltonian_sha256"].item())
    with np.load(directory / "reference_vectors.npz", allow_pickle=False) as data:
        if fingerprint != str(data["hamiltonian_sha256"].item()):
            raise ValueError("exact/reference Hamiltonian mismatch")
        h1e, eri = data["h1e"].copy(), data["eri"].copy()
    output = directory / "exact_mps"
    output.mkdir(exist_ok=True)
    ready = output / "READY.json"
    if ready.exists():
        if json.loads(ready.read_text())["source_ci_sha256"] != source_hash:
            raise ValueError("existing MPS cache belongs to different CI vectors")
        print("EXACT_MPS_CACHE_REUSED", output, flush=True)
        return
    threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    lib.num_threads(threads)
    scratch = Path(lib.param.TMPDIR) / "ag_exact_mps"
    scratch.mkdir(exist_ok=True)
    driver = DMRGDriver(scratch=str(scratch), stack_mem=block2_stack_bytes(16 << 30),
                        symm_type=SymmetryTypes.SGFCPX, n_threads=threads)
    started = time.monotonic()
    try:
        driver.initialize_system(n_sites=22, n_elec=11, orb_sym=[0] * 22)
        returned, roots = [], []
        for root, vector in enumerate(vectors.T):
            tag = f"EXACT_ROOT_{root}"
            mps = import_coefficients(driver, vector, 11, tag=tag)
            actual = fixed_n_coefficients(MPSTools.from_block2(mps), 11)
            error = float(np.linalg.norm(actual - vector))
            if error > 1e-10:
                raise RuntimeError(f"CI->MPS coefficient roundtrip defect {error:.3e}")
            returned.append(actual)
            driver.adjust_mps(mps, dot=2)
            files = copy_checkpoint(driver, mps, output)
            roots.append(dict(root=root, tag=tag, coefficient_error=error,
                              max_bond_dimension=int(mps.info.get_max_bond_dimension()), files=files))
            print("EXACT_MPS_ROOT_VERIFIED", root, error, flush=True)
        returned = np.column_stack(returned)
        absorbed = fci_dhf_slow.absorb_h1e(h1e, eri, 22, 11, 0.5)
        actions = active_action_batch(absorbed, returned.T, 22, 11).T
        residuals = np.linalg.norm(actions - returned * energies, axis=0)
        gram = returned.conj().T @ returned
        np.testing.assert_allclose(gram, np.eye(vectors.shape[1]), atol=1e-10, rtol=0)
        if max(residuals) > residual_limit:
            raise RuntimeError(f"imported MPS global residuals too large: {residuals}")
        # Verify that the persistent native files, not only RAM tensors, reload.
        driver.scratch = str(output)
        strings = cistring.make_strings(range(22), 11)
        for root, vector in enumerate(vectors.T):
            loaded = driver.load_mps(roots[root]["tag"])
            sample = np.argsort(abs(vector))[-32:]
            dets = ((strings[sample, None] >> np.arange(22)) & 1).astype(np.uint8)
            native = driver.get_csf_coefficients(loaded, cutoff=0, given_dets=dets,
                                                  max_print=0, fci_conv=True, iprint=0)[1]
            np.testing.assert_allclose(native, vector[sample], atol=1e-11, rtol=0)
        write_json(ready, dict(source_ci_sha256=source_hash, hamiltonian_sha256=fingerprint,
                              ncas=22, nelec=11, roots=roots, residual_norms=residuals,
                              residual_limit=residual_limit,
                              gram=gram, no_schmidt_truncation=True, no_dmrg_sweeps=True,
                              wall_seconds=time.monotonic() - started))
        print("EXACT_MPS_CACHE_READY", output, residuals, flush=True)
    finally:
        _active_action_links.cache_clear()
        driver.finalize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    main(parser.parse_args().directory.resolve())
