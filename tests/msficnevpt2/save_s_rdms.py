# SPDX-License-Identifier: GPL-3.0-or-later
"""Retain raw TRDMs: S ranks 1--4 or all model ranks 1--3, without IC."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
from pyscf import lib
from pyblock2.driver.core import DMRGDriver, SymmetryTypes

from socutils.mrpt import make_transition_dm1234, x2cmsficnevpt2 as ms
from socutils.mrpt import nevpt2_utils as u
from .ag_audit import block2_stack_bytes, write_json


def load_saved_pair(directory, ci_hash, bra, ket):
    """Validate only provenance and array headers; never scan dense tensors."""
    inputs = json.loads((directory / "inputs.json").read_text())
    pairs = inputs.get("ordered_pairs", [[b, k] for b in (0, 1) for k in (0, 1)])
    nranks = inputs.get("nranks", 4)
    if inputs["source_ci_sha256"] != ci_hash or [bra, ket] not in pairs or nranks not in (3, 4):
        raise ValueError("saved TRDM reference mismatch")
    pair = directory / f"bra-{bra}_ket-{ket}"
    ready = json.loads((pair / "READY.json").read_text())
    format_ = "socutils.msfic.transition-rdms" + ("-123" if nranks == 3 else "")
    if (ready["format"] != format_ or
            (ready["bra"], ready["ket"], ready["ncas"]) != (bra, ket, inputs["ncas"])):
        raise ValueError("saved ordered TRDM metadata mismatch")
    arrays = []
    try:
        for rank in range(1, nranks + 1):
            dm = np.load(pair / f"dm{rank}.npy", mmap_mode="r", allow_pickle=False)
            arrays.append(dm)
            if (dm.shape != (inputs["ncas"],) * (2 * rank) or dm.dtype != np.complex128 or
                    list(dm.shape) != ready["shapes"][rank - 1] or
                    str(dm.dtype) != ready["dtypes"][rank - 1]):
                raise ValueError("saved raw TRDM shape or dtype mismatch")
    except Exception:
        for dm in arrays:
            dm._mmap.close()
        raise
    return tuple(arrays)


def save_pairs(solver, directory, inputs):
    directory.mkdir(exist_ok=True)
    manifest = directory / "inputs.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != inputs:
            raise ValueError("TRDM directory belongs to another exact reference")
    else:
        if any(directory.iterdir()):
            raise ValueError("refusing to overwrite an unidentified TRDM directory")
        write_json(manifest, inputs)
    ncas = inputs["ncas"]
    nranks = inputs.get("nranks", 4)
    if nranks not in (3, 4):
        raise ValueError("only raw ranks 1--3 or 1--4 are supported")
    pairs = inputs.get("ordered_pairs", [[b, k] for b in (0, 1) for k in (0, 1)])
    for bra, ket in pairs:
        if (directory / f"bra-{bra}_ket-{ket}/READY.json").exists():
            arrays = load_saved_pair(directory, inputs["source_ci_sha256"], bra, ket)
            for dm in arrays:
                dm._mmap.close()
            print("TRDM_REUSED", nranks, bra, ket, flush=True)
            continue
        available = next(int(line.split()[1]) * 1024
            for line in Path("/proc/meminfo").read_text().splitlines()
            if line.startswith("MemAvailable:"))
        if available < sum(ncas**(2*r) * 16 for r in range(1, nranks + 1)) + 128 * 2**30:
            raise MemoryError("insufficient available RAM for retained ordered TRDM")
        print("TRDM_BEGIN", nranks, bra, ket, flush=True)
        if nranks == 4:
            lower = getattr(solver, "lower_directory", None)
            def provider(b, k):
                if lower is not None and (lower / "READY.json").exists():
                    return (*load_saved_pair(lower, inputs["source_ci_sha256"], b, k),
                            u.make_transition_rdm4(solver, b, k))
                return make_transition_dm1234(solver, b, k)
            with ms._disk_transition_pair(provider, bra, ket, ncas, directory, keep=True):
                pass
        else:
            pdms = u._make_transition_dm123(solver, bra, ket)
            pair = directory / f"bra-{bra}_ket-{ket}"
            pair.mkdir(exist_ok=True)
            for rank, dm in enumerate(pdms, 1):
                if dm.shape != (ncas,) * (2 * rank) or dm.dtype != np.complex128:
                    raise ValueError("raw lower TRDM shape or dtype mismatch")
                path = pair / f"dm{rank}.npy"
                temporary = path.with_suffix(".npy.tmp")
                with temporary.open("wb") as handle:
                    np.save(handle, dm, allow_pickle=False)
                temporary.replace(path)
            write_json(pair / "READY.json", dict(format="socutils.msfic.transition-rdms-123",
                bra=bra, ket=ket, ncas=ncas, shapes=[list(dm.shape) for dm in pdms],
                dtypes=[str(dm.dtype) for dm in pdms]))
            del pdms, dm
        gc.collect()
        print("TRDM_SAVED", nranks, bra, ket, flush=True)
    write_json(directory / "READY.json", {**inputs, "status": "DONE", "ordered_pairs": pairs,
                                          "tensor_contraction": False})


def main(directory, *, lower_only=False):
    threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "16"))
    lib.num_threads(threads)
    metadata = json.loads((directory / "exact_mps/READY.json").read_text())
    residual_limit = float(metadata.get("residual_limit", 1e-10))
    if not np.isfinite(residual_limit) or residual_limit <= 0:
        raise ValueError("invalid reference residual limit")
    with (directory / "exact_reference_vectors.npz").open("rb") as handle:
        ci_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    if (metadata["source_ci_sha256"] != ci_hash or
            (metadata["ncas"], metadata["nelec"]) != (22, 11) or
            max(metadata["residual_norms"]) > residual_limit or
            [root["root"] for root in metadata["roots"]] != list(range(12))):
        raise ValueError("exact native twelve-root MPS checkpoint mismatch")
    inputs = dict(source_ci_sha256=ci_hash, hamiltonian_sha256=metadata["hamiltonian_sha256"],
                  ncas=22, nelec=11, model_roots=[0, 1], raw_sgf=True)
    if lower_only:
        from .exact_spectra import MANIFOLDS
        settings = json.loads((directory / "settings.json").read_text())
        if (settings["hamiltonian_sha256"] != metadata["hamiltonian_sha256"] or
                settings["manifolds"] != {key: list(roots) for key, roots in MANIFOLDS.items()}):
            raise ValueError("model settings Hamiltonian or manifolds mismatch")
        inputs.update(nranks=3, model_roots=list(range(12)), ordered_pairs=[
            [bra, ket] for roots in settings["manifolds"].values() for bra in roots for ket in roots])
    roots = metadata["roots"] if lower_only else metadata["roots"][:2]
    scratch = Path(lib.param.TMPDIR) / "saved_s_rdms_mps"
    scratch.mkdir(exist_ok=True)
    for root in roots:
        for filename in root["files"]:
            if Path(filename).name != filename:
                raise ValueError("invalid native MPS checkpoint filename")
            shutil.copy2(directory / "exact_mps" / filename, scratch / filename)
    driver = DMRGDriver(scratch=str(scratch), symm_type=SymmetryTypes.SGFCPX,
                        stack_mem=block2_stack_bytes((128000 if lower_only else 500000) * 1000000),
                        n_threads=threads)
    try:
        driver.initialize_system(n_sites=22, n_elec=11, orb_sym=[0] * 22)
        solver = SimpleNamespace(driver=driver,
            kets=[driver.load_mps(root["tag"]) for root in roots],
            ncas=22, nelecas=11, npdm_site_type=0, npdm_cutoff=1e-24,
            lower_directory=directory / "all_model_rdms123")
        output = directory / ("all_model_rdms123" if lower_only else "S1_2_rdms")
        save_pairs(solver, output, inputs)
    finally:
        driver.finalize()
    print("TRDM_DONE", output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--lower-only", action="store_true",
                        help="save all model ranks 1--3, without computing any 4-TRDM")
    args = parser.parse_args()
    main(args.directory.resolve(), lower_only=args.lower_only)
