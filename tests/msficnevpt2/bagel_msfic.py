# SPDX-License-Identifier: GPL-3.0-or-later
"""MS-FIC using the unchanged native BAGEL 4C Hamiltonian and reference."""
import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path

import numpy as np
from pyscf import lib

from socutils.mrpt import nevpt2_utils as u, x2cmsficnevpt2 as ms
from .ag_audit import write_json
from .bagel_reference import MOIntegrals, read_binary
from .exact_spectra import MANIFOLDS, WEIGHTS, spectrum, write_pickle, solve_manifolds


def mean_field(integrals, indices, density):
    """Contract a general-spinor density without expanding the full MO ERI."""
    all_mo = np.arange(2 * integrals.n)
    out = np.zeros((len(all_mo), len(all_mo)), complex)
    for start in range(0, len(all_mo), 16):
        rows = all_mo[start:start + 16]
        direct = integrals.physicist_block((rows, indices, all_mo, indices))
        out[rows] = np.einsum("paqb,ab->pq", direct, density, optimize=True)
        del direct
        exchange = integrals.physicist_block((rows, indices, indices, all_mo))
        out[rows] -= np.einsum("pabq,ab->pq", exchange, density, optimize=True)
    return out


def mo_partition(integrals, ncore, ncas, density):
    """Core-fold and semicanonicalize with the active basis held fixed."""
    nmo = 2 * integrals.n
    if not (0 <= ncore < ncore + ncas <= nmo) or density.shape != (ncas, ncas):
        raise ValueError("invalid native MO partition or active density")
    spaces = dict(I=np.arange(ncore), A=np.arange(ncore, ncore + ncas),
                  E=np.arange(ncore + ncas, nmo))
    hbare = integrals.one_body()
    heff = hbare + mean_field(integrals, spaces["I"], np.eye(ncore))
    heff, h_error = ms._hermitian(heff, 1e-10, 1e-10, "native 4C core-dressed h1")
    core = float(np.real(.5 * np.trace((hbare + heff)[:ncore, :ncore])))
    fock = heff + mean_field(integrals, spaces["A"], density)
    fock, f_error = ms._hermitian(fock, 1e-10, 1e-10, "native 4C SA Fock")
    rotation = np.eye(nmo, dtype=complex)
    eps = fock.diagonal().real.copy()
    for label in ("I", "E"):
        block = np.ix_(spaces[label], spaces[label])
        eps[spaces[label]], rotation[block] = np.linalg.eigh(fock[block])
    rotated_fock = rotation.conj().T @ fock @ rotation
    defect = max(u._maximum_abs(rotated_fock[np.ix_(space, space)] - np.diag(eps[space]))
                 for space in (spaces["I"], spaces["E"]))
    diagnostics = dict(storage="native_bagel_wick_blocks", full_mo_eri_built=False,
                       electronic_core_energy=core, h1_hermiticity_error=h_error,
                       fock_hermiticity_error=f_error)
    eris = u._WickERIBlocks(ncore, ncas, nmo - ncore - ncas,
        {key: heff[np.ix_(*(spaces[x] for x in key))] for key in u._H1_KEYS},
        {key: integrals.physicist_block(tuple(spaces[x] for x in key)) for key in u._W_KEYS},
        diagnostics)
    return dict(eris=u._rotate_wick_eris(eris, rotation), rotation=rotation,
                mo_energy=eps, sa_density=density, electronic_core_energy=core,
                external_fock_defect=defect)


def common_partition(directory):
    work = directory / "reference"
    full, active = MOIntegrals(directory / "native_full"), MOIntegrals(directory / "native_active")
    if full.meta["mode"] != "full" or active.meta["mode"] != "active":
        raise ValueError("expected completed native full and active MO exports")
    for key in ("nclosed_pairs", "nactive_pairs", "nvirtual_pairs", "npositive_spinors",
                "nroots", "coefficient_rows", "gaunt", "breit", "nuclear_repulsion"):
        if full.meta[key] != active.meta[key]:
            raise ValueError(f"native full/active metadata mismatch: {key}")
    if (full.meta["nactive_pairs"], full.meta["nroots"]) != (11, 12):
        raise ValueError("expected Cu CAS(11e,22 spinors), twelve roots")
    if full.meta["core_energy"] != 0 or 2 * full.n != full.meta["npositive_spinors"]:
        raise ValueError("full export must contain all positive-energy MOs and bare h1")
    with (work / "exact_reference_vectors.npz").open("rb") as handle:
        ci_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    native_files = [directory / "native_full/export.json",
                    directory / "native_active/coeff.bin", directory / "native_active/sa_dm1.bin"]
    native_files += sorted((directory / "native_full").glob("*.bin"))
    source = [(str(p.relative_to(directory)), p.stat().st_size, p.stat().st_mtime_ns)
              for p in native_files]
    source_hash = hashlib.sha256((json.dumps(source) + ci_hash).encode()).hexdigest()
    cache = work / "partition.pkl"
    if cache.exists():
        with cache.open("rb") as handle:
            partition = pickle.load(handle)
        if partition["source_signature"] != source_hash:
            raise ValueError("native 4C partition inputs changed")
        return partition
    full.load_into_memory()
    ncore, ncas = 2 * full.meta["nclosed_pairs"], 2 * full.meta["nactive_pairs"]
    order = np.arange(ncas).reshape(2, ncas // 2).T.ravel()
    density = np.array(read_binary(directory / "native_active/sa_dm1.bin", (ncas, ncas)))[np.ix_(order, order)]
    # Cheap 1-RDM comparison also verifies that the native archive used twelve equal weights.
    from .save_s_rdms import load_saved_pair
    averaged = np.zeros_like(density)
    for root, weight in enumerate(WEIGHTS):
        arrays = load_saved_pair(work / "all_model_rdms123", ci_hash, root, root)
        try:
            averaged += weight * arrays[0]
        finally:
            for dm in arrays:
                dm._mmap.close()
    np.testing.assert_allclose(density, averaged, atol=1e-10, rtol=0)
    partition = mo_partition(full, ncore, ncas, density)
    eris = partition["eris"]
    np.testing.assert_allclose(eris.get_h1eff("AA"), active.one_body(), atol=1e-8, rtol=0)
    np.testing.assert_allclose(eris.get_phys("AAAA"),
        active.physicist_block((np.arange(ncas),) * 4), atol=1e-10, rtol=0)
    np.testing.assert_allclose(partition["electronic_core_energy"], active.meta["core_energy"],
                               atol=1e-8, rtol=0)
    coeff = read_binary(directory / "native_active/coeff.bin",
                        (full.meta["coefficient_rows"], full.meta["npositive_spinors"]))
    partition["mo_coeff"] = coeff @ partition.pop("rotation")
    partition["ecore"] = partition["electronic_core_energy"] + full.meta["nuclear_repulsion"]
    reference = json.loads((work / "exact_reference.json").read_text())
    digest = hashlib.sha256(source_hash.encode())
    for key in ("mo_coeff", "mo_energy", "sa_density"):
        digest.update(np.ascontiguousarray(partition[key]).tobytes())
    partition.update(source_signature=source_hash, fingerprint=digest.hexdigest(),
                     hamiltonian_sha256=reference["hamiltonian_sha256"],
                     sa_roots=list(range(12)), sa_weights=WEIGHTS.copy())
    write_pickle(cache, partition)
    write_json(work / "partition_audit.json", dict(
        source_signature=source_hash, fingerprint=partition["fingerprint"],
        core_energy=partition["ecore"], external_fock_defect=partition["external_fock_defect"],
        active_integrals_match=True, native_sa_density_match=True,
        hamiltonian="native BAGEL 4C Dirac-Coulomb-Breit RI/DF",
        ncore=ncore, ncas=ncas, nmo=2 * full.n, integral_bytes=eris.nbytes))
    print("BAGEL_MS_PARTITION_SAVED", cache, flush=True)
    return partition


def main(directory, *, prepare_only=False):
    threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "16"))
    lib.num_threads(threads)
    partition = common_partition(directory)
    if prepare_only:
        return
    work = directory / "reference"
    metadata = json.loads((work / "exact_mps/READY.json").read_text())
    with (work / "exact_reference_vectors.npz").open("rb") as handle:
        ci_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    if (metadata["source_ci_sha256"] != ci_hash or
            metadata["hamiltonian_sha256"] != partition["hamiltonian_sha256"] or
            [root["root"] for root in metadata["roots"]] != list(range(12))):
        raise ValueError("native BAGEL MPS/partition reference mismatch")
    if max(metadata["residual_norms"]) > metadata["residual_limit"]:
        raise ValueError("native BAGEL CI exceeds its original residual criterion")
    with np.load(work / "exact_reference_vectors.npz", allow_pickle=False) as saved:
        vectors, energies = saved["vectors"], saved["active_energies"]
    total = np.asarray(read_binary(directory / "native_active/energies.bin", (12,), np.float64))
    np.testing.assert_allclose(total, energies + partition["ecore"], atol=1e-8, rtol=0)
    settings = dict(symbol="Cu", hamiltonian="native BAGEL 4C Dirac-Coulomb-Breit RI/DF",
                    sa_weights=WEIGHTS, manifolds=MANIFOLDS, ansatz="ms_mr", shift=0.0,
                    nevpt2_kramers_reduction=False, partition_fingerprint=partition["fingerprint"])
    partial = dict(settings=settings, BAGEL_4C_CASSCF=spectrum(total), manifolds={})
    values = solve_manifolds(work, partition, vectors, energies, metadata, ci_hash, partial, threads)
    for label, result in (("BAGEL-4C-CASSCF", total), ("MS-FIC-NEVPT2", values)):
        print(f"{label} energy for each state", flush=True)
        for root, energy in enumerate(result):
            print(f"  State {root} weight {WEIGHTS[root]:.7g}  E = {energy:.14f}", flush=True)
    print("BAGEL_MS_SPECTRA_DONE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    main(args.directory.resolve(), prepare_only=args.prepare_only)
