# SPDX-License-Identifier: GPL-3.0-or-later
"""Translate native BAGEL storage, preserving its Hamiltonian and CI."""
import argparse
import hashlib
import itertools
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from pyscf import lib
from pyscf.fci import cistring, fci_dhf_slow
from pyblock2.driver.core import DMRGDriver, SymmetryTypes
from socutils.mrpt import nevpt2_utils as u

from .ag_audit import block2_stack_bytes, write_json
from .cu_fci_casscf import save_arrays
from .exact_mps import main as export_mps
from .exact_spectra import MANIFOLDS
from tests.nevpt2_mps_response.f_atom import active_action_batch, _active_action_links


def read_binary(path, shape, dtype=np.complex128):
    expected = int(np.prod(shape)) * np.dtype(dtype).itemsize
    if path.stat().st_size != expected:
        raise ValueError(f"BAGEL binary size mismatch: {path}")
    return np.memmap(path, mode="r", dtype=dtype, shape=shape, order="F")


def tag_bits(tag, rank):
    return tuple((tag >> i) & 1 for i in range(rank - 1, -1, -1))


class MOIntegrals:
    """Native physicist blocks; only requested spinor indices are expanded."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.meta = json.loads((self.directory / "export.json").read_text())
        self.n = int(self.meta["integral_pairs"])
        self.h = {tag_bits(t, 2): read_binary(self.directory / f"h1-{t}.bin", (self.n,)*2)
                  for t in self.meta["one_electron_tags"]}
        self.g = {tag_bits(t, 4): read_binary(self.directory / f"g2-{t}.bin", (self.n,)*4)
                  for t in self.meta["two_electron_tags"]}

    def load_into_memory(self):
        """Read native blocks sequentially before many sparse NFS selections."""
        for tag, mapped in self.g.items():
            block = np.array(mapped, copy=True, order="F")
            if isinstance(mapped, np.memmap):
                mapped._mmap.close()
            self.g[tag] = block
            print("BAGEL_MO_BLOCK_LOADED", tag, block.nbytes, flush=True)

    def one_body(self):
        out = np.empty((2*self.n,)*2, complex)
        if len(self.h) != 4:
            raise ValueError("incomplete BAGEL one-electron blocks")
        for bits, block in self.h.items():
            out[bits[0]::2, bits[1]::2] = block
        return out

    def physicist_block(self, indices):
        indices = tuple(np.asarray(i, dtype=int) for i in indices)
        if len(indices) != 4 or any(np.any((i < 0) | (i >= 2*self.n)) for i in indices):
            raise ValueError("invalid positive-energy MO indices")
        out = np.empty(tuple(len(i) for i in indices), complex)
        # Electron exchange and Hermiticity fill native omitted integral blocks.
        permutations = (((0, 1, 2, 3), False), ((1, 0, 3, 2), False),
                        ((2, 3, 0, 1), True), ((3, 2, 1, 0), True))
        for bits in itertools.product(range(2), repeat=4):
            selected = [np.flatnonzero(i % 2 == b) for i, b in zip(indices, bits)]
            if any(len(i) == 0 for i in selected):
                continue
            orbital = [i[s] // 2 for i, s in zip(indices, selected)]
            for perm, conjugate in permutations:
                source = tuple(bits[i] for i in perm)
                if source in self.g:
                    block = self.g[source][np.ix_(*(orbital[i] for i in perm))]
                    block = block.transpose(np.argsort(perm))
                    out[np.ix_(*selected)] = block.conj() if conjugate else block
                    break
            else:
                raise ValueError(f"BAGEL omitted block cannot be reconstructed: {bits}")
        return out


def ci_vectors(directory, metadata):
    n = int(metadata["nactive_pairs"])
    nelec = {s["na"] + s["nb"] for s in metadata["sectors"]}
    if len(nelec) != 1:
        raise ValueError("BAGEL CI sectors have different electron numbers")
    nelec = nelec.pop()
    strings = cistring.make_strings(range(2*n), nelec)
    vectors = np.zeros((len(strings), metadata["nroots"]), complex)
    seen = np.zeros(len(strings), bool)
    for sector in metadata["sectors"]:
        stem = directory / sector["stem"]
        a = read_binary(Path(str(stem) + ".alpha.bin"), (sector["lena"],), np.uint64)
        b = read_binary(Path(str(stem) + ".beta.bin"), (sector["lenb"],), np.uint64)
        values = read_binary(Path(str(stem) + ".ci.bin"),
                             (len(b), len(a), metadata["nroots"]))
        aa, bb = np.meshgrid(a, b, indexing="ij")
        aa, bb = aa.ravel(), bb.ravel()
        bits, crossings = np.zeros(len(aa), np.int64), np.zeros(len(aa), np.int64)
        # BAGEL stores the two Kramers labels in blocks; SGF uses paired sites.
        for p in range(n):
            bits |= ((aa >> p) & 1).astype(np.int64) << (2*p)
            bits |= ((bb >> p) & 1).astype(np.int64) << (2*p+1)
            for q in range(p):
                crossings += (((aa >> p) & 1) * ((bb >> q) & 1)).astype(np.int64)
        positions = np.searchsorted(strings, bits)
        np.testing.assert_array_equal(strings[positions], bits)
        if np.any(seen[positions]) or len(np.unique(positions)) != len(positions):
            raise ValueError("duplicate BAGEL determinants")
        seen[positions] = True
        vectors[positions] = values.transpose(1, 0, 2).reshape(len(bits), -1) * (1-2*(crossings % 2))[:, None]
    if not seen.all():
        raise ValueError("BAGEL export does not span the full fixed-N space")
    return vectors, nelec


def audit_transition_rdms(directory, work):
    metadata = json.loads((work / "exact_mps/READY.json").read_text())
    scratch = Path(lib.param.TMPDIR) / "bagel_rdm_audit"
    scratch.mkdir(exist_ok=True)
    for root in metadata["roots"][:2]:
        for filename in root["files"]:
            if Path(filename).name != filename:
                raise ValueError("invalid native MPS filename")
            shutil.copy2(work / "exact_mps" / filename, scratch / filename)
    driver = DMRGDriver(scratch=str(scratch), symm_type=SymmetryTypes.SGFCPX,
                        stack_mem=block2_stack_bytes(16 << 30), n_threads=lib.num_threads())
    errors = {}
    try:
        driver.initialize_system(n_sites=22, n_elec=11, orb_sym=[0]*22)
        solver = SimpleNamespace(driver=driver, ncas=22, nelecas=11,
            kets=[driver.load_mps(r["tag"]) for r in metadata["roots"][:2]],
            npdm_site_type=0, npdm_cutoff=1e-24)
        order = np.arange(22).reshape(2, 11).T.ravel()
        for bra, ket in itertools.product(range(2), repeat=2):
            stem = directory / "native_active" / f"bra-{bra}_ket-{ket}"
            native1 = read_binary(Path(str(stem) + ".dm1.bin"), (22,)*2)[np.ix_(order, order)]
            native2 = read_binary(Path(str(stem) + ".dm2.bin"), (22,)*4)[np.ix_(*([order]*4))]
            # BAGEL alternates creation/annihilation indices; SGF is C C D D.
            native2 = native2.transpose(0, 2, 3, 1)
            actual1 = u.make_transition_rdm1(solver, bra, ket)
            actual2 = u.make_transition_rdm2(solver, bra, ket)
            errors[f"{bra},{ket}"] = [float(np.max(abs(actual1-native1))),
                                      float(np.max(abs(actual2-native2)))]
        write_json(directory / "transition_mapping_audit.json", errors)
        if max(max(pair) for pair in errors.values()) > 1e-10:
            raise RuntimeError(f"native BAGEL/Block2 transition RDM conventions differ: {errors}")
    finally:
        driver.finalize()


def prepare(directory):
    import os
    lib.num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "16")))
    work = directory / "reference"
    ready = work / "exact_mps/READY.json"
    if ready.exists():
        metadata = json.loads(ready.read_text())
        export_mps(work, residual_limit=metadata["residual_limit"])
        audit_transition_rdms(directory, work)
        print("BAGEL_REFERENCE_PREPARED", work, flush=True)
        return
    raw = directory / "native_active"
    integrals = MOIntegrals(raw)
    meta = integrals.meta
    if (meta["nactive_pairs"], meta["nroots"], meta["mode"]) != (11, 12, "active"):
        raise ValueError("expected the Cu twelve-root CAS22 BAGEL reference")
    vectors, nelec = ci_vectors(raw, meta)
    h1e = integrals.one_body()
    phys = integrals.physicist_block([np.arange(22)] * 4)
    eri = phys.transpose(0, 2, 1, 3).copy()
    total = np.array(read_binary(raw / "energies.bin", (12,), np.float64))
    ecore = meta["core_energy"] + meta["nuclear_repulsion"]
    active = total - ecore
    absorbed = fci_dhf_slow.absorb_h1e(h1e, eri, 22, nelec, .5)
    actions = active_action_batch(absorbed, vectors.T, 22, nelec).T
    residuals = np.linalg.norm(actions - vectors * active, axis=0)
    # Native ZHarrison::compute compares RelDvector::rms(), not its norm.
    rms_target = 1e-11
    residual_limit = rms_target * np.sqrt(vectors.shape[0])
    rms_residuals = residuals / np.sqrt(vectors.shape[0])
    gram = vectors.conj().T @ vectors
    rayleigh = np.einsum("ij,ij->j", vectors.conj(), actions) / np.diag(gram)
    orthogonal_residuals = np.linalg.norm(actions-vectors*rayleigh, axis=0)
    report = dict(ncas=22, nelec=nelec, nroots=12, total_energies=total,
                  residual_norms=residuals, gram=gram, gaunt=meta["gaunt"], breit=meta["breit"],
                  rayleigh_energies=rayleigh, energy_offsets=rayleigh-active,
                  orthogonal_residual_norms=orthogonal_residuals,
                  rms_residuals=rms_residuals, rms_target=rms_target,
                  residual_limit=residual_limit,
                  h1_antihermiticity=float(np.max(abs(h1e-h1e.conj().T))))
    write_json(directory / "ci_mapping_audit.json", report)
    print("BAGEL_CI_RESIDUAL", residuals, "orthogonal", orthogonal_residuals,
          "energy_offsets", rayleigh-active, flush=True)
    np.testing.assert_allclose(gram, np.eye(12), atol=1e-10, rtol=0)
    if not np.isfinite(residuals).all() or max(residuals) > residual_limit:
        raise RuntimeError(f"BAGEL CI/integral convention check failed: {residuals}")
    digest = hashlib.sha256()
    for array in (h1e, eri, np.asarray(ecore)):
        digest.update(np.ascontiguousarray(array).tobytes())
    hsha = digest.hexdigest()
    work.mkdir(exist_ok=True)
    save_arrays(work / "reference_vectors.npz", h1e=h1e, eri=eri, hamiltonian_sha256=hsha)
    save_arrays(work / "exact_reference_vectors.npz", vectors=vectors, actions=actions,
                active_energies=active, hamiltonian_sha256=hsha)
    write_json(work / "exact_reference.json", dict(**report, hamiltonian_sha256=hsha,
               converged=[True]*12, active_energies=active, no_alpha_beta_mapping=True))
    write_json(work / "settings.json", dict(hamiltonian_sha256=hsha, manifolds=MANIFOLDS,
               hamiltonian="4C no-pair Dirac-Coulomb-Breit, native BAGEL RI/DF"))
    _active_action_links.cache_clear()
    export_mps(work, residual_limit=residual_limit)
    audit_transition_rdms(directory, work)
    print("BAGEL_REFERENCE_PREPARED", work, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    prepare(parser.parse_args().directory.resolve())
