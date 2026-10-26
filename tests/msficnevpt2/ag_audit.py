# SPDX-License-Identifier: GPL-3.0-or-later
"""Ag S1/2 diagnosis on saved orbitals/MPS; retain all four ordered 1--4 TRDMs.

An isolated Slurm diagnostic, not part of the production PT workflow. No
orbital optimization, new DMRG sweeps, RDM averaging or KR projection occurs.
"""

import argparse
import gc
import json
import os
import pickle
from pathlib import Path
import sys
import time

import numpy as np
from pyscf import lib, scf
from pyscf.fci import cistring, fci_dhf_slow

from socutils.dmrg import DMRGCI
from socutils.mcscf import zmcscf
from socutils.mrpt import make_transition_dm1234, prepare_msfic, solve_msfic
from socutils.mrpt import x2cmsficnevpt2 as ms
from socutils.scf import spinor_hf
from tests.nevpt2_mps_response.f_atom import active_action_batch, _active_action_links


def block2_stack_bytes(default):
    """Use 80% of the requested Slurm node memory (reported in MiB)."""
    requested = os.environ.get("SLURM_MEM_PER_NODE")
    if requested is None:
        return default
    memory_mib = int(requested)
    if memory_mib <= 0:
        raise ValueError("a positive explicit Slurm memory request is required")
    return (memory_mib * 2**20 * 4 + 4) // 5


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return [value.real, value.imag]
    raise TypeError(type(value).__name__)


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, default=serial, indent=2) + "\n")
    temporary.replace(path)


def fixed_n_coefficients(mps, nelec):
    """Contract owned, unfused C1 SGF tensors exactly; no Schmidt truncation.

    This is an exponential diagnostic readout, checked against Block2's native
    DeterminantTRIE below. Production NEVPT2 never calls this routine.
    """
    ncas = len(mps.tensors)
    if not 0 <= nelec <= ncas <= 24:
        raise ValueError("coefficient diagnostic is limited to at most 24 spinors")
    rows = {0: (np.zeros(1, dtype=np.uint32), np.ones((1, 1), complex))}
    for site, tensor in enumerate(mps.tensors):
        if tensor is None:
            raise ValueError("expected unfused one-site tensors")
        collected, seen = {}, set()
        for block in tensor.blocks:
            labels, matrix = block.q_labels, block.reduced
            if site == 0:
                ql, qm, qr = 0, labels[0].n, labels[1].n
                matrix = matrix[None, ...]
            elif site == ncas - 1:
                ql, qm, qr = labels[0].n, labels[1].n, nelec
                matrix = matrix[..., None]
            else:
                ql, qm, qr = (q.n for q in labels)
            if qm not in (0, 1) or qr != ql + qm or matrix.shape[1] != 1:
                raise ValueError("expected particle-number-only SGF tensors")
            if (ql, qm) in seen:
                raise ValueError("duplicate SGF block: coefficient oracle requires C1")
            seen.add((ql, qm))
            if ql not in rows or not 0 <= nelec - qr <= ncas - site - 1:
                continue
            bits, previous = rows[ql]
            collected.setdefault(qr, []).append(
                (bits | (qm << site), previous @ matrix[:, 0, :])
            )
        rows = {
            q: (np.concatenate([x[0] for x in pieces]),
                np.concatenate([x[1] for x in pieces]))
            for q, pieces in collected.items()
        }
    bits, values = rows[nelec]
    strings = cistring.make_strings(range(ncas), nelec)
    result = np.zeros(len(strings), complex)
    addresses = np.searchsorted(strings, bits)
    np.testing.assert_array_equal(strings[addresses], bits)
    result[addresses] = values[:, 0]
    return result


def extract_root(driver, ket, root, nelec):
    from pyblock2.algebra.io import MPSTools

    owned = driver.copy_mps(ket, tag=f"AG_COEFFICIENT_AUDIT_{root}")
    # Representation change on a private copy, not a one-site DMRG sweep.
    driver.adjust_mps(owned, dot=1)
    vector = fixed_n_coefficients(MPSTools.from_block2(owned), nelec)
    ncas = owned.n_sites
    strings = cistring.make_strings(range(ncas), nelec)
    rng = np.random.default_rng(7100 + root)
    sample = np.unique(np.r_[np.argsort(np.abs(vector))[-32:],
                             rng.choice(len(vector), min(32, len(vector)), replace=False)])
    occupations = ((strings[sample, None] >> np.arange(ncas)) & 1).astype(np.uint8)
    native = np.asarray(driver.get_csf_coefficients(
        owned, cutoff=0.0, given_dets=occupations,
        max_print=0, fci_conv=True, iprint=0,
    )[1])
    phase = np.vdot(vector[sample], native) / np.vdot(vector[sample], vector[sample])
    np.testing.assert_allclose(abs(phase), 1.0, atol=1e-10, rtol=0)
    vector *= phase
    np.testing.assert_allclose(vector[sample], native, atol=1e-11, rtol=1e-10)
    print("AG_COEFFICIENTS_VERIFIED", root, "native_samples", len(sample),
          "norm2", np.vdot(vector, vector).real, flush=True)
    return vector


def reference_audit(mc, solver, reference, output):
    vectors = np.column_stack([
        extract_root(solver.driver, solver.kets[root], root, 11) for root in (0, 1)
    ])
    absorbed = fci_dhf_slow.absorb_h1e(reference["h1e"], reference["eri"], 22, 11, 0.5)
    print("AG_DIRECT_H_ACTION_BEGIN", flush=True)
    actions = active_action_batch(absorbed, vectors.T, 22, 11).T
    gram = vectors.conj().T @ vectors
    projected = vectors.conj().T @ actions
    model_residual = actions - vectors @ np.linalg.solve(gram, projected)
    saved_energies = reference["e_states"][:2] - reference["ecore"]
    data = dict(
        gram=gram, projected_active_h=projected,
        saved_active_energies=saved_energies,
        saved_energy_residual_norms=np.linalg.norm(actions - vectors * saved_energies, axis=0),
        model_space_residual_norms=np.linalg.norm(model_residual, axis=0),
        native_projected_h_defect=np.max(np.abs(
            projected - (solver.projected_hamiltonian[:2, :2] - reference["ecore"] * gram)
        )),
        projected_reference_spread_cm_inverse=np.ptp(
            np.linalg.eigvalsh(projected)) * 219474.6313705,
    )
    np.testing.assert_allclose(gram, np.eye(2), atol=1e-9, rtol=0)
    np.testing.assert_allclose(data["native_projected_h_defect"], 0, atol=1e-8, rtol=0)
    np.savez(output / "reference_vectors.npz", vectors=vectors, actions=actions,
             model_residual=model_residual, h1e=reference["h1e"], eri=reference["eri"],
             hamiltonian_sha256=reference["hamiltonian_sha256"])
    write_json(output / "reference_audit.json", data)
    print("AG_DIRECT_REFERENCE_AUDIT", json.dumps(data, default=serial), flush=True)
    _active_action_links.cache_clear()
    del vectors, actions, model_residual
    gc.collect()


def main(directory):
    directory = directory.resolve()
    source = directory.parent
    state = source / "dmrg_state"
    with np.load(state / "reference.npz", allow_pickle=False) as saved:
        reference = {key: saved[key] for key in saved.files}
    fingerprint = str(reference["hamiltonian_sha256"].item())
    settings = dict(hamiltonian_sha256=fingerprint, model_roots=[0, 1],
                    sa_roots=list(range(12)), sa_weights=reference["weights"],
                    shift=0.0, metric_refinement=True, raw_rdms="independent-ordered-pairs",
                    cd_df=False, nevpt2_kramers_reduction=False)
    settings_path = directory / "audit_settings.json"
    if settings_path.exists():
        if json.loads(settings_path.read_text()) != json.loads(json.dumps(settings, default=serial)):
            raise ValueError("existing audit belongs to different inputs")
    else:
        write_json(settings_path, settings)
    np.testing.assert_array_equal(reference["weights"], np.full(12, 1 / 12))
    threads = int(os.environ["SLURM_CPUS_PER_TASK"])
    lib.num_threads(threads)
    mol, hf = scf.chkfile.load_scf(str(source / "hf_kr/hf.chk"))
    mol.stdout, mol.verbose, mol.max_memory = sys.stdout, 4, 1500000
    if mol.atom_symbol(0) != "Ag":
        raise ValueError("this diagnostic requires the Ag KR M2000 reference")
    mf = spinor_hf.SCF(mol).x2camf()
    mf.__dict__.update(hf)
    mf.converged, mf.max_memory = True, mol.max_memory
    mf.with_x2c.hcore = reference["hcore_ao"]
    mol.charge, mol.spin = 0, 1
    mc = zmcscf.CASSCF(mf, 22, 11)
    mc.fcisolver = DMRGCI(mol).init(
        ncas=22, nelecas=11, nroots=12, max_bond_dimension=2000, tol=1e-8,
        schedule_thrd_max=1e-16, n_threads=threads,
        stack_memory=block2_stack_bytes(500000 * 1000000) / 1000000,
        random_seed=1234, orbital_ordering="original", final_one_site=False,
        scratch=Path(lib.param.TMPDIR) / "dmrg_scratch",
        checkpoint_dir=state / "dmrg_checkpoint",
    )
    mc.state_average_(np.full(12, 1 / 12))
    solver = mc.fcisolver
    started = time.monotonic()
    try:
        solver.restore_checkpoint(reference["h1e"], reference["eri"], 22, 11,
                                  ecore=float(reference["ecore"]), nroots=12,
                                  max_memory=mol.max_memory)
        assert solver.kramers_adapter is None and not solver.final_one_site
        solver.driver.reorder_idx = None
        mc.mo_coeff, mc.mo_energy = reference["mo_coeff"], None
        solver.e_states = reference["e_states"].copy()
        mc.e_tot, mc.e_cas = float(np.mean(mc.e_states)), solver.e_cas
        mc.ci, mc.converged = list(solver.kets), True
        mc.canonicalization = mc.canonicalize_ = mc.natorb = False
        if not (directory / "reference_audit.json").exists():
            reference_audit(mc, solver, reference, directory)
        raw_directory = directory / "transition_rdms"
        raw_directory.mkdir(exist_ok=True)

        def pair(bra, ket):
            target = raw_directory / f"bra-{bra}_ket-{ket}"
            target.mkdir(exist_ok=True)
            ready = target / "READY.json"
            if not ready.exists():
                available = next(int(line.split()[1]) * 1024
                                 for line in Path("/proc/meminfo").read_text().splitlines()
                                 if line.startswith("MemAvailable:"))
                required = sum(22 ** (2 * rank) * 16 for rank in range(1, 5))
                if available < required + 128 * 2**30:
                    raise MemoryError("insufficient available RAM before ordered 4-TRDM")
                print("AG_RETAINED_TRDM_BEGIN", bra, ket, flush=True)
                pdms = make_transition_dm1234(solver, bra, ket)
                ms._write_transition_pair(pdms, bra, ket, 22, target, keep=True)
                del pdms
                gc.collect()
            metadata = json.loads(ready.read_text())
            if (metadata["bra"], metadata["ket"], metadata["ncas"]) != (bra, ket, 22):
                raise ValueError("retained TRDM metadata mismatch")
            print("AG_RETAINED_TRDM_READ", bra, ket, target, flush=True)
            return tuple(np.load(target / f"dm{rank}.npy", mmap_mode="r", allow_pickle=False)
                         for rank in range(1, 5))

        cache = directory / "prepared.pkl"
        if cache.exists():
            with cache.open("rb") as handle:
                prepared = pickle.load(handle)
        else:
            prepared = prepare_msfic(
                mc, sa_roots=range(12), sa_weights=np.full(12, 1 / 12), model_roots=(0, 1),
                model_overlap=solver.root_overlap[:2, :2], transition_pdms=pair,
                contraction_backend="pytblis", integral_max_memory=2000,
            )
            temporary = cache.with_suffix(".pkl.tmp")
            with temporary.open("wb") as handle:
                pickle.dump(prepared, handle, protocol=pickle.HIGHEST_PROTOCOL)
            temporary.replace(cache)
            print("AG_PREPARED_MATRICES_SAVED", cache, flush=True)
        result = solve_msfic(prepared, ansatz="ms_mr", shift=0.0, metric_refinement=True)
        write_json(directory / "S1_2.json", dict(settings=settings, **result.__dict__))
        print("AG_RETAINED_AUDIT_COMPLETE", "wall_seconds", time.monotonic() - started,
              "energies", result.energies, flush=True)
    finally:
        solver.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    main(parser.parse_args().directory)
