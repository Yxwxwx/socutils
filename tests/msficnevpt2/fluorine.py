# SPDX-License-Identifier: GPL-3.0-or-later
"""F/dyallv3z, CAS(7e,16 spinors), SA6, M0=1000: SS-SR vs MS-MR.

Restore the completed p-splittings/17/F reference without HF or DMRG sweeps.
Run on gpu01 with python -m tests.msficnevpt2.fluorine. Output and scratch
are isolated; the old p-splittings and UC calculations are not modified.
After completion, --grouped reuses the same preparation for the 4+2 comparison.
Both runs keep metric_refinement=False to reproduce the recorded benchmark;
the production solver enables Gram refinement by default.
"""

import argparse
import json
import os
import pickle
import resource
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from pyscf import gto, lib
from pyscf.fci import cistring, fci_dhf_slow
from scipy.sparse import coo_matrix
from socutils.dmrg import DMRGCI
from socutils.mcscf import zmcscf
from socutils.mrpt import nevpt2_utils as u
from socutils.mrpt import prepare_msfic, solve_msfic
from socutils.scf import spinor_hf

from ..nevpt2_mps_response.f_atom import active_action_batch
from .carbon import json_value

DIRECTORY = Path(__file__).resolve().parent / "fluorine_run"
REFERENCE = Path("/home/Yxwxwx/new-dmrgscf/p-splittings/17/F/dmrg_state")
CONVERSION = 219474.63137


def reference():
    DIRECTORY.mkdir(exist_ok=True)
    metadata = json.loads((REFERENCE / "READY.json").read_text())
    if (metadata["ncas"], metadata["nelec"], metadata["nroots"]) != (16, 7, 6):
        raise ValueError("the saved F reference has a different active space")
    mol = gto.M(
        atom="F 0 0 0",
        basis="dyallv3z",
        charge=-1,
        spin=0,
        verbose=4,
        max_memory=450000,
    )
    mf = spinor_hf.SCF(mol).x2camf()
    # Cache the same F- X2CAMF operator before changing to neutral F;
    # reconstructing the one-electron operator is not an SCF optimization.
    mf.get_hcore()
    mol.charge, mol.spin = 0, 1
    solver = DMRGCI(mol).init(
        ncas=16,
        nelecas=7,
        nroots=6,
        max_bond_dimension=1000,
        schedule_thrd_max=1e-16,
        tol=1e-8,
        n_threads=int(os.environ.get("OMP_NUM_THREADS", "32")),
        stack_memory=8192,
        random_seed=1234,
        orbital_ordering="fiedler",
        final_one_site=False,
        scratch=Path(lib.param.TMPDIR) / "fluorine_msfic",
        checkpoint_dir=REFERENCE / "dmrg_checkpoint",
    )
    mc = zmcscf.CASSCF(mf, ncas=16, nelecas=7)
    mc.fcisolver = solver
    mc.canonicalization = mc.canonicalize_ = mc.natorb = False
    mc.orbital_symmetry = None
    try:
        assert getattr(mf, "with_df", None) is None
        assert solver.kramers_adapter is None
        with np.load(REFERENCE / "active_hamiltonian.npz") as saved:
            if (
                str(saved["hamiltonian_sha256"].item())
                != metadata["hamiltonian_sha256"]
            ):
                raise ValueError("saved F metadata and Hamiltonian disagree")
            solver.restore_checkpoint(
                saved["h1e"],
                saved["eri"],
                16,
                7,
                ecore=float(saved["ecore"]),
                nroots=6,
                max_memory=mol.max_memory,
            )
        mc.mo_coeff = np.load(REFERENCE / "mo_coeff.npy")
        mc.e_states = np.load(REFERENCE / "root_energies.npy")
        np.testing.assert_allclose(mc.e_states, solver.e_tot, atol=1e-12, rtol=0)
        np.testing.assert_allclose(
            mc.mo_coeff.conj().T @ mf.get_ovlp() @ mc.mo_coeff,
            np.eye(mc.mo_coeff.shape[1]),
            atol=1e-10,
            rtol=0,
        )
        mc.e_tot, mc.e_cas = float(np.mean(mc.e_states)), solver.e_cas
        mc.ci, mc.converged = list(solver.kets), True
        h1e, ecore = mc.get_h1eff(mc.mo_coeff)
        snap = solver.checkpoint_hamiltonian
        np.testing.assert_allclose(h1e, snap["h1e"], atol=1e-9, rtol=0)
        np.testing.assert_allclose(ecore, snap["ecore"], atol=1e-9, rtol=0)
        print(
            "F_REFERENCE_RESTORED",
            REFERENCE,
            "h1e_defect",
            float(np.max(np.abs(h1e - snap["h1e"]))),
            "ecore_defect",
            float(abs(ecore - snap["ecore"])),
            "no_new_sweeps",
            flush=True,
        )
    except BaseException:
        solver.close()
        raise
    return mc, solver


def audit_reference(mc, solver):
    """Measure actual MPS residuals and J²; no FCI solve or spin selection."""
    strings = cistring.make_strings(range(16), 7)
    dets = np.array(
        [[(int(b) >> p) & 1 for p in range(16)] for b in strings], dtype=np.uint8
    )
    # Native coefficient readout uses MPS site order, even though its returned
    # determinant labels are mapped back. Audit H and J in that same basis.
    order = np.asarray(solver.driver.reorder_idx, dtype=int)
    vectors = []
    for root in range(6):
        driver, ket = u._root_ket(solver, root)
        working = driver.copy_mps(ket, tag=f"F_MS_AUDIT_{root}")
        driver.align_mps_center(working, ref=0)
        vectors.append(
            driver.get_csf_coefficients(
                working,
                cutoff=0.0,
                given_dets=dets,
                max_print=0,
                fci_conv=True,
                iprint=0,
            )[1]
        )
    vectors = np.asarray(vectors).T
    snap = solver.checkpoint_hamiltonian
    absorbed = fci_dhf_slow.absorb_h1e(
        snap["h1e"][np.ix_(order, order)],
        snap["eri"][np.ix_(order, order, order, order)],
        16,
        7,
        0.5,
    )
    action = active_action_batch(absorbed, vectors.T, 16, 7).T
    residuals = np.linalg.norm(action - vectors * (mc.e_states - snap["ecore"]), axis=0)
    gram = float(np.max(np.abs(vectors.conj().T @ vectors - np.eye(6))))
    ca, cb = mc.mol.sph2spinor_coeff()
    overlap = mc.mol.intor("int1e_ovlp_sph")
    orbital = -1j * mc.mol.intor("int1e_cg_irxp_sph", comp=3)
    spin = np.array(
        [
            0.5 * (ca.conj().T @ overlap @ cb + cb.conj().T @ overlap @ ca),
            0.5j * (cb.conj().T @ overlap @ ca - ca.conj().T @ overlap @ cb),
            0.5 * (ca.conj().T @ overlap @ ca - cb.conj().T @ overlap @ cb),
        ]
    )
    angular = (
        np.array([ca.conj().T @ a @ ca + cb.conj().T @ a @ cb for a in orbital]) + spin
    )
    mo = mc.mo_coeff[:, mc.ncore : mc.ncore + 16][:, order]
    links = cistring.gen_linkstr_index(range(16), 7)
    p, q, target, sign = links.reshape(-1, 4).T
    origin = np.repeat(np.arange(len(strings)), links.shape[1])
    acted = []
    for ao in angular:
        matrix = mo.conj().T @ ao @ mo
        operator = coo_matrix(
            (matrix[p, q] * sign, (target, origin)), shape=(len(strings),) * 2
        ).tocsr()
        acted.append(operator @ vectors)
    j2 = sum(v.conj().T @ v for v in acted)
    if not np.all(np.isfinite(residuals)) or gram > 1e-7:
        raise ValueError("invalid restored F reference vectors")
    if max(residuals) > 1e-8:
        u._warn_numerical(
            f"old F reference residual {max(residuals):.3e} exceeds 1e-8; "
            "reporting it without reoptimizing the converged reference"
        )
    audit = {
        "energies": mc.e_states,
        "residuals": residuals,
        "gram_defect": gram,
        "J2": j2,
        "fingerprint": snap["hamiltonian_sha256"],
        "reference_directory": str(REFERENCE),
        "orbital_ordering": order,
        "requested_bond_dimension": 1000,
        "no_cd_df": True,
        "no_kr": True,
    }
    print("F_REFERENCE_AUDIT", json.dumps(audit, default=json_value), flush=True)
    return audit


def run():
    start = time.perf_counter()
    DIRECTORY.mkdir(exist_ok=True)
    cache = DIRECTORY / "prepared.pkl"
    if cache.exists():
        with cache.open("rb") as handle:
            common, audit = pickle.load(handle)
        print("F_PREPARATION_CACHE_REUSED", cache, flush=True)
    else:
        mc, solver = reference()
        try:
            audit = audit_reference(mc, solver)

            def densities(bra, ket):
                begin = time.perf_counter()
                print("F_RDM_PAIR_BEGIN", bra, ket, flush=True)
                pdms = u.make_transition_dm1234(solver, bra, ket)
                print(
                    "F_RDM_PAIR_END",
                    bra,
                    ket,
                    "seconds",
                    time.perf_counter() - begin,
                    flush=True,
                )
                return pdms

            common = prepare_msfic(
                mc,
                sa_roots=range(6),
                sa_weights=np.ones(6) / 6,
                model_roots=range(6),
                transition_pdms=densities,
                transition_rdm_fallback_dir=DIRECTORY / "rdm_fallback",
            )
            with cache.open("wb") as handle:
                pickle.dump((common, audit), handle)
        finally:
            solver.close()
    data = {"reference_audit": audit, "preparation": common.diagnostics, "results": {}}
    for shift in (0.2, 0.0):
        for ansatz in ("ss_sr", "ms_mr"):
            print("F_MS_SOLVE_BEGIN", ansatz, shift, flush=True)
            # Reproduce the joint/grouped benchmark with the same numerical path.
            result = solve_msfic(
                common, ansatz=ansatz, shift=shift, metric_refinement=False
            )
            j2 = np.diag(
                result.mixing.conj().T @ np.asarray(audit["J2"]) @ result.mixing
            ).real
            spreads = {
                "P3/2": float(np.ptp(result.energies[:4]) * CONVERSION),
                "P1/2": float(np.ptp(result.energies[4:]) * CONVERSION),
            }
            data["results"][f"{ansatz}_eta_{shift}"] = {
                **result.__dict__,
                "J2": j2,
                "multiplet_spreads_cm_inverse": spreads,
            }
            print(ansatz, "eta", shift, "energy for each state", flush=True)
            for root, energy in enumerate(result.energies):
                print(
                    f"  State {root} weight {1 / 6:.7g} E = {energy:.14f}", flush=True
                )
            print("F_MULTIPLET_SPREADS", spreads, flush=True)
            print("F_J2_DIAGNOSTIC", j2, flush=True)
            data["wall_seconds"] = time.perf_counter() - start
            data["peak_rss_gib"] = (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
            )
            with (DIRECTORY / "results.json").open("w") as handle:
                json.dump(data, handle, default=json_value, indent=2)
    return data


def _restrict_model(prepared, roots):
    """Restrict IC generators and target roots before solving; keep SA Dyall."""
    roots = tuple(roots)
    if not roots or len(set(roots)) != len(roots):
        raise ValueError("nonempty distinct model roots required")
    indices = [prepared.model_roots.index(root) for root in roots]
    model = np.ix_(indices, indices)
    classes = {}
    for key, block in prepared.classes.items():
        d = block.dimension_per_reference
        rows = np.concatenate([np.arange(i * d, (i + 1) * d) for i in indices])
        ic = np.ix_(rows, rows)
        classes[key] = replace(
            block,
            **{
                name: None if getattr(block, name) is None else getattr(block, name)[ic]
                for name in ("metric", "right", "left", "active")
            },
            source=np.take(np.take(block.source, rows, axis=-2), indices, axis=-1),
        )
    return replace(
        prepared,
        model_roots=roots,
        overlap=prepared.overlap[model],
        active_reference=prepared.active_reference[model],
        reference=prepared.reference[model],
        classes=classes,
        diagnostics={**prepared.diagnostics, "restricted_model_roots": roots},
    )


def run_grouped():
    """Follow the completed joint run without rebuilding Fock, ERIs or RDMs."""
    start = time.perf_counter()
    joint = json.loads((DIRECTORY / "results.json").read_text())
    required = {f"{a}_eta_{s}" for a in ("ss_sr", "ms_mr") for s in (0.2, 0.0)}
    if not required.issubset(joint["results"]):
        raise ValueError("complete the six-state joint calculation first")
    with (DIRECTORY / "prepared.pkl").open("rb") as handle:
        common, audit = pickle.load(handle)
    if common.diagnostics["fingerprint"] != joint["preparation"]["fingerprint"]:
        raise ValueError("joint energies and cached SA6 preparation do not match")
    data = {
        "reference_audit": audit,
        "preparation": common.diagnostics,
        "same_six_root_sa_dyall": True,
        "results": {},
        "comparison": {},
    }
    for name, roots in (("P3/2", (0, 1, 2, 3)), ("P1/2", (4, 5))):
        selected = _restrict_model(common, roots)
        model_j2 = np.asarray(audit["J2"])[np.ix_(roots, roots)]
        for shift in (0.2, 0.0):
            for ansatz in ("ss_sr", "ms_mr"):
                key = f"{ansatz}_eta_{shift}"
                print("F_GROUPED_SOLVE_BEGIN", name, ansatz, shift, flush=True)
                result = solve_msfic(
                    selected, ansatz=ansatz, shift=shift, metric_refinement=False
                )
                j2 = np.diag(result.mixing.conj().T @ model_j2 @ result.mixing).real
                entry = data["results"].setdefault(key, {})
                entry[name] = {
                    **result.__dict__,
                    "model_roots": roots,
                    "J2": j2,
                    "spread_cm_inverse": float(result.spread * CONVERSION),
                }
                print(name, ansatz, "eta", shift, "energy for each state", flush=True)
                for root, energy in zip(roots, result.energies):
                    print(
                        f"  State {root} weight {1 / 6:.7g} E = {energy:.14f}",
                        flush=True,
                    )
                if len(entry) == 2:
                    energies = np.concatenate(
                        [entry[g]["energies"] for g in ("P3/2", "P1/2")]
                    )
                    difference = energies - np.asarray(
                        joint["results"][key]["energies"]
                    )
                    data["comparison"][key] = {
                        "joint_energies": joint["results"][key]["energies"],
                        "grouped_energies": energies,
                        "grouped_minus_joint_eh": difference,
                        "grouped_minus_joint_cm_inverse": difference * CONVERSION,
                        "joint_spreads_cm_inverse": joint["results"][key][
                            "multiplet_spreads_cm_inverse"
                        ],
                        "grouped_spreads_cm_inverse": {
                            g: entry[g]["spread_cm_inverse"] for g in entry
                        },
                    }
                    print(
                        "F_JOINT_GROUPED_COMPARISON",
                        key,
                        json.dumps(data["comparison"][key], default=json_value),
                        flush=True,
                    )
                data["wall_seconds"] = time.perf_counter() - start
                data["peak_rss_gib"] = (
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
                )
                with (DIRECTORY / "grouped_results.json").open("w") as handle:
                    json.dump(data, handle, default=json_value, indent=2)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grouped",
        action="store_true",
        help="run 4+2 after the completed SA6 joint calculation",
    )
    args = parser.parse_args()
    run_grouped() if args.grouped else run()
