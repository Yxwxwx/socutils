# SPDX-License-Identifier: GPL-3.0-or-later
"""F: unrestricted full-ERI six-root second-order SA-DMRG-SCF + root-0 NEVPT2.

Separate processes measure peak RSS independently; hybrid runs before full
and raises if any code requests a 4-RDM. Both restore the identical MPS image.
"""

import argparse
import json
import os
import resource
import time
from functools import partial
from pathlib import Path

import numpy as np
from pyscf import gto, lib, scf
from socutils.dmrg import DMRGCI
from socutils.mcscf import zmcscf
from socutils.mrpt import WickX2CFICNEVPT2, WickX2CSCNEVPT2
from socutils.mrpt import nevpt2_utils as u
from socutils.scf import spinor_hf

NCAS, NELEC, NROOTS = 16, 7, 6
THREADS = int(os.environ.get("OMP_NUM_THREADS", "8"))
MEMORY_MB = 200000
STACK_MB = 8192  # Leave room for the baseline's 64 GiB complex 4-RDM.


def save_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, default=lambda x: x.tolist()) + "\n")
    temporary.replace(path)


def make_solver(mol, directory, *, nroots=NROOTS):
    return DMRGCI(mol).init(
        ncas=NCAS, nelecas=NELEC, nroots=nroots,
        max_bond_dimension=1000, tol=1e-8, schedule_thrd_max=1e-16,
        n_threads=THREADS, stack_memory=STACK_MB, orbital_ordering="original",
        scratch=Path(lib.param.TMPDIR) / "dmrg_scratch",
        checkpoint_dir=directory / "dmrg_checkpoint",
        final_one_site=False, random_seed=1234,
    )


def make_mc(mf, solver):
    mc = zmcscf.CASSCF(mf, ncas=NCAS, nelecas=NELEC)
    mc.fcisolver = solver
    mc.canonicalization = mc.canonicalize_ = mc.natorb = False
    mc.conv_tol, mc.conv_tol_grad = 1e-8, 1e-4
    mc.max_cycle_macro = 50
    mc.max_stepsize = .2
    mc.superci_davidson_tol = 1e-8
    mc.superci_davidson_max_space = 500
    mc.superci_davidson_strict = True
    mc.orbital_symmetry = None
    return mc


def run_mcscf(directory):
    if (directory / "mcscf.json").exists():
        saved = json.loads((directory / "mcscf.json").read_text())
        if (saved["nroots"] != NROOTS or saved["mcscf_integrals"] != "full"
                or saved["mcscf_symm"] != "none"):
            raise ValueError("this test requires a full-ERI unrestricted six-root reference")
        print("Reusing completed full-ERI unrestricted six-root reference", flush=True)
        return
    mol = gto.M(atom="F 0 0 0", basis="dyallv3z", charge=-1, spin=0,
                verbose=4, max_memory=MEMORY_MB)
    # Match p-splittings/17/F: F- ordinary spinor HF, then neutral F CAS.
    # No KRHF, Cholesky/DF, Kramers orbital projection or result adapter.
    mf = spinor_hf.SCF(mol).x2camf()
    mf.chkfile = str(directory / "hf.chk")
    mf.conv_tol, mf.max_cycle = 1e-12, 200
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("F- full-integral X2CAMF HF did not converge")
    initial_mo = mf.mo_coeff.copy()
    mol.charge, mol.spin = 0, 1
    solver = make_solver(mol, directory)
    mc = make_mc(mf, solver)
    mc.state_average_(np.ones(NROOTS) / NROOTS)
    solver = mc.fcisolver
    mc.mo_coeff = initial_mo
    mc.chkfile = str(directory / "mcscf.chk")
    mc.callback = solver.restart_scheduler_()
    start = time.perf_counter()
    try:
        assert getattr(mf, "with_df", None) is None
        assert solver.kramers_adapter is None and mc.orbital_symmetry is None
        mc.second_order()
        if not mc.converged or not solver.converged:
            raise RuntimeError("full-ERI unrestricted six-root SA-DMRG-SCF did not converge")
        if mc.second_order_diagnostics["kramers_restricted"]:
            raise AssertionError("the ground-state test must not enable KR")
        snapshot = solver.checkpoint_hamiltonian
        if snapshot is None:
            raise RuntimeError("no final checkpoint Hamiltonian")
        states = np.asarray(mc.e_states, dtype=float)
        assert states.shape == (NROOTS,)
        np.savez(directory / "reference.npz", mo_coeff=mc.mo_coeff,
                 e_states=states, h1e=snapshot["h1e"],
                 eri=snapshot["eri"], ecore=snapshot["ecore"],
                 hcore_ao=mf.get_hcore(), overlap_ao=mf.get_ovlp(),
                 fingerprint=snapshot["hamiltonian_sha256"])
        data = dict(converged=True, basis="dyallv3z", ncas=NCAS, nelec=NELEC,
                    nroots=NROOTS, root_energies=states,
                    weights=np.ones(NROOTS) / NROOTS,
                    state_average_energy=float(mc.e_tot),
                    macro_iterations=mc.second_order_diagnostics["macro_iterations"],
                    final_gradient=float(mc.final_orbital_gradient_norm),
                    mcscf_integrals="full", mcscf_symm="none", hf_symm="none",
                    cd=False, kr=False,
                    optimizer="second_order", pt_integrals="full",
                    fingerprint=snapshot["hamiltonian_sha256"],
                    wall_seconds=time.perf_counter() - start,
                    peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20)
        save_json(directory / "mcscf.json", data)
        print("F_MCSCF_RESULT", json.dumps(data, default=lambda x: x.tolist()), flush=True)
    finally:
        solver.close()


def restore(directory):
    metadata = json.loads((directory / "mcscf.json").read_text())
    mol, saved = scf.chkfile.load_scf(str(directory / "hf.chk"))
    mol.verbose, mol.max_memory = 4, MEMORY_MB
    # Old CD/KR snapshots remain readable for historical diagnostic scripts,
    # but the new ground-state runner never selects those directories.
    mf = (spinor_hf.KRHF(mol).x2camf().cholesky(tau=1e-8)
          if metadata["mcscf_symm"] == "kramers" else spinor_hf.SCF(mol).x2camf())
    mf.__dict__.update(saved)
    mf.converged = True
    mol.charge, mol.spin = 0, 1
    solver = make_solver(mol, directory, nroots=int(metadata["nroots"]))
    with np.load(directory / "reference.npz") as ref:
        np.testing.assert_allclose(mf.get_ovlp(), ref["overlap_ao"], atol=1e-12, rtol=1e-12)
        mf.with_x2c.hcore = ref["hcore_ao"].copy()
        solver.restore_checkpoint(ref["h1e"], ref["eri"], NCAS, NELEC,
                                  ecore=float(ref["ecore"]), nroots=int(metadata["nroots"]),
                                  max_memory=MEMORY_MB)
        mc = make_mc(mf, solver)
        mc.mo_coeff = ref["mo_coeff"].copy()
        mc.mo_energy = None
        mc.e_states = ref["e_states"].copy()
        mc.e_tot, mc.e_cas = float(np.mean(mc.e_states)), np.asarray(solver.e_cas)
        mc.ci = list(solver.kets) if solver.nroots > 1 else solver.kets[0]
        mc.converged = True
        fingerprint = str(ref["fingerprint"].item())
    return mc, solver, fingerprint


def active_action_batch(eri, vectors, norb, nelec):
    """Vectorize PySCF's SGF contract_2e for a residual measurement, not a solve."""
    from pyscf.fci import cistring
    from scipy.sparse import coo_matrix
    links = cistring.gen_linkstr_index(range(norb), nelec)
    dimension, nlink = links.shape[:2]
    p, q, target, sign = links.reshape(-1, 4).T
    origin = np.repeat(np.arange(dimension), nlink)
    pq = p * norb + q
    left = coo_matrix((sign, (pq * dimension + target, origin)),
                      shape=(norb * norb * dimension, dimension)).tocsr()
    right = coo_matrix((sign, (target, pq * dimension + origin)),
                       shape=(dimension, norb * norb * dimension)).tocsr()
    # Bound temporary arrays while using the same E_pq E_rs contraction as
    # fci_dhf_slow. No alpha/beta representation or diagonalization is involved.
    result = np.empty_like(vectors)
    for start in range(0, len(vectors), 4):
        batch = vectors[start:start + 4].T
        t = (left @ batch).reshape(norb * norb, -1)
        t = eri.reshape(norb * norb, norb * norb) @ t
        result[start:start + 4] = (right @ t.reshape(
            norb * norb * dimension, batch.shape[1])).T
    return result


def check_global_residual(key, residual):
    """A finite response residual above 1e-8 is a warning, not a run failure."""
    if not np.isfinite(residual):
        raise ValueError(f"non-finite global {key} response residual")
    verified = residual <= 1e-8
    if not verified:
        u._warn_numerical(f"whole {key} global relative residual {residual:.3e} "
                          "exceeds 1e-8; retaining the energy and continuing")
    return verified


def audited_response(evaluate, mc, eris, pdms, *args, **kwargs):
    """F-only residual measurement; native pyblock2 still does every solve."""
    from pyscf.fci import fci_dhf_slow
    from socutils.mrpt import nevpt2_mps_response as response

    from .diagnose_r85 import _source_coefficients, apply_source_vector, mps_vector

    # ponytail: fixed-N coefficient oracle is exponential, used only for F
    # validation; production response never expands MPSs into CI coefficients.
    driver, reference = u._root_ket(mc.fcisolver, kwargs.get("root", 0))
    order = None if driver.reorder_idx is None else np.asarray(driver.reorder_idx).copy()
    reference_vector = mps_vector(driver, reference, mc.ncas, int(mc.nelecas))
    h = eris.get_h1eff("AA")
    g = eris.get_phys("AAAA").transpose(0, 2, 1, 3)
    h2e = {key: fci_dhf_slow.absorb_h1e(h, g, mc.ncas, int(mc.nelecas) + delta, .5)
           for key, delta in (("i", 1), ("r", -1))}
    from pyscf.fci import cistring
    native_multiply = driver.multiply
    measured = {}
    eact = float((np.einsum("pq,pq", h, pdms[0]) + .5 * np.einsum(
        "pqrs,pqsr", eris.get_phys("AAAA"), pdms[1])).real)

    def coefficients(mps, dets):
        dets = dets.copy()
        phase = 1.
        if order is not None:
            ncore = mps.n_sites - mc.ncas if int(mps.info.target.n) > int(mc.nelecas) else 0
            active = dets[:, ncore:ncore + mc.ncas][:, order]
            dets[:, ncore:ncore + mc.ncas] = active
            inversions = np.triu(order[:, None] > order[None, :], 1)
            parity = np.einsum("bi,ij,bj->b", active.astype(int), inversions.astype(int), active.astype(int)) % 2
            phase = 1 - 2 * parity
        working = driver.copy_mps(mps, tag=response._response_tag())
        try:
            driver.align_mps_center(working, ref=0)
            return phase * np.asarray(driver.get_csf_coefficients(
                working, cutoff=0., given_dets=dets, max_print=0,
                fci_conv=True, iprint=0)[1])
        finally:
            response._release_response_mps(driver, working)

    def measure_multiply(bra, mpo, ket, **options):
        value = native_multiply(bra, mpo, ket, **options)
        key = "i" if bra.info.n_ex_inactive else "r"
        print(f"F_RESPONSE_AUDIT_START root={kwargs.get('root', 0)} class={key}", flush=True)
        count = eris.ncore if key == "i" else eris.nvirt
        n = int(mc.nelecas) + (1 if key == "i" else -1)
        occupied = cistring.gen_occslst(range(mc.ncas), n)
        active_dets = np.zeros((len(occupied), mc.ncas), dtype=np.uint8)
        active_dets[np.arange(len(occupied))[:, None], occupied] = 1
        dets = []
        for index in range(count):
            external = np.full((len(occupied), count), int(key == "i"), dtype=np.uint8)
            external[:, index] = int(key == "r")
            dets.append(np.c_[external, active_dets] if key == "i" else np.c_[active_dets, external])
        x = coefficients(bra, np.concatenate(dets)).reshape(count, -1)
        cache = {}
        b = np.array([apply_source_vector(key, _source_coefficients(eris, key, index),
                                          reference_vector, mc.ncas, int(mc.nelecas), cache)
                      for index in range(count)])
        if key == "i":
            b *= ((-1.) ** (np.arange(count) + count - 1))[:, None]
        else:
            b *= (-1.) ** n
        # A global phase of the reference is physically immaterial; preserve
        # its native embedding phase when checking the complete source.
        reference_occ = cistring.gen_occslst(range(mc.ncas), int(mc.nelecas))
        reference_dets = np.zeros((len(reference_occ), mc.ncas), dtype=np.uint8)
        reference_dets[np.arange(len(reference_occ))[:, None], reference_occ] = 1
        frozen = np.full((len(reference_occ), count), int(key == "i"), dtype=np.uint8)
        embedded = coefficients(ket, np.c_[frozen, reference_dets] if key == "i"
                                else np.c_[reference_dets, frozen])
        phase = np.vdot(reference_vector, embedded) / np.vdot(reference_vector, reference_vector)
        np.testing.assert_allclose(abs(phase), 1., atol=1e-10)
        b *= phase
        orbital = np.asarray(args[0] if key == "i" else args[1])
        gap = -orbital if key == "i" else orbital
        ax = active_action_batch(h2e[key], x, mc.ncas, n)
        ax += (gap - eact)[:, None] * x
        rho = float(np.linalg.norm(ax-b) / np.linalg.norm(b))
        overlap = np.vdot(x, b)
        native_overlap = complex(driver.expectation(bra, mpo, ket))
        np.testing.assert_allclose(overlap, native_overlap, atol=1e-13, rtol=1e-8)
        quadratic = complex(driver.expectation(bra, options["left_mpo"], bra))
        np.testing.assert_allclose(np.vdot(x, ax), quadratic, atol=1e-12, rtol=1e-8)
        reference_norm = float(np.vdot(reference_vector, reference_vector).real)
        measured[key] = dict(
            global_relative_residual=rho,
            global_residual_norm=float(np.linalg.norm(ax-b)),
            source_norm2=float(np.vdot(b, b).real / reference_norm),
            hylleraas_energy=float((quadratic.real - 2 * overlap.real) / reference_norm),
            overlap_energy=float(-overlap.real / reference_norm),
            quadratic_energy=float(quadratic.real / reference_norm))
        print(f"F_RESPONSE_RESIDUAL whole_{key} relative={rho:.3e} "
              f"hylleraas={measured[key]['hylleraas_energy']:.14f}", flush=True)
        return value

    driver.multiply = measure_multiply
    try:
        result = evaluate(mc, eris, pdms, *args, **kwargs)
    finally:
        driver.multiply = native_multiply
    for key in ("i", "r"):
        entries = [e for e in result[3][key]["entries"] if not e.get("zero_source")]
        assert len(entries) == 1
        entries[0].update(measured[key])
        result[3][key]["maximum_global_relative_residual"] = measured[key]["global_relative_residual"]
        verified = check_global_residual(key, measured[key]["global_relative_residual"])
        result[3][key]["global_residual_verified"] = verified
        result[3][key]["residual_status"] = "ok" if verified else "warning"
    return result


def run_pt(directory, hybrid, root, response_options=None, *, audit_response=True,
           output_directory=None):
    stage = "hybrid" if hybrid else "full"
    mc, solver, fingerprint = restore(directory)
    ranks = []
    make_rdm = u._make_rdm

    def guarded_rdm(solver, root, rank):
        if hybrid and rank >= 4:
            raise AssertionError("the no-4-RDM branch requested rank 4")
        ranks.append(rank)
        print(f"F_RDM_START rank={rank}", flush=True)
        result = make_rdm(solver, root, rank)
        print(f"F_RDM_DONE rank={rank} bytes={result.nbytes}", flush=True)
        return result

    u._make_rdm = guarded_rdm
    if hybrid:
        from socutils.mrpt import nevpt2_mps_response as response
        unmeasured_response = response.evaluate_mps_response
        if audit_response:
            response.evaluate_mps_response = partial(audited_response, unmeasured_response)
    start = time.perf_counter()
    results = {}
    canonical_diagnostics = None
    try:
        u._root_ket(solver, root)  # Validate the requested root before forming RDMs.
        pdms = tuple(u._make_rdm(solver, root, rank) for rank in range(1, 4 if hybrid else 5))
        sc_pt = WickX2CSCNEVPT2(mc)
        for name, cls in (("SC", WickX2CSCNEVPT2), ("FIC", WickX2CFICNEVPT2)):
            pt = sc_pt if name == "SC" else cls(mc)
            if hybrid:
                pt.mps_response_options = response_options
            kwargs = {}
            if name == "FIC":
                pt.canonicalized = True
                pt.mo_energy = sc_pt.mo_energy.copy()
                kwargs = dict(mo_coeff=sc_pt.mo_coeff, eris=sc_pt.eris,
                              eris_basis="semicanonical")
            pt.kernel(root=root, pdms=pdms, mps_response=hybrid,
                      contraction_backend="pytblis", **kwargs)
            if name == "SC":
                full_mc = u._full_integral_mc(mc)
                fock = pt.mo_coeff.conj().T @ full_mc.get_fock(
                    mo_coeff=pt.mo_coeff, casdm1=pdms[0]) @ pt.mo_coeff
                canonical_diagnostics = {}
                for label, indices in (("core", slice(0, mc.ncore)),
                                       ("virtual", slice(mc.ncore + mc.ncas, None))):
                    block = fock[indices, indices]
                    offdiag = float(np.linalg.norm(block - np.diag(np.diag(block))))
                    assert offdiag < 1e-8, (label, offdiag)
                    canonical_diagnostics[f"{label}_offdiagonal_norm"] = offdiag
                np.testing.assert_array_equal(
                    pt.mo_coeff[:, mc.ncore:mc.ncore + mc.ncas],
                    mc.mo_coeff[:, mc.ncore:mc.ncore + mc.ncas])
            results[name] = dict(reference_energy=float(pt.reference_energy),
                                 e_corr=float(pt.e_corr), e_tot=float(pt.e_tot),
                                 sub_eners=pt.sub_eners, approximation=pt.approximation,
                                 reference_energy_diagnostics=pt.reference_energy_diagnostics,
                                 response_diagnostics=pt.mps_response_diagnostics,
                                 times=pt.sub_times)
            print(f"F_PT_RESULT stage={stage} method={name} "
                  f"E0={pt.reference_energy:.12f} E2={pt.e_corr:.12f} "
                  f"Etot={pt.e_tot:.12f}", flush=True)
        data = dict(stage=stage, root=root, fingerprint=fingerprint,
                    requested_rdm_ranks=ranks, rdm_bytes=sum(x.nbytes for x in pdms),
                    residual_audit_enabled=bool(hybrid and audit_response),
                    wall_seconds=time.perf_counter() - start, results=results,
                    semicanonical_diagnostics=canonical_diagnostics,
                    peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20)
        output_directory = directory if output_directory is None else output_directory
        output_directory.mkdir(parents=True, exist_ok=True)
        save_json(output_directory / f"{stage}.json", data)
    finally:
        u._make_rdm = make_rdm
        if hybrid:
            response.evaluate_mps_response = unmeasured_response
        solver.close()


def compare(directory):
    full = json.loads((directory / "full.json").read_text())
    hybrid = json.loads((directory / "hybrid.json").read_text())
    assert full["fingerprint"] == hybrid["fingerprint"]
    assert hybrid["requested_rdm_ranks"] == [1, 2, 3]
    assert full["requested_rdm_ranks"] == [1, 2, 3, 4]
    result = dict(full_peak_rss_gib=full["peak_rss_gib"],
                  hybrid_peak_rss_gib=hybrid["peak_rss_gib"], methods={})
    for name in ("SC", "FIC"):
        a, b = full["results"][name], hybrid["results"][name]
        deltas = {key: b["sub_eners"][key] - value for key, value in a["sub_eners"].items()}
        unchanged = max(abs(value) for key, value in deltas.items() if key not in ("i", "r"))
        assert unchanged < 1e-10, (name, deltas)
        assert abs(a["reference_energy"] - b["reference_energy"]) < 1e-10
        accuracy = {key: {
            "maximum_global_relative_residual": b["response_diagnostics"][key].get(
                "maximum_global_relative_residual"),
            "meets_global_residual_1e8": b["response_diagnostics"][key].get(
                "global_residual_verified", False),
        } for key in ("i", "r")}
        result["methods"][name] = dict(full_e_corr=a["e_corr"], hybrid_e_corr=b["e_corr"],
                                      delta_e_corr=b["e_corr"] - a["e_corr"],
                                      class_deltas=deltas, six_class_max_delta=unchanged,
                                      response_accuracy=accuracy)
    result["response_precision_verified"] = all(
        item["meets_global_residual_1e8"] for method in result["methods"].values()
        for item in method["response_accuracy"].values())
    save_json(directory / "comparison.json", result)
    print("F_COMPARISON", json.dumps(result), flush=True)
    if not result["response_precision_verified"]:
        print("F_PRECISION_NOT_VERIFIED: energies are diagnostic; no global 1e-8 certificate", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("mcscf", "hybrid", "full", "compare"))
    parser.add_argument("--directory", type=Path, default=Path(__file__).parent / "f_sa6_ground_run")
    parser.add_argument("--root", type=int, default=0)
    parser.add_argument("--output-directory", type=Path,
                        help="save new root-specific results without overwriting the reference run")
    parser.add_argument("--response-options", type=json.loads, default=None,
                        help="explicit JSON overrides for hybrid response controls")
    parser.add_argument("--skip-residual-audit", action="store_true",
                        help="skip the expensive validation-only fixed-N expansion, not the native solve")
    args = parser.parse_args()
    lib.num_threads(THREADS)
    args.directory.mkdir(parents=True, exist_ok=True)
    if args.stage == "mcscf":
        run_mcscf(args.directory)
    elif args.stage == "compare":
        compare(args.directory)
    else:
        run_pt(args.directory, args.stage == "hybrid", args.root, args.response_options,
               audit_response=not args.skip_residual_audit,
               output_directory=args.output_directory)
