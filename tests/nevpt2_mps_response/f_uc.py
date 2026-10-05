# SPDX-License-Identifier: GPL-3.0-or-later
"""Restore the retained six-root F reference; preflight or run full UC.

Set UC_REFERENCE to the retained f_sa6_ground_run directory. This script
does not change it. UC_PREFLIGHT defaults to 1; set it to 0 for a solve.
UC_ROOT, UC_M, UC_SWEEPS, UC_THRESHOLD select a fixed-reference convergence point.
UC_M may be comma-separated (e.g. 128,256,512): all values reuse exactly the
same prepared RDMs, semicanonical orbitals and integral blocks in one process.
UC_DIAGNOSTIC=1 requests the expensive post-solve global residual audit;
UC_COEFFICIENT_AUDIT=1 measures the solved MPS by the independent, F-only
fixed-N audit. Its certificate is reported separately from the PT API's.
UC_STRICT_CAS=1 selects the separate fully projected validation solver.
UC_RESPONSE_MODE=external_tuples selects active-only occupation blocks.
UC_TUPLE_PILOT=1 limits the TEST RUNNER to one tuple per class; its result
is labelled PILOT, never a full UC energy. UC_TOL sets native sweep stopping.
Use a unique PYSCF_TMPDIR on local scratch, and ulimit -s unlimited.
"""

import hashlib
import json
import os
import resource
import time
from pathlib import Path

import numpy as np
from pyscf import lib

from socutils.mrpt import X2CUCNEVPT2, nevpt2_utils as u, x2cucnevpt2 as uc
from . import f_atom


def product_storage_bound(mpo, mps):
    """Upper bound for uncompressed SGF tensor-product storage, no allocation.

    Follows pyblock2.algebra.MPO.__matmul__/MPS.merge_virtual_dims shapes.
    Unreachable bond sectors can make this conservative. It is NOT a peak
    RSS estimate: Q, QR, residual construction, and environments need more.
    """
    bonds = []
    for operator, state in zip(mpo.get_bond_dims(), mps.get_bond_dims()):
        combined = {}
        for qo, no in operator.items():
            for qs, ns in state.items():
                q = qo.n + qs.n
                combined[q] = combined.get(q, 0) + no * ns
        bonds.append(combined)
    sizes = []
    for site, (operator, state) in enumerate(zip(mpo.tensors, mps.tensors)):
        first, last = site == 0, site == mps.n_sites - 1
        blocks = {}
        for a in operator.blocks:
            for b in state.blocks:
                if a.q_labels[1 if first else 2] != b.q_labels[0 if first else 1]:
                    continue
                left = None if first else a.q_labels[0].n + b.q_labels[0].n
                right = None if last else a.q_labels[-1].n + b.q_labels[-1].n
                physical = a.q_labels[0 if first else 1].n
                blocks[left, physical, right] = (
                    (1 if first else bonds[site - 1][left])
                    * a.reduced.shape[0 if first else 1]
                    * (1 if last else bonds[site][right]))
        sizes.append(16 * sum(blocks.values()))
    return dict(total_tensor_gib=sum(sizes) / 2**30,
                largest_tensor_gib=max(sizes) / 2**30,
                maximum_bond_dimension=max(sum(x.values()) for x in bonds))


if __name__ == "__main__":
    f_atom.THREADS = int(os.environ.get("OMP_NUM_THREADS", "8"))
    f_atom.STACK_MB = 8192
    lib.num_threads(f_atom.THREADS)
    directory = Path(os.environ["UC_REFERENCE"]).resolve()
    metadata = json.loads((directory / "mcscf.json").read_text())
    assert metadata["nroots"] == 6 and not metadata["cd"] and not metadata["kr"]
    root = int(os.environ.get("UC_ROOT", "0"))
    bonds = [int(value) for value in os.environ.get("UC_M", "128").split(",")]
    sweeps = int(os.environ.get("UC_SWEEPS", "6"))
    options = dict(max_bond_dimension=bonds[0], n_sweeps=sweeps,
                   tol=float(os.environ.get("UC_TOL", "0")),
                   linear_threshold=float(os.environ.get("UC_THRESHOLD", "1e-24")),
                   diagnostic=os.environ.get("UC_DIAGNOSTIC", "0") == "1",
                   strict_cas=os.environ.get("UC_STRICT_CAS", "0") == "1")
    coefficient_audit = os.environ.get("UC_COEFFICIENT_AUDIT", "0") == "1"
    response_mode = os.environ.get("UC_RESPONSE_MODE", "full_chain")
    pilot = os.environ.get("UC_TUPLE_PILOT", "0") == "1"
    if pilot:
        if response_mode != "external_tuples":
            raise ValueError("UC_TUPLE_PILOT requires external_tuples")
        from itertools import islice
        from socutils.mrpt import nevpt2_external_response
        original_combinations = nevpt2_external_response.combinations
        nevpt2_external_response.combinations = lambda *args: islice(original_combinations(*args), 1)
        print("F_UC_PILOT: one tuple per class; NOT a complete UC-NEVPT2 energy", flush=True)
    if coefficient_audit and response_mode != "full_chain":
        raise ValueError("the independent full-chain coefficient audit requires UC_RESPONSE_MODE=full_chain")
    if coefficient_audit and (options["diagnostic"] or options["strict_cas"]):
        raise ValueError("the F coefficient audit hooks the native unconstrained solve; set UC_DIAGNOSTIC=0 and UC_STRICT_CAS=0")
    # A process-local guard on gpu01; never risk exhausting the entire node.
    resource.setrlimit(resource.RLIMIT_AS, (400 * 2**30, 400 * 2**30))
    start = time.perf_counter()
    mc, solver, fingerprint = f_atom.restore(directory)
    try:
        print("F_UC_REFERENCE", fingerprint, "root", root, flush=True)
        from socutils.mrpt import nevpt2_external_response
        print("F_UC_IMPLEMENTATION", json.dumps(dict(options=options, coefficient_audit=coefficient_audit,
            reference_threads=f_atom.THREADS,
            pt_threads=int(os.environ.get("UC_THREADS", str(f_atom.THREADS))), sha256={
            Path(module.__file__).name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            for module in (uc, uc.response, nevpt2_external_response)}, response_mode=response_mode)), flush=True)
        pdms = tuple(u._make_rdm(solver, root, rank) for rank in (1, 2))
        mo, eps = u.semicanonicalize(mc, mc.mo_coeff, pdms[0], root)
        eris = u._dense_eris_from_mc(mc, mo)
        print("F_UC_PREPARED_INPUT", json.dumps({name: hashlib.sha256(
            np.ascontiguousarray(array).tobytes()).hexdigest()
            for name, array in (("mo", mo), ("eps", eps), ("rdm1", pdms[0]), ("rdm2", pdms[1]))}), flush=True)
        driver, reference = u._root_ket(solver, root)
        active = uc.response._active_mps(driver, reference)
        reference_dimensions = dict(
            M0_requested=int(solver.max_bond_dimension),
            M0_actual=max(sum(x.values()) for x in active.get_bond_dims()),
            M1_requested=bonds[0])
        print("F_UC_BOND_DIMENSIONS", json.dumps(reference_dimensions), flush=True)
        eactive = float((np.einsum("pq,pq", eris.get_h1eff("AA"), pdms[0]) + .5 * np.einsum(
            "pqrs,pqsr", eris.get_phys("AAAA"), pdms[1])).real)
        order = np.arange(mc.ncas) if driver.reorder_idx is None else np.asarray(driver.reorder_idx)
        if os.environ.get("UC_PREFLIGHT", "1") == "1" and response_mode == "external_tuples":
            from math import comb
            counts = {key: comb(eris.ncore, sum(key.count(x) for x in "ij")) * comb(
                eris.nvirt, sum(key.count(x) for x in "rs")) for key in u.SUBSPACE_ORDER}
            print("F_UC_TUPLE_PREFLIGHT", json.dumps(dict(
                mode=response_mode, active_sites=mc.ncas, tuple_counts=counts, total_tuples=sum(counts.values()))), flush=True)
        elif os.environ.get("UC_PREFLIGHT", "1") == "1":
            with uc._qc_problem(driver, active, eris, order,
                                eps[:mc.ncore], eps[eris.nocc:], eactive) as problem:
                embedded, source, _, _, _ = problem
                bound = product_storage_bound(uc._algebra_mpo(source), uc.response._active_mps(driver, embedded))
            print("F_UC_PREFLIGHT", json.dumps(bound), flush=True)
        else:
            for bond in bonds:
                options["max_bond_dimension"] = bond
                reference_dimensions["M1_requested"] = bond
                print("F_UC_POINT", json.dumps(dict(options=options,
                    bond_dimensions=reference_dimensions)), flush=True)
                pt = X2CUCNEVPT2(mc)
                pt.response_mode = response_mode
                pt.stack_memory = int(os.environ.get("UC_STACK_MB", "8192"))
                pt.n_threads = int(os.environ.get("UC_THREADS", str(f_atom.THREADS)))
                pt.canonicalized, pt.mo_energy = True, eps
                pt.mps_response_options = options.copy()
                point_start = time.perf_counter()
                measured = []
                native_linear = uc._linear_response
                if coefficient_audit:
                    from .uc_audit import measure_response
                    def audit_linear(driver, state, reference, left, right, *args):
                        result = native_linear(driver, state, reference, left, right, *args)
                        measured.append(measure_response(driver, state, reference, left, right,
                            eris, order, eps[:mc.ncore], eps[eris.nocc:], eactive))
                        return result
                    uc._linear_response = audit_linear
                try:
                    pt.kernel(root=root, mo_coeff=mo, pdms=pdms, eris=eris, eris_basis="semicanonical")
                finally:
                    uc._linear_response = native_linear
                if measured:
                    assert len(measured) == 1
                    np.testing.assert_allclose(measured[0]["hylleraas_energy"], pt.e_corr, atol=1e-9, rtol=0.)
                    for key in u.SUBSPACE_ORDER:
                        np.testing.assert_allclose(measured[0]["classes"][key]["hylleraas_energy"], pt.sub_eners[key], atol=1e-9, rtol=0.)
                    measured[0]["converged"] = (measured[0]["global_relative_residual"] <= 1e-8
                        and measured[0]["max_external_relative_residual"] <= 1e-8
                        and measured[0]["max_external_pattern_relative_residual"] <= 1e-8
                        and abs(pt.diagnostics["reference_energy_difference"]) <= 1e-8)
                print("F_UC_PILOT_RESULT" if pilot else "F_UC_RESULT", json.dumps(dict(
                    fingerprint=fingerprint, root=root, bond=bond, sweeps=sweeps,
                    response_mode=response_mode,
                    bond_dimensions=reference_dimensions,
                    e_corr=pt.e_corr, e_tot=pt.e_tot, converged=pt.converged,
                    point_wall_seconds=time.perf_counter() - point_start,
                    classes=pt.sub_eners, diagnostics=pt.diagnostics,
                    independent_coefficient_audit=measured[0] if measured else None)), flush=True)
    finally:
        solver.close()
        print("F_UC_RESOURCES", json.dumps(dict(
            wall_seconds=time.perf_counter() - start,
            peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20)), flush=True)
