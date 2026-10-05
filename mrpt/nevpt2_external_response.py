# SPDX-License-Identifier: GPL-3.0-or-later
"""Exact external-occupation blocking of the spinor Dyall resolvent.

Only active sites occur in these MPSs. External tuples are unique ordered
sets, not IC basis labels. Block2 still owns the MPOs and Linear sweeps.
"""

from itertools import combinations, permutations
import time

import numpy as np
from pyscf.lib import logger

from . import nevpt2_mps_response as response
from . import nevpt2_utils as u


def tuple_sources(terms, ncore, nelec, holes, particles):
    """U_mu^dagger H U_0 in the ascending full-chain determinant convention.

Remove external operators right-to-left, retaining their Jordan--Wigner
signs. The surviving active strings are added COHERENTLY before any solve.
The core-normal-ordered integral blocks already include occupied-core
contractions; there are no spin/spatial degeneracy multipliers.
"""
    combined = {}
    for operators, tensor, labels in terms:
        active_ops = "".join(op for op, label in zip(operators, labels) if label == "A")
        coefficient = np.zeros((tensor.shape[labels.index("A")],) * len(active_ops), complex) if active_ops else np.array(0j)
        for core_order in permutations(holes):
            for virtual_order in permutations(particles):
                ci, vi = iter(core_order), iter(virtual_order)
                assigned = [next(ci) if label == "I" else next(vi) if label == "E" else None
                            for label in labels]
                core, external, nactive, sign = (1 << ncore) - 1, 0, nelec, 1
                for op, label, site in reversed(tuple(zip(operators, labels, assigned))):
                    if label == "A":
                        sign *= (-1)**core.bit_count()
                        nactive += 1 if op == "C" else -1
                    elif label == "I":
                        assert op == "D" and (core >> site & 1)
                        sign *= (-1)**(core & ((1 << site) - 1)).bit_count()
                        core ^= 1 << site
                    else:
                        assert op == "C" and not (external >> site & 1)
                        sign *= (-1)**(core.bit_count() + nactive + (external & ((1 << site) - 1)).bit_count())
                        external ^= 1 << site
                index = tuple(slice(None) if site is None else site for site in assigned)
                coefficient += sign * tensor[index]
        combined[active_ops] = combined.get(active_ops, 0.) + coefficient
    return combined


def _source_mpo(driver, coefficients, scale):
    builder = driver.expr_builder()
    sites = np.arange(driver.n_sites)
    for operators, tensor in coefficients.items():
        if operators:
            response._add_tensor(builder, operators, tensor / scale, [sites] * len(operators))
        else:
            # GeneralMPO rejects a constant-only Hamiltonian; Block2's own
            # get_identity_mpo uses this empty operator string instead.
            builder.add_term("", [], complex(tensor) / scale)
    return driver.get_mpo(builder.finalize(), cutoff=0., add_ident=False, iprint=0)


def apply_mpo(mpo, mps):
    """Untruncated pyblock2 tensor product, QR'd as it is formed.

Same contractions/fermion signs as algebra.MPO.__matmul__. Consume the
right QR factor BEFORE allocating the next product tensor: no fitted MPS,
SVD threshold, CI expansion or product-bond-squared temporary storage.
Only the one-site tensors exported by response._active_mps are accepted.
"""
    from pyblock2.algebra.core import MPS, Tensor, SubTensor
    if any(t is None or t.rank != (2 if i in (0, mps.n_sites - 1) else 3)
           for i, t in enumerate(mps.tensors)):
        raise ValueError("streamed audit requires one-site MPS tensors")
    maps = []
    for ob, sb in zip(mpo.get_bond_dims(), mps.get_bond_dims()):
        mapping, sizes = {}, {}
        for qo, no in ob.items():
            for qs, ns in sb.items():
                q = qo + qs
                mapping[qo, qs] = (q, sizes.get(q, 0), no, ns)
                sizes[q] = sizes.get(q, 0) + no * ns
        maps.append((mapping, sizes))
    tensors, factor = [None] * mps.n_sites, None
    for i in range(mps.n_sites - 1, -1, -1):
        if i == 0:
            local = Tensor.contract(mpo.tensors[i], mps.tensors[i], [1], [0])
            tensors[i] = Tensor.contract(local, factor, [1, 2], [0, 1])
            continue
        ot = (0, 2, 1) if i == mps.n_sites - 1 else (0, 3, 1, 2, 4)
        local = Tensor.contract(mpo.tensors[i], mps.tensors[i], [2], [1], 1, 0, out_trans=ot)
        if factor is not None:
            local = Tensor.contract(local, factor, [3, 4], [0, 1])
        mapping, sizes = maps[i - 1]
        merged = {}
        for block in local.blocks:
            qo, qs, *physical = block.q_labels
            q, offset, no, ns = mapping[qo, qs]
            labels = (q, *physical)
            if labels not in merged:
                merged[labels] = SubTensor(labels, np.zeros(
                    (sizes[q], *block.reduced.shape[2:]), complex))
            merged[labels].reduced[offset:offset + no * ns] += block.reduced.reshape(
                (no * ns, *block.reduced.shape[2:]))
        tensors[i] = Tensor(list(merged.values()))
        qr = tensors[i].right_canonicalize()
        factor = Tensor([SubTensor((qo, qs, q), qr[q][offset:offset + no * ns].reshape(
            no, ns, -1)) for (qo, qs), (q, offset, no, ns) in mapping.items() if q in qr])
    result = MPS(tensors)
    return result + mpo.const_e * mps if mpo.const_e != 0 else result


def evaluate_external(driver, active_mps, eris, order, core, virtual, eactive,
                      controls, bond, mc):
    """One fixed H_A, full active particle sectors, no (0,0) tuple.

All source norms are measured without truncation and normalized before
native Linear. Explicit global vector residuals are opt-in, just as for the
full-chain path. No weak nonzero source is screened.
"""
    from . import x2cucnevpt2 as uc
    nelec, ncas = int(driver.target.n), eris.ncas
    reference = None
    energies, diagnostics = {}, {}
    h = eris.get_h1eff("AA")[np.ix_(order, order)]
    # get_phys is shared by dense and exact block-storage containers.
    g = eris.get_phys("AAAA").transpose(0, 2, 1, 3)[np.ix_(order, order, order, order)]
    # Constant is shifted per tuple; h and g NEVER depend on its holes.
    hactive = uc._qc_mpo(driver, h, g, -eactive)
    left = driver.bw.bs.IdentityAddedMPO(hactive)
    normalized_reference = active_mps / uc._norm(active_mps.deep_copy())
    reference_bond = max(sum(x.values()) for x in normalized_reference.get_bond_dims())
    # One reference check enables the certified ijrs shortcut even without
    # full response audits. Disabling diagnostics must not add thousands of
    # redundant sweeps; a failed check still falls through to native Linear.
    check_reference = controls.diagnostic or (
        eris.ncore >= 2 and eris.nvirt >= 2 and reference_bond <= bond)
    active_operator = uc._algebra_mpo(hactive) if check_reference else None
    reference_defect = apply_mpo(active_operator, normalized_reference) if check_reference else None
    reference_residual = uc._norm(reference_defect) if reference_defect is not None else None
    reference_stationarity = ((normalized_reference.conj() @ reference_defect).real
                              if reference_defect is not None else None)
    try:
        reference = response._embed_reference(driver, normalized_reference, 0, 0)
        for key in u.SUBSPACE_ORDER:
            started = time.perf_counter()
            nh, npart = sum(key.count(x) for x in "ij"), sum(key.count(x) for x in "rs")
            ntarget = nelec + nh - npart
            entries, energy = [], 0.
            terms = response._source_tensors(eris, order, key)
            if 0 <= ntarget <= ncas:
                for holes in combinations(range(eris.ncore), nh):
                    for particles in combinations(range(eris.nvirt), npart):
                        start = time.perf_counter()
                        coefficients = tuple_sources(terms, eris.ncore, nelec, holes, particles)
                        scale = max((u._maximum_abs(t) for t in coefficients.values()), default=0.)
                        gap = float(sum(virtual[list(particles)]) - sum(core[list(holes)]))
                        entry = dict(holes=list(holes), particles=list(particles), active_electrons=ntarget,
                                     external_shift=gap, source_norm2=0., residual_norm2=0.,
                                     global_relative_residual=0., converged=True, zero_source=True,
                                     projected_energy=0., hylleraas_energy=0.)
                        state = source = None
                        try:
                            # For ijrs the source is exactly c*Psi_A. Test the
                            # known candidate x=c*Psi_A/Delta in the ORIGINAL
                            # full active equation. A finite reference is NOT
                            # assumed to be an eigenstate: its measured defect
                            # supplies this candidate's true relative residual.
                            # Fall through to native Linear whenever it fails,
                            # is unmeasured, or exceeds the requested M cap.
                            analytic = (key == "ijrs" and scale != 0. and gap != 0.
                                        and reference_residual is not None and reference_bond <= bond
                                        and reference_residual / abs(gap) <= controls.global_residual_tol)
                            if analytic:
                                norm2 = float(abs(complex(coefficients[""]))**2)
                                rho = reference_residual / abs(gap)
                                projected = -norm2 / gap
                                functional = projected + norm2 * reference_stationarity / gap**2
                                entry.update(zero_source=False, source_norm2=norm2,
                                    residual_norm2=float(norm2 * rho**2), global_relative_residual=float(rho),
                                    projected_energy=float(projected), hylleraas_energy=float(functional),
                                    response_bond_dimension=reference_bond, actual_sweeps=0, converged=True,
                                    solver="analytic ijrs candidate certified by original active-space vector residual")
                                energy += functional
                            elif scale != 0.:
                                source = _source_mpo(driver, coefficients, scale)
                                # Exact active-only source product, never a determinant expansion.
                                b = apply_mpo(uc._algebra_mpo(source), normalized_reference)
                                sn = uc._norm(b)
                                if sn != 0.:
                                    b = b / sn
                                    physical_norm = scale * sn
                                    right = driver.bw.bs.IdentityAddedMPO(source * (1. / sn))
                                    left.const_e = -eactive + gap
                                    state = driver.get_random_mps(response._response_tag(), bond_dim=bond,
                                        target=driver.bw.SX(ntarget, 0, 0), dot=2)
                                    reported, sweeps, discarded = uc._linear_response(
                                        driver, state, reference, left, right, controls, bond, 0)
                                    if not np.isfinite(reported):
                                        raise RuntimeError(f"nonfinite {key} tuple response: {holes}, {particles}")
                                    y = response._active_mps(driver, state)
                                    bx = b.conj() @ y
                                    xx = driver.expectation(state, left, state)
                                    projected = -physical_norm**2 * bx.real
                                    functional = physical_norm**2 * (xx.real - 2 * bx.real)
                                    rho = None
                                    if controls.diagnostic:
                                        # Original H_A-E_A plus the EXTERNAL gap; no shifted denominator trick.
                                        ay = apply_mpo(active_operator, y) + gap * y
                                        rho = uc._norm(ay - b)
                                        if not np.isfinite(rho):
                                            raise RuntimeError("nonfinite external-tuple residual")
                                    entry.update(zero_source=False, source_norm2=float(physical_norm**2),
                                        residual_norm2=None if rho is None else float(physical_norm**2 * rho**2),
                                        global_relative_residual=rho,
                                        projected_energy=float(projected), hylleraas_energy=float(functional),
                                        overlap=[float(physical_norm**2 * bx.real), float(physical_norm**2 * bx.imag)],
                                        quadratic=[float(physical_norm**2 * xx.real), float(physical_norm**2 * xx.imag)],
                                        response_bond_dimension=max(sum(x.values()) for x in y.get_bond_dims()),
                                        actual_sweeps=sweeps, discarded_weights=discarded,
                                        converged=rho is not None and rho <= controls.global_residual_tol)
                                    energy += functional
                        finally:
                            if state is not None:
                                response._release_response_mps(driver, state)
                            del source
                        entry["wall_seconds"] = time.perf_counter() - start
                        entries.append(entry)
                        if len(entries) == 1 or len(entries) % 100 == 0:
                            logger.info(mc, "UC tuple %s %d: E2=%.14f last rho=%s wall=%.3f s",
                                key, len(entries), energy, entry["global_relative_residual"],
                                time.perf_counter() - started)
            measured = all(e["global_relative_residual"] is not None for e in entries)
            source2 = sum(e["source_norm2"] for e in entries)
            residual2 = sum(e["residual_norm2"] for e in entries) if measured else None
            worst = max(entries, key=lambda e: e["global_relative_residual"]) if entries and measured else None
            energies[key] = float(energy)
            diagnostics[key] = dict(tuples=entries, tuple_count=len(entries), source_norm2=source2,
                residual_norm2=residual2, global_relative_residual=(
                    np.sqrt(residual2 / source2) if source2 else 0.) if measured else None,
                max_channel_relative_residual=(0. if measured else None) if worst is None else worst["global_relative_residual"],
                worst_tuple=None if worst is None else {k: worst[k] for k in ("holes", "particles", "global_relative_residual")},
                converged=all(e["converged"] for e in entries), hylleraas_energy=float(energy),
                projected_energy=sum(e["projected_energy"] for e in entries),
                wall_seconds=time.perf_counter() - started, controls=controls.__dict__,
                solver="pyblock2.Linear.solve/Automatic", space_restriction="full active particle sector",
                projector="fixed external occupation; (0,0) never enumerated",
                reference_active_residual_norm=reference_residual,
                positive_definiteness="not certified")
            logger.note(mc, "E(X2CUCNEVPT2-%s) = %.14f  tuples = %d  residual = %s",
                        key, energy, len(entries), diagnostics[key]["global_relative_residual"])
    finally:
        if reference is not None:
            response._release_response_mps(driver, reference)
    return energies, diagnostics
