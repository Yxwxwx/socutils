# SPDX-License-Identifier: GPL-3.0-or-later
"""Fixed-N coefficient measurements of solved UC MPSs, for F validation only.

No response is solved here. Spectator occupations are contracted exactly;
only the small active determinant sector is expanded, one pattern at a time.
The production API does not import this exponential diagnostic oracle.
"""

from itertools import combinations, permutations, product
from functools import lru_cache
import time

import numpy as np
from pyscf.fci import cistring, fci_dhf_slow

from socutils.mrpt import nevpt2_utils as u, x2cucnevpt2 as uc
from .f_atom import active_action_batch


@lru_cache(maxsize=8)
def active_bits(ncas, nelec):
    return np.sum(1 << cistring.gen_occslst(range(ncas), nelec), axis=1).astype(np.uint32)


def time_reverse_coefficients(matrix, vectors, nelec):
    """Apply wedge^N(matrix) K in fixed-N SGF order, for reference checks.

    Vectors are columns (or one 1-D vector). Second quantization of log(U)
    and sparse expm action avoid constructing the dense exterior-power U.
    This is an exponential-size coefficient diagnostic, never a PT solver.
    """
    from scipy.linalg import expm, logm
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import expm_multiply
    matrix = np.asarray(matrix, complex)
    norb = len(matrix)
    if matrix.shape != (norb, norb) or norb > 20:
        raise ValueError("the reference coefficient audit needs a square map of at most 20 spinors")
    np.testing.assert_allclose(matrix.conj().T @ matrix, np.eye(norb), atol=1e-10, rtol=0.)
    generator = logm(matrix)
    np.testing.assert_allclose(expm(generator), matrix, atol=1e-10, rtol=0.)
    np.testing.assert_allclose(generator + generator.conj().T, 0., atol=1e-10, rtol=0.)
    links = cistring.gen_linkstr_index(range(norb), nelec)
    dimension, nlink = links.shape[:2]
    p, q, target, sign = links.reshape(-1, 4).T
    origin = np.repeat(np.arange(dimension), nlink)
    action = coo_matrix((generator[p, q] * sign, (target, origin)),
                        shape=(dimension, dimension)).tocsr()
    vectors = np.asarray(vectors, complex)
    if vectors.ndim not in (1, 2) or vectors.shape[0] != dimension:
        raise ValueError("vectors must have the fixed-N SGF dimension as their first axis")
    return expm_multiply(action, vectors.conj(), traceA=action.diagonal().sum())


class SectorCoefficients:
    """Exact tensor contraction, checked against native DeterminantTRIE."""

    def __init__(self, mps, ncore, nvirt, nelec):
        self.nc, self.nv, self.total = ncore, nvirt, nelec
        self.na = mps.n_sites - ncore - nvirt
        if not ncore or not nvirt or self.na > 20:
            raise ValueError("this diagnostic is for small CAS with nonempty core/virtual spaces")
        self.matrices = []
        for site, tensor in enumerate(mps.tensors):
            matrices = {}
            for block in tensor.blocks:
                labels, array = block.q_labels, block.reduced
                if site == 0:
                    ql, qm, qr = 0, labels[0].n, labels[1].n
                    array = array[None, ...]
                elif site == mps.n_sites - 1:
                    ql, qm, qr = labels[0].n, labels[1].n, nelec
                    array = array[..., None]
                else:
                    ql, qm, qr = (q.n for q in labels)
                assert qr == ql + qm and array.shape[1] == 1
                assert (ql, qm) not in matrices, "the coefficient oracle requires C1 SGF"
                matrices[ql, qm] = array[:, 0, :]
            self.matrices.append(matrices)
        self.rows = {}
        # Empty virtual suffixes and prefix maps with 1/2 particles still
        # to their right. Their products are bounded by the solved MPS M.
        self.empty = [None] * (nvirt + 1)
        self.empty[-1] = np.ones(1, complex)
        for r in range(nvirt - 1, -1, -1):
            matrix = self.matrices[ncore + self.na + r].get((nelec, 0))
            self.empty[r] = matrix @ self.empty[r + 1] if matrix is not None and self.empty[r + 1] is not None else None
        self.prefix = {}
        for particles in (1, 2):
            q = nelec - particles
            first = next((x for (ql, _), x in self.matrices[ncore + self.na].items() if ql == q), None)
            prefix = [np.eye(first.shape[0], dtype=complex) if first is not None else None]
            for r in range(nvirt):
                matrix = self.matrices[ncore + self.na + r].get((q, 0))
                prefix.append(prefix[-1] @ matrix if matrix is not None and prefix[-1] is not None else None)
            self.prefix[particles] = prefix
        self.tails = {(): self.empty[0]}
        for r in range(nvirt):
            matrix = self.matrices[ncore + self.na + r].get((nelec - 1, 1))
            vector = matrix @ self.empty[r + 1] if matrix is not None and self.empty[r + 1] is not None else None
            prefix = self.prefix[1][r]
            self.tails[r,] = prefix @ vector if prefix is not None and vector is not None else None
            for l in range(r - 1, -1, -1):
                matrix = self.matrices[ncore + self.na + l].get((nelec - 2, 1))
                prefix = self.prefix[2][l]
                self.tails[l, r] = prefix @ matrix @ vector if prefix is not None and matrix is not None and vector is not None else None
                zero = self.matrices[ncore + self.na + l].get((nelec - 1, 0))
                vector = zero @ vector if zero is not None and vector is not None else None

    def values(self, holes, particles):
        holes, particles = tuple(holes), tuple(particles)
        nactive = self.total - self.nc + len(holes) - len(particles)
        bits = active_bits(self.na, nactive)
        tail = self.tails[particles]
        if tail is None:
            return np.zeros(len(bits), complex)
        key = holes, len(particles)
        if key not in self.rows:
            q, left = 0, np.ones(1, complex)
            for i in range(self.nc):
                occupation = int(i not in holes)
                matrix = self.matrices[i].get((q, occupation))
                left = left @ matrix if left is not None and matrix is not None else None
                q += occupation
            rows = {q: (np.zeros(1, np.uint32), left[None, :])} if left is not None else {}
            for a in range(self.na):
                collected = {}
                for (ql, qm), matrix in self.matrices[self.nc + a].items():
                    qr = ql + qm
                    remaining = nactive - (qr - q)
                    if ql not in rows or not 0 <= remaining <= self.na - a - 1:
                        continue
                    previous_bits, previous = rows[ql]
                    collected.setdefault(qr, []).append((previous_bits | (qm << a), previous @ matrix))
                rows = {qr: (np.concatenate([x[0] for x in parts]), np.concatenate([x[1] for x in parts]))
                        for qr, parts in collected.items()}
            coefficients = np.zeros((len(bits), len(tail)), complex)
            if rows:
                actual_bits, values = rows[self.total - len(particles)]
                addresses = np.full(1 << self.na, -1, int)
                addresses[bits] = np.arange(len(bits))
                coefficients[addresses[actual_bits]] = values
            self.rows[key] = coefficients
        return self.rows[key] @ tail


class Sources:
    """Coherent CAS-ket action of the tested source blocks, with fermion signs."""

    def __init__(self, eris, order, reference, nelec):
        self.nc, self.na, self.nv = eris.ncore, eris.ncas, eris.nvirt
        self.reference = reference
        self.nelec = nelec
        assert len(reference) == len(active_bits(self.na, nelec))
        self.terms = {key: uc.response._source_tensors(eris, order, key) for key in u.SUBSPACE_ORDER}
        self.bases = {}

    def basis(self, operators):
        if operators not in self.bases:
            source_bits = active_bits(self.na, self.nelec)
            ntarget = self.nelec + operators.count("C") - operators.count("D")
            target_bits = active_bits(self.na, ntarget)
            addresses = np.full(1 << self.na, -1, int)
            addresses[target_bits] = np.arange(len(target_bits))
            parity = np.array([(-1)**int(x).bit_count() for x in range(1 << self.na)])
            basis = np.zeros((self.na**len(operators), len(target_bits)), complex)
            for row, sites in enumerate(product(range(self.na), repeat=len(operators))):
                updated, sign, valid = source_bits.copy(), np.ones(len(source_bits)), np.ones(len(source_bits), bool)
                for operator, site in reversed(tuple(zip(operators, sites))):
                    occupied = (updated & (1 << site)) != 0
                    valid &= ~occupied if operator == "C" else occupied
                    sign *= parity[updated & ((1 << site) - 1)]
                    updated ^= 1 << site
                basis[row, addresses[updated[valid]]] = sign[valid] * self.reference[valid]
            self.bases[operators] = basis
        return self.bases[operators]

    def values(self, key, holes, particles):
        ntarget = self.nelec + len(holes) - len(particles)
        result = np.zeros(len(active_bits(self.na, ntarget)), complex)
        for operators, tensor, labels in self.terms[key]:
            active_operators = "".join(op for op, label in zip(operators, labels) if label == "A")
            coefficients = np.zeros(self.na**len(active_operators), complex)
            for core_order in permutations(holes):
                for virtual_order in permutations(particles):
                    assigned, ci, vi = [], iter(core_order), iter(virtual_order)
                    for label in labels:
                        assigned.append(next(ci) if label == "I" else next(vi) if label == "E" else None)
                    core, external, nactive, sign = (1 << self.nc) - 1, 0, self.nelec, 1
                    for op, label, site in reversed(tuple(zip(operators, labels, assigned))):
                        if label == "A":
                            sign *= (-1)**core.bit_count()
                            nactive += 1 if op == "C" else -1
                        elif label == "I":
                            assert op == "D"
                            sign *= (-1)**(core & ((1 << site) - 1)).bit_count()
                            core ^= 1 << site
                        else:
                            assert op == "C"
                            sign *= (-1)**(core.bit_count() + nactive + (external & ((1 << site) - 1)).bit_count())
                            external ^= 1 << site
                    index = tuple(slice(None) if site is None else site for site in assigned)
                    coefficients += sign * np.asarray(tensor[index]).reshape(-1)
            result += coefficients @ self.basis(active_operators)
        return result


def measure_response(driver, state, reference, left, right, eris, order, core, virtual, eactive):
    """Measure the actual solved MPS in all eight sectors AND the CAS sector."""
    total = int(driver.target.n)
    nc, na, nv = eris.ncore, eris.ncas, eris.nvirt
    coefficients = SectorCoefficients(uc.response._active_mps(driver, state), nc, nv, total)
    ref = SectorCoefficients(uc.response._active_mps(driver, reference), nc, nv, total).values((), ())
    sources = Sources(eris, order, ref, total - nc)
    ecore = .5 * np.trace((eris.h1e + eris.h1eff)[:nc, :nc]).real
    source_scale = complex(right.const_e) / (-eactive - ecore)
    assert abs(source_scale.imag) < 1e-13 and source_scale.real > 0
    scale = 1. / (source_scale.real * np.linalg.norm(ref))
    h = eris.get_h1eff("AA")[np.ix_(order, order)]
    g = eris.get_chem("AAAA")[np.ix_(order, order, order, order)]
    entries = {}
    for key in (*u.SUBSPACE_ORDER, "CAS"):
        start = time.perf_counter()
        nh = sum(key.count(x) for x in "ij") if key != "CAS" else 0
        npart = sum(key.count(x) for x in "rs") if key != "CAS" else 0
        nelec = total - nc + nh - npart
        absorbed = fci_dhf_slow.absorb_h1e(h, g, na, nelec, .5)
        norm2 = residual2 = state2 = 0.
        overlap = quadratic = 0j
        patterns = 0
        nonzero_patterns, worst = 0, None
        zero_source_residual2 = 0.
        print(f"UC_COEFFICIENT_AUDIT_START {key}", flush=True)
        pending = []

        def batch(patterns):
            nonlocal nonzero_patterns, worst, zero_source_residual2
            y = np.array([coefficients.values(holes, particles) for holes, particles in patterns])
            if key == "CAS":
                b = (active_action_batch(absorbed, ref[None, :], na, nelec) - eactive * ref) * source_scale
            else:
                b = np.array([sources.values(key, holes, particles) for holes, particles in patterns]) * source_scale
            gaps = np.array([sum(virtual[list(particles)]) - sum(core[list(holes)]) - eactive
                             for holes, particles in patterns])
            ly = -(active_action_batch(absorbed, y, na, nelec) + gaps[:, None] * y)
            bn = np.sum(abs(b)**2, axis=1)
            rn = np.sum(abs(ly - b)**2, axis=1)
            nonzero = np.flatnonzero(bn > 0.)  # No weak-source screening.
            nonzero_patterns += len(nonzero)
            zero_source_residual2 += float(rn[bn == 0.].sum())
            if len(nonzero):
                rhos = np.sqrt(rn[nonzero] / bn[nonzero])
                k = int(nonzero[np.argmax(rhos)])
                rho = float(np.max(rhos))
                if worst is None or rho > worst["relative_residual"]:
                    holes, particles = patterns[k]
                    worst = dict(holes=list(holes), particles=list(particles),
                                 relative_residual=rho,
                                 source_norm2=float(scale**2 * bn[k]),
                                 residual_norm2=float(scale**2 * rn[k]))
            return (float(bn.sum()), float(rn.sum()),
                    np.vdot(y, y).real, np.vdot(y, b), np.vdot(y, ly))

        for holes in combinations(range(nc), nh):
            for particles in combinations(range(nv), npart):
                pending.append((holes, particles))
                if len(pending) == 4:
                    bn, rn, yn, yb, yl = batch(pending)
                    norm2, residual2, state2 = norm2 + bn, residual2 + rn, state2 + yn
                    overlap, quadratic = overlap + yb, quadratic + yl
                    patterns += len(pending)
                    pending.clear()
                    if patterns % 512 == 0:
                        print(f"UC_COEFFICIENT_AUDIT_PROGRESS {key} patterns={patterns}", flush=True)
        if pending:
            bn, rn, yn, yb, yl = batch(pending)
            norm2, residual2, state2 = norm2 + bn, residual2 + rn, state2 + yn
            overlap, quadratic = overlap + yb, quadratic + yl
            patterns += len(pending)
        rho = np.sqrt(residual2 / norm2) if norm2 else (0. if residual2 == 0. else np.inf)
        entries[key] = dict(source_norm2=float(scale**2 * norm2), residual_norm2=float(scale**2 * residual2),
                            state_norm2=float(state2), global_relative_residual=float(rho),
                            overlap=[float(overlap.real), float(overlap.imag)],
                            quadratic=[float(quadratic.real), float(quadratic.imag)],
                            patterns=patterns, wall_seconds=time.perf_counter() - start,
                            nonzero_patterns=nonzero_patterns, worst_pattern=worst,
                            max_pattern_relative_residual=worst["relative_residual"] if worst else 0.,
                            zero_source_residual_norm2=float(scale**2 * zero_source_residual2),
                            hylleraas_energy=float(scale**2 * (-quadratic.real + 2 * overlap.real)))
        print(f"UC_COEFFICIENT_AUDIT {key} residual={rho:.6e} E2={entries[key]['hylleraas_energy']:.14f}", flush=True)
    overlap = sum(complex(*d["overlap"]) for d in entries.values())
    quadratic = sum(complex(*d["quadratic"]) for d in entries.values())
    np.testing.assert_allclose(overlap, driver.expectation(state, right, reference), atol=1e-10, rtol=1e-9)
    np.testing.assert_allclose(quadratic, driver.expectation(state, left, state), atol=1e-10, rtol=1e-9)
    norm2 = sum(d["source_norm2"] for d in entries.values())
    rho = np.sqrt(sum(d["residual_norm2"] for d in entries.values()) / norm2)
    return dict(classes=entries, global_relative_residual=float(rho),
                max_external_relative_residual=max(d["global_relative_residual"]
                                                   for key, d in entries.items() if key != "CAS"),
                max_external_pattern_relative_residual=max(d["max_pattern_relative_residual"]
                                                           for key, d in entries.items() if key != "CAS"),
                hylleraas_energy=sum(d["hylleraas_energy"] for d in entries.values()),
                method="test-only fixed-N coefficient audit of solved MPS; CAS included")
