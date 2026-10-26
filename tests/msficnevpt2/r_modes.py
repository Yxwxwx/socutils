# SPDX-License-Identifier: GPL-3.0-or-later
"""Spectral contributions and independent many-body r-mode measurements.

Read saved MS-FIC matrices, never regenerate or alter an RDM. This diagnostic
does not solve an alternative perturbation theory or enforce any symmetry.
"""
import argparse
from itertools import combinations
import json
from pathlib import Path
import pickle
import os

import numpy as np
from pyscf import lib, scf
from pyscf.fci import cistring, fci_dhf_slow
from scipy.linalg import eigh

from socutils.dmrg.kramers import ao_time_reverse, time_reverse_one_body, time_reverse_rdm1
from socutils.mrpt import x2cmsficnevpt2 as ms
from socutils.mrpt._ic_metric import congruence
from tests.nevpt2_mps_response.diagnose_r85 import _source_coefficients
from tests.nevpt2_mps_response.f_atom import active_action_batch, _active_action_links
from .ag_audit import write_json


def mode_contributions(g, denominators, shift):
    """Each (..., k, M, N) contribution in the model eigenbasis."""
    inverse = 1 / denominators
    source_response = -(g.conj()[..., :, None] * g[..., None, :]) * (
        0.5 * (inverse[..., :, None] + inverse[..., None, :]))
    norm = g.conj()[..., :, None] * g[..., None, :] * (
        inverse[..., :, None] * inverse[..., None, :])
    return source_response, -shift * norm


def doublet_decomposition(reference, corrections, norms, shift):
    """Keep traceless matrices, not a misleading sum of individual spreads."""
    def traceless(matrix):
        return matrix - np.eye(2) * np.trace(matrix) / 2

    classes = {}
    total = np.asarray(reference, complex).copy()
    for key, correction in corrections.items():
        norm_term = -shift * norms[key]
        hermitian_y = correction - norm_term
        classes[key] = dict(source_response=hermitian_y, shift_norm_correction=norm_term,
                            traceless_source_response=traceless(hermitian_y),
                            traceless_shift_norm=traceless(norm_term),
                            traceless_total=traceless(correction))
        total += correction
    return dict(classes=classes, traceless_reference=traceless(reference),
                traceless_heff=traceless(total), heff=total,
                total_spread_cm_inverse=float(2 * np.linalg.norm(traceless(total), 2) * 219474.6313705))


def partition_symmetry(prepared, time_reversal):
    """Measure the frozen SA partition; never project or rebuild its inputs."""
    nc, no = prepared.eris.ncore, prepared.eris.nocc
    active_t = time_reversal[nc:no, nc:no]
    report = dict(sa_density_time_reverse_defect=float(np.max(abs(
        prepared.sa_density - time_reverse_rdm1(active_t, prepared.sa_density)))))
    for name, part in (("inactive", slice(0, nc)), ("external", slice(no, None))):
        t = time_reversal[part, part]
        f = np.diag(prepared.mo_energy[part])
        report[name] = dict(
            subspace_unitarity_defect=float(np.max(abs(t.conj().T @ t - np.eye(len(t))), initial=0)),
            diagonal_fock_time_reverse_defect=float(np.max(abs(f - time_reverse_one_body(t, f)), initial=0)))
    return report


class RDirections:
    """Apply selected IC sums via pair-annihilated vectors, not the full B matrix.

    Columns follow the actual FIC r operators: D[a], then C[a]D[c]D[b], b<c.
    At CAS(11,22) two references need about 4 GiB annihilated vectors rather than
    the roughly 100 GiB full perturber matrix. No alpha/beta CI mapping occurs.
    """
    def __init__(self, vectors, ncas, nelec):
        self.ncas, self.nelec = ncas, nelec
        self.vectors = np.asarray(vectors)
        self.pairs = list(combinations(range(ncas), 2))
        self.dimension = ncas * (1 + len(self.pairs))
        self.bits = {n: cistring.make_strings(range(ncas), n)
                     for n in (nelec, nelec - 1, nelec - 2)}
        self.addresses = {}
        for n, bits in self.bits.items():
            addresses = np.full(1 << ncas, -1, dtype=np.int32)
            addresses[bits] = np.arange(len(bits))
            self.addresses[n] = addresses
        self.parity = np.fromiter((1 - 2 * (x.bit_count() % 2) for x in range(1 << ncas)),
                                  dtype=np.int8, count=1 << ncas)
        bits = self.bits[nelec]
        self.single = []
        self.double = []
        for vector in self.vectors.T:
            single = np.zeros((len(self.bits[nelec - 1]), ncas), complex)
            for a in range(ncas):
                origin = np.flatnonzero(bits & (1 << a))
                target = self.addresses[nelec - 1][bits[origin] ^ (1 << a)]
                single[target, a] = self.parity[bits[origin] & ((1 << a) - 1)] * vector[origin]
            double = np.zeros((len(self.bits[nelec - 2]), len(self.pairs)), complex)
            for col, (b, c) in enumerate(self.pairs):
                mask = (1 << b) | (1 << c)
                origin = np.flatnonzero((bits & mask) == mask)
                target = self.addresses[nelec - 2][bits[origin] ^ mask]
                sign = -self.parity[bits[origin] & ((1 << b) - 1)] * self.parity[
                    bits[origin] & ((1 << c) - 1)]
                double[target, col] = sign * vector[origin]
            self.single.append(single)
            self.double.append(double)

    def apply(self, coefficients):
        nout = coefficients.shape[1]
        blocks = coefficients.reshape(len(self.single), self.dimension, nout)
        result = np.zeros((len(self.bits[self.nelec - 1]), nout), complex)
        for root, c in enumerate(blocks):
            result += self.single[root] @ c[:self.ncas]
            weights = c[self.ncas:].reshape(self.ncas, len(self.pairs), nout)
            mixed = self.double[root] @ weights.transpose(1, 0, 2).reshape(len(self.pairs), -1)
            bits = self.bits[self.nelec - 2]
            for a in range(self.ncas):
                origin = np.flatnonzero((bits & (1 << a)) == 0)
                target = self.addresses[self.nelec - 1][bits[origin] | (1 << a)]
                sign = self.parity[bits[origin] & ((1 << a) - 1)]
                result[target] += sign[:, None] * mixed[origin, a*nout:(a+1)*nout]
        return result

    def source(self, eris, channel):
        one, three = _source_coefficients(eris, "r", channel)
        double = np.array([three[a, c, b] - three[a, b, c]
                           for a in range(self.ncas) for b, c in self.pairs])
        c = np.r_[one, double]
        nref = len(self.single)
        return self.apply(np.kron(np.eye(nref), c[:, None]))


def spectral_audit(prepared, shift=0.0):
    """Use precisely the retained metric basis and energy origin of MS-MR."""
    block = prepared.classes["r"]
    energies, rotation = eigh(prepared.active_reference)
    origin = float(np.mean(energies))
    centered_e = energies - origin
    print("R_MODE_METRIC_BEGIN", flush=True)
    x, _, metric = ms._metric_basis(
        block.metric, metric_atol=1e-12, metric_rcond=1e-11,
        matrix_atol=1e-10, matrix_rtol=1e-10, metric_refinement=True)
    refined = metric["gram_refinement_steps"] > 0
    dtype = np.clongdouble if refined else np.complex128
    connected = np.asarray(block.active, dtype=dtype) - origin * np.asarray(block.metric, dtype=dtype)
    f = congruence(x, connected, extended=refined, threads=lib.num_threads())
    spectrum, z = eigh(0.5 * (f + f.conj().T))
    del connected, f
    print("R_MODE_SOURCE_BEGIN", flush=True)
    source = block.source.transpose(1, 0, 2).reshape(len(block.metric), -1)
    g = (z.conj().T @ (x.conj().T @ source)).reshape(len(spectrum), -1, 2).transpose(1, 0, 2)
    g = g @ rotation
    gaps = prepared.mo_energy[prepared.eris.nocc:]
    d0 = spectrum[None, :, None] + gaps[:, None, None] - centered_e
    y, norm_term = mode_contributions(g, d0 + shift, shift)
    correction = y + norm_term
    # For a Hermitian 2x2 traceless matrix the spectral norm has this closed form.
    score = np.sqrt(((correction[..., 0, 0].real - correction[..., 1, 1].real) / 2)**2
                    + abs(correction[..., 0, 1])**2)
    count = min(20, score.size)
    top = np.argpartition(score.ravel(), -count)[-count:]
    top = top[np.argsort(score.ravel()[top])[::-1]]
    ranked = []
    for flat in top:
        channel, k = np.unravel_index(flat, score.shape)
        ranked.append(dict(channel=int(channel), mode=int(k),
                           unshifted_denominators=d0[channel, k],
                           shifted_denominators=d0[channel, k] + shift,
                           source=g[channel, k], non_scalar_norm_Eh=float(score[channel, k]),
                           correction=rotation @ correction[channel, k] @ rotation.conj().T))
    selected = list(dict.fromkeys(item["mode"] for item in ranked[:6]))
    # Two ordinary spectral directions, not additional low-denominator selections.
    for k in (len(spectrum) // 2, 3 * len(spectrum) // 4):
        if k not in selected:
            selected.append(k)
    raw = x @ z[:, selected]
    channel_c = rotation @ correction.sum(axis=1) @ rotation.conj().T
    report = dict(shift=shift, energy_origin=origin, metric=metric,
                  minimum_unshifted_denominator=float(d0.min()),
                  minimum_abs_unshifted_denominator=float(abs(d0).min()),
                  minimum_shifted_denominator=float((d0 + shift).min()),
                  minimum_abs_shifted_denominator=float(abs(d0 + shift).min()),
                  source_response=rotation @ y.sum(axis=(0, 1)) @ rotation.conj().T,
                  shift_norm_correction=rotation @ norm_term.sum(axis=(0, 1)) @ rotation.conj().T,
                  channel_corrections=channel_c, ranked_modes=ranked, selected_modes=selected,
                  selected_connected_eigenvalues=spectrum[selected])
    return report, raw


def main(directory, *, resume_direct=False):
    lib.num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    with (directory / "prepared.pkl").open("rb") as handle:
        p = pickle.load(handle)
    if len(p.model_roots) != 2:
        raise ValueError("this diagnostic is for a two-root manifold")
    if resume_direct:
        inputs = [directory / name for name in ("prepared.pkl", "S1_2.json", "reference_vectors.npz")]
        caches = [directory / name for name in ("r_modes.json", "r_mode_coefficients.npz")]
        # shortcut: freshness uses mtimes; recompute spectra after copying inputs with preserved timestamps.
        if min(path.stat().st_mtime_ns for path in caches) < max(path.stat().st_mtime_ns for path in inputs):
            raise ValueError("spectral cache predates audit inputs; rerun without --resume-direct")
        report = json.loads(caches[0].read_text())
        with np.load(caches[1], allow_pickle=False) as saved:
            c = saved["coefficients"].copy()
            np.testing.assert_array_equal(saved["modes"], report["selected_modes"])
            np.testing.assert_array_equal(saved["eigenvalues"], report["selected_connected_eigenvalues"])
        if c.shape != (len(p.classes["r"].metric), len(report["selected_modes"])):
            raise ValueError("spectral coefficient shape does not match prepared metric")
        print("R_MODE_SPECTRUM_REUSED", flush=True)
    else:
        report, c = spectral_audit(p)
        result = json.loads((directory / "S1_2.json").read_text())
        def decode(value):
            value = np.asarray(value)
            return value[..., 0] + 1j * value[..., 1]
        corrections = {key: decode(value) for key, value in result["class_corrections"].items()}
        norms = {key: decode(value) for key, value in result["class_norms"].items()}
        report["doublet_decomposition"] = doublet_decomposition(p.reference, corrections, norms, result["shift"])
        report["mode_sum_vs_solver_defect"] = report["channel_corrections"].sum(axis=0) - corrections["r"]
        write_json(directory / "r_modes.json", report)
        np.savez(directory / "r_mode_coefficients.npz", coefficients=c,
                 modes=report["selected_modes"], eigenvalues=report["selected_connected_eigenvalues"])
    with np.load(directory / "reference_vectors.npz") as saved:
        vectors, actions = saved["vectors"], saved["actions"]
    q = actions - vectors @ p.active_reference
    print("R_MODE_DIRECT_VECTORS_BEGIN", flush=True)
    operators = RDirections(vectors, p.eris.ncas, 11)
    chi = operators.apply(c)
    residual_operators = RDirections(q, p.eris.ncas, 11)
    oq = residual_operators.apply(c)
    del residual_operators
    origin = report["energy_origin"]
    ha = fci_dhf_slow.absorb_h1e(p.eris.get_h1eff("AA"), p.eris.get_phys("AAAA").transpose(0, 2, 1, 3),
                                p.eris.ncas, 10, 0.5)
    hchi = active_action_batch(ha, chi.T, p.eris.ncas, 10).T
    gram = chi.conj().T @ chi
    direct_f = chi.conj().T @ (hchi - origin * chi)
    z = chi.conj().T @ oq
    code_f = c.conj().T @ (p.classes["r"].active - origin * p.classes["r"].metric) @ c
    measured = dict(gram=gram, direct_connected_active=direct_f,
                    code_connected_active=code_f, residual_companion=z,
                    restoration_prediction_defect=code_f - direct_f + 0.5 * (z + z.conj().T))
    _active_action_links.cache_clear()
    # Record the actual time-reversed orbital, which need not be an adjacent MO.
    mol, _ = scf.chkfile.load_scf(str(directory.parent / "hf_kr/hf.chk"))
    mo = p.mo_coeff
    full_tr = mo.conj().T @ mol.intor_symmetric("int1e_ovlp_spinor") @ ao_time_reverse(mol, mo)
    measured["common_partition_symmetry"] = partition_symmetry(p, full_tr)
    measured["external_fock_diagonalization_defect"] = p.diagnostics["external_fock_defect"]
    tr = full_tr[p.eris.nocc:, p.eris.nocc:]
    channels = list(dict.fromkeys(item["channel"] for item in report["ranked_modes"][:4]))
    partners = {channel: int(np.argmax(abs(tr[:, channel]))) for channel in channels}
    channels = list(dict.fromkeys(channels + list(partners.values())))
    sources = {}
    for channel in channels:
        b = operators.source(p.eris, channel)
        direct_g = chi.conj().T @ b
        code_g = c.conj().T @ p.classes["r"].source[channel]
        sources[channel] = dict(direct=direct_g, code=code_g, defect=code_g - direct_g,
                                unshifted_direct_denominators=(
                                    np.diag(direct_f).real[:, None] / np.diag(gram).real[:, None]
                                    + p.mo_energy[p.eris.nocc + channel]
                                    - (eigh(p.active_reference)[0] - origin)))
    measured.update(source_couplings=sources, time_reverse_partners=partners,
                    time_reverse_columns={r: tr[:, r] for r in channels})
    write_json(directory / "r_direct_audit.json", measured)
    print("R_MODE_DIRECT_COMPLETE", "maximum_norm_defect", np.max(abs(gram - np.eye(len(c.T)))),
          "maximum_restoration_prediction_defect",
          np.max(abs(measured["restoration_prediction_defect"])), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--resume-direct", action="store_true", help="reuse saved spectral results for unchanged inputs")
    args = parser.parse_args()
    main(args.directory.resolve(), resume_direct=args.resume_direct)
