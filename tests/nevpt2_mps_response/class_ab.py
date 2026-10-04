# SPDX-License-Identifier: GPL-3.0-or-later
"""Whole-class versus channel-wise response using the same Block2 multiply.

The active sites are followed by all external sites, with exactly one virtual
electron/core hole. The extra U(1) counts external occupations, not spin.
Small fixed-N expansions measure residuals; they never solve CI.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from pyscf import lib
from pyscf.fci import cistring, fci_dhf_slow
from socutils.mrpt import nevpt2_utils as u
from socutils.mrpt import x2cscnevpt2 as sc
from socutils.mrpt.nevpt2_eris import wick_eris_from_mc
from socutils.mrpt.nevpt2_mps_response import (
    _release_response_mps,
    _response_tag,
)

from .diagnose_r85 import (
    _make_source_mpo,
    _shifted_mpo_constant,
    _source_coefficients,
    apply_source_vector,
    mps_vector,
    source_terms,
)
from .f_atom import THREADS, restore, save_json


def determinants(ncas, nelec):
    occ = cistring.gen_occslst(range(ncas), nelec)
    dets = np.zeros((len(occ), ncas), dtype=np.uint8)
    dets[np.arange(len(occ))[:, None], occ] = 1
    return dets


def channels(mc, eris, pdms, eps, key, bond, sweeps, *, whole_only=False):
    driver, reference = u._root_ket(mc.fcisolver, 0)
    h = eris.get_h1eff("AA")
    g = eris.get_phys("AAAA").transpose(0, 2, 1, 3)
    ncas, nelec = mc.ncas, int(mc.nelecas)
    eact = float((np.einsum("pq,pq", h, pdms[0]) + .5 * np.einsum(
        "pqrs,pqsr", eris.get_phys("AAAA"), pdms[1])).real)
    gaps = -eps[:eris.ncore] if key == "i" else eps[eris.nocc:]
    norms = np.zeros(len(gaps), dtype=complex)
    exec(sc._compile_wick_equations().norm_code[key], {"np": np},
         {**u._execution_context(eris, pdms), "norm": norms})
    sources = [_source_coefficients(eris, key, i)
               for i in range(len(gaps))]
    v = mps_vector(driver, reference, ncas, nelec)
    cache = {}
    bs = np.array([apply_source_vector(key, s, v, ncas, nelec, cache) for s in sources])
    np.testing.assert_allclose(np.sum(abs(bs)**2, axis=1), norms.real,
                               atol=1e-14, rtol=1e-8)
    working = driver.copy_mps(reference, tag=_response_tag())
    try:
        from pyblock2.algebra.io import MPSTools
        driver.align_mps_center(working, ref=0)
        _dets, raw_v = driver.get_csf_coefficients(
            working, cutoff=0., given_dets=determinants(ncas, nelec),
            max_print=0, fci_conv=False, iprint=0)
        working = driver.adjust_mps(working, dot=1)[0]
        python_mps = MPSTools.from_block2(working)
    finally:
        _release_response_mps(driver, working)
    delta = 1 if key == "i" else -1
    h2e = fci_dhf_slow.absorb_h1e(h, g, ncas, nelec + delta, .5)
    mpo = driver.get_qc_mpo(h1e=h.copy(), g2e=g.copy(), ecore=0., reorder=None, iprint=0)
    results = {}
    try:
        for name, absolute, relative, cutoff, normalize in (() if whole_only else (
                ("B_absolute", 1e-6, 0., 1e-14, False),
                ("C_normalized", 0., 1e-8, 1e-26, True))):
            entries = []
            for index, source in enumerate(sources):
                norm2 = float(norms[index].real)
                if norm2 <= 1e-14:
                    entries.append(dict(index=index, zero_source=True, e_corr=0.))
                    continue
                scale = np.sqrt(norm2) if normalize else 1.
                rhs = _make_source_mpo(driver, key, source, scale=1 / scale)
                x = driver.get_random_mps(tag=_response_tag(),
                                         bond_dim=bond, dot=2, target=reference.info.target + rhs.op.q_label)
                try:
                    with _shifted_mpo_constant(mpo, gaps[index] - eact):
                        driver.multiply(x, rhs, reference, left_mpo=mpo,
                                        n_sweeps=sweeps, tol=0., bra_bond_dims=[bond],
                                        thrds=[absolute], linear_rel_conv_thrd=relative,
                                        noises=[0.], cutoff=cutoff, linear_max_iter=4000, iprint=0)
                    xv = scale * mps_vector(driver, x, ncas, nelec + delta)
                    ax = fci_dhf_slow.contract_2e(h2e, xv, ncas, nelec + delta)
                    ax += (gaps[index] - eact) * xv
                    entries.append(dict(index=index, e_corr=float(-np.vdot(bs[index], xv).real),
                                        global_relative_residual=float(np.linalg.norm(ax-bs[index])/np.linalg.norm(bs[index]))))
                finally:
                    _release_response_mps(driver, x)
                    del rhs
                print("CHANNEL", name, key, entries[-1], flush=True)
            results[name] = dict(e_corr=sum(e["e_corr"] for e in entries), entries=entries)
    finally:
        del mpo
    raw_cache = {}
    raw_bs = np.array([apply_source_vector(key, s, np.asarray(raw_v), ncas, nelec, raw_cache)
                       for s in sources])
    return results, (h, g, eact, gaps, norms.real, sources, np.asarray(raw_v), raw_bs, python_mps)


def whole_class(data, ncas, nelec, key, bond, sweeps, scratch, *, options=None):
    from pyblock2.driver.core import DMRGDriver, SymmetryTypes
    h, g, eact, gaps, norms, sources, raw_v, bs, *embedded = data
    count, delta = len(gaps), 1 if key == "i" else -1
    driver = DMRGDriver(scratch=str(scratch), symm_type=SymmetryTypes.SAnySGFCPX,
                        n_threads=THREADS, stack_mem=2 << 30)
    driver.set_symmetry_groups("U1Fermi", "U1", "AbelianPG", hints=["SGF"])
    q = driver.bw.SX
    driver.initialize_system(n_sites=ncas+count, target=q(nelec, 0, 0), vacuum=q(0, 0, 0), hamil_init=False)
    active_ops = {"": np.eye(2), "C": np.array([[0, 0], [1, 0]]),
                  "D": np.array([[0, 1], [0, 0]])}
    label_ops = [{"": np.eye(2), "V": np.diag([0., gap]),
                  "L": np.array([[0, 0], [1, 0]])} for gap in gaps]
    driver.ghamil = driver.get_custom_hamiltonian(
        [[(q(0, 0, 0), 1), (q(1, 0, 0), 1)]] * ncas
        + [[(q(0, 0, 0), 1), (q(-delta, 1, 0), 1)]] * count,
        [active_ops] * ncas + label_ops, orb_dependent_ops="")
    # StateInfo sorts blocks: negative-charge hole labels precede vacuum.
    basis = driver.ghamil.basis[ncas]
    vacuum_index = sum(basis.n_states[j] for j in range(basis.find_state(q(0, 0, 0))))
    label_index = sum(basis.n_states[j] for j in range(basis.find_state(q(-delta, 1, 0))))
    if embedded:
        from pyblock2.algebra.core import MPS, SubTensor, Tensor
        from pyblock2.algebra.io import MPSTools
        tensors = [Tensor([SubTensor(q_labels=tuple(q(x.n, 0, x.pg) for x in b.q_labels),
                                     reduced=b.reduced.copy()) for b in t.blocks])
                   for t in embedded[0].tensors]
        tensors[-1] = Tensor([SubTensor(q_labels=b.q_labels + (q(nelec, 0, 0),),
                                        reduced=b.reduced[..., None]) for b in tensors[-1].blocks])
        tensors.extend(Tensor([SubTensor(q_labels=(q(nelec, 0, 0), q(0, 0, 0), q(nelec, 0, 0)),
                                           reduced=np.ones((1, 1, 1), dtype=complex))]) for _ in range(count-1))
        tensors.append(Tensor([SubTensor(q_labels=(q(nelec, 0, 0), q(0, 0, 0)),
                                         reduced=np.ones((1, 1), dtype=complex))]))
        reference = MPSTools.to_block2(MPS(tensors), driver.basis, center=0, tag=_response_tag())
        reference = driver.adjust_mps(reference, dot=2)[0]
    else:
        dets = np.c_[determinants(ncas, nelec), np.full((len(raw_v), count), vacuum_index, dtype=np.uint8)]
        reference = driver.get_mps_from_csf_coefficients(dets, raw_v, tag=_response_tag(), dot=2,
                                                       full_fci=False, iprint=0)
    target = q(nelec, 1, 0)
    hbuilder = driver.expr_builder()
    hbuilder.add_sum_term("CD", h, cutoff=0.)
    hbuilder.add_sum_term("CCDD", g.transpose(0, 2, 3, 1), factor=.5, cutoff=0.)
    for i in range(count):
        hbuilder.add_term("V", [ncas+i], 1.)
    hbuilder.add_const(-eact)
    a_mpo = driver.get_mpo(hbuilder.finalize(), cutoff=0., iprint=0)
    b_builder = driver.expr_builder()
    for i, source in enumerate(sources):
        for coefficient, operators in source_terms(key, source):
            b_builder.add_term("L" + "".join(op for op, _ in operators),
                               [ncas+i] + [site for _, site in operators], coefficient)
    b_mpo = driver.get_mpo(b_builder.finalize(fermionic_ops="CDL"), cutoff=0., iprint=0)
    x = driver.get_random_mps(tag=_response_tag(), bond_dim=bond, dot=2, target=target)
    try:
        controls = dict(n_sweeps=sweeps, tol=0., bra_bond_dims=[bond],
                        thrds=[1e-6], noises=[0.], cutoff=1e-14,
                        linear_max_iter=4000, iprint=1)
        controls.update({} if options is None else options)
        driver.multiply(x, b_mpo, reference, left_mpo=a_mpo, **controls)
        target_dets = determinants(ncas, nelec + delta)
        combined = []
        for i in range(count):
            labels = np.full((len(target_dets), count), vacuum_index, dtype=np.uint8)
            labels[:, i] = label_index
            combined.append(np.c_[target_dets, labels])
        combined = np.concatenate(combined)
        working = driver.copy_mps(x, tag=_response_tag())
        try:
            driver.align_mps_center(working, ref=0)
            _dets, values = driver.get_csf_coefficients(
                working, cutoff=0., given_dets=combined, fci_conv=False, max_print=0, iprint=0)
        finally:
            _release_response_mps(driver, working)
        values = np.asarray(values).reshape(count, -1)
        # The terminal odd operator crosses the final active particle sector.
        bs = bs * (-1)**(nelec + delta)
        overlap = complex(driver.expectation(x, b_mpo, reference))
        quadratic = complex(driver.expectation(x, a_mpo, x))
        np.testing.assert_allclose(np.vdot(values, bs), overlap, atol=1e-11, rtol=1e-7)
        h2e = fci_dhf_slow.absorb_h1e(h, g, ncas, nelec + delta, .5)
        residuals = np.array([fci_dhf_slow.contract_2e(h2e, v, ncas, nelec + delta)
                              + (gap-eact)*v-b for v, gap, b in zip(values, gaps, bs)])
        np.testing.assert_allclose(np.vdot(values, residuals + bs), quadratic,
                                   atol=1e-11, rtol=1e-7)
        return dict(e_corr=-overlap.real,
                    global_relative_residual=float(np.linalg.norm(residuals)/np.linalg.norm(bs)),
                    max_channel_relative_residual=max(float(np.linalg.norm(r)/np.linalg.norm(b))
                                                       for r, b in zip(residuals, bs) if np.linalg.norm(b)>1e-7),
                    representation="active spinors + all external sites, exactly one particle/hole",
                    max_bond_dimension=bond, n_sweeps=controls["n_sweeps"],
                    controls=controls, stationarity_error=abs(overlap-quadratic))
    finally:
        for state in (x, reference):
            _release_response_mps(driver, state)
        del a_mpo, b_mpo
        driver.finalize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class", dest="key", choices=("i", "r"), default="r")
    parser.add_argument("--sweeps", type=int, default=8)
    parser.add_argument("--whole-only", action="store_true", help="reuse completed B/C results")
    parser.add_argument("--whole-options", type=json.loads, default=None,
                        help="JSON overrides passed directly to DMRGDriver.multiply")
    parser.add_argument("--whole-label", default="A_whole_absolute",
                        help="result label; use a new label to preserve the baseline")
    args = parser.parse_args()
    directory = Path(__file__).parent / "f_atom_run"
    lib.num_threads(THREADS)
    mc, solver, fingerprint = restore(directory)
    try:
        pdms = tuple(u._make_rdm(solver, 0, rank) for rank in range(1, 4))
        full_mc = u._full_integral_mc(mc)
        mo, eps = u.semicanonicalize(full_mc, mc.mo_coeff, pdms[0], 0)
        eris = wick_eris_from_mc(full_mc, mo)
        result, data = channels(mc, eris, pdms, eps, args.key, 1000, args.sweeps,
                                whole_only=args.whole_only)
    finally:
        solver.close()
    output = directory / f"class_{args.key}_ab.json"
    if args.whole_only and output.exists():
        result = json.loads(output.read_text())
        assert result.pop("fingerprint") == fingerprint
    save_json(output, dict(fingerprint=fingerprint, **result))
    result[args.whole_label] = whole_class(
        data, mc.ncas, int(mc.nelecas), args.key, 1000, args.sweeps,
        Path(lib.param.TMPDIR)/"whole", options=args.whole_options)
    save_json(directory / f"class_{args.key}_ab.json", dict(fingerprint=fingerprint, **result))
    print("CLASS_AB", args.key, {k: {a: b for a, b in v.items() if a != "entries"}
                                for k, v in result.items()}, flush=True)


if __name__ == "__main__":
    main()
