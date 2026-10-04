# SPDX-License-Identifier: GPL-3.0-or-later
"""Isolate absolute/relative inner stopping, keeping the response cutoff fixed.

The fixed-N SGF coefficient expansion only measures residuals of solved MPSs;
it never solves CI or changes the response variational state. No 4-RDM.
"""
import argparse
import json
from contextlib import contextmanager
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

from .f_atom import THREADS, restore, save_json


def _source_coefficients(eris, key, index):
    """Single-index tensors for diagnostics, never for production solves."""
    if key == "r":
        return (np.asarray(eris.get_h1eff("EA"))[index],
                np.asarray(eris.get_phys("EAAA"))[index].transpose(0, 2, 1))
    if key == "i":
        return (np.asarray(eris.get_h1eff("AI"))[:, index],
                np.asarray(eris.get_phys("AAIA"))[:, :, index, :])
    raise ValueError("diagnostic sources are defined only for i/r")


def _make_source_mpo(driver, key, coefficients, *, scale=1., iprint=0):
    """Historical per-channel experiment; not imported by production."""
    one, three = coefficients
    order = getattr(driver, "reorder_idx", None)
    if order is not None:
        one, three = one[order], three[np.ix_(order, order, order)]
    builder = driver.expr_builder()
    builder.add_sum_term("C" if key == "i" else "D", one, factor=scale, cutoff=0.)
    builder.add_sum_term("CCD" if key == "i" else "CDD", three, factor=scale, cutoff=0.)
    return driver.get_mpo(builder.finalize(), cutoff=1e-20, iprint=iprint, add_ident=True)


@contextmanager
def _shifted_mpo_constant(mpo, shift):
    original = mpo.const_e
    mpo.const_e = original + shift
    try:
        yield
    finally:
        mpo.const_e = original


def mps_vector(driver, ket, ncas, nelec):
    occupied = cistring.gen_occslst(range(ncas), nelec)
    determinants = np.zeros((len(occupied), ncas), dtype=np.uint8)
    determinants[np.arange(len(occupied))[:, None], occupied] = 1
    order = getattr(driver, "reorder_idx", None)
    if order is not None:
        determinants = determinants[:, np.asarray(order)]
    working = driver.copy_mps(ket, tag=_response_tag())
    try:
        driver.align_mps_center(working, ref=0)
        _dets, vector = driver.get_csf_coefficients(
            working, cutoff=0., given_dets=determinants,
            max_print=0, fci_conv=True, iprint=0)
        vector = np.asarray(vector)
        if order is not None:
            # pyblock2 remaps the returned occupancies, but convert_phase uses
            # the internal site order. Restore the occupied-orbital permutation
            # sign when comparing to the original-order FCI residual oracle.
            inversions = np.triu(np.asarray(order)[:, None] > np.asarray(order)[None, :], 1)
            parity = np.einsum("bi,ij,bj->b", determinants.astype(int),
                               inversions.astype(int), determinants.astype(int)) % 2
            vector = vector * (1 - 2 * parity)
        return vector
    finally:
        _release_response_mps(driver, working)


def source_terms(key, coefficients):
    """Test-only explicit strings; pyblock2 builds production MPOs from tensors."""
    one, three = coefficients
    for ops, tensor in (("C" if key == "i" else "D", one),
                        ("CCD" if key == "i" else "CDD", three)):
        for sites in zip(*np.nonzero(tensor)):
            yield tensor[sites], tuple(zip(ops, sites))


def apply_source_vector(key, coefficients, vector, ncas, nelec, cache=None):
    """Measurement oracle: apply the unchanged fermion strings to SGF coefficients."""
    # ponytail: exponential fixed-N oracle, test-only; large CAS needs MPO residuals.
    cache = {} if cache is None else cache
    occupied = cistring.gen_occslst(range(ncas), nelec)
    bits = np.sum(1 << occupied, axis=1)
    target_bits = np.sum(1 << cistring.gen_occslst(
        range(ncas), nelec + (1 if key == "i" else -1)), axis=1)
    addresses = np.full(1 << ncas, -1, dtype=int)
    addresses[target_bits] = np.arange(len(target_bits))
    parity = np.fromiter(((-1)**k.bit_count() for k in range(1 << ncas)),
                         dtype=np.int8)
    result = np.zeros(len(target_bits), dtype=complex)
    for coefficient, operators in source_terms(key, coefficients):
        if operators not in cache:
            updated, sign, valid = bits.copy(), np.ones(len(bits)), np.ones(len(bits), bool)
            for operator, site in reversed(operators):
                occupied_site = (updated & (1 << site)) != 0
                valid &= ~occupied_site if operator == "C" else occupied_site
                sign *= parity[updated & ((1 << site) - 1)]
                updated ^= 1 << site
            cache[operators] = addresses[updated[valid]], sign[valid] * vector[valid]
        rows, values = cache[operators]
        result[rows] += coefficient * values
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", nargs="+", help="run only named cases; preserve earlier measurements")
    args = parser.parse_args()
    directory = Path(__file__).parent / "f_atom_run"
    lib.num_threads(THREADS)
    mc, solver, fingerprint = restore(directory)
    driver, reference = u._root_ket(solver, 0)
    states, mpos = [], []
    try:
        pdms = tuple(u._make_rdm(solver, 0, rank) for rank in range(1, 4))
        full_mc = u._full_integral_mc(mc)
        mo, eps = u.semicanonicalize(full_mc, mc.mo_coeff, pdms[0], 0)
        eris = wick_eris_from_mc(full_mc, mo)
        norms = np.zeros(eris.nvirt, dtype=complex)
        exec(sc._compile_wick_equations().norm_code["r"], {"np": np},
             {**u._execution_context(eris, pdms), "norm": norms})
        source_norm = float(norms[85].real)
        h = eris.get_h1eff("AA")
        w = eris.get_phys("AAAA")
        g = w.transpose(0, 2, 1, 3)
        active_energy = float((np.einsum("pq,pq", h, pdms[0])
                              + .5 * np.einsum("pqrs,pqsr", w, pdms[1])).real)
        gap = float(eps[eris.nocc + 85])
        active_mpo = driver.get_qc_mpo(h1e=h.copy(), g2e=g.copy(), ecore=0.,
                                     reorder=None, iprint=0)
        identity = driver.get_identity_mpo()
        source = _source_coefficients(eris, "r", 85)
        source_mpo = _make_source_mpo(driver, "r", source)
        mpos.extend((source_mpo, identity, active_mpo))
        target = reference.info.target + source_mpo.op.q_label
        bond = solver.max_bond_dimension
        initial = driver.get_random_mps(
            tag=_response_tag(), bond_dim=bond,
            center=int(reference.center), dot=2, target=target,
            left_vacuum=source_mpo.left_vacuum)
        states.append(initial)

        # Measurement-only B|Psi>: full sector fits within M=1000. Zero cutoff
        # here avoids truncating the residual oracle, not the response solve.
        rhs = driver.copy_mps(initial, tag=_response_tag())
        states.append(rhs)
        driver.multiply(rhs, source_mpo, reference, n_sweeps=8, tol=0.,
                        bra_bond_dims=[bond], noises=[0.], cutoff=0., iprint=0)
        fitted_b = mps_vector(driver, rhs, mc.ncas, int(target.n))
        reference_vector = mps_vector(driver, reference, mc.ncas, int(mc.nelecas))
        b = apply_source_vector("r", source, reference_vector, mc.ncas, int(mc.nelecas))
        np.testing.assert_allclose(fitted_b, b, atol=1e-13, rtol=1e-7)
        measured_source_norm = float(np.vdot(b, b).real)
        assert abs(measured_source_norm - source_norm) < 1e-9 * source_norm
        h2e = fci_dhf_slow.absorb_h1e(h, g, mc.ncas, int(target.n), .5)
        result = dict(fingerprint=fingerprint, channel="r85", source_norm2=source_norm,
                      measured_source_norm2=measured_source_norm, orbital_gap=gap,
                      source_fit_relative_error=float(np.linalg.norm(fitted_b-b) / np.linalg.norm(b)),
                      ncas=mc.ncas, nelec=int(target.n), sector_dimension=len(b), cases={})
        normalized_mpo = _make_source_mpo(driver, "r", source, scale=1 / np.sqrt(source_norm))
        mpos.append(normalized_mpo)
        cases = (
            ("absolute", 1e-6, 0., False, 1e-14, 8),
            ("relative", 0., 1e-8, False, 1e-14, 8),
            ("normalized_absolute", 1e-6, 0., True, 1e-14, 8),
            ("normalized_relative", 0., 1e-8, True, 1e-14, 8),
            ("normalized_cutoff20", 0., 1e-8, True, 1e-20, 8),
            ("normalized_cutoff26", 0., 1e-8, True, 1e-26, 8),
            ("normalized_sweeps16", 0., 1e-8, True, 1e-26, 16),
            ("normalized_relative10", 0., 1e-10, True, 1e-26, 16))
        if args.case:
            unknown = set(args.case) - {c[0] for c in cases}
            if unknown:
                raise ValueError(f"unknown diagnostic cases: {sorted(unknown)}")
            previous_path = directory / "r85_diagnostic.json"
            if previous_path.exists():
                previous = json.loads(previous_path.read_text())
                assert previous["fingerprint"] == fingerprint
                result["cases"].update(previous["cases"])
        for name, absolute, relative, normalized, cutoff, sweeps in cases:
            if args.case and name not in args.case:
                continue
            rhs_mpo = normalized_mpo if normalized else source_mpo
            scale = np.sqrt(source_norm) if normalized else 1.
            x_mps = driver.copy_mps(initial, tag=_response_tag())
            states.append(x_mps)
            print("R85_CASE", name, flush=True)
            with _shifted_mpo_constant(active_mpo, gap - active_energy):
                reported = driver.multiply(
                    x_mps, rhs_mpo, reference, left_mpo=active_mpo,
                    n_sweeps=sweeps, tol=0., bond_dims=[int(reference.info.bond_dim)],
                    bra_bond_dims=[bond], thrds=[absolute] * sweeps,
                    linear_rel_conv_thrd=relative, noises=[0.] * sweeps,
                    cutoff=cutoff, linear_max_iter=4000, iprint=2)
                overlap = scale**2 * complex(driver.expectation(x_mps, rhs_mpo, reference))
                quadratic = scale**2 * complex(driver.expectation(x_mps, active_mpo, x_mps))
            x = scale * mps_vector(driver, x_mps, mc.ncas, int(target.n))
            ax = fci_dhf_slow.contract_2e(h2e, x, mc.ncas, int(target.n))
            ax += (gap - active_energy) * x
            dense_overlap, dense_quadratic = np.vdot(x, b), np.vdot(x, ax)
            # Independently verify coefficient conventions against native MPOs.
            np.testing.assert_allclose(dense_overlap, overlap, atol=1e-22, rtol=1e-7)
            np.testing.assert_allclose(dense_quadratic, quadratic, atol=1e-22, rtol=1e-7)
            rho = float(np.linalg.norm(ax - b) / np.linalg.norm(b))
            case = dict(absolute_threshold=absolute, relative_threshold=relative,
                        normalized_source=normalized,
                        n_sweeps=sweeps, tol=0., noise=0., response_cutoff=cutoff,
                        max_bond_dimension=bond, solver_type="Automatic (unchanged)",
                        global_relative_residual=rho,
                        overlap=[overlap.real, overlap.imag],
                        quadratic=[quadratic.real, quadratic.imag],
                        reported=[scale**2 * complex(reported).real,
                                  scale**2 * complex(reported).imag],
                        stationarity_error=abs(overlap-quadratic))
            result["cases"][name] = case
            save_json(directory / "r85_diagnostic.json", result)
            print("R85_RESULT", name, case, flush=True)
    finally:
        for state in reversed(states):
            _release_response_mps(driver, state)
        mpos.clear()
        solver.close()


if __name__ == "__main__":
    main()
