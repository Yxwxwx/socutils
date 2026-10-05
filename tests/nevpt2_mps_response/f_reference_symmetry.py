# SPDX-License-Identifier: GPL-3.0-or-later
"""Read-only density and coefficient time-reversal audit of the F reference.

Set UC_REFERENCE to the archived f_sa6_ground_run directory and run with
PYTHONPATH=tests python -m nevpt2_mps_response.f_reference_symmetry.
The small fixed-N coefficient audit does not modify MPSs or impose KR.
"""
import json
import os
from pathlib import Path

import numpy as np
from pyscf import lib

from socutils.dmrg.kramers import (
    ao_time_reverse, time_reverse_integrals, time_reverse_one_body,
    time_reverse_rdm1, time_reverse_rdm2,
)
from socutils.mrpt import nevpt2_utils as u
from . import f_atom
from .diagnose_r85 import mps_vector
from .uc_audit import time_reverse_coefficients

f_atom.THREADS = 1
f_atom.STACK_MB = 1024
lib.num_threads(1)
directory = Path(os.environ["UC_REFERENCE"])
metadata = json.loads((directory / "mcscf.json").read_text())
assert metadata["nroots"] == 6 and not metadata["cd"] and not metadata["kr"]
mc, solver, fingerprint = f_atom.restore(directory)
try:
    assert solver.kramers_adapter is None
    mo, s = mc.mo_coeff, mc._scf.get_ovlp()
    tr = mo.conj().T @ s @ ao_time_reverse(mc.mol, mo)
    nc, no = mc.ncore, mc.ncore + mc.ncas
    partitions = (slice(0, nc), slice(nc, no), slice(no, len(tr)))
    closure = [[float(np.linalg.norm(tr[a, b])) for b in partitions] for a in partitions]
    ta = tr[nc:no, nc:no]
    with np.load(directory / "reference.npz") as saved:
        h, g = saved["h1e"], saved["eri"]
        th, tg = time_reverse_integrals(ta, h, g)
        hmo = mo.conj().T @ saved["hcore_ao"] @ mo
        ecore = float(saved["ecore"])
    d1 = [u._make_rdm(solver, root, 1) for root in range(6)]
    d2 = [u._make_rdm(solver, root, 2) for root in range(6)]
    tr1 = [time_reverse_rdm1(ta, d) for d in d1]
    tr2 = [time_reverse_rdm2(ta, d) for d in d2]
    states = np.column_stack([mps_vector(*u._root_ket(solver, root), mc.ncas, f_atom.NELEC)
                              for root in range(6)])
    gram = states.conj().T @ states
    np.testing.assert_allclose(gram, np.eye(6), atol=1e-8, rtol=0.)
    reversed_states = time_reverse_coefficients(ta, states, f_atom.NELEC)
    overlaps = states.conj().T @ reversed_states
    # In a four-dimensional manifold T may mix roots; density distances
    # are diagnostics, not many-body overlaps or a claimed fixed pairing.
    report = dict(
        fingerprint=fingerprint, energies=mc.e_states.tolist(),
        partition_time_reversal_block_norms=closure,
        active_time_reversal_square_error=float(np.max(abs(ta @ ta.conj() + np.eye(mc.ncas)))),
        one_body_time_reversal_error=float(np.max(abs(time_reverse_one_body(tr, hmo) - hmo))),
        active_h_time_reversal_error=float(np.max(abs(th - h))),
        active_g_time_reversal_error=float(np.max(abs(tg - g))),
        root_rdm1_time_reversal_distance=[[float(np.linalg.norm(a-b)) for b in d1] for a in tr1],
        root_rdm2_time_reversal_distance=[[float(np.linalg.norm(a-b)) for b in d2] for a in tr2],
        root_gram_error=float(np.max(abs(gram - np.eye(6)))),
        root_time_reversal_overlap_absolute=abs(overlaps).tolist(),
        root_time_reversal_overlap_real=overlaps.real.tolist(),
        root_time_reversal_overlap_imag=overlaps.imag.tolist(),
        many_body_time_reversal_square_error=float(np.linalg.norm(
            time_reverse_coefficients(ta, reversed_states, f_atom.NELEC) + states)),
    )
    # Audit the actual state-specific Dyall definition of the directly
    # paired roots, without averaging their densities or imposing KR.
    left, semi_left, semi_rotation, constants, energies = {}, {}, {}, {}, {}
    semi_fock_errors = []
    for root in (4, 5):
        fock = mo.conj().T @ mc.get_fock(mo_coeff=mo, casdm1=d1[root]) @ mo
        left[root] = np.zeros_like(fock)
        for sl in (partitions[0], partitions[2]):
            left[root][sl, sl] = -fock[sl, sl]
        left[root][nc:no, nc:no] = -h
        energies[root] = float((np.einsum("pq,pq", h, d1[root]) + .5 * np.einsum(
            "pqrs,pqsr", g.transpose(0, 2, 1, 3), d2[root])).real)
        constants[root] = energies[root] + float(np.trace(fock[:nc, :nc]).real)
        semi_mo, eps = u.semicanonicalize(mc, mo, d1[root], root)
        semi_rotation[root] = mo.conj().T @ s @ semi_mo
        semi_left[root] = np.diag(-np.r_[eps[:nc], np.zeros(mc.ncas), eps[no:]]).astype(complex)
        semi_left[root][nc:no, nc:no] = -h
        semi_fock = semi_mo.conj().T @ mc.get_fock(mo_coeff=semi_mo, casdm1=d1[root]) @ semi_mo
        semi_fock_errors.append(max(float(np.max(abs(semi_fock[sl, sl] - np.diag(eps[sl]))))
                                   for sl in (partitions[0], partitions[2])))
    semi_tr = semi_rotation[5].conj().T @ tr @ semi_rotation[4].conj()
    common_error = time_reverse_one_body(tr, left[4]) - left[5]
    report["paired_roots_dyall_covariance"] = dict(
        roots=[4, 5], operator="L=E0-HD; active two-body coefficients checked above",
        common_mo_one_body_max_error=float(np.max(abs(common_error))),
        common_mo_partition_max_errors=[float(np.max(abs(common_error[sl, sl]))) for sl in partitions],
        scalar_shift_error=float(abs(constants[4] - constants[5])),
        semicanonical_one_body_max_error=float(np.max(abs(
            time_reverse_one_body(semi_tr, semi_left[4]) - semi_left[5]))),
        semicanonical_representation_errors=[float(np.max(abs(
            semi_rotation[root].conj().T @ left[root] @ semi_rotation[root] - semi_left[root])))
            for root in (4, 5)],
        semicanonical_external_fock_errors=semi_fock_errors,
        reference_energy_errors=[float(energies[root] + ecore - mc.e_states[root]) for root in (4, 5)],
    )
    for name, indices in (("fourfold", range(4)), ("doublet", range(4, 6)), ("all", range(6))):
        avg1 = sum(d1[i] for i in indices) / len(indices)
        avg2 = sum(d2[i] for i in indices) / len(indices)
        report[name + "_density_closure"] = dict(
            rdm1=float(np.linalg.norm(time_reverse_rdm1(ta, avg1) - avg1)),
            rdm2=float(np.linalg.norm(time_reverse_rdm2(ta, avg2) - avg2)))
        indices = list(indices)
        block = overlaps[np.ix_(indices, indices)]
        report[name + "_coefficient_closure"] = dict(
            leakage_per_root=np.linalg.norm(reversed_states[:, indices]
                - states[:, indices] @ block, axis=0).tolist(),
            overlap_singular_values=np.linalg.svd(block, compute_uv=False).tolist())
    print("F_REFERENCE_TIME_REVERSAL", json.dumps(report), flush=True)
finally:
    solver.close()
