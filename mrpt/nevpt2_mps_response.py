# SPDX-License-Identifier: GPL-3.0-or-later
"""Block2-style uncontracted i/r response for the optional no-4-RDM branch.

This ports the ``E4 is None`` branch of pyblock2.icmr.scnevpt2 and
icnevpt2_full to native SGFCPX. Each aaac/aaav class has one enlarged response
MPS, restricted by the same native NEVPTMPSInfo used in block2main. No SC
denominator or IC basis is used in these two classes. The other six classes
retain their Wick equations.

Source MPOs and response MPSs are built with the existing pyblock2
DMRGDriver API. This is MPS-PT/Hylleraas linear response (Sharma--Chan,
2014); no time-propagation module or custom Block2 solver is used.
Our X is minus the physical first-order wavefunction: minimize
L[X] = <X|A|X> - 2 Re<X|B Psi>, then E2 = -<B Psi|X>.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np
from pyscf.lib import logger

from . import nevpt2_utils as u


@dataclass(frozen=True)
class ResponseControls:
    # Same-M schedule from pyblock2.icmr.dmrg_helper: two four-sweep stages.
    max_bond_dimension: int | None = None
    n_sweeps: int = 8
    tol: float | None = None  # inherit the reference sweep tolerance
    # dmrg_helper writes Davidson 1e-6; block2main translates the input
    # schedule to Linear.linear_conv_thrds by dividing by 50. multiply()
    # takes that final residual-squared threshold directly.
    linear_threshold: float = 1e-6 / 50
    linear_rel_conv_thrd: float = 0.0
    noise: float = 1e-5
    cutoff: float = 1e-14  # block2main's response truncation default
    linear_max_iter: int = 4000


def prepare_pdms(solver, pdms, root):
    """Request ranks 1--3 only; an explicitly supplied fourth rank is unused."""
    if pdms is None:
        return tuple(u._make_rdm(solver, root, rank) for rank in range(1, 4))
    if not isinstance(pdms, (tuple, list)) or len(pdms) not in (3, 4):
        raise ValueError("MPS response requires raw RDMs of ranks 1--3")
    return tuple(pdms[:3])


def _response_tag():
    return "NEVPT2-" + uuid4().hex


def _release_response_mps(driver, mps):
    """Release only our scratch MPS; never the retained reference/checkpoint."""
    info = mps.info
    tag = info.tag
    if not tag.startswith("NEVPT2-") or not tag[7:].isalnum():
        raise RuntimeError(f"refusing to release an unowned response MPS {tag!r}")
    # pyblock2 returns disk-backed tensors. Load before deallocating to avoid
    # Block2 0.5.4's deallocate-on-unloaded-MPS crash.
    mps.load_mutable()
    mps.deallocate()
    info.load_mutable()
    info.deallocate_mutable()
    # MPSTools saves the info image in save_dir; tensors can use a separate
    # mps_dir. Only this owned tag is eligible for removal in either location.
    directories = {Path(driver.mps_dir or driver.scratch).resolve(),
                   Path(driver.frame.save_dir).resolve()}
    for scratch in directories:
        for pattern in (f"{tag}-mps_info.bin", f"F.MPS.{tag}.*",
                        f"F.MPS.INFO.{tag}.LEFT.*", f"F.MPS.INFO.{tag}.RIGHT.*"):
            for path in scratch.glob(pattern):
                if path.is_symlink() or path.resolve().parent != scratch or not path.is_file():
                    raise RuntimeError("refusing to remove an unexpected response MPS path")
                path.unlink()


def _real_scalar(value, *, name):
    value = u._complex_scalar(value, name=name)
    imaginary = abs(value.imag)
    if imaginary > 1e-8 * max(1., abs(value.real)):
        u._warn_numerical(f"{name} has an imaginary residual of {imaginary:.3e}")
    return float(value.real), float(imaginary)


def _controls(solver, options):
    controls = ResponseControls(**({} if options is None else options))
    bond = (getattr(solver, "max_bond_dimension", None)
            if controls.max_bond_dimension is None else controls.max_bond_dimension)
    tol = getattr(solver, "tol", 1e-8) if controls.tol is None else controls.tol
    for name, value in (("max_bond_dimension", bond), ("n_sweeps", controls.n_sweeps),
                        ("linear_max_iter", controls.linear_max_iter)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name, value in (("tol", tol), ("linear_threshold", controls.linear_threshold),
                        ("linear_rel_conv_thrd", controls.linear_rel_conv_thrd),
                        ("cutoff", controls.cutoff)):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if controls.linear_threshold == 0 and controls.linear_rel_conv_thrd == 0:
        raise ValueError("at least one linear convergence threshold must be positive")
    if not np.isfinite(controls.noise) or controls.noise < 0:
        raise ValueError("noise must be finite and non-negative")
    return controls, int(bond), float(tol)


def _active_mps(driver, reference):
    """Read the existing MPS tensors with pyblock2; never expand them into CI."""
    from pyblock2.algebra.io import MPSTools
    working = driver.copy_mps(reference, tag=_response_tag())
    try:
        driver.align_mps_center(working, ref=0)
        working = driver.adjust_mps(working, dot=1)[0]
        return MPSTools.from_block2(working)
    finally:
        _release_response_mps(driver, working)


def _embed_reference(driver, active_mps, ncore, nvirt):
    """Append empty virtuals or prepend occupied core spinors, without fitting."""
    from pyblock2.algebra.core import MPS, SubTensor, Tensor
    from pyblock2.algebra.io import MPSTools
    q = driver.bw.SX
    shift = q(ncore, 0, 0)
    tensors = []
    for site, tensor in enumerate(active_mps.tensors):
        blocks = []
        for block in tensor.blocks:
            labels, array = list(block.q_labels), block.reduced.copy()
            physical = 0 if site == 0 else 1
            for axis in range(len(labels)):
                if axis != physical:
                    labels[axis] = (labels[axis] + shift)[0]
            if site == 0 and ncore:
                labels.insert(0, shift)
                array = array[None, ...]
            if site == len(active_mps.tensors) - 1 and nvirt:
                labels.append(driver.target)
                array = array[..., None]
            blocks.append(SubTensor(q_labels=tuple(labels), reduced=array))
        tensors.append(Tensor(blocks))
    prefix = []
    for site in range(ncore):
        labels = (q(1, 0, 0), q(1, 0, 0)) if site == 0 else (
            q(site, 0, 0), q(1, 0, 0), q(site + 1, 0, 0))
        prefix.append(Tensor([SubTensor(q_labels=labels,
                                        reduced=np.ones((1,) * len(labels), dtype=complex))]))
    for site in range(nvirt):
        labels = (driver.target, driver.vacuum) if site == nvirt - 1 else (
            driver.target, driver.vacuum, driver.target)
        tensors.append(Tensor([SubTensor(q_labels=labels,
                                         reduced=np.ones((1,) * len(labels), dtype=complex))]))
    mps = MPSTools.to_block2(MPS(prefix + tensors), driver.basis,
                            center=0, tag=_response_tag())
    return driver.adjust_mps(mps, dot=2)[0]


def _nevpt_mps(driver, ncore, nvirt, key, bond):
    """Native block2main NEVPTMPSInfo/MPS initialization, not a new solver."""
    bw = driver.bw
    info = bw.brs.NEVPTMPSInfo(driver.n_sites, ncore, nvirt,
                             int(key == "i"), int(key == "r"),
                             driver.vacuum, driver.target, driver.ghamil.basis)
    info.tag = _response_tag()
    info.set_bond_dimension(bond)
    info.bond_dim = bond
    mps = bw.bs.MPS(driver.n_sites, 0, 1)
    mps.initialize(info)
    mps.random_canonicalize()
    mps.tensors[mps.center].normalize()
    mps.save_mutable()
    info.save_mutable()
    mps.save_data()
    info.save_data(str(Path(driver.scratch) / f"{info.tag}-mps_info.bin"))
    return driver.adjust_mps(mps, dot=2)[0]


def _add_tensor(builder, operators, tensor, sites):
    """Batch full integral blocks through the native expression builder."""
    indices = np.array(np.nonzero(tensor))
    if indices.shape[1]:
        values = tensor[tuple(indices)]
        mapped = np.array([np.asarray(mapping)[axis]
                           for mapping, axis in zip(sites, indices)])
        builder.add_term(operators, mapped.T.ravel(), values)


def _whole_class_response(driver, active_mps, eris, order, key, orbital_energy,
                          active_energy, controls, bond, tol, iprint):
    """One aaac/aaav MPS and one native Linear solve for the complete class."""
    count, ncas = len(orbital_energy), eris.ncas
    ncore, nvirt = (count, 0) if key == "i" else (0, count)
    # Keep the original driver/frame alive. Creating a second DMRGDriver would
    # invalidate Block2's global frame and the retained reference MPS.
    original = driver.__dict__.copy()
    reference = response = source_mpo = dyall_mpo = None
    try:
        driver.initialize_system(n_sites=ncore + ncas + nvirt,
                                 n_elec=int(original["target"].n) + ncore)
        driver.reorder_idx = None
        active = np.arange(ncas) + ncore
        external = np.arange(count) if key == "i" else np.arange(count) + ncas
        reference = _embed_reference(driver, active_mps, ncore, nvirt)
        h = np.asarray(eris.get_h1eff("AA"))[np.ix_(order, order)]
        w = np.asarray(eris.get_phys("AAAA"))[np.ix_(order, order, order, order)]
        builder = driver.expr_builder()
        _add_tensor(builder, "CD", h, [active, active])
        _add_tensor(builder, "CCDD", .5 * w.transpose(0, 1, 3, 2), [active] * 4)
        builder.add_term("CD", np.column_stack((external, external)).ravel(), orbital_energy)
        builder.add_const(-active_energy - (np.sum(orbital_energy) if key == "i" else 0.))
        dyall_mpo = driver.get_mpo(builder.finalize(), cutoff=0., iprint=0)
        builder = driver.expr_builder()
        if key == "i":
            one = np.asarray(eris.get_h1eff("AI"))[order]
            three = np.asarray(eris.get_phys("AAIA"))[np.ix_(order, order, np.arange(count), order)]
            _add_tensor(builder, "CD", one, [active, external])
            _add_tensor(builder, "CCDD", three.transpose(0, 1, 3, 2),
                        [active, active, active, external])
        else:
            one = np.asarray(eris.get_h1eff("EA"))[:, order]
            three = np.asarray(eris.get_phys("EAAA"))[np.ix_(np.arange(count), order, order, order)]
            _add_tensor(builder, "CD", one, [external, active])
            _add_tensor(builder, "CCDD", three.transpose(0, 1, 3, 2),
                        [external, active, active, active])
        source_mpo = driver.get_mpo(builder.finalize(), cutoff=0., iprint=0)
        response = _nevpt_mps(driver, ncore, nvirt, key, bond)
        reported = driver.multiply(
            response, source_mpo, reference, left_mpo=dyall_mpo,
            n_sweeps=controls.n_sweeps, tol=tol,
            bond_dims=[int(reference.info.bond_dim) + 400], bra_bond_dims=[bond],
            thrds=[controls.linear_threshold] * controls.n_sweeps,
            linear_rel_conv_thrd=controls.linear_rel_conv_thrd,
            noises=[controls.noise] * controls.n_sweeps,
            cutoff=controls.cutoff, linear_max_iter=controls.linear_max_iter, iprint=iprint)
        if not np.isfinite(complex(reported)):
            raise RuntimeError("Block2 whole-class response returned a non-finite value")
        identity = driver.get_identity_mpo()
        reference_norm = _real_scalar(driver.expectation(reference, identity, reference),
                                      name="response reference norm")[0]
        if reference_norm <= 0:
            raise RuntimeError("response reference MPS has zero norm")
        overlap = driver.expectation(response, source_mpo, reference) / reference_norm
        quadratic = driver.expectation(response, dyall_mpo, response) / reference_norm
        susceptibility, imaginary = _real_scalar(overlap, name=f"whole {key} response energy")
        if susceptibility <= 0:
            raise RuntimeError(f"non-positive whole-class Dyall response {key}: {overlap!r}")
        # block2main takes the energy returned by Linear.solve. With our
        # positive left MPO its sign is reversed; post-truncation expectation
        # values are diagnostics, not a replacement energy prescription.
        e_corr = -_real_scalar(reported, name=f"Block2 whole {key} correlation energy")[0] / reference_norm
        if e_corr >= 0:
            raise RuntimeError(f"non-negative Block2 whole-class correction {key}: {reported!r}")
        return e_corr, dict(
            reference_norm=reference_norm, correlation_energy=e_corr,
            post_sweep_overlap_energy=-susceptibility,
            linear_energy_overlap_difference=float(abs(e_corr + susceptibility)),
            stationarity_residual=float(abs(complex(overlap) - complex(quadratic))),
            energy_imaginary_residual=imaginary,
            response_bond_dimension=int(response.info.bond_dim),
            reported_linear_value=[complex(reported).real, complex(reported).imag])
    finally:
        try:
            for state in (response, reference):
                if state is not None:
                    _release_response_mps(driver, state)
        finally:
            del source_mpo, dyall_mpo
            driver.__dict__.clear()
            driver.__dict__.update(original)


def evaluate_mps_response(mc, eris, pdms, core_energy, virtual_energy, *,
                          root=0, options=None, contraction_backend="numpy",
                          norm_tol=1e-14):
    """Block2 E4=None branch: whole aaac/aaav response, raw ranks 1--3 only."""
    from pyblock2.driver.core import SymmetryTypes

    from . import x2cscnevpt2 as sc
    driver, reference = u._root_ket(mc.fcisolver, root)
    if SymmetryTypes.SGF not in driver.symm_type or SymmetryTypes.CPX not in driver.symm_type:
        raise RuntimeError("MPS response requires a native complex SGF driver")
    if getattr(driver, "mpi", None) is not None:
        raise NotImplementedError("the SGFCPX response branch is currently serial/threaded")
    if np.any(np.asarray(driver.orb_sym) != 0):
        raise NotImplementedError("whole-class spinor response currently requires C1 orbitals")
    controls, bond, tol = _controls(mc.fcisolver, options)
    if u._has_frozen_orbitals(getattr(mc, "frozen", None)):
        raise NotImplementedError("nonzero frozen spinors are outside MPS response v1")
    if len(pdms) != 3:
        raise ValueError("response evaluation expects exactly ranks 1--3")
    order = np.arange(eris.ncas) if driver.reorder_idx is None else np.asarray(driver.reorder_idx)
    if not np.array_equal(np.sort(order), np.arange(eris.ncas)):
        raise RuntimeError("the retained DMRG orbital permutation is invalid")
    active_energy = _real_scalar(
        np.einsum("pq,pq", eris.get_h1eff("AA"), pdms[0])
        + .5 * np.einsum("pqrs,pqsr", eris.get_phys("AAAA"), pdms[1]),
        name="full-integral active reference energy")[0]
    active_mps = _active_mps(driver, reference)
    context = u._execution_context(eris, pdms)
    wick_globals = {"np": u._wick_einsum_namespace(contraction_backend)}
    energies, norms, gaps, diagnostics, timings = {}, {}, {}, {}, {}
    iprint = 1 if getattr(mc, "verbose", 0) >= logger.DEBUG else 0
    for key, orbital_energy in (("i", np.asarray(core_energy)), ("r", np.asarray(virtual_energy))):
        start = time.perf_counter()
        source_norms = np.zeros(len(orbital_energy), dtype=complex)
        exec(sc._compile_wick_equations().norm_code[key], wick_globals,
             {**context, "norm": source_norms})
        source_norm = _real_scalar(np.sum(source_norms), name=f"whole {key} source norm")[0]
        if np.any(source_norms.real < -norm_tol):
            raise RuntimeError(f"negative source norm in whole {key} response")
        if source_norm > norm_tol:
            energies[key], entry = _whole_class_response(
                driver, active_mps, eris, order, key, orbital_energy,
                active_energy, controls, bond, tol, iprint)
            effective_gap = source_norm / -energies[key]
        else:
            energies[key], entry, effective_gap = 0., {"zero_source": True}, float("nan")
        norms[key], gaps[key] = source_norm, (effective_gap, effective_gap)
        diagnostics[key] = dict(
            method="uncontracted_mps_response", representation="whole_class",
            response_class="aaac" if key == "i" else "aaav", solver="pyblock2.Linear",
            space_restriction="NEVPTMPSInfo", uses_4rdm=False, dyall_integrals="full",
            active_reference_energy=active_energy, source_norm=source_norm,
            entries=[entry], maximum_stationarity_residual=entry.get("stationarity_residual", 0.),
            controls={**controls.__dict__, "max_bond_dimension": bond, "tol": tol})
        timings[key] = time.perf_counter() - start
    return energies, norms, gaps, diagnostics, timings
