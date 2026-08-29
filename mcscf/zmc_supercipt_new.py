"""Compatibility entry point for the Guo--Dutta Super-CIPT driver.

The implementation lives in :mod:`socutils.mcscf.zmc_supercipt`.  This module
preserves the interface used by the contributed Nd3+(H2O)8 example without
reintroducing its older CASCI and integral-transformation code paths.
"""

from socutils.mcscf.zmc_supercipt import (
    OrbitalQuantities,
    SuperCIPTStep,
    build_koopmans_matrices,
    build_orbital_quantities,
    mcscf_supercipt,
    solve_metric_eigenproblem,
    supercipt_step,
)
from socutils.tools import analyze_casscf_spinors


def mcscf_superci_pt(
    mc,
    m=None,
    symm=None,
    max_cycle=100,
    conv_etol=1e-8,
    conv_gtol=1e-3,
    max_step=0.2,
    use_diis=False,
    use_cderi=False,
):
    """Run Super-CIPT and return ``(converged, energy, final_orbitals)``.

    ``m`` is retained for source compatibility.  The CASSCF object's attached
    SCF reference is authoritative, which prevents the two objects from
    silently supplying inconsistent Hamiltonians or overlap matrices.
    """
    if m is not None and getattr(m, "mol", None) is not mc.mol:
        raise ValueError("m and mc must refer to the same molecule")
    converged, energy, _, _, orbitals, _ = mcscf_supercipt(
        mc,
        mc.mo_coeff,
        max_stepsize=max_step,
        conv_tol=conv_etol,
        conv_tol_grad=conv_gtol,
        max_cycle=max_cycle,
        use_diis=use_diis,
        use_cderi=use_cderi,
        symm=symm,
        verbose=mc.verbose,
        cderi=getattr(mc, "_cderi", None),
        callback=getattr(mc, "callback", None),
    )
    return converged, energy, orbitals


__all__ = [
    "OrbitalQuantities",
    "SuperCIPTStep",
    "analyze_casscf_spinors",
    "build_koopmans_matrices",
    "build_orbital_quantities",
    "mcscf_superci_pt",
    "mcscf_supercipt",
    "solve_metric_eigenproblem",
    "supercipt_step",
]
