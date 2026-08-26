"""DMRG solver and Kramers result adapter for relativistic spinor CASCI."""

from socutils.dmrg.dmrgci import DMRGCI
from socutils.dmrg.kramers import (
    KramersOrbitalMap,
    KramersManifoldRDM,
    KramersPairRDM,
    KramersResultAdapter,
    align_transition_phase,
    ao_time_reverse,
    canonicalize_root_space_rdm1,
    identify_kramers_orbitals,
    kramers_residual,
    time_reverse_integrals,
    time_reverse_one_body,
    time_reverse_rdm1,
    time_reverse_rdm2,
)


def _inject_initial_dmrg():
    """Inject ``initial_dmrg`` into CASBase for convenience."""
    from socutils.mcscf.zcasbase import CASBase

    if hasattr(CASBase, 'initial_dmrg'):
        return

    def _initial_dmrg(self, nroots=1, bond_dims=None, noises=None,
                      thrds=None, n_sweeps=None, tol=1e-6, scratch=None,
                      n_threads=None, **kwargs):
        """Attach a :class:`DMRGCI` solver and return *self*."""
        fcisolv = DMRGCI(self._scf.mol)
        fcisolv.init(
            ncas=self.ncas, nelecas=self.nelecas,
            nroots=nroots,
            bond_dims=bond_dims, noises=noises, thrds=thrds,
            n_sweeps=n_sweeps, tol=tol, scratch=scratch,
            n_threads=n_threads, **kwargs,
        )
        self.fcisolver = fcisolv
        return self

    CASBase.initial_dmrg = _initial_dmrg


_inject_initial_dmrg()
