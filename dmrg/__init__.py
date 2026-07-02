"""DMRG solver for spinor (relativistic) CASCI."""

from socutils.dmrg.dmrgci import DMRGCI


def _inject_initial_dmrg():
    """Inject ``initial_dmrg`` into CASBase for convenience."""
    from socutils.mcscf.zcasbase import CASBase

    if hasattr(CASBase, 'initial_dmrg'):
        return

    def _initial_dmrg(self, nroots=1, bond_dims=None, noises=None,
                      thrds=None, tol=1e-6, scratch=None, n_threads=None):
        """Attach a :class:`DMRGCI` solver and return *self*."""
        fcisolv = DMRGCI(self._scf.mol)
        fcisolv.init(
            ncas=self.ncas, nelecas=self.nelecas,
            nroots=nroots,
            bond_dims=bond_dims, noises=noises, thrds=thrds,
            tol=tol, scratch=scratch, n_threads=n_threads,
        )
        self.fcisolver = fcisolv
        return self

    CASBase.initial_dmrg = _initial_dmrg


_inject_initial_dmrg()
