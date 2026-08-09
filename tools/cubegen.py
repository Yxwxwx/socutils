import os
import time
import numpy
import pyscf
from pyscf import lib
from pyscf import gto
from pyscf import df
from pyscf.dft import numint
from pyscf import __config__
from pyscf.tools.cubegen import Cube

RESOLUTION = getattr(__config__, 'cubegen_resolution', None)
BOX_MARGIN = getattr(__config__, 'cubegen_box_margin', 3.0)
ORIGIN = getattr(__config__, 'cubegen_box_origin', None)
# If given, EXTENT should be a 3-element ndarray/list/tuple to represent the
# extension in x, y, z
EXTENT = getattr(__config__, 'cubegen_box_extent', None)

def orbital(mol, coeff, outfile_amplitude='orbValue.cub', outfile_angle='orbPhase.cub',
            nx=80, ny=80, nz=80, resolution=RESOLUTION, margin=BOX_MARGIN):
    """Calculate orbital value on real space grid and write out in cube format.

    Args:
        mol : Mole
            Molecule to calculate the electron density for.
        
        coeff : 1D array
            coeff coefficient.

    Kwargs:
        outfile_amplitude : str
            Name of Cube file to be written; one file per spin component,
            with _alpha/_beta inserted before the extension.
        outfile_angle : str
            Name of Cube file to be written; one file per spin component,
            with _alpha/_beta inserted before the extension.
        nx : int
            Number of grid point divisions in x direction.
            Note this is function of the molecule's size; a larger molecule
            will have a coarser representation than a smaller one for the
            same value. Conflicts to keyword resolution.
        ny : int
            Number of grid point divisions in y direction.
        nz : int
            Number of grid point divisions in z direction.
        resolution: float
            Resolution of the mesh grid in the cube box. If resolution is
            given in the input, the input nx/ny/nz have no effects.  The value
            of nx/ny/nz will be determined by the resolution and the cube box
            size.
    """
    # TODO: optional global phase gauge. LAPACK eigenvector phases are random,
    # and phase values at the +-pi branch cut interpolate badly in cube viewers
    # (VESTA: linear colormap only, no cyclic wheel). Rotating the phase of the
    # largest-amplitude grid point to +pi/2 puts near-real orbitals' lobes at
    # +-pi/2, away from the cut, reproducibly across runs.
    # TODO: for genuinely complex orbitals (magnetic field / strong SOC) no
    # global phase shift avoids the cut. Optionally write cos(theta)/sin(theta)
    # cubes instead (unit phasor interpolates cleanly through the cut) and
    # reconstruct the phase per vertex in the viewer, e.g. ParaView Calculator
    # atan2(sin, cos) with a cyclic HSV colormap.
    cc = Cube(mol, nx, ny, nz, resolution, margin)

    GTOval = 'GTOval_spinor'

    # Compute the two spinor components (alpha, beta) on the .cube grid
    coords = cc.get_coords()
    ngrids = cc.get_ngrids()
    blksize = min(8000, ngrids)
    orb_on_grid = numpy.zeros((2, ngrids), dtype=numpy.complex128)
    for ip0, ip1 in lib.prange(0, ngrids, blksize):
        ao = mol.eval_gto(GTOval, coords[ip0:ip1])
        orb_on_grid[:,ip0:ip1] = numpy.dot(ao, coeff)

    # Following Al-Saadon, Shiozaki, Knizia, J. Phys. Chem. A 123, 3223 (2019):
    # each spin component of the spinor is a complex scalar field, visualized
    # by its amplitude |psi(r)| and phase angle arg[psi(r)] separately.
    for i, spin in enumerate(('alpha', 'beta')):
        amp_on_grid = numpy.abs(orb_on_grid[i]).reshape(cc.nx,cc.ny,cc.nz)
        ang_on_grid = numpy.angle(orb_on_grid[i]).reshape(cc.nx,cc.ny,cc.nz)
        base, ext = os.path.splitext(outfile_amplitude)
        cc.write(amp_on_grid, f'{base}_{spin}{ext}',
                 comment=f'Amplitude of the {spin} spinor component (1/Bohr^(3/2))')
        base, ext = os.path.splitext(outfile_angle)
        cc.write(ang_on_grid, f'{base}_{spin}{ext}',
                 comment=f'Phase angle from -pi to pi of the {spin} spinor component')
    return orb_on_grid.reshape(2,cc.nx,cc.ny,cc.nz)

def density(mol, coeff, outfile='density.cub',
            nx=80, ny=80, nz=80, resolution=RESOLUTION, margin=BOX_MARGIN):
    """Calculate density value on real space grid and write out in cube format.

    Args:
        mol : Mole
            Molecule to calculate the electron density for.
        
        coeff : 1D array
            coeff coefficient.

    Kwargs:
        outfile : str
            Name of Cube file to be written.
        nx : int
            Number of grid point divisions in x direction.
            Note this is function of the molecule's size; a larger molecule
            will have a coarser representation than a smaller one for the
            same value. Conflicts to keyword resolution.
        ny : int
            Number of grid point divisions in y direction.
        nz : int
            Number of grid point divisions in z direction.
        resolution: float
            Resolution of the mesh grid in the cube box. If resolution is
            given in the input, the input nx/ny/nz have no effects.  The value
            of nx/ny/nz will be determined by the resolution and the cube box
            size.
    """
    cc = Cube(mol, nx, ny, nz, resolution, margin)

    GTOval = 'GTOval_spinor'

    # Compute density on the .cube grid
    coords = cc.get_coords()
    ngrids = cc.get_ngrids()
    blksize = min(8000, ngrids)
    orb_on_grid = numpy.zeros((2, ngrids), dtype=numpy.complex128)
    for ip0, ip1 in lib.prange(0, ngrids, blksize):
        ao = mol.eval_gto(GTOval, coords[ip0:ip1])
        orb_on_grid[:,ip0:ip1] = numpy.dot(ao, coeff)
    den_on_grid = numpy.einsum("ip,ip->p", orb_on_grid.conj(), orb_on_grid).real
    den_on_grid = den_on_grid.reshape(cc.nx,cc.ny,cc.nz)
    cc.write(den_on_grid, outfile, comment='Density value in real space (1/Bohr^3)')
    return den_on_grid