Multiconfigurational SCF (mcscf)
================================

socutils provides CASCI and CASSCF on a two-component reference -- a spinor
mean field (``spinor_hf``) or a GHF object (``ghf.GHF``).  Because the
reference is two-component, the active space is counted in **spinor
(spin-orbital) orbitals**: ``ncas`` active spinors holding ``nelecas``
electrons, with ``nelecas`` no larger than ``ncas``.

CASCI
-----

``zcasci.CASCI(mf, ncas, nelecas, ncore=None)`` runs a complete-active-space
configuration interaction on top of a converged mean field.  ``kernel()``
returns ``(e_tot, e_cas, ci, ...)`` and stores ``mc.e_tot`` / ``mc.e_cas`` /
``mc.ci``.

.. code-block:: python

   from pyscf import gto
   from socutils.scf import spinor_hf
   from socutils.mcscf import zcasci

   mol = gto.M(atom='H 0 0 0; F 0 0 0.917', basis='ccpvdz', verbose=4)

   mf = spinor_hf.SCF(mol).x2camf()
   mf.kernel()

   mc = zcasci.CASCI(mf, 8, 6)   # 6 electrons in 8 active spinors
   mc.kernel()
   print(mc.e_tot, mc.e_cas)

The default CI solver is socutils' own ``fci.FCISolver`` (see
`Full CI and selected CI`_ below).  Useful attributes:

* ``ncore`` -- number of core (doubly counted) orbitals; inferred from the
  electron count if not given;
* ``frozen`` -- orbitals to keep frozen;
* ``natorb`` -- transform the active space to natural orbitals;
* ``canonicalization`` -- canonicalize the core/external blocks (default
  ``True``);
* ``fcisolver`` -- the CI solver, which can be replaced (see below).

CASSCF
------

``zmcscf.CASSCF(mf, ncas, nelecas, ncore=None, frozen=None)`` additionally
optimizes the orbitals.  ``kernel()`` drives a **super-CI** orbital optimizer:
each macro-iteration solves the active-space CI problem, builds the orbital
gradient and an approximate Hessian, and takes a Kramers-paired orbital
rotation step, repeating until the energy and orbital gradient are converged.

.. code-block:: python

   from pyscf import gto
   from socutils.scf import spinor_hf
   from socutils.mcscf import zmcscf

   mol = gto.M(atom='H 0 0 0; F 0 0 0.917', basis='ccpvdz', verbose=4)

   # Omitting .cholesky()/.density_fit() selects full four-index ERIs in
   # every Super-CI macroiteration.
   mf = spinor_hf.SCF(mol).x2camf()
   mf.kernel()

   mc = zmcscf.CASSCF(mf, 8, 6)   # 6 electrons in 8 active spinor orbitals
   mc.kernel()
   print(mc.e_tot)

Orbital DIIS can be enabled explicitly for the full Super-CI optimizer::

   mc.superci(use_diis=True)

For a Kramers-restricted reference or solver, DIIS must be told to retain the
time-reversal tangent space::

   mc.superci(use_diis=True, symm='kramers')

Full Super-CI DIIS extrapolates unitary orbital transformations in one fixed
reference frame.  Every extrapolated step is screened through the normal
frozen/irrep mask and, in Kramers mode, through the actual AO time-reversal
map.  Partner orbitals therefore need not be adjacent.  For ``kernel()``-style
input, set ``mc.superci_diis = True`` and, when needed,
``mc.orbital_symmetry = 'kramers'``.  Shared controls are
``orbital_diis_space=15``, ``orbital_diis_start_cycle=3`` and
``orbital_diis_start_gradient=0.02``.

Spinor orbital analysis
~~~~~~~~~~~~~~~~~~~~~~~

``analyze_casscf_spinors`` prints the dominant spinor-AO coefficients of the
optimized orbitals.  It analyzes the active space by default, making it useful
for checking or refining a spinor active-space selection::

   from socutils.tools import analyze_casscf_spinors

   analyze_casscf_spinors(mc, threshold=0.05)
   analyze_casscf_spinors(mc, threshold=0.05, mo_type='all')

The threshold is applied to the absolute value of the complex AO coefficient;
the printed real and imaginary parts are coefficients rather than
overlap-weighted AO populations.

Requirements
~~~~~~~~~~~~

* **zquatev** -- the orbital step is solved with the Kramers-paired
  (quaternion) eigensolver, so the bundled ``zquatev`` solver must be built
  (see :doc:`../install`); ``kernel()`` raises a clear error if it is missing.

The integral route follows the SCF object.  A plain mean field, with no
``with_df``, selects the full four-index spinor transformation.  Attaching
``.density_fit()`` or ``.cholesky()`` selects the factorized route instead.
The same choice is retained whenever Super-CI rebuilds transformed integrals
after an orbital step.  See
:ref:`the Cholesky decomposition section <cholesky-decomposition>` for the CD
route and its on-disk caching.

Block2 DMRG and Kramers pairs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``socutils.dmrg.DMRGCI`` is a PySCF-style Block2 solver for the same complex
spinor Hamiltonian and full-spinor 1-/2-RDM conventions as ``fci.FCISolver``.
For an ordinary general-complex calculation, attach it directly.  Kramers
mode is an explicit additional call; for an odd-electron doublet, request both
roots and give the pair equal state-average weights:

.. code-block:: python

   from socutils.dmrg import DMRGCI
   from socutils.mcscf import zmcscf

   dmrg = DMRGCI(mol).init(
       ncas=2, nelecas=1, nroots=2,
       bond_dims=[32] * 8,
       noises=[0.0] * 8,
       thrds=[1e-20] * 8,       # squared local Davidson residual
       n_sweeps=8, tol=1e-12,
       n_threads=1, stack_memory=256,
       scratch='/path/to/scratch',
   ).kramers_restricted()

   mc = zmcscf.CASSCF(mf, ncas=2, nelecas=1)
   mc.fcisolver = dmrg
   mc.state_average_([0.5, 0.5])
   dmrg = mc.fcisolver
   mc.callback = dmrg.restart_scheduler_()
   mc.kernel()

   # state_average_ installs a PySCF solver view, so read final diagnostics
   # from the solver held by mc.
   print(mc.fcisolver.kramers_diagnostics)

The CASSCF boundary derives the time-reversal matrix from the *current* active
MO coefficients at every macroiteration.  It validates active-space closure,
partner indices, and partner phases rather than assuming that ``2*i`` and
``2*i+1`` are paired.  Individual odd-electron roots are not forced to have a
Kramers-symmetric density.  All roots are optimized together in one
state-averaged ``MultiMPS`` using the same weights as the CASSCF functional.
Multi-root jobs finish with one-site sweeps, after which ``DMRGCI`` verifies
the complete split-root overlap and projected eigen-equation matrices.
Complete degenerate manifolds are then validated and averaged in the full
active spinor space.

``mc.fcisolver.kramers_diagnostics`` reports root pairs, raw partner and
ensemble time-reversal residuals, root orthogonality, the projected
Hamiltonian residual, any projection change, and ``validation_passed`` /
``validation_warnings``.  A numerical energy-degeneracy or RDM
time-reversal residual above its tolerance emits a warning and leaves the raw
result available instead of aborting the CASSCF.  Structural input errors,
inconsistent split MPS roots, and unsafe requested projections remain hard
errors.  Projection is disabled by default.
``.kramers_restricted(project=True)`` permits only a final
roundoff-level ``(D + Theta(D))/2`` projection after the raw residual has
passed ``projection_tolerance``; it is never a substitute for converging the
DMRG or fixing an index error.  The pair RDM and phase/mixing-canonicalized
transition tensor are available through
``make_kramers_pair_rdm12()`` and
``canonical_kramers_root_space_rdm1()``.

The general complex-spinor route is unchanged when
``kramers_restricted()`` is omitted.  Exact settings and numerical validation
against ``fci.FCISolver`` are recorded in
``docs/x2c_dmrg_validation.md``.

For production relativistic calculations, ``DMRGCI`` defaults to
``max_bond_dimension=1000``, ``tol=1e-8``, and a staged Davidson
squared-residual schedule from ``1e-8`` to ``schedule_thrd_max=1e-16``.
Before constructing the MPO, the solver obtains Block2's Fiedler ordering
from ``abs(h1e)`` and ``abs(g2e)`` and permutes both integral tensors.  The
permutation is registered on the driver after the sweep and MultiMPS split,
so all normal and transition RDMs exposed to PySCF retain the original active-
spinor indices.  Warm starts and disk resumes preserve the prior MPS site
ordering even if the newly evaluated Fiedler proposal changes.
The completed checkpoint is explicitly overwritten with the final canonical
one-site MultiMPS.  When resuming an older checkpoint whose saved metadata is
still two-site, the solver first performs two genuine two-site conversion
sweeps and only then runs the configured one-site restart sweeps; it never
relabels a two-site tensor image as one-site.
Restart sweeps use ``tol=0`` so the configured count is completed.  The sole
exception is a multi-root active space with at most two sites: Block2 has no
interior one-site tensor there and cannot safely perform repeated direction
changes, so that exact, negligible-cost edge case is solved cold instead of
loading a one-site warm start.
Scratch paths, checkpoint policy, thread count, and stack memory remain
machine- and job-specific input options.

Perturbative Super-CI (Super-CIPT)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``CASSCF.supercipt()`` is the Guo--Dutta two-component perturbative Super-CI
orbital optimizer.  It is an explicit alternative API: ``kernel()`` and
``superci()`` continue to use the full Super-CI/Davidson implementation.

.. code-block:: python

   from socutils.dmrg import DMRGCI
   from socutils.mcscf import zmcscf

   solver = DMRGCI(mol).init(
       ncas=4, nelecas=2, nroots=1,
       bond_dims=[64] * 8,
       noises=[0.0] * 8,
       thrds=[1e-20] * 8,
       n_sweeps=8, tol=1e-12,
       n_threads=1, stack_memory=256,
       scratch='/path/to/scratch',
   )
   mc = zmcscf.CASSCF(mf, ncas=4, nelecas=2)
   mc.fcisolver = solver
   mc.supercipt(use_diis=True, use_cderi=True)

The method uses the same exact-CI/Block2 solver contract and full or Cholesky
integral containers as Super-CI.  In the repository RDM convention,
``dm1[p,q] = <p+ q>`` and ``dm2[p,q,r,s] = <p+ r+ s q>``.  The active
two-particle contraction is

.. code-block:: text

   Q[p,t] = sum_u,v,w eris.paaa[p,u,v,w] * dm2[t,u,v,w]

Because this package stores ``D[p,q] = <p+ q>`` while the Koopmans matrices
carry the reversed projected-Hamiltonian orientation, the removal and
addition generalized eigenproblems use ``D.T`` and ``(I-D).T``, respectively.
They are positive-semidefinite metrics; numerically null directions are
removed by canonical orthogonalization.  The three paper blocks
(core--virtual, core--active, and active--virtual) form one anti-Hermitian
rotation, whose largest matrix element is capped by ``max_stepsize`` before
applying ``C <- C exp(kappa)``.

Without DIIS, Super-CIPT semicanonicalizes the redundant core and virtual
blocks after every orbital step, independently of ``mc.canonicalize_``.  With
DIIS, it follows the contributed algorithm instead: local perturbative steps
are accumulated and Pulay-extrapolated in one incremental coordinate system,
and no intervening core/virtual re-gauging is applied.  Mixing those redundant
gauge rotations into a fixed-reference DIIS history is unstable for atomic Cl.
The usual HF or previously semicanonicalized starting orbitals are therefore
recommended for the DIIS path.

Super-CIPT uses the common CASSCF attributes ``max_cycle_macro``,
``max_stepsize``, ``conv_tol``, and ``conv_tol_grad``.  Its additional
attributes are:

* ``supercipt_metric_tol`` (``1e-6``) -- discard density/hole-metric
  eigenvectors at or below this value;
* ``supercipt_denominator_tol`` (``1e-10``) -- reject a singular Dyall
  denominator rather than divide silently;
* ``supercipt_level_shift`` (``0.0``) -- optional sign-preserving shift away
  from zero;
* ``supercipt_diis`` (``False``) -- enable orbital DIIS;
* ``supercipt_use_cderi`` (``None``) -- automatically use factors attached to
  the SCF object.  ``True`` forces the factorized route and generates AO
  Cholesky vectors once if necessary; ``False`` forces full integral
  transformation.

The same options can be passed directly to ``supercipt``.  For example, a
Kramers-restricted calculation with both new accelerators is::

   mc.supercipt(
       symm='kramers',
       use_diis=True,
       use_cderi=True,
   )

The explicit ``symm='kramers'`` is mandatory when DIIS is enabled on a KRHF
reference or Kramers-adapted active-space solver.  Without DIIS, Kramers mode
is detected automatically.  The implementation identifies partner indices
and phases from ``C.H @ S @ Theta(C)``, projects every orbital generator, and
uses a quaternion eigensolve whenever core/virtual semicanonicalization is
requested.  Super-CIPT DIIS uses the perturbative correction itself as the
Pulay residual; full Super-CI retains its fixed-reference gradient DIIS.
Each extrapolated Super-CIPT step is accepted only after the next CASCI energy
evaluation.  An uphill extrapolation is recorded as rejected, the Pulay
history is reset, and the optimizer evaluates the ordinary perturbative step
from the same source orbitals instead.  If this happens at the cycle limit,
one terminal energy evaluation is reserved for that fallback so the returned
energy, RDMs, and orbitals describe the same accepted point.

``max_stepsize`` is an upper bound, not an acceleration factor.  If the
reported ``max(raw)`` is already much smaller than ``max_stepsize``, increasing
the option cannot make Super-CIPT move faster.  This commonly occurs when a
large active space must be reselected from mean-field orbitals: the
first-order Dyall denominators can produce a long linear-convergence region.
In that case use better selected starting orbitals or pre-optimize them with
the full Super-CI path; do not loosen the DMRG or denominator tolerances to
hide the orbital problem.

Convergence requires both the energy and the nonredundant orbital gradient.
``supercipt_history`` records energies, root energies, gradients, natural
occupations, RDM-energy checks, metric ranks, denominators, step scales, and
integral provenance; ``supercipt_diagnostics`` records the final settings and
status.  PySCF ``state_average_(weights)`` is supported and supplies the same
weighted energy and RDMs to every Super-CIPT equation.

At normal logging level each update prints its raw maximum rotation, applied
scale, rotation norm, minimum absolute denominator, and both metric ranks.
These distinguish an orbital-optimizer plateau from CI nonconvergence,
metric truncation, an intruder denominator, or trust-radius clipping.

``CASSCF.supercipt(...)`` is the single supported public driver and delegates
to the implementation in ``socutils.mcscf.zmc_supercipt``.  The obsolete
contributed ``zmc_supercipt_new.mcscf_superci_pt`` compatibility wrapper is
not retained.  The paper/source equation map, immutable historical output,
and exact/Pykylin/Block2 numerical ladder are recorded in
``docs/supercipt_validation.md``.

Orbital localization
~~~~~~~~~~~~~~~~~~~~

General real or complex orbitals can be Boys-localized over a selected column
interval without changing that orbital subspace::

   from socutils.lo import localize_boys

   mo_local = localize_boys(
       mol, mc.mo_coeff,
       start=mc.ncore,
       stop=mc.ncore + mc.ncas,
   )

For a Kramers-complete interval, use the time-reversal-constrained driver::

   from socutils.lo import localize_boys_kramers

   mo_local, info = localize_boys_kramers(
       mol, mc.mo_coeff,
       start=mc.ncore,
       stop=mc.ncore + mc.ncas,
       return_info=True,
   )

``localize_boys_kramers`` derives the actual partner map in the AO overlap
metric and projects all localization rotations onto the quaternionic unitary
space.  It does not assume ``(2*i, 2*i+1)`` ordering.  ``info`` records the
initial/final Boys objective, gradient, cycle history, unitary rotation and
the final time-reversal residual.  Compatibility wrappers ``boys_driver`` and
``boys_driver_kramers`` retain the old in-place calling style.

Options
~~~~~~~

The optimization is controlled by attributes set on the ``CASSCF`` object
(defaults in parentheses):

* ``max_cycle_macro`` (``50``) -- maximum number of macro-iterations;
* ``max_stepsize`` (``0.2``) -- trust radius capping each orbital-rotation step;
* ``conv_tol`` (``1e-8``) -- energy convergence threshold;
* ``conv_tol_grad`` (``1e-4``) -- orbital-gradient convergence threshold;
* ``natorb`` (``False``) -- if enabled, rotate the active orbitals at each
  macro-iteration to natural orbitals (eigenvectors of the active 1-RDM,
  ordered by descending occupation);
* ``canonicalization`` (``True``) -- after the last Super-CI macroiteration,
  diagonalize the generalized-Fock core and virtual blocks, leave the active
  orbitals and CI/MPS unchanged, and populate ``mo_energy``;
* ``canonicalize_`` (``False``) -- legacy in-macroiteration redundant-gauge
  rotation.  It is independent of the standard final ``canonicalization``
  switch and is not required by post-CASSCF methods;
* ``frozen`` (``None``) -- orbitals excluded from rotation; an ``int`` freezes the
  lowest ``frozen`` orbitals, a list/array freezes the listed indices;
* ``freeze_pair`` (``None``) -- a pair of index sets ``(set_i, set_j)`` whose
  mutual rotations are frozen (the rest are still optimized);
* ``irrep`` (``None``) -- per-orbital symmetry labels; rotations are then allowed
  only between orbitals carrying the same label.
* ``superci_solver`` (``'davidson'``) -- linear solver for the Super-CI orbital
  equation; this is independent of any local Davidson solver used by a DMRG CI
  solver;
* ``superci_davidson_tol`` (``1e-8``) -- norm tolerance for the full generalized
  augmented-Hessian residual;
* ``superci_davidson_max_space`` (``200``) -- maximum Super-CI Davidson subspace;
* ``superci_davidson_strict`` (``True``) -- raise instead of applying an orbital
  step when the configured Super-CI residual was not reached.

Convergence and results
~~~~~~~~~~~~~~~~~~~~~~~~~~

A macro-iteration is accepted as converged when **both** the energy change and
the orbital-gradient norm fall below their thresholds
(``abs(dE) < conv_tol`` and ``norm(grad) < conv_tol_grad``); otherwise the loop
stops at ``max_cycle_macro``.  ``kernel()`` returns
``(e_tot, e_cas, ci, mo_coeff, mo_energy)`` and sets the attributes

* ``mc.e_tot`` -- total CASSCF energy;
* ``mc.e_cas`` -- active-space (CI) energy;
* ``mc.ci`` -- the active-space CI vector;
* ``mc.mo_coeff`` / ``mc.mo_energy`` -- optimized orbitals and generalized-Fock
  orbital energies; ``mo_energy`` is populated by final canonicalization and is
  ``None`` when ``mc.canonicalization=False``;
* ``mc.converged`` -- whether both convergence criteria were met.
* ``mc.final_orbital_gradient_norm`` -- norm tested at the final macroiteration;
* ``mc.macro_history`` -- energy, CAS energy, gradient, applied step, natural
  occupations, CI convergence data, and Super-CI residual for each
  macroiteration;
* ``mc.cholesky_diagnostics`` -- backward-compatible integral provenance,
  including whether the route is full or factorized and, for a genuine
  ``CD`` source, its threshold and vector count;
* ``mc.canonicalization_diagnostics`` -- core/virtual Fock off-diagonal norms
  before and after the final rotation, active-orbital preservation, and
  orthonormality checks;
* ``mc.superci_diagnostics`` -- final convergence thresholds, linear-solver
  residual, and the generic ``integrals`` provenance record.

The production defaults above match the validated F CAS(7e,16 spinor)
protocol.  They remain ordinary attributes and can be overridden for smaller
validation jobs or deliberately different convergence studies.

Full CI and selected CI
-----------------------

socutils ships its own spinor CI module, ``socutils.fci``, which is the
recommended (and default) CI solver.  It is a drop-in replacement for both
PySCF's ``fci_dhf_slow`` and the Dice-based SHCI interface (``socutils.hci``).

``fci.FCISolver`` (alias ``fci.FCI``)
    Exact full CI by direct construction and diagonalization of the
    Hamiltonian in the determinant basis.  All roots come from a single
    diagonalization, which avoids the Davidson convergence problems that the
    Kramers-degenerate roots cause for iterative solvers.  This is the default
    ``mc.fcisolver``; set ``mc.fcisolver.nroots`` for several states.

    .. code-block:: python

       from socutils.mcscf import zcasci
       from socutils.fci import zfci

       mc = zcasci.CASCI(mf, 8, 6)
       mc.fcisolver = zfci.FCISolver(mol)   # (this is also the default)
       mc.fcisolver.nroots = 4
       mc.kernel()
       print(mc.fcisolver.eci)              # the individual root energies

``fci.SelectedCI(mol, occslst=...)``
    Diagonalization in a chosen list of determinants (the replacement for the
    SHCI interface).  Combined with ``zfci.gen_ras_occslst`` it also expresses
    RASCI-type determinant spaces.

The companion ``fci.addons`` module provides post-processing for any solver
that exposes ``trans_rdm1`` (``FCISolver`` and ``SelectedCI``): transition
dipoles, oscillator strengths, Einstein coefficients / radiative lifetimes,
and ``spin_square`` / ``angular_momentum_square`` for analysing states.  See
the ``examples/fci`` directory.

Configuration-averaged solvers
-------------------------------

``zcahf`` provides configuration-averaged solvers that return averaged density
matrices instead of solving a CI -- useful for averaging over a degenerate
open shell:

``zcahf.CAHF(mol)``
    Configuration-averaged Hartree-Fock: spreads the active electrons evenly
    over the active orbitals (a single averaged configuration).

    .. code-block:: python

       from socutils.mcscf import zcasci, zcahf

       mc = zcasci.CASCI(mf, 8, 6)
       mc.fcisolver = zcahf.CAHF(mol)
       mc.kernel()

``zcahf.MultiSlater(mol, det_list, weight_list)``
    Average over an explicit list of Slater determinants with given weights.

``zcahf.MultiZCAHF(mol, orb_open, elec_open)``
    Configuration averaging over multiple open shells, specified by the
    open-shell orbital and electron counts.
