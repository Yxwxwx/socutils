# SPDX-License-Identifier: GPL-3.0-or-later
"""Matrix-free evaluation of the native CASPT2 spinor Wick equations.

There are no spin labels, spin traces, spatial RDMs or spin-free generators
here. Ordered antisymmetric spinor coordinates are exactly those of the
dense oracle in x2ccaspt2. Only a fixed-external-index active metric is dense.
"""

from dataclasses import dataclass
import itertools
import math
import re
import string
import time

import numpy as np
from scipy import linalg
from scipy.sparse.linalg import LinearOperator, gmres

from . import x2ccaspt2 as native


@dataclass
class _Sector:
    name: str
    families: tuple
    indices: np.ndarray
    external: np.ndarray
    overlap: np.ndarray
    active_fock: np.ndarray
    gap: np.ndarray
    transform: np.ndarray | None = None
    denominators: np.ndarray | None = None
    orth_slice: slice | None = None


@dataclass(frozen=True)
class _ScalarFamily:
    sector: str
    suffix: str
    variables: tuple[str, ...]
    terms: tuple[tuple[float, tuple[str, str], tuple[str, str]], ...]
    active_roles: tuple[str, ...]
    restrictions: tuple[tuple[int, int, bool], ...] = ()


# These are not scalar CASPT2 equations. They define only the conventional
# Molcas spin-adapted primitive IC coordinates whose IPEA correction must be
# represented in a native spinor response space. E[p,q] is expanded exactly
# as sum_sigma C[p,sigma]D[q,sigma].
_MOLCAS_FAMILIES = (
    _ScalarFamily('ijrs', '+', tuple('ijrs'), ((1., ('r','i'), ('s','j')), (1., ('s','i'), ('r','j'))), (), ((0,1,False),(2,3,False))),
    _ScalarFamily('ijrs', '-', tuple('ijrs'), ((1., ('r','i'), ('s','j')), (-1., ('s','i'), ('r','j'))), (), ((0,1,True),(2,3,True))),
    _ScalarFamily('rsi', '+', tuple('rsia'), ((1., ('r','i'), ('s','a')), (1., ('s','i'), ('r','a'))), ('D',), ((0,1,False),)),
    _ScalarFamily('rsi', '-', tuple('rsia'), ((1., ('r','i'), ('s','a')), (-1., ('s','i'), ('r','a'))), ('D',), ((0,1,True),)),
    _ScalarFamily('ijr', '+', tuple('ijra'), ((1., ('r','j'), ('a','i')), (1., ('r','i'), ('a','j'))), ('C',), ((0,1,False),)),
    _ScalarFamily('ijr', '-', tuple('ijra'), ((1., ('r','j'), ('a','i')), (-1., ('r','i'), ('a','j'))), ('C',), ((0,1,True),)),
    _ScalarFamily('rs', '+', tuple('rsab'), ((1., ('r','b'), ('s','a')), (1., ('s','b'), ('r','a'))), ('D','D'), ((0,1,False),(2,3,False))),
    _ScalarFamily('rs', '-', tuple('rsab'), ((1., ('r','b'), ('s','a')), (-1., ('s','b'), ('r','a'))), ('D','D'), ((0,1,True),(2,3,True))),
    _ScalarFamily('ij', '+', tuple('ijab'), ((1., ('b','i'), ('a','j')), (1., ('b','j'), ('a','i'))), ('C','C'), ((0,1,False),(2,3,False))),
    _ScalarFamily('ij', '-', tuple('ijab'), ((1., ('b','i'), ('a','j')), (-1., ('b','j'), ('a','i'))), ('C','C'), ((0,1,True),(2,3,True))),
    _ScalarFamily('ir', '1', tuple('irab'), ((1., ('r','i'), ('a','b')),), ('C','D')),
    _ScalarFamily('ir', '2', tuple('irab'), ((1., ('a','i'), ('r','b')),), ('C','D')),
    _ScalarFamily('r', '', tuple('rabc'), ((1., ('r','b'), ('a','c')),), ('C','D','D')),
    _ScalarFamily('i', '', tuple('iabc'), ((1., ('b','i'), ('a','c')),), ('C','C','D')),
)


def _canonical_monomial(creators, annihilators):
    """Return a canonical fermion monomial and the sorting phase."""
    if len(set(creators)) != len(creators) or len(set(annihilators)) != len(annihilators):
        return None, 0
    sign = (-1)**(sum(creators[i] > creators[j] for i in range(len(creators)) for j in range(i+1, len(creators)))
                  + sum(annihilators[i] > annihilators[j] for i in range(len(annihilators)) for j in range(i+1, len(annihilators))))
    return (tuple(sorted(creators)), tuple(sorted(annihilators))), sign


@dataclass
class _IPEAGroup:
    sector: str
    rows: np.ndarray
    q: np.ndarray
    d0: np.ndarray
    unit_shift: np.ndarray
    orth_slice: slice


class _MolcasIPEA:
    """Molcas scalar IPEA represented in the native spinor IC space.

    Only this definition of IPEA knows about alpha/beta partners. Hamiltonian,
    source, metric and all RDM contractions remain the native spinor ones.
    The partner map is explicit because Molcas IPEA itself is a spin-free,
    primitive-coordinate prescription and has no unique SOC extension.
    """

    def __init__(self, model, *, norm_threshold=1e-10, metric_threshold=1e-8):
        self.model = model
        dims = model.dimensions
        if any(dims[s] % 2 for s in 'IAE'):
            raise ValueError('Molcas scalar IPEA requires two spinors per spatial I/A/E orbital')
        self.sdims = {s: dims[s]//2 for s in 'IAE'}
        self.pairs = {s: np.arange(dims[s]).reshape(-1, 2) for s in 'IAE'}
        self.norm_threshold = float(norm_threshold)
        self.metric_threshold = float(metric_threshold)
        self.groups = []
        self.scalar_limit_diagnostics = self._validate_scalar_limit()
        self._build()

    def _validate_scalar_limit(self):
        """Prove that adjacent spinors form a scalar alpha/beta Hamiltonian.

        Molcas IPEA is defined in spin-free primitive IC coordinates.  Merely
        having even I/A/E dimensions is therefore insufficient: accepting an
        arbitrary SOC Hamiltonian here would silently invent a partner map.
        This check concerns only the IPEA coordinate definition; the CASPT2
        Hamiltonian, source, metric, and RDM contractions remain spinorial.
        """
        nspin = len(self.model.data.h)
        nspatial = nspin//2
        errors = {}
        scales = {}
        for name, array in (('h1e', self.model.data.h),
                            ('fock', self.model.fock_mo)):
            paired = np.asarray(array).reshape(nspatial, 2, nspatial, 2)
            scale = max(1., native._maxabs(paired))
            error = max(native._maxabs(paired[:, 0, :, 0]-paired[:, 1, :, 1]),
                        native._maxabs(paired[:, 0, :, 1]),
                        native._maxabs(paired[:, 1, :, 0]))
            errors[name] = error
            scales[name] = scale

        eri = self.model.data.w.transpose(0, 2, 1, 3)
        paired = eri.reshape((nspatial, 2)*4)
        reference = paired[:, 0, :, 0, :, 0, :, 0]
        scale = max(1., native._maxabs(reference))
        error = 0.
        for spins in itertools.product(range(2), repeat=4):
            block = paired[:, spins[0], :, spins[1], :, spins[2], :, spins[3]]
            expected = reference if spins[0] == spins[1] and spins[2] == spins[3] else 0.
            error = max(error, native._maxabs(block-expected))
        errors['eri'] = error
        scales['eri'] = scale
        failed = {name: value for name, value in errors.items()
                  if value > self.model.atol+self.model.rtol*scales[name]}
        if failed:
            detail = ', '.join(f'{name}={value:.3e}' for name, value in failed.items())
            raise ValueError(
                'ipea_definition="molcas" is restricted to the scalar limit '
                'with interleaved alpha/beta spinors; spin-pair validation failed: '
                + detail)
        return dict(molcas_scalar_pair_order='interleaved_alpha_beta',
                    molcas_scalar_h1e_pair_error=float(errors['h1e']),
                    molcas_scalar_fock_pair_error=float(errors['fock']),
                    molcas_scalar_eri_pair_error=float(errors['eri']))

    @staticmethod
    def _space(label):
        return 'I' if label in 'ij' else 'E' if label in 'rs' else 'A'

    def _orbital(self, label, spatial, spin):
        space = self._space(label)
        return (space, int(self.pairs[space][spatial, spin]))

    def _native_dictionary(self, sector, parent):
        """Map canonical monomials to (selected raw-block row, phase)."""
        sec = self.model.sectors[sector]
        selected = [k for k, ext in enumerate(sec.external)
                    if tuple(int(x)//2 for x in ext) == tuple(parent)]
        result = {}
        local_offset = 0
        for family_id in sec.families:
            family = native._FAMILIES[family_id]
            nlocal = len(self.model.coords[family_id])//len(sec.external)
            for block in selected:
                rows = self.model.coords[family_id].reshape(len(sec.external), nlocal, -1)[block]
                for local, coord in enumerate(rows):
                    values = {}
                    for label, value, space in zip(family.variables, coord, family.spaces):
                        values[label] = (space, int(value))
                    ops = [(kind, values[label]) for kind, label in re.findall(r'([CD])\[([a-z])\]', family.expression)]
                    creators = [orb for kind, orb in ops if kind == 'C']
                    annihilators = [orb for kind, orb in ops if kind == 'D']
                    key, phase = _canonical_monomial(creators, annihilators)
                    position = (selected.index(block), local_offset+local)
                    if key in result:
                        raise RuntimeError('duplicate native primitive spinor monomial')
                    result[key] = (position, phase)
            local_offset += nlocal
        return selected, result

    def _expand_scalar(self, family, coordinate):
        values = dict(zip(family.variables, coordinate))
        expansion = {}
        for coefficient, first, second in family.terms:
            p, q = first
            r, s = second
            for sigma, tau in itertools.product(range(2), repeat=2):
                po = self._orbital(p, values[p], sigma)
                qo = self._orbital(q, values[q], sigma)
                ro = self._orbital(r, values[r], tau)
                so = self._orbital(s, values[s], tau)
                if qo == ro:
                    key, phase = _canonical_monomial([po], [so])
                    expansion[key] = expansion.get(key, 0.) + coefficient*phase
                key, phase = _canonical_monomial([po, ro], [qo, so])
                if phase:
                    expansion[key] = expansion.get(key, 0.) - coefficient*phase
        return {key: value for key, value in expansion.items() if value}

    def _coordinates(self, family):
        dims = [self.sdims[self._space(label)] for label in family.variables]
        return [coord for coord in itertools.product(*(range(n) for n in dims))
                if all((coord[p] < coord[q]) if strict else (coord[p] <= coord[q])
                       for p, q, strict in family.restrictions)]

    def _build_group(self, sector, parent, columns, occupations):
        sec = self.model.sectors[sector]
        selected, dictionary = self._native_dictionary(sector, parent)
        nlocal, ncol = sec.indices.shape[1], len(columns)
        c = np.zeros((len(selected), nlocal, ncol), complex)
        weights = np.zeros(ncol)
        for j, (family, coord) in enumerate(columns):
            for key, value in self._expand_scalar(family, coord).items():
                if key not in dictionary:
                    raise RuntimeError(f'Molcas generator term is absent from native {sector} basis')
                (block, local), phase = dictionary[key]
                c[block, local, j] += value/phase
            active = coord[len(sector):]
            weights[j] = sum(occupations[a]/2 if role == 'C' else (2-occupations[a])/2
                             for a, role in zip(active, family.active_roles))
        sc = np.einsum('ab,xbc->xac', sec.overlap, c, optimize=True)
        metric = np.einsum('xai,xaj->ij', c.conj(), sc, optimize=True)
        metric = (metric+metric.conj().T)*.5
        norms = np.diag(metric).real
        valid = norms > self.norm_threshold
        if not np.any(valid):
            return None
        # OpenMolcas-compatible deterministic primitive ordering.  The tiny
        # nonuniform scale is part of its historical DORT construction: in an
        # overcomplete IC basis IPEA is not invariant to changing this choice.
        family_order = {family: k for k, family in enumerate(
            [f for f in _MOLCAS_FAMILIES if f.sector == sector])}
        within = {family: 0 for family in family_order}
        row_index = np.empty(ncol, dtype=int)
        family_size = {family: sum(f is family for f, _ in columns)
                       for family in family_order}
        for j, (family, coord) in enumerate(columns):
            active = coord[len(sector):]
            if sector in ('r', 'i'):
                a, b, c = active
                index = b+self.sdims['A']*a+self.sdims['A']**2*c+1
            elif sector == 'ir':
                a, b = active
                index = a+self.sdims['A']*b+1
            else:
                index = within[family]+1
            row_index[j] = index+family_order[family]*family_size[family]
            within[family] += 1
        scale = (1+3e-6*row_index[valid])/np.sqrt(norms[valid])
        normalized = metric[np.ix_(valid, valid)]*scale[:, None]*scale[None, :]
        eigenvalues, eigenvectors = linalg.eigh((normalized+normalized.conj().T)*.5)
        keep = eigenvalues > self.metric_threshold
        y = np.zeros((ncol, int(np.count_nonzero(keep))), complex)
        y[valid] = scale[:, None]*eigenvectors[:, keep]/np.sqrt(eigenvalues[keep])[None, :]
        yld = y.astype(np.clongdouble)
        mld = metric.astype(np.clongdouble)
        gram = np.asarray(yld.conj().T@mld@yld, complex)
        gram_error = native._maxabs(gram-np.eye(y.shape[1]))
        ge, gu = linalg.eigh((gram+gram.conj().T)*.5)
        if ge[0] <= .5 or ge[-1] >= 1.5 or gram_error > 1e-6:
            raise native.CASPT2NumericalError(
                f'Molcas IPEA coordinate orthogonalization failed: {gram_error:.3e}')
        correction = (gu/np.sqrt(ge))@gu.conj().T
        y = np.asarray(yld@correction.astype(np.clongdouble), complex)
        yld = y.astype(np.clongdouble)
        gram_post = native._maxabs(np.asarray(yld.conj().T@mld@yld, complex)
                                   -np.eye(y.shape[1]))
        if gram_post > 2e-8:
            raise native.CASPT2NumericalError(
                f'Molcas IPEA coordinate reorthogonalization failed: {gram_post:.3e}')
        q = np.einsum('ar,xaj,jk->xrk', sec.transform.conj(), sc, y, optimize=True)
        q = q.reshape(len(selected)*sec.transform.shape[1], -1)
        qgram = q.conj().T@q
        qerror = native._maxabs(qgram-np.eye(q.shape[1]))
        qe, qu = linalg.eigh((qgram+qgram.conj().T)*.5)
        if qe[0] <= .5 or qe[-1] >= 1.5 or qerror > 1e-6:
            raise native.CASPT2NumericalError(
                'Molcas scalar space is not contained in retained native '
                f'metric space: sector={sector}, parent={parent}, error={qerror:.3e}')
        qcorrection = (qu/np.sqrt(qe))@qu.conj().T
        q = q@qcorrection
        qerror_post = native._maxabs(q.conj().T@q-np.eye(q.shape[1]))
        if qerror_post > 2e-10:
            raise native.CASPT2NumericalError(
                'Molcas scalar/native coordinate reorthogonalization failed: '
                f'sector={sector}, parent={parent}, error={qerror_post:.3e}')
        primitive_shift = weights*norms
        unit_shift = y.conj().T@(primitive_shift[:, None]*y)
        unit_shift = qcorrection.conj().T@unit_shift@qcorrection
        native_rows = np.concatenate([
            np.arange(sec.orth_slice.start+block*sec.transform.shape[1],
                      sec.orth_slice.start+(block+1)*sec.transform.shape[1])
            for block in selected])
        dnative = self.model.denominators[native_rows]
        d0 = q.conj().T@(dnative[:, None]*q)
        return (native_rows, q, (d0+d0.conj().T)*.5,
                (unit_shift+unit_shift.conj().T)*.5,
                dict(raw_dimension=ncol, rank=q.shape[1],
                     minimum_norm=float(np.min(norms[valid])),
                     minimum_metric_eigenvalue=float(np.min(eigenvalues[keep], initial=np.inf)),
                     orthonormalization_error_before_refinement=float(gram_error),
                     orthonormalization_error=float(gram_post),
                     native_containment_error_before_refinement=float(qerror),
                     native_containment_error=float(qerror_post)))

    def _build(self):
        occupations = np.array([np.trace(self.model.data.pdms[1][np.ix_(pair, pair)]).real
                                for pair in self.pairs['A']])
        if np.min(occupations, initial=0) < -1e-9 or np.max(occupations, initial=0) > 2+1e-9:
            raise native.CASPT2NumericalError('Molcas IPEA spatial occupations are outside [0,2]')
        grouped = {}
        raw = 0
        for family in _MOLCAS_FAMILIES:
            for coord in self._coordinates(family):
                key = (family.sector, tuple(coord[:len(family.sector)]))
                grouped.setdefault(key, []).append((family, coord))
                raw += 1
        position, rank, details = 0, 0, {}
        source_outside_sq = 0.
        for (sector, parent), columns in grouped.items():
            built = self._build_group(sector, parent, columns, occupations)
            if built is None:
                continue
            rows, q, d0, unit, detail = built
            sl = slice(position, position+q.shape[1])
            position = sl.stop
            self.groups.append(_IPEAGroup(sector, rows, q, d0, unit, sl))
            rank += q.shape[1]
            details.setdefault(sector, dict(raw_dimension=0, rank=0, groups=0))
            details[sector]['raw_dimension'] += detail['raw_dimension']
            details[sector]['rank'] += detail['rank']
            details[sector]['groups'] += 1
            details[sector]['maximum_orthonormalization_error'] = max(
                details[sector].get('maximum_orthonormalization_error', 0.),
                detail['orthonormalization_error'])
            details[sector]['maximum_orthonormalization_error_before_refinement'] = max(
                details[sector].get('maximum_orthonormalization_error_before_refinement', 0.),
                detail['orthonormalization_error_before_refinement'])
            details[sector]['maximum_native_containment_error'] = max(
                details[sector].get('maximum_native_containment_error', 0.),
                detail['native_containment_error'])
            details[sector]['maximum_native_containment_error_before_refinement'] = max(
                details[sector].get('maximum_native_containment_error_before_refinement', 0.),
                detail['native_containment_error_before_refinement'])
            b = self.model.b[rows]
            source_outside_sq += linalg.norm(b-q@(q.conj().T@b))**2
        self.rank = rank
        self.source = self.project(self.model.b)
        self.diagnostics = dict(**self.scalar_limit_diagnostics,
                                ipea_definition='molcas_spinfree_transformed_to_native_spinor',
                                molcas_scalar_raw_dimension=raw,
                                molcas_scalar_metric_rank=rank,
                                molcas_scalar_subspaces=details,
                                molcas_scalar_source_outside_norm=float(np.sqrt(source_outside_sq)),
                                ipea_spin_pairs_used=True,
                                ipea_only_spin_reduced_component=True,
                                norm_threshold=self.norm_threshold,
                                metric_threshold=self.metric_threshold,
                                active_spatial_occupations=occupations.tolist())

    def lift(self, vector):
        result = np.zeros(self.model.rank, complex)
        for group in self.groups:
            result[group.rows] += group.q@vector[group.orth_slice]
        return result

    def project(self, vector):
        result = np.zeros(self.rank, complex)
        for group in self.groups:
            result[group.orth_slice] = group.q.conj().T@vector[group.rows]
        return result

    def diagonal_model(self, shift):
        values, vectors, slices = [], [], []
        position = 0
        for group in self.groups:
            d, u = linalg.eigh(group.d0+shift*group.unit_shift)
            values.append(d)
            vectors.append(u)
            slices.append(slice(position, position+len(d)))
            position += len(d)
        return np.concatenate(values), vectors, slices

    def to_denominator_basis(self, vector, vectors):
        out = np.empty_like(vector)
        for group, u in zip(self.groups, vectors):
            out[group.orth_slice] = u.conj().T@vector[group.orth_slice]
        return out

    def from_denominator_basis(self, vector, vectors):
        out = np.empty_like(vector)
        for group, u in zip(self.groups, vectors):
            out[group.orth_slice] = u@vector[group.orth_slice]
        return out

    def shift_action(self, vector, shift):
        out = np.zeros_like(vector)
        for group in self.groups:
            out[group.orth_slice] = shift*group.unit_shift@vector[group.orth_slice]
        return out


class NativeCASPT2:
    """Reusable native-spinor contraction/metric workspace.

    The current iterative solve requires semicanonical inactive and external
    Fock blocks. Active Fock and every inter-class coupling remain complete.
    Raw S/F actions themselves support noncanonical and complex inputs.
    """

    def __init__(self, h1e, eri, ncore, ncas, nelecas, pdms, *, fock_mo=None,
                 constant=0., reference_energy=None, wick_backend='block2',
                 contraction_backend='numpy', metric_atol=1e-12,
                 metric_rcond=1e-10, atol=1e-10, rtol=1e-9,
                 check_rdms=True, max_memory_mb=2048., build_metric=True):
        started = time.perf_counter()
        self.atol, self.rtol = atol, rtol
        self.wick_backend = wick_backend
        if contraction_backend == 'numpy':
            self.einsum = np.einsum
        elif contraction_backend == 'pytblis':
            from pytblis import einsum
            self.einsum = einsum
        else:
            raise ValueError('contraction_backend must be numpy or pytblis')
        h, eri, pdms, nc, na, ne, diag = native._validate_input(
            h1e, eri, ncore, ncas, nelecas, pdms, atol, rtol, check_rdms)
        f, gamma = native.build_generalized_fock(h, eri, pdms[0], nc)
        if fock_mo is not None:
            f = native._finite(fock_mo, 'fock_mo')
        if f.shape != h.shape:
            raise ValueError('fock_mo has wrong shape')
        native._hermitian(f, 'fock_mo', atol, rtol)
        self.fock_mo = np.asarray(f, dtype=complex)
        ef = complex(np.einsum('pq,pq->', f, gamma))
        eref = complex(native._electronic_reference_energy(h, eri, nc, na, pdms)) + complex(constant)
        for name, value in (('Fock reference', ef), ('reference', eref)):
            if not np.isfinite(value) or abs(value.imag) > atol + rtol*abs(value.real):
                raise native.CASPT2NumericalError(f'{name} energy is not finite and real')
        self.ef, self.eref = ef.real, eref.real
        if reference_energy is not None:
            error = abs(complex(reference_energy) - eref)
            if error > 10*atol:
                raise native.CASPT2NumericalError(f'RDM reference energy differs by {error:.3e} Eh')
            diag['reference_energy_error'] = float(error)
        self.data = native._Data(h, eri, self.fock_mo, pdms, nc, na)
        self.dimensions = self.data.dimensions
        self.coords = tuple(native._coordinates(fam, self.dimensions) for fam in native._FAMILIES)
        sizes = [len(c) for c in self.coords]
        offsets = np.cumsum([0] + sizes)
        self.slices = tuple(slice(int(a), int(b)) for a, b in zip(offsets[:-1], offsets[1:]))
        self.size = int(offsets[-1])
        self.shapes = tuple(tuple(self.dimensions[s] for s in fam.spaces) for fam in native._FAMILIES)
        self._terms = {}
        self._constants = {}
        self._cache_bytes = 0
        self._cache_limit = int(native._positive_float(max_memory_mb, 'max_memory_mb', allow_zero=False)*2**20*.4)
        self._molcas_ipea = None
        # This is a workspace gate, not a claim to bound the user's input RDMs.
        largest_active = max((math.prod(na for _ in fam.active) for fam in native._FAMILIES), default=0)
        minimum_bytes = 16*(4*largest_active**2 + 32*self.size)
        if minimum_bytes > max_memory_mb*2**20:
            raise MemoryError(f'Native active-block workspace requires at least {minimum_bytes/2**20:.1f} MiB')
        self.diagnostics = dict(diag, raw_dimension=self.size, ncore=nc, ncas=na,
                                nvirt=self.dimensions['E'], nelecas=ne,
                                wick_backend=wick_backend, contraction_backend=contraction_backend,
                                solver_backend='native_spinor_matrix_free', maximum_rdm_rank=4,
                                spin_trace_used=False, fock_source='explicit_mo' if fock_mo is not None else 'native_spinor')
        self.rhs = self.source()
        self.sectors = {}
        if build_metric:
            self._build_metric(metric_atol, metric_rcond)
        self.diagnostics['matrix_seconds'] = time.perf_counter()-started
        self.diagnostics['constant_cache_bytes'] = self._cache_bytes

    def _prepare(self, kernel, fixed=None):
        """Lower delta constraints and cache connected Fock/RDM contractions.

        Summing the two active Fock indices of dm4 once creates at most a
        rank-six active tensor. A disconnected Fock/RDM outer product is
        never materialized. No global IC matrix is formed by this lowering.
        """
        fixed = {} if fixed is None else fixed
        output = tuple(x for x in kernel.rows if x not in fixed)
        columns = tuple(x for x in kernel.columns if x not in fixed)
        free = set(output+columns)
        spaces = dict(kernel.spaces)
        prepared = []
        for term in kernel.terms:
            coefficient = term.coefficient*math.prod(self.dimensions[s] for s in term.summed_dimensions)
            operands, labels = [], []
            for tensor in term.tensors:
                array = self.data.get(tensor)
                ix = tuple(fixed.get(x, slice(None)) for x in tensor.indices)
                array = array[ix] if ix else array
                lab = ''.join(x for x in tensor.indices if x not in fixed)
                if any(n == 0 for n in array.shape):
                    coefficient = 0
                    break
                operands.append(array)
                labels.append(lab)
            if not coefficient:
                continue
            # Only fuse connected operands when an actual summation occurs.
            all_indices = set(''.join(labels))
            retained = ''.join(sorted(all_indices & free))
            reduced = all_indices-free
            out_size = math.prod(self.dimensions[spaces[x]] for x in retained)
            if reduced and len(retained) <= 6 and 16*out_size <= self._cache_limit-self._cache_bytes:
                expr = ','.join(labels)+'->'+retained
                # Object identity, strides and normalized expression distinguish
                # views without hashing/copying a multi-GB RDM.
                alphabet = {x: string.ascii_letters[i] for i, x in enumerate(dict.fromkeys(''.join(labels)))}
                canonical = ''.join(alphabet.get(x, x) for x in expr)
                key = (canonical, tuple((a.__array_interface__['data'][0], a.shape, a.strides) for a in operands))
                if key not in self._constants:
                    value = np.asarray(self.einsum(expr, *operands, optimize=True), dtype=complex)
                    self._constants[key] = value
                    self._cache_bytes += value.nbytes
                operands, labels = [self._constants[key]], [retained]
            for x, y in term.equalities:
                if x in fixed and y in fixed:
                    coefficient *= fixed[x] == fixed[y]
                elif x in fixed or y in fixed:
                    label, index = (y, fixed[x]) if x in fixed else (x, fixed[y])
                    operands.append(np.eye(self.dimensions[spaces[label]], dtype=complex)[index])
                    labels.append(label)
                else:
                    operands.append(np.eye(self.dimensions[spaces[x]], dtype=complex))
                    labels.append(x+y)
            present = set(''.join(labels)+''.join(columns))
            for x in output:
                if x not in present:
                    operands.append(np.ones(self.dimensions[spaces[x]], dtype=complex))
                    labels.append(x)
            if coefficient:
                prepared.append((coefficient, operands, labels))
        return output, columns, prepared

    def _contract(self, plan, amplitude=None, *, matrix=False):
        rows, columns, terms = plan
        output = ''.join(rows + (columns if matrix else ()))
        result = None
        for coefficient, arrays, labels in terms:
            if amplitude is not None:
                arrays, labels = arrays+[amplitude], labels+[''.join(columns)]
            expr = ','.join(labels)+'->'+output
            value = coefficient*self.einsum(expr, *arrays, optimize=True)
            if result is None:
                result = np.array(value, dtype=complex)
            else:
                result += value
        return result

    def source(self):
        result = np.zeros(self.size, complex)
        for i, coordinates in enumerate(self.coords):
            if not len(coordinates):
                continue
            plan = self._prepare(native._compile_kernel('b', i, None, self.wick_backend))
            value = self._contract(plan)
            if value is not None:
                result[self.slices[i]] = value[tuple(coordinates.T)]
        return result

    def action(self, kind, vector, *, class_diagonal=False, offdiagonal=False):
        """Apply raw S or F in precisely the dense oracle's ordered basis."""
        if kind not in ('S', 'F'):
            raise ValueError('kind must be S or F')
        vector = np.asarray(vector)
        if vector.shape != (self.size,):
            raise ValueError('wrong native amplitude dimension')
        tensors = []
        for c, sl, shape in zip(self.coords, self.slices, self.shapes):
            t = np.zeros(shape, complex)
            if len(c):
                t[tuple(c.T)] = vector[sl]
            tensors.append(t)
        result = np.zeros(self.size, complex)
        for i, bra in enumerate(native._FAMILIES):
            if not len(self.coords[i]):
                continue
            for j, ket in enumerate(native._FAMILIES):
                if (not len(self.coords[j]) or (class_diagonal and bra.sector != ket.sector)
                        or (offdiagonal and bra.sector == ket.sector)):
                    continue
                key = (kind, i, j)
                if key not in self._terms:
                    self._terms[key] = self._prepare(native._compile_kernel(kind, i, j, self.wick_backend))
                value = self._contract(self._terms[key], tensors[j])
                if value is not None:
                    result[self.slices[i]] += value[tuple(self.coords[i].T)]
        return result

    def _local_matrix(self, kind, ids, proto, active_coords):
        sizes = [len(c) for c in active_coords]
        offsets = np.cumsum([0]+sizes)
        result = np.zeros((offsets[-1], offsets[-1]), complex)
        for ii, i in enumerate(ids):
            for jj, j in enumerate(ids):
                kernel = native._compile_kernel(kind, i, j, self.wick_backend)
                fixed = dict(zip(kernel.rows, proto))
                fixed.update(zip(kernel.columns, proto))
                value = self._contract(self._prepare(kernel, fixed), matrix=True)
                if value is None:
                    continue
                ri, ci = active_coords[ii], active_coords[jj]
                index = tuple(ri[:, k, None] for k in range(ri.shape[1]))
                index += tuple(ci[None, :, k] for k in range(ci.shape[1]))
                result[offsets[ii]:offsets[ii+1], offsets[jj]:offsets[jj+1]] = value[index] if index else value
        native._hermitian(result, 'local '+kind, self.atol, self.rtol)
        return (result+result.conj().T)*.5

    def _build_metric(self, metric_atol, metric_rcond):
        parts, largest = {}, 0.
        orbital_e = np.diag(self.fock_mo).real
        energies = {s: orbital_e[sl] for s, sl in self.data.slices.items()}
        for name in native.SUBSPACE_ORDER:
            ids = tuple(i for i, fam in enumerate(native._FAMILIES) if fam.sector == name and len(self.coords[i]))
            if not ids:
                continue
            ne = len(name)
            ext = np.unique(self.coords[ids[0]][:, :ne], axis=0)
            active = []
            indices = []
            for i in ids:
                count = len(self.coords[i])//len(ext)
                active.append(self.coords[i][:count, ne:])
                indices.append(np.arange(self.slices[i].start, self.slices[i].stop).reshape(len(ext), count))
            indices = np.concatenate(indices, axis=1)
            overlap = self._local_matrix('S', ids, ext[0], active)
            fock = self._local_matrix('F', ids, ext[0], active)
            gap = sum((-1 if label in 'ij' else 1)*energies['I' if label in 'ij' else 'E'][ext[:, k]]
                      for k, label in enumerate(name))
            active_fock = fock-(self.ef+gap[0])*overlap
            e, u = linalg.eigh(overlap)
            largest = max(largest, float(np.max(e, initial=0)))
            self.sectors[name] = _Sector(name, ids, indices, ext, overlap, active_fock, gap)
            parts[name] = (e, u)
        cutoff = max(metric_atol, metric_rcond*largest)
        pos, null_sq, details = 0, 0., {}
        for name, sector in self.sectors.items():
            e, u = parts[name]
            if np.min(e, initial=0) < -(self.atol+self.rtol*max(1., largest)):
                raise native.CASPT2NumericalError('negative native local metric eigenvalue')
            keep = e > cutoff
            x = u[:, keep]/np.sqrt(e[keep])[None, :]
            # Re-whiten the actual Gram, preserving the selected rank.
            if x.shape[1]:
                xl, sl = x.astype(np.clongdouble), sector.overlap.astype(np.clongdouble)
                gram = np.asarray(xl.conj().T@sl@xl, complex)
                ge, gu = linalg.eigh((gram+gram.conj().T)*.5)
                if ge[0] <= .5 or ge[-1] >= 1.5:
                    raise native.CASPT2NumericalError('ill-conditioned retained native metric')
                x = x@((gu/np.sqrt(ge))@gu.conj().T)
                xl = x.astype(np.clongdouble)
                a = np.asarray(xl.conj().T@sector.active_fock.astype(np.clongdouble)@xl, complex)
                native._hermitian(a, 'local orthogonal Fock', self.atol*10, self.rtol*10)
                d, v = linalg.eigh((a+a.conj().T)*.5)
                x = x@v
            else:
                d = np.empty(0)
            sector.transform = x
            sector.denominators = sector.gap[:, None]+d[None, :]
            count = sector.denominators.size
            sector.orth_slice = slice(pos, pos+count)
            pos += count
            null_sq += linalg.norm(self.rhs[sector.indices]@u[:, ~keep].conj())**2
            details[name] = dict(local_raw_dimension=len(e), local_rank=int(sum(keep)),
                                 external_blocks=len(sector.external))
        self.rank = pos
        self.denominators = np.concatenate([s.denominators.ravel() for s in self.sectors.values()]) if pos else np.empty(0)
        self.b = self.project(self.rhs)
        self.diagnostics.update(metric_rank=pos, metric_cutoff=cutoff,
                                metric_null_source_norm=float(np.sqrt(null_sq)), local_metrics=details,
                                largest_local_metric_dimension=max((len(s.overlap) for s in self.sectors.values()), default=0))

    def lift(self, vector):
        result = np.zeros(self.size, complex)
        for s in self.sectors.values():
            t = np.asarray(vector[s.orth_slice]).reshape(s.denominators.shape)
            result[s.indices] = t@s.transform.T
        return result

    def project(self, vector):
        return np.concatenate([(vector[s.indices]@s.transform.conj()).ravel() for s in self.sectors.values()]) if self.sectors else np.empty(0, complex)

    def solve(self, *, approximation='full', ipea_shift=0., regularizer='none',
              level_shift=0., regularization_basis='caspt2d', energy_functional='model',
              maxiter=100, conv_tol=1e-10, denominator_atol=1e-10, source_atol=1e-10,
              ipea_definition='spinor_diagonal', ipea_basis='input',
              scalar_norm_threshold=1e-10,
              scalar_metric_threshold=1e-8):
        if ipea_shift != 0 and ipea_definition == 'molcas':
            if ipea_basis != 'input':
                raise ValueError(
                    "Molcas IPEA uses the supplied semicanonical scalar orbital gauge; "
                    "set ipea_basis='input'")
            return self._solve_molcas_ipea(
                approximation=approximation, ipea_shift=ipea_shift,
                regularizer=regularizer, level_shift=level_shift,
                regularization_basis=regularization_basis,
                energy_functional=energy_functional, maxiter=maxiter,
                conv_tol=conv_tol, denominator_atol=denominator_atol,
                source_atol=source_atol, norm_threshold=scalar_norm_threshold,
                metric_threshold=scalar_metric_threshold)
        if ipea_shift != 0:
            raise NotImplementedError(
                "matrix-free nonzero IPEA requires ipea_definition='molcas'; "
                "the old primitive spinor diagonal prescription is not Molcas-compatible")
        if approximation not in ('full', 'caspt2d'):
            raise NotImplementedError('Native matrix-free currently supports full and caspt2d')
        if regularization_basis != 'caspt2d':
            raise NotImplementedError('Full-spectrum filtering requires the dense oracle')
        if energy_functional not in ('model', 'full'):
            raise ValueError('energy_functional must be model or full')
        regularizer, level_shift, _ = native._regularization_options(regularizer, level_shift, regularization_basis)
        if not self.sectors and self.size:
            raise ValueError('metric was not built')
        for sp in ('I', 'E'):
            sl = self.data.slices[sp]
            block = self.fock_mo[sl, sl]
            if native._maxabs(block-np.diag(np.diag(block))) > self.atol:
                raise ValueError('Matrix-free solve requires semicanonical inactive/external blocks')
        if self.diagnostics.get('metric_null_source_norm', 0.) > source_atol+self.rtol*linalg.norm(self.rhs):
            raise native.CASPT2NumericalError('source outside retained native metric span')
        b, d = self.b, self.denominators
        def full_action(t):
            # Class-diagonal H0 has already been evaluated exactly with dm4.
            # Reusing its local eigensystem avoids cancellation of EF*S after
            # ill-conditioned metric projection and repeated rank-four work.
            return d*t+self.project(self.action('F', self.lift(t), offdiagonal=True))
        unshifted = not level_shift or regularizer == 'none'
        if unshifted:
            # A zero diagonal-model pole is NOT necessarily a full-H0 pole.
            f = np.ones_like(d)
            np.divide(1., d, out=f, where=np.abs(d)>denominator_atol)
        elif regularizer == 'real':
            if np.any(np.abs(d+level_shift) <= denominator_atol):
                raise native.CASPT2IntruderError('unresolved real-shift denominator')
            f = 1/(d+level_shift)
        elif regularizer == 'imaginary':
            f = d/(d*d+level_shift**2)
        else:
            p = 1 if regularizer == 'sigma1' else 2
            if p == 1 and np.any(np.abs(d) <= denominator_atol):
                raise native.CASPT2IntruderError('sigma1 at zero denominator has no unique signed limit')
            f = np.zeros_like(d)
            np.divide(-np.expm1(-(np.abs(d)/level_shift)**p), d, out=f, where=d != 0)
        iterations = []
        if approximation == 'caspt2d':
            if unshifted and np.any((np.abs(d)<=denominator_atol) & (np.abs(b)>source_atol)):
                raise native.CASPT2IntruderError('unresolved class-diagonal denominator')
            t = -f*b
            if unshifted:
                t[np.abs(d)<=denominator_atol] = 0
            residual = np.zeros_like(t)
        elif len(b):
            if unshifted:
                matvec = lambda t: f*full_action(t)
            else:
                matvec = lambda t: t+f*(full_action(t)-d*t)
            op = LinearOperator((len(b), len(b)), matvec=matvec, dtype=complex)
            t, info = gmres(op, -f*b, rtol=conv_tol, atol=conv_tol*.01,
                           restart=min(50, len(b)), maxiter=maxiter,
                           callback=lambda x: iterations.append(float(x)), callback_type='pr_norm')
            residual = full_action(t)+b if unshifted else matvec(t)+f*b
            if info or linalg.norm(residual) > max(10*conv_tol, 10*conv_tol*linalg.norm(b)):
                raise native.CASPT2NumericalError(f'native GMRES failed: info={info}, residual={linalg.norm(residual):.3e}')
        else:
            t, residual = b.copy(), b.copy()
        af = full_action(t) if len(t) else t
        am = af if approximation == 'full' else d*t
        selected = am if energy_functional == 'model' else af
        projected = float(np.vdot(b, t).real)
        emodel = float(np.vdot(t, am).real+2*projected)
        efull = float(np.vdot(t, af).real+2*projected)
        energy = emodel if energy_functional == 'model' else efull
        sub = {name: float(np.vdot(t[s.orth_slice], selected[s.orth_slice]+2*b[s.orth_slice]).real)
               for name, s in self.sectors.items()}
        subp = {name: float(np.vdot(b[s.orth_slice], t[s.orth_slice]).real) for name, s in self.sectors.items()}
        diag = dict(self.diagnostics, residual_norm=float(linalg.norm(residual)), iterations=len(iterations),
                    energy_definition=energy_functional+'_Hylleraas_including_IPEA_shift_corrected',
                    constant_cache_bytes=self._cache_bytes)
        return native.CASPT2Result(self.eref, self.ef, energy, energy, self.lift(t),
                                  1/(1+float(np.vdot(t, t).real)), sub, diag,
                                  e_projected=projected, e_shift_correction=emodel-projected,
                                  regularizer=regularizer, level_shift=level_shift,
                                  approximation=approximation, sub_projected_eners=subp,
                                  e_constraint_correction=energy-emodel,
                                  e_full_hylleraas=efull, e_model_hylleraas=emodel,
                                  e_no_ipea_hylleraas=energy, energy_functional=energy_functional)

    def _solve_molcas_ipea(self, *, approximation, ipea_shift, regularizer,
                           level_shift, regularization_basis, energy_functional,
                           maxiter, conv_tol, denominator_atol, source_atol,
                           norm_threshold, metric_threshold):
        if approximation not in ('full', 'caspt2d'):
            raise NotImplementedError('Molcas IPEA currently supports full and caspt2d')
        if regularization_basis != 'caspt2d':
            raise NotImplementedError('Molcas IPEA spectral filtering is not implemented')
        if energy_functional not in ('model', 'full'):
            raise ValueError('energy_functional must be model or full')
        regularizer, epsilon, _ = native._regularization_options(
            regularizer, level_shift, regularization_basis)
        key = (float(norm_threshold), float(metric_threshold))
        if self._molcas_ipea is None or self._molcas_ipea[0] != key:
            self._molcas_ipea = (key, _MolcasIPEA(
                self, norm_threshold=norm_threshold,
                metric_threshold=metric_threshold))
        coordinates = self._molcas_ipea[1]
        outside = coordinates.diagnostics['molcas_scalar_source_outside_norm']
        if outside > source_atol+self.rtol*linalg.norm(self.b):
            raise native.CASPT2NumericalError(
                f'native source is outside the Molcas scalar IC component: {outside:.3e}')
        b = coordinates.source
        denominators, rotations, _ = coordinates.diagonal_model(ipea_shift)

        def unshifted_action(z):
            t = coordinates.lift(z)
            raw = self.denominators*t+self.project(
                self.action('F', self.lift(t), offdiagonal=True))
            return coordinates.project(raw)

        def shifted_action(z):
            return unshifted_action(z)+coordinates.shift_action(z, ipea_shift)

        def to_d(z):
            return coordinates.to_denominator_basis(z, rotations)

        def from_d(z):
            return coordinates.from_denominator_basis(z, rotations)

        d, bd = denominators, to_d(b)
        unregularized = epsilon == 0 or regularizer == 'none'
        if unregularized:
            filt = np.ones_like(d)
            np.divide(1., d, out=filt, where=np.abs(d) > denominator_atol)
        elif regularizer == 'real':
            if np.any(np.abs(d+epsilon) <= denominator_atol):
                raise native.CASPT2IntruderError('unresolved real-shift denominator')
            filt = 1/(d+epsilon)
        elif regularizer == 'imaginary':
            filt = d/(d*d+epsilon*epsilon)
        else:
            power = 1 if regularizer == 'sigma1' else 2
            if power == 1 and np.any(np.abs(d) <= denominator_atol):
                raise native.CASPT2IntruderError(
                    'sigma1 at zero denominator has no unique signed limit')
            filt = np.zeros_like(d)
            np.divide(-np.expm1(-(np.abs(d)/epsilon)**power), d,
                      out=filt, where=d != 0)
        iterations = []
        if approximation == 'caspt2d':
            if unregularized and np.any((np.abs(d) <= denominator_atol)
                                        & (np.abs(bd) > source_atol)):
                raise native.CASPT2IntruderError('unresolved class-diagonal denominator')
            td = -filt*bd
            if unregularized:
                td[np.abs(d) <= denominator_atol] = 0
            z = from_d(td)
            residual = np.zeros_like(z)
        elif len(b):
            if unregularized:
                def matvec(td):
                    return filt*to_d(shifted_action(from_d(td)))
            else:
                def matvec(td):
                    z = from_d(td)
                    return td+filt*to_d(shifted_action(z)-from_d(d*td))
            op = LinearOperator((len(b), len(b)), matvec=matvec, dtype=complex)
            td, info = gmres(op, -filt*bd, rtol=conv_tol,
                             atol=conv_tol*.01, restart=min(50, len(b)),
                             maxiter=maxiter,
                             callback=lambda value: iterations.append(float(value)),
                             callback_type='pr_norm')
            z = from_d(td)
            residual = shifted_action(z)+b if unregularized else matvec(td)+filt*bd
            if info or linalg.norm(residual) > max(10*conv_tol,
                                                   10*conv_tol*linalg.norm(b)):
                raise native.CASPT2NumericalError(
                    f'native Molcas-IPEA GMRES failed: info={info}, '
                    f'residual={linalg.norm(residual):.3e}')
        else:
            z, residual = b.copy(), b.copy()
        a0 = unshifted_action(z)
        ashift = a0+coordinates.shift_action(z, ipea_shift)
        ad = from_d(d*to_d(z))
        selected = ad if approximation == 'caspt2d' and energy_functional == 'model' else ashift
        projected = float(np.vdot(b, z).real)
        emodel = float(np.vdot(z, ad).real+2*projected)
        efull = float(np.vdot(z, ashift).real+2*projected)
        eno = float(np.vdot(z, a0).real+2*projected)
        energy = emodel if approximation == 'caspt2d' and energy_functional == 'model' else efull
        sub, subp = {}, {}
        for group in coordinates.groups:
            sl = group.orth_slice
            sub[group.sector] = sub.get(group.sector, 0.) + float(
                np.vdot(z[sl], selected[sl]+2*b[sl]).real)
            subp[group.sector] = subp.get(group.sector, 0.) + float(
                np.vdot(b[sl], z[sl]).real)
        orth_t = coordinates.lift(z)
        diagnostics = dict(self.diagnostics, **coordinates.diagnostics,
                           residual_norm=float(linalg.norm(residual)),
                           iterations=len(iterations),
                           ipea_shift=float(ipea_shift),
                           energy_definition=energy_functional+'_Hylleraas_including_Molcas_IPEA',
                           minimum_abs_regularization_denominator=float(
                               np.min(np.abs(d), initial=np.inf)))
        return native.CASPT2Result(
            self.eref, self.ef, energy, energy, self.lift(orth_t),
            1/(1+float(np.vdot(z, z).real)), sub, diagnostics,
            e_projected=projected, e_shift_correction=energy-projected,
            regularizer=regularizer, level_shift=epsilon,
            approximation=approximation, ipea_shift=float(ipea_shift),
            ipea_definition='molcas', ipea_basis='input',
            sub_projected_eners=subp, e_constraint_correction=energy-emodel,
            e_full_hylleraas=efull, e_model_hylleraas=emodel,
            e_no_ipea_hylleraas=eno, energy_functional=energy_functional)
