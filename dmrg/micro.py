"""Disk transactions for single-root orbital microiterations.

Block2's allocator is process-global.  This module deliberately never clones
or keeps two live DMRGDriver instances.  Snapshots contain saved GS tensor
files and numeric metadata, not native MPS handles.  A rollback reconstructs
one driver using DMRGCI.restore_checkpoint(), without any optimization sweeps.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace

import numpy as np


@dataclass
class DMRGSnapshot:
    directory: Path
    hamiltonian: dict
    controls: dict
    diagnostics: dict

    def close(self):
        shutil.rmtree(self.directory, ignore_errors=True)


class DMRGOrbitalSession:
    """Keep all tentative kernels away from the public checkpoint directory."""

    def __init__(self, solver):
        if solver.nroots != 1:
            raise NotImplementedError('MPS orbital transactions currently require nroots=1')
        self.solver = solver
        self.public_directory = solver.checkpoint_dir
        self.directory = None

    def __enter__(self):
        os.makedirs(self.solver.scratch, exist_ok=True)
        self.directory = Path(tempfile.mkdtemp(prefix='orbital_micro_', dir=self.solver.scratch))
        work = self.directory / 'working'
        try:
            if self.solver.resume:
                if self.public_directory is None:
                    raise ValueError('resume requires a checkpoint directory')
                shutil.copytree(self.public_directory, work)
            self.solver.checkpoint_dir = str(work)
            return self
        except Exception:
            shutil.rmtree(self.directory, ignore_errors=True)
            raise

    def __exit__(self, *exc):
        self.solver.checkpoint_dir = self.public_directory
        if self.directory is not None:
            shutil.rmtree(self.directory, ignore_errors=True)

    def _export(self, directory, *, orbital_converged=False):
        solver = self.solver
        h = solver.checkpoint_hamiltonian
        if h is None or not solver.converged or solver.convergence_info.get('approximate', False):
            raise RuntimeError('Only a strict converged DMRG state can be snapshotted')
        schedule = copy.deepcopy(solver.convergence_info.get('schedule'))
        if not schedule or not schedule.get('thrds'):
            raise RuntimeError('Cannot snapshot a DMRG state without its executed schedule')
        old = solver.checkpoint_dir
        try:
            solver.checkpoint_dir = str(directory)
            problem = solver._checkpoint_problem(h['h1e'], h['eri'], h['norb'],
                h['nelec'], h['nroots'], h['weights'], h['ecore'])
            if problem['hamiltonian_sha256'] != h['hamiltonian_sha256']:
                raise RuntimeError('Live DMRG Hamiltonian fingerprint changed')
            solver._begin_checkpoint(problem, 'orbital-snapshot', solver.driver.reorder_idx)
            solver._save_final_checkpoint_mps(solver._multi_mps)
            solver._complete_checkpoint(SimpleNamespace(as_dict=lambda: schedule), 'orbital-snapshot')
            manifest = directory / 'dmrgci-checkpoint.json'
            data = solver._read_json(str(manifest))
            data['orbital_converged'] = bool(orbital_converged)
            solver._write_json_atomic(str(manifest), data)
        finally:
            solver.checkpoint_dir = old
        return h

    def snapshot(self):
        directory = Path(tempfile.mkdtemp(prefix='accepted_', dir=self.directory))
        try:
            h = self._export(directory)
        except Exception:
            shutil.rmtree(directory)
            raise
        solver = self.solver
        names = ('restart', '_restart', 'resume', 'final_one_site', 'restart_diagnostics')
        controls = {name: copy.deepcopy(getattr(solver, name)) for name in names}
        # Mapping objects contain numeric arrays, not native MPS state.
        if solver.kramers_adapter is not None:
            controls['_kramers_adapter'] = copy.deepcopy(solver.kramers_adapter)
        return DMRGSnapshot(directory, h, controls, copy.deepcopy(solver.convergence_info))

    def restore(self, snapshot, *, max_memory=None, verbose=None):
        solver, h = self.solver, snapshot.hamiltonian
        old = solver.checkpoint_dir
        try:
            solver.checkpoint_dir = str(snapshot.directory)
            solver.final_one_site = snapshot.controls['final_one_site']
            solver.restore_checkpoint(h['h1e'], h['eri'], h['norb'], h['nelec'],
                ecore=h['ecore'], nroots=1, max_memory=max_memory, verbose=verbose)
            # The historical restore path admits a two-site mismatch with a
            # warning. Transactions must not silently accept an inconsistent state.
            tol = max(1e-7, 10*solver.tol,
                      10*np.sqrt(min(snapshot.diagnostics['schedule']['thrds'])))
            if (np.max(abs(solver.root_overlap - np.eye(1))) > tol or
                np.max(abs(solver.projected_hamiltonian -
                              solver.root_overlap*float(solver.e_tot))) > tol):
                raise RuntimeError('Restored MPS failed orbital-transaction validation')
            for name, value in snapshot.controls.items():
                if name == '_kramers_adapter':
                    solver.kramers_adapter = copy.deepcopy(value)
                else:
                    setattr(solver, name, copy.deepcopy(value))
            solver.convergence_info = copy.deepcopy(snapshot.diagnostics)
            solver.convergence_info['orbital_snapshot_restored'] = True
            solver.convergence_info['scratch'] = solver._scratch
        finally:
            solver.checkpoint_dir = old

    def publish(self):
        """Publish only the final, orbital-converged reference; manifest last."""
        if self.public_directory is None:
            return
        target = Path(self.public_directory)
        target.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix='.micro_publish_', dir=target))
        try:
            self._export(stage, orbital_converged=True)
            manifest_path = target / 'dmrgci-checkpoint.json'
            previous = manifest_path.read_bytes() if manifest_path.exists() else None
            old_mps = target / 'mps'
            backup = stage / 'old_mps'
            moved_old = moved_new = False
            try:
                running = json.loads((stage / 'dmrgci-checkpoint.json').read_text())
                running.update(status='publishing', converged=False)
                self.solver._write_json_atomic(str(manifest_path), running)
                if old_mps.exists():
                    os.replace(old_mps, backup)
                    moved_old = True
                os.replace(stage / 'mps', old_mps)
                moved_new = True
                os.replace(stage / 'dmrgci-checkpoint.json', manifest_path)
            except Exception:
                if moved_new:
                    shutil.rmtree(old_mps, ignore_errors=True)
                if moved_old:
                    os.replace(backup, old_mps)
                if previous is not None:
                    manifest_path.write_bytes(previous)
                else:
                    manifest_path.unlink(missing_ok=True)
                raise
        finally:
            shutil.rmtree(stage, ignore_errors=True)
