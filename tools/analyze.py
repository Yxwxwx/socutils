import pyscf
from pyscf import lib
import numpy as np
import scipy
import re


def analyze_casscf_spinors(mc, threshold=0.05, mo_type="active"):
    """Print dominant spinor-AO coefficients of two-component CASSCF MOs.

    Args:
        mc: A converged spinor CASCI/CASSCF object.  State-averaged wrappers
            are supported as long as they expose the usual ``mol``,
            ``mo_coeff``, ``ncore``, and ``ncas`` attributes.
        threshold: Print coefficients whose absolute value is greater than
            this nonnegative threshold.
        mo_type: ``"active"`` to analyze only the active orbitals or ``"all"``
            to analyze every MO.

    The reported real and imaginary numbers are AO expansion coefficients,
    not overlap-weighted populations.
    """
    threshold = float(threshold)
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("threshold must be finite and nonnegative")

    mode = str(mo_type).lower()
    if mode not in ("active", "all"):
        raise ValueError("mo_type must be 'active' or 'all'")

    mo_coeff = getattr(mc, "mo_coeff", None)
    if mo_coeff is None:
        raise ValueError("mc.mo_coeff is not available")
    mo_coeff = np.asarray(mo_coeff)
    if mo_coeff.ndim != 2:
        raise ValueError("mc.mo_coeff must be a two-dimensional matrix")

    mol = mc.mol
    labels = list(mol.spinor_labels())
    n_spinor_ao, nmo = mo_coeff.shape
    if n_spinor_ao != len(labels):
        raise ValueError(
            "mo_coeff row count (%d) does not match the number of spinor "
            "AO labels (%d)" % (n_spinor_ao, len(labels))
        )

    if mode == "active":
        start = int(mc.ncore)
        end = start + int(mc.ncas)
        if start < 0 or end > nmo:
            raise ValueError(
                "active orbital range [%d, %d) is outside the %d MOs"
                % (start, end, nmo)
            )
        print(
            "--- Analyzing Active Space Spinors "
            f"({start} to {end - 1}) ---"
        )
    else:
        start, end = 0, nmo
        print("--- Analyzing All Spinors ---")

    mo_energy = getattr(mc, "mo_energy", None)
    if mo_energy is not None:
        mo_energy = np.asarray(mo_energy)
        if mo_energy.ndim != 1 or mo_energy.size != nmo:
            raise ValueError(
                "mc.mo_energy must contain one value for each MO"
            )

    print(f"{'AO_Idx':<8} {'Spinor AO Label':<35} {'Real':<12} {'Imag':<12}")
    print("-" * 75)

    for mo_index in range(start, end):
        print(f"\nSpinor MO index: {mo_index}")
        if mo_energy is not None:
            energy = np.real_if_close(mo_energy[mo_index])
            if np.iscomplexobj(energy):
                raise ValueError(
                    f"MO energy {mo_index} has a non-negligible imaginary part"
                )
            print(f"Energy: {float(energy):.6f}")

        coeff_col = mo_coeff[:, mo_index]
        sorted_indices = np.argsort(np.abs(coeff_col))[::-1]
        selected = [
            ao_index
            for ao_index in sorted_indices
            if abs(coeff_col[ao_index]) > threshold
        ]
        if not selected:
            print(f"  No contributions found above threshold {threshold}")
            continue

        for ao_index in selected:
            value = coeff_col[ao_index]
            label = str(labels[ao_index]).strip()
            print(
                f"{ao_index:<8} {label:<35} "
                f"{value.real:10.4f}  {value.imag:10.4f}j"
            )


def analyze_from_chk(chkfile):
    mol = lib.chkfile.load_mol(chkfile)
    mo_energy = lib.chkfile.load(chkfile, 'scf/mo_energy')
    mo_coeff = lib.chkfile.load(chkfile,'scf/mo_coeff')
    analyze(mol, mo_coeff, mo_energy)

def analyze_mc_from_chk(chkfile):
    mol = lib.chkfile.load_mol(chkfile)
    mo_energy = lib.chkfile.load(chkfile, 'mcscf/mo_energy')
    mo_coeff = lib.chkfile.load(chkfile,'mcscf/mo_coeff')
    analyze(mol, mo_coeff, mo_energy)

def analyze_ghf(mol, mo_coeff, mo_energy):
    sph2spinor = np.vstack(mol.sph2spinor_coeff())
    mo_spinor = np.dot(sph2spinor.T, mo_coeff)
    return analyze(mol, mo_spinor, mo_energy)

def analyze_spinor_hf(mf):
    return analyze(mf.mol, mf.mo_coeff, mf.mo_energy)
def analyze(mol, mo_coeff, mo_energy):
    s = mol.intor('int1e_ovlp_spinor')
    s_sqrt = scipy.linalg.sqrtm(s)
    mo_normalized = np.dot(s_sqrt, mo_coeff)
    labels = np.array(mol.spinor_labels())
    
    label_list = dict()
    
    for idx, label in enumerate(labels):
        label = label.split()
        match = re.match(r'(\d+)([a-zA-Z].*)', label[2])
        spinor_label = f'{label[0]} {label[1]} {match.group(2):9s}'
        if spinor_label in label_list:
            label_list[spinor_label].append(idx)
        else:
            label_list[spinor_label]=[idx]
    
    for idx in range(mo_normalized.shape[1]):
        mo_i = abs(mo_normalized[:,idx])**2
        threshold = 0.01
        print(f'\nMO #{idx+1} Energy={mo_energy[idx]:16.8f}\nSpinor AO with contribution greater than {threshold:.2e} ')
        for label in label_list:
            contribution = sum(mo_i[label_list[label]])
            if contribution > threshold:
                print(f'{label}, {contribution:8.4f}')
        print(sum(mo_i))
        sort_idx = np.argsort(mo_i)
        print(f'Top contributing AOs')
        for i in range(5):
            print(f'{labels[sort_idx[-1-i]]:20s} {mo_i[sort_idx[-1-i]]:.4e}')

def analyze_nr(mol, mo_coeff, mo_energy):
    s = mol.intor('int1e_ovlp')
    s_sqrt = scipy.linalg.sqrtm(s)
    mo_normalized = np.dot(s_sqrt, mo_coeff)
    labels = np.array(mol.ao_labels())
    
    label_list = dict()
    
    for idx, label in enumerate(labels):
        label = label.split()
        match = re.match(r'(\d+)([a-zA-Z].*)', label[2])
        spinor_label = f'{label[0]} {label[1]} {match.group(2):9s}'
        if spinor_label in label_list:
            label_list[spinor_label].append(idx)
        else:
            label_list[spinor_label]=[idx]
    
    for idx in range(mo_normalized.shape[1]):
        mo_i = abs(mo_normalized[:,idx])**2
        threshold = 0.01
        print(f'\nMO #{idx+1} Energy={mo_energy[idx]:16.8f}\nAO with contribution greater than {threshold:.2e} ')
        for label in label_list:
            contribution = sum(mo_i[label_list[label]])
            if contribution > threshold:
                print(f'{label}, {contribution:8.4f}')
                
        sort_idx = np.argsort(mo_i)
        print(f'Top contributing AOs')
        for i in range(5):
            print(f'{labels[sort_idx[-1-i]]:20s} {mo_i[sort_idx[-1-i]]:.4e}')
