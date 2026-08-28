from importlib.metadata import version

import numpy as np
import pytest

from socutils.tools import analyze_casscf_spinors


class _SpinorMolecule:
    def __init__(self, labels):
        self._labels = labels

    def spinor_labels(self):
        return self._labels


class _SpinorCASSCF:
    def __init__(self):
        self.mol = _SpinorMolecule(["AO-0", "AO-1", "AO-2", "AO-3"])
        self.ncore = 1
        self.ncas = 2
        self.mo_energy = np.array([-2.0, -0.7, 0.2, 1.3])
        self.mo_coeff = np.array(
            [
                [0.0, 0.20 + 0.10j, 0.01, 0.0],
                [1.0, 0.60 - 0.20j, 0.02, 0.0],
                [0.0, 0.10 + 0.00j, 0.90j, 0.0],
                [0.0, 0.01 + 0.00j, 0.30, 1.0],
            ],
            dtype=complex,
        )


def test_locked_runtime_and_native_imports():
    import block2
    import pyblock2
    import pyscf
    import socutils
    import x2camf
    from socutils.lib import zquatev
    from pyblock2.driver.core import DMRGDriver, SymmetryTypes

    assert version("block2") == "0.5.4rc16"
    assert pyscf.__version__ == "2.14.0"
    assert socutils.__file__
    assert pyblock2.__file__
    assert DMRGDriver and SymmetryTypes.SGFCPX
    assert hasattr(x2camf, "amfi")
    assert hasattr(x2camf.libx2camf, "atm_integrals")
    assert hasattr(zquatev, "eigh")


def test_analyze_casscf_spinors_active_range_and_magnitude_order(capsys):
    mc = _SpinorCASSCF()

    assert analyze_casscf_spinors(mc, threshold=0.05) is None

    output = capsys.readouterr().out
    assert "Analyzing Active Space Spinors (1 to 2)" in output
    assert "Spinor MO index: 0" not in output
    assert "Spinor MO index: 1" in output
    assert "Spinor MO index: 2" in output
    assert output.index("AO-1") < output.index("AO-0") < output.index("AO-2")
    assert "AO-3" in output
    assert "0.6000     -0.2000j" in output


def test_analyze_casscf_spinors_all_and_input_validation(capsys):
    mc = _SpinorCASSCF()
    analyze_casscf_spinors(mc, threshold=0.95, mo_type="ALL")
    output = capsys.readouterr().out

    assert "Analyzing All Spinors" in output
    assert "Spinor MO index: 0" in output
    assert "Spinor MO index: 3" in output
    assert "No contributions found above threshold 0.95" in output

    with pytest.raises(ValueError, match="mo_type"):
        analyze_casscf_spinors(mc, mo_type="virtual")
    with pytest.raises(ValueError, match="threshold"):
        analyze_casscf_spinors(mc, threshold=-0.1)

    mc.mo_coeff = mc.mo_coeff[:-1]
    with pytest.raises(ValueError, match="row count"):
        analyze_casscf_spinors(mc)
