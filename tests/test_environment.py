from importlib.metadata import version


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
