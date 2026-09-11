"""The OpenFold3 data pipeline must not import a 2D molecule drawing library.

`tt_bio._vendor.openfold3...structure.component` used to import pdbeccdutils at module
scope. pdbeccdutils imports `rdkit.Chem.Draw`, which dlopens libXrender at import time,
so on a host without that system library every OpenFold3 / OpenBind fold died inside the
data pipeline before any device work -- plain protein included (7/7 targets, j10glx02,
2026-09-11). The import belongs in the one function that builds a CCD component.

Checked in a subprocess because pytest itself may already have pulled pdbeccdutils in
through another test, which would make an in-process `sys.modules` check pass for the
wrong reason.
"""
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

PROBE = """
import sys
import tt_bio.openfold3_data  # the entry point _predict_openfold3_one imports
print("pdbeccdutils" in sys.modules)
"""


def test_openfold3_data_imports_without_pdbeccdutils():
    pytest.importorskip("torch")
    out = subprocess.run([sys.executable, "-c", PROBE], cwd=REPO,
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().endswith("False"), (
        "importing tt_bio.openfold3_data pulled in pdbeccdutils, which needs libXrender:\n"
        + out.stdout)


def test_the_ccd_path_still_reaches_pdbeccdutils():
    """The negative control: deferring the import must not have removed the dependency."""
    src = (REPO / "tt_bio/_vendor/openfold3/core/data/primitives/structure"
                  "/component.py").read_text()
    body = src.split("def pdbeccdutils_component_from_ccd", 1)[1]
    assert "from pdbeccdutils.core import ccd_reader" in body.split("\ndef ", 1)[0]
