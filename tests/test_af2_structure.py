"""`tt_bio.af2_structure` against BindCraft 2's own structure module, float64, card-free.

The eight FoldIterations as `StructureOnDevice` runs them on card, rewritten as dense ops, must be
the same function as AlphaFold 2 multimer's `generate_monomer_rigids`: act and every layer's frame
to 1e-10, with and without dropout. Run in a subprocess because the reference needs JAX in float64,
which a session that already imported JAX cannot switch to.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BC2 = Path(os.environ.get("TT_BIO_BC2_CHECKOUT", Path.home() / "bcx_e2e/bc2"))
PARAMS = Path(os.environ.get("JAPANFOLD_BC2_AF2_WEIGHTS", Path.home() / "bcx_e2e/af2_params")) \
    / "params_model_1_multimer_v3.npz"


@pytest.mark.skipif(not (BC2 / "bindcraft").is_dir() or not PARAMS.is_file(),
                    reason="needs a BindCraft 2 checkout and AF2 multimer weights")
def test_structure_module_is_bindcraft2s_in_float64():
    r = subprocess.run([sys.executable, "perf/bc4_headdev/sm_check.py", "--bc2", str(BC2),
                        "--params", str(PARAMS), "--n", "40"],
                       cwd=REPO, capture_output=True, text=True, timeout=600,
                       env={**os.environ, "PYTHONPATH": str(REPO)})
    assert r.returncode == 0, r.stdout + r.stderr
