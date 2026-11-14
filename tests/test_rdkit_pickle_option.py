"""RDKit's default pickle option is process-wide. atomworks (RF3, OpenFold3) sets AllProps at
import and asserts it before every conformer generation; Nesso-1 and OpenDDE guidance need atom
properties and used to SET AtomProps, so an RF3 fold after either one in the same process died
with "Default pickle properties are not set to all properties". The 39554dfc3 gate caught it as
an order-dependent test_rf3_featurizer failure. Subprocess, so no other test's setting leaks in."""
import pickle
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _run(code: str) -> str:
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO)
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout.strip().splitlines()[-1]


def test_atom_prop_users_keep_atomworks_allprops(tmp_path):
    pkl = tmp_path / "ccd.pkl"
    pkl.write_bytes(pickle.dumps({}))
    got = _run("\n".join([
        "import sys; sys.path.insert(0, %r)" % str(REPO),
        "from rdkit import Chem",
        "import tt_bio._vendor.atomworks.ml.transforms.rdkit_utils  # noqa: F401 - sets AllProps",
        "import tt_bio.nesso1_input  # noqa: F401",
        "import tt_bio._vendor.nesso.data.featurizer  # noqa: F401",
        "from tt_bio.tfg.features import _load_pkl",
        "_load_pkl(%r)" % str(pkl),
        "print(int(Chem.GetDefaultPickleProperties()) == int(Chem.PropertyPickleOptions.AllProps))",
    ]))
    assert got == "True"


def test_alone_nesso1_still_gets_exactly_atom_props():
    got = _run("\n".join([
        "import sys; sys.path.insert(0, %r)" % str(REPO),
        "from rdkit import Chem",
        "import tt_bio.nesso1_input  # noqa: F401",
        "print(int(Chem.GetDefaultPickleProperties()) == int(Chem.PropertyPickleOptions.AtomProps))",
    ]))
    assert got == "True"
