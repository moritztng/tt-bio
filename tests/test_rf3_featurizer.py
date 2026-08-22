"""RF3 host featurizer parity, on committed fixtures, with no device and no foundry install.

`scripts/rf3_port/parity_gate.py` scores tt-bio's vendored RF3 featurizer against ten
captures of the upstream pipeline (one per capability class: protein monomer and multimer,
DNA, RNA, ligands in all three input forms, covalent glycan, non-canonical residues, MSA,
cyclic chains, templates). The captures are committed and 4.8 MB total, so this is the one
RF3 gate that runs anywhere -- a laptop, CI, a host with no card.

GAP_ENV is a skip, not a failure. `feats/ref_pos` is an RDKit-generated conformer and RDKit
moves it between releases, so on a machine whose RDKit differs from the captures' the gate
reports GAP_ENV with both versions named and every surviving mismatch inside the
RDKit-derived key set. That is an environment difference, and reporting it as a failure
would be the same inversion as guarding on a fixture's existence while depending on its
contents (see tests/of3_golden.py).
"""
import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
GATE = REPO / "scripts" / "rf3_port" / "parity_gate.py"


def _load_rf3_parity_gate():
    """Load the RF3 scorer by path, under a name of its own.

    `scripts/rf3_port/parity_gate.py` and `scripts/rfd3_port/parity_gate.py` are two
    different scorers with the same module name and the same `featurizer_parity()`
    entry point. Importing either by bare name binds `sys.modules["parity_gate"]`
    process-wide, so in a suite that already imported one, the other silently gets
    the wrong scorer's report. Neither module needs its directory on `sys.path` --
    both resolve everything from REPO -- so load by path and keep the names apart.
    """
    spec = importlib.util.spec_from_file_location("rf3_parity_gate", GATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_rf3_featurizer_parity():
    rep = _load_rf3_parity_gate().featurizer_parity()
    assert rep["mode"] == "rf3_featurizer", rep.get("mode")
    if rep["verdict"] == "GAP_ENV":
        pytest.skip("RDKit differs from the captures' ({}), and every mismatch is "
                    "RDKit-derived: {}".format(rep["env_mismatch"], rep["fixtures_pass"]))
    assert rep["verdict"] == "PASS", rep
    # the fixture count is asserted so a fixture that stops being discovered fails loudly
    # rather than passing a gate over nine of ten capability classes
    assert rep["fixtures_total"] == 10, rep["fixtures_total"]
    assert rep["fixtures_pass"] == rep["fixtures_total"], rep
    assert rep["keys_bitexact"] == rep["keys_total"], rep
