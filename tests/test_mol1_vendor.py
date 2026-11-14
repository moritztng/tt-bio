"""The vendored Mol-1 input pipeline imports without Mol-1 installed and is the pinned upstream copy."""
import importlib
import re
from pathlib import Path

import pytest

VENDOR = Path(__file__).resolve().parents[1] / "tt_bio" / "_vendor" / "mol1"


@pytest.mark.parametrize("mod", ["constants", "msa_utils", "msa_assembly", "constants_full", "tools_common",
                                 "kalign", "template_parser", "template_utils", "template_featurizer", "helpers",
                                 "search", "sources", "template_search", "rna_msa_search"])
def test_vendored_module_imports(mod):
    importlib.import_module(f"tt_bio._vendor.mol1.{mod}")


def test_every_vendored_file_names_the_same_upstream_commit():
    shas = {re.search(r"@ ([0-9a-f]{40})", p.read_text()).group(1) for p in VENDOR.glob("*.py")}
    assert len(shas) == 1, shas
    assert not any(re.search(r"^\s*(from|import) mol\b", p.read_text(), re.M) for p in VENDOR.glob("*.py"))
