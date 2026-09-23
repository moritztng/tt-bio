"""A template with two copies of the query's chains maps to the same copy in every process.

Host-only, no card. 2AD6 is an a2b2 crystal (alpha on A and C, beta on B and D) and the query
is one alpha-beta pair, so each query chain ties between two template chains. The search breaks
the tie by column order; when that order came from a set of chain names it followed the
process's string-hash seed, and boltz2 folded the same input three different ways in four runs.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "perf" / "mgx_combos" / "inputs" / "mdh_640.yaml"
MOL_DIR = os.path.expanduser("~/.boltz/mols")

pytestmark = pytest.mark.skipif(not os.path.exists(MOL_DIR),
                                reason="needs bundled CCD mol library (~/.boltz/mols)")

PROBE = """
import json, sys, yaml
from tt_bio.data.parse import parse_boltz_schema
schema = json.loads(sys.argv[1])
t = parse_boltz_schema("t", schema, ccd={}, mol_dir=sys.argv[2], boltz_2=True)
print(json.dumps(sorted((r.query_chain, r.template_chain, r.query_st, r.template_st)
                        for r in t.record.templates)))
"""


def _records(seed: int, schema: dict) -> list:
    env = dict(os.environ, PYTHONHASHSEED=str(seed), PYTHONPATH=str(REPO))
    out = subprocess.run([sys.executable, "-c", PROBE, json.dumps(schema), MOL_DIR],
                         env=env, cwd=REPO, capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_two_copy_template_maps_the_same_chains_under_every_hash_seed():
    schema = yaml.safe_load(FIXTURE.read_text())
    for s in schema["sequences"]:
        s.get("protein", {}).pop("msa", None)       # the template search never reads the MSA
    schema["templates"][0]["cif"] = str(REPO / schema["templates"][0]["cif"])
    runs = [_records(seed, schema) for seed in range(6)]
    assert all(r == runs[0] for r in runs), runs
    # File order breaks the tie, so both query chains land on the first copy.
    assert {(q, t) for q, t, *_ in runs[0]} == {("A", "A"), ("B", "B")}
