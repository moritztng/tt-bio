"""What a ceiling change has to cost, and what it must not.

`ceilings_fingerprint` decides whether every recorded capacity cell is stale, and a stale
baseline is hours of card time across fifteen models. It therefore has to fire on a row that
changes what tt-bio ADMITS, and not on a row that only improves the evidence for the same
admitted size.

The case that forced this: walking OpenDDE's ladder past its LADDER_TOP row and recording the
1088 failure that bounds it left `residues` at 1024 -- not one residue more is accepted or
refused -- and marked all fifteen cells stale, because `binds` and `mechanism` were in the hash.
A gate that prices honest evidence at hours of card time teaches people not to record it.
"""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("_cg", ROOT / "scripts" / "capacity_gate.py")
cg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cg)

from tt_bio import size_limits as sl  # noqa: E402

MODEL, ARCH = "opendde", "wormhole_b0"


@pytest.fixture
def row(monkeypatch):
    """Swap one row and hand back a setter, so nothing in CEILINGS is mutated for real."""
    base = sl.CEILINGS[MODEL][ARCH]

    def put(**kw):
        table = {m: dict(a) for m, a in sl.CEILINGS.items()}
        table[MODEL][ARCH] = replace(base, **kw)
        monkeypatch.setattr(sl, "CEILINGS", table)
        return cg.ceilings_fingerprint()

    return put


def test_moving_the_admitted_size_invalidates_every_cell(row):
    """The check this gate exists for: 2026-09-03 raised ceilings with nothing re-measured and
    three models broke in production traffic."""
    before = cg.ceilings_fingerprint()
    assert row(residues=base_plus(64)) != before


def base_plus(n: int) -> int:
    return sl.CEILINGS[MODEL][ARCH].residues + n


def test_a_ligand_allowance_is_an_admitted_size_too(row):
    """`ladder_ligand_atoms` converts residues into the tokens a cocrystal is checked against."""
    before = cg.ceilings_fingerprint()
    assert row(ladder_ligand_atoms=64) != before


def test_turning_a_row_unmeasured_invalidates_it(row):
    """An UNMEASURED row refuses nothing, which is the largest admitted-size change there is."""
    before = cg.ceilings_fingerprint()
    assert row(binds=sl.UNMEASURED, residues=None, pass_at=None, fail_at=None) != before


@pytest.mark.parametrize("change", [
    {"fail_at": 1088, "binds": sl.MEMORY, "mechanism": sl.FRAGMENTATION},
    {"mechanism": sl.DRAM},
    {"pass_at": 512},
    {"msa_rows": 14190},
    {"evidence": "reworded, same measurement"},
])
def test_recording_evidence_for_the_same_admitted_size_costs_nothing(row, change):
    """Every one of these says how the row was established, none says what it accepts."""
    before = cg.ceilings_fingerprint()
    assert row(**change) == before, f"{sorted(change)} forced a re-record without moving the cap"
