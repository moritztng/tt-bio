"""Regression: one truth predicate for the ``TT_BIO_*`` boolean gates.

The gates were read four different ways, so ``TT_BIO_SDPA_DIV_K=true`` turned that
gate on while ``TT_BIO_TRIATT_MASK_Q_SPLIT=true`` turned that one off, and an empty
value meant on for twelve gates and off for two. ``env_flag`` is the single reader.

The first test is the bit-exactness argument for the conversion: the release gate
and the fleet only ever set ``0``/``1`` or leave a gate unset, and on exactly those
three inputs ``env_flag`` agrees with all four predicates it replaced.
"""
from __future__ import annotations

import pytest

from tt_bio.envflags import env_flag

OLD = {
    "ne0":       lambda raw, d: (raw if raw is not None else ("1" if d else "0")) != "0",
    "eq1":       lambda raw, d: (raw if raw is not None else ("1" if d else "0")) == "1",
    "not_0_or_": lambda raw, d: (raw if raw is not None else ("1" if d else "0")) not in ("0", ""),
    "not_true":  lambda raw, d: (raw if raw is not None else ("1" if d else "0")) in ("1", "true", "True"),
    "not_in_f":  lambda raw, d: (raw if raw is not None else ("1" if d else "0")) not in ("0", "false", "False"),
}


@pytest.mark.parametrize("raw", [None, "0", "1"])
@pytest.mark.parametrize("default", [True, False])
def test_agrees_with_every_predicate_it_replaced(raw, default, monkeypatch):
    """On the only values the gate ever sets, the new reader is the old reader."""
    monkeypatch.delenv("TT_BIO_TEST_GATE", raising=False)
    if raw is not None:
        monkeypatch.setenv("TT_BIO_TEST_GATE", raw)
    got = env_flag("TT_BIO_TEST_GATE", default)
    for name, old in OLD.items():
        assert got == old(raw, default), f"{name} disagrees for raw={raw!r} default={default}"


@pytest.mark.parametrize("raw,want", [("1", True), ("true", True), ("True", True),
                                      ("YES", True), ("on", True), (" 1 ", True),
                                      ("0", False), ("false", False), ("no", False),
                                      ("off", False)])
def test_spelling_is_now_consistent(raw, want, monkeypatch):
    monkeypatch.setenv("TT_BIO_TEST_GATE", raw)
    assert env_flag("TT_BIO_TEST_GATE", not want) is want


@pytest.mark.parametrize("raw", ["", "   "])
def test_empty_means_the_default(raw, monkeypatch):
    monkeypatch.setenv("TT_BIO_TEST_GATE", raw)
    assert env_flag("TT_BIO_TEST_GATE", True) is True
    assert env_flag("TT_BIO_TEST_GATE", False) is False


def test_a_typo_raises_instead_of_picking_a_branch(monkeypatch):
    monkeypatch.setenv("TT_BIO_TEST_GATE", "tru")
    with pytest.raises(ValueError, match="not a boolean"):
        env_flag("TT_BIO_TEST_GATE", False)


def test_no_module_reads_a_gate_by_hand():
    """Any hand-rolled boolean gate read must go through env_flag.

    The prefix allow-list this replaced was ``(TT_BIO_|TT_PROTENIX_)``, so a whole
    family of model-scoped gates -- ``PROTENIX_``, ``OF3_``, ``OPENDDE_``, ``BOLTZ2_``,
    ``RFD3_`` -- sat outside the scanner. PXDesign then wrote its own
    ``PROTENIX_DIFFUSION_FP32_DEVICE`` read four days after env_flag landed and the
    scanner stayed green. Any upper-case name counts now, and the comparand set covers
    the tuple form (``not in ("0", "false", "False")``) as well as ``== "1"`` / ``!= "0"``.
    """
    import re
    from pathlib import Path

    import tt_bio

    root = Path(tt_bio.__file__).parent
    boolish = r'"(?:0|1|true|false|True|False|yes|no|on|off)"'
    pat = re.compile(r'[\w.]*environ\.get\(\s*"[A-Z][A-Z0-9_]*"[^)]*\)\s*'
                     r'(?:[!=]=\s*' + boolish + r'|(?:not\s+)?in\s*\([^)]*' + boolish + r')')
    bad = []
    for f in root.rglob("*.py"):
        if "_vendor" in f.parts:
            continue
        for i, ln in enumerate(f.read_text().splitlines(), 1):
            if pat.search(ln):
                bad.append(f"{f.relative_to(root)}:{i}: {ln.strip()}")
    assert not bad, "hand-rolled boolean gate; use env_flag:\n" + "\n".join(bad)
