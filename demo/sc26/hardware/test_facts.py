"""The comparison screen shows the published numbers and nothing else.

    python3 -m pytest demo/sc26/hardware/test_facts.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import facts  # noqa: E402


def test_rows_are_the_published_cells():
    src = json.loads(facts.SRC.read_text())
    cells = {m["id"]: m["cells"] for m in src["models"] + src["design"]["models"] + src["affinity"]["models"]}
    rows = facts.build()["rows"]
    # every published row measured on all four platforms, Protenix-v2 included: its weights' licence
    # governs hosting it, not the numbers of a model tt-bio runs on your own card
    assert rows and "protenix-v2" in {r["id"] for r in rows}
    for r in rows:
        c = cells[r["id"]]
        for k in ("h200", "b200", "a100"):
            assert r[k] == c[k].get("s_per_fold", c[k].get("s_per_design"))
        assert r["tt"] == c["p150a"].get("s_per_fold", c["p150a"].get("s_per_design"))


def test_bindcraft2_figures_are_still_what_the_doc_says():
    doc = (facts.REPO / "docs/bindcraft2.md").read_text()
    row = next(l for l in doc.splitlines() if l.startswith("| p300c chip |"))
    bc = facts.BINDCRAFT2
    assert f"{bc['tt']} s" in row and "median 1350, min 1300" in row
    assert f"{bc['h200']} s" in doc


def test_committed_facts_json_is_current():
    assert json.loads(facts.OUT.read_text()) == facts.build(), "run python3 demo/sc26/hardware/facts.py"
