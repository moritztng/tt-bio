#!/usr/bin/env python3
"""The stamper must not be able to classify a defect. CPU only, no card, no network.

`_of3t_donecheck.py` refuses this campaign's GO while any defect is USER-FACING, and it reads
`state/of3t/UNFIXED_TRIAGE.json`. Two programs used to write that file: `triage.py`, which asserts
its classification against the live UNFIXED set and REFUSES (rc=1) when the set has moved, and
`stamp_row_counts.py`, which loaded the same file and reconciled it in place -- appending any
newly-UNFIXED defect to CAMPAIGN-INTERNAL with "classified by stamp_row_counts.py; no user-facing
claim made". So the one path that could walk a new USER-FACING defect past the GO gate in silence
was the stamper.

These tests pin the property that fixes it: the generator owns the artifact, and its refusal is
load-bearing rather than cosmetic.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRIAGE = HERE / "triage.py"
STAMPER = HERE.parent / "stamp_row_counts.py"
ART = Path("/home/moritz/.coworker/state/of3t/UNFIXED_TRIAGE.json")


def _digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_triage_refuses_an_unclassified_unfixed_defect():
    """The refusal the stamper now inherits is real: an UNFIXED defect absent from the TABLE
    returns 1 and writes nothing. Exercised on the REAL module with its text source stubbed, so
    it tests the shipped control flow rather than a reimplementation of it."""
    sys.path.insert(0, str(HERE))
    import importlib

    tri = importlib.import_module("triage")
    real_text, real_out = tri._defects_text, tri.OUT
    text, archives = real_text()
    # A defect number that cannot collide with a filed one, UNFIXED, and not in TABLE.
    # The status must be IN THE HEADING: `status_vocab.statuses_by_defect` reads declarations out
    # of the heading line only, so a body that says "UNFIXED." is invisible to it. Writing this
    # stub the obvious way produced a green refusal test that was refusing nothing.
    stub = text + "\n\n### D99001 a defect filed while the stamper was running. **UNFIXED.**\n"
    tmp = Path("/tmp/of3t_triage_refusal_probe.json")
    if tmp.exists():
        tmp.unlink()
    tri._defects_text = lambda: (stub, archives)
    tri.OUT = tmp
    try:
        rc = tri.main()
    finally:
        tri._defects_text, tri.OUT = real_text, real_out
    assert rc == 1, "triage.py reported a classification of a set that had moved"
    assert not tmp.exists(), "triage.py wrote an artifact on the refusal path"


def test_triage_is_current_against_the_live_set():
    """And when nothing has moved it returns 0 -- otherwise the refusal above proves nothing,
    because a script that always fails also 'refuses'."""
    r = subprocess.run([sys.executable, str(TRIAGE)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_the_stamper_never_writes_the_gates_artifact():
    """Behavioural, not a source read: run the real stamper and show the artifact's bytes are
    whatever the generator last wrote. Any byte the stamper contributes is a class nobody
    classified."""
    subprocess.run([sys.executable, str(TRIAGE)], capture_output=True, text=True, check=True)
    before = _digest(ART)
    r = subprocess.run([sys.executable, str(STAMPER)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert _digest(ART) == before, "stamp_row_counts.py modified UNFIXED_TRIAGE.json"


def test_the_stamper_holds_no_classification_of_its_own():
    """The removed default class was a literal in this file. A future edit that reintroduces a
    class name here is reintroducing the defect, so name the strings rather than trust the diff."""
    # Comments are stripped first: this file's own explanation of the defect quotes the strings
    # it forbids, and so does the stamper's.
    src = "\n".join(l for l in STAMPER.read_text().splitlines() if not l.lstrip().startswith("#"))
    assert "triage.py" in src, "the stamper no longer invokes the generator"
    assert 'j["classes"]["CAMPAIGN-INTERNAL"].append' not in src
    assert "no user-facing claim made" not in src
    assert "tp.write_text" not in src, "the stamper writes the artifact again"


def test_the_stamper_writes_nothing_when_it_has_no_hold_on_the_doc():
    """The second half of the same defect. The stamper ran eight regex substitutions, wrote the
    doc, and only then checked ONE of them -- so after the doc was rewritten past all eight it
    exited 1 on every pass while still having rewritten the gate's artifact on the way. Writes
    must therefore come after the match count, not before it."""
    src = "\n".join(l for l in STAMPER.read_text().splitlines() if not l.lstrip().startswith("#"))
    refuse = src.index("STAMP REFUSED, NOTHING WRITTEN")
    assert src.index("p.write_text(t)") > refuse, "the doc is written before the sites are counted"
    assert src.index("subprocess.run") < src.index("json.loads(tp.read_text())")


def test_every_stamped_block_is_closed_in_the_doc():
    """A delimited block replaced the regex sites because a rewrite that drops a marker is
    visible, while a rewrite that moves a sentence silently zeroes a regex. That only holds while
    both markers survive, so check them."""
    doc = Path("/home/moritz/.coworker/state/of3t-orchestrator.md").read_text()
    for name in ("defects", "rows"):
        assert doc.count(f"<!-- STAMPED:{name} -->") == 1, name
        assert doc.count(f"<!-- /STAMPED:{name} -->") == 1, name


def test_the_artifact_carries_one_schema():
    """`reasons` was a bare string when the stamper had written the entry and a dict when the
    generator had; the gate and every reader had to accept both."""
    j = json.loads(ART.read_text())
    kinds = {type(v).__name__ for v in j["reasons"].values()}
    assert kinds == {"dict"}, f"reasons carries mixed schemas: {kinds}"
    assert sum(j["counts"].values()) == j["unfixed_total"]


if __name__ == "__main__":
    raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", "-q", __file__]))
