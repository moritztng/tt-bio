"""parity_from_step keeps only same-weights pairs, so a post-update step can never pass as parity."""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "parity_from_step", Path(__file__).resolve().parents[1] / "perf/r4t/parity_from_step.py")
pfs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pfs)


def rec(step, batch, **kw):
    return dict(step=step, batch=batch, terms={"a": 1.0, "b": 2.0}, upstream_terms={"a": 1.0, "b": 2.5}, **kw)


def test_only_unupdated_steps_are_kept_and_each_batch_once():
    lines = [("A", rec(1, "b08")), ("A", rec(2, "b01")), ("B", rec(1, "b08")),
             ("E", rec(5, "b03", updates_before=0)), ("E", rec(1, "b04", updates_before=3))]
    kept, dropped = pfs.pairs(lines)
    assert [p["batch"] for p in kept] == ["b08", "b03"]
    assert len(dropped) == 3
    assert kept[0]["ours"]["total"] == 3.0 and kept[0]["ref"]["total"] == 3.5


def test_a_resumed_step_one_is_not_same_weights():
    kept, _ = pfs.pairs([("R", rec(1, "b08", resumed=True))])
    assert kept == []
