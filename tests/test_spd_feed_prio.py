"""perf/spd/feed.py clamps an out-of-range grid prio into 3-8 instead of dropping the experiment."""
import importlib.util
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")
_spec = importlib.util.spec_from_file_location("spd_feed", Path(__file__).resolve().parents[1] / "perf/spd/feed.py")
feed = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(feed)


def _grid(tmp_path, prio):
    exp = {"name": "x", "arch": "wh", "prio": prio, "axes": {"n": [1]}, "done": ["/o/{n}"], "command": "echo {n}"}
    (tmp_path / "r.yaml").write_text(yaml.safe_dump({"row": "spd-r", "tier": "red", "experiments": [exp]}))
    logs = []
    return feed.cells(tmp_path, log=logs.append), logs


@pytest.mark.parametrize("prio,want", [(2, 3), (0, 3), (5, 5), (9, 8)])
def test_prio_clamped(tmp_path, prio, want):
    cells, _ = _grid(tmp_path, prio)
    assert [c["name"].split("-")[0] for c in cells] == [str(want)]


def test_non_numeric_prio_skipped(tmp_path):
    cells, logs = _grid(tmp_path, "high")
    assert cells == [] and "not a number" in logs[0]
