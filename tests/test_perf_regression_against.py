"""perf_regression.py --against: the reference tree measures itself first, then the gate reads
the baseline file that run wrote, not the committed one."""
import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _module():
    spec = importlib.util.spec_from_file_location("perf_regression_t", REPO / "scripts" / "perf_regression.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_against_measures_the_reference_tree_then_gates_on_its_file(tmp_path):
    ref = tmp_path / "v0.13.1"
    (ref / "scripts").mkdir(parents=True)
    (ref / "scripts" / "perf_regression.py").write_text(
        "import json, os, sys, pathlib\n"
        "pathlib.Path('docs').mkdir(exist_ok=True)\n"
        "pathlib.Path('docs/perf_baselines.json').write_text(json.dumps("
        "{'argv': sys.argv[1:], 'pythonpath': os.environ['PYTHONPATH']}))\n")
    m = _module()
    m._measure_reference(Namespace(against=ref, model=["nesso1"]))
    assert m.BASELINE_FILE == ref / "docs" / "perf_baselines.json"
    got = json.loads(m.BASELINE_FILE.read_text())
    assert got["argv"][:2] == ["--update-baseline", "--allow-contended"]
    assert got["argv"][-2:] == ["--model", "nesso1"]
    assert got["pythonpath"].split(":")[0] == str(ref)
    assert json.loads((REPO / "docs" / "perf_baselines.json").read_text()) != got


def test_a_failed_reference_run_fails_the_gate(tmp_path):
    ref = tmp_path / "ref"
    (ref / "scripts").mkdir(parents=True)
    (ref / "scripts" / "perf_regression.py").write_text("import sys; sys.exit(3)\n")
    m = _module()
    try:
        m._measure_reference(Namespace(against=ref, model=None))
    except SystemExit as e:
        assert "exited 3" in str(e)
    else:
        raise AssertionError("a failed reference run must fail the gate")
