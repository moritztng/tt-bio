"""CPU-only checks of the scaling driver: a fake worker stands in for a chip."""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAKE = '''
import json, os, sys, time
card = os.environ["TT_VISIBLE_DEVICES"]
print(json.dumps({"ready": True, "card": card, "pid": os.getpid()}), flush=True)
for line in sys.stdin:
    if line.startswith("quit"):
        break
    w0 = time.time(); time.sleep(0.3 if card == "3" else 0.2); w1 = time.time()
    print(json.dumps({"card": card, "designs": 2, "t_start": w0, "t_end": w1, "wall_s": w1 - w0,
                      "designs_per_s": 2 / (w1 - w0), "featurize_s": 0.01, "design_s": 0.18, "write_s": 0.01,
                      "cpu_s": 0.05, "width_achieved": 2, "seed": int(line.split()[1])}), flush=True)
'''


def test_rounds_and_report(tmp_path):
    fake = tmp_path / "fake.py"
    fake.write_text(FAKE)
    out = tmp_path / "out"
    subprocess.run([sys.executable, HERE / "scale.py", "run", "--cards", "0,3", "--spec", "x", "--weights", "x",
                    "--batch", "2", "--out", out, "--reps", "2", "--rounds", "0;3;0,3;0@1", "--worker", fake],
                   check=True, capture_output=True)
    lines = [json.loads(x) for x in open(out / "rounds.jsonl")]
    rounds = [r for r in lines if r["kind"] == "round"]
    assert len(rounds) == 8 and lines[-1]["kind"] == "end"
    assert [r["active"] for r in rounds[:4]] == [[0], [3], [0, 3], [0]]
    assert [r["active"] for r in rounds[4:]] == [[3], [0, 3], [0], [0]]  # rep 2 is rotated by one
    pinned = [r for r in rounds if r["cpus_per_chip"] == 1]
    assert all(len(r["pins"]["0"]) == 1 for r in pinned)
    seeds = [v["seed"] for r in rounds for v in r["chips"].values()]
    assert len(seeds) == len(set(seeds))  # every batch on its own seed
    both = next(r for r in rounds if r["n"] == 2)
    assert both["designs"] == 4 and both["span_s"] >= 0.3

    subprocess.run([sys.executable, HERE / "scale.py", "report", out], check=True, capture_output=True)
    rep = json.loads((out / "report.json").read_text())
    # The fake has no sysfs clock, so every round is a clock artifact and nothing reaches the table.
    assert rep["clean"] == 0 and rep["rounds"] == 8
