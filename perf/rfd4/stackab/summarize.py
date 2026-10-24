"""Read a stackab run: per leg d/s, per round B/A, AICLK during each leg.

    python3 perf/rfd4/stackab/summarize.py <run dir> [<copy evidence to>]

A leg is the median over its warm batches. designs_per_s is the fair harness's (batch / wall, writes included);
dev_per_s is batch / design_s (device time). Contended batches are read, not dropped: design_s is device time and
holds on a loud host (stage512 run4). A leg whose 1 Hz AICLK median is under 1200 MHz is dropped. Rounds pair
legs (1,2) (3,4) (5,6) of the A B B A A B order.
"""
import json
import shutil
import statistics as st
import sys
from datetime import datetime, timezone
from pathlib import Path

R = Path(sys.argv[1])
clk = [tuple(map(int, l.split())) for l in open(R / "aiclk.txt") if len(l.split()) == 2]
stamps = {}
for l in open(R / "done"):
    w = l.split()
    if len(w) >= 3 and (w[1] == "begin" or w[1].startswith("rc=")):
        t = datetime.strptime(w[2], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
        stamps.setdefault(w[0], {})["b" if w[1] == "begin" else "e"] = t


def leg(name):
    p = R / "raw" / name / "runs.jsonl"
    if not p.exists():
        return None
    recs = [json.loads(l) for l in open(p)]
    m = [r for r in recs if r["kind"] == "warm" and r.get("phase") == "measure"]
    if not m:
        return None
    # AICLK over the measured batches only (first measure start to leg end), not the compile
    t0 = datetime.strptime(m[0]["t"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp() - m[0]["wall_s"]
    t1 = stamps.get(name, {}).get("e", float("inf"))
    a = [f for t, f in clk if t0 <= t <= t1]
    return {
        "designs_per_s": round(st.median(r["designs_per_s"] for r in m), 5),
        "dev_per_s": round(st.median(r["batch"] / r["design_s"] for r in m), 5),
        "per_batch": [r["designs_per_s"] for r in m],
        "contended": sum(bool(r.get("discard")) for r in m), "batches": len(m),
        "aiclk_1hz_median": st.median(a) if a else None, "aiclk_1hz_min": min(a) if a else None,
        "aiclk_harness": [r.get("aiclk_mhz") for r in m], "sha": m[0]["sha"], "tree_dirty": m[0]["tree_dirty"],
    }


out = {}
for size in (512, 224):
    legs = {k: leg(f"s{size}_{k}{arm}") for k, arm in enumerate("ABBAAB", 1)}
    arms = dict(enumerate("ABBAAB", 1))
    dropped = [k for k, v in legs.items() if v and (v["aiclk_1hz_median"] or 0) < 1200]
    rounds = []
    for a, b in ((1, 2), (3, 4), (5, 6)):
        la, lb = (legs[a], legs[b]) if arms[a] == "A" else (legs[b], legs[a])
        if la and lb and a not in dropped and b not in dropped:
            rounds.append({"A": la["designs_per_s"], "B": lb["designs_per_s"],
                           "ratio": round(lb["designs_per_s"] / la["designs_per_s"], 4),
                           "ratio_dev": round(lb["dev_per_s"] / la["dev_per_s"], 4)})
    rs = [r["ratio"] for r in rounds]
    out[size] = {"legs": {f"{k}{arms[k]}": v for k, v in legs.items()}, "dropped_low_clock": dropped,
                 "rounds": rounds,
                 "A": st.median(r["A"] for r in rounds) if rounds else None,
                 "B": st.median(r["B"] for r in rounds) if rounds else None,
                 "ratio_median": st.median(rs) if rs else None,
                 "ratio_min": min(rs) if rs else None, "ratio_max": max(rs) if rs else None,
                 "ratio_dev_median": st.median(r["ratio_dev"] for r in rounds) if rounds else None}
print(json.dumps(out, indent=1))
if len(sys.argv) > 2:
    E = Path(sys.argv[2])
    for p in (R / "raw").glob("*/runs.jsonl"):
        (E / "raw").mkdir(parents=True, exist_ok=True)
        shutil.copy(p, E / "raw" / f"{p.parent.name}.jsonl")
    for f in ("done", "aiclk.txt"):
        shutil.copy(R / f, E / f)
    (E / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
