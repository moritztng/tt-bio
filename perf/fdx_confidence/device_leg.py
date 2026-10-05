#!/usr/bin/env python3
"""On-card leg for the confidence export: does --write_pae move a coordinate, and what does it cost.

For each model, three folds of examples/9bk6.yaml (104 + 60 aa, pinned MSA) at seed 0, each in
its own process: export off, export on, export off again. The structure must hash identically
in all three (the third run is the run-to-run control the comparison stands on). Cost is the
on-run's results.json runtime_s and peak host RSS against the mean of the two off-runs, with
AICLK sampled from sysfs once a second while each fold runs.

    TT_VISIBLE_DEVICES=<card> python perf/fdx_confidence/device_leg.py --card <card> \
        --models openfold3 boltz2 esmfold2 --out /tmp/fdxconf/device
"""
import argparse
import hashlib
import json
import os
import resource
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "examples" / "9bk6.yaml"
ARMS = (("off", False), ("on", True), ("off2", False))


def aiclk_sampler(card, stop, out):
    from tt_bio.runtime import aiclk_reading
    path = Path(f"/sys/class/tenstorrent/tenstorrent!{card}/tt_aiclk")
    while not stop.is_set():
        try:
            mhz = aiclk_reading(int(path.read_text().strip()))
            if mhz is not None:
                out.append(mhz)
        except (OSError, ValueError):
            pass
        stop.wait(1.0)


def fold(model, arm, write_pae, card, out_root, extra):
    d = out_root / model / arm
    subprocess.run(["rm", "-rf", str(d)], check=False)
    d.mkdir(parents=True)
    cmd = [sys.executable, "-m", "tt_bio.main", "predict", str(FIXTURE), "--model", model,
           "--seed", "0", "--diffusion_samples", "1", "--out_dir", str(d), *extra]
    if write_pae:
        cmd.append("--write_pae")
    env = dict(os.environ, PYTHONPATH=str(ROOT), TT_VISIBLE_DEVICES=str(card))
    clk, stop = [], threading.Event()
    t = threading.Thread(target=aiclk_sampler, args=(card, stop, clk), daemon=True)
    t.start()
    t0 = time.monotonic()
    rusage0 = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    p = subprocess.run(["/usr/bin/time", "-v", *cmd], cwd=ROOT, env=env, capture_output=True,
                       text=True)
    wall = time.monotonic() - t0
    stop.set()
    t.join()
    (d / "stdout.log").write_text(p.stdout)
    (d / "stderr.log").write_text(p.stderr)
    rss_kb = max([int(line.split()[-1]) for line in p.stderr.splitlines()
                  if "Maximum resident set size" in line] or [rusage0])
    cifs = sorted(d.rglob("*.cif"))
    best = [c for c in cifs if "_model_" not in c.name]
    res = sorted(d.rglob("results.json"))
    rt = None
    if res:
        rows = json.loads(res[0].read_text())
        rows = rows if isinstance(rows, list) else list(rows.values())
        rt = next((r.get("runtime_s") for r in rows if isinstance(r, dict)), None)
    return {"model": model, "arm": arm, "rc": p.returncode, "wall_s": round(wall, 1),
            "runtime_s": rt, "peak_rss_mb": round(rss_kb / 1024, 1),
            "cif": str(best[0]) if best else None,
            "sha256": hashlib.sha256(best[0].read_bytes()).hexdigest() if best else None,
            "aiclk_mhz": {"n": len(clk), "median": statistics.median(clk) if clk else None,
                          "min": min(clk, default=None), "max": max(clk, default=None)},
            "npz": [str(x) for x in d.rglob("*_pae.npz")]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", type=int, required=True)
    ap.add_argument("--models", nargs="+", default=["openfold3", "boltz2", "esmfold2"])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    # Boltz-2 and ESMFold-2 fold 9bk6 single-sequence on both sides (device and upstream CPU):
    # the fixture pins OpenFold3-format alignments.
    extra = {"esmfold2": ["--single_sequence"], "boltz2": ["--single_sequence"]}
    rows = []
    for m in a.models:
        for arm, on in ARMS:
            r = fold(m, arm, on, a.card, a.out, extra.get(m, []))
            rows.append(r)
            print(json.dumps(r), flush=True)
            (a.out / "device_leg.json").write_text(json.dumps(rows, indent=1))
    for m in a.models:
        rs = {r["arm"]: r for r in rows if r["model"] == m}
        same = len({r["sha256"] for r in rs.values()}) == 1 and rs["on"]["sha256"] is not None
        off = [rs[k]["runtime_s"] for k in ("off", "off2") if rs[k]["runtime_s"] is not None]
        print(f"{m}: bit-identical={same} on={rs['on']['sha256']} off={rs['off']['sha256']} "
              f"runtime on={rs['on']['runtime_s']} off={off} "
              f"rss on={rs['on']['peak_rss_mb']} off={rs['off']['peak_rss_mb']},"
              f"{rs['off2']['peak_rss_mb']} MB aiclk={rs['on']['aiclk_mhz']}")


if __name__ == "__main__":
    main()
