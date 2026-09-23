#!/usr/bin/env python3
"""of3t-stackship: inference before/after the training default, from firing and digest.

    infab.py --before TREE --after TREE --card 1 --workdir W --out INFAB.json

Per model, folds interleaved BEFORE, AFTER, BEFORE, AFTER on one card: the two BEFORE folds are
the A/A floor, BEFORE vs AFTER the A/B. of3t-verbinstall's `infaa.py` census, taken inside the
folding process at exit (D236), extended to this row's counters:

  autograd_imported        whether `tt_bio.autograd` is in the folding process at all
  EXACT_*_STATS            the exact softmax / layer norm counters, read WITHOUT importing it
  raw_ops                  which callable `ttnn.softmax` and `ttnn.layer_norm` are at exit

The non-zero control for the same counters is `STACK_SHIP_SHIP.json`, the training arm.
AICLK is sampled from tt-smi every 3 s DURING each fold and stamped beside its wall time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

SMI = "/home/ttuser/.local/bin/tt-smi"
SITE = '''import atexit, json, sys
def _dump():
    out = {"pid": __import__("os").getpid()}
    mod = sys.modules.get("tt_bio.autograd")
    out["autograd_imported"] = mod is not None
    for n in ("EXACT_SOFTMAX_STATS", "EXACT_LAYER_NORM_STATS"):
        out[n] = dict(getattr(mod, n)) if mod is not None and hasattr(mod, n) else None
    t = sys.modules.get("ttnn")
    out["raw_ops"] = ({n: getattr(getattr(t, n, None), "__qualname__", repr(getattr(t, n, None)))
                       for n in ("softmax", "layer_norm")} if t is not None else None)
    print("STACKSHIP_CENSUS " + json.dumps(out), flush=True)
atexit.register(_dump)
'''


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def clock(card: int, stop: threading.Event, out: list):
    while not stop.is_set():
        try:
            d = json.loads(subprocess.run([SMI, "-s"], capture_output=True, text=True,
                                          timeout=20).stdout)
            out.append(int(d["device_info"][card]["telemetry"]["aiclk"].strip()))
        except Exception:
            pass
        stop.wait(3)


def fold(py, tree: Path, spec, out: Path, card: str):
    subprocess.run(["rm", "-rf", str(out)], check=False)
    out.mkdir(parents=True)
    sc = out.parent / ("_sc_" + out.name)
    sc.mkdir(parents=True, exist_ok=True)
    (sc / "sitecustomize.py").write_text(SITE)
    env = {k: v for k, v in os.environ.items() if not k.startswith("TT_BIO_")}
    env.update({"TT_VISIBLE_DEVICES": card, "TT_BIO_LEASE_CARDS": card,
                "TT_BIO_LEASE_HOLDER": "worker:of3t-stackship", "OMP_NUM_THREADS": "8",
                "PYTHONPATH": str(sc) + os.pathsep + str(tree)})
    if spec["kind"] == "predict":
        cmd = [py, "-m", "tt_bio.main", "predict", str(tree / spec["fixture"]), "--model",
               spec["model"], "--single_sequence", "--sampling_steps", "6",
               "--diffusion_samples", "1", "--seed", "0", "--out_dir", str(out)]
    else:
        cmd = [py, str(tree / "scripts/af2_port/fold_timing.py"), "--pdb",
               str(tree / spec["fixture"]), "--reps", "2", "--out", str(out / "af2.json")]
    samples, stop = [], threading.Event()
    th = threading.Thread(target=clock, args=(int(card), stop, samples), daemon=True)
    th.start()
    t = time.time()
    r = subprocess.run(cmd, cwd=str(tree), env=env, capture_output=True, text=True)
    secs = time.time() - t
    stop.set(); th.join()
    census = [json.loads(l.split(" ", 1)[1]) for l in r.stdout.splitlines()
              if l.startswith("STACKSHIP_CENSUS ")]
    if spec["kind"] == "predict":
        digests = {c.name: sha(c) for c in sorted(out.rglob("*.cif"))}
    else:
        rep = json.loads((out / "af2.json").read_text()) if (out / "af2.json").exists() else {}
        digests = {"structure_sha16_all": rep.get("structure_sha16_all")}
    s = sorted(samples)
    return {"rc": r.returncode, "seconds": round(secs, 2), "digests": digests,
            "census": census,
            "aiclk_mhz_sampled_DURING": ({"n": len(s), "min": s[0], "median": s[len(s) // 2],
                                          "max": s[-1]} if s else None),
            "tail": (r.stdout + r.stderr)[-2000:] if r.returncode else ""}


MODELS = {
    "openfold3": {"kind": "predict", "model": "openfold3",
                  "fixture": "perf/size512/fixtures/cdk2x2_128.yaml"},
    "protenix-v2": {"kind": "predict", "model": "protenix-v2",
                    "fixture": "perf/size512/fixtures/cdk2x2_128.yaml"},
    "af2-ig": {"kind": "af2",
               "fixture": "scripts/af2_port/parity_artifacts/designpop_bg119/binder_complex.pdb"},
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--card", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--python", default="/home/ttuser/tt-bio-dev/env/bin/python3")
    ap.add_argument("--quiet", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    trees = {"before": Path(a.before), "after": Path(a.after)}
    commits = {k: subprocess.run(["git", "-C", str(v), "rev-parse", "HEAD"], capture_output=True,
                                 text=True).stdout.strip() for k, v in trees.items()}
    report = {"what": __doc__.strip().splitlines()[0], "host": socket.gethostname(),
              "card": int(a.card), "commits": commits, "host_quiet": a.quiet, "models": {}}
    for m in a.models.split(","):
        runs = []
        for i, side in enumerate(("before", "after", "before", "after")):
            r = fold(a.python, trees[side], MODELS[m], Path(a.workdir) / m / f"{side}{i}", a.card)
            r["side"] = side
            runs.append(r)
            print(f"INFAB {m} {side} rc={r['rc']} {r['seconds']}s clk={r['aiclk_mhz_sampled_DURING']} "
                  f"{r['digests']}", flush=True)
        ok = all(r["rc"] == 0 for r in runs)
        d = [json.dumps(r["digests"], sort_keys=True) for r in runs]
        cen = [p for r in runs for p in r["census"]]
        fired = [(p.get("EXACT_SOFTMAX_STATS") or {}, p.get("EXACT_LAYER_NORM_STATS") or {})
                 for p in cen]
        report["models"][m] = {
            "fixture": MODELS[m]["fixture"], "all_folds_succeeded": ok,
            "AA_before_identical": ok and d[0] == d[2],
            "AA_after_identical": ok and d[1] == d[3],
            "AB_identical": ok and d[0] == d[1] == d[2] == d[3],
            "digests": runs[0]["digests"],
            "autograd_imported_any": any(p.get("autograd_imported") for p in cen),
            "exact_counters_all_zero": all(not any(s.values()) and not any(l.values())
                                           for s, l in fired),
            "raw_ops_at_exit": sorted({json.dumps(p.get("raw_ops")) for p in cen}),
            "census_processes": len(cen),
            "seconds": {r["side"] + str(i): r["seconds"] for i, r in enumerate(runs)},
            "aiclk": {r["side"] + str(i): r["aiclk_mhz_sampled_DURING"] for i, r in enumerate(runs)},
            "runs": runs,
        }
        Path(a.out).write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
