#!/usr/bin/env python3
"""What the shipped accurate-softmax default costs Protenix-v2 and OpenDDE at 512 and 768 aa.

Pass 3 measured this and pass 4 could not cite it: the numbers live in the p3 state doc, the
harness and its artifacts lived in the p3 worktree, and that worktree is gone. So this is the same
question asked again on committed code, writing a committed artifact.

Two things it does NOT do, both deliberate:

  * it does not go through ``release_gate.py --model size-ladder``. That leg records exactly one
    card type (``p150a``) in ``docs/size_ladder_baseline.json`` and refuses to run on anything
    else, so on a p300c box it exits in one second without folding. Recording a second card type
    is seeding a new gate key and is not this task's call. The question here is narrower than the
    ladder's anyway: not "did any lever drift", just "what does this one default cost".
  * it does not toggle the lever inside one process. ``accurate_softmax_site`` is read in
    ``__init__``, so an arm is a construction, not a call. Each arm is its own fold subprocess with
    its own env, which also means no arm can inherit the previous one's device state.

Protocol, from the p3 pass that got a 20 aa cell wrong twice:

  * ``runtime_s`` comes from the fold's own results.json, which excludes model load and process
    startup. Wall clock would price the checkpoint read.
  * the first fold at each (model, rung) is discarded. It pays kernel compile; every ladder rung
    that skipped this read its own cold cost as the lever's.
  * arms interleave OFF/ON/OFF/ON rather than running in blocks, so a thermal or host drift over
    the cell hits both arms.
  * the A/A floor (the spread of the same-arm reps) is printed BEFORE the A/B delta and is the
    thing that decides whether the delta is a measurement at all. p3's 20 aa cell read -14.2% and
    +1.7% for two models purely from host co-tenancy; both were inside their own floor.

Run it alone on the box. Every fold here is the measurement.
"""
import argparse, json, os, statistics as st, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

FIXTURES = ROOT / "perf" / "size512" / "fixtures"
STEPS, SEED = 6, 0


def one_fold(model: str, rung: int, arm: str, workdir: Path, rep: int) -> dict:
    """One fold. arm 'on' = shipped defaults, 'off' = every site forced off."""
    from tt_bio.main import predict_results_dir_name
    fixture = FIXTURES / f"cdk2x2_{rung}.yaml"
    if not fixture.exists():
        return {"error": f"missing fixture {fixture}"}
    tag = f"{model}-{rung}-{arm}-{rep}"
    out_dir = workdir / f"out_{tag}"
    env = dict(os.environ)
    if arm == "off":
        env["TT_BIO_ACCURATE_SOFTMAX_AB"] = "-all"
    else:
        env.pop("TT_BIO_ACCURATE_SOFTMAX_AB", None)
    cmd = [sys.executable, "-m", "tt_bio.main", "predict", str(fixture),
           "--model", model, "--single_sequence", "--sampling_steps", str(STEPS),
           "--diffusion_samples", "1", "--seed", str(SEED), "--out_dir", str(out_dir)]
    log = workdir / f"{tag}.log"
    t0 = time.monotonic()
    with open(log, "w") as fp:
        rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=fp,
                            stderr=subprocess.STDOUT).returncode
    wall = time.monotonic() - t0
    if rc != 0:
        tail = "".join(log.read_text(errors="replace").splitlines(True)[-3:]).strip()
        return {"error": f"fold exited {rc}: {tail}"}
    results = out_dir / predict_results_dir_name(model, fixture.stem) / "results.json"
    try:
        rows = json.loads(results.read_text())
        ts = [r["runtime_s"] for r in rows
              if r.get("status") == "ok" and r.get("runtime_s") is not None]
    except Exception as e:
        return {"error": f"no readable results.json: {e}"}
    if not ts:
        return {"error": "fold ok but results.json carries no runtime_s"}
    return {"runtime_s": max(ts), "wall": wall}


def cell(model: str, rung: int, reps: int, workdir: Path) -> dict:
    print("\n=== %s @ %d aa ===" % (model, rung), flush=True)
    warm = one_fold(model, rung, "on", workdir, rep=0)
    if "error" in warm:
        print("  warm-up FAILED: %s" % warm["error"], flush=True)
        return {"model": model, "rung": rung, "error": warm["error"]}
    print("  warm-up (discarded)  %.4fs" % warm["runtime_s"], flush=True)
    off, on = [], []
    for rep in range(1, reps + 1):
        for arm, acc in (("off", off), ("on", on)):
            r = one_fold(model, rung, arm, workdir, rep)
            if "error" in r:
                print("  %s rep%d FAILED: %s" % (arm, rep, r["error"]), flush=True)
                return {"model": model, "rung": rung, "error": r["error"],
                        "off": off, "on": on}
            acc.append(r["runtime_s"])
            print("  %-3s rep%d  %.4fs" % (arm, rep, r["runtime_s"]), flush=True)
    # Floor first: the spread of the same-arm reps is what makes the delta a measurement.
    aa = 100.0 * (max(off) - min(off)) / st.median(off)
    ab = 100.0 * (st.median(on) - st.median(off)) / st.median(off)
    verdict = "INSIDE THE FLOOR" if abs(ab) <= aa else "outside the floor"
    print("  A/A floor %+.3f%%   A/B %+.3f%%   %s" % (aa, ab, verdict), flush=True)
    return {"model": model, "rung": rung, "off": off, "on": on,
            "off_median": st.median(off), "on_median": st.median(on),
            "aa_spread_pct": aa, "ab_median_pct": ab, "inside_aa": abs(ab) <= aa}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="protenix-v2,opendde")
    ap.add_argument("--rungs", default="512,768")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--workdir", default="/tmp/xmflip")
    ap.add_argument("--out", default=str(ROOT / "perf/xmsoftmax/results/fold_ab_flip.json"))
    a = ap.parse_args()
    workdir = Path(a.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    cells = []
    for rung in [int(r) for r in a.rungs.split(",")]:
        for model in a.models.split(","):
            cells.append(cell(model, rung, a.reps, workdir))
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(
                {"what": "cost of the shipped accurate-softmax default, per model per rung",
                 "metric": "results.json runtime_s, model load and startup excluded",
                 "steps": STEPS, "seed": SEED, "samples": 1, "single_sequence": True,
                 "fixture": "perf/size512/fixtures/cdk2x2_<rung>.yaml",
                 "off_arm": "TT_BIO_ACCURATE_SOFTMAX_AB=-all", "on_arm": "shipped defaults",
                 "cells": cells}, indent=2) + "\n")
    print("\nmodel            rung   off med    on med     A/A       A/B", flush=True)
    for c in cells:
        if "off_median" not in c:
            print("%-16s %-6d FAILED: %s" % (c["model"], c["rung"], c["error"][:60]), flush=True)
        else:
            print("%-16s %-6d %-10.4f %-10.4f %+7.3f%%  %+7.3f%%"
                  % (c["model"], c["rung"], c["off_median"], c["on_median"],
                     c["aa_spread_pct"], c["ab_median_pct"]), flush=True)
    print("\nwrote %s" % a.out, flush=True)
    return 1 if any("off_median" not in c for c in cells) else 0


if __name__ == "__main__":
    sys.exit(main())
