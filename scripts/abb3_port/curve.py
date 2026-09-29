#!/usr/bin/env python3
"""The learning curve the capped leg reports, read out of the run's own append-only history.

    PYTHONPATH=$PWD python3 scripts/abb3_port/curve.py --out runs/base-loss --world 2

Four things the write-up at the cap needs and no eyeball reading of a JSONL gives:

* **the curve**, binned so 30,000 rows fit on a page, with the loss terms beside the total,
  because a total that falls while one term climbs is a different result;
* **the cadence that integrates.** The mean of the between-step wall-clock deltas is what
  multiplies out to a step count; the median under-counts the pauses and over-predicts by
  6.6 % on this run's own first 376 steps. Both are printed, and the restart gaps are excluded
  from the cadence and charged to the leg separately, because a gap is lost grant and not a
  slow step;
* **what the restarts cost**, in steps redone and wall clock lost, from the same rows;
* **whether the replayed steps came back the same.** A resume replays the steps between the
  checkpoint and the death, so those steps are in the file twice. Comparing the two copies is a
  bit-exactness check on every resume the leg took, for free, on data the run already wrote.
  This is the leg's strongest evidence that a reset does not change the result: the 2026-09-19
  host reset replayed steps 684 and 685 and both came back digest-identical.

It shares ``tt_bio.train.history`` with the heartbeat rather than parsing the file again. That
matters: a host reset leaves one line as NUL bytes followed by a complete record, and a reader
that drops it sees a step counter that jumps.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

#: Steps skipped when the cadence is measured. The first step of a process pays for the device
#: open, the model build and the first program compile, and a restart pays it again.
WARM_AFTER = 4

#: A between-step gap longer than this is a restart, not a slow step. The longest legitimate
#: step seen on this leg is 27.7 s across a checkpoint write; a restart is ~3 minutes.
GAP_SECONDS = 120.0


def dedupe(rows: list) -> tuple:
    """One row per step, plus what the replayed copies say about the resumes.

    Last write wins. A replayed step is the same arithmetic on restored weights, so the copies
    should be identical; where they are not, the step is named here rather than averaged away.
    """
    by_step: dict = {}
    replays, disagree = 0, []
    for r in rows:
        step = int(r["step"])
        prev = by_step.get(step)
        if prev is not None:
            replays += 1
            if prev.get("digest") != r.get("digest") or prev.get("loss") != r.get("loss"):
                disagree.append({"step": step,
                                 "first": {"loss": prev.get("loss"), "digest": prev.get("digest")},
                                 "again": {"loss": r.get("loss"), "digest": r.get("digest")}})
        by_step[step] = r
    return [by_step[s] for s in sorted(by_step)], {"replayed_steps": replays,
                                                   "disagreements": disagree}


def cadence(rows: list) -> dict:
    """Wall-clock seconds per step over consecutive warm steps, and the gaps charged separately."""
    deltas, gaps = [], []
    for a, b in zip(rows, rows[1:]):
        if b["step"] != a["step"] + 1 or not (a.get("t") and b.get("t")):
            continue
        d = b["t"] - a["t"]
        if d > GAP_SECONDS:
            gaps.append(d)
        else:
            deltas.append((a["step"], d))
    warm = [d for step, d in deltas if step >= WARM_AFTER]
    use = warm or [d for _, d in deltas]
    if not use:
        return {"n": 0}
    return {"n": len(use), "mean": statistics.fmean(use), "median": statistics.median(use),
            "p90": sorted(use)[int(0.9 * (len(use) - 1))], "max": max(use),
            "gap_count": len(gaps), "gap_seconds": sum(gaps)}


def restarts(rows: list) -> list:
    """Where the step counter went backwards, and what each one cost in steps and wall clock.

    Reads the file in its own order, before :func:`dedupe`, because deduping makes the counter
    monotonic and a monotonic counter has no restarts in it to find.
    """
    out = []
    for a, b in zip(rows, rows[1:]):
        if b["step"] > a["step"]:
            continue
        out.append({"died_at": a["step"], "resumed_at": b["step"],
                    "steps_redone": a["step"] - b["step"] + 1,
                    "seconds_lost": round(b["t"] - a["t"], 1) if a.get("t") and b.get("t")
                    else None})
    return out


def bins(rows: list, n: int) -> list:
    """``n`` equal-width bins over the step axis, each carrying the mean of what it holds."""
    if not rows:
        return []
    lo, hi = rows[0]["step"], rows[-1]["step"]
    width = max((hi - lo + 1) / n, 1.0)
    terms = sorted({k for r in rows for k in (r.get("loss_terms") or {})})
    out: list = []
    for r in rows:
        idx = min(int((r["step"] - lo) / width), n - 1)
        while len(out) <= idx:
            out.append({"rows": []})
        out[idx]["rows"].append(r)
    packed = []
    for b in out:
        rs = b["rows"]
        if not rs:
            continue
        row = {"step_from": rs[0]["step"], "step_to": rs[-1]["step"], "n": len(rs),
               "loss": statistics.fmean(r["loss"] for r in rs)}
        for key in ("lr", "grad_norm"):
            vals = [r[key] for r in rs if r.get(key) is not None]
            row[key] = statistics.fmean(vals) if vals else None
        for t in terms:
            vals = [r["loss_terms"][t] for r in rs
                    if (r.get("loss_terms") or {}).get(t) is not None]
            row[t] = statistics.fmean(vals) if vals else None
        packed.append(row)
    return packed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--world", type=int, default=2)
    ap.add_argument("--rank", type=int, default=0, help="which rank's history carries the curve")
    ap.add_argument("--bins", type=int, default=40)
    ap.add_argument("--csv", help="write the binned curve here as well")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    from tt_bio.train import deadline as deadline_mod
    from tt_bio.train.history import read_rows

    out = Path(args.out)
    raw = read_rows(out / f"history-rank{args.rank}.jsonl")
    if not raw:
        print(f"no history in {out}", file=sys.stderr)
        return 2
    rows, replay = dedupe(raw)
    ckpt = out / "checkpoints"
    checkpoints = sorted(int(p.stem.split("-")[-1])
                         for p in ckpt.glob("step-*.safetensors")) if ckpt.is_dir() else []
    cad = cadence(rows)
    span = (rows[-1]["t"] - rows[0]["t"]) if rows[0].get("t") and rows[-1].get("t") else None
    realized = (span / (rows[-1]["step"] - rows[0]["step"])
                if span and rows[-1]["step"] > rows[0]["step"] else None)
    rec = deadline_mod.read(out)
    remaining = deadline_mod.remaining(out) if rec else 0.0

    state = {
        "out": str(out), "rank": args.rank,
        "rows_parsed": len(raw), "distinct_steps": len(rows),
        "first_step": rows[0]["step"], "last_step": rows[-1]["step"],
        "wall_seconds": round(rows[-1]["t"] - rows[0]["t"], 1)
        if rows[0].get("t") and rows[-1].get("t") else None,
        "cadence_seconds": cad, "restarts": restarts(raw), "replay": replay,
        # What the leg ACTUALLY achieved: every restart gap and every redone step charged to it.
        # The cadence above is the clean step; this is the one that projects a step count.
        "realized_seconds_per_step": realized,
        "checkpoints": checkpoints,
        "loss_first": rows[0]["loss"], "loss_last": rows[-1]["loss"],
        "lr_last": rows[-1].get("lr"), "deadline": rec,
        "remaining_hours": round(remaining / 3600.0, 2),
        # Two projections, because they answer different questions and the gap between them is
        # what the restarts cost. The clean one assumes no further restart; the realized one
        # assumes the leg keeps taking them at the rate it has taken them so far.
        "projected_step_at_cap": int(rows[-1]["step"] + remaining / cad["mean"])
        if cad.get("mean") else None,
        "projected_step_at_cap_realized": int(rows[-1]["step"] + remaining / realized)
        if realized else None,
        "curve": bins(rows, args.bins),
    }
    if args.csv and state["curve"]:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(state["curve"][0]))
            w.writeheader()
            w.writerows(state["curve"])

    if args.json:
        print(json.dumps(state, indent=2))
        return 0

    print(f"CURVE  {out} rank {args.rank}")
    print(f"  steps {state['first_step']}..{state['last_step']}, {state['distinct_steps']} "
          f"distinct from {state['rows_parsed']} rows "
          f"({state['rows_parsed'] - state['distinct_steps']} replayed)")
    if cad.get("n"):
        print(f"  cadence {cad['mean']:.3f} s/step mean, {cad['median']:.3f} median, "
              f"{cad['p90']:.3f} p90, {cad['max']:.3f} max, n={cad['n']}")
        print(f"  restart gaps excluded from it: {cad['gap_count']}, "
              f"{cad['gap_seconds'] / 60.0:.1f} min of grant")
    for r in state["restarts"]:
        print(f"  restart: died@{r['died_at']} -> {r['resumed_at']}, "
              f"{r['steps_redone']} steps redone, {r['seconds_lost']} s")
    d = replay["disagreements"]
    print(f"  replayed steps {replay['replayed_steps']}, "
          + ("ALL BIT-IDENTICAL on loss and digest" if not d
             else f"{len(d)} DISAGREE"))
    for x in d[:5]:
        print(f"    step {x['step']}: loss {x['first']['loss']} vs {x['again']['loss']}, "
              f"digest {x['first']['digest']} vs {x['again']['digest']}")
    print(f"  loss {state['loss_first']:.6f} -> {state['loss_last']:.6f}, "
          f"lr {state['lr_last']}")
    if realized:
        print(f"  realized {realized:.3f} s/step end to end, every restart charged to it")
    print(f"  grant {state['remaining_hours']} h left, projected step at cap "
          f"{state['projected_step_at_cap']} at the clean cadence, "
          f"{state['projected_step_at_cap_realized']} at the realized rate")
    head = state["curve"]
    if head:
        keys = [k for k in head[0] if k not in ("step_from", "step_to", "n")]
        print("  " + " ".join(f"{k:>12}" for k in ["steps", *keys]))
        for b in head:
            cells = [f"{b[k]:>12.6g}" if isinstance(b.get(k), float) else f"{'':>12}"
                     for k in keys]
            print(f"  {b['step_from']}-{b['step_to']:<6} " + " ".join(cells))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException:
        import traceback
        traceback.print_exc()
        print("CURVE: BROKEN  the reader failed and says nothing about the run", file=sys.stderr)
        raise SystemExit(2)
