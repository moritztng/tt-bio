#!/usr/bin/env python3
"""AbAg-XM deep-N pilot cost validation (state doc abag-xm-deepn-saturation-fullpanel, PHASE 2).

Reads deepn/progress.jsonl + the queue's job JSONs and answers, per model:
  1. measured wall vs the tier_a-linear projection (wall50 * N/50) -- the pilot gate is
     "validated within 25 pct" before the full panel queues.
  2. two-point fit wall = fixed + N*s using tier_a wall50 and the deepn rung wall(s):
     s = (wallN - wall50) / (N - 50), fixed = wall50 - 50*s. Targets where the fit goes
     negative (host-contention noise) are counted, not silently averaged in.
  3. full-panel projection at 64+256+512 per model from the fitted (fixed, s) medians.

Writes deepn/costfit_pilot.json and prints the same table. CPU-only.
"""
import json, sys
from pathlib import Path

BASE = Path.home() / "abag_xm" / "deepn"
PROGRESS = BASE / "progress.jsonl"
PLAN_INPUTS = BASE / "plan_inputs.json"
RUNG_COSTS = (64, 256, 512)


def load_ok_walls():
    walls = {}  # (model, target, rung) -> best (max) wall
    if not PROGRESS.exists():
        return walls
    for line in PROGRESS.read_text().splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("status") != "ok":
            continue
        k = (r["model"], r["target"], r["rung"])
        walls[k] = max(walls.get(k, 0.0), r["wall_s"])
    return walls


def med(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def main():
    pi = json.loads(PLAN_INPUTS.read_text())
    walls = load_ok_walls()
    proj = {}
    for jf in BASE.glob("queue_pilot/job_*.json"):
        j = json.loads(jf.read_text())
        proj[(j["model"], j["target"], j["rung"])] = j["proj_s"]

    per_model = {}
    for (model, target, rung), wall in sorted(walls.items()):
        wall50 = pi[target]["wall50"].get(model)
        p = proj.get((model, target, rung))
        if wall50 is None or p is None:
            continue
        d = per_model.setdefault(model, {"ratio": [], "s": [], "fixed": [], "neg": 0})
        d["ratio"].append(wall / p)
        s = (wall - wall50) / (rung - 50)
        if s <= 0:
            d["neg"] += 1
        else:
            d["s"].append(s)
            d["fixed"].append(wall50 - 50 * s)

    out = {"n_folds": len(walls), "models": {}}
    print(f"{'model':<13} {'n':>3} {'ratio med':>9} {'ratio max':>9} {'within25':>8} "
          f"{'s/samp':>7} {'fixed':>7} {'neg':>4} {'panel card-h':>12}")
    for model, d in sorted(per_model.items()):
        r = d["ratio"]
        within = sum(1 for x in r if abs(x - 1) <= 0.25)
        line = {"n": len(r), "ratio_med": round(med(r), 3), "ratio_max": round(max(r), 3),
                "within25": f"{within}/{len(r)}", "neg_fit": d["neg"]}
        if d["s"]:
            s_med, f_med = med(d["s"]), med(d["fixed"])
            panel = sum(f_med + n * s_med for n in RUNG_COSTS) * 164 / 3600
            line.update({"s_per_sample": round(s_med, 2), "fixed_s": round(f_med, 0),
                         "panel_card_h": round(panel, 1)})
            print(f"{model:<13} {len(r):>3} {med(r):>9.3f} {max(r):>9.3f} "
                  f"{within}/{len(r):<7} {s_med:>7.2f} {f_med:>7.0f} {d['neg']:>4} "
                  f"{panel:>12.1f}")
        else:
            print(f"{model:<13} {len(r):>3} {med(r):>9.3f} {max(r):>9.3f} "
                  f"{within}/{len(r):<7} {'--':>7} {'--':>7} {d['neg']:>4} {'--':>12}")
        out["models"][model] = line
    (BASE / "costfit_pilot.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote {BASE / 'costfit_pilot.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
