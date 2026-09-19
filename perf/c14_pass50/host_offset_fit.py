#!/usr/bin/env python3
"""Is the size-ladder red a scaling regression, or one constant added to every fold?

The 2026-09-19 01:59Z size-ladder run reds boltz2's 256->512 exponent at 0.75 against a
baseline of 1.42. A scaling regression and a constant per-fold overhead look identical at
one rung and completely different across six, so this fits the constant and asks whether
it explains the exponents on its own.

A constant c added to every fold leaves large rungs almost unchanged and crushes the
exponent between small ones, because the exponent is a ratio: at 256->512 a fold of 4.1 s
and one of 11.0 s differ by 2.68x, but 4.1+c and 11.0+c differ by only 1.70x at c=5.7.
That is the entire red. So: fit ONE c over every rung of every scored model by least
squares (which for this residual is just the mean offset), then recompute every exponent
from baseline+c and compare it to what the run actually measured.

If baseline+c reproduces the measured exponents, the run measured a slower HOST, not a
worse scaling law, and the arm's red is an artifact of the box it folded on.
"""
import json
import glob
import math
from pathlib import Path

WORK = Path("perf/c14_pass50/ladder_work")
BASE = Path("perf/c14_pass50/baseline/size_ladder_baseline.d")
CARD = "p300c"
RUNGS = (256, 512, 640, 768, 896, 1024)


def measured(model, rung, tag="rep0"):
    for f in sorted(glob.glob(str(WORK / f"out_{model}-{rung}-{tag}" / "*" / "results.json"))):
        rows = json.loads(Path(f).read_text())
        ts = [r["runtime_s"] for r in rows
              if r.get("status") == "ok" and r.get("runtime_s") is not None]
        if ts:
            return max(ts)
    return None


def main():
    pairs, per_model = [], {}
    for shard in sorted(BASE.glob("*.json")):
        model = shard.stem
        blk = (json.loads(shard.read_text()).get("cards") or {}).get(CARD) or {}
        entry = (blk.get("models") or {}).get(model)
        if not entry:
            continue
        base = entry.get("runtime_s") or {}
        rows = []
        for rung in RUNGS:
            b, m = base.get(str(rung)), measured(model, rung)
            if b is None or m is None:
                continue
            rows.append((rung, b, m))
            pairs.append((b, m))
        if rows:
            per_model[model] = (entry, rows)
    if not pairs:
        print("no scored rungs on disk")
        return

    # Least squares for residual (b + c - m) is the mean offset.
    c = sum(m - b for b, m in pairs) / len(pairs)
    resid = math.sqrt(sum((b + c - m) ** 2 for b, m in pairs) / len(pairs))
    print(f"fitted ONE constant over {len(pairs)} rungs of {len(per_model)} models: "
          f"c = {c:+.2f} s per fold, rms residual {resid:.2f} s")
    # A multiplicative alternative, scored the same way, so the shapes compete fairly.
    k = sum(m / b for b, m in pairs) / len(pairs)
    kresid = math.sqrt(sum((b * k - m) ** 2 for b, m in pairs) / len(pairs))
    print(f"the multiplicative alternative:               k = {k:.3f}x,      "
          f"rms residual {kresid:.2f} s")
    print()

    for model, (entry, rows) in per_model.items():
        print(f"=== {model}: baseline recorded {entry.get('recorded')} "
              f"on {entry.get('host')} @ {entry.get('commit')}, grid {entry.get('grid')} ===")
        print(f"{'rung':>6} {'base':>7} {'run':>7} {'offset':>8} {'base+c':>8} {'err':>7}")
        for rung, b, m in rows:
            print(f"{rung:>6} {b:>7.1f} {m:>7.1f} {m-b:>+8.1f} {b+c:>8.1f} {b+c-m:>+7.1f}")
        base = {r: b for r, b, _ in rows}
        run = {r: m for r, _, m in rows}
        declared = entry.get("exponents") or {}
        print(f"  {'interval':>12} {'baseline':>9} {'base+c':>8} {'measured':>9} {'tol':>5}"
              f"  verdict")
        for iv, spec in declared.items():
            lo, hi = (int(x) for x in iv.split("->"))
            if lo not in base or hi not in base or lo not in run or hi not in run:
                continue
            ratio = math.log(hi / lo)
            k_base = math.log(base[hi] / base[lo]) / ratio
            k_off = math.log((base[hi] + c) / (base[lo] + c)) / ratio
            k_run = math.log(run[hi] / run[lo]) / ratio
            tol = spec.get("tol", 0.5)
            red = abs(k_run - k_base) > tol
            # Does the constant alone explain the move the arm reds on?
            explained = abs(k_run - k_off) <= tol
            verdict = ("RED, and base+c explains it" if red and explained else
                       "RED, unexplained by c" if red else "in tolerance")
            print(f"  {iv:>12} {k_base:>9.3f} {k_off:>8.3f} {k_run:>9.3f} {tol:>5.2f}"
                  f"  {verdict}")
        print()


if __name__ == "__main__":
    main()


def spread():
    """How far apart are two folds of the SAME shape, minutes apart, on this host?

    A constant that lives in the CODE is paid identically by every fold, so it cannot be
    told from contention by the offset alone. It can be told apart by the noise: the
    baseline recorded sigma_runtime_512 = 0.41 % on a quiet box, so if the warm-up and
    rep0 folds of one rung now disagree by many times that, the box moved between them and
    the offset is the box. Warm-up carries first-fold cost so this is not an A/A floor and
    is not read as one; a 10x inflation in short-interval spread is still the signature.
    """
    print("=== same shape, two folds, minutes apart ===")
    print(f"{'model':>10} {'rung':>6} {'warmup':>8} {'rep0':>7} {'spread':>8}   "
          f"baseline sigma at 512")
    for shard in sorted(BASE.glob("*.json")):
        model = shard.stem
        blk = (json.loads(shard.read_text()).get("cards") or {}).get(CARD) or {}
        entry = (blk.get("models") or {}).get(model)
        if not entry:
            continue
        sig = entry.get("sigma_runtime_512")
        for rung in RUNGS:
            w, r = measured(model, rung, "warmup"), measured(model, rung, "rep0")
            if w is None or r is None:
                continue
            rel = abs(w - r) / min(w, r)
            note = ""
            if rung == 512 and sig:
                note = f"{sig:.4f} -> {rel/sig:.0f}x inflated"
            print(f"{model:>10} {rung:>6} {w:>8.1f} {r:>7.1f} {rel*100:>7.1f}%   {note}")


spread()
