#!/usr/bin/env python3
"""The arm-by-arm round table `perf/bcx_round/analyze.py` does not split.

Same bounds as `analyze.py` -- a round runs from one `sequence_gradients` entry to the next, so
the last collected round has no closing boundary unless a `round_stop` was written. Device time
is split forward (`_taped`) and backward (`_backward`) because the two answer different
questions: the forward is which kernel serves, the backward is which node the tape built.

Every row carries the AICLK sampled DURING the round and the 1-minute load, because a round
measured below ~1200 MHz is an artifact and a round with no clock beside it is not a measurement.
"""
import json
import statistics as st
import sys


def rounds(path):
    d = json.load(open(path))
    ev, clk = d["events"], d["aiclk"]
    starts = [e for e in ev if e["kind"] == "round_start"]
    stop = [e for e in ev if e["kind"] == "round_stop"]
    bounds = [e["t0"] for e in starts] + ([stop[0]["t0"]] if stop else [])
    out = []
    for i in range(len(bounds) - 1):
        t0, t1 = bounds[i], bounds[i + 1]
        inside = [e for e in ev if e.get("t0") is not None and e.get("t1") is not None
                  and e["t0"] >= t0 - 1e-6 and e["t1"] <= t1 + 1e-6]
        fwd = sum(e["dt"] for e in inside if e["kind"] == "device" and e["phase"] == "taped")
        bwd = sum(e["dt"] for e in inside if e["kind"] == "device" and e["phase"] == "backward")
        samples = sorted(c for t, c, _ in clk if t0 <= t <= t1)
        before = starts[i].get("reach_before") or {}
        after = starts[i].get("reach_after") or {}
        out.append({"round": starts[i]["round"], "arm": starts[i].get("arm"),
                    "wall": round(t1 - t0, 3), "fwd": round(fwd, 3), "bwd": round(bwd, 3),
                    "dev": round(fwd + bwd, 3),
                    "clkmed": samples[len(samples) // 2] if samples else None,
                    "clkmin": samples[0] if samples else None,
                    "load1": starts[i].get("load1"),
                    "served": _delta(before, after, "fused", "served"),
                    "taped": _delta(before, after, "fused", "taped"),
                    "fp32": (after.get("fp32_softmax_calls", 0)
                             - before.get("fp32_softmax_calls", 0)) or None,
                    "rp": _pair(before, after, "reblock"),
                    "rpb": _pair(before, after, "reblock_back")})
    return d, out


def _delta(before, after, group, key):
    if not after:
        return None
    return after.get(group, {}).get(key, 0) - before.get(group, {}).get(key, 0)


def _pair(before, after, key):
    if not after:
        return None
    a, b = after.get(key, [0, 0]), before.get(key, [0, 0])
    return [a[0] - b[0], a[1] - b[1]]


def med(xs):
    return round(st.median(xs), 3) if xs else None


def main(path):
    d, rows = rounds(path)
    print(f"{'rnd':>4} {'arm':<6} {'wall':>8} {'dev':>7} {'fwd':>7} {'bwd':>7} "
          f"{'clkmed':>7} {'clkmin':>7} {'load1':>6}  {'served':>6} {'taped':>6} {'fp32':>5} "
          f"{'rp':>10} {'rpb':>10}")
    for r in rows:
        print(f"{r['round']:>4} {str(r['arm']):<6} {r['wall']:>8.3f} {r['dev']:>7.3f} "
              f"{r['fwd']:>7.3f} {r['bwd']:>7.3f} {str(r['clkmed']):>7} {str(r['clkmin']):>7} "
              f"{str(r['load1']):>6}  {str(r['served']):>6} {str(r['taped']):>6} "
              f"{str(r['fp32']):>5} {str(r['rp']):>10} {str(r['rpb']):>10}")
    arms = {}
    for r in rows:
        arms.setdefault(r["arm"], []).append(r)
    print("\nwarm medians, first round of each arm dropped as its compile round")
    print(f"{'arm':<6} {'n':>3} {'wall':>8} {'dev':>8} {'fwd':>8} {'bwd':>8}")
    base = None
    for arm, rs in arms.items():
        warm = rs[1:]
        if not warm:
            continue
        m = {k: med([r[k] for r in warm]) for k in ("wall", "dev", "fwd", "bwd")}
        arms[arm] = (warm, m)
        if base is None:
            base = m
        print(f"{arm:<6} {len(warm):>3} {m['wall']:>8.3f} {m['dev']:>8.3f} "
              f"{m['fwd']:>8.3f} {m['bwd']:>8.3f}")
    if base:
        print(f"\nagainst {list(arms)[0]}:")
        for arm, v in arms.items():
            if not isinstance(v, tuple):
                continue
            m = v[1]
            print(f"  {arm:<6} dev {base['dev'] / m['dev']:.3f}x  fwd "
                  f"{base['fwd'] / m['fwd']:.3f}x  bwd {base['bwd'] / m['bwd']:.3f}x  wall "
                  f"{base['wall'] / m['wall']:.3f}x")
    print("\nstamp:", json.dumps({k: d["stamp"].get(k) for k in
                                  ("commit", "host", "card", "exact", "shipped_pool",
                                   "binder_pinned", "length_bucket_size", "arms", "registry")}))


if __name__ == "__main__":
    main(sys.argv[1])
