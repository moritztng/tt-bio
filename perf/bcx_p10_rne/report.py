#!/usr/bin/env python3
"""The arm-by-arm round table for `perf/bcx_p10_rne/round_ab.py`.

Same bounds as `perf/bcx_round/analyze.py`: a round runs from one `sequence_gradients` entry to
the next. Device time splits forward (`taped`) and backward (`backward`) because the residual
runs in both and the fold removes one call from each. Every row carries the AICLK sampled DURING
the round and the 1-minute load, because a round measured below ~1200 MHz is an artifact.

`fold`/`shipped` are the residual calls that took each branch IN that round: the arm is only an
arm if they move.
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
        delta = lambda k: (after.get(k, 0) - before.get(k, 0)) if after else None  # noqa: E731
        out.append({"round": starts[i]["round"], "arm": starts[i].get("arm"),
                    "wall": round(t1 - t0, 3), "fwd": round(fwd, 3), "bwd": round(bwd, 3),
                    "dev": round(fwd + bwd, 3),
                    "clkmed": samples[len(samples) // 2] if samples else None,
                    "clkmin": samples[0] if samples else None,
                    "load1": round(starts[i]["load1"], 1) if starts[i].get("load1") else None,
                    "resid": delta("calls"), "fold": delta("fold"),
                    "shipped": delta("shipped")})
    return d, out


def med(xs):
    return round(st.median(xs), 3) if xs else None


def main(*paths):
    """One or more dumps. Several is the normal case: the arms interleave at the PROCESS
    boundary (`ab.sh`), so each arm is a separate file and the rows are concatenated in the
    order the processes ran."""
    rows, d = [], None
    for i, path in enumerate(paths):
        d, rs = rounds(path)
        for r in rs:
            r["proc"] = i
            r["round"] = r["round"] + 100 * i
        rows += rs
    print(f"{'rnd':>4} {'arm':<6} {'wall':>8} {'dev':>7} {'fwd':>7} {'bwd':>7} "
          f"{'clkmed':>7} {'clkmin':>7} {'load1':>6} {'resid':>6} {'fold':>6} {'shipped':>8}")
    for r in rows:
        print(f"{r['round']:>4} {str(r['arm']):<6} {r['wall']:>8.3f} {r['dev']:>7.3f} "
              f"{r['fwd']:>7.3f} {r['bwd']:>7.3f} {str(r['clkmed']):>7} {str(r['clkmin']):>7} "
              f"{str(r['load1']):>6} {str(r['resid']):>6} {str(r['fold']):>6} "
              f"{str(r['shipped']):>8}")
    arms, order, seen = {}, [], set()
    for r in rows:
        if (r["arm"], r["proc"]) in seen:
            arms.setdefault(r["arm"], []).append(r)     # warm
        else:
            seen.add((r["arm"], r["proc"]))             # this process's compile round
        if r["arm"] not in order:
            order.append(r["arm"])
    print("\nwarm medians, the first round of every PROCESS dropped as its compile round")
    print(f"{'arm':<6} {'n':>3} {'wall':>8} {'dev':>8} {'fwd':>8} {'bwd':>8} "
          f"{'dev min':>8} {'dev max':>8}")
    meds = {}
    for arm in order:
        warm = arms.get(arm, [])
        if not warm:
            continue
        m = {k: med([r[k] for r in warm]) for k in ("wall", "dev", "fwd", "bwd")}
        meds[arm] = (warm, m)
        devs = [r["dev"] for r in warm]
        print(f"{arm:<6} {len(warm):>3} {m['wall']:>8.3f} {m['dev']:>8.3f} {m['fwd']:>8.3f} "
              f"{m['bwd']:>8.3f} {min(devs):>8.3f} {max(devs):>8.3f}")
    if meds:
        b = meds[order[0]][1]
        print(f"\nagainst {order[0]}:")
        for arm in order:
            if arm not in meds:
                continue
            m = meds[arm][1]
            print(f"  {arm:<6} dev {b['dev'] / m['dev']:.4f}x  fwd {b['fwd'] / m['fwd']:.4f}x  "
                  f"bwd {b['bwd'] / m['bwd']:.4f}x  wall {b['wall'] / m['wall']:.4f}x")
        # Overlap is the honest test at this size: a 3 % lever whose ranges overlap is inside
        # the spread whatever the medians say.
        if len(order) == 2 and all(a in meds for a in order):
            a, c = (sorted(r["dev"] for r in meds[o][0]) for o in order)
            print(f"\n  {order[0]} dev range [{a[0]:.3f}, {a[-1]:.3f}]  "
                  f"{order[1]} dev range [{c[0]:.3f}, {c[-1]:.3f}]  "
                  f"overlap: {'YES' if a[0] <= c[-1] and c[0] <= a[-1] else 'NO'}")
            wins = sum(1 for x, y in zip(a, c) if y < x)
            print(f"  paired by rank, {order[1]} is faster on {wins} of {min(len(a), len(c))}")
    # POSITION PAIRING. Every process shows the same shape: rounds 2, 4 and 7 have a forward
    # around 4.5-5.3 s and the rest around 2.6-2.8, on BOTH arms. That is BindCraft 2's own
    # per-round structure, not the lever, and it is what makes the raw ranges overlap. Pairing
    # round k of one arm against round k of the other removes it, and it is the sharpest test
    # this design supports: same position, same trajectory state, different arm.
    if len(order) == 2:
        by_pos = {}
        for r in rows:
            by_pos.setdefault((r["arm"], r["round"] % 100), []).append(r["dev"])
        a, b = order
        positions = sorted({k[1] for k in by_pos if k[0] == a} & {k[1] for k in by_pos if k[0] == b})
        positions = [q for q in positions if q != 1]        # each process's compile round
        print(f"\nposition-paired device seconds ({a} vs {b}, mean over processes):")
        print(f"{'pos':>4} {a:>9} {b:>9} {'delta':>8} {'ratio':>8}")
        deltas = []
        for q in positions:
            xa = sum(by_pos[(a, q)]) / len(by_pos[(a, q)])
            xb = sum(by_pos[(b, q)]) / len(by_pos[(b, q)])
            deltas.append((q, xa, xb, xa - xb))
            print(f"{q:>4} {xa:>9.3f} {xb:>9.3f} {xa - xb:>8.3f} {xa / xb:>8.4f}x")
        wins = sum(1 for _, _, _, d in deltas if d > 0)
        mean = sum(d for _, _, _, d in deltas) / len(deltas)
        print(f"  {b} faster at {wins} of {len(deltas)} positions, mean delta {mean:.3f} s")
        stable = [t for t in deltas if t[1] < 10.0 and t[2] < 10.0]
        if stable:
            ms = sum(d for _, _, _, d in stable) / len(stable)
            base = sum(x for _, x, _, _ in stable) / len(stable)
            sw = sum(1 for _, _, _, d in stable if d > 0)
            print(f"  restricted to the {len(stable)} positions where BOTH arms are under 10 s "
                  f"(the ones without BindCraft's slow forward): {b} faster at {sw} of "
                  f"{len(stable)}, mean delta {ms:.3f} s on {base:.3f} s = {base / (base - ms):.4f}x")
    print("\nstamp:", json.dumps({k: d["stamp"].get(k) for k in
                                 ("commit", "host", "card", "exact", "shipped_pool",
                                  "binder_pinned", "arms", "reach_end", "rne_fold_cast_flag_end",
                                  "rne_residual_flag", "rne_wide_dram_flag", "triatt_hifi",
                                  "rne_fold")}))


if __name__ == "__main__":
    main(*sys.argv[1:])
