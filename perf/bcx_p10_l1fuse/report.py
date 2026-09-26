#!/usr/bin/env python3
"""The arm-by-arm round table for `perf/bcx_p10_l1fuse/ab.sh`.

`perf/bcx_p10_rneker/report.py`'s method unchanged -- the bounds (one `sequence_gradients` entry
to the next), the forward/backward device split, the warm-median rule that drops the first round
of every PROCESS as its compile round, the rank pairing and the position pairing. Two things are
this row's:

**The arm is read off the ENGINE, not off a stamp.** `meter.lever_reach` writes
`lever_grad_fanin_l1` into every round boundary from `fanin_l1.FANIN_L1` itself, so a round is
labelled by what was running during it rather than by what the process was asked for. That is
what the first A/B sitting could not do and why it was thrown away.

**Reach is per round.** `fanin_l1_served` and each named decline are cumulative counters stamped
at every boundary, so the round's own reach is the difference across it. An on arm whose served
count stops moving has gone inert mid-arm, and the table shows that instead of averaging it into
a median.
"""
import json
import statistics as st
import sys

SERVED = "fanin_l1_served"
DECLINED = "fanin_l1_declined: over the L1 budget"
SPILLED = "fanin_l1_spilled: the allocator refused L1"


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
        before = starts[i].get("reach") or {}
        after = (starts[i + 1] if i + 1 < len(starts) else (stop[0] if stop else {})).get("reach") or {}
        delta = lambda k: (after.get(k, 0) - before.get(k, 0)) if after else None  # noqa: E731
        # The engine's own answer, at the boundary that opened this round.
        arm = "on" if before.get("lever_grad_fanin_l1") else "off"
        out.append({"round": starts[i]["round"], "arm": arm,
                    "wall": round(t1 - t0, 3), "fwd": round(fwd, 3), "bwd": round(bwd, 3),
                    "dev": round(fwd + bwd, 3),
                    "clkmed": samples[len(samples) // 2] if samples else None,
                    "clkmin": samples[0] if samples else None,
                    "load1": round(starts[i]["load1"], 1) if starts[i].get("load1") else None,
                    "served": delta(SERVED), "declin": delta(DECLINED),
                    "spill": delta(SPILLED)})
    return d, out


def med(xs):
    return round(st.median(xs), 3) if xs else None


def main(*paths):
    rows, d = [], None
    for i, path in enumerate(paths):
        d, rs = rounds(path)
        for r in rs:
            r["proc"] = i
            r["round"] = r["round"] + 100 * i
        rows += rs
    print(f"{'rnd':>4} {'arm':<4} {'wall':>8} {'dev':>7} {'fwd':>7} {'bwd':>7} "
          f"{'clkmed':>7} {'clkmin':>7} {'load1':>6} {'served':>7} {'declin':>7} {'spill':>6}")
    for r in rows:
        print(f"{r['round']:>4} {r['arm']:<4} {r['wall']:>8.3f} {r['dev']:>7.3f} "
              f"{r['fwd']:>7.3f} {r['bwd']:>7.3f} {str(r['clkmed']):>7} {str(r['clkmin']):>7} "
              f"{str(r['load1']):>6} {str(r['served']):>7} {str(r['declin']):>7} "
              f"{str(r['spill']):>6}")

    arms, order, seen = {}, [], set()
    for r in rows:
        if (r["arm"], r["proc"]) in seen:
            arms.setdefault(r["arm"], []).append(r)     # warm
        else:
            seen.add((r["arm"], r["proc"]))             # this process's compile round
        if r["arm"] not in order:
            order.append(r["arm"])

    # A lever that went inert mid-arm is the failure this campaign pays for most often, and it
    # is invisible in a median. Say it before the medians, not after.
    for arm in order:
        warm = arms.get(arm, [])
        srv = [r["served"] for r in warm if r["served"] is not None]
        spl = [r["spill"] or 0 for r in warm]
        if arm == "on" and srv and min(srv) == 0:
            print(f"\n!! the on arm served ZERO in {sum(1 for s in srv if s == 0)} of "
                  f"{len(srv)} warm rounds -- it went inert and the median is two arms blended")
        if arm == "on" and srv:
            print(f"\non-arm reach per warm round: served min {min(srv)} med {med(srv)} "
                  f"max {max(srv)}; declined-over-budget med "
                  f"{med([r['declin'] for r in warm if r['declin'] is not None])}; "
                  f"allocator spills total {sum(spl)}")
        if arm == "off" and srv and max(srv) > 0:
            print(f"\n!! the OFF arm served {max(srv)} -- the arms are not separated")

    print("\nwarm medians, the first round of every PROCESS dropped as its compile round")
    print(f"{'arm':<4} {'n':>3} {'wall':>8} {'dev':>8} {'fwd':>8} {'bwd':>8} "
          f"{'dev min':>8} {'dev max':>8}")
    meds = {}
    for arm in order:
        warm = arms.get(arm, [])
        if not warm:
            continue
        m = {k: med([r[k] for r in warm]) for k in ("wall", "dev", "fwd", "bwd")}
        meds[arm] = (warm, m)
        devs = [r["dev"] for r in warm]
        print(f"{arm:<4} {len(warm):>3} {m['wall']:>8.3f} {m['dev']:>8.3f} {m['fwd']:>8.3f} "
              f"{m['bwd']:>8.3f} {min(devs):>8.3f} {max(devs):>8.3f}")
    if meds:
        b = meds[order[0]][1]
        print(f"\nagainst {order[0]}:")
        for arm in order:
            if arm not in meds:
                continue
            m = meds[arm][1]
            print(f"  {arm:<4} dev {b['dev'] / m['dev']:.4f}x  fwd {b['fwd'] / m['fwd']:.4f}x  "
                  f"bwd {b['bwd'] / m['bwd']:.4f}x  wall {b['wall'] / m['wall']:.4f}x")
        if len(order) == 2 and all(a in meds for a in order):
            a, c = (sorted(r["dev"] for r in meds[o][0]) for o in order)
            print(f"\n  {order[0]} dev range [{a[0]:.3f}, {a[-1]:.3f}]  "
                  f"{order[1]} dev range [{c[0]:.3f}, {c[-1]:.3f}]  "
                  f"overlap: {'YES' if a[0] <= c[-1] and c[0] <= a[-1] else 'NO'}")
            wins = sum(1 for x, y in zip(a, c) if y < x)
            print(f"  paired by rank, {order[1]} is faster on {wins} of {min(len(a), len(c))}")

    if len(order) == 2:
        by_pos = {}
        for r in rows:
            by_pos.setdefault((r["arm"], r["round"] % 100), []).append(r["dev"])
        a, b2 = order
        positions = sorted({k[1] for k in by_pos if k[0] == a} & {k[1] for k in by_pos if k[0] == b2})
        positions = [q for q in positions if q != 1]        # each process's compile round
        print(f"\nposition-paired device seconds ({a} vs {b2}, mean over processes):")
        print(f"{'pos':>4} {a:>9} {b2:>9} {'delta':>8} {'ratio':>8}")
        deltas = []
        for q in positions:
            xa = sum(by_pos[(a, q)]) / len(by_pos[(a, q)])
            xb = sum(by_pos[(b2, q)]) / len(by_pos[(b2, q)])
            deltas.append((q, xa, xb, xa - xb))
            print(f"{q:>4} {xa:>9.3f} {xb:>9.3f} {xa - xb:>8.3f} {xa / xb:>8.4f}x")
        wins = sum(1 for _, _, _, dd in deltas if dd > 0)
        mean = sum(dd for _, _, _, dd in deltas) / len(deltas)
        print(f"  {b2} faster at {wins} of {len(deltas)} positions, mean delta {mean:.3f} s")
        stable = [t for t in deltas if t[1] < 10.0 and t[2] < 10.0]
        if stable:
            ms = sum(dd for _, _, _, dd in stable) / len(stable)
            base = sum(x for _, x, _, _ in stable) / len(stable)
            sw = sum(1 for _, _, _, dd in stable if dd > 0)
            print(f"  restricted to the {len(stable)} positions where BOTH arms are under 10 s: "
                  f"{b2} faster at {sw} of {len(stable)}, mean delta {ms:.3f} s on "
                  f"{base:.3f} s = {base / (base - ms):.4f}x")

    print("\nstamp:", json.dumps({k: d["stamp"].get(k) for k in
                                 ("commit", "host", "card", "exact", "shipped_pool",
                                  "binder_pinned", "extra_msa_on_device", "template_on_device",
                                  "triatt_hifi", "taped_kernels", "rne_kernel",
                                  "levers_expected")}))


if __name__ == "__main__":
    main(*sys.argv[1:])
