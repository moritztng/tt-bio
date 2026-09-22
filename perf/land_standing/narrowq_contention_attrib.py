#!/usr/bin/env python3
"""Attribute a narrow-q A/B's per-leg runtime to what the host was doing during that leg.

The A/B json records t_start/t_end per fold; the contention jsonl samples loadavg, per-card AICLK
and the top processes at 1 s. Joining them says whether a leg's number is a measurement or an
artifact. Written because the 2026-09-22 02:10Z run reported a 9.707 % A/A floor at 896 aa and a
failing 768 aa negative control, and neither is a statement about the lever.

CORRECTED 2026-09-22 21:45Z. An earlier version of this file concluded the 9.707 % floor was
intrinsic to the harness shape -- a cold subprocess per leg paying its own kernel compilation
inside the timed window. That is REFUTED by this repo's own artifact:
`perf/land_standing/out/narrowq_rf3_896_qb2c1.json` runs the SAME harness, `fold_ab_flip.py`, on
rf3 at 896 aa, and its two quiet reps read an A/A floor of **1.570 %**. Same harness, same size,
same host, a sixth of the floor. So the floor is the BOX, not the harness, and the earlier
conclusion was a mechanism invented to fit one number without checking the control that was
already on disk.

  python3 perf/land_standing/narrowq_contention_attrib.py <contention.jsonl> <ab.json> [<ab.json>...]
"""
import json, statistics as st, sys

# Everything the harness itself spawns. fold_ab_flip runs each leg as its own
# `-m tt_bio.main predict` subprocess (fold_ab_flip.py:63), that fold forks a multiprocessing
# spawn child, and ttnn JIT-compiles kernels through SFPI inside the timed window -- so
# tt_bio.main, spawn_main and cc1plus are OURS. Calling them foreign was this script's first
# bug: it reported "a foreign fold in 12 of 12 legs" for a run whose only company was itself.
OURS = ("fold_ab_flip", "sample_contention", "narrowq", "tt_bio.main", "multiprocessing.spawn",
        "spawn_main", "sfpi", "cc1plus", "lto1", "riscv-tt",
        # Not folds at all: host telemetry (cardtel.py shells out to tt-smi -s), the sampler's
        # own `ps`, a lock probe, and zombies whose %CPU is their final value, not current use.
        "tt-smi -s", "ps -eo", "fuser ", "<defunct>")


def clk(s):
    try:
        return int(str(s.get("aiclk", {}).get("0")).strip())
    except (TypeError, ValueError):
        return None


def _pearson(legs) -> str:
    xs = [a for a, _, _ in legs]
    ys = [b for _, b, _ in legs]
    if len(xs) < 3 or len(set(xs)) < 2:
        return "n/a (needs 3+ legs with differing load)"
    mx, my = st.mean(xs), st.mean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5
    return f"{num / den:+.3f}" if den else "n/a"


def main(argv):
    samples = [json.loads(l) for l in open(argv[1]) if l.strip()]
    print(f"{len(samples)} samples, {samples[0]['ts']} -> {samples[-1]['ts']}")
    verdicts, legs = [], []
    for path in argv[2:]:
        for cell in json.load(open(path))["cells"]:
            print(f"\n=== rf3 @ {cell['rung']} aa "
                  f"(arm 'off' = fallback forced ON, arm 'on' = shipped default) ===")
            print(f"{'arm':4s} {'rep':>3s} {'rt_s':>7s} {'clk_med':>7s} {'la_med':>6s}  foreign >20%CPU")
            for f in cell["folds"]:
                w = [s for s in samples if f["t_start"] <= s["t"] <= f["t_end"]]
                cl = [c for c in (clk(s) for s in w) if c]
                la = [s["loadavg"][0] for s in w]
                foreign = sorted({t["cmd"][:46] for s in w for t in s.get("top", [])
                                  if t.get("pcpu", 0) > 20
                                  and not any(o in t.get("cmd", "") for o in OURS)})
                print(f"{f['arm']:4s} {f['rep']:3d} {f['runtime_s']:7.1f} "
                      f"{st.median(cl) if cl else 0:7.0f} {st.median(la) if la else 0:6.2f}  "
                      f"{'; '.join(foreign)[:96]}")
                legs.append((st.median(la) if la else 0, f["runtime_s"], bool(foreign)))
            print(f"  A/A floor {cell['aa_spread_pct']:+.3f} %   A/B {cell['ab_median_pct']:+.3f} %"
                  f"   {'INSIDE THE FLOOR' if cell['inside_aa'] else 'outside the floor'}")
            # Per cell, never pooled. Runtime scales with size, so pooling two rungs correlates
            # size with size and can even flip the sign: these legs give +0.476 at 896 alone and
            # -0.238 pooled with 768.
            print(f"  r(loadavg, runtime) within this cell = {_pearson(legs):s}")
            verdicts += legs
            legs = []
    n = sum(1 for _, _, f in verdicts if f)
    print(f"\nlegs with a genuinely foreign process above 20 % CPU: {n} of {len(verdicts)}")
    print("A leg sharing the host with another row's fold is not a timing measurement.")


if __name__ == "__main__":
    main(sys.argv)
