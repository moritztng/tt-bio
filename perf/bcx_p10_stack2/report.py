#!/usr/bin/env python3
"""The wave-10 sitting, paired round against round.

Pairing matters here and it is not a nicety. Rounds 2, 4 and 7 of every process carry a forward
of 4.5-5.3 s against 2.6-2.8 elsewhere, on BOTH arms, so the raw per-round range of one arm
overlaps the other's for a structural reason and not because the levers did nothing. Round k is
therefore paired against round k, inside an adjacent process pair -- the sitting runs
off, on, on, off, so (p1, p2) and (p4, p3) are each an adjacent pair and a drift across the
sitting cancels between them rather than loading one arm.

    python3 perf/bcx_p10_stack2/report.py
"""
import json
import pathlib
import statistics as st
import sys

sys.path.insert(0, "perf/bcx_p10_stack")
import compare as C                                              # noqa: E402

ROOT = pathlib.Path("perf/bcx_p10_stack2/out")
PAIRS = [("p1_off", "p2_on"), ("p4_off", "p3_on")]
GPU_REF = 0.6958                                                 # s/round, H200, bcx-gpuref


def load(tag):
    rows, stamp = C.rounds_of(str(ROOT / tag / "round_events.json"))
    ev = json.load(open(ROOT / tag / "round_events.json"))["events"]
    starts = [e for e in ev if e["kind"] == "round_start"]
    stops = [e for e in ev if e["kind"] == "round_stop"]
    bounds = starts + stops

    def per_round(key, field):
        """Consecutive boundaries subtract to a per-round reach. A process total cannot tell a
        round that served 108 from two that served 54 and 162."""
        v = [b.get(key, {}).get(field) for b in bounds]
        return [None if (v[i] is None or v[i + 1] is None) else v[i + 1] - v[i]
                for i in range(len(v) - 1)][1:]          # round 1 dropped with the timings

    reach = {"triatt_bw_served": per_round("triatt_bw", "served"),
             "triatt_bw_calls": per_round("triatt_bw", "bw_calls"),
             "triatt_bw_declined": per_round("triatt_bw", "declined"),
             "mm_served": per_round("mm_layout", "served"),
             "rne_served": per_round("reach", "rne_add_served"),
             "rne_declined": per_round("reach", "rne_add_declined")}
    return rows, stamp, reach


def main():
    arms, reach_by_tag, stamps = {}, {}, {}
    for pair in PAIRS:
        for tag in pair:
            rows, stamp, reach = load(tag)
            arms[tag] = rows
            reach_by_tag[tag] = reach
            stamps[tag] = stamp

    deltas, wins, n = [], 0, 0
    for off_tag, on_tag in PAIRS:
        for a, b in zip(arms[off_tag], arms[on_tag]):
            deltas.append(a["wall"] - b["wall"])
            wins += a["wall"] > b["wall"]
            n += 1

    def col(tags, k):
        return [r[k] for t in tags for r in arms[t]]

    offs = [p[0] for p in PAIRS]
    ons = [p[1] for p in PAIRS]
    print("wave-10 composed stack, pc card 0, one sitting, off/on/on/off at the process "
          "boundary\n")
    hdr = f"{'arm':<6}{'n':>3}{'round s':>10}{'min':>8}{'max':>8}{'host':>8}{'dev':>8}" \
          f"{'AICLK':>7}{'clkmin':>8}{'load1':>7}"
    print(hdr)
    for name, tags in (("off", offs), ("on", ons)):
        w = col(tags, "wall")
        print(f"{name:<6}{len(w):>3}{st.median(w):>10.3f}{min(w):>8.3f}{max(w):>8.3f}"
              f"{st.median(col(tags,'host')):>8.3f}{st.median(col(tags,'dev')):>8.3f}"
              f"{st.median(col(tags,'aiclk')):>7.0f}{min(col(tags,'aiclk_min')):>8.0f}"
              f"{st.median(col(tags,'load1')):>7.2f}")

    med_off, med_on = st.median(col(offs, "wall")), st.median(col(ons, "wall"))
    pm = st.median(deltas)
    print(f"\npaired, round k against round k inside an adjacent process pair, n={n}")
    print(f"  paired median delta   {pm:+.3f} s   ->  {med_off / (med_off - pm):.4f}x")
    print(f"  paired mean delta     {st.mean(deltas):+.3f} s "
          f"(sd {st.stdev(deltas):.3f})")
    print(f"  rank-paired wins      {wins}/{n}")
    print(f"  unpaired medians      {med_off:.3f} -> {med_on:.3f} s, "
          f"{med_off / med_on:.4f}x")
    print(f"  device column         {st.median(col(offs,'dev')):.3f} -> "
          f"{st.median(col(ons,'dev')):.3f} s, "
          f"{st.median(col(offs,'dev')) / st.median(col(ons,'dev')):.4f}x")
    print(f"\nRATIO against the {GPU_REF} s H200 round: "
          f"off {med_off / GPU_REF:.2f}x  ->  on {med_on / GPU_REF:.2f}x")

    print("\nreach, per round, at the round boundary (a flag that declines measures the anchor)")
    for tag in [t for p in PAIRS for t in p]:
        r = reach_by_tag[tag]
        s = stamps[tag]
        print(f"  {tag:<8} hifi={s.get('triatt_hifi')} bw={s.get('triatt_bw_fused')} "
              f"rne={s.get('rne_kernel')} taped={s.get('taped_kernels')!r}")
        for k, v in r.items():
            print(f"      {k:<20} {v}")
    json.dump({"pairs": PAIRS, "deltas": deltas, "wins": wins, "n": n,
               "median_off": med_off, "median_on": med_on,
               "paired_median_delta": pm,
               "ratio_off": med_off / GPU_REF, "ratio_on": med_on / GPU_REF,
               "reach": reach_by_tag}, open(ROOT / "report.json", "w"), indent=1)


if __name__ == "__main__":
    main()
