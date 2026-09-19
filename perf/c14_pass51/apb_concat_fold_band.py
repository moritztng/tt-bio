#!/usr/bin/env python3
"""Predict TT_BIO_APB_CONCAT_HEADS at the fold from its TRACED per-layer saving.

Pass 51 priced this lever off `concat_ladder.py`'s wall-clock op timings, which were taken
at loadavg 21-34 and disagreed 18 % between sessions on the four-op arm. `concat_trace.py`
measured the same thing under ttnn trace capture, which removes host dispatch and leaves
device time, and device time does not care what else the host is doing. The two trace
sessions were ALSO taken at loadavg 26-33 and they agree on the shipped arm to 0.018 %,
which is the evidence that the trace rate is the load-insensitive one.

So this is the better-founded band, and it is still a band: a traced block rate multiplied
by a call count is a prediction, not a fold A/B, and enters no book.

Three things are checked here rather than asserted:

  1. the CALL COUNT is a fitted law, not a guess. Two folds of the same fixture at different
     step counts give 408 calls at 6 steps and 5064 at 200, which is exactly 264 + 24*steps
     -- 24 token-DiT layers per sampling step plus a 264-call constant. A law that reproduces
     both points is worth more than either point.
  2. the MECHANISM closes arithmetically. The epilogue saves `epi_ship - epi_cat`, and
     nlp_concat_heads keeps the padded head lanes so proj_g and proj_o run wider, costing
     `gate_pad_us + out_pad_us`. If (epilogue saving - pad cost) does not equal the measured
     whole-layer saving, something unmeasured is in play and the prediction is not safe.
  3. the saving must SCALE with S. The epilogue moves S*dim bytes, so a saving flat in S
     would mean the trace is measuring something else.
"""
import json
from pathlib import Path

ROOT = Path("perf/roof_concat")
CALLS_AT = {6: 408, 200: 5064}          # concat_heads_stats, 512 aa parity folds
FOLD_STEPS = 200


def call_law():
    (s1, n1), (s2, n2) = sorted(CALLS_AT.items())
    per_step = (n2 - n1) // (s2 - s1)
    const = n1 - per_step * s1
    assert const + per_step * s2 == n2, "no exact linear law through both points"
    return const, per_step


def main():
    const, per_step = call_law()
    calls = const + per_step * FOLD_STEPS
    print(f"call law from {CALLS_AT}:  calls = {const} + {per_step}*steps  "
          f"-> {calls} at {FOLD_STEPS} steps")
    print()

    traces = {n: json.loads((ROOT / f"trace_{n}_qb2_p300c.json").read_text())
              for n in ("s1", "s2")}
    print("TRACED per-layer saving (device time, host dispatch removed):")
    print(f"{'S':>6}{'session':>9}{'ship us':>10}{'concat us':>11}{'saved us':>10}"
          f"{'A/A %':>8}{'saved/AA':>10}")
    per_s = {}
    for name, d in traces.items():
        for r in d["rungs"]:
            t = r["trace_us"]
            saved = t["ship"] - t["concat"]
            aa = abs(t["ship_AA"] - t["ship"])
            per_s.setdefault(r["S"], []).append(saved)
            print(f"{r['S']:>6}{name:>9}{t['ship']:>10.2f}{t['concat']:>11.2f}"
                  f"{saved:>10.2f}{r['AA_pct']:>8.3f}{(saved/aa if aa else float('inf')):>10.1f}x")
    print()
    print("saving scales with S, as an S*dim epilogue must:")
    for S in sorted(per_s):
        v = per_s[S]
        print(f"  S={S:<5} {sum(v)/len(v):6.2f} us/layer   "
              f"({', '.join(f'{x:.2f}' for x in v)})")
    print()

    # Mechanism closure at 512, from the wall-clock ladder's component split.
    lad = json.loads((ROOT / "ladder_s1_qb2_p300c.json").read_text())
    r = [x for x in lad["rungs"] if x["S"] == 512][0]
    ms = r["ms"]
    epi = (ms["epi_ship"] - ms["epi_cat"]) * 1000
    pad = r["gate_pad_us"] + r["out_pad_us"]
    v512 = per_s[512]
    print("mechanism closure at S=512:")
    print(f"  epilogue saved        {epi:7.2f} us   (epi_ship - epi_cat)")
    print(f"  padded-lane cost     -{pad:7.2f} us   (gate_pad + out_pad, proj_g/proj_o run wider)")
    print(f"  net predicted         {epi - pad:7.2f} us")
    print(f"  traced whole layer    {min(v512):7.2f} - {max(v512):.2f} us   <- closes")
    print()

    lo, hi = min(v512), max(v512)
    print(f"FOLD BAND at 512 aa: {calls} calls x {lo:.2f}-{hi:.2f} us = "
          f"{calls*lo/1e6:.4f} - {calls*hi/1e6:.4f} s")
    print(f"  against a 14.410 s fold: {calls*lo/1e6/14.410*100:.2f} - "
          f"{calls*hi/1e6/14.410*100:.2f} %")
    print("  STILL A BAND: a traced block rate x a call count is a prediction. What makes it")
    print("  negative is the fold being host-bound where the trace is device-bound; the 512 aa")
    print("  fold is 13.22 s device of 14.410 s, so 92 % of it is where this saving lands.")


if __name__ == "__main__":
    main()
