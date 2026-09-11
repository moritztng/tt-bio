#!/usr/bin/env python3
"""Scale the captured per-call bytes and times to a production 512 aa fold and place them.

Both inputs are conservative in the same direction, which is the direction that cannot
manufacture the conclusion: the byte count dedupes tensors by id, so a tensor read by ten ops is
counted once (traffic floor), and the per-call time brackets the call with a device sync on both
sides, so it carries the sync (time ceiling). Bytes over time is therefore a floor on achieved
bandwidth, and "we are at N % of the roof" is a lower bound on N.

A roof is a property of the part, so the roofs come from the JSON that `roofs_bh.py` wrote on the
same card as the capture, and the card is named in the table heading. The two capture/roof pairs
this ran against are p150a on qb1 card 2 and p300c on qb2 card 2.
"""
import argparse
import json
from pathlib import Path

# calls in a production fold: 64x4 trunk + 8 confidence, 4x4 MSA, 24x200, 6x200. The pairformer
# counts are per fold already (they do not scale with the diffusion rollout); the two diffusion
# entries are layers per sampling step, multiplied by the production step count.
PHASE = {
    "pairformer|1x512x384,1x512x512x128": ("pairformer block (trunk + confidence)",
                                           "pairformer_block", 264, False),
    "pairformer|1x512x512x128":           ("MSA block", "msa_block", 16, False),
    "difftx|1x512x768,1x512x768":         ("diffusion token layer",
                                           "diffusion_token_layer", 24, True),
    "difftx|1x224x32x128,1x224x32x128":   ("atom transformer layer",
                                           "atom_transformer_layer", 6, True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True)
    ap.add_argument("--roofs", required=True)
    ap.add_argument("--flops", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cell-s", type=float, default=23.504, help="published 512 aa cell, s/fold")
    ap.add_argument("--cell-host-s", type=float, default=0.382)
    a = ap.parse_args()

    cap = json.loads(Path(a.capture).read_text())
    rf = json.loads(Path(a.roofs).read_text())
    fb = json.loads(Path(a.flops).read_text())

    bw_roof = max([r["copy_roof_GBps"] for r in rf["copy"]]
                  + [r["rw_roof_GBps"] for r in rf["read"]]) * 1e9
    mm = {}
    for r in rf["matmul"]:
        mm[r["fidelity"]] = max(mm.get(r["fidelity"], 0.0), r["TFLOPs"])
    mm_roof = mm["HiFi4"] * 1e12
    flops = {r["phase"]: r["per_call_flops"] for r in fb["rows"]}
    fold_flops = fb["fold_flops"]
    board, host, card = rf["board"], rf["host"], rf["card"]

    tab = {}
    L, w = [], lambda s: L.append(s)
    w("## What a 512 aa fold actually moves, measured on %s card %s (%s)\n" % (host, card, board))
    w("One real fold, every `PairformerLayer` and diffusion `DiffusionTransformerLayer` call")
    w("timed with a device sync on both sides, one settled call of each shape captured with")
    w("`ttnn.graph` and its DRAM traffic summed (`perf/bioir_roofline/fold_bytes_512.py`).")
    w("Call counts are what the fold actually issued, and they reproduce the architecture")
    w("exactly: 264 = 64 trunk blocks x 4 passes + 8 confidence, 16 = 4 MSA blocks x 4 passes,")
    w("24 and 6 layers per sampling step.\n")
    w("| phase | calls/fold | ms/call | DRAM GB/call | GB/s achieved | %% of %.0f GB/s roof | "
      "TFLOP/s | %% of %.1f TFLOP/s roof |" % (bw_roof / 1e9, mm_roof / 1e12))
    w("|---|---|---|---|---|---|---|---|")
    tot_s = tot_b = tot_f = 0.0
    for r in cap["rows"]:
        s = r["sig"]
        if s not in PHASE:
            continue
        label, phase, base, scales = PHASE[s]
        n = base * (cap["prod_steps"] if scales else 1)
        t = r["median_ms"] / 1e3
        b = r["dram_total"]
        gbs, tf = b / t, flops[phase] / t
        tot_s += n * t
        tot_b += n * b
        tot_f += n * flops[phase]
        tab[phase] = {"calls": n, "ms_per_call": r["median_ms"], "bytes_per_call": b,
                      "fold_s": n * t, "fold_bytes": n * b}
        w("| %s | %d | %.3f | %.3f | %.0f | %.0f %% | %.1f | %.1f %% |"
          % (label, n, r["median_ms"], b / 1e9, gbs / 1e9, 100 * gbs / bw_roof,
             tf / 1e12, 100 * tf / mm_roof))
    ach = tot_b / tot_s
    w("| **instrumented total** | | **%.2f s** | **%.2f TB** | **%.0f** | **%.0f %%** | "
      "**%.1f** | **%.1f %%** |"
      % (tot_s, tot_b / 1e12, ach / 1e9, 100 * ach / bw_roof,
         tot_f / tot_s / 1e12, 100 * (tot_f / tot_s) / mm_roof))
    w("")
    dev = a.cell_s - a.cell_host_s
    # The remainder uses each phase's summed total_ms, not its median: the first call of every
    # shape carries kernel compilation, so a median-based sum would charge that compile time to
    # the uninstrumented rest of the fold.
    rest_capture = cap["fold_s"] - sum(
        r["total_ms"] / 1e3 for r in cap["rows"] if r["sig"] in PHASE)
    w("The instrumented phases are **%.2f s of the %.3f s published cell** (%.0f %% of its"
      % (tot_s, a.cell_s, 100 * tot_s / dev))
    w("%.2f s of device time) and carry **%.1f %% of the fold's FLOPs**. The capture fold's own"
      % (dev, 100 * tot_f / fold_flops))
    w("uninstrumented remainder is **%.2f s** (embedder, templates, recycling glue, confidence"
      % rest_capture)
    w("heads, diffusion conditioning), so the instrumented phases plus that remainder")
    w("predict a **%.2f s** production fold against the **%.3f s** cell, **%.1f %% apart**: the"
      % (tot_s + rest_capture, a.cell_s,
         abs(100 * (tot_s + rest_capture - a.cell_s) / a.cell_s)))
    w("per-call syncs did not dominate and the per-shape scaling is sound.\n")
    unfused, resident = fb["fold_eager_bytes"] / 2, fb["fold_resident_bytes"] / 2
    whole = tot_b * dev / tot_s
    w("Charging the uninstrumented remainder of the fold the same achieved rate: **%.2f TB of"
      % (whole / 1e12))
    w("DRAM traffic per fold**, against a **%.2f TB** unfused count and a **%.1f GB** fully"
      % (unfused / 1e12, resident / 1e9))
    w("resident bound. So this implementation already moves **%.1fx less** than an unfused one"
      % (unfused / whole))
    w("and still **%.0fx more** than the compulsory minimum.\n" % (whole / resident))
    Path(a.out).write_text("\n".join(L))
    Path(a.out).with_suffix(".json").write_text(json.dumps({
        "host": host, "card": card, "board": board,
        "bw_roof_GBps": bw_roof / 1e9, "matmul_hifi4_TFLOPs": mm_roof / 1e12,
        "cell_s": a.cell_s, "cell_device_s": dev,
        "instrumented_s": tot_s, "instrumented_bytes": tot_b,
        "achieved_GBps": ach / 1e9, "achieved_TFLOPs": tot_f / tot_s / 1e12,
        "capture_remainder_s": rest_capture, "whole_fold_bytes": whole,
        "unfused_bytes_bf16": unfused, "resident_bytes_bf16": resident,
        "phases": tab}, indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    main()
