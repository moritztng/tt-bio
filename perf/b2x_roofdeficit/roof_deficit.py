#!/usr/bin/env python3
"""Where the Boltz-2 512 aa fold's time goes, per phase, against both roofs.

Takes measured per-call times and per-call FLOPs from the roofline campaign's capture
(`~/.coworker/state/bioir-roofline/capture_rows.json`) and the corrected per-call byte counts
from `b2x-diffusion-layer-bytes`, and reports for every phase:

  * achieved GB/s and the percent of the measured 429.9 GB/s streaming roof,
  * achieved TFLOP/s and the percent of the measured 85.96 TFLOP/s dense bf16 HiFi4 roof,
  * the roof deficit: measured time minus the time the phase's bytes would take at the roof.

A phase low on both axes is neither bandwidth- nor compute-bound, and no byte-deleting lever
will help it. That is the case this fold turns out to be in.

Byte counts must come from a buffer-address dedupe. The original counter deduped on tensor id
and charged free metadata views as full DRAM reads, which inflated the fold 1.6x; see
`perf/b2x_difflayer/real_traffic.py`.

No device needed. Run: python3 perf/b2x_roofdeficit/roof_deficit.py
"""
import json
import os

CAPTURE = os.path.expanduser("~/.coworker/state/bioir-roofline/capture_rows.json")

BW_ROOF = 429.9e9      # streaming, p300c qb2 card 2, measured
MM_ROOF = 85.96e12     # dense bf16 HiFi4 N=8192, same card, measured
SYNC_INFLATION = 1.182  # the instrumented fold syncs both sides of every call

# corrected bytes/call, buffer-address dedupe, b2x-diffusion-layer-bytes, qb2 card 2
CORRECTED_BYTES = {
    "pairformer_block": 6.651e9,
    "msa_block": 6.519e9,
    "diffusion_token_layer": 198.94e6,
    "atom_transformer_layer": 303.91e6,
}
OPS_PER_CALL = {
    "pairformer_block": 430,
    "msa_block": 378,
    "diffusion_token_layer": 60,
    "atom_transformer_layer": 66,
}


def main():
    d = json.load(open(CAPTURE))
    placed = d["fold_bytes_placed_p300c"]
    flops = {r["phase"]: r["per_call_flops"] for r in d["flops_bytes_512_qb2_recount"]["rows"]}

    hdr = ("phase", "ms/call", "calls", "fold_s", "GB/s", "%BW", "TF/s", "%MM", "deficit_s", "us/op")
    print(f"{hdr[0]:26}{hdr[1]:>9}{hdr[2]:>7}{hdr[3]:>8}{hdr[4]:>8}{hdr[5]:>7}{hdr[6]:>7}{hdr[7]:>7}{hdr[8]:>11}{hdr[9]:>7}")

    inst_s = floor_s = 0.0
    total_ops = 0
    for phase, ph in sorted(placed["phases"].items(), key=lambda kv: -kv[1]["ms_per_call"] * kv[1]["calls"]):
        ms, n = ph["ms_per_call"], ph["calls"]
        b, f = CORRECTED_BYTES[phase], flops[phase]
        gbps, tf = b / (ms * 1e-3), f / (ms * 1e-3)
        floor_ms = b / BW_ROOF * 1e3
        deficit_ms = ms - floor_ms
        inst_s += ms * n / 1e3
        floor_s += floor_ms * n / 1e3
        total_ops += OPS_PER_CALL[phase] * n
        print(f"{phase:26}{ms:9.4f}{n:7d}{ms * n / 1e3:8.3f}{gbps / 1e9:8.1f}{100 * gbps / BW_ROOF:6.1f}%"
              f"{tf / 1e12:7.2f}{100 * tf / MM_ROOF:6.1f}%{deficit_ms * n / 1e3:11.3f}"
              f"{deficit_ms * 1e3 / OPS_PER_CALL[phase]:7.1f}")

    dev = placed["cell_device_s"]
    deficit_s = inst_s - floor_s
    print()
    print(f"instrumented phase time        {inst_s:7.3f} s")
    print(f"  explained by bytes at roof   {floor_s:7.3f} s")
    print(f"  roof deficit                 {deficit_s:7.3f} s   ({100 * deficit_s / inst_s:.0f}% of it)")
    print(f"  deficit, sync-deflated       {deficit_s - inst_s * (1 - 1 / SYNC_INFLATION):7.3f} s")
    print(f"uninstrumented remainder       {placed['capture_remainder_s']:7.3f} s")
    print(f"cell device time               {dev:7.3f} s")
    print()
    print(f"Ceiling of the whole byte-deletion axis: delete every instrumented byte and the fold is")
    print(f"still {dev - floor_s:.3f} s, i.e. {dev / (dev - floor_s):.3f}x. 2x is not on this axis.")
    print()
    print("What a realistic deletion buys, at the 79-93% marginal conversion measured at the trimul:")
    tot_b = sum(CORRECTED_BYTES[p] * placed["phases"][p]["calls"] for p in CORRECTED_BYTES)
    for frac in (0.10, 0.20, 0.30, 0.40):
        lo = tot_b * frac / (0.93 * BW_ROOF)
        hi = tot_b * frac / (0.79 * BW_ROOF)
        print(f"  delete {frac:.0%}: -{lo:.2f} to -{hi:.2f} s -> {dev / (dev - lo):.3f}x to {dev / (dev - hi):.3f}x")


if __name__ == "__main__":
    main()
