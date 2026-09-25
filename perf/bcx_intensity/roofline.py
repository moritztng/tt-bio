#!/usr/bin/env python3
"""Place every part of BindCraft 2's gradient step on the roof that actually binds it.

Three instruments, each used for the one thing it is good for, and each named beside its
number:

  FLOP   `intensity.py ai`, this row. Computed from the operand shapes the taped step really
         ran, recorded call by call. Shapes do not depend on the chip, so the trace ran on
         whglx card 8 (Wormhole Galaxy) while the seconds below are Blackhole's.
  BYTES  `perf/bcx_bytes/census.json` (`bcx-bytes`, qb1 card 2, Blackhole p300c, 1350 MHz),
         joined to the device profiler rather than counted at the Python call. That is the
         better byte instrument and it already exists; this row does not re-measure it.
  TIME   `perf/bcx_bytes/join_bwd.json`, per device-op class, same run.

Roofs, both measured, neither derived here:
  compute 115.685 TFLOP/s  (dense 4096 cube, state/of3t/BACKWARD.md)
  DRAM    424.7 GB/s       (Blackhole, tt_bio/triatt_sdpa.py:293; Wormhole is 227.5)
  balance 115.685e12 / 424.7e9 = 272.4 FLOP/byte, inside the 247-338 band BACKWARD.md quotes.
"""
from __future__ import annotations

import collections
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "perf" / "bcx_intensity"

COMPUTE_ROOF = 115.685e12
DRAM_ROOF = 424.7e9
BALANCE = COMPUTE_ROOF / DRAM_ROOF


def bh_census():
    blob = subprocess.run(["git", "-C", str(ROOT), "show",
                           "origin/wk/bcx-bytes:perf/bcx_bytes/census.json"],
                          capture_output=True, text=True, check=True).stdout
    return json.loads(blob)


def bh_join():
    blob = subprocess.run(["git", "-C", str(ROOT), "show",
                           "origin/wk/bcx-bytes:perf/bcx_bytes/join_bwd.json"],
                          capture_output=True, text=True, check=True).stdout
    return json.loads(blob)


def mine(n):
    return json.loads((OUT / f"ai_n{n}.json").read_text())


# The two classifiers do not use the same words. `bcx-bytes` bins by device kernel; this row
# bins by ttnn verb. Mapped rather than renamed, so a class that exists on only one side stays
# visible instead of being folded into "other".
CLASS_MAP = {"matmul": "matmul", "eltwise": "eltwise", "typecast": "typecast",
             "layout": "layout", "softmax": "softmax", "norm": "layernorm",
             "reduce": "reduction", "view": "view"}


def fam_totals(blob, stack, phase):
    agg = collections.defaultdict(lambda: [0, 0.0, 0.0])
    for r in blob["blocks"][stack]["table"]:
        if r["phase"] != phase:
            continue
        a = agg[CLASS_MAP.get(r["family"], r["family"])]
        a[0] += r["n"]
        a[1] += r["flop"]
        a[2] += r["bytes"]
    return agg


def main():
    cen, join = bh_census(), bh_join()
    m256, m288 = mine(256), mine(288)
    out = {"roofs": {"compute_flops": COMPUTE_ROOF, "dram_bps": DRAM_ROOF,
                     "balance_flop_per_byte": BALANCE},
           "stamp_256": m256["stamp"], "stamp_288": m288["stamp"], "blocks": {}}
    for stack, jkey in (("evo", "evo"), ("extra", "extra")):
        key = f"bwd-fix {stack} n=256"
        bh_bytes = cen[key]["bwd"]["by_class"]
        bh_all_mb = cen[key]["bwd"]["moved_MB"]
        ms = join[jkey]["ms"]
        csv_ms, matched_ms = join[jkey]["csv_ms"], join[jkey]["matched_ms"]
        f256 = fam_totals(m256, stack, "bwd")
        f288 = fam_totals(m288, stack, "bwd")
        rows = []
        for cls, (nops, mb) in sorted(bh_bytes.items(), key=lambda x: -x[1][1]):
            if cls in ("free", "alloc"):
                continue
            t = ms.get(f"cls:{cls}")
            if t is None or mb <= 0:
                continue
            flop = f256.get(cls, [0, 0.0, 0.0])[1]
            b = mb * 1e6
            s = t / 1e3
            ai = flop / b
            gbs = b / s / 1e9
            tfs = flop / s / 1e12
            # The roof that binds: below balance the compute roof is unreachable, and the
            # ceiling is AI * DRAM_ROOF, not COMPUTE_ROOF.
            attainable = min(COMPUTE_ROOF, ai * DRAM_ROOF)
            rows.append({
                "class": cls, "ops": nops, "bytes_GB": b / 1e9, "ms": t,
                "flop_G": flop / 1e9, "ai": ai, "binds": "dram" if ai < BALANCE else "compute",
                "GB_s": gbs, "pct_dram_roof": 100 * gbs / (DRAM_ROOF / 1e9),
                "TFLOP_s": tfs, "pct_compute_peak": 100 * tfs * 1e12 / COMPUTE_ROOF,
                "attainable_TFLOP_s": attainable / 1e12,
                "headroom_x": (attainable / (tfs * 1e12)) if tfs else None,
                "n288_over_n256_bytes": (f288.get(cls, [0, 0, 1e-30])[2]
                                         / max(f256.get(cls, [0, 0, 1e-30])[2], 1e-30)),
                "n288_over_n256_flop": (f288.get(cls, [0, 0, 0])[1]
                                        / max(f256.get(cls, [0, 0, 0])[1], 1e-30)),
            })
        tot_b = bh_all_mb * 1e6
        tot_f = sum(v[1] for v in f256.values())
        tot_s = csv_ms / 1e3
        ai = tot_f / tot_b
        whole = {"bytes_GB": tot_b / 1e9, "ms": csv_ms, "flop_G": tot_f / 1e9, "ai": ai,
                 "binds": "dram" if ai < BALANCE else "compute",
                 "GB_s": tot_b / tot_s / 1e9,
                 "pct_dram_roof": 100 * (tot_b / tot_s) / DRAM_ROOF,
                 "TFLOP_s": tot_f / tot_s / 1e12,
                 "pct_compute_peak": 100 * (tot_f / tot_s) / COMPUTE_ROOF,
                 "attainable_TFLOP_s": min(COMPUTE_ROOF, ai * DRAM_ROOF) / 1e12,
                 "headroom_x": min(COMPUTE_ROOF, ai * DRAM_ROOF) / (tot_f / tot_s),
                 "matched_ms": matched_ms,
                 "n288_over_n256_bytes": (sum(v[2] for v in f288.values())
                                          / max(sum(v[2] for v in f256.values()), 1e-30)),
                 "n288_over_n256_flop": (sum(v[1] for v in f288.values())
                                         / max(sum(v[1] for v in f256.values()), 1e-30))}
        out["blocks"][stack] = {"rows": rows, "whole": whole}
        print(f"\n===== {stack} backward, n=256, Blackhole p300c 1350 MHz "
              f"(bytes+ms bcx-bytes, FLOP this row) =====")
        print(f"{'class':<11}{'ops':>5}{'GB':>8}{'ms':>8}{'GFLOP':>9}{'AI':>8}"
              f"{'binds':>8}{'GB/s':>8}{'%DRAM':>7}{'TF/s':>7}{'%peak':>7}{'head':>7}")
        for r in rows:
            print(f"{r['class']:<11}{r['ops']:>5}{r['bytes_GB']:>8.2f}{r['ms']:>8.2f}"
                  f"{r['flop_G']:>9.1f}{r['ai']:>8.2f}{r['binds']:>8}{r['GB_s']:>8.1f}"
                  f"{r['pct_dram_roof']:>7.1f}{r['TFLOP_s']:>7.2f}"
                  f"{r['pct_compute_peak']:>7.2f}{r['headroom_x'] or 0:>7.2f}")
        w = whole
        print(f"{'WHOLE':<11}{'':>5}{w['bytes_GB']:>8.2f}{w['ms']:>8.2f}{w['flop_G']:>9.1f}"
              f"{w['ai']:>8.2f}{w['binds']:>8}{w['GB_s']:>8.1f}{w['pct_dram_roof']:>7.1f}"
              f"{w['TFLOP_s']:>7.2f}{w['pct_compute_peak']:>7.2f}{w['headroom_x']:>7.2f}")
        print(f"  n=288/n=256 from this row's own trace: bytes x{w['n288_over_n256_bytes']:.4f}, "
              f"FLOP x{w['n288_over_n256_flop']:.4f}  (288^2/256^2 = 1.2656)")
    (OUT / "roofline.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote {OUT / 'roofline.json'}")
    print(f"balance = {BALANCE:.1f} FLOP/byte")


if __name__ == "__main__":
    main()
