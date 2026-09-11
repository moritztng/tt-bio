#!/usr/bin/env python3
"""Place Boltz-2 512 aa on both platforms' rooflines and print the table for FINDINGS.md.

Inputs: the per-phase FLOP/byte count from flops_bytes_boltz2.py, the p150a roofs measured on
this card by roofs_p150a.py, and the wall/device times already measured elsewhere (each one
cited in the table it prints). Nothing here is a datasheet number except the two H200 roofs,
which are labelled ASSERTED because this task was not allowed to rent a GPU to measure them.
"""
import argparse
import json
from pathlib import Path

# --- measured elsewhere, cited ----------------------------------------------------------
# H200 arm walls and per-fold device time: ~/.coworker/state/nvidia-bioir-boltz2-h200.md,
# artifacts/nvidia-bioir/results/{C_bioir_512,A_oss_eager_512}.json and kernels_{A,C}.json.
H200 = {
    "A_wall_s": 7.268, "A_device_s": 6.6455,
    "C_wall_s": 2.445, "C_device_s": 2.0695,
    # kernel census, arm C, per fold, ms by class
    "C_arith_ms": 446.6 + 617.6 + 111.9 + 80.9,     # cuBLAS SM90 + CuTeDSL + cuEq + Triton
    "C_traffic_ms": 655.6 + 75.9,                   # generic elementwise + memcpy/memset
    "A_arith_ms": 2722.6 + 227.4 + 197.3,           # fp32 SIMT GEMM + cuBLAS non-SM90 + cuEq
    "A_traffic_ms": 1967.7 + 604.6 + 420.0,         # elementwise + fp32 LayerNorm + memcpy
    # ASSERTED, not measured here: one H200 SXM.
    "bw_TBs": 4.8, "bf16_TFLOPs": 7916.0 / 8,
}
# p150a published cell: site/data/perf-512aa.json, boltz2.cells.p150a
P150A = {"wall_s": 23.504, "host_s": 0.382}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flops", required=True)
    ap.add_argument("--roofs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    fb = json.loads(Path(a.flops).read_text())
    rf = json.loads(Path(a.roofs).read_text())

    # The counter ran in fp32 because that is torch's CPU default; both products run bf16
    # activations and bf16 weights, so every byte number is halved. FLOPs are dtype-free.
    FL = fb["fold_flops"]
    B_UNFUSED = fb["fold_eager_bytes"] / 2
    B_RESIDENT = fb["fold_resident_bytes"] / 2

    copy_roof = max(r["copy_roof_GBps"] for r in rf["copy"]) * 1e9
    rw_roof = max(r["rw_roof_GBps"] for r in rf["read"]) * 1e9
    bw_roof = max(copy_roof, rw_roof)
    mm = {}
    for r in rf["matmul"]:
        mm[r["fidelity"]] = max(mm.get(r["fidelity"], 0.0), r["TFLOPs"])
    mm_roof = max(mm.values()) * 1e12
    hifi4 = mm.get("HiFi4", 0.0) * 1e12

    p_dev = P150A["wall_s"] - P150A["host_s"]
    plat = [
        ("H200 boltz 2.2.1 (arm A)", H200["A_wall_s"], H200["A_device_s"],
         H200["bf16_TFLOPs"] * 1e12, H200["bw_TBs"] * 1e12,
         H200["A_arith_ms"], H200["A_traffic_ms"]),
        ("H200 BioIR 0.1.0 (arm C)", H200["C_wall_s"], H200["C_device_s"],
         H200["bf16_TFLOPs"] * 1e12, H200["bw_TBs"] * 1e12,
         H200["C_arith_ms"], H200["C_traffic_ms"]),
        ("p150a tt-bio (published cell)", P150A["wall_s"], p_dev, hifi4 or mm_roof, bw_roof,
         None, None),
    ]

    L = []
    w = L.append
    w("# Boltz-2, 512 aa, on two rooflines\n")
    w("Work per fold, counted on the real modules (`flops_bytes_boltz2.py`), 3 recycles / 4 trunk")
    w("passes / 200 sampling steps / 1 diffusion sample, 35-row MSA:\n")
    w("| phase | TFLOP/fold | share |")
    w("|---|---|---|")
    for r in fb["rows"]:
        w("| %s | %.2f | %.1f %% |" % (r["phase"], r["fold_flops"] / 1e12,
                                       100 * r["fold_flops"] / FL))
    w("| **total** | **%.2f** | |\n" % (FL / 1e12))
    w("Bytes per fold, bf16, two bounds:\n")
    w("- resident (parameters only, every activation stays on chip): **%.1f GB**, "
      "arithmetic intensity **%.0f FLOP/byte**" % (B_RESIDENT / 1e9, FL / B_RESIDENT))
    w("- unfused (every aten op's inputs and outputs): **%.2f TB**, arithmetic intensity "
      "**%.1f FLOP/byte**\n" % (B_UNFUSED / 1e12, FL / B_UNFUSED))
    w("Machine balance (compute roof / bandwidth roof), the FLOP/byte above which a machine is")
    w("compute-bound:\n")
    seen = set()
    for name, wall, dev, croof, broof, _, _ in plat:
        if (croof, broof) in seen:
            continue
        seen.add((croof, broof))
        w("- %s: %.0f FLOP/byte" % (name.split(' ')[0], croof / broof))
    w("")
    w("## Where each arm actually lands\n")
    w("| arm | s/fold | device s | achieved TFLOP/s | % of compute roof | "
      "implied GB/s if unfused | % of bandwidth roof |")
    w("|---|---|---|---|---|---|---|")
    for name, wall, dev, croof, broof, _, _ in plat:
        ach = FL / dev
        impl = B_UNFUSED / dev
        w("| %s | %.3f | %.3f | %.1f | %.1f %% | %.0f | %.0f %% |"
          % (name, wall, dev, ach / 1e12, 100 * ach / croof, impl / 1e9, 100 * impl / broof))
    w("")
    w("A row whose last column exceeds 100 % is proof by contradiction that the arm does not")
    w("move the unfused byte count: no implementation exceeds its own bandwidth roof.\n")
    w("## Kernel-class split of device time, H200 (census, per fold)\n")
    w("| arm | arithmetic-shaped kernels | traffic-shaped kernels | achieved TFLOP/s over the "
      "arithmetic part | % of compute roof |")
    w("|---|---|---|---|---|")
    for name, wall, dev, croof, broof, ar, tr in plat:
        if ar is None:
            continue
        rate = FL / (ar / 1e3)
        w("| %s | %.0f ms | %.0f ms | %.0f | %.1f %% |"
          % (name, ar, tr, rate / 1e12, 100 * rate / croof))
    w("")
    w("## Roofs\n")
    w("p150a, measured on qb1 card 2 by `roofs_p150a.py`, %s, %dx%d grid:\n"
      % (rf["arch"], rf["grid"][0], rf["grid"][1]))
    w("- DRAM copy roof (read+write): **%.0f GB/s**" % (copy_roof / 1e9))
    w("- DRAM read+write roof (2 reads, 1 write): **%.0f GB/s**" % (rw_roof / 1e9))
    for k in ("LoFi", "HiFi2", "HiFi4"):
        if k in mm:
            w("- dense bf16 square matmul, %s: **%.1f TFLOP/s**" % (k, mm[k]))
    w("")
    w("H200 SXM roofs are **ASSERTED, not measured**: %.1f TB/s HBM3e and %.0f TFLOP/s bf16"
      % (H200["bw_TBs"], H200["bf16_TFLOPs"]))
    w("dense (one eighth of the 7916 TFLOP/s DGX figure). This task was not funded to rent a")
    w("GPU, so every H200 percentage above is against a vendor number and is a *lower* bound on")
    w("utilisation: a measured roof is always below the datasheet, so the real percentages are")
    w("higher than printed.\n")

    Path(a.out).write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
