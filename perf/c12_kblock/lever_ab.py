#!/usr/bin/env python3
"""Op-level A/B of TT_BIO_LINEAR_KBLOCK through the PRODUCTION call path, on device.

The sweep measured `ttnn.linear` with configs built by the sweep's own code. This measures
`tt_bio.tenstorrent._linear_blocked`, the function the eight converted call sites actually call, so
a config that is right in the sweep and wrong in the module cannot pass. Arms are interleaved rep by
rep and the flag is flipped on the module attribute between arms, which is the same object the call
sites read. Each key carries its own A/A arm: the flag-off call a second time.

Accuracy is scored against a float64 reference of the bf16 operands the device holds, and the
flag-on arm must be at least as close as the flag-off arm. This is the op-level half only; the
structural Angstrom score belongs to the fold A/B and is not claimed here.
"""
import json
import os
import sys
import time

import torch
import ttnn

WT = "/home/ttuser/.coworker/wt/c12-kblock-unlock"
sys.path.insert(0, WT)
sys.path.insert(0, os.path.join(WT, "perf/c12_kblock"))
import clk                                  # noqa: E402
import tt_bio.tenstorrent as TB             # noqa: E402

L1, DRAM = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
# (label, a_shape, w_shape, a memory_config, out memory_config). The placements are each key's
# OWN measured placement from c10-fold-census, not a single choice applied to all four: the pair
# transition runs L1-resident and the DiT and CTB projections run DRAM->DRAM. Getting this wrong is
# not cosmetic -- the first run of this script put every `a` in L1 and the two DRAM keys inverted to
# 0.9416x and 0.9545x, because a block config tuned for a DRAM operand is not the one an L1 operand
# wants. The table keys on (mt_total, kt, nt) and NOT on placement, so that inversion is a real open
# risk at any site whose operand placement differs from the one its entry was measured at.
KEYS = [
    ("fc2  (256,4,16)", (1, 16, 512, 128), (128, 512), L1, L1),
    ("fc3  (256,16,4)", (1, 16, 512, 512), (512, 128), L1, DRAM),
    ("dit  (16,24,24)", (1, 512, 768), (768, 768), DRAM, DRAM),
    ("ctb  (16,24,48)", (1, 512, 768), (768, 1536), DRAM, DRAM),
]
REPS, PIPE = 5, 4


def med(xs):
    return sorted(xs)[len(xs) // 2]


def main():
    dev = TB.get_device()
    nodes = clk.nodes_open_by_this_process()
    held = clk.force(1350, nodes)
    node = nodes[0]
    grid = TB.CORE_GRID_MAIN
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    print("grid %dx%d | nodes %s | forced on %s | aiclk %s MHz | flag default %s"
          % (grid.x, grid.y, nodes, held, clk.aiclk(node), TB._LINEAR_KBLOCK), flush=True)

    torch.manual_seed(0)
    out = {}
    for label, a_shape, w_shape, amc, omc in KEYS:
        at = torch.randn(*a_shape) * 0.1
        wt = torch.randn(*w_shape) * 0.1
        ta = ttnn.from_torch(at, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                             memory_config=amc)
        tw = ttnn.from_torch(wt, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                             memory_config=DRAM)
        ref = ttnn.to_torch(ta).double() @ ttnn.to_torch(tw).double()

        def call(on):
            TB._LINEAR_KBLOCK = on
            try:
                return TB._linear_blocked(ta, tw, compute_kernel_config=ckc, core_grid=grid,
                                          memory_config=omc)
            finally:
                TB._LINEAR_KBLOCK = False

        # Proof the flag actually changes the executed op, before anything is timed: with it off
        # the helper must fall through to core_grid, with it on it must produce a program config.
        cfg_on = TB._linear_block_cfg(a_shape, w_shape, None, None, grid)
        TB._LINEAR_KBLOCK = False
        cfg_off = TB._linear_block_cfg(a_shape, w_shape, None, None, grid)
        assert cfg_off is None and cfg_on is None, "flag is not being read where it is written"
        TB._LINEAR_KBLOCK = True
        cfg_on = TB._linear_block_cfg(a_shape, w_shape, None, None, grid)
        TB._LINEAR_KBLOCK = False
        assert cfg_on is not None, f"{label}: flag on produced no config"
        fam = "1d" if "1D" in type(cfg_on).__name__ else "2d"
        witness = "%s bw=%d drain=%dx%d" % (fam, cfg_on.in0_block_w, cfg_on.out_block_h,
                                            cfg_on.out_block_w)

        arms = [("off", lambda: ttnn.deallocate(call(False))),
                ("off_aa", lambda: ttnn.deallocate(call(False))),
                ("on", lambda: ttnn.deallocate(call(True)))]
        for _, fn in arms:
            for _ in range(2):
                fn()
        ttnn.synchronize_device(dev)
        acc = {k: [] for k, _ in arms}
        s = clk.Sampler(node)
        s.start()
        for _ in range(REPS):
            for name, fn in arms:
                ttnn.synchronize_device(dev)
                t0 = time.perf_counter()
                for _ in range(PIPE):
                    fn()
                ttnn.synchronize_device(dev)
                acc[name].append((time.perf_counter() - t0) * 1e3 / PIPE)
        clock = s.stop()

        err = {}
        for name, on in (("off", False), ("on", True)):
            y = call(on)
            err[name] = (ttnn.to_torch(y).double() - ref).abs().max().item()
            ttnn.deallocate(y)
        ms = {k: med(v) for k, v in acc.items()}
        aa = ms["off"] / ms["off_aa"]
        x = ms["off"] / ms["on"]
        res = {"witness": witness, "ms": ms, "x": round(x, 4), "aa": round(aa, 4),
               "maxabs": err, "clock": clock,
               "result": abs(x - 1) > abs(aa - 1), "accuracy_ok": err["on"] <= err["off"]}
        out[label] = res
        print("  %-17s %s %s | off %.4f ms  on %.4f ms  %.4fx  [A/A %.4fx]  maxabs %.5f->%.5f  %s%s"
              % (label, witness, "L1" if amc is L1 else "DRAM", ms["off"], ms["on"], x, aa, err["off"], err["on"],
                 "RESULT" if res["result"] else "inside floor",
                 "" if res["accuracy_ok"] else "  ACCURACY REGRESSED"), flush=True)
        ttnn.deallocate(ta)
        ttnn.deallocate(tw)

    json.dump(out, open(sys.argv[1], "w"), indent=1) if len(sys.argv) > 1 else None
    clk.release()


if __name__ == "__main__":
    main()
