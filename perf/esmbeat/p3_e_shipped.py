#!/usr/bin/env python3
"""S-E2: lever E through the SHIPPED `SwiGLUFFN.__call__`, not a hand-rolled body.

p3's screen (`p3_s_lnl1.py`) proved the L1 layer_norm destination bit-exact on a body assembled in
the screen script. That is not the same object as the shipped gate: the gate rides inside
`l1_gated`, inherits `PAIR_FFN_ROW_BLOCK_SEQ`, and carries a refusal cache that the screen body did
not have. This runs the real module, both arms, and reports three things per size:

  * `torch_equal` / `max_abs_diff` on the module output;
  * `l1_ln_stats` [served, declined] -- a [0, 0] inside the window is a vacuous arm and a FAIL,
    a [0, 0] below the window is the gate correctly not reaching;
  * ms/call for each arm, batched, never per-op-synced.
"""
import argparse, json, os, statistics as st, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import torch, ttnn
from tt_bio import tenstorrent as T
from tt_bio import esmc as EC

C_Z, D_FF = 256, 1024
CALLS_512 = 538  # ESMFold2's pair-transition call count at 512 aa; the only size it is exact for


def timed(fn, dev, reps=4, batches=5, warm=2):
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(dev)
    out = []
    for _ in range(batches):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        ttnn.synchronize_device(dev)
        out.append((time.perf_counter() - t0) * 1e3 / reps)
    return st.median(out), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=str, default="256,298,320,512,640,768,1024")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    dev = T.get_device()
    g = dev.compute_with_storage_grid_size()
    ck = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)

    torch.manual_seed(0)
    sd = {
        "0.weight": torch.randn(C_Z) * 0.05 + 1.0,
        "0.bias": torch.randn(C_Z) * 0.02,
        "1.weight": torch.randn(2 * D_FF, C_Z) * 0.02,
        "3.weight": torch.randn(C_Z, D_FF) * 0.02,
    }
    ffn = EC.SwiGLUFFN(sd, ck, fuse_swiglu=True)

    lo, hi = EC.PAIR_FFN_ROW_BLOCK_SEQ
    res = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "grid": [g.x, g.y], "rows": EC._PAIR_FFN_ROW_BLOCK, "window": [lo, hi],
           "split_swiglu": ffn.split_swiglu, "fuse_swiglu": ffn.fuse_swiglu, "sizes": {}}
    print(json.dumps({k: v for k, v in res.items() if k != "sizes"}), flush=True)

    for L in [int(s) for s in a.sizes.split(",")]:
        xt = torch.randn(1, L, L, C_Z) * 0.5
        x = ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        row = {"in_window": bool(lo <= L <= hi)}
        outs = {}
        for name, on in (("off", False), ("on", True)):
            prev = EC.set_pair_ffn_l1_ln(on)
            EC.L1_LN_STATS[0] = EC.L1_LN_STATS[1] = 0
            EC.L1_FC1_STATS[0] = EC.L1_FC1_STATS[1] = 0
            try:
                r = ffn(x)
                outs[name] = ttnn.to_torch(r)
                ttnn.deallocate(r)
                row[name + "_l1_ln_stats"] = list(EC.L1_LN_STATS)
                row[name + "_l1_fc1_stats"] = list(EC.L1_FC1_STATS)
                EC.L1_LN_STATS[0] = EC.L1_LN_STATS[1] = 0
                m, raw = timed(lambda: ttnn.deallocate(ffn(x)), dev)
                row[name + "_ms"] = round(m, 4)
                row[name + "_ms_raw"] = [round(v, 4) for v in raw]
            except Exception as e:  # noqa: BLE001 -- an OOM at 1024 is a result, not a crash
                row[name + "_error"] = f"{type(e).__name__}: {e}"[:400]
            finally:
                EC.set_pair_ffn_l1_ln(prev)

        if "off" in outs and "on" in outs:
            row["torch_equal"] = bool(torch.equal(outs["off"], outs["on"]))
            row["max_abs_diff"] = float((outs["off"] - outs["on"]).abs().max())
        if "off_ms" in row and "on_ms" in row:
            row["delta_ms"] = round(row["on_ms"] - row["off_ms"], 4)
            if L == 512:
                row["delta_s_per_fold_512"] = round(row["delta_ms"] * CALLS_512 / 1e3, 3)
        row["refused_shapes"] = len(EC._L1_LN_REFUSED)
        res["sizes"][L] = row
        print(L, json.dumps({k: v for k, v in row.items() if not k.endswith("_raw")}), flush=True)
        ttnn.deallocate(x)
        del xt, outs

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
