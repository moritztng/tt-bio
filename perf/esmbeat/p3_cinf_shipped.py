#!/usr/bin/env python3
"""C-in and F through the SHIPPED `SwiGLUFFN`, not a body reassembled in a screen script.

`p3_s_cin.py` and `p3_s_resid.py` proved both levers bit-exact on hand-rolled bodies. Those are
not the shipped object: the gates ride inside `l1_gated`, inherit `PAIR_FFN_ROW_BLOCK_SEQ`, share
a refusal ladder that drops F before C-in, and reach the model through a new `residual` entry
point. This runs the real module over four arms and reports, per size:

  * `torch_equal` / `max_abs_diff` of every arm against `off`, which with both gates down is the
    shipped chain: eager `ttnn.chunk`, `concat`, then the full-tensor add the call site used to own;
  * `l1_slice_stats` and `fused_resid_stats` [served, declined] -- a [0, 0] inside the window is a
    vacuous arm and a FAIL, a [0, 0] below the window is the gate correctly not reaching;
  * ms/call at the timed sizes, batched, never per-op-synced.

Every arm writes every gate every round. An arm that leaves a gate alone is the previous arm.
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
ARMS = (("off", False, False), ("cin", True, False), ("f", False, True), ("cinf", True, True))


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
    ap.add_argument("--time-sizes", type=str, default="512")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    timed_sizes = {int(s) for s in a.time_sizes.split(",") if s}

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
           "split_swiglu": ffn.split_swiglu, "sizes": {}}
    print(json.dumps({k: v for k, v in res.items() if k != "sizes"}), flush=True)

    for L in [int(s) for s in a.sizes.split(",")]:
        xt = torch.randn(1, L, L, C_Z) * 0.5
        x = ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        row = {"in_window": bool(lo <= L <= hi)}
        outs = {}
        for name, cin, f in ARMS:
            EC.set_pair_ffn_l1_slice(cin)
            EC.set_pair_ffn_fused_residual(f)
            EC.L1_SLICE_STATS[0] = EC.L1_SLICE_STATS[1] = 0
            EC.FUSED_RESID_STATS[0] = EC.FUSED_RESID_STATS[1] = 0
            try:
                r = ffn.residual(x)
                outs[name] = ttnn.to_torch(r)
                ttnn.deallocate(r)
                row[name + "_slice_stats"] = list(EC.L1_SLICE_STATS)
                row[name + "_resid_stats"] = list(EC.FUSED_RESID_STATS)
                if L in timed_sizes:
                    m, raw = timed(lambda: ttnn.deallocate(ffn.residual(x)), dev)
                    row[name + "_ms"] = round(m, 4)
                    row[name + "_ms_raw"] = [round(v, 4) for v in raw]
            except Exception as e:  # noqa: BLE001 -- an OOM at 1024 is a result, not a crash
                row[name + "_error"] = f"{type(e).__name__}: {e}"[:400]
            finally:
                EC.set_pair_ffn_l1_slice(True)
                EC.set_pair_ffn_fused_residual(True)

        for name in ("cin", "f", "cinf"):
            if "off" in outs and name in outs:
                row[name + "_torch_equal"] = bool(torch.equal(outs["off"], outs[name]))
                row[name + "_max_abs_diff"] = float((outs["off"] - outs[name]).abs().max())
            if name + "_ms" in row and "off_ms" in row:
                row[name + "_delta_ms"] = round(row["off_ms"] - row[name + "_ms"], 4)
                if L == 512:
                    row[name + "_s_per_fold_512"] = round(
                        row[name + "_delta_ms"] * CALLS_512 / 1e3, 3)
        row["refused_slice"] = len(EC._L1_SLICE_REFUSED)
        row["refused_resid"] = len(EC._FUSED_RESID_REFUSED)
        res["sizes"][L] = row
        print(L, json.dumps({k: v for k, v in row.items() if not k.endswith("_raw")}), flush=True)
        ttnn.deallocate(x)
        del xt, outs

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
