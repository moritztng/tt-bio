#!/usr/bin/env python3
"""S3a: the fused fp32-softmax chain at the PRODUCTION shape, not the 64-token probe shape.

`probe_fuse2.py` settled bit-exactness at S=64. This runs the shipped `_fp32_softmax_attention`
against its own `_FP32_SOFTMAX_FUSED_ADD=False` baseline at the real trunk geometry and asks the two
kill gates the plan pre-registered: is the activation list accepted at the production shape, and is
the off-fold win at S=512 at least 1.15x. `torch.equal` is the accuracy bar, nothing looser.
"""
import json, os, sys, time
from pathlib import Path

import torch, ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import tt_bio.tenstorrent as T  # noqa: E402

SIZES = tuple(int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ("256", "512", "768")))
OUT = Path(__file__).resolve().parent / ("screen_fused_full_qb1c%s_%s.json" % (
    os.environ.get("TT_VISIBLE_DEVICES", "0"), "_".join(str(x) for x in SIZES)))
H, DH = 4, 32
SCALE_INV = DH ** -0.5


def timed(fn, reps=7, warm=2):
    dev = T.get_device()
    for _ in range(warm):
        o = fn(); ttnn.synchronize_device(dev); ttnn.deallocate(o)
    ts = []
    for i in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        o = fn()
        ttnn.synchronize_device(dev)
        ts.append(time.perf_counter() - t0)
        if i < reps - 1:
            ttnn.deallocate(o)
    ts.sort()
    return ts[len(ts) // 2], [round(x * 1e3, 3) for x in ts], o


def main():
    dev = T.get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    res = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES", "0"),
           "ttnn": __import__("importlib.metadata", fromlist=["x"]).version("ttnn"),
           "n_heads": H, "head_dim": DH, "block_bytes": T._FP32_SOFTMAX_BLOCK_BYTES, "rows": []}
    for S in SIZES:
        row = {"S": S, "score_fp32_GiB": H * S ** 3 * 4 / 2 ** 30}
        torch.manual_seed(S)
        qh, kh, vh = (torch.randn(S, H, S, DH, dtype=torch.bfloat16) * 0.1 for _ in range(3))
        bh = torch.randn(1, H, S, S, dtype=torch.bfloat16) * 0.1
        mk = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev,
                                       dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        q, k, v, bias = mk(qh), mk(kh), mk(vh), mk(bh)
        call = lambda: T._fp32_softmax_attention(q, k, v, bias, scale_inv=SCALE_INV,
                                                 compute_kernel_config=ckc,
                                                 out_dtype=ttnn.bfloat16, bias_scale_inv=1.0)
        outs = {}
        for arm, fused in (("stock", False), ("fused", True)):
            T._FP32_SOFTMAX_FUSED_ADD = fused
            T.FP32_SOFTMAX_STATS.update(calls=0, blocked=0, blocks=0, fused=0, unfused=0)
            try:
                t, ts, o = timed(call)
                row[arm + "_ms"], row[arm + "_all_ms"] = round(t * 1e3, 3), ts
                row[arm + "_stats"] = dict(T.FP32_SOFTMAX_STATS)
                outs[arm] = ttnn.to_torch(o)
                ttnn.deallocate(o)
            except Exception as e:                                            # noqa: BLE001
                row[arm + "_ms"] = f"REFUSED: {type(e).__name__}: {str(e)[:300]}"
        if len(outs) == 2:
            row["torch_equal"] = bool(torch.equal(outs["stock"], outs["fused"]))
            row["max_abs_diff"] = float((outs["stock"].float() - outs["fused"].float()).abs().max())
            if isinstance(row["stock_ms"], float):
                row["speedup"] = round(row["stock_ms"] / row["fused_ms"], 4)
        for t_ in (q, k, v, bias):
            try: ttnn.deallocate(t_)
            except Exception: pass
        res["rows"].append(row)
        OUT.write_text(json.dumps(res, indent=1))
        print(json.dumps(row, indent=1), flush=True)
    print("wrote", OUT, flush=True)


main()
