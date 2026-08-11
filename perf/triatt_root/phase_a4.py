"""Phase A4: the other pf.triatt components at N=512, measured on this card, so the AI
comparison against SDPA is all one instrument.
PREDICTION: qkv ~46 TF/s, out ~38 TF/s (the 512aa ledger's shape census on this same card).
qkv AI = 103.08 GFLOP / 537.3 MB = 192 flop/byte; SDPA realised AI = 51. That ratio, not the
shapes, is the reason one runs at 46 and the other at 19.
"""
import json, sys, time
import torch, ttnn

OUT = sys.argv[1]
KC = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
    fp32_dest_acc_en=True, packer_l1_acc=True)


def timed(dev, fn, iters=7):
    fn(); ttnn.synchronize_device(dev)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter(); fn(); ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


def main():
    from tt_bio.tenstorrent import _qkv_mm_config
    dev = ttnn.open_device(device_id=0)
    res = {"rows": []}
    N, C = 512, 256
    for name, cout in (("qkv 256->768", 768), ("out 256->256", 256), ("bias_proj 256->8", 32)):
        x = ttnn.from_torch(torch.randn(N, N, C) * 0.1, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        w = ttnn.from_torch(torch.randn(C, cout) * 0.1, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        gf = 2 * N * N * C * cout / 1e9
        rd = N * N * C * 2 + C * cout * 2
        wr = N * N * cout * 2
        try:
            cfg = _qkv_mm_config(x, w)
            ms = timed(dev, lambda: ttnn.deallocate(ttnn.experimental.minimal_matmul(
                input_tensor=x, weight_tensor=w, compute_kernel_config=KC,
                dtype=ttnn.bfloat16, config=cfg)))
            row = {"op": name, "ms": round(ms, 4), "tflops": round(gf / (ms / 1e3) / 1e3, 2),
                   "read_MB": round(rd / 2 ** 20, 1), "write_MB": round(wr / 2 ** 20, 1),
                   "AI_flop_per_byte": round(gf * 1e9 / (rd + wr), 1),
                   "implied_read_GBs": round(rd / (ms / 1e3) / 1e9, 1),
                   "implied_write_GBs": round(wr / (ms / 1e3) / 1e9, 1)}
            print("  %-16s %8.4f ms %6.2f TF/s  r %6.1f MB (%5.1f GB/s)  w %6.1f MB (%5.1f GB/s)  AI %6.1f" %
                  (name, ms, row["tflops"], row["read_MB"], row["implied_read_GBs"],
                   row["write_MB"], row["implied_write_GBs"], row["AI_flop_per_byte"]), flush=True)
        except Exception as e:
            row = {"op": name, "error": str(e)[:180]}
            print("  %-16s ERR %s" % (name, str(e)[:110]), flush=True)
        res["rows"].append(row)
        ttnn.deallocate(x); ttnn.deallocate(w)
    json.dump(res, open(OUT, "w"), indent=2)
    print("wrote", OUT, flush=True)
    ttnn.close_device(dev)


main()
