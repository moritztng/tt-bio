"""Phase A probe for triatt-bottleneck-rootcause. qb2 card 1, ttnn 0.68.0.

PREDICTIONS WRITTEN BEFORE THE RUN (see state/triatt-bottleneck-rootcause.md P1-P6):
  P1  SDPA N=512 prod chunk256: 7.10 ms/call +-5%, ~19.3 TFLOP/s
  P2  read bytes/call 2.550 GB -> implied 359 GB/s, must be <= measured read roof
  P3  bias-off at N=512: 2.6-2.9 ms  => op ratio 2.45-2.73x
  P4  bias share of reads 84.2%
  P6  d=32 no-bias rate ~51.9 TFLOP/s (W9 at N=320)
"""
import json, sys, time
import torch, ttnn
from tt_bio import tenstorrent as T

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
    dev = ttnn.open_device(device_id=0)
    g = dev.compute_with_storage_grid_size()
    res = {"grid": [g.x, g.y], "cores": g.x * g.y, "l1_bytes": getattr(dev, "l1_size_per_core", lambda: 0)()}
    print("grid %dx%d = %d cores, L1/core %d B" % (g.x, g.y, g.x * g.y, getattr(dev, "l1_size_per_core", lambda: 0)()), flush=True)

    # ---- R1 compute roof: square bf16 matmul, HiFi4, fp32 acc -------------------------
    roofs = {}
    for n in (2048, 4096):
        a = ttnn.from_torch(torch.randn(n, n) * 0.1, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        b = ttnn.from_torch(torch.randn(n, n) * 0.1, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        ms = timed(dev, lambda: ttnn.deallocate(ttnn.matmul(a, b, compute_kernel_config=KC)))
        tf = 2 * n ** 3 / (ms / 1e3) / 1e12
        roofs[f"matmul_{n}_tflops"] = round(tf, 2)
        print("  compute roof N=%d: %.4f ms -> %.2f TFLOP/s" % (n, ms, tf), flush=True)
        ttnn.deallocate(a); ttnn.deallocate(b)

    # ---- R2 DRAM read roof: clone DRAM->L1, count the read only -----------------------
    for mb in (52, 105):
        el = mb * 2 ** 20 // 2
        side = 1 << (el.bit_length() // 2)
        rows = el // side
        t = ttnn.from_torch(torch.randn(rows, side) * 0.1, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        nb = rows * side * 2
        ms = timed(dev, lambda: ttnn.deallocate(ttnn.clone(t, memory_config=ttnn.L1_MEMORY_CONFIG)))
        roofs[f"read_clone_dram2l1_{mb}MB_gbs"] = round(nb / (ms / 1e3) / 1e9, 1)
        print("  read roof clone->L1 %d MB: %.4f ms -> %.1f GB/s" % (mb, ms, nb / (ms / 1e3) / 1e9), flush=True)
        # write roof: L1 source -> DRAM dest, count the write only
        try:
            tl = ttnn.clone(t, memory_config=ttnn.L1_MEMORY_CONFIG)
            msw = timed(dev, lambda: ttnn.deallocate(ttnn.clone(tl, memory_config=ttnn.DRAM_MEMORY_CONFIG)))
            roofs[f"write_clone_l12dram_{mb}MB_gbs"] = round(nb / (msw / 1e3) / 1e9, 1)
            print("  write roof clone L1->DRAM %d MB: %.4f ms -> %.1f GB/s" % (mb, msw, nb / (msw / 1e3) / 1e9), flush=True)
            ttnn.deallocate(tl)
        except Exception as e:
            print("  write roof %d MB ERR %s" % (mb, str(e)[:120]), flush=True)
        # dram->dram clone, count read+write
        ms2 = timed(dev, lambda: ttnn.deallocate(ttnn.clone(t, memory_config=ttnn.DRAM_MEMORY_CONFIG)))
        roofs[f"copy_dram2dram_{mb}MB_gbs_rw"] = round(2 * nb / (ms2 / 1e3) / 1e9, 1)
        print("  copy DRAM->DRAM %d MB (r+w): %.1f GB/s" % (mb, 2 * nb / (ms2 / 1e3) / 1e9), flush=True)
        ttnn.deallocate(t)
    res["roofs"] = roofs

    # ---- S1 the production SDPA at N=512, with and without bias -----------------------
    rows = []
    for s in (320, 512):
        b, h, d = s, 8, 32
        gf = 4 * b * h * s * s * d / 1e9
        q, k, v = (ttnn.from_torch(torch.randn(b, h, s, d) * 0.1, layout=ttnn.TILE_LAYOUT,
                                   device=dev, dtype=ttnn.bfloat16) for _ in range(3))
        bias = ttnn.from_torch(torch.randn(1, h, s, s) * 0.1, layout=ttnn.TILE_LAYOUT,
                               device=dev, dtype=ttnn.bfloat16)
        prod = T._tri_att_sdpa_program_config(s, s)
        for label, m in (("prod+bias", bias), ("prod-nobias", None)):
            try:
                ms = timed(dev, lambda mm=m: ttnn.deallocate(
                    ttnn.transformer.scaled_dot_product_attention(
                        q, k, v, attn_mask=mm, is_causal=False,
                        program_config=prod, compute_kernel_config=KC)))
                bias_B = b * h * s * s * 2 if m is not None else 0
                qkv_B = 3 * b * h * s * d * 2
                out_B = b * h * s * d * 2
                rd = bias_B + qkv_B
                rows.append({"seq": s, "cfg": label, "ms": round(ms, 4),
                             "tflops": round(gf / (ms / 1e3) / 1e3, 2),
                             "read_MB": round(rd / 2 ** 20, 1), "write_MB": round(out_B / 2 ** 20, 1),
                             "bias_frac_of_read": round(bias_B / rd, 4),
                             "implied_read_GBs": round(rd / (ms / 1e3) / 1e9, 1),
                             "implied_write_GBs": round(out_B / (ms / 1e3) / 1e9, 1),
                             "AI_flop_per_byte": round(gf * 1e9 / (rd + out_B), 2)})
                print("  SDPA seq=%d %-12s %8.4f ms %6.2f TF/s  read %7.1f MB -> %6.1f GB/s  AI %.2f" %
                      (s, label, ms, gf / (ms / 1e3) / 1e3, rd / 2 ** 20, rd / (ms / 1e3) / 1e9,
                       gf * 1e9 / (rd + out_B)), flush=True)
            except Exception as e:
                rows.append({"seq": s, "cfg": label, "error": str(e)[:200]})
                print("  SDPA seq=%d %-12s ERR %s" % (s, label, str(e)[:120]), flush=True)
        for t in (q, k, v, bias):
            ttnn.deallocate(t)
    res["sdpa"] = rows

    # ---- S2 batch sweep at N=512: isolate the bias slope ------------------------------
    sw = []
    s, h, d = 512, 8, 32
    bias = ttnn.from_torch(torch.randn(1, h, s, s) * 0.1, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    prod = T._tri_att_sdpa_program_config(s, s)
    for b in (64, 128, 256, 512):
        q, k, v = (ttnn.from_torch(torch.randn(b, h, s, d) * 0.1, layout=ttnn.TILE_LAYOUT,
                                   device=dev, dtype=ttnn.bfloat16) for _ in range(3))
        e = {}
        for label, m in (("bias", bias), ("nobias", None)):
            ms = timed(dev, lambda mm=m: ttnn.deallocate(
                ttnn.transformer.scaled_dot_product_attention(
                    q, k, v, attn_mask=mm, is_causal=False, program_config=prod,
                    compute_kernel_config=KC)), iters=5)
            e[label] = round(ms, 4)
        sw.append({"b": b, **e})
        print("  batch sweep b=%3d  bias %8.4f ms  nobias %8.4f ms  delta %8.4f" %
              (b, e["bias"], e["nobias"], e["bias"] - e["nobias"]), flush=True)
        for t in (q, k, v):
            ttnn.deallocate(t)
    ttnn.deallocate(bias)
    res["batch_sweep_512"] = sw
    if len(sw) >= 2:
        db = sw[-1]["b"] - sw[0]["b"]
        slope_bias = (sw[-1]["bias"] - sw[0]["bias"]) / db * 1e3
        slope_nb = (sw[-1]["nobias"] - sw[0]["nobias"]) / db * 1e3
        bias_per_b = h * s * s * 2
        res["bias_stream_GBs"] = round(bias_per_b / ((slope_bias - slope_nb) / 1e6) / 1e9, 1)
        print("  slope bias %.4f us/b, nobias %.4f us/b, diff %.4f -> bias stream %.1f GB/s" %
              (slope_bias, slope_nb, slope_bias - slope_nb, res["bias_stream_GBs"]), flush=True)

    json.dump(res, open(OUT, "w"), indent=2)
    print("wrote", OUT, flush=True)
    ttnn.close_device(dev)


main()
