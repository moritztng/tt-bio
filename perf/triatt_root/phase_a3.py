"""Phase A3: is the q_chunk lever bit-exact, does it stack with the dtype lever, and does it
generalise across N.

PREDICTIONS: q_chunk only splits rows; the online softmax reduces over k, so q512/k256 must be
torch.equal to q256/k256. Stacking q512+bfp8 should land ~4.4-4.6 ms.
"""
import json, sys, time
import torch, ttnn

OUT = sys.argv[1]
KC = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
    fp32_dest_acc_en=True, packer_l1_acc=True)


def timed(dev, fn, iters=5):
    fn(); ttnn.synchronize_device(dev)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter(); fn(); ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


def main():
    dev = ttnn.open_device(device_id=0)
    grid = dev.compute_with_storage_grid_size()
    res = {"rows": [], "exact": {}}
    for s in (320, 384, 512, 640):
        b, h, d = s, 8, 32
        gf = 4 * b * h * s * s * d / 1e9
        q, k, v = (ttnn.from_torch(torch.randn(b, h, s, d) * 0.1, layout=ttnn.TILE_LAYOUT,
                                   device=dev, dtype=ttnn.bfloat16) for _ in range(3))
        bt = torch.randn(1, h, s, s) * 0.1
        ref = None
        cands = [(64, 64), (256, 256), (s, 256)] + ([(s, 64)] if s <= 384 else [])
        for qc, kc in cands:
            for dt in (ttnn.bfloat16, ttnn.bfloat8_b):
                bias = ttnn.from_torch(bt, layout=ttnn.TILE_LAYOUT, device=dev, dtype=dt)
                prog = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=grid,
                                              exp_approx_mode=False, q_chunk_size=qc, k_chunk_size=kc)
                try:
                    ms = timed(dev, lambda: ttnn.deallocate(
                        ttnn.transformer.scaled_dot_product_attention(
                            q, k, v, attn_mask=bias, is_causal=False, program_config=prog,
                            compute_kernel_config=KC)))
                    o = ttnn.to_torch(ttnn.transformer.scaled_dot_product_attention(
                        q, k, v, attn_mask=bias, is_causal=False, program_config=prog,
                        compute_kernel_config=KC))
                    tag = f"s{s}_q{qc}_k{kc}_{'bf16' if dt == ttnn.bfloat16 else 'bfp8'}"
                    if dt == ttnn.bfloat16 and (qc, kc) == (256, 256):
                        ref = o
                        eq = "REF"
                    elif ref is not None:
                        eq = "torch.equal" if torch.equal(o, ref) else "differs rmsd/std %.6f" % (
                            float((o.float() - ref.float()).pow(2).mean().sqrt() / ref.float().std()))
                    else:
                        eq = "-"
                    res["rows"].append({"seq": s, "q": qc, "k": kc,
                                        "dtype": "bf16" if dt == ttnn.bfloat16 else "bfp8",
                                        "ms": round(ms, 4), "tflops": round(gf / (ms / 1e3) / 1e3, 2),
                                        "vs_ref": eq})
                    print("  s=%3d q%-4d k%-4d %-5s %8.4f ms %6.2f TF/s   %s" %
                          (s, qc, kc, "bf16" if dt == ttnn.bfloat16 else "bfp8", ms,
                           gf / (ms / 1e3) / 1e3, eq), flush=True)
                except Exception as e:
                    print("  s=%3d q%-4d k%-4d ERR %s" % (s, qc, kc, str(e)[:90]), flush=True)
                    res["rows"].append({"seq": s, "q": qc, "k": kc, "error": str(e)[:160]})
                ttnn.deallocate(bias)
        # no-bias ceiling at the best config
        for qc, kc in ((256, 256), (s, 256)):
            try:
                prog = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=grid,
                                              exp_approx_mode=False, q_chunk_size=qc, k_chunk_size=kc)
                ms = timed(dev, lambda: ttnn.deallocate(
                    ttnn.transformer.scaled_dot_product_attention(
                        q, k, v, attn_mask=None, is_causal=False, program_config=prog,
                        compute_kernel_config=KC)))
                res["rows"].append({"seq": s, "q": qc, "k": kc, "dtype": "nobias",
                                    "ms": round(ms, 4), "tflops": round(gf / (ms / 1e3) / 1e3, 2)})
                print("  s=%3d q%-4d k%-4d NOBIAS %7.4f ms %6.2f TF/s" %
                      (s, qc, kc, ms, gf / (ms / 1e3) / 1e3), flush=True)
            except Exception as e:
                print("  s=%3d q%-4d k%-4d NOBIAS ERR %s" % (s, qc, kc, str(e)[:80]), flush=True)
        for t in (q, k, v):
            ttnn.deallocate(t)
    json.dump(res, open(OUT, "w"), indent=2)
    print("wrote", OUT, flush=True)
    ttnn.close_device(dev)


main()
