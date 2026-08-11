"""Phase A2: the levers reachable from Python at N=512, plus the L1 copy roof.

PREDICTIONS (written before the run):
  Q1 bfloat8_b bias: read 1088+384=1472 MB; max(compute 2.766, 1472/469=3.14) -> ~4.0-4.6 ms, 1.6-1.8x
  Q2 bfloat4_b bias: read 576+384=960 MB -> compute-bound ~2.9-3.3 ms, 2.2-2.5x
  Q3 exp_approx_mode=True: exp is on the softmax critical path but the op is read-bound,
     so predict <3% -- if it is large the op is not read-bound and P2 is wrong
  Q4 asymmetric chunks: no change to mask bytes, so predict within +-10% of 7.15 ms, none better
  Q5 L1 copy roof ~1171 GB/s (ledger); manual-attention floor = 4 round trips of a 2.147 GB
     score tensor = 7.3 ms at that roof, i.e. no better than the 7.153 ms it would replace
"""
import json, sys, time
import torch, ttnn
from tt_bio import tenstorrent as T

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
    res = {"rows": []}
    s, h, d = 512, 8, 32
    b = s
    gf = 4 * b * h * s * s * d / 1e9
    qt = [torch.randn(b, h, s, d) * 0.1 for _ in range(3)]
    q, k, v = (ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16) for t in qt)
    bt = torch.randn(1, h, s, s) * 0.1

    # ---- L1 copy roof -----------------------------------------------------------------
    for mb in (26, 52):
        el = mb * 2 ** 20 // 2
        side = 1 << (el.bit_length() // 2)
        rows = el // side
        t = ttnn.from_torch(torch.randn(rows, side) * 0.1, layout=ttnn.TILE_LAYOUT,
                            device=dev, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
        nb = rows * side * 2
        ms = timed(dev, lambda: ttnn.deallocate(ttnn.clone(t, memory_config=ttnn.L1_MEMORY_CONFIG)))
        res[f"l1_copy_{mb}MB_gbs_rw"] = round(2 * nb / (ms / 1e3) / 1e9, 1)
        print("  L1->L1 copy %d MB (r+w): %.4f ms -> %.1f GB/s" % (mb, ms, 2 * nb / (ms / 1e3) / 1e9), flush=True)
        ttnn.deallocate(t)

    def run(label, bias_dtype, qc, kc, approx):
        try:
            bias = None if bias_dtype is None else ttnn.from_torch(
                bt, layout=ttnn.TILE_LAYOUT, device=dev, dtype=bias_dtype)
            prog = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=dev.compute_with_storage_grid_size(),
                exp_approx_mode=approx, q_chunk_size=qc, k_chunk_size=kc)
            ms = timed(dev, lambda: ttnn.deallocate(
                ttnn.transformer.scaled_dot_product_attention(
                    q, k, v, attn_mask=bias, is_causal=False, program_config=prog,
                    compute_kernel_config=KC)))
            row = {"label": label, "q_chunk": qc, "k_chunk": kc, "approx": approx,
                   "ms": round(ms, 4), "tflops": round(gf / (ms / 1e3) / 1e3, 2)}
            print("  %-28s q%-4d k%-4d approx=%-5s %8.4f ms %6.2f TF/s" %
                  (label, qc, kc, approx, ms, gf / (ms / 1e3) / 1e3), flush=True)
            if bias is not None:
                ttnn.deallocate(bias)
        except Exception as e:
            row = {"label": label, "q_chunk": qc, "k_chunk": kc, "approx": approx, "error": str(e)[:180]}
            print("  %-28s q%-4d k%-4d approx=%-5s ERR %s" % (label, qc, kc, approx, str(e)[:90]), flush=True)
        res["rows"].append(row)

    run("baseline bf16 bias", ttnn.bfloat16, 256, 256, False)
    run("bfloat8_b bias", ttnn.bfloat8_b, 256, 256, False)
    run("bfloat4_b bias", ttnn.bfloat4_b, 256, 256, False)
    run("no bias (ceiling)", None, 256, 256, False)
    run("bf16 bias exp_approx", ttnn.bfloat16, 256, 256, True)
    run("no bias exp_approx", None, 256, 256, True)
    for qc, kc in ((128, 256), (256, 128), (256, 512), (512, 256), (128, 512), (64, 256)):
        run("chunk sweep bf16", ttnn.bfloat16, qc, kc, False)

    # ---- numerics of the dtype shortcut at N=512, against the bf16-bias output ---------
    ref = None
    for name, dt in (("bfloat16", ttnn.bfloat16), ("bfloat8_b", ttnn.bfloat8_b), ("bfloat4_b", ttnn.bfloat4_b)):
        bias = ttnn.from_torch(bt, layout=ttnn.TILE_LAYOUT, device=dev, dtype=dt)
        prog = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=dev.compute_with_storage_grid_size(),
            exp_approx_mode=False, q_chunk_size=256, k_chunk_size=256)
        o = ttnn.to_torch(ttnn.transformer.scaled_dot_product_attention(
            q, k, v, attn_mask=bias, is_causal=False, program_config=prog,
            compute_kernel_config=KC)).float()
        if ref is None:
            ref = o
        else:
            dif = (o - ref)
            res[f"op_rmsd_over_std_{name}"] = round(float(dif.pow(2).mean().sqrt() / ref.std()), 6)
            res[f"op_pcc_{name}"] = round(float(torch.corrcoef(
                torch.stack([o.flatten(), ref.flatten()]))[0, 1]), 6)
            print("  numerics %-10s rmsd/std %.6f  pcc %.6f" %
                  (name, res[f"op_rmsd_over_std_{name}"], res[f"op_pcc_{name}"]), flush=True)
        ttnn.deallocate(bias)

    json.dump(res, open(OUT, "w"), indent=2)
    print("wrote", OUT, flush=True)
    ttnn.close_device(dev)


main()
