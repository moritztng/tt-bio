"""Phase B2b: the exact q_chunk cap, and the rule C1's guard needs above it.

B2 found q(seq)/k256 legal to seq 640 and illegal at 768 -- earlier than predicted. Two things are
still unknown and C1 cannot be written without them:
  1. is legality a property of q_chunk alone, or of (q_chunk, seq)? CB allocation depends on the
     chunk sizes and head_dim, not on seq, so it should be q_chunk alone. Test: run q640/k256 at
     seq 768, where B2 only tested q768.
  2. above the cap, which q_chunk wins? Two mechanisms compete and B2 conflates them:
       (a) per-q-chunk K/V re-read: every q-chunk re-reads all of padded_k x d of K and V, so
           total K/V traffic scales with n_q_chunks = padded_q / q_chunk.
       (b) padding: the mask grid read and computed is padded_q x padded_k, so a q_chunk that does
           not divide the sequence inflates the BIAS, which is 84% of the traffic.
     seq 320 isolates (a): q64 and q320 both give padded_q = 320, identical bias bytes, and it is
     still 1.5336x. seq 576 stacks both: q256 pads 576 -> 768. So the rule should be the largest
     multiple of 32 that is <= cap AND divides the padded sequence.

PREDICTIONS:
  S1  q640/k256 is LEGAL at seq 768 (legality is q_chunk-only). It should still be SLOWER than
      q384, because it pads padded_q 768 -> 1280 (1.67x of bias and compute).
  S2  at seq 768 the winner is q384: divides 768 exactly, 2 chunks instead of 3. Expect ~1.15-1.35x
      over production q256.
  S3  at seq 1024 the winner is q512: divides exactly, 2 chunks. Same range.
  S4  the cap is between 640 and 768. Testing q672/q704 pins it; I expect the CB limit lands at
      704 or 736, not on a round number.
  S5  everything here is torch.equal to production, same argument as B1.
"""
import json, sys, time
from importlib.metadata import version as _v
import torch, ttnn
TTNN_V = _v("ttnn")

OUT = sys.argv[1]
KC = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
    fp32_dest_acc_en=True, packer_l1_acc=True)


def timed(dev, fn, iters):
    fn(); ttnn.synchronize_device(dev)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter(); fn(); ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


def run(dev, grid, q, k, v, bias, qc, kc, iters):
    prog = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=grid, exp_approx_mode=False,
                                  q_chunk_size=qc, k_chunk_size=kc)
    call = lambda: ttnn.transformer.scaled_dot_product_attention(
        q, k, v, attn_mask=bias, is_causal=False, program_config=prog, compute_kernel_config=KC)
    ms = timed(dev, lambda: ttnn.deallocate(call()), iters)
    o = call(); t = ttnn.to_torch(o); ttnn.deallocate(o)
    return ms, t


def main():
    dev = ttnn.open_device(device_id=0)
    grid = dev.compute_with_storage_grid_size()
    res = {"ttnn": TTNN_V, "cap_probe": [], "above_cap": []}

    # --- S4: pin the cap. Probe q_chunk alone at a seq large enough to host it, k pinned at 256.
    # seq 768 hosts any q_chunk <= 768. Legality only, 1 iter, no timing needed.
    s = 768
    q, k, v = (ttnn.from_torch(torch.randn(s, 8, s, 32) * 0.1, layout=ttnn.TILE_LAYOUT,
                              device=dev, dtype=ttnn.bfloat16) for _ in range(3))
    bias = ttnn.from_torch(torch.randn(1, 8, s, s) * 0.1, layout=ttnn.TILE_LAYOUT,
                           device=dev, dtype=ttnn.bfloat16)
    for qc in (640, 672, 704, 736, 768):
        prog = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=grid, exp_approx_mode=False,
                                      q_chunk_size=qc, k_chunk_size=256)
        try:
            o = ttnn.transformer.scaled_dot_product_attention(
                q, k, v, attn_mask=bias, is_causal=False, program_config=prog,
                compute_kernel_config=KC)
            ttnn.synchronize_device(dev); ttnn.deallocate(o)
            ok, msg = True, ""
        except Exception as e:
            ok, msg = False, str(e)[:300]
        res["cap_probe"].append({"seq": s, "q": qc, "k": 256, "legal": ok, "error": msg})
        print("CAP  seq=%d q%-4d k256  %s  %s" % (s, qc, "LEGAL" if ok else "ILLEGAL", msg[:150]),
              flush=True)

    # --- S1/S2/S5: above the cap at seq 768, which q_chunk wins, and is it exact
    pms, pref = run(dev, grid, q, k, v, bias, 256, 256, 3)
    gf = 4 * s * 8 * s * s * 32 / 1e9
    print("seq=768 PROD q256 k256 %9.4f ms %6.2f TF/s" % (pms, gf / (pms / 1e3) / 1e3), flush=True)
    res["above_cap"].append({"seq": s, "q": 256, "k": 256, "ms": round(pms, 4), "ref": True,
                             "tflops": round(gf / (pms / 1e3) / 1e3, 2)})
    for qc in (384, 512, 640):
        try:
            ms, o = run(dev, grid, q, k, v, bias, qc, 256, 3)
            eq = "torch.equal" if torch.equal(o, pref) else "DIFFERS rmsd/std %.6f" % float(
                (o.float() - pref.float()).pow(2).mean().sqrt() / pref.float().std())
            res["above_cap"].append({"seq": s, "q": qc, "k": 256, "ms": round(ms, 4),
                                     "tflops": round(gf / (ms / 1e3) / 1e3, 2),
                                     "ratio": round(pms / ms, 4), "exact": eq,
                                     "padded_q": -(-s // qc) * qc, "n_chunks": -(-s // qc)})
            print("seq=768 CAND q%-4d k256 %9.4f ms %6.2f TF/s ratio %.4fx padded_q %d  %s" %
                  (qc, ms, gf / (ms / 1e3) / 1e3, pms / ms, -(-s // qc) * qc, eq), flush=True)
        except Exception as e:
            res["above_cap"].append({"seq": s, "q": qc, "k": 256, "error": str(e)[:200]})
            print("seq=768 CAND q%-4d k256 ILLEGAL %s" % (qc, str(e)[:120]), flush=True)
    for t in (q, k, v, bias):
        ttnn.deallocate(t)
    json.dump(res, open(OUT, "w"), indent=1)

    # --- S3: seq 1024
    s = 1024
    q, k, v = (ttnn.from_torch(torch.randn(s, 8, s, 32) * 0.1, layout=ttnn.TILE_LAYOUT,
                              device=dev, dtype=ttnn.bfloat16) for _ in range(3))
    bias = ttnn.from_torch(torch.randn(1, 8, s, s) * 0.1, layout=ttnn.TILE_LAYOUT,
                           device=dev, dtype=ttnn.bfloat16)
    gf = 4 * s * 8 * s * s * 32 / 1e9
    pms, pref = run(dev, grid, q, k, v, bias, 256, 256, 3)
    print("seq=1024 PROD q256 k256 %9.4f ms %6.2f TF/s" % (pms, gf / (pms / 1e3) / 1e3), flush=True)
    res["above_cap"].append({"seq": s, "q": 256, "k": 256, "ms": round(pms, 4), "ref": True,
                             "tflops": round(gf / (pms / 1e3) / 1e3, 2)})
    for qc in (512, 640):
        try:
            ms, o = run(dev, grid, q, k, v, bias, qc, 256, 3)
            eq = "torch.equal" if torch.equal(o, pref) else "DIFFERS rmsd/std %.6f" % float(
                (o.float() - pref.float()).pow(2).mean().sqrt() / pref.float().std())
            res["above_cap"].append({"seq": s, "q": qc, "k": 256, "ms": round(ms, 4),
                                     "tflops": round(gf / (ms / 1e3) / 1e3, 2),
                                     "ratio": round(pms / ms, 4), "exact": eq,
                                     "padded_q": -(-s // qc) * qc, "n_chunks": -(-s // qc)})
            print("seq=1024 CAND q%-4d k256 %9.4f ms %6.2f TF/s ratio %.4fx padded_q %d  %s" %
                  (qc, ms, gf / (ms / 1e3) / 1e3, pms / ms, -(-s // qc) * qc, eq), flush=True)
        except Exception as e:
            res["above_cap"].append({"seq": s, "q": qc, "k": 256, "error": str(e)[:200]})
            print("seq=1024 CAND q%-4d k256 ILLEGAL %s" % (qc, str(e)[:120]), flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
    ttnn.close_device(dev)
    print("wrote", OUT, flush=True)


main()
