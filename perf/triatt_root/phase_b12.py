"""Phase B1+B2: is q_chunk = seq bit-exact against the config PRODUCTION actually runs, and
where does it stop being legal.

B1 is the correction of phase_a3's one methodological hole: at seq 320/384 production runs q64/k64,
and phase_a3 compared q(seq)/k64 against a q256/k256 reference. Wrong reference -- it cannot prove
the swap production would make is free. Reference here is the production pick per size, taken from
tt_bio.tenstorrent._tri_att_sdpa_program_config, and the only accepted verdict is torch.equal.

PREDICTIONS, written before the run:
  R1  q(seq)/k64 at seq 320 is torch.equal to q64/k64. Mechanism: q only splits output rows, the
      online softmax reduces over k, and k_chunk is held at 64 -- so no reduction order changes.
      This is the same argument that held torch.equal at 384/512/640 against a matched k_chunk.
  R2  the 1.529x at seq 320 reproduces within +-5% (phase_a3 measured 2.9157 -> 1.9064 ms).
  R3  q_chunk = seq throws a circular-buffer clash somewhere above 640. q=k=512 already throws at
      512 (2114048 B vs 1572864 B max L1), and CB size scales with q_chunk*k_chunk, so with
      k pinned at 256 the q term grows linearly: the first failure should be 1024 or 1536, not 768.
  R4  every legal q(seq)/k(prod) is >= 1.0x of production, i.e. the lever never loses. Least
      confident of the four: at seq 256 production ALREADY has q=k=256=seq, so that row is a
      no-op identity check (ratio 1.000) and is the control.
"""
import json, sys, time
from importlib.metadata import version as _v
import torch, ttnn
TTNN_V = _v("ttnn")

OUT = sys.argv[1]
KC = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
    fp32_dest_acc_en=True, packer_l1_acc=True)

# Exactly tt_bio.tenstorrent's production policy for the tri-att SDPA, inlined so this script has
# no import dependency on the repo module while it is being edited by C1.
SDPA_CHUNK_MAX, SDPA_CHUNK_TILE = 256, 32


def prod_pick(seq):
    if 256 < seq <= 384:
        return (64, 64)
    c = min(SDPA_CHUNK_MAX, ((seq + SDPA_CHUNK_TILE - 1) // SDPA_CHUNK_TILE) * SDPA_CHUNK_TILE)
    return (c, c)


def timed(dev, fn, iters=5):
    fn(); ttnn.synchronize_device(dev)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter(); fn(); ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


def run(dev, grid, q, k, v, bias, qc, kc, iters=5):
    prog = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=grid, exp_approx_mode=False,
                                  q_chunk_size=qc, k_chunk_size=kc)
    call = lambda: ttnn.transformer.scaled_dot_product_attention(
        q, k, v, attn_mask=bias, is_causal=False, program_config=prog, compute_kernel_config=KC)
    ms = timed(dev, lambda: ttnn.deallocate(call()), iters)
    o = call()
    t = ttnn.to_torch(o)
    ttnn.deallocate(o)
    return ms, t


def main():
    dev = ttnn.open_device(device_id=0)
    grid = dev.compute_with_storage_grid_size()
    print("grid %dx%d = %d cores, ttnn %s" % (grid.x, grid.y, grid.x * grid.y, TTNN_V),
          flush=True)
    res = {"grid": [grid.x, grid.y], "ttnn": TTNN_V, "rows": []}

    for s in (256, 320, 352, 384, 448, 512, 576, 640, 768, 1024, 1536):
        b, h, d = s, 8, 32
        gf = 4 * b * h * s * s * d / 1e9  # 2 matmuls, 2 flop each
        try:
            q, k, v = (ttnn.from_torch(torch.randn(b, h, s, d) * 0.1, layout=ttnn.TILE_LAYOUT,
                                       device=dev, dtype=ttnn.bfloat16) for _ in range(3))
            bias = ttnn.from_torch(torch.randn(1, h, s, s) * 0.1, layout=ttnn.TILE_LAYOUT,
                                   device=dev, dtype=ttnn.bfloat16)
        except Exception as e:
            print("s=%4d ALLOC FAIL %s" % (s, str(e)[:120]), flush=True)
            res["rows"].append({"seq": s, "alloc_error": str(e)[:200]})
            break
        pqc, pkc = prod_pick(s)
        iters = 5 if s <= 640 else 3
        row = {"seq": s, "prod_q": pqc, "prod_k": pkc}
        try:
            pms, pref = run(dev, grid, q, k, v, bias, pqc, pkc, iters)
            row["prod_ms"] = round(pms, 4)
            row["prod_tflops"] = round(gf / (pms / 1e3) / 1e3, 2)
            print("s=%4d PROD  q%-4d k%-4d %9.4f ms %6.2f TF/s" %
                  (s, pqc, pkc, pms, gf / (pms / 1e3) / 1e3), flush=True)
        except Exception as e:
            row["prod_error"] = str(e)[:200]
            pref = None
            print("s=%4d PROD  q%-4d k%-4d ERR %s" % (s, pqc, pkc, str(e)[:100]), flush=True)
        # the C1 candidate: q_chunk = padded seq, k_chunk = the production k (unchanged)
        cqc = ((s + SDPA_CHUNK_TILE - 1) // SDPA_CHUNK_TILE) * SDPA_CHUNK_TILE
        row["cand_q"], row["cand_k"] = cqc, pkc
        if (cqc, pkc) == (pqc, pkc):
            row["note"] = "candidate == production (control row)"
            row["exact"] = "identity"
            print("       cand == prod, control row", flush=True)
        else:
            try:
                cms, co = run(dev, grid, q, k, v, bias, cqc, pkc, iters)
                row["cand_ms"] = round(cms, 4)
                row["cand_tflops"] = round(gf / (cms / 1e3) / 1e3, 2)
                if pref is not None:
                    if torch.equal(co, pref):
                        row["exact"] = "torch.equal"
                    else:
                        d0 = (co.float() - pref.float())
                        row["exact"] = "DIFFERS"
                        row["rmsd_over_std"] = round(float(d0.pow(2).mean().sqrt()
                                                           / pref.float().std()), 8)
                        row["max_abs"] = round(float(d0.abs().max()), 8)
                    row["ratio"] = round(pms / cms, 4)
                print("       CAND  q%-4d k%-4d %9.4f ms %6.2f TF/s  ratio %.4fx  %s" %
                      (cqc, pkc, cms, gf / (cms / 1e3) / 1e3, row.get("ratio", 0),
                       row.get("exact", "-")), flush=True)
            except Exception as e:
                row["cand_error"] = str(e)[:200]
                print("       CAND  q%-4d k%-4d ILLEGAL %s" % (cqc, pkc, str(e)[:110]), flush=True)
        res["rows"].append(row)
        for t in (q, k, v, bias):
            ttnn.deallocate(t)
        json.dump(res, open(OUT, "w"), indent=1)

    json.dump(res, open(OUT, "w"), indent=1)
    ttnn.close_device(dev)
    print("wrote", OUT, flush=True)


main()
