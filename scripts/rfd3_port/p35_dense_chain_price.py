"""Price every op of RFD3's dense atom-attention chain on THIS card and wheel, and screen the
bit-exact fusion candidates against it.

The p29 attributed profile that the whole host-half brief is built on was taken on qb2 at ttnn
0.68.0 and predates p30 and p31. Everything downstream of it -- which line is biggest, what a
fusion is worth, whether a custom kernel is justified -- depends on numbers that have to be
re-measured here before anything is built (`perf-method-floor-screen-predict-then-build`).

Reproduces RFD3AtomBlock.__call__'s sparse-bias branch exactly (model.py:1293-1381) at the
production shape: L=3359 atoms, 4 heads, head_dim=32, key axis tile-aligned to 3360, 128
neighbours per row, bf16 storage with an fp32 score/softmax chain.

Every timed region is bracketed by ttnn.synchronize_device (`ttnn-sync-before-every-timed-region`
-- an unsynced drain inverts the ranking). The two clone rows are the MEASURED bandwidth roof for
this card; no op below is allowed to be read as "at the roof" against an asserted number
(`roofline-roof-must-be-measured-not-asserted`).

Candidates, all required to be bit-exact (torch.equal on the real 45.1 M-element tensor):
  F1  ttnn.add_ with the scale folded in as an input activation, replacing multiply + add.
      The recipe openfold3-sizes-perf measured as a bit-exact 1.246x substitute for the
      scale_mask_softmax it could not use (`ttnn-scale-mask-softmax-and-widen-in-binary-op`).
  F2  ttnn.softmax_in_place instead of ttnn.softmax (drops one live 180 MB fp32 copy).
  F3  the two constraints that memory says are closed, asserted once here so the record is
      this card's own: scale_mask_softmax on a per-row mask, and ttnn.scatter on fp32.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import ttnn  # noqa: E402

TILE = 32
REPS = 5


def align(n):
    return -(-n // TILE) * TILE


def timed(fn, reps=REPS):
    """Median wall of `fn`, sync-bracketed. Returns (ms, last_result)."""
    out = fn()
    ttnn.synchronize_device(DEV)
    ts = []
    for _ in range(reps):
        ttnn.synchronize_device(DEV)
        t0 = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(DEV)
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts), out


def gbs(ms, mb):
    return mb / 1e3 / (ms / 1e3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atoms", type=int, default=3359)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--head_dim", type=int, default=32)
    ap.add_argument("--keys", type=int, default=128)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    global DEV
    DEV = ttnn.open_device(device_id=0)
    L, H, DH, K = a.atoms, a.heads, a.head_dim, a.keys
    N = align(L)
    scale = DH ** -0.5
    rec: dict = {"shape": [1, H, L, N], "atoms": L, "heads": H, "head_dim": DH, "n_keys": K,
                 "elements": H * L * N, "ttnn": "0.67.4"}
    mb_bf16 = H * L * N * 2 / 1e6
    mb_f32 = H * L * N * 4 / 1e6
    print(f"[shape] [1,{H},{L},{N}] = {H * L * N / 1e6:.2f} M elem, "
          f"bf16 {mb_bf16:.1f} MB, fp32 {mb_f32:.1f} MB", flush=True)

    g = torch.Generator().manual_seed(0)
    # sorted neighbour indices, the real layout (_create_attention_indices ends in torch.sort)
    idx_t = torch.stack([torch.randperm(L, generator=g)[:K].sort().values for _ in range(L)])
    idx_t = idx_t.unsqueeze(0).unsqueeze(0).expand(1, H, L, K).contiguous().to(torch.int32)
    pb_t = torch.randn(1, H, L, K, generator=g) * 2.0
    sc_t = torch.randn(1, H, L, N, generator=g) * 3.0
    v_t = torch.randn(1, H, N, DH, generator=g)

    def up(t, dt):
        return ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=DEV)

    scores = up(sc_t, ttnn.bfloat16)
    pair_bias = up(pb_t, ttnn.bfloat16)
    vv = up(v_t, ttnn.bfloat16)
    idx = ttnn.from_torch(idx_t, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=DEV)
    tmpl = ttnn.full((1, H, L, N), -1e4, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=DEV)

    ops = {}

    # --- measured roof, both dtypes -------------------------------------------------
    ms, cl = timed(lambda: ttnn.clone(scores))
    ops["clone_bf16"] = {"ms": ms, "mb": 2 * mb_bf16, "gbs": gbs(ms, 2 * mb_bf16)}
    ttnn.deallocate(cl)
    sc32 = ttnn.typecast(scores, ttnn.float32)
    ms, cl = timed(lambda: ttnn.clone(sc32))
    ops["clone_fp32"] = {"ms": ms, "mb": 2 * mb_f32, "gbs": gbs(ms, 2 * mb_f32)}
    ttnn.deallocate(cl)

    # --- the shipped chain, op by op ------------------------------------------------
    ms, bias = timed(lambda: ttnn.scatter(tmpl, 3, idx, pair_bias))
    ops["scatter"] = {"ms": ms, "mb": 2 * mb_bf16, "gbs": gbs(ms, 2 * mb_bf16),
                      "gelem_s": (H * L * N) / (ms / 1e3) / 1e9}
    ms, bias_f = timed(lambda: ttnn.typecast(bias, ttnn.float32, memory_config=bias.memory_config()))
    ops["typecast_bias_up"] = {"ms": ms, "mb": mb_bf16 + mb_f32, "gbs": gbs(ms, mb_bf16 + mb_f32)}
    ms, s_f = timed(lambda: ttnn.typecast(scores, ttnn.float32, memory_config=scores.memory_config()))
    ops["typecast_scores_up"] = {"ms": ms, "mb": mb_bf16 + mb_f32, "gbs": gbs(ms, mb_bf16 + mb_f32)}
    ms, s_scaled = timed(lambda: ttnn.multiply(s_f, scale))
    ops["multiply_scale"] = {"ms": ms, "mb": 2 * mb_f32, "gbs": gbs(ms, 2 * mb_f32)}
    ms, s_sum = timed(lambda: ttnn.add(s_scaled, bias_f))
    ops["add_bias"] = {"ms": ms, "mb": 3 * mb_f32, "gbs": gbs(ms, 3 * mb_f32)}
    ms, attn = timed(lambda: ttnn.softmax(s_sum, dim=-1))
    ops["softmax_fp32"] = {"ms": ms, "mb": 2 * mb_f32, "gbs": gbs(ms, 2 * mb_f32)}
    ms, attn_bf = timed(lambda: ttnn.typecast(attn, ttnn.bfloat16, memory_config=attn.memory_config()))
    ops["typecast_attn_down"] = {"ms": ms, "mb": mb_bf16 + mb_f32, "gbs": gbs(ms, mb_bf16 + mb_f32)}
    ms, o = timed(lambda: ttnn.matmul(attn_bf, vv))
    ops["attn_at_v"] = {"ms": ms, "mb": mb_bf16, "gbs": gbs(ms, mb_bf16)}
    ttnn.deallocate(o)

    ref = ttnn.to_torch(attn)          # the fp32 softmax output every candidate must reproduce
    chain = ["scatter", "typecast_bias_up", "typecast_scores_up", "multiply_scale", "add_bias",
             "softmax_fp32", "typecast_attn_down", "attn_at_v"]
    rec["ops"] = ops
    rec["chain_ms"] = sum(ops[k]["ms"] for k in chain)
    for k in chain + ["clone_bf16", "clone_fp32"]:
        v = ops[k]
        print(f"  {k:22s} {v['ms']:7.3f} ms  {v['gbs']:6.1f} GB/s"
              + (f"  {v['gelem_s']:5.2f} G elem/s" if "gelem_s" in v else ""), flush=True)
    print(f"  {'CHAIN TOTAL':22s} {rec['chain_ms']:7.3f} ms/call", flush=True)

    cands: dict = {}

    # --- F1: fold the scale into the add as an input activation ---------------------
    def try_f1():
        forms = []
        try:
            forms.append(("UnaryWithParam", lambda: ttnn.add(
                s_f, bias_f, input_tensor_a_activations=[
                    ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, scale)])))
        except Exception as e:                                    # pragma: no cover
            cands["f1_ctor"] = f"UnaryWithParam unavailable: {e!r}"
        forms.append(("tuple", lambda: ttnn.add(
            s_f, bias_f, input_tensor_a_activations=[(ttnn.UnaryOpType.MUL_UNARY_SFPU, scale)])))
        for name, fn in forms:
            try:
                ms, out = timed(fn)
                eq = torch.equal(ttnn.to_torch(ttnn.softmax(out, dim=-1)), ref)
                cands[f"f1_add_scaled[{name}]"] = {"ms": ms, "bit_exact_after_softmax": eq,
                                                   "replaces_ms": ops["multiply_scale"]["ms"]
                                                   + ops["add_bias"]["ms"]}
                print(f"  F1 add+scale [{name}] {ms:7.3f} ms  bit-exact={eq}"
                      f"  (replaces {ops['multiply_scale']['ms'] + ops['add_bias']['ms']:.3f})",
                      flush=True)
                return
            except Exception as e:
                cands[f"f1_add_scaled[{name}]"] = f"REJECTED: {type(e).__name__}: {str(e)[:200]}"
                print(f"  F1 add+scale [{name}] REJECTED {str(e)[:140]}", flush=True)
    try_f1()

    # in-place form: writes into s_f, so it must be the last user of it
    try:
        ms, out = timed(lambda: ttnn.add(ttnn.multiply(s_f, scale), bias_f))
        cands["f1b_multiply_then_add_fresh"] = {"ms": ms}
    except Exception as e:
        cands["f1b_multiply_then_add_fresh"] = f"REJECTED: {e!r}"

    # --- F2: softmax_in_place --------------------------------------------------------
    try:
        src = ttnn.clone(s_sum)
        ms, out = timed(lambda: ttnn.softmax_in_place(ttnn.clone(s_sum)))
        eq = torch.equal(ttnn.to_torch(out), ref)
        cands["f2_softmax_in_place"] = {"ms": ms, "bit_exact": eq,
                                        "vs_softmax_ms": ops["softmax_fp32"]["ms"]}
        print(f"  F2 softmax_in_place  {ms:7.3f} ms (incl. one clone) bit-exact={eq}", flush=True)
        ttnn.deallocate(src)
    except Exception as e:
        cands["f2_softmax_in_place"] = f"REJECTED: {type(e).__name__}: {str(e)[:200]}"
        print(f"  F2 softmax_in_place REJECTED {str(e)[:140]}", flush=True)

    # --- F3: the two closed doors, asserted on this card ----------------------------
    try:
        out = ttnn.scale_mask_softmax(s_f, scale, bias_f)
        cands["f3_scale_mask_softmax"] = {"accepted": True,
                                          "bit_exact": torch.equal(ttnn.to_torch(out), ref)}
    except Exception as e:
        cands["f3_scale_mask_softmax"] = f"REJECTED: {type(e).__name__}: {str(e)[:240]}"
        print(f"  F3 scale_mask_softmax REJECTED {str(e)[:160]}", flush=True)
    try:
        t32 = ttnn.full((1, H, L, N), -1e4, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=DEV)
        pb32 = ttnn.typecast(pair_bias, ttnn.float32)
        ms, out = timed(lambda: ttnn.scatter(t32, 3, idx, pb32))
        cands["f3_scatter_fp32"] = {"ms": ms, "accepted": True}
        print(f"  F3 scatter fp32 accepted {ms:7.3f} ms", flush=True)
    except Exception as e:
        cands["f3_scatter_fp32"] = f"REJECTED: {type(e).__name__}: {str(e)[:240]}"
        print(f"  F3 scatter fp32 REJECTED {str(e)[:160]}", flush=True)

    rec["candidates"] = cands
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rec, indent=2, default=str))
    print(f"[done] {a.out}", flush=True)
    ttnn.close_device(DEV)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
