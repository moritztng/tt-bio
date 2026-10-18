#!/usr/bin/env python3
"""ESMFold2's token-DiT attention core as one program: `_attn_fp32` (9 ops) vs the shared `_sdpa32` flash program.

The token AttentionPairBias spends ~1 ms of its ~1.4 ms a call (384 aa, Wormhole) in `esmfold2._attn_fp32`:
typecasts, q.k^T, scale_add with the pair bias, softmax and attn.v, all fp32 with the [1, 16, L, L] scores in
DRAM between them. Protenix, Boltz-2 and OpenFold3 already run their fp32 DiT attention as one flash-SDPA program
(`tenstorrent._sdpa32`, lever dit_sdpa32); ESMFold2 has its own AttentionPairBias and never reached it. Its logits
are not qk-normed and reach ~100, which is why it was kept on the fp32 chain, so the question is accuracy first.

Inside the first draw of a real fold, at diffusion step STEP, the inputs of one `_attn_fp32` call per BLOCKS entry
(q, k, v [1, 16, L, 64] bf16 with zero head padding, the cached pair bias) are captured. Per arm: float64 rel_rms
and max |err| against a float64 softmax(q k^T * scale + bias) v of the same bf16 inputs, run-to-run equality,
eager and trace-replayed ms a call. Arms:

  chain   esmfold2._attn_fp32, what ships;
  s32     tenstorrent._sdpa32 on fp32 q/k/v, mask = bias / scale in fp32 (SDPA scales its mask with the scores),
          token axis padded by sdpa32_rows / sdpa32_mask;
  s32st   the same with the five row-statistics CBs fp32 (sdpa_generic stats_dtype).

    python perf/spd/ef2_attn32_probe.py --model esmfold2-fast --fast --size 384 --out r.json
"""
import argparse, json, os, statistics, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="esmfold2-fast", choices=("esmfold2", "esmfold2-fast"))
ap.add_argument("--fast", action="store_true")
ap.add_argument("--size", type=int, default=384)
ap.add_argument("--step", type=int, default=34)
ap.add_argument("--blocks", default="0,6,11")
ap.add_argument("--reps", type=int, default=20)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch
import ttnn
import tt_bio.tenstorrent as T
import tt_baseline as B
from tt_bio.main import _resolve_recycling_steps, _resolve_sampling_steps
from tt_bio import esmfold2 as E
from tt_bio import sdpa_generic as SG

B.RECYCLING_STEPS = _resolve_recycling_steps(None, a.model)
B.SAMPLING_STEPS = _resolve_sampling_steps(None, a.model)
fix = ROOT / "perf" / "size512" / "fixtures"
one_fold, meta = B.build_fold(a.model, ROOT / f".msa_ab_{a.model}_{a.size}", fix / f"cdk2x2_{a.size}.yaml",
                              fix / f"cdk2x2_{a.size}.a3m", fast=a.fast)[:2]
res = {"model": a.model, "fast": a.fast, "size": a.size, "card": os.environ.get("TT_VISIBLE_DEVICES"),
       "host": os.uname().nodename, "step": a.step, "calls": []}
F32 = ttnn.float32
blocks = {int(b) for b in a.blocks.split(",")}
calls = [0]


def free(o):
    for t in (o if isinstance(o, (tuple, list)) else (o,)):
        if isinstance(t, ttnn.Tensor) and t.is_allocated():
            ttnn.deallocate(t)


def timed(dev, fn, n):
    """(eager ms, traced ms or None) a call."""
    for _ in range(2):
        free(fn())
    ts = []
    for _ in range(3):
        ttnn.synchronize_device(dev); t0 = time.perf_counter()
        for _ in range(n):
            free(fn())
        ttnn.synchronize_device(dev); ts.append((time.perf_counter() - t0) / n * 1e3)
    eager, traced = statistics.median(ts), None
    if T.trace_region_size() > 0:
        try:
            tid, out = T.capture_trace(dev, fn)
            ts = []
            for _ in range(3):
                ttnn.synchronize_device(dev); t0 = time.perf_counter()
                for _ in range(n):
                    ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(dev); ts.append((time.perf_counter() - t0) / n * 1e3)
            traced = statistics.median(ts)
            ttnn.release_trace(dev, tid); free(out)
        except Exception as e:
            print("trace refused:", type(e).__name__, str(e)[:160], flush=True)
    return round(eager, 4), (None if traced is None else round(traced, 4))


def s32_arm(q, k, v, bias, scale, stats):
    """`_sdpa32` as AttentionPairBias would call it: fp32 operands, mask = bias / scale, rows padded to 128."""
    dev = q.device()
    n = int(q.shape[2]); npad = T.sdpa32_rows(n)
    qf, kf, vf = (ttnn.typecast(t, F32) for t in (q, k, v))
    if npad != n:
        p = [(0, 0), (0, 0), (0, npad - n), (0, 0)]
        qf, kf, vf = (ttnn.pad(t, p, value=0.0) for t in (qf, kf, vf))
    mb = ttnn.multiply(ttnn.typecast(bias, F32), 1.0 / scale)
    m = T.sdpa32_mask(mb) if npad != n else mb

    def run():
        if not stats:
            o = T._sdpa32(qf, kf, vf, m, scale)
        else:
            chunk, k_chunk, grid = T._sdpa32_chunks(dev, int(qf.shape[0]), int(qf.shape[1]), npad)
            o = ttnn.allocate_tensor_on_device(ttnn.Shape([int(qf.shape[i]) for i in range(3)] + [int(vf.shape[3])]),
                                               F32, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG)
            SG.sdpa(dev, qf, kf, vf, m, o, chunk, k_chunk, grid, T._SDPA32_CKC, scale, stats_dtype=F32)
        if npad != n:
            o2 = ttnn.slice(o, [0, 0, 0, 0], [int(o.shape[0]), int(o.shape[1]), n, int(o.shape[3])])
            ttnn.deallocate(o); o = o2
        out = ttnn.typecast(o, E._DTYPE); ttnn.deallocate(o)
        return out
    return run, [qf, kf, vf, mb, m]


def grade(o, ref):
    o = ttnn.to_torch(o).double()[..., :ref.shape[-1]]
    d = o - ref
    return o, {"rel_rms": float(d.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()),
               "max_abs": float(d.abs().max()), "finite": bool(torch.isfinite(o).all())}


def probe(q, k, v, bias, scale, ck, accurate_softmax):
    dev = q.device()
    tq, tk, tv, tb = (ttnn.to_torch(t).double() for t in (q, k, v, bias))
    logits = tq @ tk.transpose(-1, -2) * scale + tb
    ref = torch.softmax(logits, dim=-1) @ tv
    r = {"call": calls[0], "shape": list(q.shape), "logit_absmax": float(logits.abs().max()),
         "logit_rowspan_max": float((logits.max(-1).values - logits.min(-1).values).max()), "arms": {}}
    arms = {"chain": (lambda: orig(q, k, v, bias, scale, ck, accurate_softmax), [])}
    for name, st in (("s32", False), ("s32st", True)):
        try:
            arms[name] = s32_arm(q, k, v, bias, scale, st)
        except Exception as e:
            r["arms"][name] = {"error": f"{type(e).__name__}: {str(e)[:240]}"}
    for name, (fn, keep) in arms.items():
        try:
            o1 = fn(); o2 = fn()
            t1, g = grade(o1, ref); t2 = ttnn.to_torch(o2).double()[..., :ref.shape[-1]]
            free([o1, o2])
            g["repeat_equal"] = bool(torch.equal(t1, t2))
            g["eager_ms"], g["traced_ms"] = timed(dev, fn, a.reps)
            r["arms"][name] = g
        except Exception as e:
            r["arms"][name] = {"error": f"{type(e).__name__}: {str(e)[:240]}"}
        free(keep)
        print(name, json.dumps(r["arms"][name]), flush=True)
    res["calls"].append(r)
    a.out.write_text(json.dumps(res, indent=1))


orig = E._attn_fp32


def hooked(q, k, v, attn_mask, scale, ck, accurate_softmax=False):
    i = calls[0]; calls[0] += 1
    if i // 12 == a.step and i % 12 in blocks:
        probe(q, k, v, attn_mask, scale, ck, accurate_softmax)
    return orig(q, k, v, attn_mask, scale, ck, accurate_softmax)


E._attn_fp32 = hooked
t0 = time.perf_counter(); one_fold(); res["fold_s"] = round(time.perf_counter() - t0, 3)
res["attn_calls"] = calls[0]
a.out.write_text(json.dumps(res, indent=1))
print("done", res["fold_s"], calls[0], flush=True)
T.cleanup()
