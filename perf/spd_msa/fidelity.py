"""spd-msa: math fidelity against time and float64 error on OuterProductMean's two matmuls.

* zmm: the depth contraction a [R*32, S] x b [S, J*32]^T at the campaign cell (736 tokens, depth 9947), R token
  rows (default 736 = the whole call), randn operands scaled like layer-normed c=32 projections.
* zmm_b8: the same contraction with both operands cast to bfp8 on device first (the `opm_b8` lever), at every
  fidelity including LoFi; the casts are timed on their own (ev=zmm_b8_cast) since the fold pays them per call.
* proj: the output projection [541696, 1024] x [1024, 256] (batched view, as shipped).
For each of HiFi4 / HiFi3 / HiFi2 (fp32 dest acc on, packer L1 acc on, as the trunk runs): median ms of REPS synced
calls and the error against a float64 product on the first 256 output rows (zmm) or all rows (proj), with the
count of elements off by more than 5 %, since HiFi4 was seen to put a few dozen of 138M outputs off by 1-4.

usage: TT_VISIBLE_DEVICES=<chip> python fidelity.py OUT [zmm,zmm_b8,proj] [R=736] [REPS=5]
"""
import json, statistics, sys, time
from pathlib import Path

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
WHAT = (sys.argv[2] if len(sys.argv) > 2 else "zmm,proj").split(",")
R = int(sys.argv[3]) if len(sys.argv) > 3 else 736
REPS = int(sys.argv[4]) if len(sys.argv) > 4 else 5
LOG = open(OUT / "fidelity.jsonl", "a")


def log(**kw):
    kw["t_unix"] = time.time(); line = json.dumps(kw, default=str)
    LOG.write(line + "\n"); LOG.flush(); print(line, flush=True)


from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
import tt_bio.tenstorrent as T

dev = T.get_device()
F = ttnn.MathFidelity
FIDS = [("hifi4", F.HiFi4), ("hifi3", F.HiFi3), ("hifi2", F.HiFi2)]


def ckc(fid):
    return ttnn.WormholeComputeKernelConfig(math_fidelity=fid, math_approx_mode=False,
                                           fp32_dest_acc_en=True, packer_l1_acc=True)


def grade(o, ref):
    d = (o - ref).abs()
    return dict(rel_rms=float(d.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()), max_abs=float(d.max()),
                n_bad=int((d > 0.05 * (ref.abs() + ref.abs().mean())).sum()), finite=bool(torch.isfinite(o).all()))


def sweep(name, call, ref, rows_of, fids=FIDS):
    for fname, fid in fids:
        try:
            out = call(ckc(fid)); ttnn.synchronize_device(dev)
            ts = []
            for _ in range(REPS):
                ttnn.deallocate(out)
                t0 = time.perf_counter(); out = call(ckc(fid)); ttnn.synchronize_device(dev)
                ts.append((time.perf_counter() - t0) * 1e3)
            o = rows_of(out); ttnn.deallocate(out)
            log(ev=name, fid=fname, ms=ts, ms_med=statistics.median(ts), **grade(o, ref))
        except Exception as e:
            log(ev=name + "_fail", fid=fname, err=str(e)[:300])


torch.manual_seed(0)
if "zmm" in WHAT or "zmm_b8" in WHAT:
    S, J, C = 9947, 736, 32
    a_h = torch.randn(R * C, S).bfloat16(); b_h = torch.randn(J * C, S).bfloat16()
    a = ttnn.from_torch(a_h, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    b = ttnn.from_torch(b_h, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    ref = (a_h[:256].double() @ b_h.double().t())
    if "zmm" in WHAT:
        sweep("zmm", lambda k: ttnn.matmul(a, b, transpose_b=True, compute_kernel_config=k), ref,
              lambda o: ttnn.to_torch(o[:256]).double())
    if "zmm_b8" in WHAT:
        ts = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            a8 = ttnn.typecast(a, ttnn.bfloat8_b); b8 = ttnn.typecast(b, ttnn.bfloat8_b)
            ttnn.synchronize_device(dev); ts.append((time.perf_counter() - t0) * 1e3)
            if len(ts) < REPS:
                ttnn.deallocate(a8); ttnn.deallocate(b8)
        log(ev="zmm_b8_cast", ms=ts, ms_med=statistics.median(ts))
        sweep("zmm_b8", lambda k: ttnn.matmul(a8, b8, transpose_b=True, compute_kernel_config=k), ref,
              lambda o: ttnn.to_torch(o[:256]).double(), FIDS + [("lofi", F.LoFi)])
        ttnn.deallocate(a8); ttnn.deallocate(b8)
    ttnn.deallocate(a); ttnn.deallocate(b); del a_h, b_h, ref
if "proj" in WHAT:
    ROWS, K, N = 541696, 1024, 256
    x_h = torch.randn(ROWS, K).bfloat16(); w_h = (torch.randn(K, N) / 32).bfloat16()
    x = ttnn.reshape(ttnn.from_torch(x_h, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16),
                     (46, ROWS // 46, K))
    w = ttnn.from_torch(w_h, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    ref = x_h.double() @ w_h.double()
    sweep("proj", lambda k: ttnn.linear(x, w, compute_kernel_config=k), ref,
          lambda o: ttnn.to_torch(o).double().reshape(ROWS, N))
log(ev="end")
