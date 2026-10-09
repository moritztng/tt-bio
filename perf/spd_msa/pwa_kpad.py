"""spd-msa: PairWeightedAveraging's head matmul [H, hd*rows, T] x [H, T, T]^T with K (the token axis j) at 23 tiles
(736 tokens) against 24 and 32 tiles of zero padding. Both operands are batched over the heads, so ttnn's
batched config takes in0_block_w = 2 only for an even Kt. Time per call and the max difference from K = 23.

usage: TT_VISIBLE_DEVICES=<chip> python pwa_kpad.py OUT [ROWS=512] [REPS=7]
"""
import json, statistics, sys, time
from pathlib import Path

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
ROWS = int(sys.argv[2]) if len(sys.argv) > 2 else 512
REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 7
LOG = open(OUT / "pwa_kpad.jsonl", "a")


def log(**kw):
    kw["t_unix"] = time.time(); line = json.dumps(kw, default=str)
    LOG.write(line + "\n"); LOG.flush(); print(line, flush=True)


from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
import tt_bio.tenstorrent as T

dev = T.get_device()
ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                             math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)
H, hd, Tk = 8, 8, 736
torch.manual_seed(0)
v_h = torch.randn(H, hd * ROWS, Tk).bfloat16()
w_h = torch.softmax(torch.randn(H, Tk, Tk), -1).bfloat16()
log(ev="start", rows=ROWS, arch=str(dev.arch()))
base = None
for kp in (Tk, 768, 1024):
    v = ttnn.from_torch(torch.nn.functional.pad(v_h, (0, kp - Tk)), layout=ttnn.TILE_LAYOUT, device=dev,
                        dtype=ttnn.bfloat16)
    w = ttnn.from_torch(torch.nn.functional.pad(w_h, (0, kp - Tk)), layout=ttnn.TILE_LAYOUT, device=dev,
                        dtype=ttnn.bfloat16)
    o = ttnn.matmul(v, w, transpose_b=True, compute_kernel_config=ckc); ttnn.synchronize_device(dev)
    ts = []
    for _ in range(REPS):
        ttnn.deallocate(o)
        t0 = time.perf_counter()
        o = ttnn.matmul(v, w, transpose_b=True, compute_kernel_config=ckc); ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t0) * 1e3)
    oh = ttnn.to_torch(o); ttnn.deallocate(o); ttnn.deallocate(v); ttnn.deallocate(w)
    rec = dict(ev="arm", k=kp, kt=kp // 32, ms_med=statistics.median(ts), spread=max(ts) - min(ts))
    if base is None:
        base = oh
    else:
        rec["max_abs_vs_k23"] = float((oh.double() - base.double()).abs().max())
    log(**rec)
log(ev="end")
