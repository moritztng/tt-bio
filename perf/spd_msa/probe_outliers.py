"""spd-msa: which compute setting puts a few outputs of a large bf16 matmul off by 1-4 against float64?

[ROWS, 1024] x [1024, 256], randn inputs (weights / 32), the shape of OuterProductMean's output projection.
Every production config (HiFi4, fp32 dest acc, packer L1 acc) shows 27 of 138M elements off by more than
5 % (e.g. -1.24 read as -2.23), identical across two chips, so arithmetic, not a fault. Each arm toggles one
setting; output dtype fp32 tells packing apart from accumulation.

usage: TT_VISIBLE_DEVICES=<chip> python probe_outliers.py OUT [ROWS=541696]
"""
import json, sys, time
from pathlib import Path

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
ROWS = int(sys.argv[2]) if len(sys.argv) > 2 else 541696
LOG = open(OUT / "outliers.jsonl", "a")


def log(**kw):
    kw["t_unix"] = time.time(); line = json.dumps(kw, default=str)
    LOG.write(line + "\n"); LOG.flush(); print(line, flush=True)


from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
import tt_bio.tenstorrent as T

dev = T.get_device()
K, N = 1024, 256
torch.manual_seed(0)
x_h = torch.randn(ROWS, K).bfloat16(); w_h = (torch.randn(K, N) / 32).bfloat16()
x = ttnn.from_torch(x_h, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
w = ttnn.from_torch(w_h, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
ref = x_h.double() @ w_h.double()
F = ttnn.MathFidelity
arms = [("prod", F.HiFi4, True, True, ttnn.bfloat16), ("no_l1acc", F.HiFi4, True, False, ttnn.bfloat16),
        ("no_fp32acc", F.HiFi4, False, False, ttnn.bfloat16), ("out_fp32", F.HiFi4, True, True, ttnn.float32),
        ("out_fp32_no_l1acc", F.HiFi4, True, False, ttnn.float32), ("hifi2", F.HiFi2, True, True, ttnn.bfloat16)]
for name, fid, f32, l1, odt in arms:
    try:
        ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=fid, math_approx_mode=False,
                                               fp32_dest_acc_en=f32, packer_l1_acc=l1)
        for view in ("2d", "b32"):
            xi = x if view == "2d" else ttnn.reshape(x, (32, ROWS // 32, K))
            o = ttnn.linear(xi, w, compute_kernel_config=ckc, dtype=odt)
            o_h = ttnn.to_torch(o).double().reshape(ROWS, N); ttnn.deallocate(o)
            d = (o_h - ref).abs()
            bad = (d > 0.05 * (1 + ref.abs())).nonzero()
            log(ev="arm", arm=name, view=view, n_bad=int(bad.shape[0]), max_abs=float(d.max()),
                rel_rms=float(d.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()),
                bad_vals=[(int(i), int(j), float(ref[i, j]), float(o_h[i, j])) for i, j in bad[:5].tolist()])
    except Exception as e:
        log(ev="arm_fail", arm=name, err=str(e)[:300])
log(ev="end")
