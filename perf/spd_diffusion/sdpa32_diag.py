"""Which CB format breaks sdpa_generic at fp32? One config (q256 k128), zero bias + padding mask, a dtype
matrix (operands, mask, score CB, output accumulators, statistics, destination) against float64.
usage: TT_VISIBLE_DEVICES=N python sdpa32_diag.py OUT.jsonl [NT]"""
import json, sys, time
from pathlib import Path

import torch

OUT = Path(sys.argv[1]); NT = int(sys.argv[2]) if len(sys.argv) > 2 else 730
M, H, D, NP = 1, 16, 48, 768
try:
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
except Exception:
    pass
import ttnn
import tt_bio.tenstorrent as T
from tt_bio import sdpa_generic as SG
from tt_bio.eltwise_fusion import scale_add

dev = T.get_device()
g_ = dev.compute_with_storage_grid_size(); GRID = (g_.x, g_.y)
LOG = open(OUT, "a")
F, B16 = ttnn.float32, ttnn.bfloat16
up = lambda t, dt: ttnn.from_torch(t.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=dt)
host = lambda t: torch.Tensor(ttnn.to_torch(t)).double()
s = D ** -0.5
g = torch.Generator().manual_seed(0)
q_h, k_h, v_h = (torch.randn(M, H, NT, D, generator=g) * 2 for _ in range(3))
pad = lambda t: torch.cat([t, torch.zeros(*t.shape[:2], NP - NT, D)], 2)
mask_h = torch.full((1, H, NP, NP), -1e4); mask_h[:, :, :, :NT] = 0
LS = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
q_h, k_h, v_h = q_h / 2 * LS, k_h / 2 * LS, v_h / 2 * LS
ref = torch.softmax(torch.einsum("mhid,mhjd->mhij", q_h.double(), k_h.double()) * s, -1) @ v_h.double()
# (operands, mask, im, out_im, stats, out)
ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                             fp32_dest_acc_en=True, packer_l1_acc=True)
q, k, v = up(q_h, F), up(k_h, F), up(v_h, F)
zb = up(torch.zeros(1, H, NT, NT), F)
sc = ttnn.matmul(q, k, transpose_b=True, core_grid=T.CORE_GRID_MAIN, compute_kernel_config=ckc)
pr = ttnn.softmax(scale_add(sc, s, zb), dim=-1, compute_kernel_config=ckc)
o = torch.Tensor(ttnn.to_torch(T.batched_matmul(pr, v, compute_kernel_config=ckc))).double()[..., :NT, :D]
rec = dict(case="SHIPPED fp32 chain", LS=LS, max_abs=float((o - ref).abs().max()), mean_abs=float((o - ref).abs().mean()))
LOG.write(json.dumps(rec) + "\n"); LOG.flush(); print(json.dumps(rec), flush=True)
MATRIX = [("bf16 all", B16, B16, None, None, None, B16),
          ("bf16 in, fp32 im/oim/stats", B16, B16, F, F, F, B16),
          ("bf16 in, fp32 out", B16, B16, None, None, None, F),
          ("fp32 in, bf16 mask, bf16 im", F, B16, None, None, None, F),
          ("fp32 in+mask, bf16 im", F, F, None, None, None, F),
          ("fp32 in+mask, fp32 stats only", F, F, None, None, F, F),
          ("fp32 in+mask, fp32 oim only", F, F, None, F, None, F),
          ("fp32 in+mask, fp32 oim+stats", F, F, None, F, F, F),
          ("fp32 in+mask, fp32 im only", F, F, F, None, None, F),
          ("fp32 in+mask, fp32 im+oim", F, F, F, F, None, F),
          ("fp32 all", F, F, F, F, F, F)]
for qc, kc in [(256, 128)]:
    for name, dt, mdt, im, oim, st, odt in MATRIX:
        try:
            qp, kp, vp, mask = up(pad(q_h), dt), up(pad(k_h), dt), up(pad(v_h), dt), up(mask_h, mdt)
            out = ttnn.allocate_tensor_on_device(ttnn.Shape([M, H, NP, D]), odt, ttnn.TILE_LAYOUT, dev,
                                                 ttnn.DRAM_MEMORY_CONFIG)
            SG.sdpa(dev, qp, kp, vp, mask, out, qc, kc, GRID, (ttnn.MathFidelity.HiFi4, False, True, False), s,
                    im_dtype=im, out_im_dtype=oim, stats_dtype=st)
            o = host(out)[..., :NT, :D]
            for t in (qp, kp, vp, mask, out):
                ttnn.deallocate(t)
            alpha = (o * ref).sum(-1) / (o * o).sum(-1).clamp_min(1e-30)     # per-row best scale o -> ref
            rec = dict(case=name, qc=qc, kc=kc, LS=LS, max_abs=float((o - ref).abs().max()),
                       mean_abs=float((o - ref).abs().mean()), finite=bool(torch.isfinite(o).all()),
                       row_scale_med=float(alpha.median()) if torch.isfinite(alpha).all() else None,
                       max_abs_after_row_scale=float((o * alpha[..., None] - ref).abs().max()))
        except Exception as e:
            rec = dict(case=name, qc=qc, kc=kc, error=str(e)[:300])
        rec["t_unix"] = time.time()
        LOG.write(json.dumps(rec) + "\n"); LOG.flush(); print(json.dumps(rec), flush=True)
