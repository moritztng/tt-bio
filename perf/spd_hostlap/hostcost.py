"""Host-only cost of the one-off host stalls in a Protenix-v2 c730 fold, at the serving thread share.

Every piece is the torch work plus ttnn's host-side tensor build (tilize) that the fold does before the
chip gets the tensor. Run it pinned to one chip you hold (TT_VISIBLE_DEVICES): ttnn's host tensor
constructor initialises the metal context, and without a pin that opens every chip on the box. Shapes are c730's (736 padded tokens, 9,947 MSA rows, 4
templates, 5,919 atoms). Also checks how much of each piece a second Python thread can overlap with a
main thread that is blocked in a GIL-releasing call (what a host lane would see during a device read).

    TT_VISIBLE_DEVICES=<chip> OMP_NUM_THREADS=2 python perf/spd_hostlap/hostcost.py
"""
import os, threading, time
if not os.environ.get("TT_VISIBLE_DEVICES"):
    raise SystemExit("pin one chip with TT_VISIBLE_DEVICES first")
import torch, torch.nn.functional as F
import ttnn
from tt_bio.tenstorrent import get_device
get_device()

torch.manual_seed(0)
N, D, NT_TPL, NA, CZ = 736, 9947, 4, 5919, 256
print(f"threads {torch.get_num_threads()} ncpu {os.cpu_count()}", flush=True)


def tilize(t, dtype=ttnn.bfloat16):
    return ttnn.Tensor(tensor=t.contiguous(), data_type=dtype, layout=ttnn.TILE_LAYOUT)


def timed(name, fn, reps=2):
    best = 1e9
    for _ in range(reps):
        t0 = time.perf_counter(); out = fn(); best = min(best, time.perf_counter() - t0)
    print(f"{best:7.3f} s  {name}", flush=True)
    return out


feat = dict(template_distogram=torch.rand(NT_TPL, N, N, 39), template_pseudo_beta_mask=torch.ones(NT_TPL, N, N),
            template_aatype=torch.randint(0, 32, (NT_TPL, N)), template_unit_vector=torch.randn(NT_TPL, N, N, 3),
            template_backbone_frame_mask=torch.ones(NT_TPL, N, N), msa=torch.randint(0, 32, (D, N)),
            has_deletion=torch.rand(D, N).round(), deletion_value=torch.rand(D, N))
mc = torch.ones(N, N); pm = torch.ones(N, N)


def te_at(t, dt=torch.float32):
    dg = feat["template_distogram"][t] * mc[..., None] * pm[..., None]
    pb = (feat["template_pseudo_beta_mask"][t] * mc * pm).unsqueeze(-1)
    aa = F.one_hot(feat["template_aatype"][t].long(), 32).float()
    aai = aa[None].expand(N, N, 32); aaj = aa[:, None].expand(N, N, 32)
    uv = feat["template_unit_vector"][t] * mc[..., None] * pm[..., None]
    bb = (feat["template_backbone_frame_mask"][t] * mc * pm).unsqueeze(-1)
    return torch.cat([dg, pb, aai, aaj, uv, bb], -1).to(dt)


def ms_build():
    bf = torch.bfloat16
    msa = F.one_hot(feat["msa"].long(), 32).to(bf)
    return torch.cat([msa, feat["has_deletion"].to(bf).unsqueeze(-1),
                      feat["deletion_value"].float().to(bf).unsqueeze(-1)], -1).unsqueeze(0)


tes = timed("template te_at build x4 (fp32, as today)", lambda: [te_at(t) for t in range(NT_TPL)])
timed("template te_at -> bf16 x4", lambda: [t.to(torch.bfloat16) for t in tes])
teb = [t.to(torch.bfloat16) for t in tes]
timed("template host tilize bf16 x4", lambda: [tilize(t.unsqueeze(0)) for t in teb])
ms = timed("msa feature build bf16", ms_build)
timed("msa feature host tilize bf16 (937 MB tiled)", lambda: tilize(ms))
relp = torch.rand(N, N, 139)
timed("relp fp32 -> bf16 + tilize (trunk_input)", lambda: tilize(relp.to(torch.bfloat16)))
timed("relp fp32 tilize (diffusion fp32 relpe)", lambda: tilize(relp, ttnn.float32))
zt = tilize(torch.randn(1, 730, 730, CZ).to(torch.bfloat16))
timed("z_trunk to_torch (host tile bf16 -> torch) + float", lambda: torch.Tensor(zt.to_torch()).float())
z = torch.randn(730, 730, CZ); s_in = torch.randn(730, 449); w1 = torch.randn(CZ, 449); w2 = torch.randn(CZ, 449)
zb = timed("confidence z_base fp32 build", lambda: (z + F.linear(s_in, w1).unsqueeze(1) + F.linear(s_in, w2).unsqueeze(0)).unsqueeze(0).contiguous())
timed("confidence z_base tilize fp32->bf16 (as today)", lambda: tilize(zb.float()))
timed("confidence z_base torch bf16 + tilize", lambda: tilize(zb.to(torch.bfloat16)))
lw = torch.randn(CZ); wz = torch.randn(16, CZ)
timed("_plm_z_term LN + linear (pair_z)", lambda: F.linear(F.layer_norm(z, (CZ,)) * lw, wz))
timed("_dit_z_device LN (pair_z)", lambda: F.layer_norm(z, (CZ,)).unsqueeze(0).contiguous())
timed("_dit_z_device tilize fp32 -> fp32", lambda: tilize(F.layer_norm(z, (CZ,)).unsqueeze(0).contiguous(), ttnn.float32), reps=1)
lg = torch.randn(730, 730, 64)
lgt = tilize(lg.unsqueeze(0).to(torch.bfloat16))
timed("confidence logits to_torch x2 (pae+pde)", lambda: [torch.Tensor(lgt.to_torch()).float() for _ in range(2)])


def post():
    c = (torch.arange(64, dtype=torch.float32) + 0.5) * 0.5
    for _ in range(3):
        (torch.softmax(lg, -1) * c).sum(-1)
timed("confidence post-process (3 softmax over N,N,64)", post)

# Overlap check: the lane runs while the main thread sits in a GIL-releasing call (time.sleep stands in for
# a blocking device read). Wall ~= max(lane, sleep) means the piece overlaps; ~= sum means it serialized.
for name, fn in (("te_at build x4 + bf16", lambda: [te_at(t, torch.bfloat16) for t in range(NT_TPL)]),
                 ("msa tilize", lambda: tilize(ms)), ("z_trunk to_torch", lambda: torch.Tensor(zt.to_torch()).float())):
    t0 = time.perf_counter(); fn(); alone = time.perf_counter() - t0
    th = threading.Thread(target=fn); t0 = time.perf_counter(); th.start(); time.sleep(alone); th.join()
    print(f"overlap {name}: alone {alone:.3f} s, lane + main blocked {time.perf_counter() - t0:.3f} s", flush=True)
