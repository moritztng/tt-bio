"""Does the fused SDPA's error grow with a common offset on v? (spd-q-bondradial)

The diffusion's token DiT feeds attention a v that carries absolute atom position, so a kernel error that scales
with |v| becomes a bond error that grows with distance from the origin. This probe shifts v by a constant C and
measures each attention arm against a float64 reference computed from the SAME (already rounded) operands, so
only the kernel's own arithmetic is graded.

usage: TT_VISIBLE_DEVICES=N python sdpa_offset.py OUT.jsonl

Arms: stock (ttnn fused SDPA, bf16, Boltz-2's token DiT), stock_vn (dit_vnorm: 1.0 in v's first pad lane, output
divided by it), s32 (`_sdpa32`, fp32 operands, bf16 CBs), s32_vn, s32_st32 (fp32 row statistics only),
s32_o32 (fp32 output accumulators), s32_o32_vn, s32_o32_st32, s32_all32 (score, output and statistics CBs fp32),
g16 / g16_o32 / g16_o32_vn / g16_st32 / g16_o32_st32 (bf16 operands through sdpa_generic, bf16 or fp32 output
accumulators and row statistics, fp32 output).
ARMS=a,b,... in the environment restricts the run to those arms.
Per arm and offset: rms and max |o - o64| over the real lanes, and us per call.
"""
import json, os, sys, time
from pathlib import Path

import torch

OUT = Path(sys.argv[1])
M, H, NT, D, DP, REPS = 5, 16, 512, 48, 64, 20
OFFS = (0.0, 2.0, 8.0, 32.0)

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import ttnn
import tt_bio.tenstorrent as T
from tt_bio import sdpa_generic as SG

dev = T.get_device()
LOG = open(OUT, "a")
s = D ** -0.5


def aiclk():
    vals = []
    for p in Path("/sys/class/tenstorrent").glob("tenstorrent!*"):
        try:
            vals.append(int((p / "tt_aiclk").read_text().split()[0]))
        except Exception:
            pass
    return max(vals) if vals else None


def log(**kw):
    kw.update(M=M, H=H, NT=NT, D=D, t_unix=time.time())
    LOG.write(json.dumps(kw) + "\n"); LOG.flush(); print(json.dumps(kw), flush=True)


def up(t, dt):
    return ttnn.from_torch(t.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=dt)


def host(t):
    return torch.Tensor(ttnn.to_torch(t)).double()


def ref(q, k, v, b):  # float64, b already in logit units
    return torch.softmax(torch.einsum("mhid,mhjd->mhij", q.double(), k.double()) * s + b.double(), -1) @ v.double()


def timed(fn):
    c0 = aiclk(); t0 = time.perf_counter()
    for _ in range(REPS):
        o = fn(); ttnn.deallocate(o)
    ttnn.synchronize_device(dev)
    return round((time.perf_counter() - t0) / REPS * 1e6, 1), [c0, aiclk()]


def lanepad(t):
    z = torch.zeros(*t.shape[:3], DP); z[..., :D] = t
    return z


g = torch.Generator().manual_seed(0)
q_h, k_h, v0 = (torch.randn(M, H, NT, D, generator=g) * 2 for _ in range(3))
bias_h = torch.randn(1, H, NT, NT, generator=g) * 3
lane = torch.zeros(1, 1, 1, DP); lane[..., D] = 1.0
CKC32 = T._SDPA32_CKC
chunk, k_chunk, grid, boxes = T._sdpa32_chunks(dev, M, H, NT)

for C in OFFS:
    v_h = v0 + C
    for dt, tag in ((ttnn.bfloat16, "bf16"), (ttnn.float32, "fp32")):
        rnd = (lambda t: t.bfloat16().float()) if dt == ttnn.bfloat16 else (lambda t: t)
        qr, kr, vr = rnd(lanepad(q_h)), rnd(lanepad(k_h)), rnd(lanepad(v_h))
        Q, K = up(qr, dt), up(kr, dt)
        V, Vn = up(vr, dt), up(vr + lane, dt)
        if dt == ttnn.bfloat16:
            br = (bias_h / s).bfloat16().float()          # SDPA scales its mask with the scores
            o64 = ref(qr[..., :D], kr[..., :D], vr[..., :D], br * s)
            B = up(br, dt)

            def g16(Vx, **kw):
                out = ttnn.allocate_tensor_on_device(ttnn.Shape([M, H, NT, DP]), ttnn.float32, ttnn.TILE_LAYOUT,
                                                     dev, ttnn.DRAM_MEMORY_CONFIG)
                SG.sdpa(dev, Q, K, Vx, B, out, chunk, k_chunk, grid, CKC32, s, boxes=boxes, **kw)
                return out
            arms = {"stock": lambda V=V: T.fused_sdpa(Q, K, V, attn_mask=B, scale=s),
                    "stock_vn": lambda: T._vnorm_divide(T.fused_sdpa(Q, K, Vn, attn_mask=B, scale=s), D),
                    "g16": lambda: g16(V),
                    "g16_o32": lambda: g16(V, out_im_dtype=ttnn.float32),
                    "g16_o32_vn": lambda: T._vnorm_divide(g16(Vn, out_im_dtype=ttnn.float32), D),
                    "g16_st32": lambda: g16(V, stats_dtype=ttnn.float32),
                    "g16_o32_st32": lambda: g16(V, out_im_dtype=ttnn.float32, stats_dtype=ttnn.float32)}
        else:
            br = bias_h / s
            o64 = ref(qr[..., :D], kr[..., :D], vr[..., :D], br * s)
            B = up(br, dt)

            def s32(Vx, **kw):
                out = ttnn.allocate_tensor_on_device(ttnn.Shape([M, H, NT, DP]), ttnn.float32, ttnn.TILE_LAYOUT,
                                                     dev, ttnn.DRAM_MEMORY_CONFIG)
                SG.sdpa(dev, Q, K, Vx, B, out, chunk, k_chunk, grid, CKC32, s, boxes=boxes, **kw)
                return out
            arms = {"s32": lambda: s32(V), "s32_vn": lambda: T._vnorm_divide(s32(Vn), D),
                    "s32_st32": lambda: s32(V, stats_dtype=ttnn.float32),
                    "s32_o32": lambda: s32(V, out_im_dtype=ttnn.float32),
                    "s32_o32_vn": lambda: T._vnorm_divide(s32(Vn, out_im_dtype=ttnn.float32), D),
                    "s32_o32_st32": lambda: s32(V, out_im_dtype=ttnn.float32, stats_dtype=ttnn.float32),
                    "s32_all32": lambda: s32(V, im_dtype=ttnn.float32, out_im_dtype=ttnn.float32,
                                             stats_dtype=ttnn.float32)}
        if os.environ.get("ARMS"):
            keep = set(os.environ["ARMS"].split(","))
            arms = {k: f for k, f in arms.items() if k in keep}
        for name, fn in arms.items():
            try:
                o = fn(); ttnn.synchronize_device(dev); oh = host(o)[..., :NT, :D]; ttnn.deallocate(o)
                us, clk = timed(fn)
            except Exception as e:
                log(op=name, C=C, error=str(e)[:500]); continue
            d = oh - o64
            log(op=name, C=C, dtype=tag, rms=float(d.pow(2).mean().sqrt()), max=float(d.abs().max()),
                rms_rel_to_v=float(d.pow(2).mean().sqrt() / o64.pow(2).mean().sqrt()), us=us, aiclk=clk,
                finite=bool(torch.isfinite(oh).all()))
log(op="end")
