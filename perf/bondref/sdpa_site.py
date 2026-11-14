"""OpenDDE's token-DiT attention call, sdpa_acc32 on vs off, against float64 (spd-q-bondradial).

sdpa_offset.py showed fp32 output accumulators + row statistics cut the generic SDPA's |v|-proportional error 3x,
yet at model level OpenDDE got worse with them (lddt_ca -0.0006, bond error +25 % in every octave). That probe ran
plain `_sdpa32` at 512 tokens (2 key chunks), fp32 output, HiFi4, no padding. This one runs the call the bf16 DiT
makes under dit_qkvf: `_sdpa32_fused` on a fused bf16 qkv projection, a bf16 pair bias with padded keys masked,
bf16 output, `_DIT_QKVF_CKC` (HiFi2), dit_mask_chain + dit_qbox, at OpenDDE's 9DBP length. Arms flip one factor
at a time, so a wrong path names itself.

usage: TT_VISIBLE_DEVICES=N python sdpa_site.py OUT.jsonl
Per (NT, NV, arm, v offset C): rms / max |o - o64| over valid rows and real lanes, bf16 floor of o64, us per call.
"""
import json, sys, time
from pathlib import Path

import torch

OUT = Path(sys.argv[1])
M, H, D, DP, REPS = 5, 16, 48, 64, 10
SHAPES = ((512, 512), (1024, 791), (896, 791))   # (token axis, valid tokens)
OFFS = (0.0, 8.0, 32.0)

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import ttnn
import tt_bio.tenstorrent as T

dev = T.get_device()
LOG = open(OUT, "a")
s = D ** -0.5
BASE = frozenset({"dit_mask_chain", "dit_qbox"})
# arm -> (extra levers, out dtype, ckc)
ARMS = {"def": (set(), ttnn.bfloat16, T._DIT_QKVF_CKC),
        "acc": ({"sdpa_acc32"}, ttnn.bfloat16, T._DIT_QKVF_CKC),
        "def_o32": (set(), ttnn.float32, T._DIT_QKVF_CKC),
        "acc_o32": ({"sdpa_acc32"}, ttnn.float32, T._DIT_QKVF_CKC),
        "acc_h4": ({"sdpa_acc32"}, ttnn.bfloat16, T._SDPA32_CKC),
        "def_h4": (set(), ttnn.bfloat16, T._SDPA32_CKC),
        "def_h3": (set(), ttnn.bfloat16, (ttnn.MathFidelity.HiFi3, False, True, False)),
        "acc_h3": ({"sdpa_acc32"}, ttnn.bfloat16, (ttnn.MathFidelity.HiFi3, False, True, False)),
        "acc_nmc": ({"sdpa_acc32", "-dit_mask_chain"}, ttnn.bfloat16, T._DIT_QKVF_CKC),
        "acc_nqb": ({"sdpa_acc32", "-dit_qbox"}, ttnn.bfloat16, T._DIT_QKVF_CKC)}


def aiclk():
    vals = []
    for p in Path("/sys/class/tenstorrent").glob("tenstorrent!*"):
        try:
            vals.append(int((p / "tt_aiclk").read_text().split()[0]))
        except Exception:
            pass
    return max(vals) if vals else None


def log(**kw):
    kw.update(M=M, H=H, D=D, t_unix=time.time())
    LOG.write(json.dumps(kw) + "\n"); LOG.flush(); print(json.dumps(kw), flush=True)


def up(t, dt):
    return ttnn.from_torch(t.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=dt)


def set_levers(extra):
    on = {x for x in extra if not x.startswith("-")}
    off = {x[1:] for x in extra if x.startswith("-")}
    T._LEVERS = frozenset((BASE | on) - off)


g = torch.Generator().manual_seed(0)
for NT, NV in SHAPES:
    q_h, k_h, v0 = (torch.randn(M, H, NT, D, generator=g) * 2 for _ in range(3))
    bias = torch.randn(1, H, NT, NT, generator=g) * 3 / s          # mask in score units: SDPA scales it
    bias[:, :, NV:, :] = 0.0                                          # as sdpa32_mask: new rows 0, new keys masked
    bias[..., NV:] = T._SDPA32_PAD_MASK
    br = bias.bfloat16().float()
    Z = up(br, ttnn.bfloat16)
    for C in OFFS:
        qr, kr, vr = (t.bfloat16().float() for t in (q_h, k_h, v0 + C))
        o64 = torch.softmax(torch.einsum("mhid,mhjd->mhij", qr.double(), kr.double()) * s + br.double() * s, -1) @ vr.double()
        o64 = o64[:, :, :NV]
        floor = float((o64.float().bfloat16().double() - o64).pow(2).mean().sqrt())
        pad = lambda t: torch.nn.functional.pad(t, (0, DP - D))            # noqa: E731
        qkv = torch.cat([pad(t).permute(0, 2, 1, 3).reshape(M, NT, H * DP) for t in (qr, kr, vr)], -1)
        QKV = up(qkv.unsqueeze(1), ttnn.bfloat16)
        for name, (extra, odt, ckc) in ARMS.items():
            set_levers(extra)
            fn = lambda: T._sdpa32_fused(QKV, Z, s, H, out_dtype=odt, ckc=ckc)   # noqa: E731
            try:
                o = fn(); ttnn.synchronize_device(dev)
                oh = torch.Tensor(ttnn.to_torch(o)).double().reshape(M, NT, H, DP).permute(0, 2, 1, 3)[:, :, :NV, :D]
                ttnn.deallocate(o)
                c0 = aiclk(); t0 = time.perf_counter()
                for _ in range(REPS):
                    ttnn.deallocate(fn())
                ttnn.synchronize_device(dev)
                us = round((time.perf_counter() - t0) / REPS * 1e6, 1)
            except Exception as e:
                log(NT=NT, NV=NV, arm=name, C=C, error=str(e)[:500]); continue
            d = oh - o64
            log(NT=NT, NV=NV, arm=name, C=C, rms=float(d.pow(2).mean().sqrt()), max=float(d.abs().max()),
                floor=floor, finite=bool(torch.isfinite(oh).all()), us=us, aiclk=[c0, aiclk()])
        ttnn.deallocate(QKV)
    ttnn.deallocate(Z)
log(arm="end")
