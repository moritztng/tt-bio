"""Signed error of the fp32 DiT attention arms against float64: does `_sdpa32` scale its rows?

usage: TT_VISIBLE_DEVICES=N python sdpa32_bias.py OUT.jsonl [NT] [M] [LOGIT_SCALE]

The g2 grade of dit_sdpa32 moved pLDDT by -0.0001 with a CI excluding 0, so the question is bias, not mean |err|.
Per arm: scale = <o, o64> / <o64, o64> - 1 over all outputs (a row-sum error in the bf16 normaliser shows here),
mean signed error, mean |err|, max |err|, and the spread of the per-row scale. Arms:
  shipped  matmul / scale_add / softmax / matmul, fp32, HiFi4 (what normal mode runs without the lever)
  s32      `_sdpa32` as dit_sdpa32 ships it
  s32one   `_sdpa32` with v padded 48 -> 64 by ones and the output divided by its column 48: the kernel's own
           normaliser cancels, and numerator and denominator see the same bf16 probabilities and rescales
"""
import json, sys, time
from pathlib import Path

import torch

OUT = Path(sys.argv[1]); NT = int(sys.argv[2]) if len(sys.argv) > 2 else 730
M = int(sys.argv[3]) if len(sys.argv) > 3 else 5
LS = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
H, D, REPS = 16, 48, 20

try:
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
except Exception:
    pass
import ttnn
import tt_bio.tenstorrent as T
from tt_bio.eltwise_fusion import scale_add

dev = T.get_device()
ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                             fp32_dest_acc_en=True, packer_l1_acc=True)
NP = T.sdpa32_rows(NT)
LOG = open(OUT, "a")


def aiclk():
    vals = []
    for p in Path("/sys/class/tenstorrent").glob("tenstorrent!*"):
        try:
            vals.append(int((p / "tt_aiclk").read_text().split()[0]))
        except Exception:
            pass
    return max(vals) if vals else None


def log(**kw):
    kw.update(NT=NT, M=M, NP=NP, LS=LS, t_unix=time.time())
    LOG.write(json.dumps(kw) + "\n"); LOG.flush(); print(json.dumps(kw), flush=True)


def up(t):
    return ttnn.from_torch(t.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.float32)


def host(t):
    return torch.Tensor(ttnn.to_torch(t)).double()


def timed(fn):
    c0 = aiclk(); t0 = time.perf_counter()
    for _ in range(REPS):
        o = fn(); ttnn.deallocate(o)
    ttnn.synchronize_device(dev)
    return round((time.perf_counter() - t0) / REPS * 1e6, 1), [c0, aiclk()]


def stats(o):
    o = o[..., :NT, :D]
    d = o - o64
    row = (o * o64).sum(-1) / (o64 * o64).sum(-1).clamp_min(1e-30) - 1
    return dict(scale=float((o * o64).sum() / (o64 * o64).sum() - 1), mean_signed=float(d.mean()),
                mean_abs=float(d.abs().mean()), max_abs=float(d.abs().max()),
                row_scale_mean=float(row.mean()), row_scale_std=float(row.std()))


g = torch.Generator().manual_seed(0)
s = D ** -0.5
q_h, k_h = (torch.randn(M, H, NT, D, generator=g) * LS for _ in range(2))
# v with a non-zero mean per channel, like a DiT value: a row-sum error then moves the output, not just its noise
v_h = torch.randn(M, H, NT, D, generator=g) + torch.randn(1, H, 1, D, generator=g) * 2
bias_h = torch.randn(1, H, NT, NT, generator=g) * 3
o64 = torch.softmax(torch.einsum("mhid,mhjd->mhij", q_h.double(), k_h.double()) * s + bias_h.double(), -1) \
    @ v_h.double()
q, k, v, bias = up(q_h), up(k_h), up(v_h), up(bias_h)


def shipped():
    sc = ttnn.matmul(q, k, transpose_b=True, core_grid=T.CORE_GRID_MAIN, compute_kernel_config=ckc)
    b = scale_add(sc, s, bias); ttnn.deallocate(sc)
    p = ttnn.softmax(b, dim=-1, compute_kernel_config=ckc); ttnn.deallocate(b)
    o = T.batched_matmul(p, v, compute_kernel_config=ckc); ttnn.deallocate(p)
    return o


def padrows(t):
    return ttnn.pad(t, [(0, 0), (0, 0), (0, NP - NT), (0, 0)], value=0.0)


qp, kp, vp = (ttnn.clone(padrows(t)) for t in (q, k, v))
mask = T.sdpa32_mask(ttnn.multiply(bias, 1.0 / s))
TP = -(-D // 32) * 32


def s32():
    return T._sdpa32(qp, kp, vp, mask, s)


def s32one():
    v1 = ttnn.pad(vp, [(0, 0), (0, 0), (0, 0), (0, TP - D)], value=1.0)
    o = T._sdpa32(qp, kp, v1, mask, s)
    ttnn.deallocate(v1)
    r = ttnn.divide(o[:, :, :, :D], o[:, :, :, D:D + 1])
    ttnn.deallocate(o)
    return r


for name, fn in (("shipped", shipped), ("s32", s32), ("s32one", s32one)):
    try:
        a = fn(); ttnn.synchronize_device(dev); ah = host(a); ttnn.deallocate(a)
        b = fn(); ttnn.synchronize_device(dev); bh = host(b); ttnn.deallocate(b)
        us, clk = timed(fn)
        log(op=name, us=us, aiclk=clk, finite=bool(torch.isfinite(ah).all()), equal=bool(torch.equal(ah, bh)),
            **stats(ah))
    except Exception as e:
        log(op=name, error=str(e)[:600])
log(op="end")
