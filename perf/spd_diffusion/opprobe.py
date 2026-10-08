"""spd-diffusion op probe: the fp32 token-DiT attention chain at the c730 shape, shipped form vs candidates.

usage: TT_VISIBLE_DEVICES=N python opprobe.py OUT.jsonl [NT] [M]

Shipped (protenix fp32 raw-matmul path, tenstorrent.py AttentionPairBias):
    kt = permute(k); sc = batched_matmul(q, kt); sc = scale_add(sc, s, bias); p = softmax(sc); o = batched_matmul(p, v)
Candidates, each checked for torch.equal / max |d| against the shipped intermediate it replaces:
    qk_tb      ttnn.matmul(q, k, transpose_b=True)  (no permute of k)
    sms        ttnn.scale_mask_softmax(sc, s, mask=bias)  (scale, bias add and softmax in one pass)
    sms_causal the same with is_causal_mask=True (the variant that reads a full H x W mask)
Timings are device-synced wall times over REPS calls after one warm-up call, AICLK sampled around them.
"""
import json, os, sys, time
from pathlib import Path

import torch

OUT = Path(sys.argv[1]); NT = int(sys.argv[2]) if len(sys.argv) > 2 else 730
M = int(sys.argv[3]) if len(sys.argv) > 3 else 5
H, D, DP, REPS = 16, 48, 64, 20

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
    kw.update(NT=NT, M=M, t_unix=time.time())
    LOG.write(json.dumps(kw) + "\n"); LOG.flush(); print(json.dumps(kw), flush=True)


def up(t):
    return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.float32)


def host(t):
    return torch.Tensor(ttnn.to_torch(t)).float()


def timed(name, fn, keep=None):
    try:
        out = fn(); ttnn.synchronize_device(dev)
    except Exception as e:
        log(op=name, error=str(e)[:600]); return None
    ttnn.deallocate(out)
    c0 = aiclk(); t0 = time.perf_counter()
    for _ in range(REPS):
        o = fn(); ttnn.deallocate(o)
    ttnn.synchronize_device(dev); us = (time.perf_counter() - t0) / REPS * 1e6
    log(op=name, us=round(us, 1), aiclk=[c0, aiclk()])
    return fn() if keep else None


g = torch.Generator().manual_seed(0)
s = D ** -0.5
q_h = torch.zeros(M, H, NT, DP); k_h = torch.zeros(M, H, NT, DP); v_h = torch.zeros(M, H, NT, DP)
for t in (q_h, k_h, v_h):
    t[..., :D] = torch.randn(M, H, NT, D, generator=g) * 2
bias_h = torch.randn(1, H, NT, NT, generator=g) * 3
q, k, v, bias = up(q_h), up(k_h), up(v_h), up(bias_h)

# shipped chain, piece by piece
kt = ttnn.permute(k, (0, 1, 3, 2))
sc = T.batched_matmul(q, kt, compute_kernel_config=ckc)
timed("permute_k", lambda: ttnn.permute(k, (0, 1, 3, 2)))
timed("qk_shipped", lambda: T.batched_matmul(q, kt, compute_kernel_config=ckc))
biased = scale_add(sc, s, bias)
timed("scale_add", lambda: scale_add(sc, s, bias))
p_ref = ttnn.softmax(biased, dim=-1)
timed("softmax", lambda: ttnn.softmax(biased, dim=-1))
timed("av_shipped", lambda: T.batched_matmul(p_ref, v, compute_kernel_config=ckc))
sc_h, p_h = host(sc), host(p_ref)

# float64 reference for the softmax, to say which form is closer when they differ
ref64 = torch.softmax(sc_h.double() * s + bias_h.double(), -1)
log(op="shipped_vs_f64", max_abs=float((p_h.double() - ref64).abs().max()))

# candidate: k transposed inside the matmul
for name, fn in [("qk_tb_ttnn", lambda: ttnn.matmul(q, k, transpose_b=True, compute_kernel_config=ckc,
                                                     core_grid=T.CORE_GRID_MAIN))]:
    o = timed(name, fn, keep=True)
    if o is not None:
        oh = host(o); log(op=name + "_check", equal=bool(torch.equal(oh, sc_h)),
                          max_abs=float((oh - sc_h).abs().max())); ttnn.deallocate(o)

# candidate: scale + bias + softmax in one op
for name, kw in [("sms", {}), ("sms_causal", {"is_causal_mask": True})]:
    o = timed(name, lambda kw=kw: ttnn.scale_mask_softmax(sc, s, mask=bias, **kw), keep=True)
    if o is not None:
        oh = host(o)
        log(op=name + "_check", equal=bool(torch.equal(oh, p_h)), max_abs=float((oh - p_h).abs().max()),
            max_abs_f64=float((oh.double() - ref64).abs().max())); ttnn.deallocate(o)
# candidate: q@k^T per chunk of samples with q padded to 24 M-tiles. 23 tiles is prime, so the
# batched reuse factory has no legal per_core_M (1 trips its multi-block stride bug, 23 needs a
# 2.1 MB fp32 output CB) and ttnn's default re-reads in0 and in1 for every output tile.
cores = T.COMPUTE_GRID_MAIN[0] * T.COMPUTE_GRID_MAIN[1]
MP = 768
qp_h = torch.zeros(M, H, MP, DP); qp_h[:, :, :NT] = q_h
qp = up(qp_h)


def qk_chunks(b, p, src):
    sub_h = max(h for h in range(1, 5) if p % h == 0)
    cfg = ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=T.COMPUTE_GRID_MAIN, in0_block_w=DP // 32,
        out_subblock_h=sub_h, out_subblock_w=1, per_core_M=p, per_core_N=(NT + 31) // 32)
    outs = []
    for i in range(0, M, b):
        j = min(i + b, M)
        qi = ttnn.slice(src, [i, 0, 0, 0], [j, H, MP, DP]) if (i, j) != (0, M) else src
        ki = ttnn.slice(kt, [i, 0, 0, 0], [j, H, DP, NT]) if (i, j) != (0, M) else kt
        outs.append(ttnn.matmul(qi, ki, program_config=cfg, compute_kernel_config=ckc))
    return outs


class _L(list):
    def deallocate(self):
        for t in self:
            ttnn.deallocate(t)


_dealloc = ttnn.deallocate
ttnn.deallocate = lambda t, *a, **k: t.deallocate() if isinstance(t, _L) else _dealloc(t, *a, **k)
for b in (1, 2, 5):
    for p in (3, 4, 6, 8, 12, 24):
        if b * H * (MP // 32) // p > cores:
            continue
        name = f"qk_pad24_b{b}_p{p}"
        o = timed(name, lambda b=b, p=p: _L(qk_chunks(b, p, qp)), keep=True)
        if o is not None:
            oh = torch.cat([host(t)[:, :, :NT] for t in o]); log(op=name + "_check", equal=bool(torch.equal(oh, sc_h)),
                                                                 max_abs=float((oh - sc_h).abs().max()))
            o.deallocate()
timed("pad_q", lambda: ttnn.pad(q, [(0, 0), (0, 0), (0, MP - NT), (0, 0)], 0.0))
try:
    o = ttnn.transformer.scaled_dot_product_attention(q, k, v, attn_mask=bias, is_causal=False, scale=s)
    log(op="sdpa_fp32", ok=True)
except Exception as e:
    log(op="sdpa_fp32", error=str(e)[:300])
log(op="end")
os._exit(0)
