"""Boltz-2 token-DiT attention at 512 x 1 sample: the shipped head split / SDPA / head re-assembly chain against the
fused-qkv SDPA program OpenFold3's dit_fuse runs (`tenstorrent._sdpa32_fused`), each against a float64 reference.

usage: TT_VISIBLE_DEVICES=N python ditattn.py OUT.jsonl [NT]

Arms, all from the same padded [1, NT, 3 * 16 * 64] bf16 qkv projection and [1, 16, NT, NT] bias:
  ship     unsqueeze, nlp_create_qkv_heads, fused_sdpa at the token_dit program config, slice 48 lanes, permute,
           reshape, permute (tenstorrent.py AttentionPairBias, the non-concat branch)
  cat      the same SDPA, nlp_concat_heads (apb_cat): [1, NT, 1024] on padded lanes
  qkvf4    _sdpa32_fused at its own HiFi4 fp32-dest config, bf16 out, bf16 mask
  qkvf2/3  the same program at HiFi2 / HiFi3
Per arm: device-synced us per call over REPS (eager, and as a replayed trace = device time), rel_rms and max_abs vs float64 on the 48 real lanes, AICLK.
"""
import json, sys, time
from pathlib import Path

import torch

OUT = Path(sys.argv[1]); NT = int(sys.argv[2]) if len(sys.argv) > 2 else 512
H, D, DP, REPS = 16, 48, 64, 50

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import ttnn
import tt_bio.tenstorrent as T
from tt_bio import sdpa_generic as SG

dev = T.get_device(trace="diffusion")
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
    kw.update(NT=NT, arch=str(dev.arch()), t_unix=time.time())
    LOG.write(json.dumps(kw) + "\n"); LOG.flush(); print(json.dumps(kw), flush=True)


g = torch.Generator().manual_seed(0)
s = D ** -0.5
q_h, k_h, v_h = (torch.randn(1, H, NT, D, generator=g) * 1.5 for _ in range(3))
bias_h = torch.randn(1, H, NT, NT, generator=g) * 4
o64 = torch.softmax((torch.einsum("bhid,bhjd->bhij", q_h.double(), k_h.double()) + bias_h.double()) * s, -1) \
    @ v_h.double()                                         # [1, H, NT, D]


def lanes(t):                                              # [1, H, NT, D] -> [1, NT, H * DP], zero pad lanes
    return torch.nn.functional.pad(t, (0, DP - D)).permute(0, 2, 1, 3).reshape(1, NT, H * DP)


qkv_t = ttnn.from_torch(torch.cat([lanes(q_h), lanes(k_h), lanes(v_h)], -1), layout=ttnn.TILE_LAYOUT,
                        device=dev, dtype=ttnn.bfloat16)
bias_t = ttnn.from_torch(bias_h, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)


def ship():
    qkv = ttnn.unsqueeze(qkv_t, 1)
    q, k, v = ttnn.experimental.nlp_create_qkv_heads(qkv, num_heads=H, num_kv_heads=H, transpose_k_heads=False)
    o = T.fused_sdpa(q, k, v, attn_mask=bias_t, scale=s,
                     program_config=T._sdpa_program_config_for_lengths(NT, NT, H, site="token_dit", d=DP))
    for t in (q, k, v):
        ttnn.deallocate(t)
    return o


def ship_full():
    o = ship()
    o = o[:, :, :, :D]
    o = ttnn.permute(o, (0, 1, 3, 2))
    o = ttnn.reshape(o, (o.shape[0], -1, o.shape[3]))
    return ttnn.permute(o, (0, 2, 1))                      # [1, NT, H * D]


def cat():
    o = ship()
    o = ttnn.experimental.nlp_concat_heads(o)
    return ttnn.reshape(o, (1, NT, H * DP))


def qkvf(fid):
    ckc = (fid,) + T._SDPA32_CKC[1:]

    def run():
        qkv = ttnn.unsqueeze(qkv_t, 1)
        chunk, k_chunk, grid = T._sdpa32_chunks(dev, 1, H, NT)
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, NT, H * DP]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev,
                                             ttnn.DRAM_MEMORY_CONFIG)
        SG.sdpa(dev, qkv, qkv, qkv, bias_t, out, chunk, k_chunk, grid, ckc, s, qkv_fused=H)
        return ttnn.reshape(out, (1, NT, H * DP))
    return run


def err(o, padded):
    h = torch.Tensor(ttnn.to_torch(o)).double().reshape(1, NT, -1)
    if padded:
        h = h.reshape(1, NT, H, DP)[..., :D]
    h = h.reshape(1, NT, H, D).permute(0, 2, 1, 3)
    d = h - o64
    return float(d.pow(2).mean().sqrt() / o64.pow(2).mean().sqrt()), float(d.abs().max())


ARMS = [("ship", ship_full, False), ("cat", cat, True), ("qkvf4", qkvf(ttnn.MathFidelity.HiFi4), True),
        ("qkvf3", qkvf(ttnn.MathFidelity.HiFi3), True), ("qkvf2", qkvf(ttnn.MathFidelity.HiFi2), True)]
for rnd in range(2):
    for name, fn, padded in ARMS:
        try:
            o = fn(); ttnn.synchronize_device(dev)
            rr, ma = err(o, padded); ttnn.deallocate(o)
            o2 = fn(); h2 = torch.Tensor(ttnn.to_torch(o2)); ttnn.deallocate(o2)
            c0 = aiclk(); t0 = time.perf_counter()
            for _ in range(REPS):
                o = fn(); ttnn.deallocate(o)
            ttnn.synchronize_device(dev)
            us = round((time.perf_counter() - t0) / REPS * 1e6, 1)
            tid, o = T.capture_trace(dev, fn)              # device time alone: the fold's diffusion is device-bound
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
            t0 = time.perf_counter()
            for _ in range(REPS):
                ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(dev)
            dus = round((time.perf_counter() - t0) / REPS * 1e6, 1)
            ttnn.release_trace(dev, tid); ttnn.deallocate(o)
            log(round=rnd, arm=name, us=us, trace_us=dus, rel_rms=rr, max_abs=ma, aiclk=[c0, aiclk()],
                bitexact_rerun=bool(torch.equal(h2, torch.Tensor(ttnn.to_torch(fn())))))
        except Exception as e:  # noqa: BLE001
            log(round=rnd, arm=name, error=repr(e)[:400])
