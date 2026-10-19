"""The token DiT's fused SDPA at a ragged length, padded once per rollout to a chunk-friendly length.

At JapanFold's production shape (cs5, 9LLG) the DiT attends over 1199 tokens: q, k, v [1, 16, 1216, 64] (head dim 48
padded) against a [1, 16, 1216, 1199] bias, q = k = 256 (no 32-aligned divisor of 1216 in dit_chunk's [96, 320]), so
the op pads to its chunk grid inside, 1.0 ms a call x 4800. The bias' ragged key tail is also re-filled every call
(ttnn.pad in `_sdpa_pad_ragged`, 165 us). Here the bias is padded ONCE to Np with the new keys masked and q, k, v
padded per call to Np, the way `_stage_token_bias` + `_sdpa_to_bias_width` already serve Boltz-2.
Per arm: us/call of the SDPA alone, plus the per-call q/k/v pad cost per Np, and rel_rms / max abs of the logical
[:L] rows against a float64 softmax attention of the same bf16 inputs.
  TT_VISIBLE_DEVICES=N python perf/spd_m_opendde/ditpad_sweep.py --L 1199 --b 1 5 --out OUT.jsonl
"""
import argparse
import json
import os
import statistics as st
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--L", type=int, default=1199)
ap.add_argument("--b", type=int, nargs="+", default=[1, 5])
ap.add_argument("--np", type=int, nargs="+", default=[1216, 1248, 1280, 1344, 1536])
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
os.environ.setdefault("TT_BIO_LEVERS", "normal")

from tt_bio.main import ensure_p300_mesh_descriptor  # noqa: E402
ensure_p300_mesh_descriptor()
import torch  # noqa: E402
import ttnn  # noqa: E402
import tt_bio.tenstorrent as T  # noqa: E402

dev = T.get_device()
OPENED = sorted({int(os.readlink(f"/proc/self/fd/{fd}").rsplit("/", 1)[1]) for fd in os.listdir("/proc/self/fd")
                 if os.path.exists(f"/proc/self/fd/{fd}") and
                 os.readlink(f"/proc/self/fd/{fd}").startswith("/dev/tenstorrent/")})


def aiclk():
    out = []
    for n in OPENED:
        try:
            out.append(int(Path(f"/sys/class/tenstorrent/tenstorrent!{n}/tt_aiclk").read_text().split()[0]))
        except Exception:  # noqa: BLE001
            pass
    return out


def timed(fn, free=True):
    for _ in range(2):
        o = fn()
        if free:
            ttnn.deallocate(o)
    ttnn.synchronize_device(dev)
    ts, clk = [], []
    for _ in range(5):
        t = time.perf_counter()
        for _ in range(5):
            o = fn()
            if free:
                ttnn.deallocate(o)
        ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t) / 5 * 1e6)
        clk += aiclk()
    return dict(us_med=round(st.median(ts), 1), us_min=round(min(ts), 1),
                aiclk_med=sorted(clk)[len(clk) // 2] if clk else None, aiclk_min=min(clk) if clk else None)


def emit(rec):
    print(json.dumps(rec), flush=True)
    with a.out.open("a") as f:
        f.write(json.dumps(rec) + "\n")


H, D, DL, L = 16, 64, 48, a.L
up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)  # noqa: E731
scale = DL ** -0.5
for B in a.b:
    g = torch.Generator().manual_seed(B)
    qkv = [torch.zeros(B, H, L, D) for _ in range(3)]
    for t in qkv:
        t[..., :DL] = torch.randn(B, H, L, DL, generator=g)
    qkv = [t.bfloat16() for t in qkv]
    bias = (2 * torch.randn(1, H, L, L, generator=g)).bfloat16()
    q64, k64, v64 = (t.double() for t in qkv)
    ref = torch.softmax(scale * (q64 @ k64.transpose(-1, -2) + bias.double()), dim=-1) @ v64
    del q64, k64, v64
    lq = [up(t) for t in qkv]   # the logical [B, H, L, D] the fold's projection hands over
    # what the fold pays per call today: the bias' ragged key tail re-filled (ttnn.pad aliases at a tile multiple)
    bl = up(bias)
    emit(dict(B=B, arm="bias_tail_pad", **timed(lambda: ttnn.pad(bl, [(0, 0), (0, 0), (0, 0), (0, (-L) % 32)],
                                                                 value=T._SDPA_PAD_MASK), free=False)))
    ttnn.deallocate(bl)
    for Np in a.np:
        bp = torch.zeros(1, H, Np, Np)
        bp[..., :L, :L] = bias.float()
        bp[..., :, L:] = T._SDPA_PAD_MASK
        db = up(bp)
        pads = [(0, 0), (0, 0), (0, Np - L), (0, 0)]
        dq, dk, dv = (ttnn.pad(t, pads, value=0.0) for t in lq)
        row = timed(lambda: ttnn.pad(lq[0], pads, value=0.0), free=Np - L >= 32)
        emit(dict(B=B, Np=Np, arm="qkv_pad_one", **row))
        divs = [c for c in range(64, 417, 32) if Np % c == 0]
        arms = [(qc, kc) for qc in divs for kc in divs if kc >= 64]
        if Np == 1216:
            arms = [(256, 256)] + arms
        for qc, kc in arms + ["pm"]:
            if qc == "pm":
                fn = lambda: T.token_dit_sdpa_pm(dq, dk, dv, db, scale)  # noqa: E731
                T._DIT_PM_PICK.clear()
                name = "pm"
            else:
                cfg = T._sdpa_program_config(q_chunk_size=qc, k_chunk_size=kc)
                fn = lambda cfg=cfg: T.fused_sdpa(dq, dk, dv, attn_mask=db, scale=scale, program_config=cfg)  # noqa: E731
                name = f"q{qc},k{kc}"
            rec = dict(B=B, Np=Np, arm=name)
            try:
                o = fn()
                if o is None:
                    rec["err"] = "declined"
                else:
                    got = ttnn.to_torch(o).double()[:, :, :L, :]
                    ttnn.deallocate(o)
                    rec.update(rel_rms=float((got - ref).norm() / ref.norm()), max_abs=float((got - ref).abs().max()))
                    rec.update(timed(fn))
                    if qc == "pm":
                        rec["pick"] = T._DIT_PM_PICK.get((B, Np, H, D, dq.dtype))
            except Exception as e:  # noqa: BLE001
                rec["err"] = str(e).splitlines()[0][:200]
            emit(rec)
        # a pad inside the last tile aliases its source (`_sdpa_pad_ragged`): freeing it would free lq
        for t in (dq, dk, dv) if Np - L >= 32 else ():
            ttnn.deallocate(t)
        ttnn.deallocate(db)
    for t in lq:
        ttnn.deallocate(t)
