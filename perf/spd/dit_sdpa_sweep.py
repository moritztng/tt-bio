"""dit_sdpa_sweep.py [--seq 544] [--out OUT.json] : the token DiT's fused SDPA at one sequence length, every chunk config.

BoltzGen's 512 aa page design runs its token transformer at 544 tokens. 544 is 17 tiles and 17 is prime, so no chunk
between one tile and the whole sequence divides it: `_sdpa_program_config_for_lengths` falls back to q = k = 256,
ttnn pads both axes to 768, and the call costs ~0.88 ms on a Wormhole chip against ~0.1 ms at 512. This times the
op at q/k/v [1, 16, S, 64] bf16 with a [1, 16, S, S] bf16 bias for each (q_chunk, k_chunk, pad_to), where pad_to > S
pads q/k/v and masks the new keys exactly as `_sdpa_masked` does (`_sdpa_pad_ragged(to=)`), and grades each against
a float64 torch reference on the unpadded inputs plus torch.equal against the shipped pick.

    TT_VISIBLE_DEVICES=N python perf/spd/dit_sdpa_sweep.py --seq 544 --out sweep.json
"""
import argparse
import json
import time

import torch
import ttnn

from tt_bio import tenstorrent as T

ap = argparse.ArgumentParser()
ap.add_argument("--seq", type=int, default=544)
ap.add_argument("--heads", type=int, default=16)
ap.add_argument("--reps", type=int, default=20)
ap.add_argument("--out", default="dit_sdpa_sweep.json")
ap.add_argument("--arms", default="", help="q:k:pad,... instead of the full sweep (the shipped pick always runs first)")
a = ap.parse_args()

dev = T.get_device()
S, H, D, HD = a.seq, a.heads, 64, 48
g = torch.Generator().manual_seed(0)
q = torch.zeros(1, H, S, D); k = torch.zeros(1, H, S, D); v = torch.zeros(1, H, S, D)
q[..., :HD] = torch.randn(1, H, S, HD, generator=g)
k[..., :HD] = torch.randn(1, H, S, HD, generator=g)
v[..., :HD] = torch.randn(1, H, S, HD, generator=g)
b = 2.0 * torch.randn(1, H, S, S, generator=g)
scale = HD ** -0.5
up = lambda x: ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
qt, kt, vt, bt = up(q), up(k), up(v), up(b)
# reference on the bf16-rounded inputs, in float64
r = [x.to(torch.bfloat16).double() for x in (q, k, v, b)]
ref = torch.softmax(r[0] @ r[1].transpose(-1, -2) * scale + r[3], -1) @ r[2]


def run(qc, kc, pad_to):
    cfg = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=T.COMPUTE_GRID_MAIN, exp_approx_mode=False,
                                 q_chunk_size=qc, k_chunk_size=kc)
    if pad_to > S:
        qp, kp, vp, bp, _ = T._sdpa_pad_ragged(qt, kt, vt, bt, to=pad_to)
    else:
        qp, kp, vp, bp = qt, kt, vt, bt

    def once():
        o = T.fused_sdpa(qp, kp, vp, attn_mask=bp, scale=scale, program_config=cfg)
        return o
    o = once()
    out = ttnn.to_torch(o)[:, :, :S, :].float()
    ttnn.deallocate(o)
    for _ in range(3):
        ttnn.deallocate(once())
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(a.reps):
        ttnn.deallocate(once())
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) / a.reps * 1e3, out


ship = T._sdpa_program_config_for_lengths(S, S, H, site="token_dit", d=D)
arms = [(ship.q_chunk_size, ship.k_chunk_size, S)]
if a.arms:
    arms += [tuple(map(int, x.split(":"))) for x in a.arms.split(",")]
pads = [] if a.arms else sorted({S, -(-S // 64) * 64, -(-S // 128) * 128, -(-S // 256) * 256})
for p in pads:
    for qc in (32, 64, 128, 256):
        for kc in (32, 64, 128, 256, 512):
            if (qc, kc, p) not in arms and p % qc == 0 and p % kc == 0:
                arms.append((qc, kc, p))
for qc in (() if a.arms else (32, 64, 128, 256)):
    for kc in (64, 128, 256):
        if (qc, kc, S) not in arms:
            arms.append((qc, kc, S))     # non-dividing: ttnn pads internally
res, base = [], None
for qc, kc, p in arms:
    try:
        ms, out = run(qc, kc, p)
    except Exception as e:
        res.append(dict(q=qc, k=kc, pad=p, err=str(e).splitlines()[0][:200])); print(res[-1], flush=True)
        continue
    if base is None:
        base = out
    err = (out.double()[..., :HD] - ref[..., :HD]).abs()
    rel = (err.norm() / ref[..., :HD].norm()).item()
    res.append(dict(q=qc, k=kc, pad=p, ms=round(ms, 4), finite=bool(torch.isfinite(out).all()),
                    rel_l2_f64=rel, max_abs_f64=err.max().item(), equal_ship=bool(torch.equal(out, base))))
    print(res[-1], flush=True)
json.dump(dict(seq=S, heads=H, ship=[ship.q_chunk_size, ship.k_chunk_size], arms=res,
               aiclk_note="sampled by the harness"), open(a.out, "w"), indent=1)
