"""Float64 error and device time of the fp32 token DiT's attention with q/k/v written bf16.

    TT_VISIBLE_DEVICES=<chip> TT_BIO_LEVERS=normal python perf/spd_difflin/qkv16_check.py

dit_mm16 keeps q/k/v fp32. The qkv linear writing bf16 is 1175 -> 821 us on Wormhole (op_probe), and the
head split and `_sdpa32` then read half the bytes. `_sdpa32` sizes each operand's CB from its own dtype,
so bf16 q/k/v with the fp32 mask is a legal program; this checks that it is also a right one (the mixed-
format gated multiply was not, k1_check.py --elementwise). Each arm prints its error against float64 on
the fp32 inputs, against float64 on the bf16-rounded inputs (what rounding alone costs), and its time.
"""
import time

import torch

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()
import ttnn  # noqa: E402
import tt_bio.tenstorrent as T  # noqa: E402

B, H, S, D = 5, 16, 768, 48


def ref(q, k, v, m, scale):
    a = torch.softmax((q @ k.transpose(-1, -2) + m) * scale, -1)
    b = torch.softmax(q @ k.transpose(-1, -2) * scale + m, -1)
    return a @ v, b @ v


def err(y, r):
    e = y - r
    return f"rms {e.pow(2).mean().sqrt():.3e} max {e.abs().max():.3e} mean {e.mean():+.3e}"


def timed(fn, n=20):
    dev = T.get_device()
    fn(); ttnn.synchronize_device(dev)
    t = time.perf_counter()
    for _ in range(n):
        y = fn()
        ttnn.deallocate(y) if isinstance(y, ttnn.Tensor) else [ttnn.deallocate(x) for x in y]
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t) / n * 1e6


def main():
    dev = T.get_device()
    print("arch", T.arch_name(), flush=True)
    torch.manual_seed(11)
    scale = D ** -0.5
    q, k, v = (torch.randn(B, H, S, D) for _ in range(3))
    m = torch.randn(1, H, S, S) * 2
    rf = [r.double() for r in ref(q.double(), k.double(), v.double(), m.double(), scale)]
    qb, kb, vb = (x.bfloat16().float() for x in (q, k, v))
    rb = [r.double() for r in ref(qb.double(), kb.double(), vb.double(), m.double(), scale)]
    up = lambda x, dt: ttnn.from_torch(x, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
    tm = up(m, ttnn.float32)
    for lab, dt in (("f32", ttnn.float32), ("bf16", ttnn.bfloat16)):
        tq, tk, tv = up(q, dt), up(k, dt), up(v, dt)
        y = ttnn.to_torch(T._sdpa32(tq, tk, tv, tm, scale)).double()[..., :D]
        sem = min((0, 1), key=lambda i: (y - rf[i]).pow(2).mean())
        print(f"sdpa32 qkv={lab} semantics={'(qk+m)*s' if sem == 0 else 'qk*s+m'} vs f64 {err(y, rf[sem])} | "
              f"vs f64 of bf16-rounded inputs {err(y, rb[sem])} | ref_rms {rf[sem].pow(2).mean().sqrt():.3f}",
              flush=True)
        print(f"sdpa32 qkv={lab} {timed(lambda: T._sdpa32(tq, tk, tv, tm, scale)):.1f} us", flush=True)
    # The head split the two formats feed (nlp_create_qkv_heads on [B, 1, S, 3 * H * D]).
    x = torch.randn(B, 1, S, 3 * H * D)
    for lab, dt in (("f32", ttnn.float32), ("bf16", ttnn.bfloat16)):
        tx = up(x, dt)
        us = timed(lambda: ttnn.experimental.nlp_create_qkv_heads(tx, num_heads=H, num_kv_heads=H,
                                                                  transpose_k_heads=False))
        print(f"create_qkv_heads {lab} {us:.1f} us", flush=True)


if __name__ == "__main__":
    main()
