"""Float64 error and device time of the atom transformer's superset attention with q/k/v written bf16.

    TT_VISIBLE_DEVICES=<chip> TT_BIO_LEVERS=normal python perf/spd_difflin/atom16_check.py

`atom_mm16` writes the fp32 atom transformer's q/k/v bf16, so `_attention_superset` hands the windowed
`sdpa_generic` kernel bf16 frames with the fp32 superset bias. This is the same mixed-format program
qkv16_check.py measured for the token DiT, read through the kv_window path: each arm prints its error
against float64 on the fp32 inputs, against float64 on the bf16-rounded inputs (what rounding alone
costs), and its time, at k chunk 32 (Wormhole's) and at the whole window.
"""
import time

import torch

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()
import ttnn  # noqa: E402
import tt_bio.tenstorrent as T  # noqa: E402
from tt_bio import sdpa_generic as SG  # noqa: E402
from tt_bio.protenix import _ATOM_SDPA32_CKC  # noqa: E402

M, H, DH, NQ, W, LEAD = 5, 4, 32, 32, 160, 64   # AtomTransformer._superset(): whole tiles around the 128 keys
NB = 185                                    # 5919 atoms, the c730 cell
F = (NB + W // NQ - 1) * NQ                 # frame rows


def ref(q, k, v, m, scale):
    """Both bias semantics: (qk + m) * s and qk * s + m. q/k/v (M, H, F, dh), m (1, H*nb, nq, W)."""
    qw = q[:, :, LEAD:LEAD + NB * NQ].reshape(M, H, NB, NQ, DH)
    kw = k.unfold(2, W, NQ)[:, :, :NB].transpose(-1, -2)          # (M, H, nb, W, dh)
    vw = v.unfold(2, W, NQ)[:, :, :NB].transpose(-1, -2)
    qk = qw @ kw.transpose(-1, -2)                                 # (M, H, nb, nq, W)
    mm = m.reshape(1, H, NB, NQ, W)
    outs = []
    for sc in ((qk + mm) * scale, qk * scale + mm):
        outs.append((torch.softmax(sc, -1) @ vw).reshape(M, H * NB, NQ, DH))
    return outs


def err(y, r):
    e = y - r
    return f"rms {e.pow(2).mean().sqrt():.3e} max {e.abs().max():.3e} mean {e.mean():+.3e}"


def main():
    dev = T.get_device()
    print("arch", T.arch_name(), flush=True)
    torch.manual_seed(11)
    scale = DH ** -0.5
    q, k, v = (torch.randn(M, H, F, DH) for _ in range(3))
    m = torch.randn(1, H * NB, NQ, W) * 2
    rf = ref(q.double(), k.double(), v.double(), m.double(), scale)
    qb, kb, vb = (x.bfloat16().double() for x in (q, k, v))
    rb = ref(qb, kb, vb, m.double(), scale)
    up = lambda x, dt: ttnn.from_torch(x, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
    tm = up(m, ttnn.float32)
    g = dev.compute_with_storage_grid_size()
    for kc in (32, W):
        for lab, dt in (("f32", ttnn.float32), ("bf16", ttnn.bfloat16)):
            tq, tk, tv = up(q, dt), up(k, dt), up(v, dt)

            def run():
                o = ttnn.allocate_tensor_on_device(ttnn.Shape([M, H * NB, NQ, DH]), dt, ttnn.TILE_LAYOUT, dev,
                                                   ttnn.DRAM_MEMORY_CONFIG)
                SG.sdpa(dev, tq, tk, tv, tm, o, NQ, kc, (g.x, g.y), _ATOM_SDPA32_CKC, scale,
                        kv_window=(NB, W, LEAD, NQ))
                return o

            y = ttnn.to_torch(run()).double()
            sem = min((0, 1), key=lambda i: (y - rf[i]).pow(2).mean())
            print(f"atom sdpa kc={kc} qkv={lab} semantics={'(qk+m)*s' if sem == 0 else 'qk*s+m'} "
                  f"vs f64 {err(y, rf[sem])} | vs f64 of bf16-rounded inputs {err(y, rb[sem])} | "
                  f"ref_rms {rf[sem].pow(2).mean().sqrt():.3f}", flush=True)
            run(); ttnn.synchronize_device(dev)
            t = time.perf_counter()
            for _ in range(20):
                ttnn.deallocate(run())
            ttnn.synchronize_device(dev)
            print(f"atom sdpa kc={kc} qkv={lab} {(time.perf_counter() - t) / 20 * 1e6:.1f} us", flush=True)


if __name__ == "__main__":
    main()
