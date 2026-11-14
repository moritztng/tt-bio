#!/usr/bin/env python3
"""gap_pfdpb: the token encoder's pairformer pair bias, shipped ttnn chain against dit_pair_bias (RFD3_PF_DPB).
  shipped  _tuned_linear(z, W [128, 16]) -> permute (0, 3, 1, 2) -> typecast fp32 -> pad keys to n at -1e4
  pfdpb    PairformerAttention._pair_bias_dpb: W^T zero-padded to 32 heads, scale 1, the key pad from the mask
Both [1, 16, I, n] fp32, against a float64 reference of the same bf16 inputs; the key pad must score <= -9984 (a
bf16 -1e4) so its softmax weight is exactly 0. Sizes: the 685-token page and 418 tokens (partial last tile).
    python scripts/rfd3_port/gap_pfdpb.py OUT.json [reps]
"""
import json, os, pathlib, statistics, sys, time, types

import torch
import ttnn

sys.path.insert(0, os.getcwd())
from tt_bio.rfd3 import model as M                                       # noqa: E402
from tt_bio.rfd3.tiles import pad_axis                                   # noqa: E402
from tt_bio.tenstorrent import get_device                                # noqa: E402

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "out/gap_pfdpb.json")
N = int(sys.argv[2]) if len(sys.argv) > 2 else 5
C, H = 128, 16


def timeit(fn, dev):
    o = fn()
    ttnn.synchronize_device(dev)
    ts = []
    for _ in range(N):
        ttnn.deallocate(o)
        t0 = time.perf_counter()
        o = fn()
        ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t0) * 1e3)
    return o, statistics.median(ts), min(ts), max(ts)


def case(dev, ckc, I):
    n = (I + 31) // 32 * 32
    g = torch.Generator().manual_seed(I)
    z_t = torch.randn(1, I, I, C, generator=g).bfloat16()
    w_t = (torch.randn(H, C, generator=g) / C ** 0.5).bfloat16()       # to_b.weight is [heads, c_z]
    up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)  # noqa: E731
    z, w = up(z_t), up(w_t.t().contiguous())
    ref = torch.einsum("bijc,hc->bhij", z_t.double(), w_t.double())
    attn = types.SimpleNamespace(weights={"to_b.weight": w_t}, n_head=H, device=dev, dtype=ttnn.bfloat16,
                                 compute_kernel_config=ckc)

    def shipped():
        b = M._tuned_linear(z, w, ckc=ckc, dtype=ttnn.bfloat16, core_grid=M.CORE_GRID_MAIN)
        b = ttnn.permute(b, (0, 3, 1, 2))
        bf = ttnn.typecast(b, ttnn.float32, memory_config=b.memory_config())
        return pad_axis(bf, n, 3, -1e4)

    def pfdpb():
        return M.PairformerAttention._pair_bias_dpb(attn, z, I, n)

    rec, outs = {"I": I, "n": n}, {}
    for name, fn in (("shipped", shipped), ("pfdpb", pfdpb)):
        try:
            o, med, lo, hi = timeit(fn, dev)
            r = {"shape": [int(x) for x in o.shape], "dtype": str(o.dtype), "ms_median": med, "ms_min": lo, "ms_max": hi}
            d = ttnn.to_torch(o).double()
            ttnn.deallocate(o)
            assert tuple(d.shape) == (1, H, I, n), tuple(d.shape)
            e = d[..., :I] - ref
            r.update(rel_err_f64=float(e.norm() / ref.norm()), max_abs_f64=float(e.abs().max()),
                     key_pad_max=float(d[..., I:].max()) if n > I else None)
            outs[name] = d
        except Exception as ex:                                          # noqa: BLE001
            r = {"error": repr(ex)[:400]}
        rec[name] = r
        print(I, name, r, flush=True)
    if len(outs) == 2:
        a, b = outs["shipped"][..., :I], outs["pfdpb"][..., :I]
        rec["pfdpb_vs_shipped"] = {"bit_equal_frac": float((a == b).double().mean()), "max_abs": float((a - b).abs().max())}
        print(I, "pfdpb_vs_shipped", rec["pfdpb_vs_shipped"], flush=True)
    return rec


def main():
    dev = get_device()
    ckc = M._default_compute_kernel_config()
    rec = {"reps": N, "card": os.environ.get("TT_VISIBLE_DEVICES"), "host": os.uname().nodename,
           "arch": str(dev.arch()), "cases": [case(dev, ckc, I) for I in (685, 418)]}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
