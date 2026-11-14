#!/usr/bin/env python3
"""gap_dit_permute: the DiT's stacked pair-bias projection + head-first permute, against transpose-first forms.

`LocalTokenTransformer._dit_sdpa_masks` projects z [1, I, IP, 128] to every block's heads at once ([.., 288]) and
permutes to [1, 288, I, IP]: 6.2 ms a call in Tracy fast (2 calls/step), ~4x its DRAM floor, because a tiled permute
that moves the last dim transposes inside every tile. The candidates move the intra-tile transpose onto z (128
channels, 123 MB, less than half of the 285 MB output) and leave a tile-only permute of non-last dims:
  shipped  _tuned_linear(z, W) then permute (0, 3, 1, 2)
  tmm      transpose(z, -2, -1) -> matmul(W^T broadcast over rows, zt) -> [1, I, 288, IP] -> permute (0, 2, 1, 3)
  tmm_rep  as tmm with W^T repeated over the I rows once (weights only, cacheable), for a matmul without in0 broadcast
Every arm is scored against a float64 einsum of the same bf16 inputs.

    python scripts/rfd3_port/gap_dit_permute.py OUT.json [reps]
"""
import json, os, pathlib, statistics, sys, time
import torch, ttnn
sys.path.insert(0, os.getcwd())
from tt_bio.rfd3 import model as M                                       # noqa: E402
from tt_bio.tenstorrent import get_device                                # noqa: E402

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "out/gap_dit_permute.json")
N = int(sys.argv[2]) if len(sys.argv) > 2 else 5
I, IP, C, H = 685, 704, 128, 16 * 18


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


def main():
    dev = get_device()
    torch.manual_seed(42)
    ckc = M._default_compute_kernel_config()
    z_t = torch.randn(1, I, IP, C).bfloat16()
    w_t = (torch.randn(C, H) / C ** 0.5).bfloat16()
    ref = torch.einsum("bijc,ch->bhij", z_t.double(), w_t.double())
    up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)  # noqa: E731
    z, w = up(z_t), up(w_t)
    wt = up(w_t.t().contiguous().reshape(1, 1, H, C))

    def shipped():
        pb = M._tuned_linear(z, w, ckc=ckc, dtype=ttnn.bfloat16, core_grid=M.CORE_GRID_MAIN)
        out = ttnn.permute(pb, (0, 3, 1, 2))
        ttnn.deallocate(pb)
        return out

    def tmm_with(w_rows):
        def run():
            zt = ttnn.transpose(z, -2, -1)
            pb = ttnn.matmul(w_rows, zt, compute_kernel_config=ckc, dtype=ttnn.bfloat16)
            ttnn.deallocate(zt)
            out = ttnn.permute(pb, (0, 2, 1, 3))
            ttnn.deallocate(pb)
            return out
        return run

    arms = {"shipped": shipped, "tmm": tmm_with(wt)}
    try:
        arms["tmm_rep"] = tmm_with(ttnn.repeat(wt, ttnn.Shape([1, I, 1, 1])))
    except Exception as e:                                               # noqa: BLE001
        print("tmm_rep: repeat failed:", repr(e)[:200], flush=True)
    rec = {"shape": [1, I, IP, C], "heads": H, "reps": N, "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "host": os.uname().nodename, "arch": str(dev.arch()), "arms": {}}
    for name, fn in arms.items():
        try:
            o, med, lo, hi = timeit(fn, dev)
            d = ttnn.to_torch(o).double()
            assert tuple(d.shape) == tuple(ref.shape), (tuple(d.shape), tuple(ref.shape))
            assert torch.isfinite(d).all(), "non-finite output"
            r = {"ms_median": med, "ms_min": lo, "ms_max": hi,
                 "rel_err_f64": float((d - ref).norm() / ref.norm()), "max_abs_f64": float((d - ref).abs().max())}
            ttnn.deallocate(o)
        except Exception as e:                                           # noqa: BLE001
            r = {"error": repr(e)[:400]}
        rec["arms"][name] = r
        print(name, r, flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
