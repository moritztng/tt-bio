#!/usr/bin/env python3
"""gap_maskw: where the atom-SDPA mask writer (rfd3_bias.sparse_mask_bf16) spends its time on this chip.

The page fixture's shape: pair bias [1, 4, 6051, 128] bf16, neighbour index [1, 1, 6051, 128] uint32 sorted per row,
mask out [1, 4, 6080, 6080] bf16 (296 MB). Modes (rfd3_bias.DIAG):
  0  shipped
  1  every tile from the fill tile, scan kept (no pokes, no slot refill)
  2  shipped work, no DRAM writes
  3  every tile from the fill tile, no scan: the write alone
and two write roofs at the mask's shape: `full` (ttnn.full, a write-only op) and `clone` (reads + writes it), with
GB/s counting written bytes only, so mode 3 can be put against what this chip's DRAM takes in writes.
Mode 0 is checked element for element against a torch scatter of the same inputs. The index is synthetic (each row
draws 128 distinct keys around its own position, spread like a folded chain's neighbours); `occ` reports the
fraction of key tiles that hold any index, the number that sets how much of the op is pokes.
    python scripts/rfd3_port/gap_maskw.py OUT.json [reps]
"""
import json, os, pathlib, statistics, sys, time

import torch
import ttnn

sys.path.insert(0, os.getcwd())
from tt_bio import rfd3_bias as RB                                       # noqa: E402
from tt_bio.tenstorrent import get_device                                # noqa: E402

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "out/gap_maskw.json")
N_REP = int(sys.argv[2]) if len(sys.argv) > 2 else 7
H, L, K = 4, 6051, 128


def synthetic_index(g):
    idx = torch.empty(L, K, dtype=torch.int64)
    for i in range(L):
        near = torch.randint(-48, 49, (96,), generator=g) + i
        far = (torch.randn(96, generator=g) * 900).long() + i
        cand = torch.unique(torch.cat([torch.tensor([i]), near, far]).clamp(0, L - 1))
        while cand.numel() < K:
            cand = torch.unique(torch.cat([cand, torch.randint(0, L, (K,), generator=g)]))
        keep = cand[torch.randperm(cand.numel(), generator=g)[:K]]
        idx[i] = torch.sort(keep).values
    return idx


def main():
    dev = get_device()
    g = torch.Generator().manual_seed(0)
    idx = synthetic_index(g)
    pb_t = torch.randn(1, H, L, K, generator=g).bfloat16()
    n = (L + 31) // 32 * 32
    occ = torch.zeros(n // 32, n // 32, dtype=torch.bool)
    occ[(torch.arange(L) // 32).unsqueeze(1).expand(L, K), idx // 32] = True
    pb = ttnn.from_torch(pb_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    ix = ttnn.from_torch(idx.to(torch.int32).reshape(1, 1, L, K), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                         device=dev)
    rec = {"reps": N_REP, "card": os.environ.get("TT_VISIBLE_DEVICES"), "host": os.uname().nodename,
           "arch": str(dev.arch()), "occ": float(occ.float().mean()), "floor_ms_248GBs": H * n * n * 2 / 248e6}
    for mode in (0, 1, 2, 3):
        RB.DIAG = mode
        o = RB.sparse_mask_bf16(pb, ix)
        ttnn.synchronize_device(dev)
        ts = []
        for _ in range(N_REP):
            ttnn.deallocate(o)
            t0 = time.perf_counter()
            o = RB.sparse_mask_bf16(pb, ix)
            ttnn.synchronize_device(dev)
            ts.append((time.perf_counter() - t0) * 1e3)
        r = {"ms_median": statistics.median(ts), "ms_min": min(ts), "ms_max": max(ts)}
        if mode == 0:
            d = ttnn.to_torch(o)[:, :, :L, :L]
            ref = torch.full((1, H, L, L), -1e4).bfloat16()
            ref.scatter_(3, idx.reshape(1, 1, L, K).expand(1, H, L, K), pb_t)
            r["equal_torch_scatter"] = bool(torch.equal(d, ref))
        ttnn.deallocate(o)
        rec[f"mode{mode}"] = r
        print(mode, r, flush=True)
    RB.DIAG = 0
    shape, wbytes = [1, H, n, n], H * n * n * 2
    src = ttnn.full(shape, -1e4, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    for name, fn in (("full", lambda: ttnn.full(shape, -1e4, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)),
                     ("clone", lambda: ttnn.clone(src))):
        ttnn.deallocate(fn())
        ttnn.synchronize_device(dev)
        ts = []
        for _ in range(N_REP):
            t0 = time.perf_counter()
            o = fn()
            ttnn.synchronize_device(dev)
            ts.append((time.perf_counter() - t0) * 1e3)
            ttnn.deallocate(o)
        rec[name] = {"ms_median": statistics.median(ts), "write_GBs": wbytes / statistics.median(ts) / 1e6}
        print(name, rec[name], flush=True)
    ttnn.deallocate(src)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
