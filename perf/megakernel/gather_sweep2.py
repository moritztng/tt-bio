#!/usr/bin/env python3
"""W4 milestone 1c: the exchange sweep with the degenerate cases excluded.

1b swept tiles-per-block and found the copy floor is set by the reader/writer block
granularity, not by the exchange: 937 GB/s at 4 tiles per block against 461 at 32. But two of
its rows are not measurements of an exchange at all. The permutation is
`m -> (m%32)*(M/32) + m/32` over M = tpb*2048/R pieces, so when M == 32 it is the identity.
Every row here has M > 32 by construction and is checked to differ from the input, so the
"perm" note means a real shuffle happened, not a copy that passed a sorted-values check.

    TT_VISIBLE_DEVICES=3 python3 perf/megakernel/gather_sweep2.py
"""
import json, sys, time
from pathlib import Path
import torch, ttnn
from tt_bio.tenstorrent import get_device
import importlib.util

d = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("gs", str(d / "gather_sweep.py"))
gs = importlib.util.module_from_spec(spec)
sys.argv = [sys.argv[0]]
spec.loader.exec_module(gs)

N = C = None


def main():
    global N, C
    N, C = 320, 64
    dev = get_device()
    L1 = ttnn.L1_MEMORY_CONFIG
    torch.manual_seed(0)
    tin = torch.randn(1, N, N, C)
    inp = ttnn.from_torch(tin, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                          memory_config=L1)
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, N, N, C]), ttnn.bfloat16,
                                         ttnn.TILE_LAYOUT, dev, L1)
    mb = 2 * (N * N * C * 2) / 1e6
    ref = tin.to(torch.bfloat16)
    ref_sorted = torch.sort(ref.flatten().float()).values
    rows = []
    print(f"\n=== real (non-identity) 32-way exchange, [1,{N},{N},{C}] L1->L1 {mb:.1f} MB r+w,"
          f" grid 10x10; ttnn.clone 1168.9, ttnn.permute(0,2,1,3) 205.6 GB/s ===", flush=True)
    for tpb in (8, 16, 32):
        for depth in (2, 4):
            for piece in (None, 64, 128, 256, 512):
                if piece is not None:
                    M = tpb * 2048 // piece
                    if M <= 32 or M % 32:
                        continue
                lbl = f"tpb={tpb:2d} d={depth} " + ("copy    " if piece is None
                                                    else f"R={piece:4d}B")
                try:
                    pd = gs.build(dev, inp, out, tpb, depth, 10, 10, piece)
                    got = ttnn.to_torch(ttnn.generic_op([inp, out], pd))
                    if piece is None:
                        note = "exact" if torch.equal(got, ref) else "MISMATCH"
                    else:
                        isperm = torch.equal(torch.sort(got.flatten().float()).values,
                                             ref_sorted)
                        diff = not torch.equal(got, ref)
                        note = ("perm+shuffled" if (isperm and diff)
                                else f"perm={isperm} differs={diff}")
                    for _ in range(2):
                        for _ in range(16):
                            ttnn.generic_op([inp, out], pd)
                    ttnn.synchronize_device(dev)
                    ts = []
                    for _ in range(5):
                        ttnn.synchronize_device(dev)
                        t0 = time.perf_counter()
                        for _ in range(16):
                            ttnn.generic_op([inp, out], pd)
                        ttnn.synchronize_device(dev)
                        ts.append((time.perf_counter() - t0) * 1e3 / 16)
                    ms = sorted(ts)[len(ts) // 2]
                    rows.append(dict(arm=lbl, ms=round(ms, 4), eff_gbs=round(mb / ms, 1),
                                     note=note))
                    print("  %-24s %8.4f ms %7.1f GB/s  %s" % (lbl, ms, mb / ms, note),
                          flush=True)
                except Exception as e:
                    print("  %-24s FAILED %s: %s" % (lbl, type(e).__name__, str(e)[:110]),
                          flush=True)
    Path(d / "gather_sweep2_n320.json").write_text(
        json.dumps(dict(n=N, c=C, mb=mb, grid="10x10", rows=rows), indent=2) + "\n")
    from tt_bio.tenstorrent import cleanup
    cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
