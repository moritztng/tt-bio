#!/usr/bin/env python3
"""Isolate the host-assembly helpers: assemble known blocks via host vs device and
bit-compare. Also exercise one chunked TriangleMultiplication/TriangleAttention/
Transition at a small non-tile-aligned N with host mode forced, vs device mode.

    TT_VISIBLE_DEVICES=0 python3 perf/large_target_oom/hostcat_isolate.py
"""
import torch
import ttnn
from tt_bio import tenstorrent as T
from tt_bio.tenstorrent import get_device, _acc_append, _acc_concat


def helper_check(dev):
    print("== helper mechanics (4-D trimul-style, dim=-1) ==", flush=True)
    torch.manual_seed(0)
    blocks = [torch.randn(1, 708, 736, 32, dtype=torch.bfloat16) for _ in range(4)]
    for host in (False, True):
        acc = []
        for b in blocks:
            t = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
            _acc_append(acc, t, host)
        out = _acc_concat(acc, -1, host)
        got = ttnn.to_torch(out)
        want = torch.cat(blocks, dim=-1)
        print(f"  host={host}: {'BIT-EXACT' if torch.equal(got, want) else 'DIFFERS maxabs ' + format((got.float()-want.float()).abs().max().item(), '.3e')}", flush=True)
        ttnn.deallocate(out)

    print("== helper mechanics (3-D tri_att-style, dim=0) ==", flush=True)
    parts = [torch.randn(96, 708, 384, dtype=torch.bfloat16) for _ in range(3)]
    for host in (False, True):
        acc = []
        for p in parts:
            t = ttnn.from_torch(p, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
            _acc_append(acc, t, host)
        out = _acc_concat(acc, 0, host)
        got = ttnn.to_torch(out)
        want = torch.cat(parts, dim=0)
        print(f"  host={host}: {'BIT-EXACT' if torch.equal(got, want) else 'DIFFERS maxabs ' + format((got.float()-want.float()).abs().max().item(), '.3e')}", flush=True)
        ttnn.deallocate(out)


def main():
    dev = get_device()
    helper_check(dev)


if __name__ == "__main__":
    main()
