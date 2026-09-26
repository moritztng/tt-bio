#!/usr/bin/env python3
"""What one un-checkpointed block costs in DRAM, and how deep a stack fits on the card.

`recompute=False` OOMed the shipped round on its first no-checkpoint forward: the allocator
filled every DRAM bank and failed on a 42 467 328 B buffer -- a [288, 288, 128] float32 pair
tensor -- with 253 952 B free. That is the verdict on the lever, and it does not say by how
much. This does: it takes a taped forward one block at a time and reads DRAM after each, with
and without `autograd.checkpoint`, so the per-block footprint is measured rather than divided
out of an OOM message.

No JAX, no BindCraft 2 campaign, so it does not touch the host-global XLA compile lock and
runs in under a minute. The blocks, the tape and the checkpoint are the shipped ones.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import pathlib
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "perf" / "bcx_stack")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from perf.bcx_stack import stack as S            # noqa: E402
from perf.bcx_afgrad import afgrad as A          # noqa: E402

OUT = ROOT / "perf" / "bcx_p10_resid"
GB = 1 << 30


def dram(ttnn, device):
    mv = ttnn.get_memory_view(device, ttnn.BufferType.DRAM)
    return (int(mv.total_bytes_allocated_per_bank) * int(mv.num_banks),
            int(mv.total_bytes_per_bank) * int(mv.num_banks))


def walk(dev, ttnn, m0, z0, mask, blocks, ckpt):
    """A taped forward of `blocks` Evoformer blocks, DRAM read after each one."""
    ag = dev.ag
    gc.collect()
    base, total = dram(ttnn, dev.device)
    ml, zl = dev.leaf(m0), dev.leaf(z0)
    trail = []
    failed = None
    t0 = time.time()
    try:
        with dev.tt.tape():
            m, z = ml, zl
            for i in range(blocks):
                if ckpt:
                    m, z = ag.checkpoint(
                        lambda a, b, i=i: dev.evo(i, a, b, mask, (None, None)), m, z)
                else:
                    m, z = dev.evo(i, m, z, mask, (None, None))
                dev.sync()
                trail.append(dram(ttnn, dev.device)[0])
    except Exception as exc:                        # the OOM is the measurement
        failed = f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"
    wall = time.time() - t0
    got = len(trail)
    out = {"ckpt": ckpt, "blocks_asked": blocks, "blocks_done": got,
           "wall_s": round(wall, 3), "failed": failed,
           "dram_total_gb": round(total / GB, 3),
           "dram_base_gb": round(base / GB, 4),
           "dram_after_gb": [round(x / GB, 4) for x in trail]}
    if got >= 2:
        steps = [trail[i] - trail[i - 1] for i in range(2, got)]
        if steps:
            steps.sort()
            out["per_block_gb"] = round(steps[len(steps) // 2] / GB, 4)
            out["per_block_gb_min"] = round(steps[0] / GB, 4)
            out["per_block_gb_max"] = round(steps[-1] / GB, 4)
    try:
        del m, z
    except Exception:
        pass
    del ml, zl
    ag.release_pins()
    gc.collect()
    dev.sync()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default=A.DEFAULT_PARAMS)
    ap.add_argument("--card", type=int, default=int(os.environ.get("TT_VISIBLE_DEVICES", "0")))
    ap.add_argument("--n", type=int, default=288)
    ap.add_argument("--blocks", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default="ckptmem_n288.json")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    import ttnn
    lv, dev, ref = S.open_all(args)
    clock = S.Clock()
    m0, z0, _, _ = S.inputs(ref, args.n, args.seed)
    mask = dev.up(torch.ones(1, args.n))

    blob = {"stamp": S.stamp(args, clock), "n": args.n, "blocks": args.blocks,
            "loadavg_start": os.getloadavg(), "arms": []}
    # Checkpointed first: it is the shipped arm and it always fits, so the card is in a known
    # state when the arm that may not fit runs.
    for ckpt in (True, False):
        got = walk(dev, ttnn, m0, z0, mask, args.blocks, ckpt)
        got["aiclk"] = clock.window([(time.time() - got["wall_s"], time.time())])
        got["loadavg"] = os.getloadavg()[0]
        blob["arms"].append(got)
        print(json.dumps({k: v for k, v in got.items() if k != "dram_after_gb"}), flush=True)
    clock.stop()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / args.out).write_text(json.dumps(blob, indent=1, default=str))
    print(f"wrote {OUT / args.out}", flush=True)


if __name__ == "__main__":
    main()
