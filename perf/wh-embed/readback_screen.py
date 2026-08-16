#!/usr/bin/env python3
"""Lever 2's screen: what does one discarded hidden-state readback cost?

ESMCHiddenStatesModel.__call__ copies every one of its n_layers+1 hidden states to the host,
because ESMFold2's LanguageModelShim consumes all of them. The standalone embed path does not:
_trunk_forward takes model(tokens)[-1, 0] and throws the other n_layers away. For esmc-6b that
is 80 discarded [B, Lb, 2560] copies per forward.

This times the readback in isolation so the saving can be predicted BEFORE the change is built.
Two arms, because they answer different questions:

  t_rb       ttnn.to_torch(t).float() on a [1, Lb, 2560] bf16 device tensor -- the copy itself.
  t_rb_sync  the same, but with a preceding op queued so the copy also pays the pipeline drain
             it forces in the real forward. The real cost sits between these two: in the model
             the host cannot dispatch block i+1 until block i's copy has returned.

Reports both, plus bytes/time against the measured DRAM roof so the number is placed on a roof
rather than quoted bare.
"""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-model", type=int, default=2560)
    ap.add_argument("--lb", type=int, default=128, help="bucketed length (76 aa -> 128)")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--n-discarded", type=int, default=80,
                    help="esmc-6b has n_layers=80 discarded readbacks per forward")
    ap.add_argument("--dram-roof-GBps", type=float, default=None,
                    help="measured DRAM roof for the bytes/time sanity check")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import ttnn

    from tt_bio.tenstorrent import get_device
    dev = get_device()

    shape = (args.batch, args.lb, args.d_model)
    host = torch.randn(*shape, dtype=torch.float32).to(torch.bfloat16)
    t = ttnn.from_torch(host, device=dev, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    nbytes = args.batch * args.lb * args.d_model * 2

    def med(xs):
        return statistics.median(xs)

    # arm 1: the copy alone
    for _ in range(3):
        torch.Tensor(ttnn.to_torch(t)).float()
    a1 = []
    for _ in range(args.reps):
        t0 = time.perf_counter()
        torch.Tensor(ttnn.to_torch(t)).float()
        a1.append(time.perf_counter() - t0)

    # arm 2: the copy behind queued device work, so it also pays the drain it forces
    def one_sync():
        u = ttnn.mul(t, 1.0)
        out = torch.Tensor(ttnn.to_torch(u)).float()
        ttnn.deallocate(u)
        return out

    for _ in range(3):
        one_sync()
    a2 = []
    for _ in range(args.reps):
        t0 = time.perf_counter()
        one_sync()
        a2.append(time.perf_counter() - t0)

    t_rb, t_rb_sync = med(a1), med(a2)
    res = dict(
        arch=os.environ.get("PROBE_ARCH", "unknown"),
        visible_devices=os.environ.get("TT_VISIBLE_DEVICES", ""),
        shape=list(shape), bytes=nbytes, reps=args.reps,
        n_discarded=args.n_discarded,
        t_rb_ms=round(t_rb * 1000, 4),
        t_rb_sync_ms=round(t_rb_sync * 1000, 4),
        t_rb_ms_all=[round(x * 1000, 4) for x in a1[:5]],
        implied_GBps=round(nbytes / t_rb / 1e9, 2),
        implied_GBps_sync=round(nbytes / t_rb_sync / 1e9, 2),
        predicted_saving_lower_ms=round(args.n_discarded * t_rb * 1000, 2),
        predicted_saving_upper_ms=round(args.n_discarded * t_rb_sync * 1000, 2),
        loadavg=round(os.getloadavg()[0], 2),
    )
    if args.dram_roof_GBps:
        res["dram_roof_GBps"] = args.dram_roof_GBps
        res["frac_of_roof"] = round(res["implied_GBps"] / args.dram_roof_GBps, 4)
        res["exceeds_roof"] = res["implied_GBps"] > args.dram_roof_GBps
    ttnn.deallocate(t)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
