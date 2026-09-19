#!/usr/bin/env python3
"""How many deallocates one shipped PairformerLayer makes, split DRAM against L1.

The brief names 88 deallocates in the fast pairformer chain and asks whether the training
path retains what inference frees. `tape_profile.py`'s keeper counts only the DRAM ones,
because holding an L1 intermediate past its op changes which kernel runs. This counts both,
so the 88 can be reconciled rather than contradicted.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from perf.clocksample import during  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sizes", default="256,384,512")
    ap.add_argument("--ckpt", type=Path,
                    default=Path(os.environ.get("PTX_V2_CKPT",
                                                "/home/ttuser/protenix_ckpt/protenix-v2.pt")))
    args = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio as _TB
    from tt_bio.tenstorrent import get_device, Pairformer
    from tt_bio import protenix_weights as PW
    from tt_bio.protenix import n_blocks
    assert Path(_TB.__file__).resolve().is_relative_to(REPO)

    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
        packer_l1_acc=True)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    ck = ck.get("model", ck)
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in ck.items()}
    c_z = int(sd["layernorm_z_cycle.weight"].shape[0])
    blk = {k[len("pairformer_stack.blocks.0."):]: v for k, v in sd.items()
           if k.startswith("pairformer_stack.blocks.0.")}
    comb = {f"layers.0.{k}": v for k, v in PW.remap_pairformer_block(blk).items()}
    del ck, sd

    out = {"doc": __doc__, "env": {
        "host": socket.gethostname(), "arch": str(dev.arch()),
        "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
        "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
        "c_z": c_z, "blocks": 1, "ckpt_blocks": n_blocks({}, "x") or 48,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, "runs": []}

    shipped = ttnn.deallocate
    with during() as clk:
        pf = Pairformer(1, 32, c_z // 32, 384 // 16, 16, True, comb, ckc)
        ttnn.synchronize_device(dev)
        for n in [int(x) for x in args.sizes.split(",") if x.strip()]:
            tally = {"dram": 0, "l1": 0, "unknown": 0, "dram_b": 0}

            def counting(t, *a, **k):
                # The classification and the byte read are separate try blocks on purpose: an
                # element_size() this wheel declines must not also lose the DRAM-vs-L1 count,
                # which is the number this census exists for.
                try:
                    dram = t.memory_config().buffer_type == ttnn.BufferType.DRAM
                    tally["dram" if dram else "l1"] += 1
                except Exception:
                    dram = False
                    tally["unknown"] += 1
                if dram:
                    try:
                        v = 1
                        for d in list(t.shape):
                            v *= int(d)
                        tally["dram_b"] += v * t.element_size()
                    except Exception:
                        tally["dram_b_unknown"] = tally.get("dram_b_unknown", 0) + 1
                return shipped(t, *a, **k)

            st = ttnn.from_torch(torch.randn(1, n, 384) * 0.5, dtype=ttnn.bfloat16,
                                 layout=ttnn.TILE_LAYOUT, device=dev)
            zt = ttnn.from_torch(torch.randn(1, n, n, c_z) * 0.5, dtype=ttnn.bfloat16,
                                 layout=ttnn.TILE_LAYOUT, device=dev)
            ttnn.deallocate = counting
            try:
                so, zo = pf(st, zt, None, None, None)
                ttnn.synchronize_device(dev)
            finally:
                ttnn.deallocate = shipped
            tally["total"] = tally["dram"] + tally["l1"] + tally["unknown"]
            tally["tokens"] = n
            out["runs"].append(tally)
            for t_ in (st, zt, so, zo):
                try:
                    shipped(t_)
                except Exception:
                    pass
            print("%4d aa  deallocates total %3d  dram %3d (%.3f GB)  l1 %3d"
                  % (n, tally["total"], tally["dram"], tally["dram_b"] / 1e9, tally["l1"]),
                  flush=True)
    out["aiclk"] = clk.summary()
    out["clock_line"] = clk.line(0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(out["clock_line"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
