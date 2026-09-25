#!/usr/bin/env python3
"""Does the zero swap move the gradient? Asked where the answer is deterministic.

`gradcompare.py` asked this over the whole 2,482-node tape and could not answer it: the
backward's own run-to-run spread at crop 384 under `exact_training(False)` reaches a worst-case
relative L2 of 1.03-1.57 on three `msa_module.blocks.*.pwa.z_norm_bias` gradients, the SAME
arm twice, so the floor is the size of anything a zero swap could do and the verdict flipped
between two runs of that very comparison. A floor that large cannot clear a change, and it
cannot convict one either.

This asks the question one level down, where it IS deterministic and where the whole of the
change lives. The change substitutes one all-zeros bf16 tensor for another inside
`_v_create_qkv_heads`' backward. Two things decide it completely:

  1. are the two zero tensors byte-identical to each other, and are both exactly zero;
  2. is the closure's OUTPUT -- `concat([rows, zero, zero])` at the real pair-track shape --
     byte-identical between the two zero sources.

If (2) holds, the closure emits the same bytes it emitted before and the VJP through it cannot
have moved. Anything `gradcompare.py` then sees is the backward's own nondeterminism, which it
measures A/A anyway. Run at every slot, because slot 0, 1 and 2 place the rows at a different
offset in the packed width and only one of them is the contiguous-front case.

    python3 perf/of3t_zerosfill/zeroident.py --out perf/of3t_zerosfill/out/zeroident.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import ttnn                                                              # noqa: E402

# B, 1, L, H*dh at crop 384 -- the shape of the 324 calls of3t-bwattrib attributed.
B, L, W = 384, 384, 128
SHAPE = [B, 1, L, W]


def sha(t):
    import torch
    x = ttnn.to_torch(t).contiguous()
    return hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest()[:32]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="perf/of3t_zerosfill/out/zeroident.json")
    a = ap.parse_args()

    import torch
    from tt_bio import autograd as ag

    res = {"doc": __doc__.splitlines()[0], "argv": sys.argv, "shape": SHAPE, "env": {
        "host": socket.gethostname(),
        "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                 capture_output=True, text=True).stdout.strip(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "board": "pc card 0 -- Blackhole p150a, custom 130-core firmware"}}

    dev = ttnn.open_device(device_id=0)
    try:
        dt, lay = ttnn.bfloat16, ttnn.TILE_LAYOUT
        # A stand-in for merge_heads_value(g): arbitrary, nonzero, and the same tensor for
        # both arms, so the only thing that varies is where the zeros came from.
        g = torch.linspace(-3.0, 3.0, B * L * W).reshape(SHAPE).to(torch.bfloat16)
        rows = ttnn.from_torch(g, layout=lay, dtype=dt, device=dev)

        ag.DEVICE_ZEROS = False
        z_host = ag.grad_zeros(SHAPE, dt, dev)
        ag.DEVICE_ZEROS = True
        z_dev = ag.grad_zeros(SHAPE, dt, dev)
        ttnn.synchronize_device(dev)

        th, td = ttnn.to_torch(z_host), ttnn.to_torch(z_dev)
        res["zeros"] = {
            "host_sha256": sha(z_host), "device_sha256": sha(z_dev),
            "host_absmax": float(th.abs().max()), "device_absmax": float(td.abs().max()),
            "host_numel": int(th.numel()), "device_numel": int(td.numel()),
            "byte_identical": sha(z_host) == sha(z_dev),
            "both_exactly_zero": bool(th.abs().max() == 0 and td.abs().max() == 0),
        }
        print("zeros:", json.dumps(res["zeros"]), flush=True)

        # The closure's output, at every slot.
        res["concat_by_slot"] = {}
        for s in (0, 1, 2):
            oh = ttnn.concat([rows if i == s else z_host for i in range(3)], dim=-1)
            od = ttnn.concat([rows if i == s else z_dev for i in range(3)], dim=-1)
            ttnn.synchronize_device(dev)
            hh, hd = sha(oh), sha(od)
            r = {"host_sha256": hh, "device_sha256": hd, "byte_identical": hh == hd,
                 "shape": [int(d) for d in oh.shape]}
            res["concat_by_slot"][str(s)] = r
            print("slot", s, json.dumps(r), flush=True)
            ttnn.deallocate(oh)
            ttnn.deallocate(od)

        res["verdict"] = {
            "zeros_byte_identical": res["zeros"]["byte_identical"],
            "both_exactly_zero": res["zeros"]["both_exactly_zero"],
            "closure_output_byte_identical_every_slot":
                all(v["byte_identical"] for v in res["concat_by_slot"].values()),
        }
        res["verdict"]["vjp_unchanged_through_this_closure"] = bool(
            res["verdict"]["both_exactly_zero"]
            and res["verdict"]["closure_output_byte_identical_every_slot"])
    finally:
        ttnn.close_device(dev)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print()
    print(json.dumps(res["verdict"], indent=1))
    print("wrote", a.out)
    return 0 if res["verdict"]["vjp_unchanged_through_this_closure"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
