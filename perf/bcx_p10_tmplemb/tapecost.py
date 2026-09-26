#!/usr/bin/env python3
"""Why does a block-level device timing read 25x the round's own meter?

`vjp.py` timed the four extra-MSA pair blocks at 26.317 s warm (1 fwd + 1 bwd, n=288, AICLK
1350). `bcx-p10-resident` measured the SAME four blocks at **1.054 s** per round inside a real
BindCraft 2 round, and a round is 2 taped forwards plus 1 backward. Something in the standalone
harness costs ~25x, and until it is named no block-level device number in this campaign means
anything.

Three arms on the same blocks, same tensors, same card, in one process:

* `notape` -- forward only, no `tt_bio.autograd` tape at all.
* `tape`   -- forward under the tape, no backward.
* `taped`  -- forward under the tape plus the backward, which is what `vjp.py` timed.
* `ckpt`   -- forward under the tape with `ag.checkpoint` per block, which is what
  `_Trunk.extra_msa(recompute=True)` does on the production path, and `ckpt_bwd` adds its
  backward.

`notape` fast and `tape` slow puts the cost in the tape's bookkeeping. All three slow puts it in
the blocks or the masks, and the round's 1.054 s then has to be re-read.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics as st
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_PARAMS = "/home/ttuser/bcx_e2e/af2_params/params_model_1_multimer_v3.npz"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=288)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--params", default=DEFAULT_PARAMS)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import ttnn

    from tt_bio import autograd as ag
    from tt_bio import taped_ttnn
    from tt_bio.af2 import af2_pair_masks, load_af2_device_model
    from tt_bio.af2_weights import load_af2_state_dict

    sys.path.insert(0, str(ROOT / "perf" / "bcx_stack"))
    from stack import sysfs_node
    node, pci = sysfs_node()
    clk = lambda: int(open(f"{node}/tt_aiclk").read().split()[0])  # noqa: E731

    state = load_af2_state_dict(args.params, multimer=True)
    model = load_af2_device_model(state, template=True, multimer=True, structure=False,
                                  trunk_dtype=torch.bfloat16).eval()
    dev = model._device
    masks = af2_pair_masks(torch.ones(args.n, args.n), dev)

    def up(t):
        return ttnn.from_torch(t.detach().unsqueeze(0).to(torch.bfloat16),
                               layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)

    g = torch.Generator().manual_seed(0)
    rows = {}
    for stack, blocks in (("extra_msa", model.device_extra_msa),
                          ("template", model.device_template)):
        c = 128 if stack == "extra_msa" else 64
        act = torch.randn(args.n, args.n, c, generator=g)
        cot = torch.randn(args.n, args.n, c, generator=g)
        for arm in ("notape", "tape", "taped", "ckpt", "ckpt_bwd"):
            times, clocks = [], []
            for _ in range(args.reps):
                clocks.append(clk())
                t0 = time.perf_counter()
                if arm == "notape":
                    z = up(act)
                    for block in blocks:
                        z = block(z, *masks)
                    ttnn.synchronize_device(dev)
                else:
                    leaf = ag.Tensor(up(act), requires_grad=True)
                    ckpt = arm.startswith("ckpt")
                    with taped_ttnn.tape():
                        z = leaf
                        for block in blocks:
                            if ckpt:
                                z = ag.checkpoint(lambda t, b=block: b(t, *masks), z)
                            else:
                                z = block(z, *masks)
                    ttnn.synchronize_device(dev)
                    if arm in ("taped", "ckpt_bwd"):
                        ag.backward([z], [up(cot)])
                        ttnn.synchronize_device(dev)
                    ag.release_pins()
                times.append(time.perf_counter() - t0)
            warm = times[1:] or times
            rows[f"{stack}/{arm}"] = {
                "blocks": len(blocks), "c": c,
                "warm_s": round(st.median(warm), 3),
                "per_block_s": round(st.median(warm) / len(blocks), 4),
                "all_s": [round(x, 3) for x in times],
                "aiclk_min": min(clocks), "aiclk_max": max(clocks),
                "load1": round(os.getloadavg()[0], 2)}
            print(f"{stack:10s} {arm:7s} {rows[f'{stack}/{arm}']['warm_s']:8.3f} s  "
                  f"({rows[f'{stack}/{arm}']['per_block_s']:.4f} s/block)", flush=True)

    doc = {"n": args.n, "pci": pci, "reps": args.reps, "arms": rows,
           "round_meter_reference": {
               "extra_msa_s_per_round": 1.054,
               "what_it_covers": "4 blocks, 2 taped forwards + 1 backward, n=288",
               "source": "bcx-p10-resident, state/perf10/bcx-RESIDENT.md"},
           "finished_utc": time.strftime("%FT%TZ", time.gmtime())}
    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"tapecost_n{args.n}.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps(doc, indent=1), flush=True)


if __name__ == "__main__":
    main()
