#!/usr/bin/env python3
"""Device seconds a seam extension would ADD, measured on the card at the round's own n.

The CEILING leg subtracts host seconds and adds device seconds, and a device figure for an
unbuilt path is a prediction. The extra-MSA stack's pair track and the template pair stack
already run on this card in tt-bio (`tt_bio/af2.py` `device_extra_msa`, `device_template`),
so both are timed here rather than estimated:

* extra-MSA, 4 blocks, TAPED forward then backward, checkpointed like the Evoformer, and the
  primal forward -- the round runs it twice forward (recycle + differentiated) and once
  backward, so the per-round cost is `2 x taped fwd + bwd`, the same pattern as `splice.py`.
* template pair stack, 2 blocks, forward only: the template stack's inputs are the target's
  template features, which carry no gradient to the sequence, so JAX runs it forward only
  and so would the card.

Masks: the pair track takes `af2_pair_masks(mask_2d)`; the round's state is 211 real of 224
on the device axis, so the mask used here zeroes the last 13 rows/columns. The extra-MSA
blocks in `Dev.stack` pass no mask (`afgrad.py:183`), so this also times the masked call
directly through the block, which is what a seam would have to run.
"""
import argparse
import json
import os
import pathlib
import statistics as st
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
_ROOT = HERE.parents[1]
for _p in (str(_ROOT / "perf" / "bcx_round"), str(_ROOT), str(_ROOT / "perf" / "bcx_afgrad"),
           str(_ROOT / "perf" / "bcx_stack")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch                                                           # noqa: E402
import meter as M                                                      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=224)
    ap.add_argument("--real", type=int, default=211)
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--params", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import afgrad as A
    from tt_bio.af2 import af2_pair_masks, load_af2_device_model
    from tt_bio.af2_weights import load_af2_state_dict

    clk = M.Clock(0.25).start()
    A._float64_layernorm()
    state = load_af2_state_dict(args.params or A.DEFAULT_PARAMS)
    dm = load_af2_device_model(state, template=True, trunk_dtype=torch.bfloat16)
    dev = A.Dev(dm)
    ag, ttnn = dev.ag, dev.ttnn
    n = args.n
    g = torch.Generator().manual_seed(0)
    z0 = torch.randn(n, n, 128, generator=g)
    zt0 = torch.randn(n, n, 64, generator=g)
    mask = torch.zeros(n)
    mask[:args.real] = 1
    mask_2d = mask[:, None] * mask[None, :]
    masks = af2_pair_masks(mask_2d, dev.device)

    def extra(i, z):
        blk = dm.device_extra_msa[i]
        const = dm._up(dm.opm_constant[i].reshape(1, 1, -1))
        return blk(blk._residual(z, const), *masks)

    res = {"n": n, "real": args.real, "reps": args.reps, "host": os.uname().nodename,
           "card": os.environ.get("TT_VISIBLE_DEVICES"), "pci": clk.pci,
           "loadavg_start": os.getloadavg(), "timings": {}}

    def timed(name, fn):
        out = []
        for r in range(args.reps + 1):   # rep 0 compiles kernels and is dropped
            dev.sync()
            t0 = time.time()
            fn()
            dev.sync()
            t1 = time.time()
            if r:
                out.append((t0, t1))
        dts = [b - a for a, b in out]
        w = clk.window(out[0][0], out[-1][1])
        res["timings"][name] = {"med": round(st.median(dts), 4), "min": round(min(dts), 4),
                                "max": round(max(dts), 4), **w}
        print(name, res["timings"][name], flush=True)

    def extra_primal():
        z = dev.up(z0)
        for i in range(4):
            z = extra(i, z)
        ttnn.deallocate(z)

    box = {}

    def extra_taped():
        zl = dev.leaf(z0)
        with dev.tt.tape():
            z = zl
            for i in range(4):
                z = ag.checkpoint(lambda t, i=i: extra(i, t), z)
        box["zl"], box["z"] = zl, z

    def extra_backward():
        zl, z = box.pop("zl"), box.pop("z")
        ag.backward([z], [dev.seed(z0, z)])
        dev.grad(zl, tuple(z0.shape))
        ag.release_pins()

    def extra_fwd_bwd():
        extra_taped()
        extra_backward()

    def template_fwd():
        dm._template_stack(zt0, mask_2d)

    timed("extra_msa_primal_4blk", extra_primal)
    timed("extra_msa_taped_fwd_4blk", lambda: (extra_taped(), box.clear()))
    timed("extra_msa_taped_fwd_plus_bwd_4blk", extra_fwd_bwd)
    timed("template_pair_stack_fwd_2blk", template_fwd)
    t = res["timings"]
    t["extra_msa_bwd_4blk_derived"] = {
        "med": round(t["extra_msa_taped_fwd_plus_bwd_4blk"]["med"]
                     - t["extra_msa_taped_fwd_4blk"]["med"], 4)}
    clk.stop()
    res["loadavg_end"] = os.getloadavg()
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
