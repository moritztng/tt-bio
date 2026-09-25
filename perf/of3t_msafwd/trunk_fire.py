#!/usr/bin/env python3
"""TRUNK firing count for of3t-msafwd: every ttnn.add_ issued from PairformerLayer.__call__,
keyed by the stack that owns the layer (the MSA module's pair_stack or the trunk's 48-block
Pairformer), the operand dtypes and the result dtype. Both stacks are built by OF3Trunk.__init__
from the shipped checkpoint, so each carries the construction arguments inference uses, and both
run on the 64-token crop boundary: the trunk Pairformer on the MSA module's own z_out with a
random bf16 s. A count, not a code read."""
import json
import os
import socket
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))


def main():
    import ttnn
    from tt_bio.tenstorrent import get_device, PairformerLayer
    from tt_bio.openfold3_trunk import OF3Trunk
    B = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    (m0, z0), kw = B["inputs"]["args"], B["inputs"]["kwargs"]
    n, n_seq = int(z0.shape[-2]), int(m0.shape[-3])
    sd = torch.load(os.path.expanduser("~/of3-weights/of3-p2-155k.pt"), map_location="cpu",
                    weights_only=False)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
        packer_l1_acc=True)
    tr = OF3Trunk(sd, ckc)
    del sd
    owner = {}
    for blk in tr.msa_module.blocks:
        owner[id(blk.pair_stack)] = "msa_module.pair_stack"
    for v in vars(tr.pairformer).values():
        if isinstance(v, (list, tuple)):
            for lay in v:
                if isinstance(lay, PairformerLayer):
                    owner[id(lay)] = "trunk.pairformer"
    orig = ttnn.add_
    census = {}

    def add_(a, b, *ar, **k):
        f = sys._getframe(1)
        if f.f_code.co_name == "_z_residual":   # the add's home since the z_fp32_residual patch
            f = f.f_back
        me = f.f_locals.get("self")
        out = orig(a, b, *ar, **k)
        if isinstance(me, PairformerLayer) and f.f_code.co_name == "__call__":
            key = "%s | add_(%s, %s) line %d -> %s" % (
                owner.get(id(me), "other"), a.dtype, b.dtype, f.f_lineno, out.dtype)
            census[key] = census.get(key, 0) + 1
        return out

    ttnn.add_ = add_
    up = lambda x: ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=dev,
                                   dtype=ttnn.bfloat16)
    pm = kw["pair_mask"].reshape(1, n, n)
    m1 = torch.diagonal(pm.reshape(n, n)).reshape(1, n).clamp(0, 1)
    pm_d, attn_d = up(pm), up((1 - m1).reshape(1, 1, 1, n) * -1e9)
    _, z = tr.msa_module(up(m0.reshape(1, n_seq, n, -1)), up(z0.reshape(1, n, n, -1)),
                         pm_d, attn_d)
    torch.manual_seed(0)
    s = up(torch.randn(1, n, 384))
    tr.pairformer(s, z, pm_d, attn_d, attn_d)
    ttnn.add_ = orig
    vals = list(owner.values())
    n_layers = {v: vals.count(v) for v in set(vals)}
    out = {"host": socket.gethostname(), "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "board": open("/sys/class/tenstorrent/tenstorrent!%s/tt_card_type"
                         % os.environ.get("TT_VISIBLE_DEVICES", "0")).read().strip(),
           "boundary": sys.argv[1], "n_tokens": n, "layers": n_layers, "census": census}
    for k_, v in sorted(census.items()):
        print("%5d  %s" % (v, k_))
    print("layers per owner:", n_layers)
    Path(sys.argv[2]).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
