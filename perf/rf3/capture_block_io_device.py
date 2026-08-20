#!/usr/bin/env python3
"""A Pairformer-block parity fixture at a REAL fold size, captured off the device trunk.

Every fixture `scripts/rf3_port/capture_trunk_io.py` produced is 8 to 53 tokens, so the padded
sequence the device sees is one or two tiles. `perf/rf3/triatt_len_ladder.py` measured an order of
magnitude of accuracy difference between a one-tile sequence and everything above 64 tokens, which
means those fixtures cannot score an attention kernel at any size anyone folds -- and that is a gap
in the port's gate, not in any one lever.

This closes it without a reference forward. The (s, z) entering the 48-block stack is taken off a
real device recycle at `--aa`, which is on-manifold by construction, and written in the same format
`--golden` reads. `parity_pairformer.py` then computes its own torch golden from that input, so
both arms and the reference all see identical numbers and the measurement stays real.

The device tensors are tile-padded; only the logical `n_token` rows are kept, so the fixture is a
clean N x N block with no padding of its own.
"""
from __future__ import annotations

import argparse
import enum
import sys
from pathlib import Path

if not hasattr(enum, "StrEnum"):
    class StrEnum(str, enum.Enum):
        def __str__(self):
            return str(self.value)
    enum.StrEnum = StrEnum

import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aa", type=int, default=128)
    ap.add_argument("--ckpt", default="/home/ttuser/rf3_perf_work/rf3_latest.ckpt")
    ap.add_argument("--recycle", type=int, default=1,
                    help="how many recycles to run before capturing; 1 is the first pass")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from tt_bio.rf3.featurize import featurize
    fo = featurize(str(REPO / f"perf/rf3/inputs/rf3_{args.aa}.json"),
                   n_recycles=2, diffusion_batch_size=1, seed=args.seed)[0]

    import ttnn
    import tt_bio.tenstorrent as T
    from tt_bio.rf3 import model as rf3_model
    from tt_bio.rf3.host import HostInputs
    from perf.rf3.tt_rf3_bench import net_config

    cfg = net_config(args.ckpt)
    device = T.get_device()
    kcfg = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    tt = rf3_model.load(
        args.ckpt, kcfg,
        n_pairformer_blocks=cfg["recycler"]["n_pairformer_blocks"],
        n_msa_blocks=cfg["recycler"]["msa_module"]["n_block"],
        n_dit_blocks=cfg["diffusion_module"]["diffusion_transformer"]["n_block"],
        num_timesteps=50, with_confidence=False)

    host = HostInputs.build(fo["feats"], device)
    n = int(host.n_token)
    grab = {}
    pf = tt.recycler.pairformer

    class Spy:
        """A wrapper, not an instance attribute: Python looks `__call__` up on the type, so
        assigning `pf.__call__` is silently ignored and the stack runs unhooked."""

        def __init__(self, inner):
            self.inner = inner

        def __call__(self, s, z):
            if not grab:
                grab["s"] = ttnn.to_torch(s)
                grab["z"] = ttnn.to_torch(z)
            return self.inner(s, z)

    tt.recycler.pairformer = Spy(pf)
    s_inputs, s_init, z_init = tt.feature_initializer(
        host.single_in, host.pair_in, host.pair_v, host.keys_indexing,
        host.atom_to_token_mean, host.window_mask, host.n_atom_padded,
        host.token_feats, host.relpos_feat, host.bond_feat)
    tmpl = tt.recycler.template_embedder.embed_template_feats(host.template_feats)
    s, z = ttnn.mul(s_init, 0.0), ttnn.mul(z_init, 0.0)
    for i in range(args.recycle):
        s, z = tt.recycler(host, tmpl, host.msa_stack[i % len(host.msa_stack)],
                           s_inputs, s_init, z_init, s, z)
    tt.recycler.pairformer = pf
    assert grab, "the Pairformer stack was never entered"

    sh, zh = grab["s"].float(), grab["z"].float()
    sh = sh.reshape(-1, sh.shape[-1])[:n]                 # [I, C_S], padding dropped
    zh = zh.reshape(zh.shape[-3], zh.shape[-2], -1)[:n, :n]   # [I, I, C_Z]
    torch.save({"in": (sh, zh), "tokens": n, "aa": args.aa,
                "source": "device trunk, tt_bio.rf3 recycler"}, args.out)
    print(f"wrote {args.out}: s{list(sh.shape)} z{list(zh.shape)} "
          f"tokens={n} s_std={sh.std():.4f} z_std={zh.std():.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
