#!/usr/bin/env python3
"""The t-independent hoist: is it bit-exact, and what does it buy on the rollout.

`DiffusionConditioning.pair` takes no `t`, and inside the atom encoder and decoder the
noisy coordinates reach the single track only, so `z_cond`, the whole atom-pair track and
both stacks of windowed biases are the same arithmetic on the same inputs on all 49 x D
denoiser calls. Hoisting them is bit-exact BY CONSTRUCTION, which is a reason to verify it
and not a reason to assert it: the allocation order changes, and this port has already been
bitten by allocation-order sensitivity once.

Both arms in one process off one checkpoint load, arm A run twice so the A/A floor is
measured, and arm B replays arm A's recorded draws -- a cross-RNG coordinate comparison
produces a plausible structure and a meaningless number.
"""
from __future__ import annotations

import argparse
import enum
import json
import sys
import time
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
    ap.add_argument("--aa", type=int, default=512)
    ap.add_argument("--ckpt", default="/home/ttuser/rf3_perf_work/rf3_latest.ckpt")
    ap.add_argument("--n_recycles", type=int, default=2)
    ap.add_argument("--num_steps", type=int, default=50)
    ap.add_argument("--diffusion_batch_size", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--feat_cache", default="/home/ttuser/rf3_perf_work/featcache")
    ap.add_argument("--lever", choices=("hoist", "dit_bias"), default="hoist",
                    help="hoist: arm B hoists the t-independent half of the denoiser. "
                         "dit_bias: both arms hoist, arm B also builds the token DiT's 24 "
                         "pair biases once instead of on every call.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import ttnn
    from tt_bio.rf3 import model as rf3_model
    from tt_bio.rf3.host import HostInputs
    from tt_bio.rf3.sampler import Draws
    from tt_bio.tenstorrent import get_device
    from perf.rf3.featcache import featurized
    from perf.rf3.tt_rf3_bench import net_config

    fo = featurized(str(REPO / f"perf/rf3/inputs/rf3_{args.aa}.json"),
                    n_recycles=max(args.n_recycles, 2),
                    diffusion_batch_size=args.diffusion_batch_size, seed=args.seed,
                    cache_dir=args.feat_cache or None)
    f = fo["feats"]

    cfg = net_config(args.ckpt)
    device = get_device()
    kcfg = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    tt = rf3_model.load(
        args.ckpt, kcfg,
        n_pairformer_blocks=cfg["recycler"]["n_pairformer_blocks"],
        n_msa_blocks=cfg["recycler"]["msa_module"]["n_block"],
        n_dit_blocks=cfg["diffusion_module"]["diffusion_transformer"]["n_block"],
        num_timesteps=args.num_steps, with_confidence=False)

    host = HostInputs.build(f, device)
    s_inputs, s, z = tt.trunk(host, args.n_recycles)
    D = args.diffusion_batch_size
    dm = tt.diffusion_module

    def prep(arm_b: bool):
        """The prepared state each arm runs with, for whichever lever is under test."""
        if args.lever == "hoist":
            return dm.prepare(host, s_inputs, s, z) if arm_b else None
        rf3_model._HOIST_DIT_BIAS = arm_b
        try:
            return dm.prepare(host, s_inputs, s, z)
        finally:
            rf3_model._HOIST_DIT_BIAS = True

    def roll(hoist: bool, draws):
        prepared = prep(hoist)
        calls = [0]

        def denoise(x_noisy, t):
            calls[0] += 1
            return dm(host, x_noisy, t, s_inputs, s, z, prepared)

        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        x, dr = tt.sampler.sample(denoise, torch.zeros(D, host.n_atom, 3), D,
                                  draws=draws)
        ttnn.synchronize_device(device)
        return x, dr, time.perf_counter() - t0, calls[0]

    # ---- 1. one denoiser call, fixed inputs, no RNG anywhere in the comparison ------
    g = torch.Generator().manual_seed(7)
    x_fixed = torch.randn(D, host.n_atom, 3, generator=g) * 10.0
    t_fixed = torch.full((D,), 4.0)

    def one_call(hoist: bool):
        return dm(host, x_fixed, t_fixed, s_inputs, s, z, prep(hoist))

    ca, cb, ca2 = one_call(False), one_call(True), one_call(False)
    call_ab = float((ca.double() - cb.double()).abs().max())
    call_aa = float((ca.double() - ca2.double()).abs().max())

    # ---- 2. the rollout, warm, arm B replaying arm A's recorded draws ---------------
    _, rec, _, _ = roll(False, Draws())               # warm-up AND the draw record
    xa, _, ta, na = roll(False, Draws(list(rec.values)))
    xb, _, tb, nb = roll(True, Draws(list(rec.values)))
    xa2, _, ta2, _ = roll(False, Draws(list(rec.values)))

    d_ab = float((xa.double() - xb.double()).abs().max())
    d_aa = float((xa.double() - xa2.double()).abs().max())
    rep = {"lever": args.lever, "aa": args.aa, "n_token": host.n_token, "n_atom": host.n_atom,
           "diffusion_batch_size": D, "denoiser_calls": na,
           "one_call_max_abs_diff": {"hoisted_vs_base": call_ab,
                                     "base_vs_base": call_aa},
           "one_call_bit_exact": call_ab == 0.0,
           "rollout_s": {"base": round(ta, 4), "hoisted": round(tb, 4),
                         "base_again": round(ta2, 4)},
           "per_call_ms": {"base": round(ta / na * 1e3, 3),
                           "hoisted": round(tb / nb * 1e3, 3)},
           "speedup": round(ta / tb, 4), "aa_floor": round(ta / ta2, 4),
           "rollout_max_abs_diff_coords": {"hoisted_vs_base": d_ab,
                                           "base_vs_base": d_aa},
           "rollout_bit_exact": d_ab == 0.0}
    print(json.dumps(rep, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
