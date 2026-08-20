#!/usr/bin/env python3
"""Where the RF3 confidence head's time goes, and what the global layer norm costs.

Pass 3's two triangle-attention levers cut the trunk 2.6x, and the confidence head went
from under 5% of a 512 aa fold to 14.7% of it without anything about the head changing
(`perf-mechanism-label-expires-when-lever-removes-its-traffic`). It has never been
decomposed. Its own 4-block Pairformer costs 4 x 48 ms = 0.19 s at 512 aa against the
5.99 s the phase reads, so the answer is in `embed` and `heads`, not in the stack.

Both arms of `global_layer_norm` run in one process off one checkpoint load, the flag
flipped per call rather than per process, and arm A twice so the A/A floor is measured
rather than assumed.

x_pred comes from a 3-step rollout, not the shipped 50. The head's cost is set by the
SHAPE of the binned distance one-hot, which is [1, I, I, 40] either way; only the bin
contents differ. Timing only -- do not read accuracy off this harness.
"""
from __future__ import annotations

import argparse
import contextlib
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

from perf.rf3.trunk_decompose import Acc  # noqa: E402


def instrument(head, acc: Acc):
    acc.wrap(head, "embed", "conf.embed")
    acc.wrap(head, "heads", "conf.heads")
    acc.wrap(head, "pairformer", "conf.pairformer")
    return head


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aa", type=int, default=512)
    ap.add_argument("--ckpt", default="/home/ttuser/rf3_perf_work/rf3_latest.ckpt")
    ap.add_argument("--n_recycles", type=int, default=2)
    ap.add_argument("--num_steps", type=int, default=3)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from tt_bio.rf3.featurize import featurize
    fo = featurize(str(REPO / f"perf/rf3/inputs/rf3_{args.aa}.json"),
                   n_recycles=max(args.n_recycles, 2), diffusion_batch_size=1,
                   seed=args.seed)[0]
    f, rep_atom_idxs = fo["feats"], fo["ground_truth"]["rep_atom_idxs"]

    import ttnn
    from tt_bio.rf3 import model as rf3_model
    from tt_bio.rf3 import confidence_head as ch
    from tt_bio.rf3.host import HostInputs
    from tt_bio.rf3.host import distance_onehot
    from tt_bio.rf3.sampler import Draws
    from tt_bio.tenstorrent import get_device
    from perf.rf3.tt_rf3_bench import net_config

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
        num_timesteps=args.num_steps, with_confidence=True)

    host = HostInputs.build(f, device)
    s_inputs, s, z = tt.trunk(host, args.n_recycles)
    x_pred, _ = tt.sampler.sample(
        lambda x, t: tt.diffusion_module(host, x, t, s_inputs, s, z),
        torch.zeros(1, host.n_atom, 3), 1, draws=Draws())
    dist_tt = distance_onehot(x_pred[0:1], rep_atom_idxs, device)

    head = tt.confidence_head
    acc = Acc(device)
    instrument(head, acc)

    def one(fold: bool):
        ch._GLN_ROW_FOLD = fold
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        out = head(s_inputs, s, z, dist_tt)
        ttnn.synchronize_device(device)
        dt = time.perf_counter() - t0
        for v in out.values():
            ttnn.deallocate(v)
        return dt

    one(False)                              # warm
    legs: dict[str, list[float]] = {"A": [], "B": [], "A2": []}
    split: dict[str, dict] = {}
    for r in range(args.reps):
        for tag, fold in (("A", False), ("B", True), ("A2", False)):
            acc.reset()
            acc.on = True
            legs[tag].append(one(fold))
            acc.on = False
            if r == args.reps - 1:
                split[tag] = {k: round(v, 5) for k, v in acc.t.items()}

    def med(xs):
        xs = sorted(xs)
        return xs[len(xs) // 2]

    a, b, a2 = med(legs["A"]), med(legs["B"]), med(legs["A2"])
    rep = {"aa": args.aa, "n_token": host.n_token, "reps": args.reps,
           "head_s": {"one_row_flatten": round(a, 5), "row_fold": round(b, 5),
                      "one_row_flatten_again": round(a2, 5)},
           "speedup": round(a / b, 4), "aa_floor": round(a / a2, 4),
           "legs_all_s": {k: [round(x, 5) for x in v] for k, v in legs.items()},
           "split_per_call_s": split}
    print(json.dumps(rep, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
