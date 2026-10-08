"""Where does the taped/untaped difference enter, and does it enter once or everywhere?

The whole-stack difference is 0.0333 relative L2 on the pair track at every token axis measured,
and four whole-stack levers have attributed 16% of it (the softmax forward's compute kernel
config) and left the rest. Two of those four were argued from source before being run and were
wrong, so this stops arguing and walks the stack a block at a time.

Each row is one Evoformer block run twice on the same input, untaped against taped. Both arms
are fed the untaped carry, so a row is what ONE block contributes with no history behind it. A
flat curve means every block contributes alike and the cause is an op the blocks share; a spike
at one block means it is that block's own. The 48-block number already measured, 0.0333, is the
accumulated endpoint this curve has to add up to.

No checkpointing, no recompute, no memory mode: `recompute=False` on both arms, so the only
difference is the tape. That is the base defect isolated in the trunk VJP script, now resolved
per block.

  TT_VISIBLE_DEVICES=<card> python3 perf/bci_accept/block_taped_diff.py --card <card> \
      --af2-weights ~/bcx_e2e/af2_params --tokens 64
"""
from __future__ import annotations

import argparse
import os

import numpy as np

#: The op classes `AF2PairBlock._update` dispatches, which `set_skip` drops.
OP_CLASSES = ("msa_row_attn", "msa_col_attn", "msa_transition", "tri_mul_out",
              "tri_mul_in", "tri_att_start", "tri_att_end", "pair_transition")


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(b.ravel()))
    if denominator == 0.0:
        return float("nan")
    return float(np.linalg.norm((a - b).ravel()) / denominator)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--card", type=int, required=True)
    parser.add_argument("--af2-weights", required=True)
    parser.add_argument("--model", default="model_1_multimer_v3")
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--n-seq", type=int, default=1)
    parser.add_argument("--c-m", type=int, default=256)
    parser.add_argument("--c-z", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scale", type=float, default=0.1)
    parser.add_argument("--exact", action="store_true")
    parser.add_argument("--skip-sweep", action="store_true",
                        help="attribute the settled-carry difference to an op class. For the "
                             "control and then each op class in turn, drop that class from both "
                             "device stacks (`AF2DeviceModel.set_skip`, the shipped cost-census "
                             "lever) and remeasure. The arm that drops is the one carrying the "
                             "difference. Skipping also removes work, so a class is only "
                             "implicated if its leg falls much further than the others.")
    parser.add_argument("--also-skip", action="append", default=[],
                        help="an extra sweep leg, comma-separated op classes. Dropping msa_row_attn "
                             "zeroes the MSA track, but that op is also the ONLY MSA op that reads "
                             "the pair track, so the zero may be imported divergence rather than a "
                             "defect of its own. Dropping every pair op instead decides it: if the "
                             "MSA difference collapses with msa_row_attn still running, it was "
                             "coming through the pair bias.")
    parser.add_argument("--warmup", type=int, default=2,
                        help="blocks the carry is advanced untaped before any row is measured. "
                             "The raw random input is off-distribution and the first block "
                             "amplifies it; a settled carry is what the per-block reading needs.")
    parser.add_argument("--rows", type=int, default=6,
                        help="blocks measured per configuration")
    parser.add_argument("--order", default="forward", choices=("forward", "reverse"),
                        help="block order. The forward run put the whole 48-block difference in "
                             "the FIRST row and ~0.0014 in every other, which is either the "
                             "block or the position it sits in. Reversing separates them: if "
                             "the spike stays on the first ROW it is the position and the raw "
                             "input it sees, if it follows block 0 to the last row it is that "
                             "block's own.")
    args = parser.parse_args()

    if "TT_VISIBLE_DEVICES" not in os.environ:
        raise SystemExit("set TT_VISIBLE_DEVICES to the leased card before running this")
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor(device=args.card)

    from tt_bio import autograd, bindcraft2

    rng = np.random.default_rng(args.seed)
    n = args.tokens
    msa = (rng.standard_normal((args.n_seq, n, args.c_m)) * args.scale).astype(np.float32)
    pair = (rng.standard_normal((n, n, args.c_z)) * args.scale).astype(np.float32)
    mask = np.ones((args.n_seq, n), dtype=np.float32)
    pair_mask = np.ones((n, n), dtype=np.float32)

    pool = bindcraft2.TrunkPool(args.af2_weights)
    pool.require([args.model])
    if args.model in pool.absent:
        raise SystemExit(f"{args.model} not in {args.af2_weights}: {pool.absent[args.model]}")
    pool.use(args.model)

    print(f"tokens {n}, msa {msa.shape}, pair {pair.shape}, seed {args.seed}, "
          f"exact={args.exact}", flush=True)
    print("each row is ONE block run twice on the SAME input, untaped against taped, "
          "recompute=False both ways", flush=True)

    with bindcraft2.refusals_unwrapped(), autograd.exact_training(args.exact), \
            bindcraft2.fast_round():
        evo = bindcraft2.EvoformerOnDevice(pool, blocks=bindcraft2.EVOFORMER_BLOCKS,
                                           recompute=False,
                                           memory=bindcraft2._Memory("fast"))
        trunk = evo._trunk("")
        blocks = list(trunk.model.device_evoformer)
        order = list(range(len(blocks)))
        if args.order == "reverse":
            order.reverse()

        if args.skip_sweep:
            configs = ([()] + [(name,) for name in OP_CLASSES]
                       + [tuple(group.split(",")) for group in args.also_skip])
            warm_msa, warm_pair = msa, pair
            try:
                for index in range(args.warmup):
                    trunk.model.device_evoformer = [blocks[index]]
                    warm_msa, warm_pair = evo._primal("", warm_msa, warm_pair, mask, pair_mask)
                print(f"carry settled through {args.warmup} untaped blocks, pair max "
                      f"{float(np.max(np.abs(warm_pair))):.6g}", flush=True)

                print(f"\n{'skipped':>58} {'mean msa':>10} {'mean pair':>10} {'rows':>5}",
                      flush=True)
                for names in configs:
                    # `set_skip` walks `device_evoformer` to set each block's `skip`, so the
                    # list has to be whole when it is called -- the row loop below leaves it
                    # sliced to one block.
                    trunk.model.device_evoformer = blocks
                    trunk.model.set_skip(names)
                    carry_msa, carry_pair = warm_msa, warm_pair
                    msa_readings, pair_readings = [], []
                    for index in range(args.warmup, args.warmup + args.rows):
                        trunk.model.device_evoformer = [blocks[index]]
                        untaped = evo._primal("", carry_msa, carry_pair, mask, pair_mask)
                        taped_msa, taped_pair, _t = evo._taped("", carry_msa, carry_pair,
                                                               mask, pair_mask)
                        msa_readings.append(rel_l2(taped_msa, untaped[0]))
                        pair_readings.append(rel_l2(taped_pair, untaped[1]))
                        carry_msa, carry_pair = untaped
                    label = "+".join(names) if names else "(control)"
                    print(f"{label:>58} {float(np.mean(msa_readings)):>10.6f} "
                          f"{float(np.mean(pair_readings)):>10.6f} {len(pair_readings):>5}",
                          flush=True)
            finally:
                trunk.model.device_evoformer = blocks
                trunk.model.set_skip(())
            return

        carry_msa, carry_pair = msa, pair
        print(f"\n{'row':>4}{'block':>6} {'msa rel L2':>12} {'pair rel L2':>12} "
              f"{'pair max|d|':>12} {'ref max|.|':>12}", flush=True)
        try:
            for row, index in enumerate(order):
                block = blocks[index]
                # One block at a time, through the SAME entry points a design round uses. An
                # earlier version of this script called the block object directly and segfaulted
                # in the first layer_norm: `_primal` and `_taped` do setup -- the memory mode, the
                # trunk's arm, the refusal and card contexts -- that a bare call skips. Slicing
                # the block list is the smallest way to keep all of it and vary only the depth.
                trunk.model.device_evoformer = [block]
                untaped = evo._primal("", carry_msa, carry_pair, mask, pair_mask)
                taped_msa, taped_pair, _token = evo._taped("", carry_msa, carry_pair,
                                                           mask, pair_mask)
                reference_pair = untaped[1]
                print(f"{row:>4}{index:>6} {rel_l2(taped_msa, untaped[0]):>12.6f} "
                      f"{rel_l2(taped_pair, reference_pair):>12.6f} "
                      f"{float(np.max(np.abs(taped_pair - reference_pair))):>12.6g} "
                      f"{float(np.max(np.abs(reference_pair))):>12.6g}", flush=True)
                # The untaped arm is the carry, so every row reads one block's own contribution
                # with no history behind it. The carry round-trips through bf16 on each upload,
                # equally for both arms, which is why this is a per-block reading and not a
                # second estimate of the 48-block number.
                carry_msa, carry_pair = untaped
        finally:
            trunk.model.device_evoformer = blocks


if __name__ == "__main__":
    main()
