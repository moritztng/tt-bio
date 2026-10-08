"""Where does the taped/untaped difference enter, and does it enter once or everywhere?

The whole-stack difference is 0.0333 relative L2 on the pair track at every token axis measured,
and four whole-stack levers have attributed 16% of it (the softmax forward's compute kernel
config) and left the rest. Two of those four were argued from source before being run and were
wrong, so this stops arguing and walks the stack a block at a time.

Two readings per block, and they answer different questions:

* **per-block**: both arms are fed the SAME input, the untaped carry. This is the difference ONE
  block creates, with no history. A flat curve here means every block contributes alike and the
  cause is an op the blocks share; a spike at one block means it is that block's own.
* **accumulated**: each arm carries its own output forward, which is what the stack actually does.
  Comparing this against the per-block curve says whether the difference compounds or saturates.

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
    args = parser.parse_args()

    if "TT_VISIBLE_DEVICES" not in os.environ:
        raise SystemExit("set TT_VISIBLE_DEVICES to the leased card before running this")
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor(device=args.card)

    from tt_bio import autograd, bindcraft2

    rng = np.random.default_rng(args.seed)
    n = args.tokens
    msa_np = (rng.standard_normal((args.n_seq, n, args.c_m)) * args.scale).astype(np.float32)
    pair_np = (rng.standard_normal((n, n, args.c_z)) * args.scale).astype(np.float32)
    mask_np = np.ones((args.n_seq, n), dtype=np.float32)
    pair_mask_np = np.ones((n, n), dtype=np.float32)

    pool = bindcraft2.TrunkPool(args.af2_weights)
    pool.require([args.model])
    if args.model in pool.absent:
        raise SystemExit(f"{args.model} not in {args.af2_weights}: {pool.absent[args.model]}")
    pool.use(args.model)

    print(f"tokens {n}, msa {msa_np.shape}, pair {pair_np.shape}, seed {args.seed}, "
          f"exact={args.exact}", flush=True)

    with bindcraft2.refusals_unwrapped(), autograd.exact_training(args.exact), \
            bindcraft2.fast_round():
        evo = bindcraft2.EvoformerOnDevice(pool, blocks=bindcraft2.EVOFORMER_BLOCKS,
                                           recompute=False,
                                           memory=bindcraft2._Memory("fast"))
        m, z, mask, pair_mask, n_real = evo._inputs(msa_np, pair_np, mask_np, pair_mask_np)
        m_shape, z_shape = tuple(m.shape), tuple(z.shape)
        trunk = evo._trunk("")
        msa_mask = evo._msa_mask(trunk, mask)
        pair_masks = evo._pair_masks(trunk, pair_mask)

        def down(mo, zo):
            return (trunk.down(mo, m_shape)[:, :n_real].numpy(),
                    trunk.down(zo, z_shape)[:n_real, :n_real].numpy())

        # The untaped carry, which both arms are measured against and which the per-block arm is
        # re-fed at every step.
        carry_m, carry_z = trunk.up(m), trunk.up(z)
        acc_m, acc_z = trunk.up(m), trunk.up(z)

        print(f"\n{'block':>6} {'per-block msa':>14} {'per-block pair':>15} "
              f"{'accum msa':>12} {'accum pair':>12}")
        for index, block in enumerate(trunk.model.device_evoformer):
            # Untaped: the reference carry advances one block.
            with autograd.no_grad():
                next_m, next_z = block(carry_m, carry_z, msa_mask, *pair_masks)
            trunk.sync()

            # Per-block: the taped arm runs THIS block on the untaped carry, so the reading is
            # this block's own contribution with no history behind it. A fresh tape each time,
            # dropped immediately; nothing is differentiated here.
            with trunk.taped.tape():
                # `trunk.leaf` uploads a torch tensor; these are already on the card, so the
                # tape wrapper goes straight round them. Same tensor, same bytes, both arms.
                leaf_m = trunk.ag.Tensor(carry_m, requires_grad=True)
                leaf_z = trunk.ag.Tensor(carry_z, requires_grad=True)
                taped_m, taped_z = block(leaf_m, leaf_z, msa_mask, *pair_masks)
                per_block = down(taped_m.value, taped_z.value)
            trunk.ag.release_pins()

            # Accumulated: the taped arm carries its own output, which is what the stack does.
            with trunk.taped.tape():
                leaf_m = trunk.ag.Tensor(acc_m, requires_grad=True)
                leaf_z = trunk.ag.Tensor(acc_z, requires_grad=True)
                out_m, out_z = block(leaf_m, leaf_z, msa_mask, *pair_masks)
                acc_m, acc_z = out_m.value, out_z.value
                accumulated = down(acc_m, acc_z)
            trunk.ag.release_pins()

            reference = down(next_m, next_z)
            carry_m, carry_z = next_m, next_z
            print(f"{index:>6} {rel_l2(per_block[0], reference[0]):>14.6f} "
                  f"{rel_l2(per_block[1], reference[1]):>15.6f} "
                  f"{rel_l2(accumulated[0], reference[0]):>12.6f} "
                  f"{rel_l2(accumulated[1], reference[1]):>12.6f}", flush=True)


if __name__ == "__main__":
    main()
