"""Does the on-card Evoformer's backward depend on how its activations were kept?

It must not. `fast`, `lean` and `offload` differ only in what the tape holds and what the backward
recomputes (`_Trunk.arm` sets `block.step_runner`, nothing else), and `recompute=False` holds the
whole stack and recomputes nothing at all. All four are the same mathematics. A difference between
any two of them is a defect in that machinery, provable on ONE card with no host reference, so it
cannot be read as precision between two vendors' float.

The A/B this comes from measured exactly that at 192 tokens through the whole BindCraft 2 design
round: `fast` and `lean` agreed on i_pTM and pLDDT to every digit printed and disagreed on the
binder gradient by 7.2% in L2 norm and 10.6% in its largest element, reproducibly. This script cuts
the design loop, the losses and host JAX out of that comparison and asks the trunk directly.

It uses RANDOM activations on purpose. The claim under test is that two configurations compute the
same function, and that is a claim about every input, not about a trajectory's. Random inputs also
free the token axis: `recompute=False` cannot run at 192 tokens on a p150a (measured: it asks for
9.4 MB against 7.8 MB free, holding 34.218 of 34.226 GB) but fits easily at 128, which is what
makes the no-recompute referee reachable at all.

  TT_VISIBLE_DEVICES=<card> python3 perf/bci_accept/trunk_vjp_modes.py --card <card> \
      --af2-weights ~/bcx_e2e/af2_params --tokens 128
"""
from __future__ import annotations

import argparse
import os

import numpy as np


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    """||a-b|| / ||b||, the same statistic the card A/B quotes."""
    denominator = float(np.linalg.norm(b.ravel()))
    if denominator == 0.0:
        return float("nan")
    return float(np.linalg.norm((a - b).ravel()) / denominator)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel(), b.ravel()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0.0:
        return float("nan")
    return float(np.dot(a, b) / denominator)


def report(name: str, got: np.ndarray, ref: np.ndarray) -> dict:
    """One row. `max abs` is reported beside the norms because a direction error that nearly
    cancels in the norm is precisely the shape this investigation has been chasing: at 192 tokens
    the binder gradient's norm agreed to 1.8% while its elements disagreed by 26%."""
    return {"what": name, "rel_l2": rel_l2(got, ref), "cosine": cosine(got, ref),
            "max_abs_diff": float(np.max(np.abs(got - ref))) if got.size else float("nan"),
            "ref_max_abs": float(np.max(np.abs(ref))) if ref.size else float("nan"),
            "bit_identical": bool(got.shape == ref.shape and np.array_equal(got, ref))}


def run_one(pool, memory_mode: str, recompute: bool, blocks: int,
            msa, pair, mask, pair_mask, g_msa, g_pair):
    """One (memory mode, recompute) configuration: taped forward, then backward on a FIXED
    cotangent. A fresh `EvoformerOnDevice` per configuration, because `_Memory` caches the mode it
    picked per token axis and a reused one would quietly serve the first configuration's choice to
    the second."""
    from tt_bio import bindcraft2

    evo = bindcraft2.EvoformerOnDevice(pool, blocks=blocks, recompute=recompute,
                                       memory=bindcraft2._Memory(memory_mode))
    msa_out, pair_out, token = evo._taped("", msa, pair, mask, pair_mask)
    grad_msa, grad_pair = evo._backward("", token, g_msa, g_pair)
    live = evo.live_tapes("")
    if live:
        # The tape bank is keyed by token and swept per slot. A configuration that leaves one
        # behind has not run the program the next configuration will run.
        raise SystemExit(f"{memory_mode}/recompute={recompute} left {live} live tape(s)")
    return (np.asarray(msa_out), np.asarray(pair_out),
            np.asarray(grad_msa), np.asarray(grad_pair), dict(evo.calls))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--card", type=int, required=True)
    parser.add_argument("--af2-weights", required=True)
    parser.add_argument("--tokens", type=int, default=128,
                        help="token axis, a multiple of 32 (default 128: small enough that the "
                             "recompute=False referee fits on a p150a)")
    parser.add_argument("--n-seq", type=int, default=1,
                        help="MSA rows. BindCraft 2 designs from a single row.")
    parser.add_argument("--c-m", type=int, default=256)
    parser.add_argument("--c-z", type=int, default=128)
    parser.add_argument("--model", default="model_1_multimer_v3",
                        help="AF2 checkpoint, the same preset the card A/B runs")
    parser.add_argument("--blocks", type=int, default=None)
    parser.add_argument("--exact", action="store_true",
                        help="run under autograd.exact_training(True). The input-scale sweep "
                             "cannot separate a structural defect from accumulated bf16: "
                             "floating-point relative error is scale-invariant too, so a flat "
                             "sweep is consistent with both. This lever can. If the "
                             "configurations converge under exact training the disagreement is "
                             "precision; if they still disagree it is structural.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scale", type=float, default=0.1,
                        help="standard deviation of the random activations")
    parser.add_argument("--configs", nargs="+",
                        default=["none:False", "fast:True", "lean:True"],
                        help="<memory mode>:<recompute> entries. The FIRST is the reference every "
                             "other is read against; 'none:False' is the natural one, since it "
                             "recomputes nothing and so cannot be wrong about a recompute.")
    args = parser.parse_args()

    if args.tokens % 32:
        raise SystemExit(f"--tokens must be a multiple of 32, got {args.tokens}")
    if "TT_VISIBLE_DEVICES" not in os.environ:
        raise SystemExit("set TT_VISIBLE_DEVICES to the leased card before running this")

    # qb1's p150a is single-chip and qb2's p300 is not; either way a script that reaches the
    # device path directly has to set the mesh descriptor itself, because only tt-bio's CLI entry
    # points do it (tt_bio/tenstorrent.py::_open_and_init_device).
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor(device=args.card)

    from tt_bio import autograd, bindcraft2

    blocks = args.blocks if args.blocks is not None else bindcraft2.EVOFORMER_BLOCKS
    rng = np.random.default_rng(args.seed)
    n = args.tokens
    msa = (rng.standard_normal((args.n_seq, n, args.c_m)) * args.scale).astype(np.float32)
    pair = (rng.standard_normal((n, n, args.c_z)) * args.scale).astype(np.float32)
    mask = np.ones((args.n_seq, n), dtype=np.float32)
    pair_mask = np.ones((n, n), dtype=np.float32)
    # One fixed cotangent for every configuration. The comparison is of the VJP at one point in
    # one direction; a cotangent drawn per configuration would compare two different quantities.
    g_msa = (rng.standard_normal(msa.shape) * args.scale).astype(np.float32)
    g_pair = (rng.standard_normal(pair.shape) * args.scale).astype(np.float32)

    print(f"tokens {n}, msa {msa.shape}, pair {pair.shape}, blocks {blocks}, seed {args.seed}",
          flush=True)
    print("every configuration below computes the same function; they differ only in what the "
          "tape holds and what the backward recomputes", flush=True)

    # Match what a BindCraft 2 design round runs: `predictor`'s `fast` defaults to `not exact`,
    # and `fast_round()` is what turns the fp32 softmax backward off for it. Measured null at 192
    # tokens either way, but the point of this script is to vary ONE thing.
    pool = bindcraft2.TrunkPool(args.af2_weights)
    # The pool resolves names lazily and `trunk_for` only falls back to `names[0]` once something
    # has asked for one, so an unasked pool is empty rather than defaulted.
    pool.require([args.model])
    if args.model in pool.absent:
        raise SystemExit(f"{args.model} not in {args.af2_weights}: {pool.absent[args.model]}")
    pool.use(args.model)
    results = {}
    print(f"exact_training={args.exact}", flush=True)
    with bindcraft2.refusals_unwrapped(), autograd.exact_training(args.exact), \
            bindcraft2.fast_round():
        for config in args.configs:
            mode, _, recompute_text = config.partition(":")
            recompute = recompute_text.lower() in ("1", "true", "yes")
            # `none` is the label for "no checkpointing at all"; the memory mode is then unused
            # by `evoformer()` and only arms `step_runner`, so it is pinned to fast to keep the
            # one varying thing varying.
            memory_mode = "fast" if mode == "none" else mode
            print(f"\n=== {config}  (memory={memory_mode}, recompute={recompute}) ===",
                  flush=True)
            results[config] = run_one(pool, memory_mode, recompute, blocks,
                                      msa, pair, mask, pair_mask, g_msa, g_pair)
            print(f"  seams {results[config][4]}", flush=True)

    reference_name = args.configs[0]
    msa_ref, pair_ref, g_msa_ref, g_pair_ref, _ = results[reference_name]
    print(f"\nread against {reference_name}:\n")
    header = f"{'config':>14} {'quantity':>14} {'rel L2':>12} {'cosine':>10} " \
             f"{'max|diff|':>12} {'ref max|.|':>12}  bit-identical"
    print(header)
    rows = []
    for config in args.configs[1:]:
        msa_out, pair_out, g_msa_out, g_pair_out, _ = results[config]
        for label, got, ref in (("forward msa", msa_out, msa_ref),
                                ("forward pair", pair_out, pair_ref),
                                ("grad msa", g_msa_out, g_msa_ref),
                                ("grad pair", g_pair_out, g_pair_ref)):
            row = report(label, got, ref)
            rows.append((config, row))
            print(f"{config:>14} {label:>14} {row['rel_l2']:>12.6f} {row['cosine']:>10.6f} "
                  f"{row['max_abs_diff']:>12.6g} {row['ref_max_abs']:>12.6g}  "
                  f"{row['bit_identical']}")

    # The verdict is a line, so a log can be grepped for it rather than read.
    worst = max((row["rel_l2"] for _config, row in rows if np.isfinite(row["rel_l2"])),
                default=float("nan"))
    backward_worst = max((row["rel_l2"] for _config, row in rows
                          if row["what"].startswith("grad") and np.isfinite(row["rel_l2"])),
                         default=float("nan"))
    forward_worst = max((row["rel_l2"] for _config, row in rows
                         if row["what"].startswith("forward") and np.isfinite(row["rel_l2"])),
                        default=float("nan"))
    print(f"\nVERDICT worst relative L2 {worst:.6g} "
          f"(forward {forward_worst:.6g}, backward {backward_worst:.6g})")
    print("VERDICT " + ("CONSISTENT: every configuration agrees"
                        if worst == 0.0 else
                        "INCONSISTENT: configurations that must agree do not"))


if __name__ == "__main__":
    main()
