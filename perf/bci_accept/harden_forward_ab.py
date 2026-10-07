"""One round at the `harden` boundary, on card and on host JAX, from the same ProteinStates.

Issue #17: the on-card design loop holds i_pTM through `anneal` and collapses at `harden`. Two
campaigns per arm is about 7 hours of card to answer that. This answers it in four gradient rounds.

`harden` changes exactly one thing no earlier stage does: `one_hot_weight` goes 0 -> 1
(bindcraft/trajectory.py:219), so the trunk is handed an *exact one-hot* sequence instead of
`softmax(2z/T)`. At the end of `anneal` the temperature is already 0.01, but against a logit spread
of a few hundredths that is still a soft distribution, not a one-hot. So the arms are compared at
both settings on the same logits:

    one_hot_weight = 0   the input `anneal` ends on
    one_hot_weight = 1   the input `harden` starts from

If the two arms agree at 0 and diverge at 1, the divergence is the one-hot forward and the fix is
in the trunk's handling of it. If they diverge at both, it is not specific to `harden` and the
collapse has another cause. If they agree at both, the trunk forward is not the difference at all
and the gradient route is what is left.

Card: this opens a device. Run it only when state/bci/CHIPS.md names this row as the holder, under
its RULES (TT_VISIBLE_DEVICES, the mesh descriptor, flock, nice, a timeout, AICLK recorded).
"""
import argparse
import os
import pickle

import numpy as np


def interface_ptm(predictions):
    for prediction in predictions.values():
        value = prediction.metrics.get("i_ptm", prediction.metrics.get("iptm"))
        if value is not None:
            return float(value)
    return float("nan")


def plddt(predictions):
    for prediction in predictions.values():
        value = prediction.metrics.get("plddt")
        if value is not None:
            return float(np.mean(np.asarray(value)))
    return float("nan")


def binder_gradient(gradients):
    return np.concatenate([np.asarray(value).ravel() for _, value in sorted(gradients.items())])


#: BindCraft 2 builds its design model at campaign.py:262. `predictor` yields the factory that
#: stands in for `AlphaFoldDesignModel`, so the model has to come from the factory and not from the
#: class, or the trunk routing in the subclass is never installed.
def design_model(build, af2_weights, presets, recycles, bucket):
    return build(presets=presets, data_dir=af2_weights, max_cache_size=16, num_recycle=recycles,
                 models=presets, length_bucket_size=bucket, multi_chain_binders=(),
                 target_pad_length=0)


def one_round(model, protein_states, losses, one_hot_weight, key_seed):
    """One `sequence_gradients` call at the harden stage parameters, from a fixed key.

    The key reset is not a nicety, it is what makes the comparison a comparison.
    `sequence_gradients` hands `self.key` to the jitted gradient (af2.py:401) and
    `_resolve_model_name` splits it on the way in (`self.key, model_random_key =
    jax.random.split(self.key)`, af2.py:260), so the attribute advances on every call. Two settings
    run back to back therefore see two different dropout masks, and two arms run back to back start
    from whatever key the arm before them left behind. Measured on 2026-10-07: the same
    one_hot_weight=1.0 round read i_pTM 0.7664 as the second call of a run and 0.8125 as the fifth,
    a 0.046 swing from call order alone, which is larger than the effect the A/B is looking for.
    """
    import jax
    model.key = jax.random.PRNGKey(key_seed)
    predictions, gradients, *_rest = model.sequence_gradients(
        protein_states, losses, None,
        softmax_weight=1.0, one_hot_weight=one_hot_weight, temperature=0.01, logit_scale=2.0)
    return predictions, gradients


def check_states_match(protein_states, binder_length):
    """Refuse a pickle written by a different design from the one these settings build.

    The losses and the design settings are rebuilt here from --binder-length and --target-pdb while
    the states come from whichever trajectory wrote the pickle. If those two disagree the round
    still runs and still prints numbers, so the mismatch has to be caught before the card opens.
    """
    from bindcraft.prediction import collect_shared_chains
    from bindcraft.protein import has_residue_flag, ResidueFlags

    _, chains = collect_shared_chains(protein_states)
    for name, chain in chains.items():
        designed = int(np.asarray(has_residue_flag(chain.flags, ResidueFlags.DESIGN)).sum())
        if designed and designed != binder_length:
            raise SystemExit(f"--states chain {name!r} designs {designed} residues, "
                             f"--binder-length says {binder_length}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", required=True, help="harden-entry ProteinStates pickle")
    parser.add_argument("--af2-weights", required=True)
    parser.add_argument("--card", type=int, default=None,
                        help="chip to open; omit to run the host arm alone")
    parser.add_argument("--target-pdb", required=True, help="ABSOLUTE path to the target structure")
    parser.add_argument("--target-chains", default="A")
    parser.add_argument("--hotspots", default="54,56,66,115")
    parser.add_argument("--binder-length", type=int, required=True,
                        help="binder length of the trajectory that produced --states")
    parser.add_argument("--models", nargs="+", default=["model_1_multimer_v3"])
    parser.add_argument("--recycles", type=int, default=1)
    parser.add_argument("--bucket", type=int, default=32)
    #: The card arm compares the two settings `harden` actually crosses, so this defaults to
    #: exactly 0 and 1 and the device job is unchanged. The host arm is seconds, so sweeping it
    #: card-free is how you tell a step at the one-hot boundary from a smooth trend.
    parser.add_argument("--one-hot-weights", type=float, nargs="+", default=[0.0, 1.0],
                        help="one_hot_weight settings to run (default: the 0 and 1 harden crosses)")
    parser.add_argument("--key-seed", type=int, default=0,
                        help="PRNG key every round is reset to, so settings and arms are matched")
    #: BindCraft 2 turns dropout OFF for `harden` alone (bindcraft/trajectory.py:219, :301), while
    #: AlphaFoldDesignModel defaults it ON (af2.py:209). A harden A/B left at the constructor
    #: default measures a program the stage never runs, and one the on-card trunk cannot run at
    #: all, since tt-bio's Evoformer applies no dropout. Off is the matched setting; the flag is
    #: here so the dropout-on case stays reachable without editing the script.
    parser.add_argument("--dropout", action="store_true",
                        help="run with AF2 dropout on (default off, which is what harden does)")
    args = parser.parse_args()

    from bindcraft.loss import build_losses
    from bindcraft.settings import build_design_settings
    from tt_bio import bindcraft2

    if args.card is not None:
        # qb2 is a two-chip p300 board, which ttnn reads as a CUSTOM cluster. Only tt-bio CLI entry
        # points set the mesh descriptor; a script that reaches the device path directly, as this
        # one does, gets TT_FATAL about the fabric mesh graph descriptor (tt_bio/tenstorrent.py,
        # _open_and_init_device). TT_VISIBLE_DEVICES is the wrapper's job.
        from tt_bio.main import ensure_p300_mesh_descriptor
        ensure_p300_mesh_descriptor(device=args.card)

    # The target is named by ABSOLUTE path, never by the `"target": "hPDL1"` preset. That preset
    # carries `"target_path": "structures/hPDL1.pdb"`, which is relative, and from any cwd but
    # BindCraft 2's own settings tree it resolves to nothing without a word. The campaign then runs
    # with no target chain at all: it folds the binder alone, reports 64 tokens for a 40-aa binder
    # and scores iptm exactly 0.000 at every stage. An A/B set up that way compares two arms on a
    # fold with no interface, which is the one thing this script exists to measure.
    if not os.path.isabs(args.target_pdb) or not os.path.exists(args.target_pdb):
        raise SystemExit(f"--target-pdb must be an existing absolute path, got {args.target_pdb!r}")
    settings = {
        "targets": [{"name": "hPDL1", "target_path": args.target_pdb,
                     "chains": args.target_chains, "hotspots": args.hotspots}],
        "binder_lengths": [args.binder_length, args.binder_length],
        "max_trajectories": 1, "number_of_final_designs": 1, "trajectory_only": True,
        "design_models": 1, "campaign_seed": 42,
    }
    design_settings = build_design_settings(settings)
    losses = build_losses(design_settings.settings, seed=design_settings.seed)
    with open(args.states, "rb") as handle:
        protein_states = pickle.load(handle)
    check_states_match(protein_states, args.binder_length)

    presets = tuple(args.models)
    arms = {"host-JAX": dict(trunk="jax", checkpoints=args.af2_weights)}
    if args.card is not None:
        arms["on-card"] = dict(card=args.card, checkpoints=args.af2_weights)

    results = {}
    for arm_name, predictor_arguments in arms.items():
        with bindcraft2.predictor(**predictor_arguments) as build:
            model = design_model(build, args.af2_weights, presets, args.recycles, args.bucket)
            model.dropout = args.dropout
            for one_hot_weight in args.one_hot_weights:
                predictions, gradients = one_round(model, protein_states, losses, one_hot_weight,
                                                   args.key_seed)
                results[(arm_name, one_hot_weight)] = (
                    interface_ptm(predictions), plddt(predictions), binder_gradient(gradients))

    print(f"{'one_hot_weight':>15}{'arm':>12}{'i_pTM':>9}{'pLDDT':>9}")
    for (arm_name, one_hot_weight), (iptm, mean_plddt, _) in sorted(
            results.items(), key=lambda item: (item[0][1], item[0][0])):
        print(f"{one_hot_weight:>15.1f}{arm_name:>12}{iptm:>9.4f}{mean_plddt:>9.4f}")

    if len(arms) == 2:
        print("\nbinder sequence gradient, on-card against host JAX, on the same logits:")
        for one_hot_weight in args.one_hot_weights:
            host = results[("host-JAX", one_hot_weight)][2]
            card = results[("on-card", one_hot_weight)][2]
            cosine = float(np.dot(host, card) / (np.linalg.norm(host) * np.linalg.norm(card) + 1e-30))
            relative = float(np.linalg.norm(card - host) / (np.linalg.norm(host) + 1e-30))
            print(f"  one_hot_weight={one_hot_weight:.1f}  cosine={cosine:.6f}  "
                  f"relative L2 difference={relative:.6f}")


if __name__ == "__main__":
    main()
