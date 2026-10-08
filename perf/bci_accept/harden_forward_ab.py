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


def gradient_by_key(gradients):
    """Per-key norm and shape. A whole-dict norm of 0 cannot say whether every chain is zero or
    the dict simply does not hold the chain being designed, and those need different fixes."""
    return {key: (np.asarray(value).shape, float(np.linalg.norm(np.asarray(value), ord=None)))
            for key, value in sorted(gradients.items())}


#: BindCraft 2 builds its design model at campaign.py:262. `predictor` yields the factory that
#: stands in for `AlphaFoldDesignModel`, so the model has to come from the factory and not from the
#: class, or the trunk routing in the subclass is never installed.
def design_model(build, af2_weights, presets, recycles, bucket):
    return build(presets=presets, data_dir=af2_weights, max_cache_size=16, num_recycle=recycles,
                 models=presets, length_bucket_size=bucket, multi_chain_binders=(),
                 target_pad_length=0)


def one_round(model, protein_states, losses, one_hot_weight, key_seed, temperature=0.01):
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
    predictions, gradients, *rest = model.sequence_gradients(
        protein_states, losses, None,
        softmax_weight=1.0, one_hot_weight=one_hot_weight, temperature=temperature,
        logit_scale=2.0)
    # A design loss that never attaches to the state gives a finite forward and an exactly zero
    # gradient, which is the signature seen here. The scalar loss is the cheapest way to tell that
    # from a real zero, so it is printed rather than dropped into `*_rest`.
    loss_value = float(rest[0]) if rest else float("nan")
    print(f"  [loss] design_loss={loss_value:.6g}  losses={sorted(losses)}  "
          f"states={sorted(protein_states)}  "
          f"required={sorted({r for entry in losses.values() for r in entry.required_states})}",
          flush=True)
    return predictions, gradients


def captured_losses_for(states_path):
    """The ACTIVE loss names recorded beside the states, or None if the capture predates them."""
    import json
    sidecar = states_path + ".losses.json"
    if not os.path.exists(sidecar):
        return None
    with open(sidecar) as handle:
        return set(json.load(handle))


def active_losses(settings, available_states, captured=None):
    """The loss set a gradient round at `harden` should actually use, or raise saying why not.

    Three distinct traps live here and each one returns a plausible-looking forward with a
    silently wrong gradient, so each is a refusal rather than a fallback.

    1. `build_design_settings` stores its settings dict RAW (settings.py:655), so a hand-built
       one never gains the `weights_<loss>` keys `build_losses` reads (loss.py:105) and EVERY
       loss is dropped. `load_settings` is the merge BindCraft 2's own entry point uses
       (`read_settings`, settings.py:714). Without it the design loss is the constant 0 and the
       sequence gradient is exactly zero.
    2. `weighted_design_loss` (loss.py:115) seeds its sum with a constant `0.0` and skips a loss
       whose `required_states` are absent, so an empty or fully unsatisfied set does not raise.
    3. The jitted `sequence_design_loss` (af2.py:369-371) does NOT skip: it indexes
       `prediction_arrays[state_name]` and raises KeyError inside a traced function. A
       default-configured loss keeps `required_states=('complex',)` (loss.py:57) while these
       states are keyed by target name, which is what the design loop hands `update_sequence`.

    `captured` is the active set recorded at capture time, which is the captured article rather
    than a reconstruction: trajectory.py:131 passes `active_losses` beside the
    `active_protein_states` of :145.
    """
    from bindcraft.loss import build_losses
    from bindcraft.settings import build_design_settings, load_settings

    design_settings = build_design_settings(load_settings(settings))
    losses = build_losses(design_settings.settings, seed=design_settings.seed)
    if not losses:
        raise SystemExit("build_losses returned no losses: the gradient would be identically "
                         "zero and the A/B would compare two arms on a constant")
    if captured is not None:
        missing = captured - set(losses)
        if missing:
            raise SystemExit(f"the capture recorded active losses this settings build does not "
                             f"produce: {sorted(missing)}")
        losses = {name: entry for name, entry in losses.items() if name in captured}
        print(f"losses from the capture's own active set: {sorted(losses)}", flush=True)
    dropped = {name: sorted(entry.required_states - available_states)
               for name, entry in losses.items()
               if not entry.required_states <= available_states}
    if dropped:
        print(f"dropping {len(dropped)} loss(es) whose required states are absent "
              f"(states present: {sorted(available_states)}): {dropped}", flush=True)
        losses = {name: entry for name, entry in losses.items() if name not in dropped}
    if not losses:
        raise SystemExit("every loss was dropped for missing states; the gradient would be zero")
    print(f"losses in play, identical for both arms: {sorted(losses)}", flush=True)
    return losses


def protein_states_preview(path):
    """The state names in the pickle, read before any model is built so a refusal costs nothing."""
    with open(path, "rb") as handle:
        return pickle.load(handle)


def cast_sequences(protein_states, dtype):
    """Re-type every chain's sequence logits, leaving flags and coordinates alone.

    BindCraft 2 stores `Protein.sequence` as float16. The design loss is differentiated with
    respect to exactly that array (`af2.py:352` takes `sequences`, `:41` gates it on the DESIGN
    flag), so the returned gradient is float16 too and anything below float16's smallest
    subnormal, about 6e-8, lands as exactly zero. That is indistinguishable from "no gradient"
    until the same call is made in float32.
    """
    return {state: {chain: protein.replace(sequence=np.asarray(protein.sequence, dtype=dtype))
                    for chain, protein in complex_.items()}
            for state, complex_ in protein_states.items()}


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
    #: `harden` runs at 0.01 and `logit_scale` is 2.0, so the trunk sees softmax(200*z). The host
    #: arm's binder gradient came back exactly 0 at that setting, which is what a saturated softmax
    #: does: its Jacobian is diag(p) - p p^T, and that underflows to zero once p is one-hot in
    #: float32. Raising the temperature is the control that tells a real saturation from a harness
    #: that is extracting the wrong array: at 1.0 the same call must produce a nonzero gradient.
    parser.add_argument("--sequence-dtype", default="keep",
                        choices=("keep", "float32", "float64"),
                        help="re-type the sequence logits before differentiating (default keep, "
                             "which is BindCraft 2's own float16)")
    parser.add_argument("--temperature", type=float, default=0.01,
                        help="softmax temperature (default 0.01, which is harden's own)")
    parser.add_argument("--key-seed", type=int, default=0,
                        help="PRNG key every round is reset to, so settings and arms are matched")
    #: BindCraft 2 turns dropout OFF for `harden` alone (bindcraft/trajectory.py:219, :301), while
    #: AlphaFoldDesignModel defaults it ON (af2.py:209). A harden A/B left at the constructor
    #: default measures a program the stage never runs, and one the on-card trunk cannot run at
    #: all, since tt-bio's Evoformer applies no dropout. Off is the matched setting; the flag is
    #: here so the dropout-on case stays reachable without editing the script.
    parser.add_argument("--dropout", action="store_true",
                        help="run with AF2 dropout on (default off, which is what harden does)")
    #: `predictor(exact=True)` runs softmax and layer norm on the host in float64 INSIDE the
    #: tape, which is the closest this port gets to AF2's own gradient. It costs 24.87x on a
    #: `sequence_gradients` call at n=192, so it is not a shipping setting -- it is the control
    #: that says whether the binder gradient's 26% elementwise error against host JAX is bf16
    #: precision inside the tape or a program difference. If the error survives `exact`, no
    #: amount of precision work in the tape closes it.
    parser.add_argument("--exact", action="store_true",
                        help="run the on-card arm with predictor(exact=True): host float64 "
                             "softmax and layer norm inside the tape (about 25x slower)")
    parser.add_argument("--memory", default=None,
                        choices=("auto", "fast", "lean"),
                        help="device memory mode for the on-card arm (default: predictor's own "
                             "'auto'). 'fast' and 'lean' recompute and offload differently and "
                             "must return the same gradient; a difference between them is a bug "
                             "in that machinery, provable without leaving the card.")
    parser.add_argument("--no-recompute", action="store_true",
                        help="run the on-card arm with gradient checkpointing OFF "
                             "(predictor(recompute=False)). The taped forward checkpoints by "
                             "default and the backward recomputes each block; if that recompute "
                             "does not reproduce the taped forward bit for bit, the VJP is taken "
                             "at a different point than the forward was. Costs memory: the whole "
                             "stack's activations stay resident.")
    args = parser.parse_args()

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
    losses = active_losses(settings, set(protein_states_preview(args.states)),
                           captured_losses_for(args.states))
    with open(args.states, "rb") as handle:
        protein_states = pickle.load(handle)
    check_states_match(protein_states, args.binder_length)
    if args.sequence_dtype != "keep":
        protein_states = cast_sequences(protein_states, args.sequence_dtype)
        print(f"sequence logits cast to {args.sequence_dtype}", flush=True)

    presets = tuple(args.models)
    # The device blocks apply no dropout, so `evoformer_on_device` refuses the swap by default
    # rather than fold a different program than the host arm. `use_dropout` is traced, so that
    # guard cannot see that THIS A/B has turned dropout off on both arms -- which is `harden`'s
    # own setting (bindcraft/trajectory.py:219) and the one #17 is about. With it off there is no
    # dropout for the swap to lose and the refusal is a false positive, so waive it. With
    # --dropout the card really would drop it silently, and then refusing is the correct answer.
    dropout_policy = "refuse" if args.dropout else "ignore"
    arms = {"host-JAX": dict(trunk="jax", checkpoints=args.af2_weights)}
    if args.card is not None:
        arms["on-card"] = dict(card=args.card, checkpoints=args.af2_weights,
                               dropout=dropout_policy,
                               recompute=not args.no_recompute)
        if args.memory is not None:
            arms["on-card"]["memory"] = args.memory
        if args.exact:
            arms["on-card"]["exact"] = True

    results = {}
    for arm_name, predictor_arguments in arms.items():
        # Print the arm's actual predictor arguments. Every lever A/B in this row is graded on a
        # difference of a few parts in a thousand, so "which program did this arm run" has to come
        # out of the run rather than out of the launcher.
        print(f"  [{arm_name}] predictor {dict(sorted(predictor_arguments.items()))}", flush=True)
        with bindcraft2.predictor(**predictor_arguments) as build:
            model = design_model(build, args.af2_weights, presets, args.recycles, args.bucket)
            model.dropout = args.dropout
            for one_hot_weight in args.one_hot_weights:
                predictions, gradients = one_round(model, protein_states, losses, one_hot_weight,
                                                   args.key_seed, args.temperature)
                print(f"  [{arm_name} one_hot={one_hot_weight}] gradient keys: "
                      f"{gradient_by_key(gradients)}", flush=True)
                results[(arm_name, one_hot_weight)] = (
                    interface_ptm(predictions), plddt(predictions), binder_gradient(gradients))
                # Say whether the fp32 softmax backward actually fired, rather than inferring it
                # from the env var having been set. `bindcraft2.fast_round()` turns
                # SOFTMAX_BW_FP32 OFF for a BindCraft 2 predictor (bindcraft2.py:2152) even though
                # tt_bio.autograd defaults it ON, and `fast` itself defaults to `not exact`. A
                # lever A/B whose only evidence is the variable it exported cannot tell a lever
                # that did nothing from a lever that changed nothing.
                if arm_name != "host-JAX":
                    from tt_bio import autograd as _autograd
                    print(f"  [{arm_name} one_hot={one_hot_weight}] SOFTMAX_BW_FP32="
                          f"{_autograd.SOFTMAX_BW_FP32} stats={_autograd.SOFTMAX_BW_FP32_STATS}",
                          flush=True)

    print(f"{'one_hot_weight':>15}{'arm':>12}{'i_pTM':>9}{'pLDDT':>9}")
    for (arm_name, one_hot_weight), (iptm, mean_plddt, _) in sorted(
            results.items(), key=lambda item: (item[0][1], item[0][0])):
        print(f"{one_hot_weight:>15.1f}{arm_name:>12}{iptm:>9.4f}{mean_plddt:>9.4f}")

    # A nan or dead gradient is itself the #17 signal, and a cosine of nan cannot say which arm
    # produced it. Report each arm's own gradient before comparing them.
    print("\nbinder sequence gradient, per arm:")
    print(f"{'one_hot_weight':>15}{'arm':>12}{'L2 norm':>14}{'nan':>7}{'max|g|':>12}")
    for (arm_name, one_hot_weight), (_iptm, _plddt, gradient) in sorted(
            results.items(), key=lambda item: (item[0][1], item[0][0])):
        finite = gradient[np.isfinite(gradient)]
        print(f"{one_hot_weight:>15.1f}{arm_name:>12}"
              f"{float(np.linalg.norm(finite)):>14.6g}"
              f"{int((~np.isfinite(gradient)).sum()):>7}"
              f"{(float(np.abs(finite).max()) if finite.size else float('nan')):>12.6g}")

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
