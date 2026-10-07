"""Run a short host-JAX BindCraft 2 trajectory on CPU and save the sequence logits each stage ends on.

`stage_conditioning.py` answers "how much of a design round does the trunk's low bits decide" from
synthetic logits. The answer depends on the gap between each position's top two amino acids, which
is the one thing synthetic logits get wrong. This produces the real ones.

`--trunk card --card N` turns the same script into #17's device arm: the design loop's
Evoformer runs on the chip and every other stage stays on host JAX, so the two arms differ
in exactly the component the issue is about, at the same campaign seed.

No card by default: `trunk="jax"` opens no device (docs/bindcraft2.md:325). Short stage rounds and a 40-aa
binder against PD-L1, because a CPU gradient round at Siddhant's 288 tokens is minutes. The stage
*parameters* are untouched, and they are what the conditioning depends on.
"""
import argparse
import json
import os

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-pdb", required=True, help="ABSOLUTE path to the target structure")
    parser.add_argument("--target-chains", default="A")
    parser.add_argument("--hotspots", default="54,56,66,115")
    parser.add_argument("--af2-weights", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--binder-length", type=int, default=40)
    # BindCraft 2's own defaults (settings.py:604). A 64-token round is seconds on CPU, so the
    # real schedule is affordable and the logits are the ones a real trajectory reaches.
    parser.add_argument("--screen", type=int, default=50)
    parser.add_argument("--refine", type=int, default=25)
    parser.add_argument("--anneal", type=int, default=45)
    parser.add_argument("--harden", type=int, default=5)
    parser.add_argument("--dump-states", default=None,
                        help="pickle the ProteinStates entering harden, for the device A/B")
    parser.add_argument("--project", default="/tmp/bci17_capture")
    # The on-card Evoformer applies no dropout at all: tt_bio/bindcraft2.py and tt_bio/tenstorrent.py
    # contain the string zero times, while BindCraft 2 threads batch["use_dropout"] into every
    # Evoformer sub-layer (af/alphafold/model/modules_multimer.py:419, :701-737). So a host-JAX run
    # with design_dropout=false is what the device arm effectively does for the 120 rounds of
    # screen+refine+anneal, and the default is what the host control does. Running both on the host
    # is a card-free test of whether that difference alone makes the harden signature.
    parser.add_argument("--design-dropout", choices=["true", "false"], default="true")
    parser.add_argument("--trajectories", type=int, default=1)
    parser.add_argument("--binder-lengths", type=int, nargs=2, default=None)
    # The two arms of #17. `jax` is BindCraft 2 unmodified; `card` swaps the design loop's
    # Evoformer onto a chip and leaves every other stage, validation included, on host JAX
    # (campaign_predictor's `validation="jax"` default), which is what makes the design-loop
    # Evoformer the only difference between the arms.
    parser.add_argument("--trunk", choices=["jax", "card"], default="jax")
    parser.add_argument("--card", type=int, default=None,
                        help="chip index for --trunk card; must match TT_VISIBLE_DEVICES")
    # Siddhant ran stock BindCraft 2, where design_dropout defaults true and the device trunk
    # silently dropped it. This branch's fix refuses that swap, so reproducing his arm needs the
    # refusal waived explicitly. "refuse" is the fixed behaviour; "ignore" is what he actually ran.
    parser.add_argument("--evoformer-dropout", choices=["refuse", "ignore"], default="ignore")
    # One trajectory at a time, BindCraft 2's own loop, on both arms. The interleaving width is
    # chosen from free host memory, so two arms launched together on different boxes get different
    # widths and the comparison silently stops being matched (measured 2026-10-07: the pc pair drew
    # 1 and 2). Pinning it costs wall time and buys a comparison.
    parser.add_argument("--trajectories-per-card", type=int, default=1)
    args = parser.parse_args()

    if (args.trunk == "card") != (args.card is not None):
        parser.error("--trunk card needs --card N, and --card N is meaningless without it")
    if args.trunk == "card" and os.environ.get("TT_VISIBLE_DEVICES") != str(args.card):
        parser.error(f"TT_VISIBLE_DEVICES={os.environ.get('TT_VISIBLE_DEVICES')!r} does not name "
                     f"card {args.card}; the launcher sets both or neither")

    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    import jax.numpy as jnp
    from bindcraft.sequence_optimization import GradientSequenceOptimizer
    from tt_bio import bindcraft2

    #: stage name -> logits that stage ended on, filled by the hook below. The optimizer object is
    #: per stage (bindcraft/trajectory.py:215-218), so its class is the one place every stage's
    #: update passes through, and `__class__.__name__` says which stage it is.
    captured: dict[str, np.ndarray] = {}
    stage_of = {
        "LogitSequenceOptimizer": "screen_or_refine",
        "SequenceAnnealingOptimizer": "anneal",
        "OneHotSequenceOptimizer": "harden",
    }
    original_update = GradientSequenceOptimizer.update_sequence
    #: The ProteinStates the `harden` stage starts from. This is the input the device arm and the
    #: host arm have to be handed identically for a one-round comparison to mean anything.
    harden_entry = {"states": None}

    def recording_update(self, protein_states, accumulated_gradients):
        if type(self).__name__ == "OneHotSequenceOptimizer" and harden_entry["states"] is None:
            harden_entry["states"] = protein_states
        updated = original_update(self, protein_states, accumulated_gradients)
        from bindcraft.prediction import collect_shared_chains
        from bindcraft.protein import has_residue_flag, ResidueFlags
        _, chains = collect_shared_chains(updated)
        for name, chain in chains.items():
            designed = np.asarray(has_residue_flag(chain.flags, ResidueFlags.DESIGN))
            if not designed.any():
                continue
            stage = stage_of.get(type(self).__name__, type(self).__name__)
            softmax_weight, one_hot_weight, temperature, _ = self.sequence_parameters()
            key = f"{stage}:{name}"
            captured[key] = np.asarray(jnp.asarray(chain.sequence)[designed], dtype=np.float32)
            captured[key + ":params"] = np.asarray(
                [float(softmax_weight), float(one_hot_weight), float(temperature)], dtype=np.float32)
        return updated

    GradientSequenceOptimizer.update_sequence = recording_update

    # The target has to be given explicitly with an ABSOLUTE path. The `"target": "hPDL1"` preset
    # carries `"target_path": "structures/hPDL1.pdb"`, which is relative, and from any cwd but
    # BindCraft 2's own settings directory it resolves to nothing -- silently. The campaign then
    # runs with NO TARGET CHAIN: it folds the binder alone, reports 64 tokens for a 40-aa binder,
    # scores iptm exactly 0.000 at every stage, and dies in the interface loss with
    # KeyError: 'target_seed'. An interface experiment set up that way measures nothing.
    settings = {
        "targets": [{"name": "hPDL1", "target_path": args.target_pdb,
                     "chains": args.target_chains, "hotspots": args.hotspots}],
        "binder_lengths": args.binder_lengths or [args.binder_length, args.binder_length],
        "max_trajectories": args.trajectories,
        "number_of_final_designs": args.trajectories,
        "design_dropout": args.design_dropout == "true",
        "trajectory_only": True,
        "design_models": 1,
        "campaign_seed": 42,
        "screen_steps": args.screen,
        "refine_steps": args.refine,
        "anneal_steps": args.anneal,
        "harden_steps": args.harden,
        "mutate_steps": 0,
        "project_folder": args.project,
    }
    print("settings:", json.dumps(settings), flush=True)

    predictor_arguments = {"checkpoints": args.af2_weights}
    if args.trunk == "jax":
        predictor_arguments["trunk"] = "jax"
    else:
        # Nothing that reaches ttnn outside tt-bio's own CLI gets the mesh descriptor, and a lone
        # p300 chip without it raises TT_FATAL (tt_bio/tenstorrent.py::_open_and_init_device).
        from tt_bio.main import ensure_p300_mesh_descriptor
        ensure_p300_mesh_descriptor()
        predictor_arguments["card"] = args.card
        predictor_arguments["dropout"] = args.evoformer_dropout
    print("arm:", json.dumps({k: v for k, v in predictor_arguments.items()
                              if k != "checkpoints"}), flush=True)

    try:
        with bindcraft2.campaign_predictor(**predictor_arguments):
            bindcraft2.run_campaign(settings, args.project, af2_weights=args.af2_weights,
                                    trajectories_per_card=args.trajectories_per_card)
    finally:
        GradientSequenceOptimizer.update_sequence = original_update
        if args.dump_states and harden_entry["states"] is not None:
            import pickle
            with open(args.dump_states, "wb") as handle:
                pickle.dump(harden_entry["states"], handle)
            print(f"wrote {args.dump_states}", flush=True)
        if captured:
            np.savez(args.out, **captured)
            print(f"wrote {args.out}: {sorted(k for k in captured if not k.endswith(':params'))}",
                  flush=True)
        else:
            print("captured nothing", flush=True)


if __name__ == "__main__":
    main()
