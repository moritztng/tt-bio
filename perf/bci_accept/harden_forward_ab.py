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


def one_round(build, protein_states, losses, one_hot_weight, model=None):
    """One `sequence_gradients` call at the harden stage parameters."""
    with build() as design_model:
        predictions, gradients = design_model.sequence_gradients(
            protein_states, losses, model,
            softmax_weight=1.0, one_hot_weight=one_hot_weight, temperature=0.01, logit_scale=2.0)
    return predictions, gradients


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", required=True, help="harden-entry ProteinStates pickle")
    parser.add_argument("--af2-weights", required=True)
    parser.add_argument("--card", type=int, default=None,
                        help="chip to open; omit to run the host arm alone")
    parser.add_argument("--binder-length", type=int, default=40)
    args = parser.parse_args()

    from bindcraft.loss import build_losses
    from bindcraft.settings import build_design_settings
    from tt_bio import bindcraft2

    settings = {
        "target": "hPDL1",
        "binder_lengths": [args.binder_length, args.binder_length],
        "max_trajectories": 1, "number_of_final_designs": 1, "trajectory_only": True,
        "design_models": 1, "campaign_seed": 42,
    }
    design_settings = build_design_settings(settings)
    losses = build_losses(design_settings.settings, seed=design_settings.seed)
    with open(args.states, "rb") as handle:
        protein_states = pickle.load(handle)

    arms = {"host-JAX": lambda: bindcraft2.predictor(trunk="jax", checkpoints=args.af2_weights)}
    if args.card is not None:
        arms["on-card"] = lambda: bindcraft2.predictor(card=args.card, checkpoints=args.af2_weights)

    results = {}
    for arm_name, build in arms.items():
        for one_hot_weight in (0.0, 1.0):
            predictions, gradients = one_round(build, protein_states, losses, one_hot_weight)
            results[(arm_name, one_hot_weight)] = (
                interface_ptm(predictions), plddt(predictions), binder_gradient(gradients))

    print(f"{'one_hot_weight':>15}{'arm':>12}{'i_pTM':>9}{'pLDDT':>9}")
    for (arm_name, one_hot_weight), (iptm, mean_plddt, _) in sorted(
            results.items(), key=lambda item: (item[0][1], item[0][0])):
        print(f"{one_hot_weight:>15.1f}{arm_name:>12}{iptm:>9.4f}{mean_plddt:>9.4f}")

    if len(arms) == 2:
        print("\nbinder sequence gradient, on-card against host JAX, on the same logits:")
        for one_hot_weight in (0.0, 1.0):
            host = results[("host-JAX", one_hot_weight)][2]
            card = results[("on-card", one_hot_weight)][2]
            cosine = float(np.dot(host, card) / (np.linalg.norm(host) * np.linalg.norm(card) + 1e-30))
            relative = float(np.linalg.norm(card - host) / (np.linalg.norm(host) + 1e-30))
            print(f"  one_hot_weight={one_hot_weight:.1f}  cosine={cosine:.6f}  "
                  f"relative L2 difference={relative:.6f}")


if __name__ == "__main__":
    main()
