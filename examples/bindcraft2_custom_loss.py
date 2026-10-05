"""Add, reweight and replace a BindCraft 2 loss term, and grade the one you added.

Runs on CPU, needs no Tenstorrent card and no AlphaFold 2 weights: it exercises the seam
``bindcraft2.loss_terms`` patches and the gradient grade, which is everything that is specific to
tt-bio. Put the same ``with`` block around ``campaign.run_campaign(...)`` inside
``bindcraft2.campaign_predictor(card=0)`` and the terms are live on card.

    PYTHONPATH=<tt-bio>:<bindcraft2> python3 examples/bindcraft2_custom_loss.py \\
        --settings <bindcraft2>/examples/pdl1_denovo.json
"""
import argparse
import warnings

import jax
import jax.numpy as jnp
from bindcraft.loss import (build_design_losses, chain_residue_slices,
                            distogram_bin_distances, resolve_prediction_state)
from bindcraft.protein import ResidueFlags, has_residue_flag
from bindcraft.settings import read_settings

from tt_bio import bindcraft2


def hotspot_contacts(protein_states, predictions, prediction_state="complex",
                     binder="binder", target="target", cutoff=8.0):
    """Reward contacts with the target's hotspot residues, from the model's own distogram.

    A term may read anything in ``bindcraft2.INTERMEDIATES``. This one takes the distogram,
    softmaxes it into a contact probability under ``cutoff`` Angstrom using BindCraft 2's own bin
    edges, and returns one minus the mean probability over (binder, hotspot) pairs -- so driving
    the term down drives those contacts up.

    Every rule a term has to follow is visible here: resolve the state name before indexing it,
    a scalar out, no Python branch on a traced value, a mask instead of a slice so padding cannot
    leak in, and an eps on the divide.
    """
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    metrics = predictions[prediction_state].metrics
    complex_chains = protein_states[prediction_state]
    rows = chain_residue_slices(complex_chains)

    edges = distogram_bin_distances(metrics["distogram"].shape[-1])[1:]
    contact = jax.nn.softmax(metrics["distogram"], axis=-1)[..., 1:][..., edges <= cutoff].sum(-1)

    binder_rows = jnp.zeros(contact.shape[0], bool).at[rows[binder]].set(True)
    hotspot = has_residue_flag(complex_chains[target].flags, ResidueFlags.HOTSPOT)
    target_rows = jnp.zeros(contact.shape[0], bool).at[rows[target]].set(hotspot)
    pairs = (binder_rows[:, None] & target_rows[None, :]).astype(contact.dtype)
    return 1.0 - (contact * pairs).sum() / (pairs.sum() + 1e-8)


def softer_plddt(protein_states, predictions, prediction_state="binder_alone", chain="binder"):
    """A replacement for BindCraft 2's ``plddt_loss``: the same quantity, squared.

    A replacement keeps the name, the weight and the target weighting of the term it replaces, so
    its signature has to carry the same state parameters -- ``prediction_state`` here. tt-bio
    refuses a replacement that changes them, because that silently changes how BindCraft 2 fans
    the term out over states and the weight would no longer mean the same thing.
    """
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    rows = chain_residue_slices(protein_states[prediction_state])[chain]
    confidence = predictions[prediction_state].metrics["plddt"][rows]
    designed = has_residue_flag(protein_states[prediction_state][chain].flags,
                                ResidueFlags.DESIGN).astype(confidence.dtype)
    return jnp.square(1 - confidence).dot(designed) / (designed.sum() + 1e-8)


def dead_on_arrival(protein_states, predictions, prediction_state="complex"):
    """What a term with no gradient looks like. tt-bio refuses this one; that is the point."""
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    return jnp.argmax(predictions[prediction_state].metrics["distogram"], axis=-1).mean() * 1.0


def table(losses, title):
    print(f"\n{title}")
    for name, entry in sorted(losses.items()):
        print(f"  {name:<34s} weight {entry.weight:+.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", required=True, help="a BindCraft 2 settings JSON")
    parser.add_argument("--binder", type=int, default=110)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--probes", type=int, default=8)
    arguments = parser.parse_args()

    settings = read_settings(arguments.settings, {})
    targets = {"complex": 1.0}
    baseline = build_design_losses(settings, targets, arguments.binder, arguments.seed)
    table(baseline, f"BindCraft 2's own terms for {arguments.settings}:")

    print(f"\n{len(bindcraft2.terms())} terms are registered in total; "
          f"bindcraft2.terms() lists them with their weight setting and what they read.")

    with bindcraft2.loss_terms(
            add={"hotspot_contacts": (hotspot_contacts, 0.8)},
            weight={"interface_contacts": 0.5, "binder_helicity": 0.0},
            replace={"plddt_loss": softer_plddt}) as installed:
        changed = build_design_losses(settings, targets, arguments.binder, arguments.seed)
        table(changed, "with the hook open -- added, reweighted, replaced, switched off:")
        print(f"\n  hotspot_contacts is now registered: {'hotspot_contacts' in installed}, "
              f"weight setting {installed['hotspot_contacts'].setting}, "
              f"reads {installed['hotspot_contacts'].reads}")

    restored = build_design_losses(settings, targets, arguments.binder, arguments.seed)
    print(f"\nafter the block: {len(restored)} terms, identical to the baseline: "
          f"{ {n: e.weight for n, e in restored.items()} == {n: e.weight for n, e in baseline.items()} }"
          f", and hotspot_contacts is gone: {'hotspot_contacts' not in bindcraft2.terms()}")

    print("\nthe gradient of the added term, float64 central differences:")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rows, worst, dtype = bindcraft2.check_gradient(hotspot_contacts, probes=arguments.probes)
    for row in rows:
        if row.probes:
            print("   ", row)
    print(f"    worst {worst:.2e} relative, the term's own forward in {dtype}")

    print("\nand a term with no gradient, which does not get installed:")
    try:
        with bindcraft2.loss_terms(add={"dead_on_arrival": (dead_on_arrival, 1.0)}):
            print("    INSTALLED -- the screen missed it")
    except bindcraft2.DeadGradient as refusal:
        print(f"    {type(refusal).__name__}: {refusal}")


if __name__ == "__main__":
    main()
