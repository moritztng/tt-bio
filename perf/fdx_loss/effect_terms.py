"""The A/B arm for `design_run.py --loss-module`: one added term, and the quantity it targets.

The point of the measurement is not that the loss falls. A loss always falls. The point is that
the *geometric quantity the added term asks about* moves in the direction it asks for, measured on
the final coordinates rather than read off the objective that was being optimised.

So the term and the measurement are written separately here, from different arrays: the term reads
the predicted CA coordinates through BindCraft 2's own helper, and `radius_of_gyration` recomputes
the same quantity in NumPy from the coordinates the run wrote out.
"""
import contextlib

import numpy as np

from tt_bio import bindcraft2

#: Heavy on purpose. The arms differ by two gradient steps, so a weight comparable to BindCraft
#: 2's own (0.5 for `compactness`) would move the structure by less than the run-to-run spread and
#: the measurement would say nothing either way.
WEIGHT = 8.0


def tight_binder(protein_states, predictions, prediction_state="binder_alone", chain="binder"):
    """Pull the binder's CA atoms towards their own centroid: a radius-of-gyration penalty.

    Deliberately the simplest term that has an unambiguous geometric signature, so the effect
    measurement cannot be mistaken for the optimiser doing something else. Masked on the real
    residues, so padding does not drag the centroid.
    """
    import jax.numpy as jnp
    from bindcraft.loss import chain_atom_coordinates, resolve_prediction_state
    from bindcraft.protein import real_residue_mask

    # Always resolve the state name first, the way BindCraft 2's own terms do: a campaign names
    # its states after its targets, so a term that indexes `predictions["complex"]` directly
    # raises KeyError on any design whose state is called something else.
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    folded = predictions[prediction_state].protein_complex[chain]
    coordinates, present = chain_atom_coordinates(folded)
    mask = (present * real_residue_mask(protein_states[prediction_state][chain].flags)
            ).astype(coordinates.dtype)
    weight = mask / (mask.sum() + 1e-8)
    centroid = (coordinates * weight[:, None]).sum(0)
    return jnp.sqrt((jnp.square(coordinates - centroid).sum(-1) * weight).sum() + 1e-8)


@contextlib.contextmanager
def with_tight_binder():
    """What `design_run.py --loss-module perf.fdx_loss.effect_terms --loss-attr with_tight_binder`
    wraps the design loop in."""
    with bindcraft2.loss_terms(add={"tight_binder": (tight_binder, WEIGHT)}) as installed:
        yield installed


@contextlib.contextmanager
def baseline():
    """The control arm: the hook is imported and opened, and changes nothing.

    Not the same as not calling it. This arm proves the machinery itself is inert, which is the
    other half of the bit-identical claim.
    """
    with bindcraft2.loss_terms() as installed:
        yield installed


def radius_of_gyration(coordinates, mask=None) -> float:
    """The quantity `tight_binder` asks about, recomputed in NumPy from the written coordinates.

    float64 and independent of anything in the design loop, so the A/B is not reading back the
    objective it optimised.
    """
    points = np.asarray(coordinates, dtype=np.float64).reshape(-1, 3)
    if mask is not None:
        points = points[np.asarray(mask, dtype=bool).reshape(-1)]
    centroid = points.mean(0)
    return float(np.sqrt(np.square(points - centroid).sum(-1).mean()))
