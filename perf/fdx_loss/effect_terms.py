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


def resolved_binder(protein_states, predictions, prediction_state="complex", chain="binder"):
    """Push up the binder's mean probability of being experimentally resolved.

    The arm the effect measurement actually uses, chosen because **no term in BindCraft 2's
    default set reads `experimentally_resolved_ca`**: its own `experimentally_resolved` term is
    registered but weighted 0, so the default objective is indifferent to this quantity and the
    two arms can separate on it. A term that duplicates one of the eight default terms cannot
    demonstrate anything, which is what the `tight_binder` arm found out the hard way: it
    restates `compactness`, so the control was already optimising it.
    """
    import jax.numpy as jnp
    from bindcraft.loss import chain_residue_slices, resolve_prediction_state
    from bindcraft.protein import ResidueFlags, has_residue_flag

    prediction_state = resolve_prediction_state(predictions, prediction_state)
    complex_chains = protein_states[prediction_state]
    rows = chain_residue_slices(complex_chains)[chain]
    resolved = predictions[prediction_state].metrics["experimentally_resolved_ca"][rows]
    designed = has_residue_flag(complex_chains[chain].flags,
                                ResidueFlags.DESIGN).astype(resolved.dtype)
    return 1.0 - resolved.dot(designed) / (designed.sum() + 1e-8)


#: Comparable to BindCraft 2's own weights rather than dominant. The quantity is 0-1, like the
#: one `plddt_loss` reads at 0.1, and the whole default objective sits around 4.5, so 1.0 makes
#: the term about a tenth of the loss: enough to move the design, not enough to take the
#: optimiser over. `tight_binder` at 8.0 did take it over and oscillated from step 5.
RESOLVED_WEIGHT = 1.0


@contextlib.contextmanager
def with_resolved_binder():
    """The effect arm: one added term on a quantity the default objective ignores."""
    with bindcraft2.loss_terms(
            add={"resolved_binder": (resolved_binder, RESOLVED_WEIGHT)}) as installed:
        yield installed


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
