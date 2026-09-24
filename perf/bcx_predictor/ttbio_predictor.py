#!/usr/bin/env python3
"""A BindCraft 2 predictor backed by tt-bio's AlphaFold 2 trunk on a Tenstorrent card.

BindCraft 2 takes its predictor as a parameter everywhere. `prediction.py:9`
`ProteinPredictor` declares `predict`, `:14` `DifferentiableProteinPredictor` adds
`sequence_gradients`, ten entry points in `trajectory.py` are typed to them, and
`campaign.py:262` is the only place in the repository that builds one. So a backend is a
class, not a fork of the loop, and this is that class.

It subclasses `AlphaFoldDesignModel` rather than reimplementing it. Everything between
`ProteinStates` and the trunk -- padding, canonical names, templates, frozen interfaces,
`alphafold_input_features`, `alphafold_prediction_metrics`, the Kabsch alignment, the
`StructurePrediction` assembly -- stays BindCraft 2's own code. The only thing replaced is
the Evoformer, which is what tt-bio has on card.

WHERE THE TRUNK GOES IN. `sequence_gradients` compiles one jitted program and
differentiates it w.r.t. the sequences alone (`af2.py:328-378`), so the trunk cannot be
swapped at the Python level. The seam is inside the vendored haiku model: the extra-MSA
stack (`modules.py:1528`) and the Evoformer stack (`modules.py:1594`) both consume
`{'msa', 'pair'}` activations and masks and return the same, and `single_activations`
(`:1599`) projects the first MSA row. `bcx-e2e` proved that cut at `(single, pair)`, with
the projection on the DEVICE side, and showed the tail's cotangents seed those two roots.

THE MASK NO LONGER DECIDES THE CONFIGURATION, AND THE REFUSAL BELOW IS STALE. This class
was written when tt-bio's trunk served an all-ones mask only and asserted it. It does not:
`af2.af2_pair_masks` builds the pair track's multiply and key bias, `AF2EvoformerBlock`
takes the MSA mask, and `splice.py` passes both. `length_bucket_size` 32 is therefore
available, which is the setting the device wants -- the PD-L1 complex costs 4.504 s at 211
and 1.369 s padded to 224.

And the refusal below never covered the padding that actually broke a fold.
`masked_residue_count` counts what `pad_design_chains` pads, which is the DESIGN chain and
only when `target_pad_length` is set (`bindcraft/af2.py:56` and `:295`). `predict` pads the
TOTAL at `bindcraft/af2.py:298` and `real_residue_weights` zeroes `seq_mask` on that tail.
Nothing here counts it: on the 115-residue PD-L1 target folded ALONE the count is 0 while
the fold carries 13 masked residues and a pair mask 19.3% zero
(`perf/bcx_mono/capture.json`). That is the fold that read pLDDT 0.534 against BindCraft 2's
own 0.950, so the guard would have waved it through. Counting the tail is `bcx-predictor`'s
to land with its own test; `bcx-mono` left the function alone and recorded the undercount
(`state/bcx-mono.md`).
"""
import numpy as np

from bindcraft.af2 import (AlphaFoldDesignModel, padded_prediction_length,
                           protein_state_shapes, pad_design_chains, concatenate_chain_arrays)
from bindcraft.prediction import DifferentiableProteinPredictor
from bindcraft.protein import real_residue_weights

DEVICE_TOKEN_BUCKET = 32


def masked_residue_count(protein_states, length_bucket_size: int, target_pad_length: int = 0) -> int:
    """How many residues BindCraft 2's own padding would mask out of this complex."""
    padded = pad_design_chains(protein_states, length_bucket_size, target_pad_length)
    masked = 0
    for state_name, chain_names, _ in protein_state_shapes(padded):
        flags = concatenate_chain_arrays(chain_names, padded[state_name], "flags")["flags"]
        masked += int((np.asarray(real_residue_weights(flags)) == 0).sum())
    return masked


def refuse_masked_state(protein_states, length_bucket_size: int, target_pad_length: int = 0) -> None:
    masked = masked_residue_count(protein_states, length_bucket_size, target_pad_length)
    if masked:
        raise ValueError(
            f"tt-bio's AF2 trunk runs an unmasked fold only (tt_bio/af2.py:24-28, asserted at "
            f":385 and :509), and this complex masks {masked} residues because BindCraft 2 pads "
            f"the design chain to length_bucket_size={length_bucket_size}. Run the campaign with "
            f"length_bucket_size=1, which leaves the complex unpadded and changes no step, "
            f"recycle, loss or filter.")


class TTBioAlphaFoldDesignModel(AlphaFoldDesignModel):
    """`AlphaFoldDesignModel` with its Evoformer trunk on a Tenstorrent card.

    `trunk='jax'` keeps BindCraft 2's own trunk and is the control arm: same class, same
    call path, same bucket rounding, so a device result is compared against this rather
    than against a differently-shaped program.
    """

    #: `campaign.py:59` negotiates this; the trunk returns AF2's distogram head unchanged.
    provides_distogram = True

    def __init__(self, *args, trunk: str = "jax", card: int | None = None, **kwargs):
        if trunk not in ("jax", "device"):
            raise ValueError(f"trunk must be 'jax' or 'device', not {trunk!r}")
        super().__init__(*args, **kwargs)
        self.trunk = trunk
        self.card = card
        self._device = None

    # -------------------------------------------------------------- the Protocol surface

    def predict(self, protein_states, *args, **kwargs):
        if self.trunk == "device":
            self._open_device()
        return super().predict(protein_states, *args, **kwargs)

    def sequence_gradients(self, protein_states, losses, *args, **kwargs):
        if self.trunk == "device":
            self._open_device()
        return super().sequence_gradients(protein_states, losses, *args, **kwargs)

    # -------------------------------------------------------------- the device side

    def _open_device(self):
        """The splice owns the device context, not this class.

        `splice.evoformer_on_device` replaces `modules.py`'s Evoformer `layer_stack` for
        the duration of a campaign, so by the time this class runs the trunk is already on
        card and there is nothing here to open. `trunk='device'` is therefore a label on
        the stamp and a statement of intent; `trunk='jax'` is the control arm, the same
        class on BindCraft 2's trunk, which is what a device result gets compared against.
        """
        return None

    @staticmethod
    def device_padded_length(residue_count: int) -> int:
        """tt-bio's token axis buckets to 32. Measured on qb2 card 3: the PD-L1 complex at
        211 costs 4.504 s on the trunk forward and the same design padded to 224 costs
        1.369 s, so rounding up does more arithmetic and is 3.29x faster."""
        return padded_prediction_length(residue_count, DEVICE_TOKEN_BUCKET)


def conforms(model) -> bool:
    """`DifferentiableProteinPredictor` is `@runtime_checkable`, so this is a method-presence
    check and a non-JAX backend can satisfy it."""
    return isinstance(model, DifferentiableProteinPredictor)
