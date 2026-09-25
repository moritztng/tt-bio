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
import json
import time

import numpy as np

from bindcraft.af2 import (AlphaFoldDesignModel, padded_prediction_complex,
                           padded_prediction_length, pad_design_chains,
                           concatenate_chain_arrays)
from bindcraft.prediction import DifferentiableProteinPredictor
from bindcraft.protein import AMINO_ACIDS, real_residue_weights

DEVICE_TOKEN_BUCKET = 32


def sequence_letters(protein) -> str:
    """The sequence a filter would read off this chain.

    BindCraft 2 carries a design chain as (L, 20) logits and takes its argmax wherever it
    needs letters (`bindcraft/filters.py:229`), so this is that same reading.
    """
    return "".join(AMINO_ACIDS[i] for i in np.asarray(protein.sequence).argmax(-1))


def masked_residue_count(protein_states, length_bucket_size: int, target_pad_length: int = 0,
                         path: str = "predict") -> int:
    """How many residues BindCraft 2's padding masks out of this complex.

    BindCraft 2 pads in TWO places and they do not overlap, which is what this function
    got wrong: it counted only the first and returned 0 for the single-chain PD-L1 fold
    that reads 0.534 pLDDT with a 19.28% zero pair mask, so the refusal built to catch
    exactly that padding would have waved it through.

    * `pad_design_chains` pads each DESIGN chain up to the bucket. That is the
      `sequence_gradients` path (`bindcraft/af2.py:385`) and it is what the old count saw.
    * `predict` pads the TOTAL token axis to the bucket (`bindcraft/af2.py:298`), and it
      only pads chains at all when `target_pad_length` is set. A single chain of 115 is
      unpadded by the first rule and padded to 128 by the second.

    `path` selects which one, because a caller asking about a gradient step and a caller
    asking about a fold are asking different questions.
    """
    if path not in ("predict", "sequence_gradients"):
        raise ValueError(f"path must be 'predict' or 'sequence_gradients', not {path!r}")
    masked = 0
    for state_name, protein_complex in protein_states.items():
        if path == "sequence_gradients":
            padded = pad_design_chains({state_name: protein_complex},
                                       length_bucket_size, target_pad_length)[state_name]
        else:
            padded = (padded_prediction_complex(protein_complex, length_bucket_size,
                                                target_pad_length)
                      if target_pad_length else protein_complex)
        names = tuple(sorted(padded))
        flags = concatenate_chain_arrays(names, padded, "flags")["flags"]
        masked += int((np.asarray(real_residue_weights(flags)) == 0).sum())
        if path == "predict":
            # The token-axis pad `predict` adds on top, which is pure mask.
            residues = sum(len(padded[n]) for n in names)
            masked += padded_prediction_length(residues, length_bucket_size) - residues
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

    def __init__(self, *args, trunk: str = "jax", card: int | None = None,
                 pool=None, seqlog: str | None = None, **kwargs):
        if trunk not in ("jax", "device"):
            raise ValueError(f"trunk must be 'jax' or 'device', not {trunk!r}")
        super().__init__(*args, **kwargs)
        self.trunk = trunk
        self.card = card
        #: A `multimer_pool.MultimerPool`, or None for the single-checkpoint arm. BindCraft 2
        #: samples one design model per gradient step, so with a pool the device trunk has to
        #: follow that choice rather than hold one checkpoint for the campaign.
        self.pool = pool
        #: Where to append one JSON line per `predict`, or None. The first shipped-pool
        #: trajectory was rejected on a mutate stage reporting ptm = iptm = 1.0, and the
        #: sequence that produced it is in no artifact BindCraft 2 writes for a REJECTED
        #: trajectory -- the losses CSV carries metrics and no sequence, and the PDB is
        #: only written on acceptance. So asking whether that degeneracy was in the
        #: sequence or in us needed the whole 2.25 h trajectory over again. One line per
        #: call is cheaper than one trajectory.
        self.seqlog = seqlog
        self._device = None

    # -------------------------------------------------------------- the Protocol surface

    def _resolved(self, model):
        """Resolve BindCraft 2's per-step checkpoint ONCE, and hand the name to both sides.

        `predict` and `sequence_gradients` resolve `model` themselves
        (`bindcraft/af2.py:287,381`), and for `model=None` that does not look a name up, it
        SAMPLES one and splits `self.key` doing it. So resolving a second time here to pick
        the card's trunk drew a second, independent name: the card ran one checkpoint's 48
        Evoformer blocks while the embedder, the template stack, the structure module and the
        heads around them came from another, in four calls out of five. Nothing downstream can
        see that -- the shapes agree and the loss still falls -- and it is what made the first
        full shipped-pool trajectory report ptm = iptm = 1.0 for every mutate round and then
        fail the final pLDDT filter.

        Only the pool arm resolves here. With no pool there is nothing to select and the extra
        draw would move `self.key`, and the monomer-pinned arm is the one every published
        number in this campaign was measured on.
        """
        return self._resolve_model_name(model) if self.pool is not None else model

    def _select(self, model):
        """Point the device trunk at the checkpoint the JAX side is about to use."""
        if self.pool is not None:
            self.pool.use(model)

    def _record(self, model, protein_states):
        """Append the state this call is about to fold, with the checkpoint folding it."""
        if not self.seqlog:
            return
        row = {"t": round(time.time(), 1), "trunk": self.trunk, "model": model,
               "states": {state: {chain: sequence_letters(protein)
                                  for chain, protein in complex_.items()}
                          for state, complex_ in protein_states.items()}}
        with open(self.seqlog, "a") as handle:
            handle.write(json.dumps(row) + "\n")

    def predict(self, protein_states, model=None, *args, **kwargs):
        if self.trunk == "device":
            self._open_device()
            model = self._resolved(model)
            self._select(model)
        self._record(model, protein_states)
        return super().predict(protein_states, model, *args, **kwargs)

    def sequence_gradients(self, protein_states, losses, model=None, *args, **kwargs):
        if self.trunk == "device":
            self._open_device()
            model = self._resolved(model)
            self._select(model)
        return super().sequence_gradients(protein_states, losses, model, *args, **kwargs)

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
