"""The engine: features in, one batch of designs out.

An engine takes a feature batch and returns four tensors; ``write_design`` needs exactly those
and reads nothing else off the model:

    coords_nm        (designs, tokens, 37, 3)   nanometres, atom37 slots
    residue_type     (designs, tokens)          vocabulary indices, -1 outside the emitted set
    atom_mask        (designs, tokens, 37)      which atom37 slots are real
    insertion_logits (designs, tokens, tokens)  unindexed-motif insertion mapping

:class:`DesignEngine` is the real one: the sampler (`sampler.py`) integrates the flow, calling the
flow network (`network.py`) once per step, which runs the trunk (`trunk.py`) on the card at that
step's recycle depth; the autoencoder's decoder (`autoencoder.py`) turns the final latents into
atoms and residue types. That is upstream's `VanillaStrategy.generate` plus
`Proteina._format_sample_local_latents`, in the order upstream runs them.

The recipe is the released one and is not negotiable: `thinking_medium` is 60 steps and 180 trunk
passes, and the sampler refuses a depth list that is not as long as the step count. Doing fewer
passes is a different model, not a faster one.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

#: The released bundle, as `tt_bio.weights` resolves it.
DEFAULT_WEIGHTS = "nvidia/RFD4-Proteina"


@dataclass(frozen=True)
class DesignOutput:
    """What an engine returns. Attribute names are upstream's, so write_cif takes it as is."""

    coords_nm: object
    residue_type: object
    atom_mask: object
    insertion_logits: object


@runtime_checkable
class Engine(Protocol):
    """Generate designs from one featurized example."""

    def design(self, features: dict, *, num_designs: int, nsteps: int, seed: int) -> DesignOutput:
        ...


class DesignEngine:
    """RFD4 on Tenstorrent: the sampler driving the trunk and the autoencoder on one card.

    The weights are loaded once and stay on the device, so a second design on the same engine
    pays no load. `recipe` names a shipped recycle strategy (`thinking_low|medium|high`); the
    step count comes from the recipe unless the caller overrides it.
    """

    def __init__(self, weights: str | Path = DEFAULT_WEIGHTS, *, recipe: str = "thinking_medium",
                 max_parallel_samples: int | None = 5):
        from tt_bio.rfd4 import network as _network
        from tt_bio.rfd4 import sampler as _sampler
        from tt_bio.rfd4.autoencoder import RFD4AutoEncoder
        from tt_bio.rfd4.trunk import Trunk, compute_kernel_config

        import ttnn

        path = _resolve_weights(weights)
        state, config = _network.load_checkpoint(path)
        cfg = config["nn"]
        self.spec = _sampler.recycle_strategy(recipe)
        self.config = config
        self.glue = _network.FlowGlue(state, config)
        from tt_bio.tenstorrent import WeightScope

        self.trunk = Trunk(WeightScope({k: v.float() for k, v in _network._sub(state, "nn.").items()}),
                           compute_kernel_config(), nlayers=cfg["nlayers"], n_heads=cfg["nheads"],
                           every_n=cfg["update_pair_repr_every_n"],
                           residual_dtype=ttnn.float32)
        # The bundle stores the AE under `autoencoder.`; it is byte-identical to the AE's own
        # checkpoint (699/699 tensors), which is what the AE was graded on.
        self.autoencoder = RFD4AutoEncoder(_network._sub(state, "autoencoder."), parts=("decoder",))
        self.max_parallel_samples = max_parallel_samples
        self._device = self.autoencoder.dev["decoder"].device

    def design(self, features: dict, *, num_designs: int, nsteps: int, seed: int,
               rng: str = "per_design") -> DesignOutput:
        """`rng="per_design"` gives every design its own stream (`sampler.DesignRng`).
        `rng="upstream"` seeds the global stream with `seed` and draws in upstream's order, which
        is how a design is compared one to one against upstream's run of the same seed."""
        import torch

        from tt_bio._vendor.rfd4_prep._cif.model_masks import compute_flow_emit_masks
        from tt_bio.rfd4 import network as _network
        from tt_bio.rfd4 import sampler as _sampler

        batch = preprocess(features)
        n_tokens = batch["padding_mask"].shape[1]
        spec = self.spec
        if nsteps != spec.nsteps:
            # Fewer steps is a different (worse) recipe, not a faster one; say which ran.
            logger.info("rfd4: %d steps instead of the recipe's %d", nsteps, spec.nsteps)
            spec = replace(spec, nsteps=nsteps)

        def trunk(s_init, z_init, conds, mask):
            return self.trunk.run_bucketed(s_init, z_init, conds, mask, self._device)

        network = _network.FlowNetwork(self.glue, trunk, batch)
        chunked = _sampler.ChunkedNetwork(network, spec,
                                          max_parallel_samples=self.max_parallel_samples)
        if rng == "upstream":
            torch.manual_seed(seed)
        out = _sampler.generate(spec, chunked, n_designs=num_designs, n_tokens=n_tokens,
                                seed=seed, rng_mode=rng, emit_masks=compute_flow_emit_masks(batch),
                                groups=_groups(batch), coord_cond_mask=batch.get("coord_cond_mask"))
        logger.info("rfd4: %d designs, %d steps, %d trunk passes", num_designs, nsteps,
                    out.trunk_passes)

        decoded = self.decode(batch, out, num_designs=num_designs)
        return DesignOutput(coords_nm=decoded["coors_nm"], residue_type=decoded["residue_type"],
                            atom_mask=decoded["atom_mask"],
                            insertion_logits=out.insertion_logits)

    def decode(self, batch: dict, out, *, num_designs: int) -> dict:
        """The autoencoder's decoder on the final latents, then upstream's assembly of the result.

        With the release's `condition_decoder_on_sequence_prediction`, the flow's sequence readout
        picks the residue type for every emitted token the user did not pin through `C_SEQ`, and
        the decoder is conditioned on that, so the structure and the sequence agree.
        """
        import torch

        from tt_bio._vendor.rfd4_prep._cif.model_masks import compute_ae_masks
        from tt_bio.rfd4 import network as _network
        from tt_bio.rfd4.ae_fold import C_SEQ

        tables = self.autoencoder.tab
        padding_mask = batch["padding_mask"]
        decode_batch = _network._broadcast_batch(dict(batch), num_designs)
        decode_batch["z_latent"] = out.x["local_latents"]
        decode_batch["ca_coors_nm"] = out.x["bb_ca"]

        if out.sequence_logits is not None:
            logits = out.sequence_logits
            if "token_type" in decode_batch:
                allowed = tables["allowed"][decode_batch["token_type"].long()]
                logits = logits.masked_fill(~allowed, torch.finfo(logits.dtype).min)
            predicted = logits.argmax(dim=-1)
            _, emit = compute_ae_masks(decode_batch)
            pinned = decode_batch[C_SEQ] != tables["mask_token"]
            decode_batch[C_SEQ] = torch.where(emit & ~pinned, predicted, decode_batch[C_SEQ])

        coord_groups = decode_batch.get("condition_CoordinateGroup_1_atom")
        decoded = self.autoencoder.decode(
            decode_batch, required_atom_mask=None if coord_groups is None else coord_groups >= 0)

        emit = decoded["emit_mask"]
        coords_nm = decoded["coors_nm"].clone()
        atom_mask = decoded["atom_mask"].clone()
        residue_type = decoded["residue_type"].long()
        # Tokens with no latents (unfixed ligands and the like) keep the sampled CA in slot 1 and
        # are marked unencoded, which is how upstream reports a token the decoder did not emit.
        no_latent = (~emit) & _as_bool(padding_mask, num_designs)
        atom_mask[..., 1] |= no_latent
        coords_nm[..., 1, :] = torch.where(no_latent[..., None], out.x["bb_ca"], coords_nm[..., 1, :])
        residue_type = torch.where(emit, residue_type, torch.full_like(residue_type, -1))
        return {"coors_nm": coords_nm, "residue_type": residue_type, "atom_mask": atom_mask}


def preprocess(features: dict) -> dict:
    """`Proteina.preprocess_batch_for_generation`: the two derived keys the network reads."""
    batch = dict(features)
    batch["coords_nm"] = features["coords"] / 10.0
    batch["coord_mask"] = features["atom37_mask"]
    return batch


def _groups(batch: dict) -> dict:
    """The token-level group tensors stochastic centering is scoped by (`extract_groups`)."""
    crdgrp = batch.get("condition_CoordinateGroup_1_atom")
    out = {name: (batch[name].long() if name in batch else None)
           for name in ("chain_iid", "pn_unit_iid", "molecule_iid")}
    out["coord_group"] = crdgrp.long().max(dim=-1).values if crdgrp is not None else None
    return out


def _as_bool(mask, num_designs: int):
    import torch

    mask = mask.to(torch.bool)
    return mask.expand(num_designs, *mask.shape[1:]) if mask.shape[0] == 1 else mask


def _resolve_weights(weights: str | Path) -> Path:
    """The released bundle's directory, downloading it on first use."""
    path = Path(str(weights)).expanduser()
    if path.exists():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(str(weights)))


def get_engine(name: str = "auto", **kwargs) -> Engine:
    """The engine to design with."""
    if name not in ("auto", "device"):
        raise ValueError(f"unknown RFD4 engine {name!r}")
    return DesignEngine(**kwargs)
