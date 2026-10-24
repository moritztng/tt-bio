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
from dataclasses import dataclass, field, replace
from pathlib import Path
from collections.abc import Callable, Sequence
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
    #: Saved trajectory snapshots, empty unless the run asked for them
    #: (`tt_bio.rfd4.trajectory.write_trajectory` puts them on disk).
    trajectory: list = field(default_factory=list)


def auto_width(n_tokens: int) -> int:
    """The default sample width at `n_tokens`: the most designs whose pair rows together fit in one 512-token
    design's, twice that below 256 tokens, and at most 16, the widest graded (128 tokens). That is 10 at 224, 4 at
    256, 1 from 363. The trunk's triangle multiplication takes such a batch as one call on the 512-token route
    (`pair_batch`): at 128 tokens a design costs 2.27 s at width 16 against 4.86 s alone (p300c, ~1330 MHz). Below
    256 a call still pays per launch, so the doubled budget buys 1.056x a design at 224 and 1.024x at 192; from 384
    a wider call is slower (traced trunk, p300c, 1275-1350 MHz). A width DRAM refuses narrows by itself."""
    n = max(1, int(n_tokens))
    return min(16, max(1, 512 ** 2 * (2 if n < 256 else 1) // n ** 2))


@runtime_checkable
class Engine(Protocol):
    """Generate designs from one featurized example."""

    def design(self, features: dict, *, num_designs: int, nsteps: int, seed: int) -> DesignOutput:
        ...


def load_trunk(glue, state: dict, cfg: dict):
    """The device trunk over the `nn.` weights the heads left to it. Refuses a checkpoint where a
    weight is read by neither, so a renamed or extra tensor fails the load instead of designing."""
    import ttnn

    from tt_bio.rfd4 import network as _network
    from tt_bio.rfd4.trunk import Trunk, compute_kernel_config
    from tt_bio.tenstorrent import WeightScope

    scope = WeightScope.recording({k: v.float() for k, v in _network._sub(state, "nn.").items()})
    trunk = Trunk(scope, compute_kernel_config(), nlayers=cfg["nlayers"], n_heads=cfg["nheads"],
                  every_n=cfg["update_pair_repr_every_n"], residual_dtype=ttnn.float32)
    glue.check_trunk_reads(scope.reads)
    return trunk


class DesignEngine:
    """RFD4 on Tenstorrent: the sampler driving the trunk and the autoencoder on one card.

    The weights are loaded once and stay on the device, so a second design on the same engine
    pays no load. `recipe` names a shipped recycle strategy (any of `sampler.RECYCLE_SCHEDULES`:
    the three `thinking_*` aliases, and `{low,medium,high}/{flat,front_loaded,midpeak}`); the
    step count comes from the recipe unless the caller overrides it. `tsr_k` is the sequence
    temperature, upstream's `local_latents.simulation_step_params.tsr_k`: lower is hotter.
    """

    def __init__(self, weights: str | Path = DEFAULT_WEIGHTS, *, recipe: str = "thinking_medium",
                 tsr_k: float | None = None, adapter: str | Path | None = None,
                 max_parallel_samples: int | None | str = "auto"):
        from tt_bio.rfd4 import network as _network
        from tt_bio.rfd4 import sampler as _sampler
        from tt_bio.rfd4.autoencoder import RFD4AutoEncoder
        path = _resolve_weights(weights)
        state, config = _network.load_checkpoint(path, adapter=adapter)
        cfg = config["nn"]
        self.spec = _sampler.recycle_strategy(recipe)
        if tsr_k is not None:
            latents = self.spec.modalities[_sampler.LOCAL_LATENTS]
            self.spec = replace(self.spec, modalities=dict(self.spec.modalities,
                                                           **{_sampler.LOCAL_LATENTS: replace(latents, tsr_k=tsr_k)}))
        self.config = config
        self.glue = _network.FlowGlue(state, config)
        self.trunk = load_trunk(self.glue, state, cfg)
        # The bundle stores the AE under `autoencoder.`; it is byte-identical to the AE's own
        # checkpoint (699/699 tensors), which is what the AE was graded on.
        self.autoencoder = RFD4AutoEncoder(_network._sub(state, "autoencoder."), parts=("decoder",))
        # "auto" batches as many designs a device step as `auto_width` allows. A batched design is not
        # bit-identical to the same design alone (the token transition's matmul config moves with M), and
        # the 31-seed grade puts the difference far under the seed floor (state/rfd4/bets/b128.md).
        self.max_parallel_samples = max_parallel_samples
        self._device = self.autoencoder.dev["decoder"].device
        from tt_bio.rfd4.pair import PairOnCard

        self.pair = (None if self.glue.pair_split is None
                     else PairOnCard(self.glue.pair_split, self._device))
        self._weights = path
        self._encoder = None

    @property
    def encoder(self):
        """The AE's encoder, brought up on first use.

        Only partial diffusion and inverse folding need it, and they are the minority of runs, so
        a generation from pure noise pays neither the host memory nor the device bring-up. The
        cost of asking later is one re-read of the bundle.
        """
        if self._encoder is None:
            from tt_bio.rfd4 import network as _network
            from tt_bio.rfd4.autoencoder import RFD4AutoEncoder

            # No adapter here: a LoRA adapts the trunk (`nn.`), and the encoder is the
            # `autoencoder.` half of the same bundle.
            state, _ = _network.load_checkpoint(self._weights)
            self._encoder = RFD4AutoEncoder(_network._sub(state, "autoencoder."), parts=("encoder",))
        return self._encoder

    @classmethod
    def from_pretrained(cls, weights: str | Path = DEFAULT_WEIGHTS, **kwargs) -> "DesignEngine":
        """Load the released bundle onto a card. Upstream spells this
        ``rfd4.Proteina.from_pretrained``; the arguments and the default checkpoint are the same."""
        return cls(weights, **kwargs)

    def design(self, features: dict, *, num_designs: int, nsteps: int, seed: int,
               rng: str = "per_design", partial_t_backbone: float | None = None,
               partial_t_latent: float | None = None, couple_partial_t: bool = False,
               partial_trajectory_fraction_steps: float | None = None,
               inverse_folding: bool = False,
               inverse_folding_method: str = "resampled_forward_marginal",
               transforms: "Sequence[Callable]" = (), guidance=None,
               save_trajectory_every: int = 0) -> DesignOutput:
        """`rng="per_design"` gives every design its own stream (`sampler.DesignRng`).
        `rng="upstream"` seeds the global stream with `seed` and draws in upstream's order, which
        is how a design is compared one to one against upstream's run of the same seed.

        `transforms` are per-replicate sampling transforms (`tt_bio.rfd4.replicates`): each one
        runs once per design, so a batch can cover a surface instead of repeating one condition.
        `guidance` is a metric-gradient callable (`tt_bio.rfd4.guidance`).

        The partial-diffusion and inverse-folding arguments are upstream's, with upstream's
        meanings and upstream's mutual exclusions (`sampler.resolve_partial_start`). Any of them
        starts the trajectory from the input structure rather than from pure noise, which needs
        the clean state `x_1` and so the AE's encoder.
        """
        import gc

        import torch

        from tt_bio._vendor.rfd4_prep._cif.model_masks import compute_flow_emit_masks
        from tt_bio.rfd4 import network as _network
        from tt_bio.rfd4 import sampler as _sampler
        from tt_bio.sample_chunks import resolve_sample_chunk_width
        from tt_bio.size_limits import is_alloc_refusal

        from tt_bio.rfd4.replicates import apply_transforms

        batch = preprocess(features)
        if transforms:
            batch = apply_transforms(batch, num_designs=num_designs, transforms=transforms,
                                     seed=seed)
        n_tokens = batch["padding_mask"].shape[1]
        spec = self.spec
        if nsteps != spec.nsteps:
            # Fewer steps is a different (worse) recipe, not a faster one; say which ran.
            logger.info("rfd4: %d steps instead of the recipe's %d", nsteps, spec.nsteps)
            spec = replace(spec, nsteps=nsteps)

        start = _sampler.resolve_partial_start(
            spec, partial_t_backbone=partial_t_backbone, partial_t_latent=partial_t_latent,
            couple_partial_t=couple_partial_t,
            partial_trajectory_fraction_steps=partial_trajectory_fraction_steps,
            inverse_folding=inverse_folding, inverse_folding_method=inverse_folding_method)
        x_1 = self.clean_sample(batch, num_designs=num_designs, seed=seed) if start.is_partial else None

        network = _network.FlowNetwork(self.glue, _CardTrunk(self.trunk, self._device, self.pair), batch,
                                       plan=_sampler.step_plan(spec, start))
        args = dict(n_designs=num_designs, n_tokens=n_tokens, seed=seed, rng_mode=rng,
                    emit_masks=compute_flow_emit_masks(batch), groups=_groups(batch),
                    coord_cond_mask=batch.get("coord_cond_mask"), start=start, x_1=x_1,
                    guidance=guidance, save_trajectory_every=save_trajectory_every)
        out = None
        cap = auto_width(n_tokens) if self.max_parallel_samples == "auto" else self.max_parallel_samples
        width = resolve_sample_chunk_width(num_designs, cap)
        if rng == "per_design" and guidance is None and width < num_designs:
            # The chunks ChunkedNetwork would run one after another, run as interleaved lanes:
            # same designs, but the host prepares one chunk while the card runs another.
            # Guidance sees the whole batch, so it stays on the chunker.
            try:
                out = _sampler.generate(spec, network, lane_width=width, **args)
                self.chunk_width = width
            except Exception as exc:  # noqa: BLE001 -- only a DRAM refusal falls through
                if not is_alloc_refusal(exc):
                    raise
                logger.info("rfd4: %d-wide lanes refused DRAM, narrowing in chunks instead", width)
            if out is None:  # the refused call's device tensors go before the chunker allocates
                gc.collect()
        if out is None:
            if rng == "upstream":
                torch.manual_seed(seed)
            chunked = _sampler.ChunkedNetwork(network, spec, max_parallel_samples=cap)
            out = _sampler.generate(spec, chunked, **args)
            #: The sample width the trajectory finished at: `max_parallel_samples`, or less where DRAM
            #: refused a chunk and the chunker halved it. A throughput number quotes this, not the cap.
            self.chunk_width = chunked.width
        logger.info("rfd4: %d designs, %d steps, %d trunk passes, %d at a time%s", num_designs,
                    start.nsteps or nsteps, out.trunk_passes, self.chunk_width,
                    f", from t={start.t}" if start.is_partial else "")

        if self.pair is not None:
            self.pair.release()  # the design's constant pair projection
        decoded = self.decode(batch, out, num_designs=num_designs)
        return DesignOutput(coords_nm=decoded["coors_nm"], residue_type=decoded["residue_type"],
                            atom_mask=decoded["atom_mask"],
                            insertion_logits=out.insertion_logits,
                            trajectory=out.trajectory)

    def clean_sample(self, batch: dict, *, num_designs: int, seed: int) -> dict:
        """`Proteina.add_clean_samples`: the input structure as the flow's clean state `x_1`.

        The backbone is the CA slot of the input coordinates. The latents are the AE encoder's
        draw, `mean + eps * exp(log_scale)` — upstream's encode is stochastic, so a parent
        structure does not have one latent, it has a distribution, and every design gets its own
        draw from it. The draw is seeded off the design seed, so a run reproduces.
        """
        import torch

        from tt_bio.rfd4 import network as _network
        from tt_bio.rfd4.sampler import BB_CA, LOCAL_LATENTS

        wide = _network._broadcast_batch(dict(batch), num_designs)
        encoded = self.encoder.encode(wide, noise=None)
        gen = torch.Generator().manual_seed(seed)
        eps = torch.randn(encoded["log_scale"].shape, generator=gen,
                          dtype=encoded["log_scale"].dtype)
        latents = (encoded["mean"] + eps * torch.exp(encoded["log_scale"])) * encoded["emit_mask"][..., None]
        return {BB_CA: wide["coords_nm"][:, :, 1, :], LOCAL_LATENTS: latents}

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


class _CardTrunk:
    """The trunk on one card, as `FlowNetwork` calls it: `launch` enqueues, `collect` reads s.
    With a `PairOnCard` it also builds z_init on the card."""

    def __init__(self, trunk, device, pair=None):
        self.trunk, self.device, self.pair = trunk, device, pair
        self.builds_pair = pair is not None

    def __call__(self, s_init, z_init, conds, mask):
        return self.collect(self.launch(s_init, z_init, conds, mask))

    def launch(self, s_init, z_init, conds, mask):
        return self.trunk.launch(s_init, z_init, conds, mask, self.device, pair=False,
                                 z_builder=self.pair)

    def collect(self, pending):
        return self.trunk.collect(pending)


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
    """The released bundle's directory, downloading it on first use.

    The default goes through the weights registry (pinned revision, `TT_BIO_RFD4` override,
    `tt-bio weights` sees it). Another hub id is snapshotted whole, at its HF_REVISIONS pin if any."""
    from tt_bio import weights as _w

    path = Path(str(weights)).expanduser()
    if path.exists():
        return path
    if str(weights) == DEFAULT_WEIGHTS:
        return _w.fetch("rfd4")
    return _w.fetch_hf_repo(str(weights))


def get_engine(name: str = "auto", **kwargs) -> Engine:
    """The engine to design with."""
    if name not in ("auto", "device"):
        raise ValueError(f"unknown RFD4 engine {name!r}")
    return DesignEngine(**kwargs)
