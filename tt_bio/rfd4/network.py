"""The flow network around the trunk: features in, model outputs out.

`tt_bio.rfd4.trunk` holds the part worth putting on a card, the 14 attention blocks and the 7
pair updates with their recycles, which is 99 % of a design's arithmetic. Everything around it
is one matmul deep and runs once per sampler step, not once per trunk pass:

    cond_factory + transition_c_1/2   time embeddings, CATH code, recycle_remaining -> [b, n, 256]
    init_repr_factory                 21 token features -> 897 -> [b, n, 768]
    pair_repr_builder                 10 pair features -> 107 -> [b, n, n, 384], AdaLN on z
    heads                             latents 32, CA 3, sequence 147, insertion bilinear

Those four are upstream's own modules, vendored under `tt_bio._vendor.rfd4_prep._cif`, so the
features the trunk reads are built by NVIDIA's code rather than a transcription of it. The census
(`state/rfd4/census/census.json`) prices them at under 2 % of a design, and the pair feature build
is a gather, so a device port of the glue would buy nothing and could only introduce drift.

:class:`FlowNetwork` is the :class:`tt_bio.rfd4.sampler.Network` the sampler drives: one call per
step, it owns the recycle loop, and the trunk it calls is injected. Pass the device trunk for a
real design, or a host reference for a parity test on a machine with no card.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

import torch

logger = logging.getLogger(__name__)

#: PEFT LoRA on the trunk's `to_qkv` and `to_out_node`: W = base + (alpha / r) B A.
LORA_ALPHA, LORA_RANK = 64, 32


def merge_lora(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Fold PEFT LoRA adapters back into their base weights, under upstream's own key names.

    The released flow checkpoint keeps the trunk's q/k/v and output projections as a PEFT
    wrapped layer: `...to_qkv.base_layer.weight` plus `lora_A.default` / `lora_B.default`.
    Upstream loads them through PEFT, which computes `base + (alpha / r) B A` on the fly. We
    merge once at load, so what the trunk sees is one weight per projection.
    """
    scale = LORA_ALPHA / LORA_RANK
    merged, drop = {}, set()
    for key, value in state_dict.items():
        if ".lora_A.default.weight" not in key:
            continue
        stem = key[: -len(".lora_A.default.weight")]
        b = state_dict[f"{stem}.lora_B.default.weight"]
        base = state_dict[f"{stem}.base_layer.weight"]
        merged[f"{stem}.weight"] = base + scale * (b.double() @ value.double()).to(base.dtype)
        drop |= {key, f"{stem}.lora_B.default.weight", f"{stem}.base_layer.weight"}
        for suffix in ("bias",):  # the base layer's bias is unchanged by the adapter
            if f"{stem}.base_layer.{suffix}" in state_dict:
                merged[f"{stem}.{suffix}"] = state_dict[f"{stem}.base_layer.{suffix}"]
                drop.add(f"{stem}.base_layer.{suffix}")
    if not merged:
        return dict(state_dict)
    out = {k: v for k, v in state_dict.items() if k not in drop}
    out.update(merged)
    logger.debug("merged %d LoRA adapters", len(merged))
    return out


def merge_adapter(state_dict: dict[str, torch.Tensor], adapter: str | Path, *,
                  prefix: str = "nn.") -> dict[str, torch.Tensor]:
    """Fold a standalone PEFT LoRA adapter directory into ``state_dict``.

    ``adapter`` is a PEFT export: ``adapter_config.json`` (rank, alpha, target modules) beside
    ``adapter_model.safetensors``, whose keys are the wrapped module path plus ``lora_A``/``lora_B``
    and carry no model prefix, so ``prefix`` is prepended to reach the base weight.

    Merging the bundle's own ``adapter/`` onto its ``base/`` reproduces NVIDIA's merged
    ``model.safetensors`` on all 28 adapted projections to 3e-08 max absolute difference, which is
    fp32 rounding on weights of order 0.5.
    """
    from safetensors.torch import load_file

    adapter = Path(adapter).expanduser()
    config = json.loads((adapter / "adapter_config.json").read_text())
    scale = config["lora_alpha"] / config["r"]
    weights = load_file(str(adapter / "adapter_model.safetensors"))

    out = dict(state_dict)
    for key, a in weights.items():
        if not key.endswith(".lora_A.weight"):
            continue
        stem = key[: -len(".lora_A.weight")]
        b = weights[f"{stem}.lora_B.weight"]
        target = f"{prefix}{stem}.weight"
        if target not in out:
            raise KeyError(f"{adapter.name} adapts {stem}, which the checkpoint does not have "
                           f"as {target}")
        base = out[target]
        out[target] = base + scale * (b.double() @ a.double()).to(base.dtype)
    logger.info("merged %d LoRA adapters from %s", sum(k.endswith(".lora_A.weight") for k in weights),
                adapter)
    return out


def load_checkpoint(path: str | Path, adapter: str | Path | None = None
                    ) -> tuple[dict[str, torch.Tensor], dict]:
    """`(state_dict, config)` from the released bundle, with the trunk's LoRA already merged.

    The bundle ships the same weights twice: `model.safetensors`, NVIDIA's own export with the
    adapters already folded in, and `flow_checkpoints/rfd4_proteina_flow-ema.ckpt`, the training
    checkpoint that still carries them. The export is preferred, because merging ourselves lands
    within one fp32 ULP of it on the 28 adapted projections and over 14 layers and 6 recycles that
    ULP grows into a 2.5e-7 difference in the model's output. Taking upstream's own numbers keeps
    a design bit-comparable with upstream's.

    `path` is the bundle directory or a checkpoint file; the config comes from the bundle's
    `config.json`, which is what upstream builds the model from. `adapter` names a PEFT LoRA
    directory to fold in on top, for a fine-tune of your own; the bundle's `base/` plus its
    `adapter/` is the released model itself, which is how that path is graded.
    """
    path = Path(path).expanduser()
    if path.is_dir():
        config = json.loads((path / "config.json").read_text())
        export = path / "model.safetensors"
        if export.is_file():
            from safetensors.torch import load_file

            state = load_file(str(export))
            return (state if adapter is None else merge_adapter(state, adapter)), config["config"]
        ckpt = path / "flow_checkpoints" / "rfd4_proteina_flow-ema.ckpt"
    else:
        ckpt = path
        config = json.loads((path.parent.parent / "config.json").read_text())
    state = merge_lora(torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"])
    return (state if adapter is None else merge_adapter(state, adapter)), config["config"]


def _sub(state_dict: dict, prefix: str) -> dict:
    return {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}


class FlowGlue(torch.nn.Module):
    """Upstream's feature factories, conditioning transitions and heads, on the host in fp32.

    Built from the released config so the feature list is the checkpoint's, not a repo example's:
    the release is the 14-layer 768-wide net with `pair_repr_dim` 384, and its `feats_seq` is 21
    features wide. Norm type is set before the modules are built, as upstream does, because
    `make_norm` reads a module global.
    """

    def __init__(self, state_dict: dict, config: dict):
        super().__init__()
        from tt_bio._vendor.rfd4_prep._cif.nn_feature_factory import FeatureFactory
        from tt_bio._vendor.rfd4_prep._cif.nn_modules_norm import make_norm, set_norm_type
        from tt_bio._vendor.rfd4_prep._cif.nn_modules_pair_rep_initial import PairReprBuilder
        from tt_bio._vendor.rfd4_prep._cif.nn_modules_seq_transition_af3 import Transition

        cfg = dict(config["nn"])
        # The latent width is the autoencoder's, which is why upstream passes it in rather than
        # reading it off the nn config (`proteina.py:421`).
        cfg.setdefault("latent_dim", config["autoencoder_config"]["nn_ae"]["latent_z_dim"])
        set_norm_type(cfg.get("norm_type"))
        self.cfg = cfg
        token_dim, dim_cond = cfg["token_dim"], cfg["dim_cond"]

        self.init_repr_factory = FeatureFactory(feats=cfg["feats_seq"], dim_feats_out=token_dim,
                                                use_ln_out=False, mode="seq", **cfg)
        self.cond_factory = FeatureFactory(feats=cfg["feats_cond_seq"], dim_feats_out=dim_cond,
                                           use_ln_out=False, mode="seq", **cfg)
        self.condition_on_recycle_remaining = "recycle_remaining" in (cfg["feats_cond_seq"] or [])
        self.transition_c_1 = Transition(dim_cond, expansion_factor=2)
        self.transition_c_2 = Transition(dim_cond, expansion_factor=2)
        self.pair_repr_builder = PairReprBuilder(feats_repr=cfg["feats_pair_repr"],
                                                 feats_cond=cfg["feats_pair_cond"],
                                                 dim_feats_out=cfg["pair_repr_dim"],
                                                 dim_cond_pair=dim_cond, **cfg)
        self.local_latents_linear = torch.nn.Sequential(
            make_norm(token_dim), torch.nn.Linear(token_dim, cfg["latent_dim"], bias=False))
        self.ca_linear = torch.nn.Sequential(
            make_norm(token_dim), torch.nn.Linear(token_dim, 3, bias=False))
        readout = cfg.get("sequence_readout") or {}
        self.sequence_readout_enabled = bool(readout.get("enabled", False))
        if self.sequence_readout_enabled:
            self.sequence_readout = torch.nn.Sequential(
                make_norm(token_dim),
                torch.nn.Linear(token_dim, readout.get("num_classes", 147), bias=False))
        self.insertion_q_proj = torch.nn.Linear(token_dim, token_dim, bias=False)
        self.insertion_k_proj = torch.nn.Linear(token_dim, token_dim, bias=False)
        self._insertion_head_scale = token_dim ** -0.5

        self.load_state_dict(_sub(state_dict, "nn."), strict=False)
        self.eval()
        self.requires_grad_(False)

    # -- the three pieces a step needs, each called once per network call ------------------

    def conditioning(self, batch: dict, recycle_remaining: int | None) -> torch.Tensor:
        """`[b, n, dim_cond]`: time embeddings, CATH code and the remaining-pass count."""
        mask = batch["padding_mask"]
        c = self.cond_factory({**batch, "recycle_remaining": recycle_remaining})
        return self.transition_c_2(self.transition_c_1(c, mask), mask)

    def initial_states(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """`(s_init, z_init)` for this step's noisy state."""
        mask = batch["padding_mask"]
        s_init = self.init_repr_factory(batch) * mask[..., None]
        return s_init, self.pair_repr_builder(batch)

    def heads(self, s: torch.Tensor, batch: dict) -> dict[str, dict[str, torch.Tensor]]:
        """The four readouts. `emit_masks` decide which tokens a modality may move."""
        from tt_bio._vendor.rfd4_prep._cif.model_masks import compute_flow_emit_masks

        emit = compute_flow_emit_masks(batch)
        param = self.cfg["output_parameterization"]
        out = {
            "bb_ca": {param["bb_ca"]: self.ca_linear(s) * emit["bb_ca"][..., None]},
            "local_latents": {param["local_latents"]:
                              self.local_latents_linear(s) * emit["local_latents"][..., None]},
        }
        if self.sequence_readout_enabled:
            out["sequence"] = {"logits": self.sequence_readout(s) * batch["padding_mask"][..., None]}
        q, k = self.insertion_q_proj(s), self.insertion_k_proj(s)
        out["insertion"] = {"logits": torch.bmm(q, k.transpose(1, 2)) * self._insertion_head_scale}
        return out


class FlowNetwork:
    """One network call per sampler step: features, the trunk at this step's recycle depth, heads.

    `trunk` is `(s_init, z_init, conds, mask) -> (s, z)`, where `conds` carries one conditioning
    tensor per recycle because upstream rebuilds it each cycle from the remaining-pass count. The
    device trunk (`tt_bio.rfd4.trunk.Trunk.run_bucketed`) and a host reference have the same shape,
    so a parity test and a real design run the same code around them. A trunk that also has
    `launch(...)` and `collect(pending)` is enqueued by :meth:`launch` and read by :meth:`finish`,
    which is what lets `sampler.run_lanes` keep the host busy while the card runs.

    The batch the glue reads is the featurized example (`tt_bio.rfd4.io.featurize`) with this
    step's noisy state written in, exactly as upstream's sampler writes `x_t` and `t` into the
    batch it hands the network.
    """

    def __init__(self, glue: FlowGlue, trunk: Callable, batch: dict, *,
                 condition_on_recycle_remaining: bool = True):
        self.glue = glue
        self.trunk = trunk
        self.batch = batch
        self.condition_on_recycle_remaining = condition_on_recycle_remaining
        self.passes = 0

    def __call__(self, x_t: dict[str, Any], t: dict[str, float], *, recycle_depth: int,
                 step: int) -> dict[str, dict[str, Any]]:
        return self.finish(self.launch(self.prepare(x_t, t, recycle_depth=recycle_depth, step=step)))

    # The call in three stages, for `sampler.run_lanes`: the host builds the trunk's inputs, the
    # trunk is enqueued, and the heads read its output once it is back.

    def prepare(self, x_t: dict[str, Any], t: dict[str, float], *, recycle_depth: int, step: int):
        n_designs = next(iter(x_t.values())).shape[0]
        batch = dict(self.batch)
        batch["x_t"] = dict(x_t)
        batch["t"] = {mode: torch.full((n_designs,), float(value)) for mode, value in t.items()}
        batch = _broadcast_batch(batch, n_designs)

        depth = max(1, int(recycle_depth))
        condition = self.condition_on_recycle_remaining and self.glue.condition_on_recycle_remaining
        with torch.no_grad():
            s_init, z_init = self.glue.initial_states(batch)
            conds = [self.glue.conditioning(batch, None)]
            for cycle in range(1, depth):
                conds.append(self.glue.conditioning(batch, depth - cycle - 1) if condition
                             else conds[0])
        return batch, depth, (s_init, z_init, conds, batch["padding_mask"].float())

    def launch(self, prepared):
        """Start the trunk. A trunk with a `launch` only enqueues here; a plain callable runs."""
        batch, depth, inputs = prepared
        launch = getattr(self.trunk, "launch", None)
        return batch, depth, launch(*inputs) if launch else self.trunk(*inputs)

    def finish(self, launched) -> dict[str, dict[str, Any]]:
        batch, depth, pending = launched
        s, _z = self.trunk.collect(pending) if hasattr(self.trunk, "collect") else pending
        self.passes += depth
        with torch.no_grad():
            return self.glue.heads(s, batch)


def _broadcast_batch(batch: dict, n_designs: int) -> dict:
    """Give every batched tensor the design axis the sampler runs at.

    `featurize` returns one example with a leading axis of 1. A batch of designs shares that one
    input: the conditions, the contig and the target are the same, only the noise differs. So the
    features are expanded, not recomputed, which is also what upstream's dataloader does with
    `num_replicates`.
    """
    if n_designs == 1:
        return batch
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.dim() > 0 and value.shape[0] == 1:
            out[key] = value.expand(n_designs, *value.shape[1:])
        elif isinstance(value, dict):
            out[key] = _broadcast_batch(value, n_designs)
        else:
            out[key] = value
    return out
