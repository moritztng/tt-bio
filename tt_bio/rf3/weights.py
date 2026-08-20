"""Load the RF3 torch reference model from an upstream checkpoint.

This is the reference the ttnn port is scored against, not the production path.
Production inference runs on Tenstorrent via ``tt_bio.rf3`` and only borrows the
featurizer from here.

Two things the checkpoint layout makes easy to get wrong:

- The weights live under ``ck["model"]`` with both a ``model.*`` and a ``shadow.*``
  copy. ``shadow`` is the EMA. Upstream's ``EMA.forward`` dispatches to ``shadow``
  whenever the module is not training, so **inference uses the EMA weights**;
  loading ``model.*`` silently gives a worse network that looks like a port bug.
- The architecture config is in ``ck["train_cfg"].model.net``, carrying a hydra
  ``_target_`` key that the constructor does not accept.
"""

from __future__ import annotations

import os
from typing import Any

import torch

#: What upstream instantiates for inference, per `ck["train_cfg"].model.net._target_`.
NET_TARGETS = {
    "rf3.model.RF3.RF3WithConfidence": "RF3WithConfidence",
    "rf3.model.RF3.RF3": "RF3",
}


def _strip(prefix: str, sd: dict) -> dict:
    out = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    if not out:
        raise KeyError(f"no weights under prefix {prefix!r}")
    return out


def load_reference(
    ckpt_path: str | os.PathLike,
    *,
    use_ema: bool = True,
    num_steps: int | None = None,
    device: str = "cpu",
) -> tuple[Any, dict]:
    """Build the RF3 torch reference and load the checkpoint into it.

    Args:
        ckpt_path: an upstream RF3 checkpoint, e.g. ``rf3_foundry_01_24_latest_remapped.ckpt``.
        use_ema: load the EMA (``shadow``) weights. This is what inference uses;
            pass False only to compare against the raw training weights.
        num_steps: override the sampler's timestep count.

    Returns:
        ``(net, cfg)``, with ``net`` in eval mode and ``cfg`` the resolved net config.
    """
    from omegaconf import OmegaConf

    from tt_bio._vendor.rf3.model import RF3 as rf3_model

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    net_cfg = OmegaConf.to_container(ck["train_cfg"].model.net, resolve=True)

    target = net_cfg.pop("_target_")
    if target not in NET_TARGETS:
        raise ValueError(f"unexpected net target {target!r}")
    cls = getattr(rf3_model, NET_TARGETS[target])

    if num_steps is not None:
        net_cfg["inference_sampler"]["num_timesteps"] = num_steps

    net = cls(**net_cfg)

    prefix = "shadow." if use_ema else "model."
    state = _strip(prefix, ck["model"])
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint does not match the model: {len(missing)} missing, "
            f"{len(unexpected)} unexpected (first: {(missing or unexpected)[:3]})"
        )

    net.eval().to(device)
    for p in net.parameters():
        p.requires_grad_(False)
    return net, net_cfg
