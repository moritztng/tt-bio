# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, Institute for Protein Design, University of Washington
# Full copyright, license conditions and disclaimer: THIRD_PARTY_LICENSES.txt.
# Adapted from RFD4-Proteina 2db07bb444d91695f8c10e6acb7648e9297f9221.
"""Config-selectable LayerNorm/RMSNorm factory for the score-network trunk."""

import logging

import torch
from torch.nn import functional as F

logger = logging.getLogger(__name__)

# Process-global norm selection, set once before the model is built. Assumes
# single-threaded model construction (set_norm_type is called before make_norm).
_NORM_TYPE = "layer_norm"


def _rms_norm_preserve_dtype(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    """Apply RMSNorm and cast back to the input dtype."""
    return F.rms_norm(x, (x.shape[-1],), weight, eps).to(x.dtype)


_compiled_rms_norm = torch.compile(_rms_norm_preserve_dtype, fullgraph=True, dynamic=True)


class CompiledRMSNorm(torch.nn.RMSNorm):
    """RMSNorm that preserves activation dtype and fuses the output cast on CUDA."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eps = self.eps if self.eps is not None else torch.finfo(x.dtype).eps
        if x.device.type == "cuda" and not torch.compiler.is_compiling():
            return _compiled_rms_norm(x, self.weight, eps)
        return _rms_norm_preserve_dtype(x, self.weight, eps)


def set_norm_type(norm_type: str | None) -> None:
    """Set the process-wide norm implementation used by ``make_norm``."""
    global _NORM_TYPE
    _NORM_TYPE = norm_type or "layer_norm"
    if _NORM_TYPE == "rmsnorm_torch_fused":
        # Sequence, pair, and attention layouts specialize separately in grad and no-grad mode.
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 32)
    logger.info(f"Score-network norm type: {_NORM_TYPE!r}")


def is_identity_mask(mask: torch.Tensor, x: torch.Tensor) -> bool:
    """True when ``x * mask`` returns ``x`` bit for bit: every entry is 1 and the product keeps
    ``x``'s dtype. A design without padding is the common case, and there each pair-sized multiply
    by the mask is a full pass over the pair for nothing."""
    return torch.result_type(x, mask) == x.dtype and bool((mask == 1).all())


def make_norm(
    normalized_shape: int,
    *,
    eps: float = 1e-5,
    elementwise_affine: bool = True,
    bias: bool = True,
) -> torch.nn.Module:
    """Build a norm layer of the configured type; signature mirrors :py:class:`torch.nn.LayerNorm`.

    ``bias`` is ignored by the RMSNorm variants (RMSNorm has no bias by construction).
    """
    if _NORM_TYPE == "layer_norm":
        return torch.nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine, bias=bias)
    if _NORM_TYPE == "rmsnorm_torch":
        return torch.nn.RMSNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
    if _NORM_TYPE == "rmsnorm_torch_fused":
        return CompiledRMSNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
    if _NORM_TYPE == "rmsnorm_apex":
        # Fail loudly rather than silently falling back to a slower kernel.
        from apex.normalization import FusedRMSNorm

        return FusedRMSNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
    raise ValueError(
        f"Unknown norm_type {_NORM_TYPE!r}; expected layer_norm, rmsnorm_torch, rmsnorm_torch_fused, or rmsnorm_apex."
    )
