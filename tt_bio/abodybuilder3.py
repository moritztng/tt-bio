"""ABodyBuilder3 ttnn port — structure-module components on device.

ABodyBuilder3 (Exscientia, Apache-2.0; vendored under ``tt_bio/_vendor/abodybuilder3``)
is a single, MSA-free, one-hot antibody Fv structure module: 8 invariant-point-
attention (IPA) update blocks + a single pLDDT head. This module ports the
structure module to ttnn, reusing the shared ``tt_bio.tenstorrent`` primitives.

Scope of this file (first ttnn chunk): the standard, non-novel structure-module
submodules — the post-IPA LayerNorm, the single-representation Transition, the
BackboneUpdate linear, the AngleResnet, and the pLDDT head — ported to ttnn and
PCC-gated against the reference golden (``scripts/abb3_golden.py``). These are
all Linear/LayerNorm/ReLU/residual ops that map directly onto ``ttnn.linear`` /
``ttnn.layer_norm`` / ``ttnn.relu``.

The genuinely novel op — InvariantPointAttention's per-residue rigid-frame point
rotation, N^2*H*P_q squared-distance logits, value-point aggregation, and
``invert_apply`` under ttnn's tiled shape/layout constraints — is the long pole
of the port and lands in a follow-on chunk; it has no reusable primitive in tt-bio
(tt-bio's ESMFold2 is a diffusion folder, not an AF2 structure module).

The AngleResnet's final size-2 torsion normalization (sqrt(sum(s**2, -1)+eps),
divide) is a tiny, cheap op on an awkward (<32) trailing dim, so it is kept as a
host-side fp32 tail (``normalize_angles``) — the same "keep cheap host code as-is"
boundary the ESMFold2 port uses for its confidence head.
"""
from __future__ import annotations

import math

import torch
import ttnn

from tt_bio.tenstorrent import (
    Module,
    WeightScope,
    Weights,
    CORE_GRID_MAIN,
    _dtype,
    get_device,
)


def abb3_compute_kernel_config() -> ttnn.DeviceComputeKernelConfig:
    """HiFi4 + fp32 dest accumulation for the ABodyBuilder3 linears — the same
    config the other tt-bio models use, so bf16 weights still accumulate in fp32
    (parity-safe for the 128/256-channel linears)."""
    dev = get_device()
    kernel_cls = (
        ttnn.types.WormholeComputeKernelConfig
        if dev.arch() == ttnn.Arch.WORMHOLE_B0
        else ttnn.types.BlackholeComputeKernelConfig
    )
    return kernel_cls(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def _from_torch(x: torch.Tensor) -> ttnn.Tensor:
    return ttnn.from_torch(
        x, layout=ttnn.TILE_LAYOUT, device=get_device(), dtype=_dtype(),
    )


def _to_torch(x: ttnn.Tensor) -> torch.Tensor:
    return ttnn.to_torch(x).to(torch.float32)


class IPALayerNorm(Module):
    """The LayerNorm applied after each IPA block (c=embed_dim=128)."""

    def __init__(self, state_dict: Weights, ck: ttnn.DeviceComputeKernelConfig):
        super().__init__(state_dict, ck)
        self.weight = self.torch_to_tt("weight")
        self.bias = self.torch_to_tt("bias")

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.layer_norm(
            x, weight=self.weight, bias=self.bias, epsilon=1e-5,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )


class BackboneUpdate(Module):
    """BackboneUpdate (Alg. 23 part): a single Linear(c_s -> 6) producing the
    per-residue quaternion-translation update vector."""

    def __init__(self, state_dict: Weights, ck: ttnn.DeviceComputeKernelConfig):
        super().__init__(state_dict, ck)
        self.weight = self.torch_to_tt("linear.weight")
        self.bias = self.torch_to_tt("linear.bias")

    def __call__(self, s: ttnn.Tensor) -> ttnn.Tensor:
        return self._lin(
            s, self.weight, bias=self.bias, dtype=_dtype(),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )


class PLDDTHead(Module):
    """PerResidueLDDTCaPredictor: LayerNorm -> Linear -> ReLU -> Linear -> ReLU ->
    Linear(no_bins). Returns [*, N, 50] pLDDT logits."""

    def __init__(self, state_dict: Weights, ck: ttnn.DeviceComputeKernelConfig):
        super().__init__(state_dict, ck)
        self.norm_weight = self.torch_to_tt("layer_norm.weight")
        self.norm_bias = self.torch_to_tt("layer_norm.bias")
        self.l1_weight = self.torch_to_tt("linear_1.weight")
        self.l1_bias = self.torch_to_tt("linear_1.bias")
        self.l2_weight = self.torch_to_tt("linear_2.weight")
        self.l2_bias = self.torch_to_tt("linear_2.bias")
        self.l3_weight = self.torch_to_tt("linear_3.weight")
        self.l3_bias = self.torch_to_tt("linear_3.bias")

    def __call__(self, s: ttnn.Tensor) -> ttnn.Tensor:
        x = ttnn.layer_norm(
            s, weight=self.norm_weight, bias=self.norm_bias, epsilon=1e-5,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        x = self._lin(x, self.l1_weight, bias=self.l1_bias, dtype=_dtype(),
                      memory_config=ttnn.L1_MEMORY_CONFIG)
        x = ttnn.relu(x, memory_config=ttnn.L1_MEMORY_CONFIG)
        x = self._lin(x, self.l2_weight, bias=self.l2_bias, dtype=_dtype(),
                      memory_config=ttnn.L1_MEMORY_CONFIG)
        x = ttnn.relu(x, memory_config=ttnn.L1_MEMORY_CONFIG)
        x = self._lin(x, self.l3_weight, bias=self.l3_bias, dtype=_dtype(),
                      memory_config=ttnn.L1_MEMORY_CONFIG)
        return x


class StructureModuleTransition(Module):
    """StructureModuleTransition (Alg. 23 lines 8-9): one TransitionLayer
    (Linear->ReLU->Linear->ReLU->Linear + residual) then dropout (identity at
    eval) + LayerNorm. no_transition_layers=1 for the pldt-loss checkpoint."""

    def __init__(self, state_dict: Weights, ck: ttnn.DeviceComputeKernelConfig):
        super().__init__(state_dict, ck)
        self.norm_weight = self.torch_to_tt("layer_norm.weight")
        self.norm_bias = self.torch_to_tt("layer_norm.bias")
        # Single transition layer (no_transition_layers=1).
        self.l1_weight = self.torch_to_tt("layers.0.linear_1.weight")
        self.l1_bias = self.torch_to_tt("layers.0.linear_1.bias")
        self.l2_weight = self.torch_to_tt("layers.0.linear_2.weight")
        self.l2_bias = self.torch_to_tt("layers.0.linear_2.bias")
        self.l3_weight = self.torch_to_tt("layers.0.linear_3.weight")
        self.l3_bias = self.torch_to_tt("layers.0.linear_3.bias")

    def __call__(self, s: ttnn.Tensor) -> ttnn.Tensor:
        s_initial = s
        x = self._lin(s, self.l1_weight, bias=self.l1_bias, dtype=_dtype(),
                      memory_config=ttnn.L1_MEMORY_CONFIG)
        x = ttnn.relu(x, memory_config=ttnn.L1_MEMORY_CONFIG)
        x = self._lin(x, self.l2_weight, bias=self.l2_bias, dtype=_dtype(),
                      memory_config=ttnn.L1_MEMORY_CONFIG)
        x = ttnn.relu(x, memory_config=ttnn.L1_MEMORY_CONFIG)
        x = self._lin(x, self.l3_weight, bias=self.l3_bias, dtype=_dtype(),
                      memory_config=ttnn.L1_MEMORY_CONFIG)
        x = ttnn.add(x, s_initial, memory_config=ttnn.L1_MEMORY_CONFIG)
        # dropout is identity at eval; final LayerNorm.
        return ttnn.layer_norm(
            x, weight=self.norm_weight, bias=self.norm_bias, epsilon=1e-5,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )


class AngleResnet(Module):
    """AngleResnet (Alg. 20 lines 11-14), use_original_sm=True path:
    relu+linear_initial on s_initial, relu+linear_in on s, sum, then 2 resnet
    blocks (relu+linear_2, relu+linear_3, +residual), relu, linear_out -> [*, 14].

    Returns the UNNORMALIZED [*, N, 14] tensor on device. The final size-2 torsion
    normalization is a tiny host-side fp32 tail (``normalize_angles``) — the
    awkward <32 trailing dim is not worth a tiled device reduction, mirroring the
    ESMFold2 "cheap host code" boundary."""

    def __init__(self, state_dict: Weights, ck: ttnn.DeviceComputeKernelConfig,
                 epsilon: float = 1e-7):
        super().__init__(state_dict, ck)
        self.eps = epsilon
        self.lin_in_weight = self.torch_to_tt("linear_in.weight")
        self.lin_in_bias = self.torch_to_tt("linear_in.bias")
        self.lin_initial_weight = self.torch_to_tt("linear_initial.weight")
        self.lin_initial_bias = self.torch_to_tt("linear_initial.bias")
        self.b0_l2_weight = self.torch_to_tt("layers.0.linear_2.weight")
        self.b0_l2_bias = self.torch_to_tt("layers.0.linear_2.bias")
        self.b0_l3_weight = self.torch_to_tt("layers.0.linear_3.weight")
        self.b0_l3_bias = self.torch_to_tt("layers.0.linear_3.bias")
        self.b1_l2_weight = self.torch_to_tt("layers.1.linear_2.weight")
        self.b1_l2_bias = self.torch_to_tt("layers.1.linear_2.bias")
        self.b1_l3_weight = self.torch_to_tt("layers.1.linear_3.weight")
        self.b1_l3_bias = self.torch_to_tt("layers.1.linear_3.bias")
        self.out_weight = self.torch_to_tt("linear_out.weight")
        self.out_bias = self.torch_to_tt("linear_out.bias")

    def _block(self, s, l2_w, l2_b, l3_w, l3_b):
        a = ttnn.relu(s, memory_config=ttnn.L1_MEMORY_CONFIG)
        a = self._lin(a, l2_w, bias=l2_b, dtype=_dtype(),
                      memory_config=ttnn.L1_MEMORY_CONFIG)
        a = ttnn.relu(a, memory_config=ttnn.L1_MEMORY_CONFIG)
        a = self._lin(a, l3_w, bias=l3_b, dtype=_dtype(),
                      memory_config=ttnn.L1_MEMORY_CONFIG)
        return ttnn.add(a, s, memory_config=ttnn.L1_MEMORY_CONFIG)

    def __call__(self, s: ttnn.Tensor, s_initial: ttnn.Tensor) -> ttnn.Tensor:
        si = ttnn.relu(s_initial, memory_config=ttnn.L1_MEMORY_CONFIG)
        si = self._lin(si, self.lin_initial_weight, bias=self.lin_initial_bias,
                       dtype=_dtype(), memory_config=ttnn.L1_MEMORY_CONFIG)
        x = ttnn.relu(s, memory_config=ttnn.L1_MEMORY_CONFIG)
        x = self._lin(x, self.lin_in_weight, bias=self.lin_in_bias,
                      dtype=_dtype(), memory_config=ttnn.L1_MEMORY_CONFIG)
        x = ttnn.add(x, si, memory_config=ttnn.L1_MEMORY_CONFIG)
        x = self._block(x, self.b0_l2_weight, self.b0_l2_bias,
                        self.b0_l3_weight, self.b0_l3_bias)
        x = self._block(x, self.b1_l2_weight, self.b1_l2_bias,
                        self.b1_l3_weight, self.b1_l3_bias)
        x = ttnn.relu(x, memory_config=ttnn.L1_MEMORY_CONFIG)
        x = self._lin(x, self.out_weight, bias=self.out_bias, dtype=_dtype(),
                      memory_config=ttnn.L1_MEMORY_CONFIG)
        return x  # [*, N, 14] unnormalized


def normalize_angles(unnorm: torch.Tensor, epsilon: float = 1e-7) -> torch.Tensor:
    """Host-side fp32 tail for AngleResnet: reshape [*, N, 14] -> [*, N, 7, 2] and
    L2-normalize over the last (size-2) dim, matching the reference."""
    s = unnorm.view(unnorm.shape[:-1] + (-1, 2))
    norm_denom = torch.sqrt(
        torch.clamp(torch.sum(s ** 2, dim=-1, keepdim=True), min=epsilon)
    )
    return s / norm_denom
