"""AlphaFold2 (`model_1_ptm`) on ttnn: the blocks that are AF2's and nobody else's.

Everything AF2 shares with the four models already in `tenstorrent.py` -- the triangle
multiplication, the triangle attention, the outer product mean -- is that module's class, driven
through its constructor flags. What lives here is the two things AF2 does differently:

* `ReluTransition`. Every other model in the repo has a SwiGLU transition; AF2 has LayerNorm,
  linear, ReLU, linear. One caller, so it does not belong on a hot shared file.
* `AF2PairBlock`. AF2's pair track in AF2's order, with no single representation and no
  attention-pair-bias. It deliberately does NOT subclass `PairformerLayer`: fitting AF2 into
  that class needs three new constructor hooks (transition class, transition scope, bias
  plumbing) on a class four other models run through, and the reuse would end at the MSA track,
  which `PairformerLayer` does not model at all.

The reference is `tt_bio/af2_reference.py`, scored against a captured JAX run by
`scripts/af2_port/tap_gate.py`. This file is scored against the reference's own activations by
`scripts/af2_port/device_gate.py`.

**The masks.** AF2 builds `mask_2d` from `seq_mask`, and PXDesign folds every residue it is
given, so `mask_2d` is all ones for every fold this port serves and both mask arguments are
None. That is not a shortcut that can be relaxed silently: AF2's triangle multiplication masks
BOTH halves of the fused projection (`mask * p_in(x)` before the split) where
`TriangleMultiplication` masks only the `a` half, so a genuinely masked AF2 fold needs that
difference resolved first.
"""
from __future__ import annotations

import ttnn

from .tenstorrent import (
    PAIR_ROW_BLOCK,
    Module,
    TriangleAttention,
    TriangleMultiplication,
    Weights,
    get_device,
)

# The pair-track shape constants of `model_1_ptm`, from the checkpoint's own config
# (`scripts/af2_port/af2ig_model_config.json`).
C_Z = 128
TRI_MUL_HIDDEN = 128
TRI_ATT_HEADS = 4
TRI_ATT_HEAD_DIM = 32
PAIR_TRANSITION_FACTOR = 4

# Row-block the transition once its hidden activation would be the biggest tensor in the block.
# `factor` is 4, so the hidden is 4 pair tensors; at 208 tokens that is 88 MB and fits, and the
# block is row-local (LayerNorm over channels, two matmuls over channels), so blocking changes
# nothing a row computes.
TRANSITION_ROWBLOCK_BYTES = 256 * 2 ** 20


def compute_kernel_config() -> ttnn.DeviceComputeKernelConfig:
    """The repo's trunk kernel config: HiFi4 with an fp32 accumulator, per part."""
    device = get_device()
    cls = (ttnn.types.WormholeComputeKernelConfig
           if device.arch() == ttnn.Arch.WORMHOLE_B0
           else ttnn.types.BlackholeComputeKernelConfig)
    return cls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
               fp32_dest_acc_en=True, packer_l1_acc=True)


class ReluTransition(Module):
    """AF2's `Transition`: LayerNorm, linear, ReLU, linear, all three with a bias.

    The ReLU is fused into the first matmul's pack, so the expanded hidden is written once.
    """

    def __init__(self, state_dict: Weights,
                 compute_kernel_config: ttnn.DeviceComputeKernelConfig):
        super().__init__(state_dict, compute_kernel_config)
        self.norm_weight = self.torch_to_tt("norm.weight")
        self.norm_bias = self.torch_to_tt("norm.bias")
        self.fc1_weight = self.torch_to_tt("fc1.weight")
        self.fc1_bias = self.torch_to_tt("fc1.bias")
        self.fc2_weight = self.torch_to_tt("fc2.weight")
        self.fc2_bias = self.torch_to_tt("fc2.bias")

    def _rows(self, x: ttnn.Tensor) -> ttnn.Tensor:
        xn = ttnn.layer_norm(
            x, weight=self.norm_weight, bias=self.norm_bias, epsilon=1e-5,
            compute_kernel_config=self.compute_kernel_config,
        )
        h = self._lin(xn, self.fc1_weight, bias=self.fc1_bias, activation="relu",
                      memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(xn)
        out = self._lin(h, self.fc2_weight, bias=self.fc2_bias,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(h)
        return out

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        shape = [int(d) for d in x.shape]
        hidden_bytes = 1
        for d in shape[:-1]:
            hidden_bytes *= d
        hidden_bytes *= int(self.fc1_weight.shape[-1]) * 2
        if hidden_bytes <= TRANSITION_ROWBLOCK_BYTES:
            return self._rows(x)
        rows = shape[-3]
        blocks = [self._rows(x[:, s:min(s + PAIR_ROW_BLOCK, rows)])
                  for s in range(0, rows, PAIR_ROW_BLOCK)]
        out = ttnn.concat(blocks, dim=-3)
        for b in blocks:
            ttnn.deallocate(b)
        return out


class AF2PairBlock(Module):
    """AF2's pair track: two triangle multiplications, two triangle attentions, a transition.

    `evoformer_order=False` is the template pair stack, which runs the attentions before the
    multiplications (`modules.py:212-241` against `modules.py:1330-1356`).

    `scale_pair_bias=False, fp32_softmax=True` on both attentions, which is openfold3's
    combination for the identical reference convention: AF2 wants `softmax(qk / sqrt(d) + b)` with
    the pair bias raw (`af2_reference.Attention._attend`), and the fp32-softmax path adds it raw
    after scaling the logits. The two flags are coupled, measured on card at Evoformer block 0
    against the reference's own activations (rms against the torch bf16 arm, with the torch fp32
    arm on the same input as the envelope):

        arm                                  tri_att_start   tri_att_end
        fused SDPA, bias unscaled                  9.0x           53x
        fused SDPA, bias pre-scaled by sqrt(d)     9.0x          21.3x
        fp32 softmax, bias raw                     2.75x          2.78x

    The fused SDPA computes `softmax(scale * (qk + attn_mask))`, so an unscaled bias arrives
    sqrt(32) = 5.66x too flat -- that is the 53x column. Pre-scaling fixes the convention but
    leaves the softmax reduction in bfloat16, and at AF2's 4-head pair bias that alone is worth
    3.4-7.7x. The fused triangle-attention kernel (K2) is bit-identical on this shape either way,
    so it is not implicated.

    `gated_move=False` because the fused chunk+gate forward move takes no bias.
    """

    def __init__(
        self,
        state_dict: Weights,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
        head_dim: int = TRI_ATT_HEAD_DIM,
        n_heads: int = TRI_ATT_HEADS,
        evoformer_order: bool = True,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.evoformer_order = evoformer_order
        self.tri_mul_out = TriangleMultiplication(
            False, self.scope("tri_mul_out"), compute_kernel_config)
        self.tri_mul_in = TriangleMultiplication(
            True, self.scope("tri_mul_in"), compute_kernel_config)
        self.tri_att_start = TriangleAttention(
            head_dim, n_heads, False, self.scope("tri_att_start"), compute_kernel_config,
            scale_pair_bias=False, fp32_softmax=True)
        self.tri_att_end = TriangleAttention(
            head_dim, n_heads, True, self.scope("tri_att_end"), compute_kernel_config,
            scale_pair_bias=False, fp32_softmax=True)
        self.pair_transition = ReluTransition(
            self.scope("pair_transition"), compute_kernel_config)

    def __call__(self, z: ttnn.Tensor, mask: ttnn.Tensor | None = None,
                 attn_mask: ttnn.Tensor | None = None) -> ttnn.Tensor:
        order = ((self.tri_mul_out, mask), (self.tri_mul_in, mask),
                 (self.tri_att_start, attn_mask), (self.tri_att_end, attn_mask))
        if not self.evoformer_order:
            order = order[2:] + order[:2]
        for module, module_mask in order + ((self.pair_transition, None),):
            update = (module(z) if module is self.pair_transition
                      else module(z, module_mask))
            z = ttnn.add_(z, update)
            ttnn.deallocate(update)
        return z
