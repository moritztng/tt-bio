"""AlphaFold2 (`model_1_ptm`) on ttnn: the blocks that are AF2's and nobody else's.

Everything AF2 shares with the four models already in `tenstorrent.py` -- the triangle
multiplication, the triangle attention, the outer product mean -- is that module's class, driven
through its constructor flags. What lives here is what AF2 does differently:

* `ReluTransition`. Every other model in the repo has a SwiGLU transition; AF2 has LayerNorm,
  linear, ReLU, linear. One caller, so it does not belong on a hot shared file.
* `AF2PairBlock`. AF2's pair track in AF2's order, with no single representation and no
  attention-pair-bias. It deliberately does NOT subclass `PairformerLayer`: fitting AF2 into
  that class needs three new constructor hooks (transition class, transition scope, bias
  plumbing) on a class four other models run through, and the reuse would end at the MSA track,
  which `PairformerLayer` does not model at all.
* `AF2Attention`. The MSA track's two attention users in one class. Its row variant is the
  shared `TriangleAttention` with the bias taken from a second tensor through a second
  LayerNorm, and its column variant softmaxes over the MSA depth, which needs a compute kernel
  config the shared fp32-softmax helper does not pass.
* `AF2EvoformerBlock`. `AF2PairBlock` plus the MSA track and the outer product mean.

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

import torch
import ttnn

from .tenstorrent import (
    PAIR_ROW_BLOCK,
    Module,
    OuterProductMean,
    TriangleAttention,
    TriangleMultiplication,
    Weights,
    _fp32_softmax_attention,
    _pair_bias_from_z,
    batched_matmul,
    get_device,
)

# The pair-track shape constants of `model_1_ptm`, from the checkpoint's own config
# (`scripts/af2_port/af2ig_model_config.json`).
C_Z = 128
TRI_MUL_HIDDEN = 128
TRI_ATT_HEADS = 4
TRI_ATT_HEAD_DIM = 32
PAIR_TRANSITION_FACTOR = 4
MSA_ATT_HEADS = 8
MSA_ATT_HEAD_DIM = 32

# Row-block the MSA row attentions pair bias once LN(pair) would be the biggest tensor in the
# block. It is 11 MB at 208 tokens and 184 MB at 848, and the norm is row-local.
PAIR_BIAS_ROWBLOCK_BYTES = 128 * 2 ** 20

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
class AF2Attention(Module):
    """AF2's MSA attention: row-wise with a pair bias, and column-wise, in one class.

    Both are `af2_reference.Attention` with a LayerNorm in front, and the two flags are exactly
    what the reference's two subclasses differ by:

    * `pair_bias` adds `linear(pair_norm(z))` to the logits. That is the only thing separating
      the row variant from the shared `TriangleAttention` -- its bias comes from a second tensor
      through a second LayerNorm instead of from its own input -- and it is why this class exists
      rather than a `bias_from=` hook on a class four other models run through.
    * `column` attends over the MSA depth axis instead of the residue axis.

    The softmax is the one place the two paths genuinely differ. The row softmax is L wide and
    takes `_fp32_softmax_attention` with the bias RAW (`bias_scale_inv=1.0`): AF2 scales q by
    `key_dim**-0.5` and adds the bias unscaled, so a raw bias handed to the fused SDPA arrives
    sqrt(32) = 5.66x too flat. The column softmax is only as wide as the MSA is deep, so its
    exp-sum sits in (1, depth] -- the regime where `ttnn.softmax` called without a compute kernel
    config loses up to 2.9e-2, and `_fp32_softmax_attention` passes none. The column path
    therefore writes its four ops out here with the config attached: 4.9e-4 instead of 2.9e-2 on
    a probability, measured by `scripts/af2_port/softmax_ckc_probe.py`. Fixing the shared helper
    instead would move shipped numbers on Boltz-2, Protenix-v2, OpenFold3 and ESMFold2.

    `msa_mask` is all ones for every fold this port serves, so AF2s `1e9 * (msa_mask - 1)` logit
    bias is identically zero and is not built. ttnn masks its own tile padding, so a sequence
    length that is not a multiple of 32 needs no mask either (measured: an explicit -1e9 fill is
    bit-identical).
    """

    def __init__(
        self,
        state_dict: Weights,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
        n_heads: int = MSA_ATT_HEADS,
        head_dim: int = MSA_ATT_HEAD_DIM,
        pair_bias: bool = False,
        column: bool = False,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.pair_bias = pair_bias
        self.column = column
        self.scale_inv = head_dim**-0.5
        self.norm_weight = self.torch_to_tt("layer_norm.weight")
        self.norm_bias = self.torch_to_tt("layer_norm.bias")
        # One fused q/k/v weight, as `TriangleAttention.__init__` builds it. AF2s q/k/v are
        # `Projection`, so there is no bias to fuse. The key names the first of the three; the
        # transform argument is unused because the weight is a concatenation of all three.
        self.qkv_weight = self.torch_to_tt(
            "linear_q.weight",
            lambda _: torch.cat([self.weights[f"linear_{p}.weight"] for p in "qkv"], dim=0).t(),
        )
        self.g_weight = self.torch_to_tt("linear_g.weight")
        self.g_bias = self.torch_to_tt("linear_g.bias")
        self.o_weight = self.torch_to_tt("linear_o.weight")
        self.o_bias = self.torch_to_tt("linear_o.bias")
        if pair_bias:
            self.pair_norm_weight = self.torch_to_tt("pair_norm.weight")
            self.pair_norm_bias = self.torch_to_tt("pair_norm.bias")
            # No sqrt(head_dim) pre-scale, which is `TriangleAttention`s own
            # `scale_pair_bias=False`: AF2 adds the bias raw.
            self.bias_weight = self.torch_to_tt("linear.weight")

    def _bias(self, z: ttnn.Tensor) -> ttnn.Tensor:
        if len(z.shape) == 4:
            z = ttnn.reshape(z, tuple(z.shape)[1:])
        rows, cols, channels = (int(d) for d in z.shape)
        chunk = (None if rows * cols * channels * 2 <= PAIR_BIAS_ROWBLOCK_BYTES
                 else PAIR_ROW_BLOCK)
        return _pair_bias_from_z(z, self.pair_norm_weight, self.pair_norm_bias, self.bias_weight,
                                 self.compute_kernel_config, chunk)

    def _attend(self, q: ttnn.Tensor, k: ttnn.Tensor, v: ttnn.Tensor,
                bias: ttnn.Tensor | None) -> ttnn.Tensor:
        if bias is not None:
            out = _fp32_softmax_attention(
                q, k, v, bias, scale_inv=self.scale_inv,
                compute_kernel_config=self.compute_kernel_config,
                out_dtype=ttnn.bfloat16, bias_scale_inv=1.0)
        else:
            kt = ttnn.permute(k, (0, 1, 3, 2))
            scores = batched_matmul(q, kt, compute_kernel_config=self.compute_kernel_config)
            ttnn.deallocate(kt)
            scores = ttnn.multiply_(scores, self.scale_inv)
            probs = ttnn.softmax(scores, dim=-1,
                                 compute_kernel_config=self.compute_kernel_config)
            ttnn.deallocate(scores)
            out = batched_matmul(probs, v, compute_kernel_config=self.compute_kernel_config,
                                 dtype=ttnn.bfloat16)
            ttnn.deallocate(probs)
        for t in (q, k, v):
            ttnn.deallocate(t)
        return out

    def __call__(self, msa: ttnn.Tensor, pair: ttnn.Tensor | None = None,
                 msa_mask: ttnn.Tensor | None = None) -> ttnn.Tensor:
        assert (pair is not None) == self.pair_bias, "pair_bias and the pair argument disagree"
        assert msa_mask is None, "a masked AF2 MSA is not wired up; see the class docstring"
        if len(msa.shape) == 4:
            msa = ttnn.reshape(msa, tuple(msa.shape)[1:])
        if self.column:
            msa = ttnn.permute(msa, (1, 0, 2))
        x = ttnn.layer_norm(
            msa, weight=self.norm_weight, bias=self.norm_bias, epsilon=1e-5,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if self.column:
            ttnn.deallocate(msa)  # the permutes own copy, not the callers tensor
        qkv = self._lin(x, self.qkv_weight, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = self._attend(*self._split_heads(qkv, self.n_heads),
                           self._bias(pair) if self.pair_bias else None)
        out = self._merge_heads(out)
        gate = self._lin(x, self.g_weight, bias=self.g_bias,
                         memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(x)
        out = ttnn.multiply_(out, gate, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
        ttnn.deallocate(gate)
        projected = self._lin(out, self.o_weight, bias=self.o_bias,
                              memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(out)
        if self.column:
            transposed = ttnn.permute(projected, (1, 0, 2))
            ttnn.deallocate(projected)
            projected = transposed
        return projected


class AF2EvoformerBlock(AF2PairBlock):
    """One `EvoformerIteration`: the MSA track, the outer product mean, then the pair track.

    Subclasses `AF2PairBlock` for the reason the reference subclasses too. The checkpoint remap
    makes `evoformer.0.tri_mul_out` and `evoformer.0.opm` siblings, so a flat scope is the
    checkpoints own layout, and `outer_product_mean.first` is False so the MSA track runs first.
    """

    def __init__(self, state_dict: Weights,
                 compute_kernel_config: ttnn.DeviceComputeKernelConfig, **kwargs):
        super().__init__(state_dict, compute_kernel_config, **kwargs)
        self.msa_row_attn = AF2Attention(
            self.scope("msa_row_attn"), compute_kernel_config, pair_bias=True)
        self.msa_col_attn = AF2Attention(
            self.scope("msa_col_attn"), compute_kernel_config, column=True)
        self.msa_transition = ReluTransition(
            self.scope("msa_transition"), compute_kernel_config)
        # `scale_bias=True` puts the proj_o bias inside the division by the pair norm, which is
        # AF2s own semantics.
        self.opm = OuterProductMean(
            self.scope("opm"), compute_kernel_config, scale_bias=True)

    def _msa_track(self, msa: ttnn.Tensor, pair: ttnn.Tensor) -> ttnn.Tensor:
        for module in (self.msa_row_attn, self.msa_col_attn, self.msa_transition):
            update = module(msa, pair) if module is self.msa_row_attn else module(msa)
            msa = ttnn.add_(msa, update)
            ttnn.deallocate(update)
        return msa

    def __call__(self, msa: ttnn.Tensor, z: ttnn.Tensor,
                 msa_mask: ttnn.Tensor | None = None, mask: ttnn.Tensor | None = None,
                 attn_mask: ttnn.Tensor | None = None) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        assert msa_mask is None, "a masked AF2 MSA is not wired up; see AF2Attention"
        msa = self._msa_track(msa, z)
        # AF2 divides the outer product mean by `eps + norm`, and at an all-ones mask the norm
        # is the MSA depth everywhere. `eps` is 1e-3 and the trunk is bfloat16, whose spacing at
        # 2.0 is 0.0078, so `eps + norm` rounds back to the depth exactly -- at any depth, since
        # the spacing scales with the value. Adding it anyway measures 1.7x worse on card at
        # Evoformer 0 and 47 (`device_gate.py --opm-eps 1e-3`). `None` reads the depth off the
        # tensor, which is that divisor.
        update = self.opm(msa, None)
        z = ttnn.add_(z, update)
        ttnn.deallocate(update)
        return msa, super().__call__(z, mask, attn_mask)
