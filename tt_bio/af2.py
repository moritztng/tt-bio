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

import hashlib

import torch
import ttnn

from .af2_reference import AF2Model, load_af2_model
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

# The template pair stack's own dims, read off `template.pair_stack.0.` in the checkpoint:
# `tri_att_start.linear.weight (4, 64)` is 4 heads and `linear_q.weight (64, 64)` is 4 x 16.
# Every other width in the block is inferred from the weights by the ops themselves.
TEMPLATE_TRI_ATT_HEADS = 4
TEMPLATE_TRI_ATT_HEAD_DIM = 16

#: The three pair stacks `AF2DeviceModel.triatt_fused` can send to the fused SDPA. The template's
#: two blocks are separable from the 52 trunk ones because they are a different `head_dim` (16
#: against 32, both padded to a 32-channel tile) on a tensor the trunk never sees.
TRIATT_FUSED_STACKS = ("extra_msa", "evoformer", "template")

# Row-block the MSA row attentions pair bias once LN(pair) would be the biggest tensor in the
# block. It is 11 MB at 208 tokens and 184 MB at 848, and the norm is row-local.
PAIR_BIAS_ROWBLOCK_BYTES = 128 * 2 ** 20

# Row-block the transition once its hidden activation would be the biggest tensor in the block.
# `factor` is 4, so the hidden is 4 pair tensors; at 208 tokens that is 88 MB and fits, and the
# block is row-local (LayerNorm over channels, two matmuls over channels), so blocking changes
# nothing a row computes.
TRANSITION_ROWBLOCK_BYTES = 256 * 2 ** 20


#: The op classes `scripts/af2_port/tap_gate.py --substitute` moves to host torch, one class per
#: run. A class is every attribute running the same arithmetic, so both triangle-multiplication
#: directions are one class and both triangle attentions are another. `all` is the control: with
#: every op substituted the device blocks only carry the residual adds, so the arm has to
#: reproduce the torch trunk's error growth or the instrument is not measuring what it claims.
SUBSTITUTION_CLASSES = {
    "trimul": ("tri_mul_out", "tri_mul_in"),
    "triatt": ("tri_att_start", "tri_att_end"),
    "msa_row": ("msa_row_attn",),
    "msa_col": ("msa_col_attn",),
    "transitions": ("pair_transition", "msa_transition"),
    "opm": ("opm",),
}
SUBSTITUTION_CLASSES["all"] = tuple(
    name for names in list(SUBSTITUTION_CLASSES.values()) for name in names)


def _host_twins(block, msa_mask: torch.Tensor, pair_mask: torch.Tensor) -> dict:
    """Each substitutable op's reference module, curried with the masks the reference takes.

    The ttnn blocks take their masks implicitly -- all ones for every fold this port serves --
    and `af2_reference` takes them as arguments, so the currying is where the two signatures
    meet.
    """
    return {
        "tri_mul_out": lambda z: block.tri_mul_out(z, pair_mask),
        "tri_mul_in": lambda z: block.tri_mul_in(z, pair_mask),
        "tri_att_start": lambda z: block.tri_att_start(z, pair_mask),
        "tri_att_end": lambda z: block.tri_att_end(z, pair_mask),
        "pair_transition": lambda z: block.pair_transition(z),
        "msa_row_attn": lambda m, z: block.msa_row_attn(m, msa_mask, z),
        "msa_col_attn": lambda m: block.msa_col_attn(m, msa_mask),
        "msa_transition": lambda m: block.msa_transition(m),
        "opm": lambda m: block.opm(m, msa_mask),
    }


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


def sigmoid_gate(x: ttnn.Tensor, gate: ttnn.Tensor, wide: bool) -> ttnn.Tensor:
    """`x * sigmoid(gate)`, owning `gate`. `wide` takes the sigmoid in float32.

    The SFPU's bfloat16 sigmoid disagrees with torch's on 10.38% of elements at 1.77e-03 rms
    relative; taking it in float32 and narrowing once is bit-identical to torch at 0 of 5,537,792
    (`scripts/af2_port/eltwise_rounding_probe.py`). Unlike the residual add the disagreement is
    not one-sided -- 46.2% of it grows the magnitude -- so it compounds as a random walk rather
    than linearly, which is why it survived the residual fix as a residue no single op class
    owned. Measured, and it is a regression: see `AF2Attention.rne_sigmoid`. Kept as the
    instrument that says so.
    """
    if not wide:
        out = ttnn.multiply_(x, gate, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
        ttnn.deallocate(gate)
        return out
    prob = ttnn.typecast(gate, ttnn.float32)
    ttnn.deallocate(gate)
    prob = ttnn.sigmoid(prob)
    narrow = ttnn.typecast(prob, ttnn.bfloat16)
    ttnn.deallocate(prob)
    out = ttnn.multiply_(x, narrow)
    ttnn.deallocate(narrow)
    return out


class AF2PairBlock(Module):
    """AF2's pair track: two triangle multiplications, two triangle attentions, a transition.

    `evoformer_order=False` is the template pair stack, which runs the attentions before the
    multiplications (`modules.py:212-241` against `modules.py:1330-1356`).

    `fused_hifi` picks which attention kernel serves that fp32 softmax: `None` follows the
    process-wide `TT_BIO_TRIATT_FUSED_HIFI`, a bool pins this block. `AF2DeviceModel.triatt_fused`
    is what sets it per stack, and says why AF2 does not use the variable.

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

    #: Op attribute names whose ttnn output is replaced, in chain, by the host-torch twin's.
    #: `AF2DeviceModel._install_substitution` sets it together with `host_ops`; a fold leaves it
    #: empty and pays nothing for it.
    substitute: frozenset = frozenset()

    #: `(download, upload, twins)`, the bridge a substituted op crosses.
    host_ops: tuple | None = None

    #: Op attribute names dropped entirely: the op does not run and its residual add becomes the
    #: identity. `substitute` is the wrong instrument for a COST screen because it moves an op to
    #: host torch, so a leg measures `host_X - device_X` rather than `device_X`. This one measures
    #: the device cost of the op class plus the one residual add that carries it. It is
    #: arithmetically wrong on purpose -- a leg times synthetic inputs and makes no accuracy claim,
    #: the same convention `trunk_timing.py` ships -- and, like `substitute`, a fold leaves it
    #: empty. Set by `scripts/af2_port/fold_timing.py --skip`.
    skip: frozenset = frozenset()

    #: Route every residual add through float32 so the bfloat16 result rounds ties to even, which
    #: is what torch and JAX do. `ttnn.add` breaks them away from zero and its bfloat16 datapath
    #: is narrower than float32, so it disagrees with the reference on 11.2% of elements at equal
    #: operand magnitudes -- 1 ulp each, 9 adds per Evoformer block, 432 over the stack
    #: (`scripts/af2_port/residual_add_probe.py`). On by default because it is the whole of this
    #: trunk's error growth: it takes the four-recycle device leg from 52 failed taps of 94 and
    #: 0.084555 of i_pTM to 9 and 0.002605, and costs 0.42 s over four trunk passes.
    rne_residual = True

    #: Where the two float32 temporaries live. They are twice the width of the bfloat16 pair, and
    #: inheriting its memory config puts them in L1 with it: at 512 tokens the bfloat16 pair fits
    #: (64 MB) and its float32 copy does not (128 MB across 130 banks, 943 KB free per bank), so
    #: the trunk OOMs at a length it runs at with the fix off, and runs again at 848 where the
    #: bfloat16 pair is itself too big for L1. DRAM has room at every length and the arithmetic
    #: does not depend on where the operands sit, so the temporaries go there and the result
    #: comes back to whatever memory config the input arrived in.
    rne_wide_dram = True

    def _residual(self, x: ttnn.Tensor, update: ttnn.Tensor | None) -> ttnn.Tensor:
        """`x + update`, and it owns `update`.

        The wide path is bit-identical to torch's bfloat16 add at every operand ratio measured,
        which the in-place `ttnn.add_` is not.

        `None` is a skipped op (see `skip`): there is no update, so the residual is the identity.
        """
        if update is None:
            return x
        if not self.rne_residual:
            out = ttnn.add_(x, update)
            ttnn.deallocate(update)
            return out
        config = x.memory_config()
        wide_config = ttnn.DRAM_MEMORY_CONFIG if self.rne_wide_dram else config
        wide = ttnn.typecast(x, ttnn.float32, memory_config=wide_config)
        other = ttnn.typecast(update, ttnn.float32, memory_config=wide_config)
        ttnn.deallocate(update)
        ttnn.deallocate(x)
        wide = ttnn.add_(wide, other)
        ttnn.deallocate(other)
        out = ttnn.typecast(wide, ttnn.bfloat16, memory_config=config)
        ttnn.deallocate(wide)
        return out

    def __init__(
        self,
        state_dict: Weights,
        compute_kernel_config: ttnn.DeviceComputeKernelConfig,
        head_dim: int = TRI_ATT_HEAD_DIM,
        n_heads: int = TRI_ATT_HEADS,
        evoformer_order: bool = True,
        fused_hifi: bool | None = None,
    ):
        super().__init__(state_dict, compute_kernel_config)
        self.evoformer_order = evoformer_order
        self.tri_mul_out = TriangleMultiplication(
            False, self.scope("tri_mul_out"), compute_kernel_config)
        self.tri_mul_in = TriangleMultiplication(
            True, self.scope("tri_mul_in"), compute_kernel_config)
        self.tri_att_start = TriangleAttention(
            head_dim, n_heads, False, self.scope("tri_att_start"), compute_kernel_config,
            scale_pair_bias=False, fp32_softmax=True, fused_hifi=fused_hifi)
        self.tri_att_end = TriangleAttention(
            head_dim, n_heads, True, self.scope("tri_att_end"), compute_kernel_config,
            scale_pair_bias=False, fp32_softmax=True, fused_hifi=fused_hifi)
        self.pair_transition = ReluTransition(
            self.scope("pair_transition"), compute_kernel_config)

    def _update(self, name: str, device, x: ttnn.Tensor,
                *args: ttnn.Tensor) -> ttnn.Tensor | None:
        """One op's residual update: from the card, or from its host-torch twin if substituted.

        The residual add stays on card either way, so a substitution changes exactly one op's
        arithmetic and nothing about how the block is chained. That is the whole point of the
        instrument: an isolated per-op screen scores an op against its own captured input and
        measures how much error it injects, while this one leaves the op in the chain and
        measures how fast the block's error grows with it swapped out.

        A skipped op has no update at all, which `_residual` turns into the identity.
        """
        if name in self.skip:
            return None
        if name not in self.substitute:
            return device(x, *args)
        down, up, twins = self.host_ops
        return up(twins[name](*[down(t) for t in (x, *args)]))

    def __call__(self, z: ttnn.Tensor, mask: ttnn.Tensor | None = None,
                 attn_mask: ttnn.Tensor | None = None) -> ttnn.Tensor:
        order = [("tri_mul_out", lambda t: self.tri_mul_out(t, mask)),
                 ("tri_mul_in", lambda t: self.tri_mul_in(t, mask)),
                 ("tri_att_start", lambda t: self.tri_att_start(t, attn_mask)),
                 ("tri_att_end", lambda t: self.tri_att_end(t, attn_mask))]
        if not self.evoformer_order:
            order = order[2:] + order[:2]
        for name, device in order + [("pair_transition", self.pair_transition)]:
            z = self._residual(z, self._update(name, device, z))
        return z
class AF2DeviceTemplatePairStack:
    """The template's two `PairBlock`s in ttnn: host torch in, host torch out.

    `AF2PairBlock` with `evoformer_order=False` -- the template runs the attentions before the
    multiplications -- at the template's own widths. `mask_2d` is asserted all ones rather than
    plumbed, for `evoformer_stack`'s reason: AF2 masks BOTH halves of the triangle
    multiplication's fused projection where `TriangleMultiplication` masks only the `a` half, so
    a genuinely masked fold needs that difference resolved before a mask can be honoured here.
    """

    def __init__(self, blocks: list, up, down):
        self.blocks, self._up, self._down = blocks, up, down

    def __call__(self, act: torch.Tensor, mask_2d: torch.Tensor) -> torch.Tensor:
        assert bool((mask_2d == 1).all()), (
            "a masked AF2 template pair stack is not wired up; see AF2PairBlock")
        shape = tuple(act.shape)
        z = self._up(act)
        for block in self.blocks:
            z = block(z)
        out = self._down(z, shape)
        ttnn.deallocate(z)
        return out


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

    #: Take the gating sigmoid in float32. OFF, and measured that way: widening it is
    #: bit-identical to torch per op and worse end to end, pair growth 1.0465 -> 1.0492 and the
    #: structure module 3.51e-03 -> 6.75e-03 of 1-pcc. The add's rounding is one-sided (100% of
    #: its disagreements grow the magnitude) so removing it removes a bias; the sigmoid's is
    #: not (46.2%), so removing it only redraws a random walk, and this draw landed worse. One
    #: probe column, `grew`, separates the two cases and it is the one to read.
    rne_sigmoid = False

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
        out = sigmoid_gate(out, gate, self.rne_sigmoid)
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
        msa = self._residual(msa, self._update("msa_row_attn", self.msa_row_attn, msa, pair))
        for name, module in (("msa_col_attn", self.msa_col_attn),
                             ("msa_transition", self.msa_transition)):
            msa = self._residual(msa, self._update(name, module, msa))
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
        z = self._residual(z, self._update("opm", lambda m: self.opm(m, None), msa))
        return msa, super().__call__(z, mask, attn_mask)


class AF2DeviceModel(AF2Model):
    """`AF2Model` with its two block stacks on card and everything else in torch.

    Host keeps the embeddings, the recycling state, the template, the structure module and the
    two confidence heads. The card keeps the 4 extra-MSA blocks and the 48 Evoformer blocks,
    which is every O(L^3) op in the trunk. The boundary is one round trip per stack per
    recycling pass: the pair and the MSA go up, the same two come back down.

    Two things the reference does per pass, this class does once per design, both bit-exact
    rather than approximate (`scripts/af2_port/host_screen.py` proves each by substitution):

    * **The template embedding.** It is constant in its `pair` argument -- with one template the
      cross-attention softmaxes over a single key, so the weight is exactly 1.0 and the query
      never reaches the output -- and everything else it reads is fixed for a design. Four calls
      become one, which is what keeps it on host at 0.44 s per design against a 1.0 s bar.
    * **The extra-MSA stack's MSA track.** It reaches the pair through the outer product mean and
      nothing else, and with `extra_msa_mask` all zeros that output is `proj_o.bias / eps`, one
      vector repeated over every pair position. This class computes the constant on host and
      injects it, so `MsaColumnGlobalAttention` never has to exist in ttnn. The assert holds the
      claim to its precondition: a featurisation with a real extra MSA fails here rather than
      folding silently against the wrong constant.
    """

    #: `(tag, payload) -> None`, set by `scripts/af2_port/tap_gate.py`. When it is set the
    #: stacks download every block's output, the extra-MSA stack runs its dead MSA track on
    #: host, and the memoised template re-emits its two taps on the passes it does not
    #: recompute -- so the device leg owes exactly the taps the torch leg owes. A fold never
    #: sets it and never pays for any of it.
    block_tap = None

    #: True runs the four extra-MSA blocks in host torch instead of on card. That stack sets
    #: the pair representation the 48 Evoformer blocks start from, so this is the other half of
    #: the substitution instrument: it moves the block-0 input between the two arms without
    #: touching anything in the 48 blocks that follow.
    extra_msa_host = False

    #: Op classes the Evoformer stack runs in host torch instead of on card, from
    #: `SUBSTITUTION_CLASSES`. Set by `scripts/af2_port/tap_gate.py --substitute` and empty for
    #: every fold.
    substitute: frozenset = frozenset()

    #: Op classes dropped from BOTH device stacks, for the cost census. `set_skip` sets it, a fold
    #: leaves it empty, and `AF2PairBlock.skip` says why it is not `substitute`.
    skip: frozenset = frozenset()

    #: True runs the template's two `PairBlock`s in host torch instead of on card. It is the
    #: arm that prices the seam in one process, and the control that has to reproduce pass 16's
    #: committed device numbers -- the template was on host when they were taken.
    template_host = False

    #: Which pair stacks run their two triangle attentions on the fused persistent-mask SDPA
    #: instead of the materialised fp32 softmax, from `TRIATT_FUSED_STACKS`. `None` follows the
    #: process-wide `TT_BIO_TRIATT_FUSED_HIFI` for every stack, which is what the perf branch's
    #: A/B legs assign. A set pins AF2's own blocks and leaves the variable alone, because
    #: PXDesign runs the Protenix filter in the same process and the same variable flips its
    #: triangle attention too. Not bit-exact against the materialised path: an online softmax
    #: reduces over k in a different order, so a change here is an accuracy question, scored by
    #: `filter_flip_rate.py` over both design populations.
    triatt_fused: frozenset | None = None

    #: Off recomputes the template every pass. It must change no number anywhere, which is what
    #: `tap_gate.py --device --no-template-cache` checks against the same reference taps. On, the
    #: cache is keyed by `_template_key`, so it saves the three recycles of one design and is
    #: invalidated by the next design rather than served to it.
    template_cached = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device_extra_msa: list = []
        self.device_evoformer: list = []
        self.device_template: list = []
        self.opm_constant: list = []
        self._device = None
        self._template_cache = None

    def to_device(self) -> "AF2DeviceModel":
        """Build the ttnn blocks from the parameters already loaded into the torch modules."""
        state = self.state_dict()
        ckc = compute_kernel_config()
        self._device = get_device()

        def scoped(prefix: str) -> Weights:
            return {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}

        self.device_extra_msa = [AF2PairBlock(scoped(f"extra_msa.{i}."), ckc,
                                              fused_hifi=self._fused_hifi("extra_msa"))
                                 for i in range(len(self.extra_msa))]
        self.device_evoformer = [AF2EvoformerBlock(scoped(f"evoformer.{i}."), ckc,
                                                   fused_hifi=self._fused_hifi("evoformer"))
                                 for i in range(len(self.evoformer))]
        if self.template is not None:
            self.device_template = [
                AF2PairBlock(scoped(f"template.pair_stack.{i}."), ckc,
                             head_dim=TEMPLATE_TRI_ATT_HEAD_DIM,
                             n_heads=TEMPLATE_TRI_ATT_HEADS, evoformer_order=False,
                             fused_hifi=self._fused_hifi("template"))
                for i in range(len(self.template.pair_stack))]
            self._template_stack = AF2DeviceTemplatePairStack(
                self.device_template, self._up, self._down)
            self.set_template_host(self.template_host)
        zero = torch.zeros((), dtype=self.trunk_dtype)
        self.opm_constant = [
            block.opm.proj_o.bias.to(self.trunk_dtype) / (block.opm.eps + zero)
            for block in self.extra_msa]
        return self

    # ------------------------------------------------------------------ the boundary

    def _up(self, t: torch.Tensor) -> ttnn.Tensor:
        return ttnn.from_torch(t.unsqueeze(0).to(torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                               device=self._device, dtype=ttnn.bfloat16)

    def _down(self, t: ttnn.Tensor, shape: tuple) -> torch.Tensor:
        x = torch.Tensor(ttnn.to_torch(t))
        while x.dim() > len(shape) and x.shape[0] == 1:
            x = x.squeeze(0)
        assert tuple(x.shape) == tuple(shape), f"device gave {tuple(x.shape)}, want {shape}"
        return x.to(self.trunk_dtype)

    @property
    def _device_blocks(self) -> list:
        """Every `AF2PairBlock` on card, so a global arm cannot miss a stack."""
        return self.device_extra_msa + self.device_evoformer + self.device_template

    def _fused_hifi(self, stack: str) -> bool | None:
        """Whether `stack`'s triangle attentions take the fused SDPA. See `triatt_fused`."""
        return None if self.triatt_fused is None else stack in self.triatt_fused

    def set_triatt_fused(self, stacks) -> None:
        """Pin which pair stacks take the fused SDPA, without rebuilding the blocks.

        The construction-time route is `triatt_fused` read by `to_device`; this one exists because
        an A/B has to interleave both arms in one process to be believable, and rebuilding 54
        blocks between arms costs more than the leg. See `triatt_fused` for what the values mean.
        """
        if stacks is not None:
            stacks = frozenset(stacks)
            unknown = stacks - set(TRIATT_FUSED_STACKS)
            assert not unknown, f"unknown pair stacks {sorted(unknown)}, want {TRIATT_FUSED_STACKS}"
        self.triatt_fused = stacks
        for stack, blocks in (("extra_msa", self.device_extra_msa),
                              ("evoformer", self.device_evoformer),
                              ("template", self.device_template)):
            for block in blocks:
                block.tri_att_start.fused_hifi = self._fused_hifi(stack)
                block.tri_att_end.fused_hifi = self._fused_hifi(stack)

    def set_template_host(self, enabled: bool) -> None:
        """Run the template's pair stack in host torch. See `template_host`."""
        self.template_host = enabled
        if self.template is not None:
            self.template.pair_stack_device = None if enabled else self._template_stack

    def set_rne_residual(self, enabled: bool) -> None:
        """Route every residual add in both trunk stacks through float32. See
        `AF2PairBlock.rne_residual`."""
        for block in self._device_blocks:
            block.rne_residual = enabled

    def set_rne_wide_dram(self, enabled: bool) -> None:
        """Put the float32 residual temporaries in DRAM instead of inheriting the pair's memory
        config. See `AF2PairBlock.rne_wide_dram`; off is the arm that OOMs at 512 tokens."""
        for block in self._device_blocks:
            block.rne_wide_dram = enabled

    def set_rne_sigmoid(self, enabled: bool) -> None:
        """Route both MSA attentions' gating sigmoid through float32. A screening arm, not a
        default: see `AF2Attention.rne_sigmoid` for what it measured."""
        for block in self._device_blocks:
            for name in ("msa_row_attn", "msa_col_attn"):
                if hasattr(block, name):
                    getattr(block, name).rne_sigmoid = enabled

    def set_skip(self, names) -> None:
        """Drop an op class from both device stacks. See `AF2PairBlock.skip`; a fold never calls
        this, and every leg that does is a timing leg on synthetic inputs."""
        self.skip = frozenset(names)
        for block in self.device_extra_msa + self.device_evoformer:
            block.skip = self.skip

    def _down_unshaped(self, t: ttnn.Tensor) -> torch.Tensor:
        """`_down` for the substitution bridge, which knows the op but not the rank."""
        x = torch.Tensor(ttnn.to_torch(t))
        while x.dim() > 3 and x.shape[0] == 1:
            x = x.squeeze(0)
        return x.to(self.trunk_dtype)

    def _install_substitution(self, msa_mask: torch.Tensor, pair_mask: torch.Tensor) -> None:
        """Point every Evoformer block's substituted ops at their host-torch twins.

        The Evoformer stack only, deliberately: the number being measured is the 48-block error
        growth rate, so leaving the extra-MSA stack entirely on card gives every arm the same
        block-0 input and the same intercept, and a substitution can then only move the slope.
        """
        for block, host in zip(self.device_evoformer, self.evoformer):
            block.substitute = self.substitute
            block.host_ops = (self._down_unshaped, self._up,
                              _host_twins(host, msa_mask, pair_mask))

    def _tap(self, tag: str, **payload) -> None:
        if self.block_tap is not None:
            self.block_tap(tag, payload)

    # ------------------------------------------------------------------ the two stacks

    def extra_msa_stack(self, extra: torch.Tensor, pair: torch.Tensor,
                        extra_mask: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
        if self.extra_msa_host:
            # `AF2Model.extra_msa_stack`'s loop, with the taps the device path would emit. The
            # dead MSA track runs for real here rather than collapsing to its constant, which
            # `scripts/af2_port/host_screen.py` already proved is the same pair either way.
            for block in self.extra_msa:
                extra, pair = block(extra, pair, extra_mask, pair_mask)
                self._tap("extra_msa_stack", msa=extra, pair=pair)
            return pair
        assert bool((extra_mask == 0).all()), (
            "this port replaces the extra-MSA track with the constant its outer product mean "
            "collapses to under an all-zero mask; a real extra MSA needs the track written")
        shape = tuple(pair.shape)
        z = self._up(pair)
        for index, block in enumerate(self.device_extra_msa):
            if self.block_tap is not None:
                # The dead track, on host, only so the device leg owes the torch leg's taps. It
                # reads the block's INPUT pair, which is what the reference hands it.
                extra = self.extra_msa[index]._msa_track(extra, self._down(z, shape), extra_mask)
            const = self._up(self.opm_constant[index].reshape(1, 1, -1))
            z = block(block._residual(z, const))
            if self.block_tap is not None:
                self._tap("extra_msa_stack", msa=extra, pair=self._down(z, shape))
        out = self._down(z, shape)
        ttnn.deallocate(z)
        return out

    def evoformer_stack(self, msa: torch.Tensor, pair: torch.Tensor, msa_mask: torch.Tensor,
                        pair_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert bool((msa_mask == 1).all()), "a masked AF2 MSA is not wired up; see AF2Attention"
        if self.substitute:
            self._install_substitution(msa_mask, pair_mask)
        msa_shape, pair_shape = tuple(msa.shape), tuple(pair.shape)
        m, z = self._up(msa), self._up(pair)
        for block in self.device_evoformer:
            m, z = block(m, z)
            if self.block_tap is not None:
                self._tap("evoformer_iteration", msa=self._down(m, msa_shape),
                          pair=self._down(z, pair_shape))
        out = self._down(m, msa_shape), self._down(z, pair_shape)
        ttnn.deallocate(m)
        ttnn.deallocate(z)
        return out

    # ------------------------------------------------------------------ the template, once

    @staticmethod
    def _template_key(feats: dict, mask_2d: torch.Tensor,
                      multichain_mask: torch.Tensor) -> tuple:
        """Content key over every input the template reads except `pair`.

        The cache is worth having because `AF2Template.forward` is constant in `pair`: with one
        template the pointwise attention softmaxes over a single key, so the weight is exactly
        1.0 and the query drops out. It is NOT constant in the template features, and those
        change with the design: `complex_features` masks the template sequence, so
        `template_aatype` is identical for every design and the whole design dependence sits in
        the coordinates. Two PXDesign backbones against the same target share their target block
        bit for bit and differ by 34 A in the binder, which a key on nothing serves to the wrong
        design.

        Every `template_*` feature goes in, not just the ones read today, so the key cannot go
        stale if the module starts reading one more. The hashed bytes are ~200 KB per call
        against a 0.44 s template pass.
        """
        parts = [feats[k] for k in sorted(feats) if k.startswith("template_")]
        parts += [mask_2d, multichain_mask]

        def digest(t: torch.Tensor) -> tuple:
            # `mask_2d` arrives in the trunk dtype, and numpy has no bfloat16. Widening to float64
            # is exact from every float dtype this model uses, and the dtype string is in the key
            # anyway, so a bf16 arm and an fp32 arm still hash apart.
            raw = t.detach().contiguous()
            raw = raw.double() if raw.dtype.is_floating_point else raw
            return (tuple(t.shape), str(t.dtype),
                    hashlib.blake2b(raw.cpu().numpy().tobytes(), digest_size=16).digest())

        return tuple(digest(t) for t in parts)

    def template_embedding(self, pair: torch.Tensor, feats: dict, mask_2d: torch.Tensor,
                           multichain_mask: torch.Tensor) -> torch.Tensor:
        key = self._template_key(feats, mask_2d, multichain_mask)
        if self._template_cache is not None and self._template_cache[0] == key:
            # The pass that computed it already fired every hook a tap gate installed; only the
            # passes served from the cache have to re-emit, or the tap counts diverge.
            _, stack_out, embedding = self._template_cache
            self._tap("template_pair_stack", out=stack_out)
            self._tap("template_embedding", out=embedding)
            return embedding
        # A forward hook on the last torch block is dead once the stack is on card, so the tap
        # comes off `run_pair_stack`, which both arms go through.
        stack = []
        run = self.template.run_pair_stack

        def record(act, mask):
            act = run(act, mask)
            stack.append(act)
            self._tap("template_pair_stack", out=act)
            return act

        self.template.run_pair_stack = record
        try:
            embedding = super().template_embedding(pair, feats, mask_2d, multichain_mask)
        finally:
            # Removes the instance-dict entry and restores the bound class method.
            del self.template.run_pair_stack
        if self.template_cached:
            self._template_cache = (key, stack[-1], embedding)
        return embedding


def load_af2_device_model(state_dict: dict, *, template: bool = True, **kwargs):
    """`load_af2_model`, then the ttnn stacks. One device context per process."""
    return load_af2_model(state_dict, template=template, cls=AF2DeviceModel,
                          **kwargs).to_device()
