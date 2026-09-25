"""AlphaFold2 on ttnn: the blocks that are AF2's and nobody else's.

Serves `model_1_ptm` and AF2-multimer_v3 from one set of classes. The two variants run the same
ops at the same widths on card and differ by two orderings: multimer_v3 folds the outer product
mean into the pair before the row attention reads it, and its template pair stack runs the
multiplications before the attentions where the monomer's runs the attentions first. Everything
else that separates them -- the 73-channel relative encoding, the nine summed template feature
embeddings, the backbone unit vectors -- is host featurisation, in `af2_reference.py`.


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

**The masks.** AF2 builds `mask_2d = seq_mask[:, None] * seq_mask[None, :]`, so a masked fold
arrives as an outer product and never as an arbitrary matrix. That is what makes the shared
triangle multiplication usable: it masks the `a` half of the fused projection alone
(`tenstorrent.py:7326`) where AF2 masks both halves before the split
(`af2_reference.py:242`), and on an outer-product mask the two are the same number on every
real residue pair. `af2_pair_masks` carries the algebra and the guard that keeps it true. An
all-ones mask still takes the None path, so every fold PXDesign runs today is unchanged.
"""
from __future__ import annotations

import hashlib

import torch
import ttnn

from .af2_reference import AF2Model, load_af2_model
from .tenstorrent import (
    CORE_GRID_MAIN,
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

#: The named arms `scripts/af2_port/*.py --triatt-fused` offers, so three scripts cannot drift
#: into three spellings of the same configuration. `trunk` is the fallback the accuracy verdict
#: prices if `all` flips a design: the lever where pass 8 measured it, template left materialised.
TRIATT_FUSED_ARMS = {
    "inherit": None,
    "none": frozenset(),
    "trunk": frozenset({"extra_msa", "evoformer"}),
    "all": frozenset(TRIATT_FUSED_STACKS),
}

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

    The ttnn blocks take their masks as two prepared tensors (`af2_pair_masks`) and
    `af2_reference` takes the one `mask_2d` they were built from, so the currying is where the
    two signatures meet.
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


#: AF2's masked-logit constant (`modules.py`: `1e9 * (mask - 1)`), shared by the pair track's key
#: bias and `AF2EvoformerBlock._mask_biases`.
MASK_LOGIT_BIAS = 1e9


def af2_pair_masks(mask_2d: torch.Tensor,
                   device=None) -> tuple[ttnn.Tensor, ttnn.Tensor] | tuple[None, None]:
    """`mask_2d` -> the two tensors `AF2PairBlock` takes, or `(None, None)` if it is all ones.

    AF2 builds `mask_2d = seq_mask[:, None] * seq_mask[None, :]`, so it is an OUTER PRODUCT of a
    0/1 vector. Everything below rests on that, and the assert is what keeps it honest.

    *The multiply.* AF2 masks the fused projection before the split -- `mask * p_in(x)`, both
    halves (`af2_reference.py:242`) -- and `TriangleMultiplication` masks the `a` half alone
    (`tenstorrent.py:7326`, and its own comment at :7155 says so). On an outer-product mask the
    two are the same number on every real pair: masking both gives
    `s_i s_j sum_k s_k a_ik b_jk` and masking `a` gives `s_i sum_k s_k a_ik b_jk`, equal wherever
    `s_j = 1`, and the padded `k` is killed by `a`'s own mask in either. They differ only on
    masked rows and columns, which nothing downstream reads: the padded key `j` is removed by the
    bias below, the padded MSA column by `AF2EvoformerBlock._mask_biases`, and the padded
    residue pair by `AF2MaskedOuterProductMean`. Measured in float64 on both directions,
    max |difference| 0.0 over the real block (`perf/bcx_mask/one_sided_algebra.json`). So this
    port needs no both-halves triangle multiplication, and the shared class stays shared.

    *The key bias.* AF2's triangle attention adds `1e9 * (mask_2d - 1)` per query row
    (`af2_reference.py:262`). This returns `1e9 * (seq_mask - 1)` broadcast over the query axis
    instead: `[1, 1, 1, n]` against `[n, 1, 1, n]`, the same numbers on every real row, and the
    rows where they differ are the masked queries whose output nothing reads. The pair mask is
    the outer product and not the 1-D mask for the reason `token_axis.py:420` records.
    """
    assert mask_2d.dim() == 2 and mask_2d.shape[0] == mask_2d.shape[1], mask_2d.shape
    if bool((mask_2d == 1).all()):
        return None, None
    seq = torch.diagonal(mask_2d).float()
    assert bool(((seq == 0) | (seq == 1)).all()), "mask_2d's diagonal is not 0/1"
    assert torch.equal(mask_2d.float(), seq[:, None] * seq[None, :]), (
        "AF2's mask_2d is an outer product of seq_mask; the `a`-half triangle multiplication is "
        "only exact on one, so a general pair mask needs the both-halves form written first")
    up = lambda t: ttnn.from_torch(t.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                                   device=device or get_device(), dtype=ttnn.bfloat16)
    return (up(mask_2d.float().unsqueeze(0)),
            up(((seq - 1.0) * MASK_LOGIT_BIAS).reshape(1, 1, 1, -1)))


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
        # `bias_in_matmul="o"` keeps linear_o.bias inside the output projection's matmul, which
        # is 9 failing taps at pcc 0.99690 against 13 at 0.99180 for a separate `ttnn.add_`. It is
        # named here rather than defaulted in the shared block because RF3 biases the same
        # projection and the same form costs it accuracy (state/pxdesign-af2ig-port.md, pass 21).
        # `l1_padded_plan=True` derives the fp32-softmax L1 block from the tile-padded token
        # extent, which is the extent the shard takes. AF2-IG's token counts are ragged at every
        # rung it runs (208/336/592/848 are all 16 mod 32), so the logical extent under-sizes the
        # plan and the block loses residency it has already paid for: 1.3621x on the 848 trunk
        # pass, bit-exact, `structure_sha16 cd80f8e274306706` unchanged. Named here rather than
        # defaulted in the shared block because the pairformer stacks share this class and were
        # priced with the lever off (state/pxdesign-af2ig-land.md, D2).
        self.tri_att_start = TriangleAttention(
            head_dim, n_heads, False, self.scope("tri_att_start"), compute_kernel_config,
            scale_pair_bias=False, fp32_softmax=True, fused_hifi=fused_hifi,
            bias_in_matmul="o", l1_padded_plan=True)
        self.tri_att_end = TriangleAttention(
            head_dim, n_heads, True, self.scope("tri_att_end"), compute_kernel_config,
            scale_pair_bias=False, fp32_softmax=True, fused_hifi=fused_hifi,
            bias_in_matmul="o", l1_padded_plan=True)
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
    multiplications -- at the template's own widths. It takes the same `mask_2d` the trunk does,
    through the same `af2_pair_masks`; the template's pair stack is the same four ops.
    """

    def __init__(self, blocks: list, up, down):
        self.blocks, self._up, self._down = blocks, up, down

    def __call__(self, act: torch.Tensor, mask_2d: torch.Tensor) -> torch.Tensor:
        masks = af2_pair_masks(mask_2d)
        shape = tuple(act.shape)
        z = self._up(act)
        for block in self.blocks:
            z = block(z, *masks)
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

    An all-ones `msa_mask` makes AF2s `1e9 * (msa_mask - 1)` logit bias identically zero, so the
    caller passes no `mask_bias` and none is built; multimer_v3, whose template MSA rows are
    masked by chi 1, passes one. ttnn masks its own tile padding, so a sequence length that is
    not a multiple of 32 needs no mask either (measured: an explicit -1e9 fill is bit-identical).
    """

    #: Take the gating sigmoid in float32. OFF, and measured that way: widening it is
    #: bit-identical to torch per op and worse end to end, pair growth 1.0465 -> 1.0492 and the
    #: structure module 3.51e-03 -> 6.75e-03 of 1-pcc. The add's rounding is one-sided (100% of
    #: its disagreements grow the magnitude) so removing it removes a bias; the sigmoid's is
    #: not (46.2%), so removing it only redraws a random walk, and this draw landed worse. One
    #: probe column, `grew`, separates the two cases and it is the one to read.
    rne_sigmoid = False

    #: Derive the fp32-softmax L1 block from the tile-PADDED token extent, which is the extent
    #: the shard takes. On for AF2-IG, whose token counts are ragged at every rung it runs; see
    #: `AF2PairBlock.tri_att_start` for the measurement and `AF2DeviceModel.set_l1_padded_plan`
    #: for the arm. Only the row variant reads it -- the column softmax has no pair bias, so it
    #: never reaches `_fp32_softmax_attention` at all.
    l1_padded_plan = True

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
        # Route on the variant, not on whether a bias came in. The column attention's only bias
        # is the MSA mask's, and routing it to the fp32 helper took it off its own softmax: at one
        # MSA row, where the answer is v exactly, the helper misses by rel L2 2.8e-2 and this path
        # by 0.0 (perf/bcx_maskbias/colop.json). That was the whole of the masked block's `mo`
        # error at an all-ones mask (perf/bcx_maskbias/fwd_b0.json).
        if self.pair_bias:
            out = _fp32_softmax_attention(
                q, k, v, bias, scale_inv=self.scale_inv,
                compute_kernel_config=self.compute_kernel_config,
                out_dtype=ttnn.bfloat16, bias_scale_inv=1.0,
                l1_padded_plan=self.l1_padded_plan)
        else:
            kt = ttnn.permute(k, (0, 1, 3, 2))
            scores = batched_matmul(q, kt, compute_kernel_config=self.compute_kernel_config)
            ttnn.deallocate(kt)
            scores = ttnn.multiply_(scores, self.scale_inv)
            if bias is not None:
                # AF2 adds the mask bias AFTER the scale, unscaled, so it goes here and not
                # into `scale_inv`.
                scores = ttnn.add(scores, bias)
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
                 mask_bias: ttnn.Tensor | None = None) -> ttnn.Tensor:
        """`mask_bias` is AF2's `1e9 * (msa_mask - 1)` already shaped for the softmax axis.

        It arrives prepared rather than as a mask because the two variants need different
        layouts of the same numbers -- `[rows, 1, 1, n]` for the row attention, whose keys
        are residues, and `[n, 1, 1, rows]` for the column attention, whose keys are rows --
        and `AF2EvoformerBlock` builds both once per block instead of once per attention.
        """
        assert (pair is not None) == self.pair_bias, "pair_bias and the pair argument disagree"
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
        bias = self._bias(pair) if self.pair_bias else None
        if mask_bias is not None:
            # AF2 adds the pair bias and the mask bias to the same logits. The pair bias is
            # [1, heads, n, n] and broadcasts over rows; the mask bias is per row, so the sum
            # is [rows, heads, n, n] -- 1.2 MB at n=192 with 2 rows, which is why this is an
            # add rather than a second softmax argument.
            bias = mask_bias if bias is None else ttnn.add(bias, mask_bias)
        out = self._attend(*self._split_heads(qkv, self.n_heads), bias)
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


class AF2MaskedOuterProductMean(OuterProductMean):
    """AF2's outer product mean with a real MSA mask.

    The shared `OuterProductMean` cannot express this and must not be changed to: it is
    Boltz's, BoltzGen's, OpenFold3's, Protenix's, OpenDDE's, RF3's and AF2-IG's as well.
    It differs from AF2 in exactly two ways, and both matter once a mask is not all ones.

    It masks only the `a` operand -- the same one-sided masking `AF2PairBlock` documents for
    the triangle multiplication -- where AF2 masks both, so its sum is
    `sum_s m_si a_si b_sj` against AF2's `sum_s m_si m_sj a_si b_sj`. And it divides by a
    scalar depth, where AF2 divides by `eps + sum_s m_si m_sj`, which is a matrix once the
    rows disagree about which residues are real.

    `eps` stops being cosmetic here. `AF2EvoformerBlock` drops it at an all-ones mask
    because bf16 rounds `eps + depth` back to the depth; with a real mask the norm is 0
    wherever both residues are masked in every row, and 1e-3 is what keeps that finite.
    """

    EPS = 1e-3

    def _sum_rows(self, a: ttnn.Tensor, b: ttnn.Tensor) -> ttnn.Tensor:
        """`sum_s a_sic b_sjd W_cdk + o_bias`, unscaled, without `ttnn.repeat`.

        `OuterProductMean._small_depth` gives `b` the I batch with `ttnn.repeat`, which has
        no tape entry, so the masked OPM's backward cannot reach it. The same contraction
        runs transposed here: `[I, c_z, D] x [D, J]` broadcasts a 2D in1 over in0's batch,
        the direction ttnn supports, and both permutes are verbs the trunk already tapes.
        """
        S, I, C = (int(d) for d in a.shape)
        _, J, D = (int(d) for d in b.shape)
        w = self._proj_o_folded(C, D)
        c_z = int(w.shape[1]) // D
        out = None
        for s_i in range(S):
            a_s = ttnn.reshape(a if S == 1 else a[s_i:s_i + 1], (I, C))
            A = ttnn.matmul(a_s, w, compute_kernel_config=self.compute_kernel_config,
                            core_grid=CORE_GRID_MAIN)
            A = ttnn.to_layout(A, ttnn.ROW_MAJOR_LAYOUT)
            A = ttnn.reshape(A, (I, D, c_z))
            A = ttnn.to_layout(A, ttnn.TILE_LAYOUT)
            A = ttnn.permute(A, (0, 2, 1))
            b_s = ttnn.reshape(b if S == 1 else b[s_i:s_i + 1], (J, D))
            bt = ttnn.permute(b_s, (1, 0))
            part = ttnn.matmul(A, bt, compute_kernel_config=self.compute_kernel_config)
            part = ttnn.permute(part, (0, 2, 1))
            out = part if out is None else ttnn.add(out, part)
        out = ttnn.add(out, self.o_bias)
        return ttnn.reshape(out, (1, *tuple(out.shape)))

    def masked(self, x: ttnn.Tensor, msa_mask: ttnn.Tensor) -> ttnn.Tensor:
        if len(x.shape) == 4:
            x = ttnn.reshape(x, tuple(x.shape)[1:])
        if len(msa_mask.shape) == 3:
            msa_mask = ttnn.reshape(msa_mask, tuple(msa_mask.shape)[1:])
        rows, n = (int(d) for d in msa_mask.shape)
        mask_col = ttnn.reshape(msa_mask, (rows, n, 1))

        normed = ttnn.layer_norm(x, weight=self.norm_weight, bias=self.norm_bias,
                                 epsilon=1e-5,
                                 compute_kernel_config=self.compute_kernel_config)
        a = ttnn.linear(normed, self.a_weight, bias=self.a_bias,
                        compute_kernel_config=self.compute_kernel_config,
                        core_grid=CORE_GRID_MAIN)
        b = ttnn.linear(normed, self.b_weight, bias=self.b_bias,
                        compute_kernel_config=self.compute_kernel_config,
                        core_grid=CORE_GRID_MAIN)
        ttnn.deallocate(normed)
        # BOTH operands, which is the difference from the shared class.
        a = ttnn.multiply_(a, mask_col)
        b = ttnn.multiply_(b, mask_col)

        # `n_msa=1` asks for the raw sum over rows plus proj_o's bias, unscaled: AF2's own
        # numerator. The divisor is applied below because it is per residue pair.
        # `_sum_rows` rather than `_small_depth`: same contraction, no `ttnn.repeat`, so the
        # backward can reach it. Both score 0.0040 against a float64 reference of the same
        # numbers (`perf/bcx_predictor/opm_unit.json`).
        z = self._sum_rows(a, b)

        # norm_ij = sum_s m_si m_sj, as a matmul over the row axis.
        mt = ttnn.permute(msa_mask, (1, 0))
        norm = ttnn.matmul(mt, msa_mask, compute_kernel_config=self.compute_kernel_config)
        ttnn.deallocate(mt)
        norm = ttnn.add_(norm, self.EPS)
        norm = ttnn.reshape(norm, (1, n, n, 1))
        out = ttnn.divide(z, norm)
        ttnn.deallocate(z)
        ttnn.deallocate(norm)
        return out


class AF2EvoformerBlock(AF2PairBlock):
    """One `EvoformerIteration`: the MSA track, the outer product mean, then the pair track.

    Subclasses `AF2PairBlock` for the reason the reference subclasses too. The checkpoint remap
    makes `evoformer.0.tri_mul_out` and `evoformer.0.opm` siblings, so a flat scope is the
    checkpoints own layout, and `outer_product_mean.first` is False so the MSA track runs first.
    """

    def __init__(self, state_dict: Weights,
                 compute_kernel_config: ttnn.DeviceComputeKernelConfig,
                 opm_first: bool = False, **kwargs):
        super().__init__(state_dict, compute_kernel_config, **kwargs)
        self.opm_first = opm_first
        self.msa_row_attn = AF2Attention(
            self.scope("msa_row_attn"), compute_kernel_config, pair_bias=True)
        self.msa_col_attn = AF2Attention(
            self.scope("msa_col_attn"), compute_kernel_config, column=True)
        self.msa_transition = ReluTransition(
            self.scope("msa_transition"), compute_kernel_config)
        # `scale_bias=True` puts the proj_o bias inside the division by the pair norm, which is
        # AF2s own semantics.
        self.opm = AF2MaskedOuterProductMean(
            self.scope("opm"), compute_kernel_config, scale_bias=True)

    def _mask_biases(self, msa_mask: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """`[rows, n]` mask -> the row and column additive logit biases.

        Built once per block rather than once per attention: the two variants are the same
        numbers in different layouts, because the row attention's keys are residues and the
        column attention's keys are rows.

        Both reshapes move a tile row into tiles of their own and write only its logical
        elements, so the new tiles' padding is whatever the buffer last held, which after a
        backward is often NaN or Inf. The broadcast add copies row 0's padding columns into the
        score tile's padding, and `softmax_in_place` masks a finite padding value but not a NaN:
        every row of that tile comes back NaN. On identical inputs the 48-block backward
        alternated clean, NaN, clean, NaN (perf/bcx_nan/). The padding is zeroed where it is
        created: tracing every op of a block's forward, these two reshapes are the only ones
        whose output padding is not finite (perf/bcx_nan/optrace_pad.json).
        """
        if len(msa_mask.shape) == 3:
            msa_mask = ttnn.reshape(msa_mask, tuple(msa_mask.shape)[1:])
        rows, n = (int(d) for d in msa_mask.shape)
        flat = ttnn.multiply(ttnn.subtract(msa_mask, 1.0), MASK_LOGIT_BIAS)
        row_bias = ttnn.fill_implicit_tile_padding(ttnn.reshape(flat, (rows, 1, 1, n)), 0.0)
        transposed = ttnn.permute(flat, (1, 0))
        col_bias = ttnn.fill_implicit_tile_padding(ttnn.reshape(transposed, (n, 1, 1, rows)), 0.0)
        return row_bias, col_bias

    def _opm_update(self, msa: ttnn.Tensor, z: ttnn.Tensor,
                    msa_mask: ttnn.Tensor | None) -> ttnn.Tensor:
        """The outer product mean residual, at either position in the block.

        AF2 divides the outer product mean by `eps + norm`, and at an all-ones mask the norm is
        the MSA depth everywhere. `eps` is 1e-3 and the trunk is bfloat16, whose spacing at 2.0
        is 0.0078, so `eps + norm` rounds back to the depth exactly -- at any depth, since the
        spacing scales with the value. Adding it anyway measures 1.7x worse on card at Evoformer
        0 and 47 (`device_gate.py --opm-eps 1e-3`). `None` reads the depth off the tensor, which
        is that divisor.
        """
        call = ((lambda m: self.opm(m, None)) if msa_mask is None
                else (lambda m: self.opm.masked(m, msa_mask)))
        return self._residual(z, self._update("opm", call, msa))

    def _msa_track(self, msa: ttnn.Tensor, pair: ttnn.Tensor,
                   row_bias: ttnn.Tensor | None = None,
                   col_bias: ttnn.Tensor | None = None) -> ttnn.Tensor:
        msa = self._residual(msa, self._update("msa_row_attn", self.msa_row_attn, msa, pair,
                                               row_bias))
        msa = self._residual(msa, self._update("msa_col_attn", self.msa_col_attn, msa, None,
                                               col_bias))
        msa = self._residual(msa, self._update("msa_transition", self.msa_transition, msa))
        return msa

    def __call__(self, msa: ttnn.Tensor, z: ttnn.Tensor,
                 msa_mask: ttnn.Tensor | None = None, mask: ttnn.Tensor | None = None,
                 attn_mask: ttnn.Tensor | None = None) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        row_bias, col_bias = (self._mask_biases(msa_mask) if msa_mask is not None
                              else (None, None))
        if self.opm_first:
            # multimer_v3 sets `outer_product_mean.first`, so the pair carries the MSA before
            # the row attention reads it as a bias. Same ops and same weights as the monomer
            # block; only this order differs, and it differs for all 52 blocks.
            z = self._opm_update(msa, z, msa_mask)
        msa = self._msa_track(msa, z, row_bias, col_bias)
        if not self.opm_first:
            z = self._opm_update(msa, z, msa_mask)
        return msa, super().__call__(z, mask, attn_mask)


class AF2SingleActivations(Module):
    """`evoformer/single_activations`: MSA row 0 -> the single track, on card.

    AF2 computes this inside the Evoformer scope, so its weights are trunk weights and it is the
    only path from the structure module's cotangent back into the MSA track. Keeping it on the
    device side of a trunk/tail split is what makes `(single, pair)` the hand-off: a split that
    hands over `(msa, pair)` instead receives exactly zero on the MSA side, because no head or
    loss BindCraft 2 runs reads the MSA track directly.
    """

    def __init__(self, state_dict: Weights,
                 compute_kernel_config: ttnn.DeviceComputeKernelConfig):
        super().__init__(state_dict, compute_kernel_config)
        self.weight = self.torch_to_tt("weight")
        self.bias = self.torch_to_tt("bias")

    def __call__(self, msa: ttnn.Tensor) -> ttnn.Tensor:
        """[.., rows, n, C_M] -> [.., 1, n, C_S], reading row 0 only."""
        rows = int(msa.shape[-3])
        row0 = msa if rows == 1 else msa[..., :1, :, :]
        return self._lin(row0, self.weight, bias=self.bias)


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
        self.device_single = None
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
                                                   opm_first=self.multimer,
                                                   fused_hifi=self._fused_hifi("evoformer"))
                                 for i in range(len(self.evoformer))]
        if self.template is not None:
            self.device_template = [
                AF2PairBlock(scoped(f"template.pair_stack.{i}."), ckc,
                             head_dim=TEMPLATE_TRI_ATT_HEAD_DIM,
                             n_heads=TEMPLATE_TRI_ATT_HEADS,
                             # The monomer's template stack runs the attentions first;
                             # multimer_v3's runs the Evoformer order.
                             evoformer_order=self.multimer,
                             fused_hifi=self._fused_hifi("template"))
                for i in range(len(self.template.pair_stack))]
            self._template_stack = AF2DeviceTemplatePairStack(
                self.device_template, self._up, self._down)
            self.set_template_host(self.template_host)
        self.device_single = AF2SingleActivations(scoped("single_activations."), ckc)
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

    def set_l1_padded_plan(self, enabled: bool) -> None:
        """Pin every AF2 attention's fp32-softmax L1 plan to the padded or the logical extent.

        The construction-time route is the `l1_padded_plan=True` AF2PairBlock and AF2Attention
        already carry; this one exists so an A/B does not have to go through
        `TT_BIO_FP32_SOFTMAX_L1_PADDED`, which is read at import and therefore costs a process an
        arm. It reaches AF2's blocks only: the protenix filter's `AttentionPairBias` runs in the
        same process on PXDesign and keeps following the env var either way.
        """
        for block in self._device_blocks:
            block.tri_att_start.l1_padded_plan = enabled
            block.tri_att_end.l1_padded_plan = enabled
        for block in self.device_evoformer:
            block.msa_row_attn.l1_padded_plan = enabled

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
        masks = af2_pair_masks(pair_mask, self._device)
        z = self._up(pair)
        for index, block in enumerate(self.device_extra_msa):
            if self.block_tap is not None:
                # The dead track, on host, only so the device leg owes the torch leg's taps. It
                # reads the block's INPUT pair, which is what the reference hands it.
                extra = self.extra_msa[index]._msa_track(extra, self._down(z, shape), extra_mask)
            # `outer_product_mean.first` does not reach this path: the constant does not
            # depend on the MSA, and it lands on the pair before the pair track either way.
            const = self._up(self.opm_constant[index].reshape(1, 1, -1))
            z = block(block._residual(z, const), *masks)
            if self.block_tap is not None:
                self._tap("extra_msa_stack", msa=extra, pair=self._down(z, shape))
        out = self._down(z, shape)
        ttnn.deallocate(z)
        return out

    def evoformer_stack(self, msa: torch.Tensor, pair: torch.Tensor, msa_mask: torch.Tensor,
                        pair_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.substitute:
            self._install_substitution(msa_mask, pair_mask)
        msa_shape, pair_shape = tuple(msa.shape), tuple(pair.shape)
        # One set of mask tensors for all 48 blocks. `None` on an all-ones mask, which keeps
        # every fold PXDesign runs today on the arithmetic it ran before, bit for bit.
        pair_masks = af2_pair_masks(pair_mask, self._device)
        msa_mask_tt = (None if bool((msa_mask == 1).all())
                       else self._up(msa_mask.float()))
        m, z = self._up(msa), self._up(pair)
        for block in self.device_evoformer:
            m, z = block(m, z, msa_mask_tt, *pair_masks)
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
