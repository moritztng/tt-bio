"""A tape-native pair track for one Protenix v2 pairformer block, with LoRA adapters.

WHY THIS EXISTS AND WHAT IT IS NOT. The production `tenstorrent.PairformerLayer` is
written against raw ttnn with every L1-residency and chunking lever tt-bio ships; it
cannot be differentiated and must not be slowed down to make it differentiable. So the
taped segment is a TWIN: the same maths on the same weights over `tt_bio.autograd`, and
`perf/ptxft/block_parity.py` holds it to the production layer by PCC. The twin is the
thing that is differentiated; production is the thing that is fast; parity is what makes
training the twin train the model that will serve.

THE PAIR TRACK IS THE WHOLE REACHABLE SET, and this is a structural fact rather than a
scoping choice. In `PairformerLayer.__call__` all five z updates --
tri_mul_out(z), tri_mul_in(z), tri_att_start(z), tri_att_end(z), transition_z(z) -- read
only z. s is updated FROM z (attention_pair_bias, transition_s) and never feeds back into
it. So across the 48-block stack z is a closed autoregression, and a distogram objective,
which reads z, has EXACTLY ZERO gradient into attention_pair_bias or single_transition --
2,513,280 of each block's 4,749,184 parameters, 52.9 %. Not a small gradient: zero, the
same way the diffusion module's 204 M is zero. `block_parity.py --s-independence`
measures it rather than trusting this paragraph.

Maths, read off the production modules rather than recalled:

* trimul (`TriangleMultiplication`): `zn = LN_in(z)`; `a = sigmoid(g_a(zn)) * p_a(zn)`,
  `b = sigmoid(g_b(zn)) * p_b(zn)`; contract outgoing `sum_k a[i,k,c] b[j,k,c]` or
  incoming `sum_k a[k,i,c] b[k,j,c]`; `LN_out`; `p_out`; gate by `sigmoid(g_out(zn))`,
  which reads the INPUT norm (tenstorrent.py, `g_out = _trimul_out_proj(x_norm_in, ...)`).
  tt-bio fuses the a/b halves into one `g_in`/`p_in` weight (`protenix_weights.
  remap_triangle_multiplication` concatenates on the out axis); this splits them back,
  which needs no slice op on the tape and puts the adapter at the upstream granularity.
* triangle attention (`TriangleAttention`): for the ending variant transpose the pair
  tensor first; `zn = LN(z)`; q/k/v from three linears reshaped to head-major
  `[S, H, S, d]` with the pair ROW as the leading axis; the bias is `linear(zn)` giving
  `[S, S, H]`, permuted to `[1, H, S, S]` and broadcast over that leading axis; gate by
  `sigmoid(g(zn))`; project with `o`. The scale convention matters and is the production
  one: `_bias_scale = sqrt(head_dim)` pre-multiplies the bias weight because the fused
  kernel computes `exp((qk + bias) / sqrt(d))`, so the equivalent unfused form adds the
  RAW bias to `qk / sqrt(d)`. Getting this backwards makes the softmax sqrt(d) ~ 5.7x too
  peaky, which is the documented root cause of the OpenFold3 MSA pair-stack degradation.
* transition (`Transition`): `zn = LN(z)`; `silu(fc1(zn)) * fc2(zn)`; `fc3`. silu is
  composed as `x * sigmoid(x)` from the tape's own ops, so it needs no new backward.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import ttnn

from tt_bio import autograd as ag
from tt_bio import finetune as ft

EPS = 1e-5  # every layer norm in the production block

# The 34 adapted linears of a block's pair track, by the name this module gives them.
# attention_pair_bias and single_transition are absent because a distogram loss cannot
# reach them at all -- see the module docstring.
PAIR_TRACK_TARGETS = tuple(
    [f"tri_mul_{d}.{n}" for d in ("out", "in")
     for n in ("g_in_a", "g_in_b", "p_in_a", "p_in_b", "p_out", "g_out")]
    + [f"tri_att_{d}.{n}" for d in ("start", "end")
       for n in ("linear", "mha.linear_q", "mha.linear_k", "mha.linear_v",
                 "mha.linear_g", "mha.linear_o")]
    + [f"transition_z.{n}" for n in ("fc1", "fc2", "fc3")]
)

# The other 52.9 % of a block. A distogram objective cannot reach these -- z is a closed
# autoregression, measured by `block_parity.py --s-independence` -- but the confidence
# head's pLDDT reads s_single, so a complete block backward needs them.
SINGLE_TRACK_TARGETS = tuple(
    [f"attention.proj_{n}" for n in ("q", "k", "v", "g", "o")]
    + ["attention.proj_z.1"]
    + [f"transition_s.{n}" for n in ("fc1", "fc2", "fc3")]
)


def _t2d(t, device, dtype):
    """A checkpoint tensor (out, in) -> a frozen tape Tensor in ttnn's (in, out) layout.

    `.t()` is what `Module.torch_to_tt` applies by default, so this is the same layout the
    production layer reads. On a 1-D norm gain `.t()` is a no-op, so the reshape below is
    what gives it a rank the pair tensor can broadcast against.
    """
    import numpy as np
    a = t.detach().to("cpu").float().numpy()
    a = a.T if a.ndim == 2 else a.reshape(1, 1, -1)
    return ag.Tensor(ft.to_device(np.ascontiguousarray(a), device, dtype=dtype),
                     requires_grad=False)


class PairTrackBlock:
    """One pairformer block's pair track, differentiable, optionally LoRA-adapted.

    ``sd`` is the output of ``protenix_weights.remap_pairformer_block`` for one block.
    ``adapt`` is the subset of ``PAIR_TRACK_TARGETS`` to give adapters; ``()`` means the
    block is frozen, which is what the blocks below the adapted window run as.
    """

    def __init__(self, sd: Dict, device, *, n_heads: int = 8, head_dim: int = 32,
                 dtype=ttnn.bfloat16, adapter_dtype=ttnn.float32,
                 lora: Optional[ft.LoraConfig] = None, adapt=(), rng=None,
                 chunk: Optional[int] = None, q_chunk: Optional[int] = None,
                 train_base: bool = False):
        import numpy as np
        self.device, self.dtype = device, dtype
        self.n_heads, self.head_dim = n_heads, head_dim
        self.chunk, self.q_chunk = chunk, q_chunk
        self.lora = lora
        self.adapt = set(adapt)
        self.rng = np.random.default_rng() if rng is None else rng
        self.params: Dict[str, ag.Tensor] = {}      # the TRAINABLE set
        self.w: Dict[str, ag.Tensor] = {}           # the frozen set

        def take(key, name=None):
            self.w[name or key] = _t2d(sd[key], device, dtype)

        for d in ("out", "in"):
            p = f"tri_mul_{d}"
            src = "tri_mul_out" if d == "out" else "tri_mul_in"
            for n in ("norm_in.weight", "norm_in.bias", "norm_out.weight", "norm_out.bias"):
                take(f"{src}.{n}", f"{p}.{n}")
            # g_in / p_in arrive concatenated on the OUT axis; split them back.
            for fused, halves in (("g_in", ("g_in_a", "g_in_b")),
                                  ("p_in", ("p_in_a", "p_in_b"))):
                w = sd[f"{src}.{fused}.weight"]
                h = int(w.shape[0]) // 2
                for i, half in enumerate(halves):
                    self.w[f"{p}.{half}"] = _t2d(w[i * h:(i + 1) * h], device, dtype)
            self.w[f"{p}.p_out"] = _t2d(sd[f"{src}.p_out.weight"], device, dtype)
            self.w[f"{p}.g_out"] = _t2d(sd[f"{src}.g_out.weight"], device, dtype)

        for d in ("start", "end"):
            p = f"tri_att_{d}"
            for n in ("layer_norm.weight", "layer_norm.bias"):
                take(f"{p}.{n}")
            self.w[f"{p}.linear"] = _t2d(sd[f"{p}.linear.weight"], device, dtype)
            for n in ("q", "k", "v", "g", "o"):
                self.w[f"{p}.mha.linear_{n}"] = _t2d(
                    sd[f"{p}.mha.linear_{n}.weight"], device, dtype)

        for n in ("norm.weight", "norm.bias"):
            take(f"transition_z.{n}")
        for n in ("fc1", "fc2", "fc3"):
            self.w[f"transition_z.{n}"] = _t2d(sd[f"transition_z.{n}.weight"], device, dtype)

        # The single track. attention.proj_q is the block's only linear WITH a bias, and
        # proj_z.1 is stored unscaled here: production folds sqrt(head_dim) into that
        # weight because its fused kernel computes exp((qk + bias)/sqrt(d)), while this
        # adds the raw bias to an already-scaled qk. Same arithmetic, different place to
        # put the constant, and getting it wrong makes the softmax sqrt(d) too peaky.
        for n in ("pre_norm_s.weight", "pre_norm_s.bias",
                  "attention.proj_z.0.weight", "attention.proj_z.0.bias",
                  "transition_s.norm.weight", "transition_s.norm.bias"):
            take(n)
        zw = sd["attention.proj_z.1.weight"]
        self.s_heads = H = int(zw.shape[0])
        self.s_head_dim = hd = int(sd["attention.proj_q.weight"].shape[0]) // H
        # THE HEAD DIMENSION IS 24, WHICH IS NOT A TILE. 384 channels over 16 heads gives
        # 24 lanes per head, and a ttnn tile is 32x32, so [1, H, S, 24] in TILE_LAYOUT is
        # a padded tensor whose last 8 lanes are not defined by us. Production solves this
        # at load time -- `head_dim_padding = -head_dim % 32` and `_pad_head_lanes` in
        # `tenstorrent.AttentionPairBias.__init__` -- and the twin has to solve it the same
        # way, because the alternative is a matmul over 32 lanes of which 8 are whatever
        # the allocator left there. Unpadded, this reads 1.51e-01 relative against a
        # float64 reference on a forward that is otherwise correct.
        # q/k/v gain zero lanes on their OUTPUT axis, o gains zero ROWS on its INPUT axis
        # so the pad contributes nothing, and g is re-laned to line up with them.
        self.s_pad_dim = pd = hd + (-hd % 32)

        def _lanes(w, axis):
            """Pad each head's block from hd to pd lanes with zeros, along `axis`."""
            import torch as _t
            if pd == hd:
                return w
            shp = list(w.shape)
            shp[axis:axis + 1] = [H, hd]
            w = w.reshape(shp)
            pad = [0] * (2 * len(shp))
            pad[2 * (len(shp) - 1 - (axis + 1))] = 0
            pad[2 * (len(shp) - 1 - (axis + 1)) + 1] = pd - hd
            w = _t.nn.functional.pad(w, pad)
            shp[axis:axis + 2] = [H * pd]
            return w.reshape(shp)

        self.w["attention.proj_q.bias"] = _t2d(_lanes(sd["attention.proj_q.bias"], 0),
                                               device, dtype)
        for n in ("q", "k", "v", "g"):
            self.w[f"attention.proj_{n}"] = _t2d(
                _lanes(sd[f"attention.proj_{n}.weight"], 0), device, dtype)
        self.w["attention.proj_o"] = _t2d(_lanes(sd["attention.proj_o.weight"], 1),
                                          device, dtype)
        self.w["attention.proj_z.1"] = _t2d(sd["attention.proj_z.1.weight"], device, dtype)
        for n in ("fc1", "fc2", "fc3"):
            self.w[f"transition_s.{n}"] = _t2d(sd[f"transition_s.{n}.weight"], device, dtype)

        # Full-parameter training: every base weight becomes a trainable fp32 leaf.
        # Used by the overfit-from-random-init arm, where there is no pretrained weight
        # for an adapter to sit beside, so LoRA would be adapting noise. fp32 for the
        # same reason the adapters are: the update-magnitude control reads 0.767 of a
        # step at lr 1e-5 on a bf16 leaf and 1.000 on an fp32 one.
        if train_base:
            for _n, _t in list(self.w.items()):
                _new = ag.Tensor(ft.to_device(ft.to_host(_t.value), device,
                                              dtype=adapter_dtype), requires_grad=True)
                self.w[_n] = _new
                self.params[_n] = _new

        # Adapters, after every base weight exists so `in`/`out` come from the real shapes.
        self.adapters: Dict[str, tuple] = {}
        if lora is not None and self.adapt:
            for name in sorted(self.adapt):
                base = self.w[name].value
                fin, fout = int(base.shape[-2]), int(base.shape[-1])
                a, b = ft.lora_factors(fin, fout, lora, device, dtype=adapter_dtype,
                                       rng=self.rng)
                self.adapters[name] = (a, b)
                self.params[f"{name}.lora_A"] = a
                self.params[f"{name}.lora_B"] = b

    # ------------------------------------------------------------------ pieces

    def _lin(self, x: ag.Tensor, name: str, bias: Optional[str] = None) -> ag.Tensor:
        """The base linear, adapted if this site has an adapter."""
        w = self.w[name]
        b0 = self.w[bias] if bias else None
        if name in self.adapters:
            a, b = self.adapters[name]
            return ft.lora_linear(x, w, a, b, b0, scaling=self.lora.scaling)
        return ag.linear(x, w, b0)

    def _ln(self, x: ag.Tensor, prefix: str) -> ag.Tensor:
        return ag.layer_norm(x, self.w[f"{prefix}.weight"], self.w[f"{prefix}.bias"],
                             eps=EPS)

    def _silu(self, x: ag.Tensor) -> ag.Tensor:
        """x * sigmoid(x), composed from the tape rather than added as an op."""
        return ag.mul(x, ag.sigmoid(x))

    def trimul(self, z: ag.Tensor, p: str, *, incoming: bool) -> ag.Tensor:
        zn = self._ln(z, f"{p}.norm_in")
        a = ag.mul(ag.sigmoid(self._lin(zn, f"{p}.g_in_a")), self._lin(zn, f"{p}.p_in_a"))
        b = ag.mul(ag.sigmoid(self._lin(zn, f"{p}.g_in_b")), self._lin(zn, f"{p}.p_in_b"))
        out = ag.pair_contract(a, b, incoming=incoming)
        out = self._ln(out, f"{p}.norm_out")
        out = self._lin(out, f"{p}.p_out")
        return ag.mul(ag.sigmoid(self._lin(zn, f"{p}.g_out")), out)

    def triatt(self, z: ag.Tensor, p: str, *, ending: bool) -> ag.Tensor:
        H, d = self.n_heads, self.head_dim
        if ending:
            z = ag.permute(z, (1, 0, 2))
        S = int(z.value.shape[0])
        zn = self._ln(z, f"{p}.layer_norm")

        def heads(name):
            x = self._lin(zn, f"{p}.mha.linear_{name}")          # [S, S, H*d]
            return ag.permute(ag.reshape(x, [S, S, H, d]), (0, 2, 1, 3))  # [S, H, S, d]

        q, k, v = heads("q"), heads("k"), heads("v")
        # RAW bias against a qk already scaled by 1/sqrt(d): see the module docstring on
        # `_bias_scale`. [S, S, H] -> [1, H, S, S], broadcast over the leading pair row.
        bias = ag.reshape(ag.permute(self._lin(zn, f"{p}.linear"), (2, 0, 1)), [1, H, S, S])
        o = ag.triangle_attention(q, k, v, bias, scale=d ** -0.5,
                                  chunk=self.chunk, q_chunk=self.q_chunk)
        o = ag.reshape(ag.permute(o, (0, 2, 1, 3)), [S, S, H * d])
        o = ag.mul(o, ag.sigmoid(self._lin(zn, f"{p}.mha.linear_g")))
        out = self._lin(o, f"{p}.mha.linear_o")
        return ag.permute(out, (1, 0, 2)) if ending else out

    def transition(self, z: ag.Tensor, p: str = "transition_z") -> ag.Tensor:
        zn = self._ln(z, f"{p}.norm")
        x1 = self._silu(self._lin(zn, f"{p}.fc1"))
        x2 = self._lin(zn, f"{p}.fc2")
        return self._lin(ag.mul(x1, x2), f"{p}.fc3")

    def attention_pair_bias(self, s: ag.Tensor, z: ag.Tensor) -> ag.Tensor:
        """`AttentionPairBias` over the single representation, biased by the pair one.

        The same shape as `triatt` with the pair-row axis collapsed to 1: q/k/v are
        [1, H, S, d] and the bias is [1, H, S, S], so it reuses the tape'"'"'s already
        gradchecked triangle_attention rather than adding a second attention backward.
        """
        H, d, pd = self.s_heads, self.s_head_dim, self.s_pad_dim
        S = int(s.value.shape[-2])
        sn = self._ln(s, "pre_norm_s")

        def heads(name, bias=None):
            x = self._lin(sn, f"attention.proj_{name}", bias=bias)
            return ag.permute(ag.reshape(x, [1, S, H, pd]), (0, 2, 1, 3))

        # s carries a leading batch axis, [1, S, c_s], because every layer-norm gain in
        # this tree is stored as (1, 1, C) by _t2d and broadcasting one against a rank-2
        # input silently promotes the OUTPUT to rank 3, which then fails the residual add
        # with a shape error one op later rather than where it happened.

        q = heads("q", bias="attention.proj_q.bias")
        k, v = heads("k"), heads("v")
        zn = self._ln(z, "attention.proj_z.0")
        bias = ag.reshape(ag.permute(self._lin(zn, "attention.proj_z.1"), (2, 0, 1)),
                          [1, H, S, S])
        o = ag.triangle_attention(q, k, v, bias, scale=d ** -0.5,
                                  chunk=self.chunk, q_chunk=self.q_chunk)
        o = ag.reshape(ag.permute(o, (0, 2, 1, 3)), [1, S, H * pd])
        o = ag.mul(o, ag.sigmoid(self._lin(sn, "attention.proj_g")))
        return self._lin(o, "attention.proj_o")

    def single(self, s: ag.Tensor, z: ag.Tensor) -> ag.Tensor:
        """The two s updates, in the production layer'"'"'s order (tenstorrent.py:9300-9316)."""
        s = ag.add(s, self.attention_pair_bias(s, z))
        return ag.add(s, self.transition(s, "transition_s"))

    # ------------------------------------------------------------------ the block

    def __call__(self, z: ag.Tensor) -> ag.Tensor:
        """The five residual updates, in the production layer's order."""
        z = ag.add(z, self.trimul(z, "tri_mul_out", incoming=False))
        z = ag.add(z, self.trimul(z, "tri_mul_in", incoming=True))
        z = ag.add(z, self.triatt(z, "tri_att_start", ending=False))
        z = ag.add(z, self.triatt(z, "tri_att_end", ending=True))
        z = ag.add(z, self.transition(z))
        return z


class DistogramHead:
    """The objective's own head: one `(64, c_z)` linear plus bias, trained in full.

    Trained rather than adapted: a rank-8 adapter on a (64, 256) linear is 2560 params
    against the head's own 16,448, so LoRA there buys nothing and costs a hyperparameter.
    """

    def __init__(self, sd: Dict, device, *, dtype=ttnn.bfloat16,
                 param_dtype=ttnn.float32, trainable: bool = True):
        import numpy as np
        w = sd["distogram_head.linear.weight"].detach().to("cpu").float().numpy().T
        b = sd["distogram_head.linear.bias"].detach().to("cpu").float().numpy().reshape(1, -1)
        dt = param_dtype if trainable else dtype
        self.w = ag.Tensor(ft.to_device(np.ascontiguousarray(w), device, dtype=dt),
                           requires_grad=trainable)
        self.b = ag.Tensor(ft.to_device(np.ascontiguousarray(b), device, dtype=dt),
                           requires_grad=trainable)
        self.params = ({"distogram_head.weight": self.w, "distogram_head.bias": self.b}
                       if trainable else {})

    def __call__(self, z: ag.Tensor) -> ag.Tensor:
        return ag.linear(z, self.w, self.b)


def symmetrize_bins(t):
    """The distogram is over UNORDERED pairs, so the head's output is symmetrised.

    Upstream is a SUM, not a mean: Protenix's DistogramHead.forward is
    `logits = linear(z); logits = logits + logits.transpose(-2, -3)`
    (protenix/model/modules/head.py:54-56, AF3 Algorithm 1 line 17). The mean this
    originally used computes exactly HALF of that head, bias included, which flattens
    every predicted distribution toward uniform and scales the loss gradient with it.
    Argmax is scale-invariant, so no calibration readout can see the difference -- only
    the cross-entropy can, which is why this is fixed against the source rather than
    against a plot.

    It is applied to the LOGITS, not to z, which is what upstream does too. The same
    function symmetrises the gradient seed: for S = L + L^T the seed on L is g + g^T,
    the identical operation.
    """
    return t + t.transpose(1, 0, 2)
