"""ESMFold2 on Tenstorrent (ttnn) — folding trunk.

ESMFold2 is a diffusion all-atom structure predictor built on ESMC-6B
embeddings. Reference architecture: the Biohub fork of transformers
(models/esmfold2/modeling_esmfold2_common.py), see tests/esmfold2_reference.py.

This module implements the `folding_trunk`: 48 PairUpdateBlocks, each
    z = z + tri_mul_out(z); z = z + tri_mul_in(z); z = z + pair_transition(z)
operating on the pair representation z [B, L, L, c_z=256].

The triangle-multiplication math is identical to AlphaFold3 / Boltz, so we
reuse tt-bio's proven ttnn `TriangleMultiplication` kernel and the ESMC
`SwiGLUFFN`, remapping ESMFold2's bundled weights into the expected layout:
  proj_bundle[:512]->p_in, proj_bundle[512:]->g_in, proj_emit->p_out,
  proj_gate->g_out, norm_start->norm_in, norm_mix->norm_out.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F
import ttnn

from tt_bio.esmc import SwiGLUFFN, apply_rotary
from tt_bio.envflags import env_flag
from tt_bio.tenstorrent import (
    CORE_GRID_MAIN,
    Module,
    TorchWrapper,
    TriangleMultiplication,
    Weights,
    WeightScope,
    capture_trace,
    _accurate_softmax,
    _sdpa_program_config_for_lengths,
    accurate_softmax_site,
    attn_value_matmul,
    fused_sdpa,
    lever,
    _ColWin,
    _win_addcmul,
    sdpa32_mask,
    sdpa32_rows,
    _sdpa32_fused,
    trunk_compute_kernel_config,
    trace_region_size,
)
from tt_bio.eltwise_fusion import scale_add
from tt_bio.addcmul_at import addcmul_at

_ROW = lambda x: x.reshape(1, -1)

_DTYPE = ttnn.bfloat16  # activation dtype for the diffusion / encoders

# Cache step-invariant diffusion quantities (pair_bias from z, atom conditioning
# c, the atom adaLN modulation) across the ~20 sampling steps, keyed by source-
# tensor identity. Bit-identical to recomputing.
_STEP_CACHE = True

# The token DiT's q/k/v/g with the 48 -> 64 head padding folded into the weights (AttentionPairBias
# ._folded): two matmuls per block instead of six plus a chunk and a concat. Same arithmetic.
_APB_FOLD = env_flag("TT_BIO_ESMFOLD2_APB_FOLD", False)


# Optional progress callback for the CLI display: report_progress(stage, step, total).
# A no-op unless a worker installs one via set_progress(); never affects compute.
_PROGRESS = None

def set_progress(fn):
    global _PROGRESS
    _PROGRESS = fn

def report_progress(stage, step=0, total=0):
    if _PROGRESS is not None:
        try:
            _PROGRESS(stage, step, total)
        except Exception:
            pass


# Optional trajectory hook: dump(step, x, x_denoised) receives the sampler's real coordinates,
# step -1 being the initial noise and x_denoised the network's prediction at that step. Off unless
# set_trajectory_dump() installs one. It only reads tensors the sampler already made, so a fold is
# bit-identical with or without it. An exception raised by the hook aborts the fold.
_DUMP = None

# The atom transformer's SwiGLU up-projection as two linears (one per half of w_up) instead of one linear and a
# ttnn.chunk: 6 atom blocks x every sampler step, the chunk was 0.30 s of device time per 512-aa fold (census c3).
ATOM_FFN_SPLIT = env_flag("TT_BIO_ESMFOLD2_ATOM_FFN_SPLIT", None)  # unset: the ef2_atom_ffn_split lever


def gate(flag, name):
    """A branch-stack switch: its setting when given, else whether lever `name` is active in the fold's scope.
    esmfold2_runtime.fold_levers decides which ef2_* levers each mode and arch carries."""
    return lever(name) if flag is None else flag


#: Replay the diffusion step from a ttnn trace (StructureHead._step) when the device was opened
#: with get_device(trace="esmfold2"). Lossless. Off: it lost 0.46 s/fold on p150a while the dense
#: atom SDPA kept the step device-bound; re-measured with the windowed atom attention.
DIFFUSION_TRACE = env_flag("TT_BIO_ESMFOLD2_DIFFUSION_TRACE", False)
DIFFUSION_TRACE_STATS = [0, 0]  # [steps replayed, steps run eager]


def set_diffusion_trace(on: bool) -> None:
    global DIFFUSION_TRACE
    DIFFUSION_TRACE = bool(on)


def set_trajectory_dump(fn):
    global _DUMP
    _DUMP = fn


def _attn_fp32(q, k, v, attn_mask, scale, ck, accurate_softmax: bool = False):
    """Manual fp32 attention (matmul + softmax + matmul). Used for the token
    AttentionPairBias, whose logits are NOT qk-normed and reach ~100+, making the
    softmax argmax too peaked for bf16 SDPA precision. q/k/v [B,H,L,Dp].

    This is ESMFold2's only softmax site: 100% of its softmax volume, and the shared
    ``accurate_softmax`` keyword on tenstorrent.AttentionPairBias never reached it, because
    ESMFold2 has its own AttentionPairBias class. ``accurate_softmax`` swaps the fused
    kernel, whose rows sum to 0.9944 here, for the 5-op chain."""
    f32 = ttnn.float32
    qf = ttnn.typecast(q, f32); kf = ttnn.typecast(k, f32); vf = ttnn.typecast(v, f32)
    kt = ttnn.permute(kf, (0, 1, 3, 2))  # [B,H,Dp,L]
    logits = ttnn.matmul(qf, kt, compute_kernel_config=ck, dtype=f32)
    if attn_mask is None:
        logits = ttnn.multiply(logits, scale)
    else:
        logits = scale_add(logits, scale, ttnn.typecast(attn_mask, f32))
    attn = (_accurate_softmax(logits, ck) if accurate_softmax
            else ttnn.softmax(logits, dim=-1))  # over keys
    ctx = attn_value_matmul(attn, vf, ck, f32)  # [B,H,L,Dp]
    return ttnn.typecast(ctx, _DTYPE)


def _sdpa32_mask(bias, scale):
    """The cached pair bias as the `_sdpa32` mask under dit_qkvf: fp32, divided by `scale` (the program scales its mask with
    the scores), padded to `sdpa32_rows`. Built once per block per fold, like the bias it replaces."""
    f32 = ttnn.float32
    b = ttnn.typecast(bias, f32)
    m = ttnn.multiply(b, 1.0 / scale)
    ttnn.deallocate(b)
    if sdpa32_rows(int(m.shape[-1])) == int(m.shape[-1]):
        return m
    mp = sdpa32_mask(m)
    ttnn.deallocate(m)
    return mp


def _sdpa_bf16(q, k, v, attn_mask, scale):
    """Scaled-dot-product attention; ttnn SDPA needs bf16, so cast i/o when _DTYPE is fp32."""
    if _DTYPE == ttnn.bfloat16:
        return fused_sdpa(
            q, k, v, attn_mask=attn_mask, scale=scale,
            program_config=_sdpa_program_config_for_lengths(q.shape[2], k.shape[2], q.shape[0] * q.shape[1], site="esmfold2", d=q.shape[3]))
    qb, kb, vb = (ttnn.typecast(t, ttnn.bfloat16) for t in (q, k, v))
    mb = ttnn.typecast(attn_mask, ttnn.bfloat16) if attn_mask is not None else None
    ctx = fused_sdpa(
        qb, kb, vb, attn_mask=mb, scale=scale,
        program_config=_sdpa_program_config_for_lengths(q.shape[2], k.shape[2], q.shape[0] * q.shape[1], site="esmfold2", d=q.shape[3]))
    ctx = ttnn.typecast(ctx, _DTYPE)
    ttnn.deallocate(qb); ttnn.deallocate(kb); ttnn.deallocate(vb)
    return ctx

C_Z = 256  # pair channels
from .token_axis import bucket_multiple as _bucket_multiple, pad_amount

# token-dim tile bucket for the folding trunk (pads + masks + slices), from the fleet value.
PAD_MULTIPLE = _bucket_multiple("esmfold2")
#                    Stays at 32 (not 64): the trimul's matmul contraction size depends on
#                    the padded length, and in bf16 that is NOT exact across sizes — a 64
#                    bucket measurably shifts RMSD-vs-truth (~0.065 Å on ubiquitin). Unlike
#                    the ESMC LM (whose padded attention is masked exactly → bit-identical
#                    hidden states), the trunk can't be re-bucketed without an accuracy hit.


def _remap_trimul(sd: dict, prefix: str) -> WeightScope:
    """ESMFold2 TriangleMultiplicativeBlock weights -> tt-bio TriangleMultiplication."""
    g = lambda k: sd[f"{prefix}.{k}"]
    pb = g("proj_bundle.weight")  # [4*latent, c_z]; rows: signal(2*latent) | gate(2*latent)
    half = pb.shape[0] // 2
    return WeightScope({
        "norm_in.weight": g("norm_start.weight"),
        "norm_in.bias": g("norm_start.bias"),
        "norm_out.weight": g("norm_mix.weight"),
        "norm_out.bias": g("norm_mix.bias"),
        "p_in.weight": pb[:half],   # signal (left|right values)
        "g_in.weight": pb[half:],   # gate (left|right gates)
        "p_out.weight": g("proj_emit.weight"),
        "g_out.weight": g("proj_gate.weight"),
    })


def _remap_transition(sd: dict) -> WeightScope:
    """ESMFold2 pair_transition (LN + SwiGLUMLP w12/w3) -> ESMC SwiGLUFFN keys."""
    return WeightScope({
        "0.weight": sd["pair_transition.norm.weight"],
        "0.bias": sd["pair_transition.norm.bias"],
        "1.weight": sd["pair_transition.ffn.w12.weight"],
        "3.weight": sd["pair_transition.ffn.w3.weight"],
    })


class PairUpdateBlock(Module):
    """z = z + tri_mul_out(z); z = z + tri_mul_in(z); z = z + pair_transition(z)."""

    def __init__(self, state_dict: Weights, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        sd = self.weights.as_dict()
        # tri_mul_out = outgoing (ending=False); tri_mul_in = incoming (ending=True).
        self.tri_out = TriangleMultiplication(
            False, _remap_trimul(sd, "tri_mul_out._engine"), compute_kernel_config,
            gated_move=True,
        )
        self.tri_in = TriangleMultiplication(
            True, _remap_trimul(sd, "tri_mul_in._engine"), compute_kernel_config,
            gated_move=True,
        )
        self.transition = SwiGLUFFN(
            _remap_transition(sd), compute_kernel_config, fuse_swiglu=True
        )

    def __call__(self, z: ttnn.Tensor, mask: ttnn.Tensor | None = None) -> ttnn.Tensor:
        # `z + tri(z)` through the trimul's own residual path, as the Pairformer runs it: the
        # in-place add (`_add_input`), or under the trimul_tail lever the add folded into the
        # tail's epilogue, which saves one pair-tensor pass per trimul.
        z = self.tri_out(z, mask, add_to_input=True)
        z = self.tri_in(z, mask, add_to_input=True)
        # `z + transition(z)`, but the add belongs to the transition: on the row-blocked path it
        # runs per block so fc2's output never round-trips DRAM (lever F, esmc.PAIR_FFN_FUSED_
        # RESIDUAL). Bit-exact -- the add is elementwise and row-independent.
        out = self.transition.residual(z)
        if out is not z:  # ws_resid wrote into z
            ttnn.deallocate(z)
        return out


class FoldingTrunkModel(Module):
    """ModuleList of PairUpdateBlocks over the pair representation z."""

    def __init__(self, n_layers: int, state_dict: Weights, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self.blocks = [
            PairUpdateBlock(self.scope(f"blocks.{i}"), compute_kernel_config)
            for i in range(n_layers)
        ]

    def __call__(self, z: ttnn.Tensor, mask: ttnn.Tensor | None = None) -> ttnn.Tensor:
        for block in self.blocks:
            z = block(z, mask)
        return z


class FoldingTrunk(TorchWrapper):
    """Top-level folding trunk (torch z [B,L,L,256] in/out)."""

    def __init__(self, n_layers: int = 48, d_pair: int = C_Z):
        super().__init__()
        self.n_layers = n_layers
        self.d_pair = d_pair

    def _create_module(self, weights: WeightScope) -> FoldingTrunkModel:
        # The trunk's matmul fidelity, as every Pairformer-family trunk takes it (`trunk_hifi3`).
        return FoldingTrunkModel(self.n_layers, weights,
                                 trunk_compute_kernel_config(self.compute_kernel_config))

    def forward(self, z):
        # Pad both sequence axes to a tile multiple and mask padding out of the
        # triangle contraction, then slice back (mirrors tt-bio PairformerModule).
        seq_len = z.shape[1]
        pad = pad_amount(seq_len, PAD_MULTIPLE)
        mask = None
        if pad:
            z = F.pad(z, (0, 0, 0, pad, 0, pad))
            real = torch.zeros(1, z.shape[1], z.shape[1])
            real[:, :seq_len, :seq_len] = 1.0
            mask = self._from_torch(real)
        out = self.module(self._from_torch(z), mask)
        return self._to_torch(out)[:, :seq_len, :seq_len, :]


# ===========================================================================
# Diffusion structure head — token transformer (DiT with pair bias)
# ===========================================================================


class AdaLN(Module):
    """Adaptive LayerNorm (adaLN-Zero): sigmoid(s_gate(LN_s(s)))*LN(a) + s_shift(LN_s(s))."""

    def __init__(self, state_dict: Weights, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self.s_scale = self.torch_to_tt("s_scale", transform=lambda x: x)
        self.s_gate_w = self.torch_to_tt("s_gate.weight")
        self.s_gate_b = self.torch_to_tt("s_gate.bias", transform=_ROW)
        self.s_shift_w = self.torch_to_tt("s_shift.weight")

    def folded(self):
        """(scale weight, scale bias, shift weight) on the un-weighted LN(s): the s-norm's gamma folded in."""
        g = self.weights["s_scale"].float()[:, None]
        return (g * self.weights["s_gate.weight"].float().t(), self.weights["s_gate.bias"].float(),
                g * self.weights["s_shift.weight"].float().t())

    def __call__(self, a: ttnn.Tensor, s: ttnn.Tensor, terms=None) -> ttnn.Tensor:
        """`terms`: this block's (sigmoid scale, shift) windows of the step's wide conditioning (dit_shoist)."""
        ck = self.compute_kernel_config
        a_norm = ttnn.layer_norm(a, epsilon=1e-5, compute_kernel_config=ck)
        if terms is not None:
            (sc_t, sc_off, w), (sh_t, sh_off, _) = terms
            out = addcmul_at(sh_t, a_norm, sc_t, w, x_off=sh_off, z_off=sc_off)
            if out is None:
                sc, sh = terms[0].cut(), terms[1].cut()
                out = ttnn.addcmul(sh, a_norm, sc, value=1.0)
                ttnn.deallocate(sc); ttnn.deallocate(sh)
            ttnn.deallocate(a_norm)
            return out
        s_norm = ttnn.layer_norm(s, weight=self.s_scale, epsilon=1e-5, compute_kernel_config=ck)
        gate = ttnn.sigmoid(self._lin(s_norm, self.s_gate_w, bias=self.s_gate_b))
        shift = self._lin(s_norm, self.s_shift_w)
        return ttnn.add(ttnn.multiply(gate, a_norm), shift)


class AttentionPairBias(Module):
    """Gated multi-head attention conditioned on s (adaLN) with per-head pair bias from z."""

    def __init__(self, num_heads: int, state_dict: Weights, compute_kernel_config,
                 accurate_softmax: bool = False):
        super().__init__(state_dict, compute_kernel_config)
        self.num_heads = num_heads
        self.accurate_softmax = accurate_softmax
        self.adaln = AdaLN(self.scope("adaln"), compute_kernel_config)
        self.q_w = self.torch_to_tt("q_proj.weight")
        self.q_b = self.torch_to_tt("q_proj.bias", transform=_ROW)
        self.kv_w = self.torch_to_tt("kv_proj.weight")
        self.g_w = self.torch_to_tt("g_proj.weight")
        self.out_gate_w = self.torch_to_tt("out_gate.weight")
        self.out_gate_b = self.torch_to_tt("out_gate.bias", transform=_ROW)
        self.pair_norm_w = self.torch_to_tt("pair_norm.weight")
        self.pair_norm_b = self.torch_to_tt("pair_norm.bias")
        self.pair_bias_w = self.torch_to_tt("pair_bias_proj.weight")
        self._pb = None       # cached pair_bias (see __call__)
        self._pb_src = None   # identity of the z it was built from

        # head_dim may not be a tile multiple (e.g. 768/16=48 -> padded to 64 by
        # the nlp head ops). Fold the head un-padding into two constant scatter
        # matmuls: pad g (d_model->Dpad) and a padded out_proj (Dpad->d_model).
        d_model = self.weights["out_proj.weight"].shape[0]
        H, hd = num_heads, d_model // num_heads
        self.head_dim = hd
        hdp = -(-hd // 32) * 32  # round up to tile multiple
        self.head_dim_pad = hdp
        Dpad = H * hdp
        Sg = torch.zeros(d_model, Dpad)  # g[.,768] @ Sg -> g[.,1024] (real slots only)
        for h in range(H):
            Sg[h * hd : (h + 1) * hd, h * hdp : h * hdp + hd] = torch.eye(hd)
        owt = self.weights["out_proj.weight"].t()  # [in d_model, out d_model]
        Op = torch.zeros(Dpad, d_model)  # padded out_proj input dim
        for h in range(H):
            Op[h * hdp : h * hdp + hd, :] = owt[h * hd : (h + 1) * hd, :]
        to_tt = lambda x: ttnn.from_torch(
            x, layout=ttnn.TILE_LAYOUT, device=self.device, dtype=_DTYPE
        )
        self.scatter_g = to_tt(Sg)
        self.out_w_padded = to_tt(Op)
        self._Sg = Sg
        self._fold = None  # (qkv_w, qkv_b, g_w) with the head padding folded in, built on first use

    def _folded(self):
        """q/k/v/g projections with scatter_g folded into the weights: W @ Sg only moves columns, so
        x @ (W @ Sg) is the padded projection exactly, one matmul for q|k|v and one for g."""
        if self._fold is None:
            Sg, d = self._Sg, self._Sg.shape[0]
            W = lambda k: self.weights[k].float()
            kv = W("kv_proj.weight")
            qkv = torch.cat([W("q_proj.weight").t() @ Sg, kv[:d].t() @ Sg, kv[d:].t() @ Sg], dim=1)
            b = torch.cat([W("q_proj.bias").reshape(1, -1) @ Sg, torch.zeros(1, 2 * Sg.shape[1])], dim=1)
            to_tt = lambda x: ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=self.device, dtype=_DTYPE)
            self._fold = (to_tt(qkv), to_tt(b), to_tt(W("g_proj.weight").t() @ Sg))
        return self._fold

    def __call__(self, a: ttnn.Tensor, s: ttnn.Tensor, z: ttnn.Tensor, terms=None) -> ttnn.Tensor:
        """`terms` (dit_shoist): the adaLN windows; the output then comes back without its s-gate, which the
        caller applies with the residual add."""
        ck = self.compute_kernel_config
        d_model = a.shape[-1]
        head_dim = d_model // self.num_heads

        x = self.adaln(a, s, terms)
        s = s if terms is None else None  # no s: _out skips the gate
        lin = self._lin
        if lever("dit_qkvf"):
            # dit_qkvf: one q|k|v projection (head padding folded in) read straight by the shared `_sdpa32_fused`
            # program, which writes the heads concatenated on their padded lanes: no head split, no casts, no
            # re-assembly. The token axis arrives padded to sdpa32_rows (DiffusionTransformerModel).
            qkv_w, qkv_b, g_w = self._folded()
            qkv = ttnn.unsqueeze(lin(x, qkv_w, qkv_b), 1)
            g_pad = ttnn.sigmoid(lin(x, g_w))
            ttnn.deallocate(x)
            B, _, n, w = (int(d) for d in qkv.shape)
            ctx = _sdpa32_fused(qkv, self._pair_bias(z, head_dim, ck), head_dim**-0.5, self.num_heads,
                                out_dtype=_DTYPE)
            ttnn.deallocate(qkv)
            return self._out(ttnn.reshape(ctx, (B, n, w // 3)), g_pad, s)
        if _APB_FOLD:
            qkv_w, qkv_b, g_w = self._folded()
            qh, kh, vh = self._split_heads(lin(x, qkv_w, qkv_b), self.num_heads)
            g_pad = ttnn.sigmoid(lin(x, g_w))  # pad lanes read 0.5, against ctx pad lanes that are 0
            ttnn.deallocate(x)
            return self._attend(qh, kh, vh, g_pad, s, z, head_dim, ck)
        q = lin(x, self.q_w, self.q_b)
        kv = lin(x, self.kv_w)
        k, v = ttnn.chunk(kv, 2, dim=-1)
        ttnn.deallocate(kv)
        g = ttnn.sigmoid(lin(x, self.g_w))
        ttnn.deallocate(x)

        # Pack q,k,v into heads. nlp_create_qkv_heads splits at stride = head_dim
        # rounded UP to a tile multiple (64 for head_dim 48), so a raw stride-48
        # pack would be misread (heads bleed into neighbours). Pad each head
        # 48->64 via scatter_g FIRST so nlp's stride-64 split lands on real
        # boundaries; the padded dims are zero (no effect on q.k or on ctx).
        qp = lin(q, self.scatter_g)  # [B, L, Dpad]  per-head 48 real + 16 zero
        kp = lin(k, self.scatter_g)
        vp = lin(v, self.scatter_g)
        ttnn.deallocate(q); ttnn.deallocate(k); ttnn.deallocate(v)
        qkv = ttnn.concat([qp, kp, vp], dim=-1)  # [B, L, 3*Dpad]
        ttnn.deallocate(qp); ttnn.deallocate(kp); ttnn.deallocate(vp)
        qh, kh, vh = self._split_heads(qkv, self.num_heads)
        g_pad = lin(g, self.scatter_g)  # [B, L, Dpad]
        ttnn.deallocate(g)
        return self._attend(qh, kh, vh, g_pad, s, z, head_dim, ck)

    def _pair_bias(self, z, head_dim, ck):
        """pair_bias = pair_bias_proj(layer_norm(z)) [B, H, L, L], or under dit_qkvf `_sdpa32`'s mask
        made from it (`_sdpa32_mask`).

        A pure function of the pair conditioning z, which is constant across the diffusion sampling steps (the
        resident sampler feeds the same cached z object each step). So it is built once and reused while the same
        z is fed; a new protein passes a fresh z, which rebuilds it. Bit-identical to recomputing, and saves an
        O(L^2 . H) layer_norm + projection per step per layer, the single largest step-invariant cost in the
        token DiT."""
        if _STEP_CACHE and self._pb_src is z:
            return self._pb
        if self._pb is not None:
            ttnn.deallocate(self._pb)
        z_norm = ttnn.layer_norm(
            z, weight=self.pair_norm_w, bias=self.pair_norm_b, epsilon=1e-5, compute_kernel_config=ck
        )
        pb = self._lin(z_norm, self.pair_bias_w)  # [B, L, L, H]
        ttnn.deallocate(z_norm)
        pair_bias = ttnn.permute(pb, (0, 3, 1, 2))  # [B, H, L, L]
        ttnn.deallocate(pb)
        if lever("dit_qkvf"):  # cache the fp32 mask in place of the bf16 bias
            m = _sdpa32_mask(pair_bias, head_dim**-0.5)
            ttnn.deallocate(pair_bias)
            pair_bias = m
        self._pb = pair_bias
        self._pb_src = z
        return pair_bias

    def _attend(self, qh, kh, vh, g_pad, s, z, head_dim, ck):
        pair_bias = self._pair_bias(z, head_dim, ck)
        ctx = _attn_fp32(qh, kh, vh, pair_bias, head_dim**-0.5, ck, self.accurate_softmax)
        ttnn.deallocate(qh); ttnn.deallocate(kh); ttnn.deallocate(vh)
        return self._out(self._merge_heads(ctx), g_pad, s)  # [B, L, H*head_dim_pad]  (pad value dims are 0)

    def _out(self, ctx, g_pad, s):
        """Gate + out_proj in the padded head layout, then project back to d_model; times the s gate."""
        ctx = ttnn.multiply(ctx, g_pad)
        ttnn.deallocate(g_pad)
        out = self._lin(ctx, self.out_w_padded)  # [B, L, d_model]
        ttnn.deallocate(ctx)
        if s is None:
            return out
        out_gate = ttnn.sigmoid(self._lin(s, self.out_gate_w, self.out_gate_b))
        return ttnn.multiply(out_gate, out)


class ConditionedTransitionBlock(Module):
    """adaLN-conditioned SwiGLU transition, gated by sigmoid(output_gate(s))."""

    def __init__(self, state_dict: Weights, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self.adaln = AdaLN(self.scope("adaln"), compute_kernel_config)
        self.swish_w = self.torch_to_tt("lin_swish.weight")
        self.out_w = self.torch_to_tt("lin_out.weight")
        self.gate_w = self.torch_to_tt("output_gate.weight")
        self.gate_b = self.torch_to_tt("output_gate.bias", transform=_ROW)

    def __call__(self, a: ttnn.Tensor, s: ttnn.Tensor, terms=None) -> ttnn.Tensor:
        """`terms` as AttentionPairBias's: the output then comes back without its s-gate."""
        lin = self._lin
        x = self.adaln(a, s, terms)
        sw = lin(x, self.swish_w)
        ttnn.deallocate(x)
        a1, a2 = ttnn.chunk(sw, 2, dim=-1)
        ttnn.deallocate(sw)
        if lever("ef2_atom_fuse"):
            b = ttnn.multiply(a1, a2, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        else:
            b = ttnn.multiply(ttnn.silu(a1), a2)
        ttnn.deallocate(a1); ttnn.deallocate(a2)
        out = lin(b, self.out_w)
        ttnn.deallocate(b)
        if terms is not None:
            return out
        gate = ttnn.sigmoid(lin(s, self.gate_w, self.gate_b))
        return ttnn.multiply(gate, out)


class DiffusionTransformerModel(Module):
    """Token DiT: x = x + attn(x,s,z); x = x + transition(x,s), repeated."""

    def __init__(self, num_heads: int, num_blocks: int, state_dict: Weights, compute_kernel_config,
                 accurate_softmax: bool = False):
        super().__init__(state_dict, compute_kernel_config)
        self.attn = [
            AttentionPairBias(num_heads, self.scope(f"attn_blocks.{i}"), compute_kernel_config,
                              accurate_softmax=accurate_softmax)
            for i in range(num_blocks)
        ]
        self.trans = [
            ConditionedTransitionBlock(self.scope(f"transition_blocks.{i}"), compute_kernel_config)
            for i in range(num_blocks)
        ]
        self._shoist_w = None

    def _shoist(self, s):
        """`dit_shoist`: every block's s-terms as three wide linears off this step's `s`: (sigmoid adaLN scales,
        adaLN shifts, sigmoid output gates), one 768-column window per sub-block in order (attention 0,
        transition 0, attention 1, ...). The adaLN pair reads the un-weighted LN(s), its gamma folded in."""
        if self._shoist_w is None:
            sw, sb, hw, gw, gb = [], [], [], [], []
            for attn, trans in zip(self.attn, self.trans):
                for blk, g in ((attn, "out_gate."), (trans, "output_gate.")):
                    w_sc, b_sc, w_sh = blk.adaln.folded()
                    sw.append(w_sc); sb.append(b_sc); hw.append(w_sh)
                    gw.append(blk.weights[g + "weight"].float().t()); gb.append(blk.weights[g + "bias"].float())
            up = lambda t: ttnn.from_torch(t.contiguous(), dtype=_DTYPE, layout=ttnn.TILE_LAYOUT,  # noqa: E731
                                           device=self.device)
            self._shoist_w = (up(torch.cat(sw, 1)), up(torch.cat(sb).reshape(1, -1)), up(torch.cat(hw, 1)),
                              up(torch.cat(gw, 1)), up(torch.cat(gb).reshape(1, -1)))
        w_sc, b_sc, w_sh, w_g, b_g = self._shoist_w
        s_hat = ttnn.layer_norm(s, epsilon=1e-5, compute_kernel_config=self.compute_kernel_config)
        out = (self._lin(s_hat, w_sc, b_sc, activation="sigmoid"), self._lin(s_hat, w_sh),
               self._lin(s, w_g, b_g, activation="sigmoid"))
        ttnn.deallocate(s_hat)
        return out

    def _blocks_shoist(self, a, s, z):
        """The block loop under dit_shoist: each sub-block reads its adaLN windows and returns ungated; the gate
        window multiplies it into the residual in one op."""
        sc, sh, g = self._shoist(s)
        W = lambda t, j: _ColWin(t, j * 768, 768)  # noqa: E731
        x = a
        for i, (attn, trans) in enumerate(zip(self.attn, self.trans)):
            for j, blk in ((2 * i, attn), (2 * i + 1, trans)):
                terms = (W(sc, j), W(sh, j))
                y = blk(x, s, z, terms) if blk is attn else blk(x, s, terms)
                x_new = _win_addcmul(x, y, W(g, j))
                if x_new is None:
                    gj = W(g, j).cut()
                    x_new = ttnn.addcmul(x, y, gj, value=1.0)
                    ttnn.deallocate(gj)
                ttnn.deallocate(y)
                if x is not a:
                    ttnn.deallocate(x)
                x = x_new
        for t in (sc, sh, g):
            ttnn.deallocate(t)
        return x

    def __call__(self, a: ttnn.Tensor, s: ttnn.Tensor, z: ttnn.Tensor) -> ttnn.Tensor:
        # dit_qkvf runs the attention at sdpa32_rows. The token stream is padded to it here, once a call, not
        # q, k and v in every block: rows are independent everywhere but the attention, and its mask
        # (`_sdpa32_mask`) hides the new keys. The same padding Protenix's and OpenFold3's token DiTs do.
        n = int(a.shape[-2])
        npad = sdpa32_rows(n) if lever("dit_qkvf") else n
        if npad != n:
            a, s = (ttnn.pad(t, [(0, 0), (0, npad - n), (0, 0)], value=0.0) for t in (a, s))
        if lever("dit_shoist"):
            x = self._blocks_shoist(a, s, z)
        else:
            x = a
            for attn, trans in zip(self.attn, self.trans):
                x = ttnn.add(x, attn(x, s, z))
                x = ttnn.add(x, trans(x, s))
        if npad != n:
            out = ttnn.slice(x, [0, 0, 0], [int(x.shape[0]), n, int(x.shape[2])])
            for t in (s, x):
                ttnn.deallocate(t)
            x = out
        return x


class DiffusionTransformer(TorchWrapper):
    """Top-level token transformer (torch a[B,L,768], s[B,L,768], z[B,L,L,256] in/out)."""

    def __init__(self, num_heads: int = 16, num_blocks: int = 12):
        super().__init__()
        self.num_heads = num_heads
        self.num_blocks = num_blocks

    def _create_module(self, weights: WeightScope) -> DiffusionTransformerModel:
        return DiffusionTransformerModel(
            self.num_heads, self.num_blocks, weights, self.compute_kernel_config
        )

    def forward(self, a, s, z):
        out = self.module(self._from_torch(a), self._from_torch(s), self._from_torch(z))
        return self._to_torch(out)


# ===========================================================================
# Diffusion conditioning (noise + s/z projections)
# ===========================================================================


#: First block size tried after a refusal. 384 is the outer product's first rung, and the two
#: transients are the same order, so it starts there rather than at a number of its own.
_TRANSITION_FALLBACK_ROWS = 384


class TransitionLayer(Module):
    """SwiGLU transition with separate a/b projections: out_proj(silu(a(LN(x)))*b(LN(x)))."""

    def __init__(self, state_dict: Weights, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        # Rows a refused pair shape settled at. Per layer, because the pair transitions and
        # the single ones carry different channel widths at the same shape, so what one
        # settles at says nothing about another.
        self._rows_refused: dict = {}
        self.norm_w = self.torch_to_tt("norm.weight")
        self.norm_b = self.torch_to_tt("norm.bias")
        self.a_w = self.torch_to_tt("a_proj.weight")
        self.b_w = self.torch_to_tt("b_proj.weight")
        self.out_w = self.torch_to_tt("out_proj.weight")

    def _body(self, x: ttnn.Tensor) -> ttnn.Tensor:
        ck = self.compute_kernel_config
        lin = self._lin
        xn = ttnn.layer_norm(x, weight=self.norm_w, bias=self.norm_b, epsilon=1e-5, compute_kernel_config=ck)
        a = lin(xn, self.a_w)
        b = lin(xn, self.b_w)
        ttnn.deallocate(xn)
        gated = ttnn.multiply(ttnn.silu(a), b)
        ttnn.deallocate(a); ttnn.deallocate(b)
        out = lin(gated, self.out_w)
        ttnn.deallocate(gated)
        return out

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # On the pair tensor z [B,L,L,c] the a/b projections are ~1 GiB at L=1024,
        # which (with the 6B LM resident through the fold) OOMs the 12 GB/chip
        # Wormhole DRAM. The transition is row-independent over dim=1 (per-position
        # LayerNorm + matmuls + pointwise gate), so tiling is bit-exact. A pair
        # input (4D, transient ~ rows*L) uses the area-bounded tile; a single
        # input (3D, per-token) uses the fixed row tile. Blackhole has its own, much
        # larger, pair budget (tenstorrent.BH_PAIR_TILE_AREA): single pass to 1024, blocked
        # above it, where one of these transients passes 4 GiB and gets refused.
        from tt_bio import tenstorrent
        L = x.shape[1]
        if len(x.shape) == 4:
            chunk = tenstorrent.pair_row_tile(L)
        else:
            t = tenstorrent.SMALL_GRID_SEQ_TILE
            chunk = t if (t and L > t) else 0
        if chunk:
            return self._tiled(x, chunk)
        # Blackhole runs this in a single pass (`pair_row_tile` returns 0 on a big grid), which
        # is right at every size that fits and is not at 1536: the a/b projections off a
        # [1,1536,1536,512] pair are 2.25 GiB, and DRAM refused exactly that on a p150a with
        # 288.0 MiB wanted per bank, 561.2 MiB free and a 264.2 MiB largest free block -- room
        # enough, no single block big enough. Tiling is bit-exact here by the row-independence
        # argument this op already relies on for the Wormhole path above.
        return tenstorrent.row_block_after_refusal(
            self._rows_refused, tuple(x.padded_shape),
            lambda: self._body(x), lambda rows: self._tiled(x, rows),
            rows=_TRANSITION_FALLBACK_ROWS, tag="transition")

    def _tiled(self, x: ttnn.Tensor, chunk: int) -> ttnn.Tensor:
        # Sliced one block at a time and assembled by `_acc_concat`, as the shared Transition
        # does: `ttnn.chunk` copied every block up front, a whole second x, and the device concat
        # then wanted the whole output beside every part. That concat was esmfold2-fast's next
        # refusal at 1248 once the pair conditioning fitted. Same blocks, same order.
        from tt_bio import tenstorrent
        L = int(x.shape[1])
        host = tenstorrent._host_concat(x)
        parts = []
        for s in range(0, L, chunk):
            e = min(s + chunk, L)
            p = x[:, s:e]
            tenstorrent._acc_append(parts, self._body(p), host)
            if (s, e) != (0, L):          # a whole-range slice may alias x itself
                ttnn.deallocate(p)
        return tenstorrent._acc_concat(parts, 1, host)


class DiffusionConditioningModel(Module):
    """Builds conditioning single s [B,L,c_s] and pair z [B,L,L,c_z] from inputs + noise."""

    def __init__(self, state_dict: Weights, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self.z_in_w = self.torch_to_tt("z_input_norm.weight")
        self.z_in_b = self.torch_to_tt("z_input_norm.bias")
        self.z_proj_w = self.torch_to_tt("z_proj.weight")
        self.z_trans = [TransitionLayer(self.scope(f"z_transitions.{i}"), compute_kernel_config) for i in range(2)]
        self.s_in_w = self.torch_to_tt("s_input_norm.weight")
        self.s_in_b = self.torch_to_tt("s_input_norm.bias")
        self.s_proj_w = self.torch_to_tt("s_proj.weight")
        self.noise_n_w = self.torch_to_tt("noise_norm.weight")
        self.noise_n_b = self.torch_to_tt("noise_norm.bias")
        self.noise_proj_w = self.torch_to_tt("noise_proj.weight")
        self.s_trans = [TransitionLayer(self.scope(f"s_transitions.{i}"), compute_kernel_config) for i in range(2)]
        self._rows_refused: dict = {}

    def _pair_in(self, z_trunk, relpos):
        """`z_proj(LN(concat(z_trunk, relpos)))` over the whole pair tensor."""
        # Freed in `finally`, so a refused norm or projection leaves nothing behind for the
        # row-blocked retry to allocate around.
        zc = ttnn.concat([z_trunk, relpos], dim=-1)
        try:
            zn = ttnn.layer_norm(zc, weight=self.z_in_w, bias=self.z_in_b, epsilon=1e-5,
                                 compute_kernel_config=self.compute_kernel_config)
        finally:
            ttnn.deallocate(zc)
        try:
            return self._lin(zn, self.z_proj_w)
        finally:
            ttnn.deallocate(zn)

    def _pair_in_rows(self, z_trunk, relpos, rows):
        """`_pair_in` in row blocks. The concat, the norm and the projection are all row-local,
        so no [B,L,L,c_z+c_rel] tensor ever exists whole and the blocks reassemble exactly."""
        from tt_bio import tenstorrent
        L = int(z_trunk.shape[1])
        host = tenstorrent._host_concat(z_trunk)
        blocks = []
        for s in range(0, L, rows):
            e = min(s + rows, L)
            tenstorrent._acc_append(
                blocks, self._pair_in(z_trunk[:, s:e], relpos[:, s:e]), host)
        return tenstorrent._acc_concat(blocks, 1, host)

    def cond_pair(self, z_trunk, relpos):
        """Pair conditioning z = f(z_trunk, relpos). Step-INVARIANT (no t / no
        coords) — computed once and cached across the diffusion trajectory."""
        # Three full-size pair tensors used to be live when z_proj allocates its own
        # [B,L,L,256]: the concat and its layer_norm, both [B,L,L,c_z+c_rel], plus the
        # concat's own inputs which the caller still owns. At L=1095 on a trunk that has
        # already left ~10 GiB resident that is the allocation esmfold2's 9j4c exclusion
        # dies on at the campaign's 64 samples — 627916800 B = 1095x1120x256x2 refused
        # with 34281632 B/bank free, i.e. 96.8 pct occupancy, measured on the WH Galaxy.
        #
        # Both intermediates are dead the moment the next op has read them, so freeing
        # them there costs nothing and changes no arithmetic. The concat alone is wider
        # than the refused buffer, so releasing it before the matmul is what makes 9j4c
        # fit; releasing the norm after it keeps the transitions off the same ceiling.
        #
        # Even freed promptly the concat and its norm are two [B,L,L,512] tensors live at once, and
        # at 1248 tokens on the Galaxy that second 1.49 GiB was refused with 127.5 MiB free per bank
        # against 126.8 wanted, in pieces (esmfold2-fast). After a refusal the same three ops run in
        # row blocks, so the widest thing alive is one [B,L,L,256] output.
        from tt_bio import tenstorrent
        z = tenstorrent.row_block_after_refusal(
            self._rows_refused, tuple(z_trunk.padded_shape),
            lambda: self._pair_in(z_trunk, relpos),
            lambda rows: self._pair_in_rows(z_trunk, relpos, rows),
            rows=_TRANSITION_FALLBACK_ROWS, tag="cond_pair")
        # In place: `z` is this function's own tensor, and a fresh sum would be a third
        # [B,L,L,256] beside `z` and the update -- esmfold2-fast's next refusal at 1248, once the
        # two lines above fitted (797442048 B against a 62.1 MiB/bank largest block).
        for t in self.z_trans:
            u = t(z)
            ttnn.add_(z, u)
            ttnn.deallocate(u)
        return z

    def cond_single(self, s_inputs, n_raw):
        """Single conditioning s = f(s_inputs, fourier(t)). t-DEPENDENT (cheap,
        [B,L,c]) — recomputed each step."""
        ck = self.compute_kernel_config
        lin = self._lin
        s = lin(ttnn.layer_norm(s_inputs, weight=self.s_in_w, bias=self.s_in_b, epsilon=1e-5, compute_kernel_config=ck), self.s_proj_w)
        n = lin(ttnn.layer_norm(n_raw, weight=self.noise_n_w, bias=self.noise_n_b, epsilon=1e-5, compute_kernel_config=ck), self.noise_proj_w)
        s = ttnn.add(s, ttnn.unsqueeze(n, 1))  # broadcast noise over the L axis
        for t in self.s_trans:
            s = ttnn.add(s, t(s))
        return s

    def __call__(self, s_inputs, z_trunk, relpos, n_raw):
        return self.cond_single(s_inputs, n_raw), self.cond_pair(z_trunk, relpos)


class DiffusionConditioning(TorchWrapper):
    """Top-level conditioning (torch in/out). forward(t_hat, s_inputs, z_trunk, relpos) -> (s, z)."""

    def __init__(self, sigma_data: float = 16.0):
        super().__init__()
        self.sigma_data = sigma_data

    def _create_module(self, weights: WeightScope) -> DiffusionConditioningModel:
        self._fw = weights["fourier.w"]  # [fourier_dim] buffers (host fourier embed)
        self._fb = weights["fourier.b"]
        return DiffusionConditioningModel(weights, self.compute_kernel_config)

    def forward(self, t_hat, s_inputs, z_trunk, relpos):
        bsz = z_trunk.shape[0]
        t = torch.as_tensor(t_hat, dtype=torch.float32).reshape(-1)
        if t.numel() == 1:
            t = t.expand(bsz)
        t_noise = 0.25 * torch.log((t / self.sigma_data).clamp(min=1e-20))
        n_raw = torch.cos(2.0 * math.pi * (t_noise[:, None] * self._fw[None, :] + self._fb[None, :]))
        s, z = self.module(
            self._from_torch(s_inputs), self._from_torch(z_trunk),
            self._from_torch(relpos), self._from_torch(n_raw.float()),
        )
        return self._to_torch(s), self._to_torch(z)


# ===========================================================================
# Atom encoder/decoder — SWA (sliding-window) attention with 3D RoPE
# ===========================================================================


def build_3d_rope_tables(ref_pos, ref_space_uid, head_dim, n_spatial_per_axis,
                         n_uid_pairs, spatial_base_freq, uid_base_freq):
    """Host-side 3D RoPE cos/sin, returned duplicated as [B, 1, N, head_dim]
    (matches modeling_esmfold2_common.build_3d_rope + apply_rotary_emb_3d)."""
    B, N = ref_pos.shape[:2]
    half = head_dim // 2
    spatial_inv = 1.0 / (spatial_base_freq ** (torch.arange(n_spatial_per_axis, dtype=torch.float32) / n_spatial_per_axis))
    uid_inv = 1.0 / (uid_base_freq ** (torch.arange(n_uid_pairs, dtype=torch.float32) / n_uid_pairs))
    spatial = torch.einsum("bna,k->bnak", ref_pos.float(), spatial_inv).reshape(B, N, 3 * n_spatial_per_axis)
    uid = torch.einsum("bn,k->bnk", ref_space_uid.float(), uid_inv)
    freqs = torch.cat([spatial, uid], dim=-1)
    if freqs.shape[-1] < half:
        freqs = torch.cat([freqs, torch.zeros(B, N, half - freqs.shape[-1])], dim=-1)
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).view(B, 1, N, head_dim)
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).view(B, 1, N, head_dim)
    return cos, sin


def _valid_rank(n_or_mask):
    """Atom validity [N] (bool) and each atom's rank among the valid ones, cumsum(valid) - 1."""
    if isinstance(n_or_mask, int):
        valid = torch.ones(n_or_mask, dtype=torch.bool)
    else:
        valid = (n_or_mask.reshape(-1) > 0.5)
    return valid, torch.cumsum(valid.long(), 0) - 1


def _allowed(valid, rank, i, j, half_window):
    """Whether query atom i may attend key atom j (index tensors that broadcast, all in range)."""
    within = (rank[i] - rank[j]).abs() <= half_window
    return (within & valid[i] & valid[j]) | (i == j)


def band_mask(n_or_mask, half_window: int):
    """Sliding-window additive attention bias [1,1,N,N], validity-aware.

    Accepts either an int N (all atoms valid) or an atom mask [1,N]/[N] (1=valid).
    Mirrors SWA3DRoPEAttention: window is over the *valid* atom ranking
    (rank=cumsum(valid)-1), padded atoms are excluded, and the diagonal is always
    allowed (so a fully-masked/invalid row doesn't produce NaN under softmax).
    """
    valid, rank = _valid_rank(n_or_mask)
    idx = torch.arange(valid.shape[0])
    allowed = _allowed(valid, rank, idx[:, None], idx[None, :], half_window)
    return torch.where(allowed, 0.0, float("-inf")).view(1, 1, *allowed.shape)


# The atom transformer's sliding window as ONE windowed SDPA: block i of 32 queries reads the 160 keys
# [32 i - 64, 32 i + 96) straight from a zero-framed K/V (sdpa_generic `kv_window`, the program
# Protenix's and OpenFold3's atom attention run), instead of a dense [N, N] band mask over every atom.
# Keys outside the band get zero weight on both paths, so it is the same attention; the softmax reduces
# over 160 keys instead of N, which is ~26x fewer scores at 512 tokens. When a valid atom's window does
# not fit the 160 keys (padding inside the atom list) the dense band runs instead. p300c 512 aa: -0.82 s ESMFold2,
# -1.05 s ESMFold2-Fast; on a Wormhole Galaxy chip -11.0 s per 5-sample ESMFold2-Fast grade fold. Graded on both
# models on both architectures (perf/spd/grade.py, 11 complexes x 4 seeds), so on by default; 0 restores the dense band.
_SWA_WINDOW = env_flag("TT_BIO_ESMFOLD2_SWA_WINDOW", True)
SWA_STATS = [0, 0]  # atom transformer masks built (windowed, dense)


def _window_bias(n_or_mask, half_window, nq=32):
    """`band_mask`'s 0/-inf band cut to per-block bias [nb, nq, W] over keys [nq*i - lead, nq*i - lead + W),
    or None when some valid query row has an allowed key outside its window. Built from the ranks over
    the W keys a row can see, never the dense [N, N] (at 512 aa ~4k atoms, that cost ~0.5 s of host
    time per build)."""
    valid, rank = _valid_rank(n_or_mask)
    N = valid.shape[0]
    lead = -(-half_window // nq) * nq
    W, NP = 2 * lead + nq, -(-N // nq) * nq
    i = torch.arange(NP)[:, None]
    j = (i // nq) * nq - lead + torch.arange(W)[None, :]                     # [NP, W] key atom per slot
    inside = (i < N) & (j >= 0) & (j < N)
    allowed = inside & _allowed(valid, rank, i.clamp(max=N - 1), j.clamp(0, N - 1), half_window)
    # A dense row allows every valid atom within half_window ranks of a valid query (a contiguous rank
    # range clipped to [0, V)), or only the diagonal for a padding query.
    V = int(valid.sum())
    r = rank[:N]
    need = torch.where(valid, (r + half_window).clamp(max=V - 1) - (r - half_window).clamp(min=0) + 1, 1)
    if not torch.equal(allowed[:N].sum(1), need):
        return None
    return torch.where(allowed, 0.0, float("-inf")).view(NP // nq, nq, W)


class SWAWindow:
    """The windowed atom attention's device state: the 0/-big bias [1, H*nb, 32, W] and the zero rows
    that frame K and V. Stands where the dense band mask stood."""

    NQ = 32

    def __init__(self, win, n, heads, head_dim, upload, device):
        nb, nq, W = win.shape
        self.n, self.nb, self.W, self.lead = n, nb, W, (W - nq) // 2
        # finite, so a fully masked padding row stays finite; pre-scaled because the kernel scales the
        # mask with the scores (OpenFold3 _superset does the same)
        bias = win.clamp(min=-1e4) * head_dim ** 0.5
        self.bias = upload(bias.unsqueeze(0).expand(heads, nb, nq, W).reshape(1, heads * nb, nq, W).contiguous())
        self.zero = ttnn.zeros((1, heads, self.lead, head_dim), dtype=_DTYPE, layout=ttnn.TILE_LAYOUT, device=device)
        self._batched = {}  # B -> (bias, zero) for B diffusion samples folded into the heads

    def _frame(self, B):
        """Bias and zero rows for a batch of B samples run as B * H heads: query block (b H + h) nb + i
        reads frame head b H + h, which is the row-major [B, H] -> [B H] reshape of q, k and v."""
        if B == 1:
            return self.bias, self.zero
        if B not in self._batched:
            H, lead, dh = self.zero.shape[1], self.zero.shape[2], self.zero.shape[3]
            self._batched[B] = (ttnn.repeat(self.bias, [1, B, 1, 1]),
                                ttnn.zeros((1, B * H, lead, dh), dtype=_DTYPE, layout=ttnn.TILE_LAYOUT,
                                           device=self.zero.device()))
        return self._batched[B]

    def attend(self, q, k, v, scale):
        """q, k, v [B, H, N, dh] -> ctx [B, H, N, dh]."""
        import tt_bio.tenstorrent as _T
        from tt_bio import sdpa_generic as SG
        B, dh, nb, nq, N = q.shape[0], q.shape[3], self.nb, self.NQ, self.n
        H, NP = B * q.shape[1], nb * nq
        bias, zero = self._frame(B)
        if NP != N:
            q, k, v = (ttnn.pad(t, [(0, 0), (0, 0), (0, NP - N), (0, 0)], 0.0) for t in (q, k, v))
        if B > 1:  # tile-aligned rows, so a view
            q, k, v = (ttnn.reshape(t, (1, H, NP, dh)) for t in (q, k, v))
        kf = ttnn.concat([zero, k, zero], dim=2)
        vf = ttnn.concat([zero, v, zero], dim=2)
        qb = ttnn.reshape(q, (1, H * nb, nq, dh))
        o = ttnn.allocate_tensor_on_device(ttnn.Shape([1, H * nb, nq, dh]), q.dtype, ttnn.TILE_LAYOUT,
                                           q.device(), ttnn.DRAM_MEMORY_CONFIG)
        # the wheel SDPA's default compute config, which the dense band path runs at
        SG.sdpa(q.device(), qb, kf, vf, bias, o, nq, self.W, tuple(_T.COMPUTE_GRID_MAIN),
                (ttnn.MathFidelity.HiFi2, True, False, False), scale, kv_window=(nb, self.W))
        ttnn.deallocate(kf); ttnn.deallocate(vf)
        if NP != N:
            for t in (q, k, v):
                ttnn.deallocate(t)
        o = ttnn.reshape(o, (B, H // B, NP, dh))
        return ttnn.slice(o, [0, 0, 0, 0], [B, H // B, N, dh]) if NP != N else o

    def deallocate(self):
        for t in (self.bias, self.zero, *(t for f in self._batched.values() for t in f)):
            ttnn.deallocate(t)
        self._batched = {}


def swa_band(n_or_mask, half_window, upload, device, heads=4, head_dim=32):
    """The atom transformer's attention mask on the device: a SWAWindow when the windowed path holds,
    else the dense [1, 1, N, N] band."""
    win = _window_bias(n_or_mask, half_window) if _SWA_WINDOW else None
    SWA_STATS[win is None] += 1
    if win is None:
        return upload(band_mask(n_or_mask, half_window).float())
    return SWAWindow(win, _valid_rank(n_or_mask)[0].shape[0], heads, head_dim, upload, device)


_ROT_HALF = {}  # (device id, head_dim) -> the rotate-half matrix on that device


def _rope_mm(x, cos, sin):
    """apply_rotary with rotate_half(x) = cat([-x2, x1]) as one matmul against a signed permutation.

    ttnn.experimental.rotary_embedding needs head_dim % 64 == 0 and the atom heads are 32 wide. Each output element
    is one product with +-1 and zeros, so the matmul is exact and replaces chunk, neg and concat."""
    d = x.shape[-1]
    dev = x.device()
    key = (dev.id(), d)
    if key not in _ROT_HALF:
        r = torch.zeros(d, d)
        h = d // 2
        r[h:, :h] = -torch.eye(h)
        r[:h, h:] = torch.eye(h)
        _ROT_HALF[key] = ttnn.from_torch(r, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    ck = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi3, math_approx_mode=False,
                                          fp32_dest_acc_en=True, packer_l1_acc=False)
    rot = ttnn.matmul(x, _ROT_HALF[key], compute_kernel_config=ck)
    out = ttnn.add(ttnn.multiply(x, cos), ttnn.multiply(rot, sin))
    ttnn.deallocate(rot)
    return out


def _rms_norm(x, ck, eps=1e-6):
    return ttnn.rms_norm(x, epsilon=eps, compute_kernel_config=ck)


class SWAAttention(Module):
    """Sliding-window attention with 3D RoPE, qk RMSNorm, sigmoid gate."""

    def __init__(self, n_heads: int, state_dict: Weights, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self.n_heads = n_heads
        self.qkv_w = self.torch_to_tt("Wqkv.weight")
        self.gate_w = self.torch_to_tt("gate_proj.weight")
        self.out_w = self.torch_to_tt("out_proj.weight")

    def __call__(self, x, cos, sin, attn_mask, valid=None):
        ck = self.compute_kernel_config
        lin = self._lin
        head_dim = x.shape[-1] // self.n_heads
        q, k, v = self._split_heads(lin(x, self.qkv_w), self.n_heads)
        q, k = _rms_norm(q, ck), _rms_norm(k, ck)
        if lever("ef2_atom_rope"):
            q, k = _rope_mm(q, cos, sin), _rope_mm(k, cos, sin)
        else:
            q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
        if isinstance(attn_mask, SWAWindow):
            ctx = attn_mask.attend(q, k, v, head_dim**-0.5)
        else:
            ctx = _sdpa_bf16(q, k, v, attn_mask, head_dim**-0.5)
        ttnn.deallocate(q); ttnn.deallocate(k); ttnn.deallocate(v)
        ctx = self._merge_heads(ctx)
        # Zero invalid (padding) atom outputs so they can't blow up / NaN-contaminate
        # valid atoms via masked 0*inf in later blocks (matches reference out*valid).
        if valid is not None:
            ctx = ttnn.multiply(ctx, valid)
        g = lin(x, self.gate_w)
        if lever("ef2_atom_fuse"):
            ctx = ttnn.multiply(g, ctx, input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID])
        else:
            gate = ttnn.sigmoid(g)
            ttnn.deallocate(g)
            ctx = ttnn.multiply(ctx, gate)
            g = gate
        ttnn.deallocate(g)
        out = lin(ctx, self.out_w)
        ttnn.deallocate(ctx)
        return out


class SWAAtomBlock(Module):
    """adaLN-Zero (RMSNorm * (1+scale) + shift) + SWA attn + SwiGLU FFN, gated residuals."""

    def __init__(self, n_heads: int, state_dict: Weights, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self.adaln_w = self.torch_to_tt("adaln_modulation.1.weight")
        self.attn = SWAAttention(n_heads, self.scope("attn"), compute_kernel_config)
        self.ffn_up = self.torch_to_tt("ffn.w_up.weight")
        h = self.weights["ffn.w_up.weight"].shape[0] // 2
        self.ffn_up_ab = tuple(self.torch_to_tt("ffn.w_up.weight", lambda w, s=s: w[s].t())
                               for s in (slice(0, h), slice(h, None)))
        self.ffn_down = self.torch_to_tt("ffn.w_down.weight")
        self._mod = None       # cached adaLN modulation chunks (see __call__)
        self._mod_src = None   # identity of the conditioning c_l they were built from
        self._mod_fused = False  # whether they hold 1 + scale (ef2_atom_fuse)

    def _modulate(self, x, scale, shift):
        """RMSNorm(x) * (1 + scale) + shift. Under ef2_atom_fuse `scale` already holds 1 + scale (cached with the
        modulation) and the multiply-add is one addcmul."""
        ck = self.compute_kernel_config
        if lever("ef2_atom_fuse"):
            n = _rms_norm(x, ck)
            out = ttnn.addcmul(shift, n, scale, value=1.0)
            ttnn.deallocate(n)
            return out
        return ttnn.add(ttnn.multiply(_rms_norm(x, ck), ttnn.add(scale, 1.0)), shift)

    @staticmethod
    def _gated_add(x, g, y):
        """x + g * y: one addcmul under ef2_atom_fuse."""
        if lever("ef2_atom_fuse"):
            return ttnn.addcmul(x, g, y, value=1.0)
        return ttnn.add(x, ttnn.multiply(g, y))

    def __call__(self, x, c_l, cos, sin, attn_mask, valid=None):
        ck = self.compute_kernel_config
        lin = self._lin
        # The adaLN modulation is a pure function of the conditioning c_l, which
        # is step-invariant during diffusion sampling (the encoder feeds the same
        # cached c each step), so cache the 6 chunks keyed by c_l identity.
        # Bit-identical to recomputing; saves a silu + O(N.6d) projection per
        # step per atom block (3 encoder + 3 decoder).
        fuse = lever("ef2_atom_fuse")
        if _STEP_CACHE and self._mod_src is c_l and self._mod_fused == fuse:
            sa, ca, ga, sf, cf, gf = self._mod
        else:
            if self._mod is not None:
                for _t in self._mod:
                    ttnn.deallocate(_t)
            mod = lin(ttnn.silu(c_l), self.adaln_w)  # [B,N,6d]
            sa, ca, ga, sf, cf, gf = ttnn.chunk(mod, 6, dim=-1)
            ttnn.deallocate(mod)
            if fuse:    # 1 + scale once per conditioning, not every step
                ca1, cf1 = ttnn.add(ca, 1.0), ttnn.add(cf, 1.0)
                ttnn.deallocate(ca); ttnn.deallocate(cf)
                ca, cf = ca1, cf1
            self._mod = (sa, ca, ga, sf, cf, gf)
            self._mod_src, self._mod_fused = c_l, fuse

        attn_in = self._modulate(x, ca, sa)
        attn_out = self.attn(attn_in, cos, sin, attn_mask, valid)
        x = self._gated_add(x, ga, attn_out)
        ttnn.deallocate(attn_out)

        ffn_in = self._modulate(x, cf, sf)
        if gate(ATOM_FFN_SPLIT, "ef2_atom_ffn_split"):
            h1, h2 = (lin(ffn_in, w) for w in self.ffn_up_ab)
        else:
            h1, h2 = ttnn.chunk(lin(ffn_in, self.ffn_up), 2, dim=-1)
        ttnn.deallocate(ffn_in)
        if lever("ef2_atom_fuse"):
            hb = ttnn.multiply(h1, h2, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        else:
            hb = ttnn.multiply(ttnn.silu(h1), h2)
        ttnn.deallocate(h1); ttnn.deallocate(h2)
        ffn_out = lin(hb, self.ffn_down)
        ttnn.deallocate(hb)
        x = self._gated_add(x, gf, ffn_out)
        ttnn.deallocate(ffn_out)
        return x


class SWAAtomTransformerModel(Module):
    def __init__(self, n_heads: int, n_blocks: int, state_dict: Weights, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self.blocks = [
            SWAAtomBlock(n_heads, self.scope(f"blocks.{i}"), compute_kernel_config)
            for i in range(n_blocks)
        ]

    def __call__(self, q_l, c_l, cos, sin, attn_mask, valid=None):
        for block in self.blocks:
            q_l = block(q_l, c_l, cos, sin, attn_mask, valid)
        return q_l


# Atom 3D-RoPE config (DiffusionModule defaults).
ATOM_ROPE = dict(n_spatial_per_axis=2, n_uid_pairs=10, spatial_base_freq=20.0, uid_base_freq=10000.0)


class SWAAtomTransformer(TorchWrapper):
    """Token-free SWA atom transformer (torch in/out).
    forward(q_l[B,N,d], c_l[B,N,d], ref_pos[B,N,3], ref_space_uid[B,N]) -> [B,N,d]."""

    def __init__(self, n_blocks: int = 3, n_heads: int = 4, swa_window_size: int = 128, d_atom: int = 128):
        super().__init__()
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.half_window = swa_window_size // 2
        self.head_dim = d_atom // n_heads

    def _create_module(self, weights: WeightScope) -> SWAAtomTransformerModel:
        return SWAAtomTransformerModel(self.n_heads, self.n_blocks, weights, self.compute_kernel_config)

    def forward(self, q_l, c_l, ref_pos, ref_space_uid):
        n = q_l.shape[1]
        cos, sin = build_3d_rope_tables(ref_pos, ref_space_uid, self.head_dim, **ATOM_ROPE)
        mask = swa_band(n, self.half_window, self._from_torch, self.tt_device, self.n_heads, self.head_dim)
        out = self.module(
            self._from_torch(q_l), self._from_torch(c_l),
            self._from_torch(cos.float()), self._from_torch(sin.float()), mask,
        )
        return self._to_torch(out)


# ===========================================================================
# Atom encoder / decoder wrappers + DiffusionModule
# ===========================================================================

SIGMA_DATA = 16.0


class AtomEncoder(Module):
    """ref atom features (+ noisy coords) -> token-aggregated repr a, atom repr q, c."""

    def __init__(self, n_heads, n_blocks, state_dict: Weights, compute_kernel_config,
                 structure_prediction: bool = True):
        super().__init__(state_dict, compute_kernel_config)
        self.structure_prediction = structure_prediction
        self.atom_linear_w = self.torch_to_tt("atom_linear.weight")
        self.atom_norm_w = self.torch_to_tt("atom_norm.weight")
        self.atom_norm_b = self.torch_to_tt("atom_norm.bias")
        self.coords_w = self.torch_to_tt("coords_linear.weight") if structure_prediction else None
        self.a2t_w = self.torch_to_tt("atom_to_token_linear.weight")
        self.transformer = SWAAtomTransformerModel(n_heads, n_blocks, self.scope("atom_transformer"), compute_kernel_config)
        self._c = None       # cached atom conditioning c (see __call__)
        self._c_src = None   # identity of the atom_feats it was built from

    def __call__(self, atom_feats, cos, sin, band, scatter_m, r_input=None, valid=None):
        ck = self.compute_kernel_config
        lin = self._lin
        # c = layer_norm(atom_linear(atom_feats)) depends only on the (step-
        # invariant) reference atom features, so cache it across sampling steps
        # keyed by the atom_feats identity. This also keeps the SAME c object
        # flowing into both atom transformers, which lets each SWAAtomBlock cache
        # its adaLN modulation too. Bit-identical to recomputing.
        if _STEP_CACHE and self._c_src is atom_feats:
            c = self._c
        else:
            if self._c is not None:
                ttnn.deallocate(self._c)
            c = ttnn.layer_norm(lin(atom_feats, self.atom_linear_w), weight=self.atom_norm_w, bias=self.atom_norm_b, epsilon=1e-5, compute_kernel_config=ck)
            self._c = c
            self._c_src = atom_feats
        q = ttnn.add(c, lin(r_input, self.coords_w)) if self.structure_prediction else c
        q = self.transformer(q, c, cos, sin, band, valid)
        q_to_a = ttnn.relu(lin(q, self.a2t_w))  # [B,N,d_token]
        a = ttnn.matmul(scatter_m, q_to_a, compute_kernel_config=ck, dtype=_DTYPE, core_grid=CORE_GRID_MAIN)
        ttnn.deallocate(q_to_a)
        return a, q, c


class AtomDecoder(Module):
    """token repr a + atom skip (q,c) -> per-atom coordinate update r_update [B,N,3]."""

    def __init__(self, n_heads, n_blocks, state_dict: Weights, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self.t2a_w = self.torch_to_tt("token_to_atom_linear.weight")
        self.norm_w = self.torch_to_tt("norm.weight")
        self.norm_b = self.torch_to_tt("norm.bias")
        self.out_w = self.torch_to_tt("output_linear.weight")
        self.transformer = SWAAtomTransformerModel(n_heads, n_blocks, self.scope("atom_transformer"), compute_kernel_config)

    def __call__(self, a, q_skip, c_skip, cos, sin, band, gather_g, valid=None):
        ck = self.compute_kernel_config
        lin = self._lin
        a_to_q = lin(a, self.t2a_w)  # [B,L,d_atom]
        a_to_q = ttnn.matmul(gather_g, a_to_q, compute_kernel_config=ck, dtype=_DTYPE, core_grid=CORE_GRID_MAIN)  # [B,N,d_atom]
        q = ttnn.add(q_skip, a_to_q)
        q = self.transformer(q, c_skip, cos, sin, band, valid)
        q = ttnn.layer_norm(q, weight=self.norm_w, bias=self.norm_b, epsilon=1e-5, compute_kernel_config=ck)
        return lin(q, self.out_w)  # [B,N,3]


class DiffusionModuleModel(Module):
    def __init__(self, state_dict: Weights, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        ck = compute_kernel_config
        self.conditioning = DiffusionConditioningModel(self.scope("conditioning"), ck)
        self.atom_encoder = AtomEncoder(4, 3, self.scope("atom_encoder"), ck)
        self.atom_decoder = AtomDecoder(4, 3, self.scope("atom_decoder"), ck)
        self.token_transformer = DiffusionTransformerModel(
            16, 12, self.scope("token_transformer"), ck,
            accurate_softmax=accurate_softmax_site("esmfold2.token_transformer"))
        self.s_step_norm_w = self.torch_to_tt("s_step_norm.weight")
        self.s_step_norm_b = self.torch_to_tt("s_step_norm.bias")
        self.s_to_token_w = self.torch_to_tt("s_to_token.weight")
        self.token_norm_w = self.torch_to_tt("token_norm.weight")
        self.token_norm_b = self.torch_to_tt("token_norm.bias")

    def _denoise(self, s, z, atom_feats, r_input, cos, sin, band, scatter_m, gather_g, valid):
        ck = self.compute_kernel_config
        lin = self._lin
        a, q_skip, c_skip = self.atom_encoder(atom_feats, cos, sin, band, scatter_m, r_input=r_input, valid=valid)
        s_step = ttnn.layer_norm(s, weight=self.s_step_norm_w, bias=self.s_step_norm_b, epsilon=1e-5, compute_kernel_config=ck)
        a = ttnn.add(a, lin(s_step, self.s_to_token_w))
        a = self.token_transformer(a, s, z)
        a = ttnn.layer_norm(a, weight=self.token_norm_w, bias=self.token_norm_b, epsilon=1e-5, compute_kernel_config=ck)
        return self.atom_decoder(a, q_skip, c_skip, cos, sin, band, gather_g, valid=valid)

    def __call__(self, s_inputs, z_trunk, relpos, n_raw, atom_feats, r_input,
                 cos, sin, band, scatter_m, gather_g, valid=None):
        s, z = self.conditioning(s_inputs, z_trunk, relpos, n_raw)
        out = self._denoise(s, z, atom_feats, r_input, cos, sin, band, scatter_m, gather_g, valid)
        ttnn.deallocate(z); ttnn.deallocate(s)
        return out

    # --- Cached sampling path: prepare step-invariant device tensors once, then
    # run each diffusion step transferring only the (tiny) noisy coords. Avoids
    # re-transferring z_trunk/relpos (~L²·256) and re-computing the pair
    # conditioning at every one of the ~20 sampling steps. ---
    def prepare(self, s_inputs, z_trunk, relpos, atom_feats, cos, sin, band,
                scatter_m, gather_g, valid=None):
        self._ctx = dict(
            s_inputs=s_inputs, atom_feats=atom_feats, cos=cos, sin=sin, band=band,
            scatter_m=scatter_m, gather_g=gather_g, valid=valid,
            z=self.conditioning.cond_pair(z_trunk, relpos),  # cached pair conditioning
        )
        ttnn.deallocate(z_trunk); ttnn.deallocate(relpos)

    def step(self, n_raw, r_input):
        c = self._ctx
        s = self.conditioning.cond_single(c["s_inputs"], n_raw)
        out = self._denoise(s, c["z"], c["atom_feats"], r_input, c["cos"], c["sin"],
                            c["band"], c["scatter_m"], c["gather_g"], c["valid"])
        ttnn.deallocate(s)
        return out

    def release_cache(self):
        for k, v in getattr(self, "_ctx", {}).items():
            if isinstance(v, (ttnn.Tensor, SWAWindow)):
                try:
                    v.deallocate() if isinstance(v, SWAWindow) else ttnn.deallocate(v)
                except Exception:
                    pass
        self._ctx = {}
        # The per-layer step caches are keyed on the identity of tensors freed above, so nothing can
        # hit them again. Freed here rather than when the next fold rebuilds them: left in place, the
        # token DiT's 12 pair biases alone ([1, 16, L, L], 384 MiB at 1024 tokens) sat through the
        # next fold's language model and trunk, and a warm 1024-token ESMFold2-Fast fold on a
        # Wormhole chip ran out of DRAM where the cold one fit.
        owners = ([(m, "_pb", "_pb_src") for m in self.token_transformer.attn]
                  + [(b, "_mod", "_mod_src") for b in
                     self.atom_encoder.transformer.blocks + self.atom_decoder.transformer.blocks]
                  + [(self.atom_encoder, "_c", "_c_src")])
        for owner, cache, src in owners:
            held = getattr(owner, cache)
            for t in held if isinstance(held, tuple) else (held,):
                if t is not None and t.is_allocated():
                    ttnn.deallocate(t)
            setattr(owner, cache, None)
            setattr(owner, src, None)


class DiffusionModule(TorchWrapper):
    """One diffusion denoising step. forward(x_noisy[B,N,3], t_hat, feats...) -> x_denoised[B,N,3]."""

    def __init__(self, sigma_data: float = SIGMA_DATA):
        super().__init__()
        self.sigma_data = sigma_data

    def _create_module(self, weights: WeightScope) -> DiffusionModuleModel:
        self._fw = weights["conditioning.fourier.w"]
        self._fb = weights["conditioning.fourier.b"]
        return DiffusionModuleModel(weights, self.compute_kernel_config)

    def forward(self, x_noisy, t_hat, ref_pos, ref_charge, ref_mask, ref_element,
                ref_atom_name_chars, ref_space_uid, tok_idx, s_inputs, z_trunk, relpos):
        sigma, B, N = self.sigma_data, x_noisy.shape[0], x_noisy.shape[1]
        L = s_inputs.shape[1]
        t = torch.as_tensor(t_hat, dtype=torch.float32).reshape(-1)
        if t.numel() == 1:
            t = t.expand(B)

        # host: noise embedding, normalized coords, atom features, rope/band/scatter/gather
        t_noise = 0.25 * torch.log((t / sigma).clamp(min=1e-20))
        n_raw = torch.cos(2.0 * math.pi * (t_noise[:, None] * self._fw[None, :] + self._fb[None, :])).float()
        denom = torch.sqrt(t * t + sigma * sigma)
        r_noisy = x_noisy / denom[:, None, None]
        r_input = torch.cat([r_noisy, torch.zeros_like(r_noisy)], dim=-1)  # [B,N,6]
        atom_feats = torch.cat([
            ref_pos, ref_charge.unsqueeze(-1), ref_mask.unsqueeze(-1).float(),
            ref_element, ref_atom_name_chars.reshape(B, N, -1),
        ], dim=-1)  # [B,N,389]
        cos, sin = build_3d_rope_tables(ref_pos, ref_space_uid, 32, **ATOM_ROPE)
        ft = self._from_torch
        band = swa_band(ref_mask, 64, ft, self.tt_device)
        valid = ref_mask.reshape(1, N, 1).float()  # [1,N,1] zero out padding atoms
        oh = F.one_hot(tok_idx[0].long(), L).float() * ref_mask[0].float()[:, None]  # [N,L]
        gather_g = oh.unsqueeze(0)  # [1,N,L]
        scatter_m = (oh / oh.sum(0).clamp(min=1)).t().unsqueeze(0)  # [1,L,N]

        r_update = self.module(
            ft(s_inputs), ft(z_trunk), ft(relpos), ft(n_raw), ft(atom_feats.float()),
            ft(r_input.float()), ft(cos.float()), ft(sin.float()), band,
            ft(scatter_m.float()), ft(gather_g.float()), ft(valid),
        )
        r_update = self._to_torch(r_update)

        sigma2, t2 = sigma * sigma, (t * t)
        out = (sigma2 / (sigma2 + t2))[:, None, None] * x_noisy
        out = out + ((sigma * t) / torch.sqrt(sigma2 + t2))[:, None, None] * r_update
        return out


# ===========================================================================
# Inputs embedder + relative-position (pair init)
# ===========================================================================


def relpos_features(residue_index, asym_id, sym_id, entity_id, token_index,
                    r_bins=32, c_bins=2):
    """Host-build the relative-position one-hot features [B,L,L,n_feats]
    (mirrors ResIdxAsymIdSymIdEntityIdEncoding.forward)."""
    same_chain = asym_id.unsqueeze(2) == asym_id.unsqueeze(1)
    same_res = residue_index.unsqueeze(2) == residue_index.unsqueeze(1)
    same_entity = entity_id.unsqueeze(2) == entity_id.unsqueeze(1)

    dij = torch.clip(residue_index.unsqueeze(2) - residue_index.unsqueeze(1) + r_bins, 0, 2 * r_bins)
    dij = torch.where(same_chain, dij, torch.full_like(dij, 2 * r_bins + 1))
    a_res = F.one_hot(dij, 2 * r_bins + 2)

    dtok = torch.clip(token_index.unsqueeze(2) - token_index.unsqueeze(1) + r_bins, 0, 2 * r_bins)
    dtok = torch.where(same_chain & same_res, dtok, torch.full_like(dtok, 2 * r_bins + 1))
    a_tok = F.one_hot(dtok, 2 * r_bins + 2)

    dch = torch.clip(sym_id.unsqueeze(2) - sym_id.unsqueeze(1) + c_bins, 0, 2 * c_bins)
    dch = torch.where(same_chain, torch.full_like(dch, 2 * c_bins + 1), dch)
    a_ch = F.one_hot(dch, 2 * c_bins + 2)

    return torch.cat([a_res.float(), a_tok.float(), same_entity.float().unsqueeze(-1), a_ch.float()], dim=-1)


class PairLookup:
    """A [B,L,L,c] pair tensor held as `rows[codes]`: one int code per pair, one fp32 row per distinct code.

    The vendored forward only stores it, adds it or calls `.float()` / `.detach()` on it; `host()` is the
    dense fp32 tensor and `device(dev)` builds it on the device with one embedding (an L*L index and a
    [U,c] table cross PCIe instead of the 134 MB bf16 pair at 512 tokens)."""

    dtype = torch.float32

    def __init__(self, codes, rows):
        self.codes, self.rows = codes, rows

    @property
    def shape(self):
        return (*self.codes.shape, self.rows.shape[-1])

    def float(self):
        return self

    detach = float

    def host(self):
        return F.embedding(self.codes, self.rows)

    def device(self, dev):
        assert self.codes.shape[0] == 1, self.codes.shape
        from tt_bio.tenstorrent import pair_lookup_device
        return pair_lookup_device(self.codes[0], self.rows, dev)


def relpos_rows(w, residue_index, asym_id, sym_id, entity_id, token_index, r_bins=32, c_bins=2) -> PairLookup:
    """`relpos_features(...) @ w.T` without the one-hots: every feature group is one-hot (entity is one
    0/1 column), so the product is a sum of rows of w.T picked by index, and each distinct index tuple is
    one row of the table. Rows are summed in fp32 in the order relpos_lookup always used."""
    n = 2 * r_bins + 2
    t = w.t().float()  # [n_feats, d_pair]
    same_chain = asym_id.unsqueeze(2) == asym_id.unsqueeze(1)
    same_res = residue_index.unsqueeze(2) == residue_index.unsqueeze(1)
    same_entity = entity_id.unsqueeze(2) == entity_id.unsqueeze(1)
    dij = torch.clip(residue_index.unsqueeze(2) - residue_index.unsqueeze(1) + r_bins, 0, 2 * r_bins)
    dij = torch.where(same_chain, dij, 2 * r_bins + 1)
    dtok = torch.clip(token_index.unsqueeze(2) - token_index.unsqueeze(1) + r_bins, 0, 2 * r_bins)
    dtok = torch.where(same_chain & same_res, dtok, 2 * r_bins + 1)
    dch = torch.clip(sym_id.unsqueeze(2) - sym_id.unsqueeze(1) + c_bins, 0, 2 * c_bins)
    dch = torch.where(same_chain, 2 * c_bins + 1, dch)
    nc = 2 * c_bins + 2
    uniq, codes = torch.unique(((dij.long() * n + dtok) * nc + dch) * 2 + same_entity.long(), return_inverse=True)
    se, rest = uniq % 2, uniq // 2
    ch, rest = rest % nc, rest // nc
    rows = t[:n][rest // n] + t[n:2 * n][rest % n]
    rows += t[2 * n + 1:][ch]
    rows += se.bool().unsqueeze(-1) * t[2 * n]
    return PairLookup(codes, rows)


def relpos_lookup(w, residue_index, asym_id, sym_id, entity_id, token_index, r_bins=32, c_bins=2):
    """Dense fp32 `relpos_features(...) @ w.T` (relpos_rows on the host)."""
    return relpos_rows(w, residue_index, asym_id, sym_id, entity_id, token_index, r_bins, c_bins).host()


# Relative positions by table lookup on the host (relpos_lookup); False = one-hot features through a device linear.
# Unset follows the ef2_relpos_lookup lever.
RELPOS_LOOKUP = env_flag("TT_BIO_ESMFOLD2_RELPOS_LOOKUP", None)


class RelPosEncoding(TorchWrapper):
    """Relative-position pair encoding: host one-hot features -> Linear(n_feats, d_pair)."""

    def __init__(self, r_bins: int = 32, c_bins: int = 2):
        super().__init__()
        self.r_bins, self.c_bins = r_bins, c_bins

    def _create_module(self, weights: WeightScope):
        self._w = weights["embed.weight"]
        return weights  # no ttnn submodule; we just hold the weight

    def rows(self, residue_index, asym_id, sym_id, entity_id, token_index) -> PairLookup:
        return relpos_rows(self._w, residue_index, asym_id, sym_id, entity_id, token_index, self.r_bins, self.c_bins)

    def forward(self, residue_index, asym_id, sym_id, entity_id, token_index):
        if gate(RELPOS_LOOKUP, "ef2_relpos_lookup"):
            return self.rows(residue_index, asym_id, sym_id, entity_id, token_index).host()
        feats = relpos_features(residue_index, asym_id, sym_id, entity_id, token_index, self.r_bins, self.c_bins)
        w = ttnn.from_torch(self._w.t(), layout=ttnn.TILE_LAYOUT, device=self.tt_device, dtype=_DTYPE)
        out = ttnn.linear(self._from_torch(feats.float()), w, compute_kernel_config=self.compute_kernel_config,
                          dtype=_DTYPE, core_grid=CORE_GRID_MAIN)
        return self._to_torch(out)


class InputsEmbedder(TorchWrapper):
    """Atom encoder (no coords) -> concat [a, aatype, profile, deletion_mean] = s_inputs [B,L,451]."""

    def __init__(self, n_heads: int = 4, n_blocks: int = 3):
        super().__init__()
        self.n_heads, self.n_blocks = n_heads, n_blocks

    def _create_module(self, weights: WeightScope) -> AtomEncoder:
        return AtomEncoder(self.n_heads, self.n_blocks, weights.child("atom_attention_encoder"),
                           self.compute_kernel_config, structure_prediction=False)

    def forward(self, aatype, profile, deletion_mean, ref_pos, atom_mask, ref_space_uid,
                ref_charge, ref_element, ref_atom_name_chars, atom_to_token):
        B, N, L = ref_pos.shape[0], ref_pos.shape[1], aatype.shape[1]
        atom_feats = torch.cat([
            ref_pos, ref_charge.unsqueeze(-1), atom_mask.unsqueeze(-1).float(),
            ref_element, ref_atom_name_chars.reshape(B, N, -1),
        ], dim=-1)
        cos, sin = build_3d_rope_tables(ref_pos, ref_space_uid, 32, **ATOM_ROPE)
        band = swa_band(atom_mask, 64, self._from_torch, self.tt_device)
        valid = atom_mask.reshape(1, N, 1).float()
        oh = F.one_hot(atom_to_token[0].long(), L).float() * atom_mask[0].float()[:, None]
        scatter_m = (oh / oh.sum(0).clamp(min=1)).t().unsqueeze(0)
        ft = self._from_torch
        a, _q, _c = self.module(ft(atom_feats.float()), ft(cos.float()), ft(sin.float()),
                                band, ft(scatter_m.float()), valid=ft(valid))
        a = self._to_torch(a)  # [B,L,384]
        return torch.cat([a, aatype, profile, deletion_mean.unsqueeze(-1)], dim=-1)


# ===========================================================================
# LanguageModelShim (ESMC hidden states -> pair init)
# ===========================================================================


class LanguageModelShimModel(Module):
    def __init__(self, state_dict: Weights, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self.zl_ln_w = self.torch_to_tt("base_z_linear.0.weight"); self.zl_ln_b = self.torch_to_tt("base_z_linear.0.bias")
        self.zl_w = self.torch_to_tt("base_z_linear.1.weight")
        self.dp_w = self.torch_to_tt("base_z_mlp.0.downproject.weight"); self.dp_b = self.torch_to_tt("base_z_mlp.0.downproject.bias", transform=_ROW)
        self.o0_w = self.torch_to_tt("base_z_mlp.0.output_mlp.0.weight"); self.o0_b = self.torch_to_tt("base_z_mlp.0.output_mlp.0.bias", transform=_ROW)
        self.o2_w = self.torch_to_tt("base_z_mlp.0.output_mlp.2.weight"); self.o2_b = self.torch_to_tt("base_z_mlp.0.output_mlp.2.bias", transform=_ROW)
        self.fln_w = self.torch_to_tt("base_z_mlp.1.weight"); self.fln_b = self.torch_to_tt("base_z_mlp.1.bias")

    def __call__(self, hidden, combine_w, precombined=False):  # hidden [B, L*K, d_model] (flat_layers); combine_w torch [K]
        ck = self.compute_kernel_config
        lin = self._lin
        ln = lambda t, w, b: ttnn.layer_norm(t, weight=w, bias=b, epsilon=1e-5, compute_kernel_config=ck)
        K = 1 if precombined else combine_w.numel()
        B, L = hidden.shape[0], hidden.shape[1] // K
        if precombined:  # K = 1: the ESMC already summed the weighted layer norms (combine_lm)
            x = lin(lin(hidden, self.zl_w), self.dp_w, self.dp_b)
            return self._pair_init(x, L)
        lz = lin(ln(hidden, self.zl_ln_w, self.zl_ln_b), self.zl_w)  # [B, L*K, 256]
        # Softmax-weighted layer combine on host (non-learned given combine_w).
        lz_t = torch.Tensor(ttnn.to_torch(lz)).float().reshape(B, L, K, 256)
        x_t = (combine_w.view(1, 1, K, 1) * lz_t).sum(2)  # [B,L,256]
        x = lin(ttnn.from_torch(x_t, layout=ttnn.TILE_LAYOUT, device=self.device, dtype=_DTYPE), self.dp_w, self.dp_b)
        return self._pair_init(x, L)

    def _pair_init(self, x, L):
        # The outer-product pair init holds a, b, their product and difference, and the
        # 512-wide concat all at full [B,L,L,*]: ~4.4 GiB at L=1095, which is the
        # allocation esmfold2's 9j4c exclusion dies on (1255833600 B = 1095x1120x512x2,
        # measured on the WH Galaxy). Every op below is row-local over dim=1 (a varies
        # along rows, b is a broadcast of x along rows, and the MLP/LN are per-row), so
        # row tiles reassemble bit-identically and the full-size tensors never exist.
        from tt_bio import tenstorrent
        tile = tenstorrent.pair_row_tile(L)
        if tile:
            parts = [self._pair_init_rows(x, s, min(s + tile, L), L)
                     for s in range(0, L, tile)]
            return ttnn.concat(parts, dim=1)
        return self._pair_init_rows(x, 0, L, L)

    def _pair_init_rows(self, x, s, e, L):
        ck = self.compute_kernel_config
        lin = self._lin
        ln = lambda t, w, b: ttnn.layer_norm(t, weight=w, bias=b, epsilon=1e-5, compute_kernel_config=ck)
        xa = x if (s == 0 and e == L) else x[:, s:e]
        a = ttnn.repeat(ttnn.unsqueeze(xa, 2), (1, 1, L, 1))  # [B,R,L,256]
        b = ttnn.repeat(ttnn.unsqueeze(x, 1), (1, e - s, 1, 1))
        feat = ttnn.concat([ttnn.multiply(a, b), ttnn.subtract(a, b)], dim=-1)  # [B,R,L,512]
        ttnn.deallocate(a); ttnn.deallocate(b)
        h = ttnn.gelu(lin(feat, self.o0_w, self.o0_b))
        ttnn.deallocate(feat)
        h2 = lin(h, self.o2_w, self.o2_b)
        ttnn.deallocate(h)
        return ln(h2, self.fln_w, self.fln_b)


def flat_layers(hidden_states):
    """ESMC hidden states [B, L, K, d] -> [B, L*K, d], the layout the shim projects, on the host.

    A host reshape is a view. The device reshape it replaces went through the tile-padded
    [B, L, K, d] (K = 81 pads to 96) and left a buffer the process never got back, held by no
    Python object: 451,215,360 B at 1024 tokens, which a warm 1088-token ESMFold2-Fast fold on a
    Wormhole chip then ran out of DRAM for. The upload is also 16 % smaller without the padding.
    """
    B, L, K, d = hidden_states.shape
    return hidden_states.reshape(B, L * K, d)


class LanguageModelShim(TorchWrapper):
    """ESMC hidden states [B,L,n_layers+1,d_model] -> pair init z [B,L,L,256]."""

    def _create_module(self, weights: WeightScope) -> LanguageModelShimModel:
        self._combine = weights["base_z_combine"]  # [n_layers+1]
        self._ln_bias = weights["base_z_linear.0.bias"].float()
        return LanguageModelShimModel(weights, self.compute_kernel_config)

    def combine_lm(self):
        """What the ESMC needs to combine its layers itself: (softmax weights, LayerNorm weight, bias)."""
        return torch.softmax(self._combine, 0).tolist(), self.module.zl_ln_w, self.module.zl_ln_b

    @staticmethod
    def is_precombined(hidden_states) -> bool:
        """The ESMC hands over one layer only when it combined them itself (combine_lm); the 6B has 81."""
        return hidden_states.shape[2] == 1

    def fill_absent(self, hidden_states):
        """Non-protein tokens reach the shim as zero rows, and the reference's LayerNorm maps a zero row to
        its bias in every layer, so their combined state is that bias (the weights sum to 1)."""
        absent = (hidden_states == 0).all(-1, keepdim=True)
        return torch.where(absent, self._ln_bias.to(hidden_states.dtype), hidden_states)

    def forward(self, hidden_states):
        w = torch.softmax(self._combine, 0)  # [K]
        pre = self.is_precombined(hidden_states)
        if pre:
            hidden_states = self.fill_absent(hidden_states)
        return self._to_torch(self.module(self._from_torch(flat_layers(hidden_states)), w, precombined=pre))


# ===========================================================================
# MSA encoder (optional trunk; conditions pair on an MSA)
# ===========================================================================


def _msa_row_blocks(L: int, M: int):
    """Row blocks over L for the MSA encoder's [B,L,M,c] ops, or None for a single
    pass. Every op split here is row-independent (LayerNorm over the last dim,
    per-position linears, elementwise), so the split is bit-exact; it exists
    because one encoder block's transients are ~8x the [B,L,M,c] tensor itself —
    2 GiB at L=1024, M=8192 — which a 12 GB Wormhole chip cannot hold."""
    from tt_bio import tenstorrent
    chunk = tenstorrent.msa_row_tile(L, M)
    if not chunk:
        return None
    return [(s, min(s + chunk, L)) for s in range(0, L, chunk)]


def _msa_transition_residual(m, ffn):
    """`m + ffn(m)` for the MSA transition, one row block at a time.

    Uses `SwiGLUFFN`'s own dim=1 partition (`pair_row_tile` + `ttnn.chunk`, the
    identical call it makes internally), so each block sees exactly the tensor it
    would have seen anyway and the arithmetic is bit-identical. What changes is
    lifetime: the FFN no longer concatenates a full [B,L,M,c] result that then has
    to sit next to `m` for the residual add. At L=1024, M=8192 that is two 2 GiB
    buffers the chip never has to hold."""
    from tt_bio import tenstorrent
    L = m.shape[1]
    chunk = tenstorrent.pair_row_tile(L) if len(m.shape) == 4 else 0
    if not chunk:
        return ttnn.add(m, ffn(m))
    out = []
    for part in ttnn.chunk(m, -(-L // chunk), dim=1):
        upd = ffn(part)
        out.append(ttnn.add(part, upd))
        ttnn.deallocate(upd)
    # The concat needs the whole result while `m` and every block are live. With a real MSA
    # (2ad6, 1280 tokens x 8192 rows) DRAM refused it: 106.7 MiB/bank against a 74.7 MiB
    # largest block. `m` is the MPWA output and nothing else holds it, so it is consumed.
    return tenstorrent._acc_concat(out, 1, False, consume=m)


#: L -> outer-product row block, once DRAM has refused the single pass at that L. Keyed on L
#: because the refused shape is a function of L alone, and remembered so the 48 trunk blocks x
#: 10 recycles do not each pay a failed 4.5 GiB allocation.
_OPM_ROWS_REFUSED: dict[int, int] = {}


def _opm_fallback_rows(L: int) -> int:
    """Rows per block for a refused outer product, tile-aligned and at most a quarter of L.

    The transient is rows*32 x L*32 and two are live at once across the permute, so the block
    has to shrink with L rather than sit at a constant: a quarter keeps the pair inside the
    504088512 B largest free block measured at L=1536 with room to spare, and `_row_blocked`
    halves again if the chip disagrees.
    """
    return max(32, (max(32, L // 4) // 32) * 32)


#: How the outer product meets Wout. "tile" (the original): the [L*32, L*32] product, then two tile
#: reshapes that turn each 32x32 tile into a row of 1024; at 512 aa those reshapes are 95 of the op's
#: ~114 ms (census c3, p150a), and with a 1024-row MSA 8.0 s of a 49 s fold on Wormhole (wpo). "pk": the
#: same product, the move done by `permute_back_pk.tile_rows` (bit-exact, a copy). "kq": contract a with
#: Wout's q-blocks first (K = 32 x MSA depth), so no tile ever changes shape. "rm": the original with its
#: reshapes done row-major. A/B in perf/spd_gap_esmf/opm_bench.py. Unset: "pk". It replaced the ef2_opm_kq
#: lever (kq for MSAs up to 64 rows): Wormhole, 512 aa, 35 rows, 24.8 ms a call against kq's 63.7 and tile's 231.8,
#: and exact where kq moves the result (opk/mic).
_OPM_PATH = os.environ.get("TT_BIO_ESMFOLD2_OPM") or None
_OPM_KQ_MAX_DEPTH = 64


class OuterProductMean(Module):
    """MSA -> pair update via outer-product mean (d_hidden=32, tile-aligned -> full ttnn)."""

    def __init__(self, state_dict: Weights, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self.norm_w = self.torch_to_tt("norm.weight"); self.norm_b = self.torch_to_tt("norm.bias")
        self.W = self.torch_to_tt("W.weight")
        self.Wout = self.torch_to_tt("Wout.weight"); self.Wout_b = self.torch_to_tt("Wout.bias", transform=_ROW)
        self._wq = None

    def _wout_q(self):
        """Wout as 32 [32p, 256] blocks, one per q: Wout's input index is p*32 + q."""
        if self._wq is None:
            w = self.weights["Wout.weight"].t().reshape(32, 32, -1)  # [p, q, c]
            self._wq = [ttnn.from_torch(w[:, q].contiguous(), layout=ttnn.TILE_LAYOUT, device=self.device,
                                        dtype=_DTYPE) for q in range(32)]
        return self._wq

    def _rows_kq(self, a_blk, b, recip_blk, L, M):  # a_blk [B,Bl,M,32], b [B,L,M,32] -> [B,Bl,L,256]
        """out[i,j,c] = sum over (q,m) of b[j,m,q] * U[i,(q,m),c], U[i,m,q,c] = sum_p a[i,m,p] Wout[p*32+q,c].

        Same sum as `_rows`, reassociated: the 32x32 outer block is never formed, so nothing is
        reshaped across tiles. The MSA axis is zero-padded to a tile so it can be the inner run of K."""
        ck = self.compute_kernel_config
        B, Bl = int(a_blk.shape[0]), int(a_blk.shape[1])
        Mp = -(-M // 32) * 32
        if Mp != M:
            a_blk = ttnn.pad(a_blk, [(0, 0), (0, 0), (0, Mp - M), (0, 0)], 0.0)
            b = ttnn.pad(b, [(0, 0), (0, 0), (0, Mp - M), (0, 0)], 0.0)
        u = ttnn.concat([ttnn.matmul(a_blk, w, compute_kernel_config=ck, dtype=_DTYPE) for w in self._wout_q()],
                        dim=2)  # [B,Bl,32*Mp,256], row q*Mp + m
        ut = ttnn.transpose(u, -2, -1)  # [B,Bl,256,32*Mp]
        ttnn.deallocate(u)
        bt = ttnn.reshape(ttnn.permute(b, (0, 3, 2, 1)), (1, 1, 32 * Mp, L))  # [(q,m), j]  (B == 1)
        o = ttnn.matmul(ut, bt, compute_kernel_config=ck, dtype=_DTYPE, core_grid=CORE_GRID_MAIN)  # [B,Bl,256,L]
        ttnn.deallocate(ut)
        out = ttnn.add(ttnn.transpose(o, -2, -1), self.Wout_b)
        ttnn.deallocate(o)
        return ttnn.multiply(out, recip_blk)

    def _rows_rm(self, a_blk, b2, recip_blk, L, M):
        """`_rows` with its two cross-tile reshapes done row-major (views) around one untilize/tilize."""
        ck = self.compute_kernel_config
        B, Bl = a_blk.shape[0], a_blk.shape[1]
        a2 = ttnn.reshape(ttnn.permute(a_blk, (0, 1, 3, 2)), (B, Bl * 32, M))
        prod = ttnn.matmul(a2, b2, compute_kernel_config=ck, dtype=_DTYPE, core_grid=CORE_GRID_MAIN)
        r = ttnn.to_layout(prod, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(prod)
        r = ttnn.permute(ttnn.reshape(r, (B, Bl, 32, L, 32)), (0, 1, 3, 2, 4))
        outer = ttnn.to_layout(ttnn.reshape(r, (B, Bl, L, 1024)), ttnn.TILE_LAYOUT)
        ttnn.deallocate(r)
        out = self._lin(outer, self.Wout, self.Wout_b)
        return ttnn.multiply(out, recip_blk)

    def _rows(self, a_blk, b2, recip_blk, L, M):  # a_blk [B,Bl,M,32] -> [B,Bl,L,256]
        ck = self.compute_kernel_config
        B, Bl = a_blk.shape[0], a_blk.shape[1]
        a2 = ttnn.reshape(ttnn.permute(a_blk, (0, 1, 3, 2)), (B, Bl * 32, M))   # [B,Bl*32,M]
        prod = ttnn.matmul(a2, b2, compute_kernel_config=ck, dtype=_DTYPE, core_grid=CORE_GRID_MAIN)  # [B,Bl*32,L*32]
        prod = ttnn.permute(ttnn.reshape(prod, (B, Bl, 32, L, 32)), (0, 1, 3, 2, 4))  # [B,Bl,L,32,32]
        outer = ttnn.reshape(prod, (B, Bl, L, 1024))
        out = self._lin(outer, self.Wout, self.Wout_b)  # [B,Bl,L,256]
        return ttnn.multiply(out, recip_blk)

    def _rows_pk(self, a_blk, b2, recip_blk, L, M):
        """`_rows` with the tile -> row move on `permute_back_pk.tile_rows`; `_rows` where that kernel does not serve."""
        from . import permute_back_pk as PK
        from .tenstorrent import COMPUTE_GRID_MAIN
        ck = self.compute_kernel_config
        B, Bl = int(a_blk.shape[0]), int(a_blk.shape[1])
        a2 = ttnn.reshape(ttnn.permute(a_blk, (0, 1, 3, 2)), (B, Bl * 32, M))
        prod = ttnn.matmul(a2, b2, compute_kernel_config=ck, dtype=_DTYPE, core_grid=CORE_GRID_MAIN)  # [B,Bl*32,L*32]
        outer = ttnn.allocate_tensor_on_device(ttnn.Shape([B, Bl, L, 1024]), _DTYPE, ttnn.TILE_LAYOUT,
                                               prod.device(), prod.memory_config())
        if not PK.tile_rows_ok(prod, outer):
            ttnn.deallocate(outer)
            ttnn.deallocate(prod)
            return self._rows(a_blk, b2, recip_blk, L, M)
        PK.tile_rows(prod, outer, tuple(COMPUTE_GRID_MAIN))
        ttnn.deallocate(prod)
        out = self._lin(outer, self.Wout, self.Wout_b)  # [B,Bl,L,256]
        ttnn.deallocate(outer)
        return ttnn.multiply(out, recip_blk)

    def _project(self, mm, mf=None):  # [B,rows,M,128] -> a, b each [B,rows,M,32]
        ck = self.compute_kernel_config
        x = self._lin(ttnn.layer_norm(mm, weight=self.norm_w, bias=self.norm_b, epsilon=1e-5, compute_kernel_config=ck), self.W)
        if mf is not None:  # masked MSA entries contribute nothing to the outer product
            x = ttnn.multiply(x, mf)
        return ttnn.chunk(x, 2, dim=-1)

    def __call__(self, m, recip_nvalid, mf=None):  # m [B,L,M,128]; recip_nvalid [B,L,L,1]; mf [B,L,M,1] or None
        ck = self.compute_kernel_config
        B, L, M = m.shape[0], m.shape[1], m.shape[2]
        # The LayerNorm transient is a full [B,L,M,128] (2 GiB at L=1024, M=8192);
        # rows are independent, so build a/b in row blocks.
        blocks = _msa_row_blocks(L, M)
        if blocks:
            parts = [self._project(m[:, s:e], None if mf is None else mf[:, s:e]) for s, e in blocks]
            a = ttnn.concat([p[0] for p in parts], dim=1)
            b = ttnn.concat([p[1] for p in parts], dim=1)
        else:
            a, b = self._project(m, mf)  # [B,L,M,32]
        # kq costs 16384 L^2 Mp flops against tile's 2048 L^2 M + 524288 L^2 and holds a [L, 32 Mp, 256] transient,
        # so it is for shallow MSAs only (the page fixture has 35 rows); deep ones take pk.
        path = _OPM_PATH or "pk"
        if path == "kq" and M <= _OPM_KQ_MAX_DEPTH:
            b2, rows = b, self._rows_kq  # kq contracts the [B,L,M,32] b itself
        else:
            b2 = ttnn.reshape(ttnn.permute(b, (0, 2, 1, 3)), (B, M, L * 32))   # [B,M,L*32]
            rows = {"rm": self._rows_rm, "tile": self._rows}.get(path, self._rows_pk)  # a deep MSA under kq too
        # The a2@b2 product is [B,L*32,L*32] (~2 GiB at L=1024), which OOMs the
        # 12 GB/chip Wormhole DRAM. Output row i depends only on a[i] and b, so
        # tiling over the i (row) dim is bit-exact. Pair op -> area-bounded tile
        # (transient ~ rows*L). This is the tensor that ends a 1536 fold when it is not
        # blocked: 2048*L^2 bytes, 4.50 GiB at L=1536.
        from tt_bio import tenstorrent
        chunk = tenstorrent.pair_row_tile(L)
        if chunk:
            return self._row_blocked(a, b2, recip_nvalid, L, M, chunk, rows)
        # `pair_row_tile` returns 0 on a big grid, so Blackhole runs this op in a single
        # pass -- which was right at every size it was measured at and is not at 1536.
        # `_rows` builds a [B, Bl*32, L*32] product, quadratic in L with a 32x blow-up on
        # BOTH axes: at L=1536 that is 49152 x 49152 in bf16, 4831838208 B, and the permute
        # on the next line needs a second one live at the same time. Measured on qb1
        # card 0: DRAM refused exactly that, 603979776 B per bank against a 4278190016 B
        # bank, with 698519488 B free and a 504088512 B largest block -- room enough, no
        # single block big enough.
        # Row-blocking is bit-exact here by this op's own argument, stated in the comment
        # above: output row i depends only on a[i] and b, so tiling the i axis partitions
        # independent rows and reassociates nothing. That is a stronger correctness story
        # than the pair-FFN row block, which can move a bf16 bit through the matmul program
        # ttnn derives -- nothing here changes a reduction.
        # After the refusal, never before it, so every size that fits keeps its single pass.
        # The try/refuse/halve/remember loop itself is `tenstorrent.row_block_after_refusal`,
        # shared with the pair transition rather than written out again here.
        return tenstorrent.row_block_after_refusal(
            _OPM_ROWS_REFUSED, L,
            lambda: rows(a, b2, recip_nvalid, L, M),
            lambda n: self._row_blocked(a, b2, recip_nvalid, L, M, n, rows),
            rows=_opm_fallback_rows(L), tag="opm")

    def _row_blocked(self, a, b2, recip_nvalid, L, M, chunk, rows):
        """`rows` (one of the `_rows*` paths) over `chunk`-row blocks. A further refusal is the caller's to shrink."""
        outs = [rows(a[:, s:min(s + chunk, L), :, :], b2,
                           recip_nvalid[:, s:min(s + chunk, L), :, :], L, M)
                for s in range(0, L, chunk)]
        return ttnn.concat(outs, dim=1)


class MSAPairWeightedAveraging(Module):
    """Pair-biased MSA row update on tt-bio's shared `PairWeightedAveraging` (AF3 Algorithm 10).

    The shared module holds the MSA depth-major, [1, M, L, c]; ESMFold2 holds it token-major,
    [1, L, M, c], so the update is one permute in and one out. The old path ran the head contraction
    on the host in fp32 because head_width 16 is sub-tile; the shared module's unpadded head path
    does it on the chip. With a 1024-row MSA at 512 tokens that host einsum was 1.3 s a call, 33
    calls a fold (11 loops x 3 blocks).
    """

    def __init__(self, n_heads, head_width, state_dict: Weights, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        from tt_bio import tenstorrent
        sd = self.weights.as_dict()
        names = {"norm_m": "norm_single", "norm_z": "compute_bias.0", "proj_z": "compute_bias.1",
                 "proj_m": "Wv", "proj_g": "Wgate", "proj_o": "Wout"}
        self.pwa = tenstorrent.PairWeightedAveraging(head_width, n_heads, WeightScope(
            {f"{k}.{t}": sd[f"{v}.{t}"] for k, v in names.items() for t in ("weight", "bias")
             if f"{v}.{t}" in sd}), compute_kernel_config)

    def __call__(self, m, pair):
        """Returns `m + update`."""
        mt = ttnn.permute(m, (0, 2, 1, 3))
        o = self.pwa(mt, pair)
        ttnn.deallocate(mt)
        u = ttnn.permute(o, (0, 2, 1, 3))
        ttnn.deallocate(o)
        out = ttnn.add(m, u)
        ttnn.deallocate(u)
        return out


class MSAEncoderBlock(Module):
    def __init__(self, n_heads_msa, msa_head_width, is_final, state_dict: Weights, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self.is_final = is_final
        self.opm = OuterProductMean(self.scope("outer_product_mean"), compute_kernel_config)
        if not is_final:
            self.mpwa = MSAPairWeightedAveraging(n_heads_msa, msa_head_width, self.scope("msa_pair_weighted_averaging"), compute_kernel_config)
            self.msa_transition = SwiGLUFFN(_remap_transition_named(self.weights.as_dict(), "msa_transition"), compute_kernel_config)
        self.tri_out = TriangleMultiplication(False, _remap_trimul(self.weights.as_dict(), "tri_mul_out._engine"), compute_kernel_config)
        self.tri_in = TriangleMultiplication(True, _remap_trimul(self.weights.as_dict(), "tri_mul_in._engine"), compute_kernel_config)
        self.pair_transition = SwiGLUFFN(_remap_transition_named(self.weights.as_dict(), "pair_transition"), compute_kernel_config)

    def __call__(self, m, pair, recip_nvalid, mf=None):
        # `pair` is the encoder's own tensor (block 0 gets MSAEncoder.forward's upload), so every
        # residual adds in place. A fresh sum is a third pair tensor beside the MSA: 3abq at 1536
        # with its MSA was refused the pair-transition sum, 1193803776 B against a 64.7 MiB/bank
        # largest block.
        u = self.opm(m, recip_nvalid, mf)
        ttnn.add_(pair, u)
        ttnn.deallocate(u)
        if not self.is_final:
            m = self.mpwa(m, pair)  # residual included
            m = _msa_transition_residual(m, self.msa_transition)
        for f in (lambda z: self.tri_out(z, None), lambda z: self.tri_in(z, None), self.pair_transition):
            u = f(pair)
            ttnn.add_(pair, u)
            ttnn.deallocate(u)
        return m, pair


def _remap_transition_named(sd: dict, prefix: str) -> WeightScope:
    return WeightScope({
        "0.weight": sd[f"{prefix}.norm.weight"], "0.bias": sd[f"{prefix}.norm.bias"],
        "1.weight": sd[f"{prefix}.ffn.w12.weight"], "3.weight": sd[f"{prefix}.ffn.w3.weight"],
    })


class MSAEncoderModel(Module):
    def __init__(self, n_layers, n_heads_msa, msa_head_width, state_dict: Weights, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self.embed_w = self.torch_to_tt("embed.weight")
        self.project_w = self.torch_to_tt("project_inputs.weight")
        self.blocks = [
            MSAEncoderBlock(n_heads_msa, msa_head_width, i == n_layers - 1, self.scope(f"blocks.{i}"), compute_kernel_config)
            for i in range(n_layers)
        ]

    def __call__(self, x_pair, x_inputs, m_feat, recip_nvalid, mf=None):
        ck = self.compute_kernel_config
        lin = self._lin
        m = ttnn.add(lin(m_feat, self.embed_w), ttnn.unsqueeze(lin(x_inputs, self.project_w), 2))
        # The features are an MSA-sized upload used only here; the blocks hold `m` beside it.
        ttnn.deallocate(m_feat)
        pair = x_pair
        for block in self.blocks:
            m, pair = block(m, pair, recip_nvalid, mf)
        return pair


class MSAEncoder(TorchWrapper):
    """MSA encoder: refine pair using an MSA. forward(x_pair, x_inputs, msa_oh, has_deletion, deletion_value, msa_mask)."""

    def __init__(self, n_layers: int = 4, n_heads_msa: int = 8, msa_head_width: int = 16):
        super().__init__()
        self.n_layers, self.n_heads_msa, self.msa_head_width = n_layers, n_heads_msa, msa_head_width

    def _create_module(self, weights: WeightScope) -> MSAEncoderModel:
        return MSAEncoderModel(self.n_layers, self.n_heads_msa, self.msa_head_width, weights, self.compute_kernel_config)

    def forward(self, x_pair, x_inputs, msa_oh, has_deletion, deletion_value, msa_mask, to_host=True):
        """`to_host=False` returns the device pair, for a caller that keeps the injection on the chip."""
        m_feat = torch.cat([msa_oh, has_deletion.unsqueeze(-1), deletion_value.unsqueeze(-1)], dim=-1)  # [B,L,M,35]
        mask_f = msa_mask.float()
        n_valid = (mask_f @ mask_f.transpose(-1, -2)).clamp(min=1.0).unsqueeze(-1)  # [B,L,L,1]
        ft = self._from_torch
        # An all-ones mask (no column masking) multiplies by one: skip it.
        mf = None if bool(mask_f.all()) else ft(mask_f.unsqueeze(-1))
        # A device x_pair (ESMFold2's z_init built on the device) is copied: the blocks update the pair in place.
        xp = ttnn.clone(x_pair) if isinstance(x_pair, ttnn.Tensor) else ft(x_pair)
        out = self.module(xp, ft(x_inputs), ft(m_feat.float()), ft((1.0 / n_valid).float()), mf)
        return self._to_torch(out) if to_host else out


# ===========================================================================
# Diffusion sampler (host orchestration over the ttnn DiffusionModule)
# ===========================================================================


def karras_schedule(steps, sigma_data=16.0, s_min=0.0004, s_max=160.0, p=7.0):
    """Karras power-law noise schedule (mirrors DiffusionStructureHead)."""
    inv_p = 1.0 / p
    k = torch.arange(steps, dtype=torch.float32)
    base = s_max ** inv_p + (k / (steps - 1)) * (s_min ** inv_p - s_max ** inv_p)
    return F.pad(sigma_data * base.pow(p), (0, 1), value=0.0)


def _random_rotations(n, gen):
    q = torch.randn((n, 4), generator=gen)
    scale = torch.sqrt((q * q).sum(1))
    signs = torch.where(q[:, 0] < 0, -scale, scale)
    q = q / signs[:, None]
    r, i, j, k = torch.unbind(q, -1)
    two_s = 2.0 / (q * q).sum(-1)
    return torch.stack((
        1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r),
        two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r),
        two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j),
    ), -1).reshape(n, 3, 3)


def _center_random_augmentation(x, atom_mask, gen, second=None):
    mask = atom_mask.unsqueeze(-1)
    mean = (x * mask).sum(1, keepdim=True) / mask.sum(1, keepdim=True).clamp(min=1)
    x = x - mean
    if second is not None:
        second = second - mean
    r = _random_rotations(x.shape[0], gen).to(x.dtype)
    x = torch.einsum("bmd,bds->bms", x, r)
    if second is not None:
        second = torch.einsum("bmd,bds->bms", second, r)
    t = torch.randn(x[:, 0:1, :].shape, generator=gen)
    return x + t, (None if second is None else second + t)


def _weighted_rigid_align(x, x_gt, w, mask):
    w = (mask * w).unsqueeze(-1)
    denom = w.sum(-2, keepdim=True).clamp(min=1e-8)
    mu, mu_gt = (x * w).sum(-2, keepdim=True) / denom, (x_gt * w).sum(-2, keepdim=True) / denom
    H = torch.einsum("bni,bnj->bij", w * (x_gt - mu_gt), (x - mu))
    U, _, Vh = torch.linalg.svd(H.float())
    det = torch.linalg.det(U @ Vh)
    R = (U @ torch.diag_embed(torch.stack([torch.ones_like(det), torch.ones_like(det), det], -1)) @ Vh).to(H.dtype)
    return torch.einsum("bni,bij->bnj", x - mu, R.transpose(-1, -2)) + mu_gt


def sample_structure(denoise_fn, n_atoms, ref_mask, *, steps=14, sigma_data=16.0,
                     s_min=0.0004, s_max=160.0, p=7.0, gamma_0=0.8, gamma_min=1.0,
                     noise_scale=1.003, step_scale=1.5, max_inference_sigma=256.0, seed=0):
    """Run reverse diffusion (Algorithm 18). denoise_fn(x_noisy[B,N,3], t_hat[B]) -> x_denoised."""
    # By default the sampler draws from a private generator seeded with `seed`
    # so a fold is reproducible independent of global RNG state. The torch
    # reference's DiffusionStructureHead.sample instead draws from the GLOBAL
    # RNG (controlled by torch.manual_seed). When TT_BIO_ESMFOLD2_DIFFUSION_SHARED_RNG
    # is set, match that convention (gen=None -> torch.randn uses the global
    # generator): the device sampler then shares draws with the reference for
    # same-seed parity, and the public `seed` param actually controls the
    # sampler (the private-generator path ignores it, because the forward never
    # threads the seed kwarg -- the mp-spawn-worker-unseeded-rng-pattern).
    gen = None if env_flag("TT_BIO_ESMFOLD2_DIFFUSION_SHARED_RNG", False) else torch.Generator().manual_seed(seed)
    # Shared-draws hook for the integration-parity gate (see tt_bio/boltz2.py
    # AtomDiffusion.sample). The private generator above makes a fold reproducible
    # regardless of global RNG state, but for the envelope test the device and CPU
    # references must draw byte-identical noise. TT_BIO_SHARED_DRAW_SEED forces the
    # GLOBAL RNG (gen=None) into an identical state immediately before the first
    # draw on both paths, so every torch.randn(..., generator=gen) below shares
    # draws. Unset in production, so normal folds keep the private-generator path.
    _sds = os.environ.get("TT_BIO_SHARED_DRAW_SEED")
    if _sds:
        gen = None
        torch.manual_seed(int(_sds))
    schedule = karras_schedule(steps, sigma_data, s_min, s_max, p)
    if max_inference_sigma is not None:
        schedule = schedule[schedule <= float(max_inference_sigma)]
        schedule = F.pad(schedule, (1, 0), value=float(max_inference_sigma))
    lam, eta = noise_scale, step_scale
    B = ref_mask.shape[0]
    x = schedule[0] * torch.randn(B, n_atoms, 3, generator=gen)
    if _DUMP is not None:
        _DUMP(-1, x, None)
    atom_mask = ref_mask.float()
    gammas = torch.where(schedule > gamma_min, torch.full_like(schedule, gamma_0), torch.zeros_like(schedule))
    x_prev = None
    n_steps = len(schedule) - 1
    for i, (s_tm, s_t, gamma) in enumerate(zip(schedule[:-1], schedule[1:], gammas[1:])):
        report_progress("diffusion", i, n_steps)
        x, x_prev = _center_random_augmentation(x, atom_mask, gen, x_prev)
        s_tm_v = float(s_tm); t_hat = s_tm_v * (1.0 + float(gamma))
        eps = lam * max(t_hat ** 2 - s_tm_v ** 2, 0.0) ** 0.5
        x_noisy = x + eps * torch.randn(x.shape, generator=gen)
        x_den = denoise_fn(x_noisy, torch.full((B,), t_hat))
        x_noisy = _weighted_rigid_align(x_noisy.float(), x_den.float(), atom_mask, atom_mask)
        x = x_noisy + eta * (float(s_t) - t_hat) * (x_noisy - x_den) / t_hat
        x_prev = x_den
        if _DUMP is not None:
            _DUMP(i, x, x_den)
    return x


class StructureHead(TorchWrapper):
    """ttnn DiffusionModule + diffusion sampler -> 3D coordinates."""

    def __init__(self, sigma_data: float = SIGMA_DATA):
        super().__init__()
        self.sigma_data = sigma_data

    def _create_module(self, weights: WeightScope) -> DiffusionModuleModel:
        self._fw = weights["diffusion_module.conditioning.fourier.w"]
        self._fb = weights["diffusion_module.conditioning.fourier.b"]
        return DiffusionModuleModel(weights.child("diffusion_module"), self.compute_kernel_config)

    def sample(self, *args, steps=14, seed=0, multiplicity=1):
        self.prepare(*args)
        try:
            return self.draw(steps=steps, seed=seed, multiplicity=multiplicity)
        finally:
            self.release()

    def prepare(self, z_trunk, s_inputs, relpos, ref_pos, ref_charge, ref_mask, ref_element,
                ref_atom_name_chars, ref_space_uid, tok_idx):
        # Build the step-invariant tensors ONCE and keep them resident on the
        # device until release(): the pair conditioning, atom features,
        # 3D-RoPE / band / scatter / gather tables. Only the (tiny) noisy coords
        # cross PCIe each step — z_trunk/relpos (~L²·256) and the pair
        # conditioning are not re-transferred / re-computed per step, nor per
        # sample chunk: the fp32 z_trunk upload is one contiguous L²·256·4 B
        # request, and re-making it per chunk is what failed 1152 tokens x 2
        # samples on Wormhole once the first chunk had fragmented DRAM.
        dmm = self.module
        B, N, L = ref_pos.shape[0], ref_pos.shape[1], s_inputs.shape[1]
        ft = self._from_torch
        atom_feats = torch.cat([
            ref_pos, ref_charge.unsqueeze(-1), ref_mask.unsqueeze(-1).float(),
            ref_element, ref_atom_name_chars.reshape(B, N, -1),
        ], dim=-1)
        cos, sin = build_3d_rope_tables(ref_pos, ref_space_uid, 32, **ATOM_ROPE)
        band = swa_band(ref_mask, 64, ft, self.tt_device)
        valid = ref_mask.reshape(1, N, 1).float()
        oh = F.one_hot(tok_idx[0].long(), L).float() * ref_mask[0].float()[:, None]
        gather_g = oh.unsqueeze(0)
        scatter_m = (oh / oh.sum(0).clamp(min=1)).t().unsqueeze(0)
        relpos = relpos.device(self.tt_device) if isinstance(relpos, PairLookup) else ft(relpos)
        dmm.prepare(ft(s_inputs), z_trunk if isinstance(z_trunk, ttnn.Tensor) else ft(z_trunk), relpos, ft(atom_feats.float()),
                    ft(cos.float()), ft(sin.float()), band,
                    ft(scatter_m.float()), ft(gather_g.float()), ft(valid))
        self._base, self._ref_mask, self._n_atoms = dmm._ctx, ref_mask, N

    def draw(self, steps=14, seed=0, multiplicity=1):
        """One trajectory of `multiplicity` samples on the tensors prepare() left resident."""
        dmm, sigma, ft = self.module, self.sigma_data, self._from_torch
        ref_mask, N = self._ref_mask, self._n_atoms

        # Best-of-N batching: the conditioning is molecule-only (identical
        # across diffusion samples), so replicate the resident tensors to
        # `multiplicity` — then a SINGLE B=N trajectory draws N distinct
        # samples (different noise per batch row) for ~1x the device cost of one
        # (B=1 underutilizes the grid). B=1 is untouched (bit-identical). The
        # only data crossing PCIe per step stays the tiny noisy coords. The base
        # stays resident beside the replicas for the next chunk; multiplicity > 1
        # only happens where B·L² fits the budget, so that costs at most one
        # L≈390 pair conditioning.
        if multiplicity > 1:
            rep = lambda v: (ttnn.repeat(v, [multiplicity] + [1] * (len(v.shape) - 1))
                             if isinstance(v, ttnn.Tensor) else v)
            dmm._ctx = {k: rep(v) for k, v in self._base.items()}
            ref_mask = ref_mask.repeat(multiplicity, 1)

        def denoise(x_noisy, t_hat):
            t = torch.as_tensor(t_hat, dtype=torch.float32).reshape(-1)
            if t.numel() == 1:
                t = t.expand(x_noisy.shape[0])
            t_noise = 0.25 * torch.log((t / sigma).clamp(min=1e-20))
            n_raw = torch.cos(2.0 * math.pi * (t_noise[:, None] * self._fw[None, :] + self._fb[None, :])).float()
            denom = torch.sqrt(t * t + sigma * sigma)
            r_noisy = x_noisy / denom[:, None, None]
            r_input = torch.cat([r_noisy, torch.zeros_like(r_noisy)], dim=-1)  # [B,N,6]
            r_update = self._step(n_raw, r_input.float())
            sigma2, t2 = sigma * sigma, t * t
            out = (sigma2 / (sigma2 + t2))[:, None, None] * x_noisy
            return out + ((sigma * t) / torch.sqrt(sigma2 + t2))[:, None, None] * r_update

        try:
            return sample_structure(denoise, N, ref_mask, steps=steps,
                                    sigma_data=sigma, seed=seed)
        finally:
            self._release_trace()
            if dmm._ctx is not self._base:
                # the replicas, not what they share with the base (the SWAWindow is batched in place)
                dmm._ctx = {k: v for k, v in dmm._ctx.items() if v is not self._base.get(k)}
                dmm.release_cache()
                dmm._ctx = self._base

    def _step(self, n_raw, r_input):
        """One denoiser call. Replayed from a ttnn trace when the device has a region for it.

        The step is shape-stable within a trajectory and reads only two small inputs, so the
        graph captured once replays every later step with new input contents: the same program
        on the same buffers, bit-identical to the eager call. What it removes is the host
        dispatch of every op of every step, which the eager loop pays in full because each step
        ends in a download. Steps 1-2 run eager (they fill the modules' lazy device caches,
        which must not be created inside a capture); step 3 is captured and replayed.
        The trace bakes in the addresses of the trajectory's resident tensors, so it lives
        exactly as long as one draw().
        """
        tr = getattr(self, "_trace", None)
        key = (tuple(n_raw.shape), tuple(r_input.shape))
        if tr is not None and tr["key"] != key:
            self._release_trace()
            tr = None
        if tr is None:
            self._eager_steps = getattr(self, "_eager_steps", 0) + 1
            if not (DIFFUSION_TRACE and trace_region_size() > 0 and self._eager_steps > 2):
                DIFFUSION_TRACE_STATS[1] += 1
                out = self.module.step(self._from_torch(n_raw), self._from_torch(r_input))
                res = self._to_torch(out)
                ttnn.deallocate(out)
                return res
            tr = self._capture_step(n_raw, r_input, key)
        host = lambda x: ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        ttnn.copy_host_to_device_tensor(host(n_raw), tr["n"])
        ttnn.copy_host_to_device_tensor(host(r_input), tr["r"])
        ttnn.execute_trace(self.tt_device, tr["tid"], cq_id=0, blocking=False)
        DIFFUSION_TRACE_STATS[0] += 1
        return self._to_torch(tr["out"])

    def _capture_step(self, n_raw, r_input, key):
        dev = self.tt_device
        n_t, r_t = self._from_torch(n_raw), self._from_torch(r_input)
        ttnn.synchronize_device(dev)
        tid, out = capture_trace(dev, lambda: self.module.step(n_t, r_t))
        self._trace = {"tid": tid, "n": n_t, "r": r_t, "out": out, "key": key}
        return self._trace

    def _release_trace(self):
        tr = getattr(self, "_trace", None)
        self._trace, self._eager_steps = None, 0
        if tr is None:
            return
        ttnn.synchronize_device(self.tt_device)
        ttnn.release_trace(self.tt_device, tr["tid"])
        for k in ("n", "r", "out"):
            ttnn.deallocate(tr[k])

    def release(self):
        self.module._ctx = getattr(self, "_base", {}) or {}
        self.module.release_cache()
        self._base = self._ref_mask = None


# ===========================================================================
# Distogram head
# ===========================================================================


class DistogramHead(Module):
    """Plain pair->bins linear. Symmetrizing z is the caller's job: the reference
    forward passes `z + z.transpose(-2, -3)` in, so this module must not do it again."""

    def __init__(self, state_dict: Weights, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self.w = self.torch_to_tt("weight")
        self.b = self.torch_to_tt("bias", transform=_ROW)

    def __call__(self, z: ttnn.Tensor) -> ttnn.Tensor:
        return self._lin(z, self.w, bias=self.b)


class DistogramHeadModel(TorchWrapper):
    def _create_module(self, weights: WeightScope) -> DistogramHead:
        return DistogramHead(weights, self.compute_kernel_config)

    def forward(self, z):
        return self._to_torch(self.module(self._from_torch(z)))

