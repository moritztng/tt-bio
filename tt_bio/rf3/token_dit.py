"""RF3's 24-block token diffusion transformer on ttnn.

Full attention, not windowed: the call site passes `Beta_II=None`, which takes
`AttentionPairBiasDiffusion.forward` rather than the `atom_attention` branch the
atom stacks take. Same three flags as every other RF3 DiT stack
(kq_norm / no_residual / 2-factor SwiGLU); head_dim is 48, so kq_norm runs its
un-pad path.

`DiffusionTransformerLayer` hardcodes `compute_pair_bias=False`, so the pair bias is
precomputed here, once per block, as `to_b(ln_0(Z_II))`. RF3 adds it unscaled -- Q is
already divided by sqrt(c) -- so nothing rescales it.
"""

from __future__ import annotations

import torch
import ttnn

from tt_bio.rf3.remap_encoder import ADALN, ATTENTION
from tt_bio.tenstorrent import DiffusionTransformer, Module, _dtype

N_HEADS = 16
C_TOKEN = 768
HEAD_DIM = C_TOKEN // N_HEADS      # 48, deliberately not tile-aligned


def remap_block(src: dict, i: int) -> dict:
    """One RF3 DiT block -> tt-bio `layers.{i}.*`. Same shape as the atom remap."""
    p, o = f"blocks.{i}.", f"layers.{i}."
    apb, ctb = f"{p}attention_pair_bias.", f"{p}conditioned_transition_block."
    out: dict[str, torch.Tensor] = {}
    for a, b in ADALN.items():
        out[f"{o}adaln.{b}"] = src[f"{apb}ada_ln_1.{a}"]
        out[f"{o}transition.adaln.{b}"] = src[f"{ctb}ada_ln.{a}"]
    for a, b in ATTENTION.items():
        out[f"{o}pair_bias_attn.{b}"] = src[f"{apb}{a}"]
    out[f"{o}pair_bias_attn.proj_q.bias"] = torch.zeros(
        src[f"{apb}to_q.weight"].shape[0], dtype=src[f"{apb}to_q.weight"].dtype)
    out[f"{o}output_projection_linear.weight"] = src[f"{apb}linear_output_project.0.weight"]
    out[f"{o}output_projection_linear.bias"] = src[f"{apb}linear_output_project.0.bias"]
    # chunk 0 is the NON-SiLU'd operand, so linear_2 comes first
    out[f"{o}transition.swish_gate.0.weight"] = torch.cat(
        [src[f"{ctb}linear_2.weight"], src[f"{ctb}linear_1.weight"]], dim=0)
    out[f"{o}transition.b_to_a.weight"] = src[f"{ctb}linear_3.weight"]
    out[f"{o}transition.output_projection.0.weight"] = src[f"{ctb}linear_output_project.0.weight"]
    out[f"{o}transition.output_projection.0.bias"] = src[f"{ctb}linear_output_project.0.bias"]
    return out


class TokenDiffusionTransformer(Module):
    def __init__(self, state_dict, compute_kernel_config, n_block: int = 24,
                 fp32_softmax: bool = False):
        # NOTE fp32_softmax is INERT for this stack. The non-atom branch of
        # AttentionPairBias.__call__ has its own attention implementation and never
        # calls _attention, which is where the flag is read -- which is why both arms
        # scored identically to 7 digits. Kept as a parameter only so the signature
        # matches the atom stacks; do not read anything into it here.
        super().__init__(state_dict, compute_kernel_config)
        raw = self.weights.as_dict()
        self.n_block = n_block
        remapped: dict[str, torch.Tensor] = {}
        for i in range(n_block):
            remapped.update(remap_block(raw, i))
        self.ln0_w, self.ln0_b, self.to_b = [], [], []
        for i in range(n_block):
            p = f"blocks.{i}.attention_pair_bias."
            self.ln0_w.append(ttnn.from_torch(
                raw[f"{p}ln_0.weight"].float(), layout=ttnn.TILE_LAYOUT,
                device=self.device, dtype=_dtype(ttnn.bfloat16)))
            self.ln0_b.append(ttnn.from_torch(
                raw[f"{p}ln_0.bias"].float(), layout=ttnn.TILE_LAYOUT,
                device=self.device, dtype=_dtype(ttnn.bfloat16)))
            # tt-bio's token attention adds the bias to the raw q@k^T and scales the
            # SUM by head_dim**-0.5 (matching SDPA, which scales its additive mask along
            # with QK). RF3 adds the bias unscaled, having already divided Q by sqrt(c).
            # So the bias has to carry sqrt(head_dim) to survive the shared scale -- the
            # same compensation AttentionPairBias bakes into z_weight when it computes
            # the bias itself. Without it the bias arrives 1/sqrt(48) too small, which
            # is worth 34.65x the ceiling and still reads as pcc 0.996.
            self.to_b.append(ttnn.from_torch(
                (raw[f"{p}to_b.weight"] * (HEAD_DIM ** 0.5)).t().contiguous(),
                layout=ttnn.TILE_LAYOUT,
                device=self.device, dtype=_dtype(ttnn.bfloat16)))
        self.stack = DiffusionTransformer(
            n_layers=n_block, dim=C_TOKEN, n_heads=N_HEADS, atom_level=False,
            state_dict=remapped, compute_kernel_config=compute_kernel_config,
            no_residual=True, a_to_b_gate=False, fp32_softmax=fp32_softmax,
        )

    def bias(self, z: ttnn.Tensor, i: int) -> ttnn.Tensor:
        b = ttnn.layer_norm(z, weight=self.ln0_w[i], bias=self.ln0_b[i], epsilon=1e-5,
                            compute_kernel_config=self.compute_kernel_config)
        b = ttnn.linear(b, self.to_b[i],
                        compute_kernel_config=self.compute_kernel_config)
        return ttnn.permute(b, (0, 3, 1, 2))          # [1, heads, I, I]

    def __call__(self, a: ttnn.Tensor, s: ttnn.Tensor, z: ttnn.Tensor,
                 bias: list | None = None) -> ttnn.Tensor:
        """`bias` is the 24 per-head pair biases, built once per fold by the caller.

        They are a pure function of `z`, which is the diffusion pair conditioning: no `t`
        and no diffusion-batch index reaches them, so rebuilding them per denoiser call
        produced the same bytes 49 x D times over.
        """
        if bias is None:
            bias = [self.bias(z, i) for i in range(self.n_block)]
        return self.stack(a, s, bias)
