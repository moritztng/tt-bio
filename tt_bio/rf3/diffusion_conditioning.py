"""RF3's diffusion conditioning on ttnn.

    Z = to_zii(cat([Z_trunk, relpos(f)]));  Z += transition_1[b](Z)  for b in 0,1
    S = to_si(cat([S_trunk, S_inputs]))
    S = process_n(fourier(1/4 * log(t / sigma_data))) + S
    S += transition_2[b](S)  for b in 0,1

Two things not to infer from names:

  * the `relative_position_encoding` here is a SECOND instance with its own learned
    linear -- the same host one-hot features as the feature initializer's, different
    weights. Reusing the feature initializer's linear would be a silent 128-channel
    error on every pair.
  * `c_t_embed` is overwritten to 256 inside `__init__`, so the config value for it is
    dead. The checkpoint agrees (fourier w is [256]).

The Fourier embedding is computed on host in fp32 deliberately: `torch.cos` is not on
autocast's list and its operands stay fp32, so the reference evaluates it in fp32 even
under autocast. Matching where the reference actually computes is what took C_L to
bit-exact in the encoder.
"""

from __future__ import annotations

import math

import torch
import ttnn

from tt_bio.tenstorrent import CORE_GRID_MAIN, Module, Transition, _dtype


def fourier_embedding(t: torch.Tensor, w: torch.Tensor, b: torch.Tensor,
                      sigma_data: float = 16.0) -> torch.Tensor:
    """[D] noise levels -> [D, c_t_embed], as `FourierEmbedding` does it."""
    n = 0.25 * torch.log(t.float() / sigma_data)
    return torch.cos(2 * math.pi * (n[..., None] * w.float() + b.float()))


# RF3 -> tt-bio leaf names for the shared Transition block. Same table the Pairformer
# and MSA remaps use; scoping straight into the RF3 dict hands it `layer_norm_1` and
# it asks for `norm`.
TRANSITION_LEAVES = {
    "layer_norm_1": "norm",
    "linear_1": "fc1",
    "linear_2": "fc2",
    "linear_3": "fc3",
}


def _transition(raw: dict, prefix: str) -> dict:
    out = {}
    for k, v in raw.items():
        if not k.startswith(prefix + "."):
            continue
        leaf, _, tail = k[len(prefix) + 1:].partition(".")
        out[f"{TRANSITION_LEAVES[leaf]}.{tail}"] = v
    return out


class DiffusionConditioning(Module):
    def __init__(self, state_dict, compute_kernel_config, sigma_data: float = 16.0):
        super().__init__(state_dict, compute_kernel_config)
        self.sigma_data = sigma_data

        self.relpos_w = self.torch_to_tt("relative_position_encoding.linear.weight")
        self.zii_norm_w = self.torch_to_tt("to_zii.0.weight")
        self.zii_norm_b = self.torch_to_tt("to_zii.0.bias")
        self.zii_w = self.torch_to_tt("to_zii.1.weight")
        self.si_norm_w = self.torch_to_tt("to_si.0.weight")
        self.si_norm_b = self.torch_to_tt("to_si.0.bias")
        self.si_w = self.torch_to_tt("to_si.1.weight")
        self.n_norm_w = self.torch_to_tt("process_n.0.weight")
        self.n_norm_b = self.torch_to_tt("process_n.0.bias")
        self.n_w = self.torch_to_tt("process_n.1.weight")

        self.fourier_w = self.weights["fourier_embedding.w"].float()
        self.fourier_b = self.weights["fourier_embedding.b"].float()

        raw = self.weights.as_dict()
        self.transition_1 = [
            Transition(_transition(raw, f"transition_1.{i}"), compute_kernel_config)
            for i in range(2)
        ]
        self.transition_2 = [
            Transition(_transition(raw, f"transition_2.{i}"), compute_kernel_config)
            for i in range(2)
        ]

    def pair(self, relpos_feat: ttnn.Tensor, z_trunk: ttnn.Tensor) -> ttnn.Tensor:
        r = ttnn.linear(relpos_feat, self.relpos_w,
                        compute_kernel_config=self.compute_kernel_config)
        z = ttnn.concat([z_trunk, r], dim=-1)
        ttnn.deallocate(r)
        z = ttnn.layer_norm(z, weight=self.zii_norm_w, bias=self.zii_norm_b,
                            epsilon=1e-5,
                            compute_kernel_config=self.compute_kernel_config)
        z = ttnn.linear(z, self.zii_w,
                        compute_kernel_config=self.compute_kernel_config)
        for tr in self.transition_1:
            z = ttnn.add(z, tr(z))
        return z

    def single(self, s_trunk: ttnn.Tensor, s_inputs: ttnn.Tensor,
               n_embed: ttnn.Tensor) -> ttnn.Tensor:
        s = ttnn.concat([s_trunk, s_inputs], dim=-1)
        s = ttnn.layer_norm(s, weight=self.si_norm_w, bias=self.si_norm_b,
                            epsilon=1e-5,
                            compute_kernel_config=self.compute_kernel_config)
        s = ttnn.linear(s, self.si_w, compute_kernel_config=self.compute_kernel_config,
                        core_grid=CORE_GRID_MAIN)
        n = ttnn.layer_norm(n_embed, weight=self.n_norm_w, bias=self.n_norm_b,
                            epsilon=1e-5,
                            compute_kernel_config=self.compute_kernel_config)
        n = ttnn.linear(n, self.n_w, compute_kernel_config=self.compute_kernel_config)
        s = ttnn.add(s, ttnn.unsqueeze(n, -2))
        for tr in self.transition_2:
            s = ttnn.add(s, tr(s))
        return s

    def single_at(self, s_trunk, s_inputs, t: torch.Tensor) -> ttnn.Tensor:
        """The t-dependent half on its own. `pair` takes no `t`, so a rollout that has
        hoisted it needs only this one."""
        n = fourier_embedding(t, self.fourier_w, self.fourier_b, self.sigma_data)
        n_embed = ttnn.from_torch(n.reshape(1, -1, n.shape[-1]),
                                  layout=ttnn.TILE_LAYOUT, device=self.device,
                                  dtype=_dtype(ttnn.bfloat16))
        return self.single(s_trunk, s_inputs, n_embed)

    def __call__(self, relpos_feat, z_trunk, s_trunk, s_inputs, t: torch.Tensor):
        return (self.single_at(s_trunk, s_inputs, t),
                self.pair(relpos_feat, z_trunk))
