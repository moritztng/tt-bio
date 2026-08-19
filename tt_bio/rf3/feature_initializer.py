"""RF3's FeatureInitializer on ttnn.

    S_inputs = cat([A_I from the atom encoder, restype, profile, deletion_mean])
    S_init   = to_s_init(S_inputs)
    Z_init   = to_z_init_i(S_inputs).unsqueeze(-3) + to_z_init_j(S_inputs).unsqueeze(-2)
    Z_init  += relpos_linear(relpos_features(f))
    Z_init  += process_token_bonds(token_bonds)

`deletion_mean` is the only feature that needs an unsqueeze -- upstream keeps that in a
`features_to_unsqueeze` list precisely because it is 1-D where the others are not.
"""

from __future__ import annotations

import torch
import ttnn

from tt_bio.rf3.atom_encoder import AtomAttentionEncoder
from tt_bio.tenstorrent import CORE_GRID_MAIN, Module

#: the order is load-bearing: it fixes the 449 columns to_s_init and to_z_init read
FEATURES = ["restype", "profile", "deletion_mean"]
FEATURES_TO_UNSQUEEZE = {"deletion_mean"}


def token_features(f: dict, n_token: int) -> torch.Tensor:
    """The per-token block appended to A_I to make S_inputs [I, 449]."""
    cols = []
    for name in FEATURES:
        t = f[name].float()
        if name in FEATURES_TO_UNSQUEEZE:
            t = t.unsqueeze(-1)
        cols.append(t.reshape(n_token, -1))
    return torch.cat(cols, dim=-1)


class FeatureInitializer(Module):
    def __init__(self, state_dict, compute_kernel_config, mlff_const: torch.Tensor):
        super().__init__(state_dict, compute_kernel_config)
        enc = {k[len("input_feature_embedder.atom_attention_encoder."):]: v
               for k, v in self.weights.as_dict().items()
               if k.startswith("input_feature_embedder.atom_attention_encoder.")}
        self.encoder = AtomAttentionEncoder(enc, compute_kernel_config, mlff_const)
        self.to_s_init = self.torch_to_tt("to_s_init.weight")
        self.to_z_i = self.torch_to_tt("to_z_init_i.weight")
        self.to_z_j = self.torch_to_tt("to_z_init_j.weight")
        self.relpos = self.torch_to_tt("relative_position_encoding.linear.weight")
        self.bonds = self.torch_to_tt("process_token_bonds.weight")

    def __call__(self, single_in, pair_in, v, keys_indexing, atom_to_token, mask,
                 n_pad, token_feats, relpos_feat, bond_feat):
        a_i, _, _, _ = self.encoder(single_in, pair_in, v, keys_indexing,
                                    atom_to_token, mask, n_pad)
        s_inputs = ttnn.concat([a_i, token_feats], dim=-1)
        s_init = ttnn.linear(s_inputs, self.to_s_init,
                             compute_kernel_config=self.compute_kernel_config,
                             core_grid=CORE_GRID_MAIN)
        zi = ttnn.linear(s_inputs, self.to_z_i,
                         compute_kernel_config=self.compute_kernel_config)
        zj = ttnn.linear(s_inputs, self.to_z_j,
                         compute_kernel_config=self.compute_kernel_config)
        # The reference gives to_z_init_i the -3 unsqueeze and to_z_init_j the -2, NOT
        # the other way round. i and j are different learned projections, so swapping
        # them transposes the asymmetric part of Z_init: pcc stays at 0.93 because the
        # symmetric part dominates, while rel_rms goes to 133x the ceiling.
        z_init = ttnn.add(ttnn.unsqueeze(zi, -3), ttnn.unsqueeze(zj, -2))
        z_init = ttnn.add(z_init, ttnn.linear(
            relpos_feat, self.relpos,
            compute_kernel_config=self.compute_kernel_config))
        z_init = ttnn.add(z_init, ttnn.linear(
            bond_feat, self.bonds,
            compute_kernel_config=self.compute_kernel_config))
        return s_inputs, s_init, z_init
