"""Host-side input assembly for the RF3 port.

Every ported component takes device tensors on a padded atom axis, and until now each
parity harness rebuilt that padding itself. The duplication is not just untidy: the
window padding, the two atom-to-token matrices (a mean-pooling one for the encoder and
a one-hot one for the decoder, transposed relative to each other) and the 32-column
pair block whose last 27 columns are zero are all easy to get subtly wrong, and getting
one wrong produces a plausible structure rather than an error.

`HostInputs.build` does it once. Per-step quantities -- the noisy coordinates and the
chirality gradients, which change every diffusion step -- are `step_inputs`, kept
separate because the rest is built once per target.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import ttnn

from tt_bio.boltz2 import get_indexing_matrix
from tt_bio.rf3.atom_encoder import window_mask
from tt_bio.rf3.atom_encoder_host import (ATOM_KEYS, ATOM_WINDOW,
                                          atom_to_token_mean,
                                          pair_inputs_windowed, single_features,
                                          token_to_atom_windowed)
from tt_bio.rf3.confidence_head import predicted_distance_onehot
from tt_bio.rf3.feature_init import (assert_mlff_inputs_zero, relpos_features,
                                     token_bond_features)
from tt_bio.rf3.feature_initializer import token_features
from tt_bio.rf3.template import template_features

#: width of the atom-encoder's single-feature block, and of its pair block. The pair
#: block is 32 wide but only 5 columns carry features; the rest is the tile pad.
C_SINGLE_IN = 393
C_PAIR_IN = 32


def to_device(x: torch.Tensor, device) -> ttnn.Tensor:
    return ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=device,
                           dtype=ttnn.bfloat16)


@dataclass
class HostInputs:
    """Everything the device model needs that does not change between steps."""

    n_atom: int
    n_atom_padded: int
    n_token: int

    single_in: ttnn.Tensor
    #: the pair track is windowed: [K, ATOM_WINDOW, ATOM_KEYS, C_PAIR_IN], not
    #: [1, Lp, Lp, C_PAIR_IN]. The atom transformer reads Lp x 128 of the Lp^2 pairs a
    #: dense build produces, so the dense one is 32.5x too big at 512 aa and 4.40 GB at
    #: 1024 aa, where it did not fit at all.
    pair_in: ttnn.Tensor
    pair_v: ttnn.Tensor
    keys_indexing: ttnn.Tensor
    window_mask: ttnn.Tensor
    atom_to_token_mean: ttnn.Tensor
    atom_to_token: ttnn.Tensor
    #: [1, K, I, ATOM_KEYS]: the token->atom one-hot restricted to each window's keys,
    #: for the second gather in the diffusion encoder's `_trunk_pair`.
    token_to_atom_win: ttnn.Tensor

    token_feats: ttnn.Tensor
    relpos_feat: ttnn.Tensor
    bond_feat: ttnn.Tensor
    template_feats: ttnn.Tensor

    #: [n_recycle] of [1, n_msa, I, c_msa]; the featurizer draws one per recycle
    msa_stack: list[ttnn.Tensor]

    #: kept on host: the sampler and the chirality gradients need them per step
    atom_to_token_map: torch.Tensor
    chiral_centers: torch.Tensor
    chiral_dihedrals: torch.Tensor
    rep_atom_idxs: torch.Tensor | None

    @staticmethod
    def build(f: dict, device, *, r_max: int = 32, s_max: int = 2) -> "HostInputs":
        # The MLFF constant folded into every atom's C_L is only valid while the MACE
        # embeddings are absent. Check here, once, rather than in each component.
        assert_mlff_inputs_zero(f)

        ff = {k: (v.clone() if isinstance(v, torch.Tensor) else v)
              for k, v in f.items()}
        L = int(ff["atom_to_token_map"].shape[0])
        I = int(ff["atom_to_token_map"].max()) + 1
        Lp = ((L + ATOM_WINDOW - 1) // ATOM_WINDOW) * ATOM_WINDOW
        K = Lp // ATOM_WINDOW
        ff["ref_atom_name_chars"] = ff["ref_atom_name_chars"].reshape(L, -1)

        s_in = torch.zeros(1, Lp, C_SINGLE_IN)
        s_in[0, :L] = single_features(ff, L)
        p_raw, v_in = pair_inputs_windowed(ff, L, Lp)
        p_in = torch.zeros(K, ATOM_WINDOW, ATOM_KEYS, C_PAIR_IN)
        p_in[..., :p_raw.shape[-1]] = p_raw

        # Two different matrices, and the difference matters. The encoder aggregates
        # atoms into tokens by MEAN (so its rows sum to 1); the decoder broadcasts a
        # token back to its atoms, which is the one-hot, transposed.
        a2t_mean = torch.zeros(1, I, Lp)
        a2t_mean[0, :, :L] = atom_to_token_mean(ff, L, I)
        a2t = torch.zeros(1, Lp, I)
        a2t[0, torch.arange(L), ff["atom_to_token_map"].long()[:L]] = 1.0

        msa_stack = ff["msa_stack"]
        return HostInputs(
            n_atom=L, n_atom_padded=Lp, n_token=I,
            single_in=to_device(s_in, device),
            pair_in=to_device(p_in, device),
            pair_v=to_device(v_in, device),
            keys_indexing=to_device(
                get_indexing_matrix(K, ATOM_WINDOW, ATOM_KEYS, torch.device("cpu")),
                device),
            window_mask=to_device(window_mask(L, Lp), device),
            atom_to_token_mean=to_device(a2t_mean, device),
            atom_to_token=to_device(a2t, device),
            token_to_atom_win=to_device(token_to_atom_windowed(a2t, Lp), device),
            token_feats=to_device(token_features(ff, I).reshape(1, I, -1), device),
            relpos_feat=to_device(
                relpos_features(ff, r_max=r_max, s_max=s_max).unsqueeze(0), device),
            bond_feat=to_device(token_bond_features(ff).unsqueeze(0), device),
            template_feats=to_device(template_features(ff).unsqueeze(0), device),
            msa_stack=[to_device(msa_stack[i].unsqueeze(0), device)
                       for i in range(msa_stack.shape[0])],
            atom_to_token_map=ff["atom_to_token_map"],
            chiral_centers=ff["chiral_centers"],
            chiral_dihedrals=ff["chiral_center_dihedral_angles"],
            rep_atom_idxs=None,
        )

    def step_inputs(self, r_noisy: torch.Tensor, device
                    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """The two per-step atom-level tensors: scaled coordinates and chirality.

        The chirality gradient is autograd through the loss module's dihedral term at
        inference -- RF3's implicit-chirality representation, and the reason this port
        needs a backward pass in a forward-only engine. It stays on host: it is
        per-chiral-centre gather/scatter arithmetic on a handful of atoms.
        """
        from tt_bio._vendor.rf3.loss.loss import calc_chiral_grads_flat_impl

        L, Lp = self.n_atom, self.n_atom_padded
        r_in = torch.zeros(1, Lp, 3)
        r_in[0, :L] = r_noisy.reshape(-1, 3)[:L]
        with torch.no_grad(), torch.autocast("cpu", enabled=False):
            ch = calc_chiral_grads_flat_impl(
                r_noisy.detach().clone().float(), self.chiral_centers.long(),
                self.chiral_dihedrals.float(), False).nan_to_num()
        ch_in = torch.zeros(1, Lp, 3)
        ch_in[0, :L] = ch.reshape(-1, 3)[:L]
        return to_device(r_in, device), to_device(ch_in, device)


def distance_onehot(x_pred: torch.Tensor, rep_atom_idxs: torch.Tensor,
                    device) -> ttnn.Tensor:
    """The confidence head's binned representative-atom distances, on device."""
    return to_device(predicted_distance_onehot(x_pred, rep_atom_idxs), device)
