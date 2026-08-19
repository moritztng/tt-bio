"""RF3's diffusion atom attention encoder on ttnn.

Shares the windowing, the -1e9 additive mask and the 3-block atom transformer with the
feature-initializer encoder (subclassed from it), and adds the four things that make it
a different module -- see "Diffusion atom encoder: pre-flighted" in the state file.

The ORDER below is load-bearing and is not the order the additions read in:

    Q = C                       # binds; Q keeps the PRE-trunk single track
    C = C + s_trunk_atom        # rebinds C; the transformer gets the POST-trunk one
    Q = proc_r(R) + Q
    Q = proc_ch(chiral) + Q
    P = P + sl(relu(C)) + sm(relu(C))     # uses the POST-trunk C

Upstream asserts `not (C_L == Q_L).all()`, which is the tell that Q and C are meant to
diverge. Writing the two additions in the natural order folds the trunk into Q as well
and costs a few percent rather than crashing.
"""

from __future__ import annotations

import torch
import ttnn

from tt_bio.rf3.atom_encoder import (ATOM_KEYS, ATOM_WINDOW, AtomAttentionEncoder)
from tt_bio.tenstorrent import CORE_GRID_MAIN, _dtype, batched_matmul

C_TOKEN_DIFFUSION = 768


class DiffusionAtomEncoder(AtomAttentionEncoder):
    def __init__(self, state_dict, compute_kernel_config, mlff_const: torch.Tensor,
                 n_block: int = 3):
        super().__init__(state_dict, compute_kernel_config, mlff_const, n_block)
        self.s_trunk_norm_w = self.torch_to_tt("process_s_trunk.0.weight")
        self.s_trunk_norm_b = self.torch_to_tt("process_s_trunk.0.bias")
        self.s_trunk_w = self.torch_to_tt("process_s_trunk.1.weight")
        self.z_norm_w = self.torch_to_tt("process_z.0.weight")
        self.z_norm_b = self.torch_to_tt("process_z.0.bias")
        self.z_w = self.torch_to_tt("process_z.1.weight")
        self.r_w = self.torch_to_tt("process_r.weight")
        self.ch_w = (self.torch_to_tt("process_ch.weight")
                     if "process_ch.weight" in self.weights else None)

    def _trunk_single(self, s_trunk: ttnn.Tensor, a2t: ttnn.Tensor) -> ttnn.Tensor:
        """process_s_trunk(S)[..., tok_idx, :] -- gather written as a one-hot matmul."""
        s = ttnn.layer_norm(s_trunk, weight=self.s_trunk_norm_w, bias=self.s_trunk_norm_b,
                            epsilon=1e-5,
                            compute_kernel_config=self.compute_kernel_config)
        s = ttnn.linear(s, self.s_trunk_w,
                        compute_kernel_config=self.compute_kernel_config,
                        core_grid=CORE_GRID_MAIN)
        return ttnn.matmul(a2t, s, compute_kernel_config=self.compute_kernel_config)

    def _trunk_pair(self, z: ttnn.Tensor, a2t: ttnn.Tensor, a2t_win: ttnn.Tensor,
                    n_pad: int) -> ttnn.Tensor:
        """process_z(Z)[..., tok_idx, :, :][..., tok_idx, :], windowed.

        With `broadcast_trunk_feats_on_1dim_old=False` this checkpoint takes the DOUBLE
        gather, i.e. both axes are expanded token->atom. As a matmul that is
        A @ Z @ A^T per channel.

        The query gather has to produce every atom row, so it is the dense
        [Lp, I] @ [I, I*c] and is unchanged. The key gather only ever needs each query
        block's own 128 keys, so it becomes a batched [K, 32c, I] @ [K, I, 128] against
        the windowed one-hot -- the Lp/128 saving, on the one matmul that carried the
        atom-pair track's cost. Folding the channel axis into the row axis keeps both
        operands rank 4 with matching leading dims, which is what `batched_matmul`
        needs to spread the K windows over the core grid.
        """
        p = ttnn.layer_norm(z, weight=self.z_norm_w, bias=self.z_norm_b, epsilon=1e-5,
                            compute_kernel_config=self.compute_kernel_config)
        p = ttnn.linear(p, self.z_w, compute_kernel_config=self.compute_kernel_config)
        _, i_tok, _, c_pair = p.shape
        k = n_pad // ATOM_WINDOW
        g = ttnn.matmul(a2t, ttnn.reshape(p, (1, i_tok, i_tok * c_pair)),
                        compute_kernel_config=self.compute_kernel_config)
        g = ttnn.reshape(g, (1, n_pad, i_tok, c_pair))
        g = ttnn.permute(g, (0, 1, 3, 2))                      # [1, Lp, c_pair, I]
        g = ttnn.reshape(g, (1, k, ATOM_WINDOW * c_pair, i_tok))
        g = batched_matmul(g, a2t_win,
                           compute_kernel_config=self.compute_kernel_config)
        g = ttnn.reshape(g, (k, ATOM_WINDOW, c_pair, ATOM_KEYS))
        return ttnn.permute(g, (0, 1, 3, 2))                   # [K, W, KEYS, c_pair]

    def __call__(self, single_in, pair_in, v, keys_indexing, atom_to_token,
                 mask, n_pad, s_trunk, z_trunk, r_noisy, chiral, a2t_onehot,
                 a2t_win):
        c = self.single(single_in)
        p = ttnn.multiply(
            ttnn.linear(pair_in, self.pair_w,
                        compute_kernel_config=self.compute_kernel_config), v)

        q = c                                        # PRE-trunk, deliberately
        c = ttnn.add(c, self._trunk_single(s_trunk, a2t_onehot))
        p = ttnn.add(p, self._trunk_pair(z_trunk, a2t_onehot, a2t_win, n_pad))

        q = ttnn.add(ttnn.linear(r_noisy, self.r_w,
                                 compute_kernel_config=self.compute_kernel_config), q)
        if self.ch_w is not None and chiral is not None:
            q = ttnn.add(ttnn.linear(chiral, self.ch_w,
                                     compute_kernel_config=self.compute_kernel_config), q)

        p = self.pair_terms(p, c, keys_indexing, n_pad)

        biases = [self.bias(p, i, mask, n_pad) for i in range(self.n_block)]
        k = n_pad // 32
        qw = ttnn.reshape(q, (1, k, 32, -1))
        cw = ttnn.reshape(c, (1, k, 32, -1))
        out = self.transformer(qw, cw, biases, keys_indexing)
        out = ttnn.reshape(out, (1, n_pad, -1))
        a = ttnn.linear(out, self.proc_q,
                        compute_kernel_config=self.compute_kernel_config,
                        core_grid=CORE_GRID_MAIN, activation="relu")
        return ttnn.matmul(atom_to_token, a,
                           compute_kernel_config=self.compute_kernel_config), out, c, p
