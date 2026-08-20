"""RF3's atom attention encoder on ttnn.

The formulas here are already proven bit-exact in host torch against the captured
reference (`scripts/rf3_port/probe_pair_formula.py --autocast`), so any gap this
module shows is the ttnn's, not the formula's.

Three things are not what a reading of the reference's type hints would suggest, and
all three are load-bearing (see the state file):

  * the atom transformer runs LOCAL WINDOWED attention -- `Beta_lm = True` is a
    sentinel that diverts `AttentionPairBiasDiffusion.forward` into `atom_attention`
    on its first line, so it is built with `atom_level=True`
  * out-of-range keys must be masked ADDITIVELY. tt-bio's windowed gather yields zero
    key rows, and a zeroed key still enters the softmax with logit `0*q + bias`; at
    L=115 that is 30% of key slots
  * the blocks set `no_residual=True` (transition reads the block input) and
    `a_to_b_gate=False` (2-factor SwiGLU, not Boltz-2's 3-factor one)

Because the attention is local, the pair track is carried WINDOWED end to end, as
[K, 32, 128, c] rather than [L_atom, L_atom, c]. Nothing outside this module and the
decoder ever reads it, and every operation on it is elementwise in the pair index or a
matmul over the channel axis, so the windowed track holds the same numbers the dense one
held at the entries the attention reads -- 1/32.5 of them at 512 aa.
"""

from __future__ import annotations

import torch
import ttnn

from tt_bio.rf3.atom_encoder_host import (ATOM_KEYS, ATOM_WINDOW, PAD_LEFT,
                                          fused_pair_weight)
from tt_bio.rf3.remap_encoder import (atom_transformer_bias_weights,
                                      remap_atom_transformer)
from tt_bio.tenstorrent import (CORE_GRID_MAIN, DiffusionTransformer, Module,
                                _dtype)

NEG = -1e9
N_HEADS = 4
C_ATOM = 128
C_ATOMPAIR = 16


def window_mask(n_atom: int, n_atom_padded: int) -> torch.Tensor:
    """[K, 1, ATOM_WINDOW, ATOM_KEYS] additive mask: 0 in range, -1e9 out.

    Covers both kinds of out-of-range at once -- keys that fall outside the sequence
    because the window overhangs its end, and keys that are only there because the
    atom count was padded up to a multiple of the window.
    """
    k = n_atom_padded // ATOM_WINDOW
    idx = (torch.arange(k)[:, None] * ATOM_WINDOW
           + torch.arange(ATOM_KEYS)[None, :] - PAD_LEFT)      # [K, ATOM_KEYS]
    bad = (idx < 0) | (idx >= n_atom)
    return torch.where(bad, torch.full_like(idx, NEG, dtype=torch.float32),
                       torch.zeros_like(idx, dtype=torch.float32))[:, None, None, :] \
        .expand(k, 1, ATOM_WINDOW, ATOM_KEYS).contiguous()


def key_window(x, keys_indexing, n_pad: int, compute_kernel_config) -> ttnn.Tensor:
    """[1, n_pad, c] -> [K, 1, ATOM_KEYS, c]: a key-indexed term, gathered per window.

    The same one-hot matmul the atom transformer already applies to its own key track
    (`boltz2.single_to_keys`). A window's 128 keys are eight consecutive 16-atom
    half-blocks, so `keys_indexing` maps 2K half-blocks to 8K window slots in one
    matmul, with all-zero columns where the window runs off the end. It is a one-hot, so
    each gathered value is carried through exactly.
    """
    k, c = n_pad // ATOM_WINDOW, x.shape[-1]
    y = ttnn.reshape(x, (1, 2 * k, ATOM_WINDOW // 2, c))
    y = ttnn.permute(y, (0, 2, 3, 1))                     # [1, W/2, c, 2K]
    y = ttnn.matmul(y, keys_indexing, compute_kernel_config=compute_kernel_config,
                    core_grid=CORE_GRID_MAIN)             # [1, W/2, c, 8K]
    y = ttnn.permute(y, (0, 3, 1, 2))                     # [1, 8K, W/2, c]
    return ttnn.reshape(y, (k, 1, ATOM_KEYS, c))


def windowed_bias(p, ln0_w, ln0_b, to_b, mask, n_pad, compute_kernel_config):
    """Masked pair bias for one block: [1, K*n_heads, W, ATOM_KEYS].

    Shared by the encoder and the decoder, which run the same 3-block atom transformer
    over the same pair track.

    The pair track is already windowed -- [K, W, ATOM_KEYS, c] -- so this is the
    projection, a permute and the additive mask, with no per-window gather left to do.
    `mask` puts -1e9 on the key slots that exist only because the window overhangs the
    sequence or the atom count was padded up to a multiple of the window; those lanes
    leave the softmax at exactly zero, so whatever the pair track holds there does not
    reach the output.
    """
    b = ttnn.layer_norm(p, weight=ln0_w, bias=ln0_b, epsilon=1e-5,
                        compute_kernel_config=compute_kernel_config)
    b = ttnn.linear(b, to_b, compute_kernel_config=compute_kernel_config)
    b = ttnn.permute(b, (0, 3, 1, 2))                     # [K, heads, W, KEYS]
    b = ttnn.add(b, mask)                                 # additive -1e9
    return ttnn.reshape(b, (1, (n_pad // ATOM_WINDOW) * N_HEADS,
                            ATOM_WINDOW, ATOM_KEYS))


class AtomAttentionEncoder(Module):
    def __init__(self, state_dict, compute_kernel_config, mlff_const: torch.Tensor,
                 n_block: int = 3):
        super().__init__(state_dict, compute_kernel_config)
        raw = self.weights.as_dict()
        self.n_block = n_block

        self.proc_in = self.torch_to_tt("process_input_features.weight")
        # constant, not an MLP: the atom-level-embedding inputs are all-zero at public
        # inference but the track is NOT a no-op (biases + a LayerNorm). Precomputed
        # in the reference's own precision -- see feature_init.mlff_constant.
        self.mlff = ttnn.from_torch(
            mlff_const.reshape(1, 1, -1).float(), layout=ttnn.TILE_LAYOUT,
            device=self.device, dtype=_dtype(ttnn.bfloat16))
        # one fused [16, 5] instead of three sub-tile matmuls; all three terms are
        # multiplied by V, so the sum of linears is a linear of the concat
        pw = torch.nn.functional.pad(fused_pair_weight(raw), (0, 32 - 5))
        self.pair_w = ttnn.from_torch(
            pw.t().contiguous(), layout=ttnn.TILE_LAYOUT, device=self.device,
            dtype=_dtype(ttnn.bfloat16))
        self.sl = self.torch_to_tt("process_single_l.1.weight")
        self.sm = self.torch_to_tt("process_single_m.1.weight")
        self.mlp = [self.torch_to_tt(f"pair_mlp.{i}.weight") for i in (1, 3, 5)]
        self.proc_q = self.torch_to_tt("process_q.0.weight")

        # per-block pair bias, held out of the block so the -1e9 mask can be folded in
        self.ln0_w, self.ln0_b, self.to_b = [], [], []
        for i in range(n_block):
            bw = atom_transformer_bias_weights(raw, i)
            self.ln0_w.append(ttnn.from_torch(
                bw["ln_0.weight"].float(), layout=ttnn.TILE_LAYOUT,
                device=self.device, dtype=_dtype(ttnn.bfloat16)))
            self.ln0_b.append(ttnn.from_torch(
                bw["ln_0.bias"].float(), layout=ttnn.TILE_LAYOUT,
                device=self.device, dtype=_dtype(ttnn.bfloat16)))
            self.to_b.append(ttnn.from_torch(
                bw["to_b.weight"].t().contiguous(), layout=ttnn.TILE_LAYOUT,
                device=self.device, dtype=_dtype(ttnn.bfloat16)))

        self.transformer = DiffusionTransformer(
            n_layers=n_block, dim=C_ATOM, n_heads=N_HEADS, atom_level=True,
            state_dict=remap_atom_transformer(raw, n_block),
            compute_kernel_config=compute_kernel_config,
            no_residual=True,      # transition reads the block input
            a_to_b_gate=False,     # 2-factor SwiGLU
            # Measured, not assumed. Off, this stack sits 20x above the pair-track
            # bf16 ceiling (Q_L rel_rms 0.1123 vs 0.0049); on, it sits at it (0.0063).
            # Same failure this port already paid four passes for in the trunk.
            fp32_softmax=True,
        )

    # ------------------------------------------------------------------ pieces

    def single(self, single_in: ttnn.Tensor) -> ttnn.Tensor:
        c = ttnn.linear(single_in, self.proc_in,
                        compute_kernel_config=self.compute_kernel_config,
                        core_grid=CORE_GRID_MAIN)
        return ttnn.add(c, self.mlff)

    def pair(self, pair_in, v, c, keys_indexing, n_pad: int) -> ttnn.Tensor:
        p = ttnn.linear(pair_in, self.pair_w,
                        compute_kernel_config=self.compute_kernel_config)
        return self.pair_terms(ttnn.multiply(p, v), c, keys_indexing, n_pad)

    def pair_terms(self, p, c, keys_indexing, n_pad: int) -> ttnn.Tensor:
        """The two single-track terms and the 3-layer MLP, shared with the diffusion
        encoder, which reaches the same tail after adding its trunk terms.

        `sl` indexes the QUERY atom and `sm` the KEY atom. The query axis is the window's
        own 32 rows, so that term is added through the flat [1, Lp, KEYS, c] view of the
        same buffer -- a leading-dim reshape, no relayout. The key axis is the one that
        needs the windowed gather.
        """
        k, cp = n_pad // ATOM_WINDOW, p.shape[-1]
        rc = ttnn.relu(c)
        sl = ttnn.linear(rc, self.sl, compute_kernel_config=self.compute_kernel_config)
        sm = ttnn.linear(rc, self.sm, compute_kernel_config=self.compute_kernel_config)
        p = ttnn.add(ttnn.reshape(p, (1, n_pad, ATOM_KEYS, cp)),
                     ttnn.unsqueeze(sl, -2))
        p = ttnn.add(ttnn.reshape(p, (k, ATOM_WINDOW, ATOM_KEYS, cp)),
                     key_window(sm, keys_indexing, n_pad, self.compute_kernel_config))
        m = p
        for w in self.mlp:
            m = ttnn.linear(ttnn.relu(m), w,
                            compute_kernel_config=self.compute_kernel_config)
        return ttnn.add(p, m)

    def bias(self, p: ttnn.Tensor, i: int, mask: ttnn.Tensor, n_pad: int) -> ttnn.Tensor:
        return windowed_bias(p, self.ln0_w[i], self.ln0_b[i], self.to_b[i], mask, n_pad,
                             self.compute_kernel_config)

    # ----------------------------------------------------------------- forward

    def __call__(self, single_in, pair_in, v, keys_indexing, atom_to_token,
                 mask, n_pad: int):
        c = self.single(single_in)
        p = self.pair(pair_in, v, c, keys_indexing, n_pad)
        biases = [self.bias(p, i, mask, n_pad) for i in range(self.n_block)]
        # tt-bio's atom-level path takes the single track ALREADY windowed as
        # [B, K, W, D], not flat [B, L, D].
        k = n_pad // ATOM_WINDOW
        cw = ttnn.reshape(c, (1, k, ATOM_WINDOW, C_ATOM))
        q = self.transformer(cw, cw, biases, keys_indexing)
        q = ttnn.reshape(q, (1, n_pad, C_ATOM))
        a = ttnn.linear(q, self.proc_q, compute_kernel_config=self.compute_kernel_config,
                        core_grid=CORE_GRID_MAIN, activation="relu")
        return ttnn.matmul(atom_to_token, a,
                           compute_kernel_config=self.compute_kernel_config), q, c, p
