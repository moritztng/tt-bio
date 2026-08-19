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


def windowed_bias(p, ln0_w, ln0_b, to_b, mask, n_pad, compute_kernel_config, device):
    """Windowed, masked pair bias for one block: [1, K*n_heads, W, ATOM_KEYS].

    Shared by the encoder and the decoder, which run the same 3-block atom transformer
    over the same pair track.

    Shifting the bias right by PAD_LEFT makes each 128-key window the contiguous slice
    [32k, 32k+128); the shifted-in columns are exactly the out-of-range slots, and the
    additive `mask` puts -1e9 there. Padding with zeros rather than -1e9 is fine because
    the mask lands on top -- what matters is that these lanes never reach the softmax
    carrying a real value.
    """
    b = ttnn.layer_norm(p, weight=ln0_w, bias=ln0_b, epsilon=1e-5,
                        compute_kernel_config=compute_kernel_config)
    b = ttnn.linear(b, to_b, compute_kernel_config=compute_kernel_config)
    b = ttnn.permute(b, (0, 3, 1, 2))                     # [1, heads, L, L]
    k = n_pad // ATOM_WINDOW
    blocks = []
    for j in range(k):
        lo, hi = j * ATOM_WINDOW - PAD_LEFT, j * ATOM_WINDOW - PAD_LEFT + ATOM_KEYS
        lo_c, hi_c = max(lo, 0), min(hi, n_pad)
        piece = b[:, :, j * ATOM_WINDOW:(j + 1) * ATOM_WINDOW, lo_c:hi_c]
        pads = []
        if lo_c > lo:
            pads.append(ttnn.zeros((1, N_HEADS, ATOM_WINDOW, lo_c - lo),
                                   layout=ttnn.TILE_LAYOUT, device=device,
                                   dtype=piece.dtype))
        pads.append(piece)
        if hi > hi_c:
            pads.append(ttnn.zeros((1, N_HEADS, ATOM_WINDOW, hi - hi_c),
                                   layout=ttnn.TILE_LAYOUT, device=device,
                                   dtype=piece.dtype))
        blocks.append(ttnn.concat(pads, dim=-1) if len(pads) > 1 else piece)
    w = ttnn.concat(blocks, dim=0)                        # [K, heads, W, KEYS]
    w = ttnn.add(w, mask)                                 # additive -1e9
    return ttnn.reshape(w, (1, k * N_HEADS, ATOM_WINDOW, ATOM_KEYS))


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

    def pair(self, pair_in: ttnn.Tensor, v: ttnn.Tensor, c: ttnn.Tensor) -> ttnn.Tensor:
        p = ttnn.linear(pair_in, self.pair_w,
                        compute_kernel_config=self.compute_kernel_config)
        p = ttnn.multiply(p, v)
        rc = ttnn.relu(c)
        sl = ttnn.linear(rc, self.sl, compute_kernel_config=self.compute_kernel_config)
        sm = ttnn.linear(rc, self.sm, compute_kernel_config=self.compute_kernel_config)
        # unsqueeze(-2) indexes the QUERY atom, unsqueeze(-3) the KEY atom
        p = ttnn.add(p, ttnn.unsqueeze(sl, -2))
        p = ttnn.add(p, ttnn.unsqueeze(sm, -3))
        m = p
        for w in self.mlp:
            m = ttnn.linear(ttnn.relu(m), w,
                            compute_kernel_config=self.compute_kernel_config)
        return ttnn.add(p, m)

    def bias(self, p: ttnn.Tensor, i: int, mask: ttnn.Tensor, n_pad: int) -> ttnn.Tensor:
        return windowed_bias(p, self.ln0_w[i], self.ln0_b[i], self.to_b[i], mask, n_pad,
                             self.compute_kernel_config, self.device)

    # ----------------------------------------------------------------- forward

    def __call__(self, single_in, pair_in, v, keys_indexing, atom_to_token,
                 mask, n_pad: int):
        c = self.single(single_in)
        p = self.pair(pair_in, v, c)
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
