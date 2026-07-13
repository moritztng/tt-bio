"""OpenFold3 confidence heads (AF3 Algorithm 31) on Tenstorrent.

Ports ``aux_heads`` (PAE / PDE / pLDDT / distogram / experimentally_resolved) to device,
reusing the gated 4-block confidence ``Pairformer`` (``aux_heads.pairformer_embedding.
pairformer_stack``) for the confidence trunk's pair channel, and the Protenix-v2
``ConfidenceHead`` discipline (z-embedding + output heads host-fp32).

Hybrid confidence Pairformer (precision-motivated split):

  * **z-path on device (bf16, HiFi4 + fp32 dest acc)** -- the heavy pair compute
    (TriangleMultiplication / TriangleAttention / Transition on [N, N, 128]). Gates at
    zij_conf PCC 0.99624 vs the reference.
  * **s-path on host (fp32)** -- LN + AttentionPairBias + Transition on [N, 384]. The
    confidence Pairformer receives the trunk's raw final single ``si_trunk`` at ~196k
    magnitude (the reference passes it with NO glue/LayerNorm, unlike the trunk's own
    Pairformer which starts each cycle at s~187 via the s-glue). At 196k, bf16
    (resolution ~1024) corrupts the small per-block s-updates and the attention amplifies
    the error across the 4 blocks; the plddt / experimentally_resolved LayerNorm then
    strips the dominant residual and exposes the corruption. Running the s-path in fp32
    on host (fed the per-block device z for the pair bias) recovers it -- the s-track is
    light ([N, 384] attention) and the device keeps the heavy pair compute.

The host AttentionPairBias uses the reference formula (q scaled by 1/sqrt(c_hidden),
pair bias = ``linear_z(LN_z(z))`` with no further scaling), matching the fp32 golden.

The OF3 confidence Pairformer block layout is bit-identical to the trunk's
``pairformer_stack`` block (c_z=128, 4 tri-att heads, 16 attn-pair-bias heads,
c_hidden_pair_bias=24, c_s=384), so ``remap_pairformer_stack`` + ``Pairformer`` build it
with no new primitive code. The output heads differ from Protenix's layout (OF3 pLDDT /
resolved are a flat ``[N_tok, 23 * c_out]`` linear -> ``masked_select`` to atoms; PDE /
distogram symmetrise as ``L(z) + L(z).T``; distogram reads the trunk pair, not the
confidence-Pairformer output), so they are ported here rather than reused.

Validated vs the real OF3 reference golden
(``~/of3_ref_out.pkl["intermediates"]["confidence_heads_real"]``, captured by
``scripts/of3_confidence_golden.py``); see ``tests/test_openfold3_confidence.py`` for
per-head PCC.
"""

import math

import torch
import torch.nn.functional as F

from .tenstorrent import Pairformer
from .openfold3_weights import remap_pairformer_stack

# aux_heads.pairformer_embedding distance bins (config.model_config).
_MIN_BIN, _MAX_BIN, _NO_BIN, _INF = 3.25, 50.75, 39, 1e8
_MAX_ATOMS_PER_TOKEN = 23
# Confidence pairformer dims (config.model_config.heads.pairformer_embedding.pairformer
# + attn_pair_bias: c_s=384, c_z=128, c_s_input=449, 16 attn-pair-bias heads, head_dim 24).
_C_S, _C_Z, _C_S_INPUT = 384, 128, 449
_APB_HEADS, _APB_HEAD_DIM = 16, 24
_BLK = "pairformer_embedding.pairformer_stack.blocks.%d."


class OF3ConfidenceHead:
    """OF3 AF3-family confidence heads (device z-path + host-fp32 s-path).

    Args:
        aux_state_dict: ``aux_heads`` sub-state-dict (keys stripped of ``aux_heads.``).
        device: ttnn device.
        compute_kernel_config: HiFi4 + fp32 dest acc.
    """

    def __init__(self, aux_state_dict, device, compute_kernel_config):
        import re

        self._w = dict(aux_state_dict)
        self.dev = device
        self.compute_kernel_config = compute_kernel_config

        _pf_prefix = "pairformer_embedding.pairformer_stack"
        pf_sd = remap_pairformer_stack(self._w, prefix=_pf_prefix)
        _blk = re.compile(rf"^{re.escape(_pf_prefix)}\.blocks\.(\d+)\.")
        n_blocks = 1 + max(int(_blk.match(k).group(1)) for k in self._w if _blk.match(k))
        b0z = _BLK % 0 + "pair_stack."
        tri_att_n_heads = self._w[b0z + "tri_att_start.linear_z.weight"].shape[0]
        tri_att_head_dim = self._w[b0z + "tri_att_start.mha.linear_q.weight"].shape[0] // tri_att_n_heads
        apb0 = _BLK % 0 + "attn_pair_bias."
        att_n_heads = self._w[apb0 + "linear_z.weight"].shape[0]
        att_head_dim = self._w[apb0 + "mha.linear_q.weight"].shape[0] // att_n_heads
        # Pairformer holds the z-path sub-modules (and the s-path sub-modules, unused --
        # the s-path runs host-fp32 via _host_s_block for precision).
        self.pf = Pairformer(n_blocks, tri_att_head_dim, tri_att_n_heads,
                             att_head_dim, att_n_heads, True, pf_sd, compute_kernel_config)
        self.n_blocks = n_blocks

        bins = torch.linspace(_MIN_BIN, _MAX_BIN, _NO_BIN, dtype=torch.float32)
        self._squared_bins = bins ** 2
        self._upper = torch.cat([self._squared_bins[1:], self._squared_bins.new_tensor([_INF])])

    def _g(self, k):
        return self._w[k].float()

    def _bias(self, k):
        return self._w[k].float() if k in self._w else 0.0

    def _bw(self, i, name):
        return self._w[(_BLK % i) + name].float()

    def _host_s_block(self, s, z_host, i):
        """One host-fp32 s-path block: AttentionPairBias + Transition (reference formula).

        ``s`` is [N, c_s] fp32; ``z_host`` is the device-computed [N, N, c_z] pair for this
        block (brought host-side for the pair-bias LN+linear, which must match the
        reference's no-sqrt-scaling formula). Returns the updated ``s``.
        """
        pfx = "attn_pair_bias."
        a = F.layer_norm(s, (_C_S,), self._bw(i, pfx + "layer_norm_a.weight"),
                         self._bw(i, pfx + "layer_norm_a.bias"))
        zn = F.layer_norm(z_host, (_C_Z,), self._bw(i, pfx + "layer_norm_z.weight"),
                          self._bw(i, pfx + "layer_norm_z.bias"))
        bias = F.linear(zn, self._bw(i, pfx + "linear_z.weight")).permute(2, 0, 1)  # [H, N, N]
        q = F.linear(a, self._bw(i, pfx + "mha.linear_q.weight"),
                     self._bw(i, pfx + "mha.linear_q.bias"))
        k = F.linear(a, self._bw(i, pfx + "mha.linear_k.weight"))
        v = F.linear(a, self._bw(i, pfx + "mha.linear_v.weight"))
        N = a.shape[0]
        q = q.view(N, _APB_HEADS, _APB_HEAD_DIM).permute(1, 0, 2) / math.sqrt(_APB_HEAD_DIM)
        k = k.view(N, _APB_HEADS, _APB_HEAD_DIM).permute(1, 0, 2)
        v = v.view(N, _APB_HEADS, _APB_HEAD_DIM).permute(1, 0, 2)
        scores = torch.einsum("hqd,hkd->hqk", q, k) + bias
        o = torch.einsum("hqk,hkd->hqd", F.softmax(scores, dim=-1), v)
        o = o.permute(1, 0, 2).reshape(N, _APB_HEADS * _APB_HEAD_DIM)
        g = torch.sigmoid(F.linear(a, self._bw(i, pfx + "mha.linear_g.weight")))
        o = F.linear(o * g, self._bw(i, pfx + "mha.linear_o.weight"))
        s = s + o
        # single_transition (SwiGLU)
        tpfx = "single_transition."
        xn = F.layer_norm(s, (_C_S,), self._bw(i, tpfx + "layer_norm.weight"),
                          self._bw(i, tpfx + "layer_norm.bias"))
        t = F.silu(F.linear(xn, self._bw(i, tpfx + "swiglu.linear_a.weight"))) * \
            F.linear(xn, self._bw(i, tpfx + "swiglu.linear_b.weight"))
        s = s + F.linear(t, self._bw(i, tpfx + "linear_out.weight"))
        return s

    def forward(self, si_input, si_trunk, zij_trunk, repr_x_pred,
                max_atom_per_token_mask, use_zij_trunk_embedding=True):
        """Confidence forward -> dict of head logits (host fp32) + the confidence
        Pairformer (si_conf, zij_conf).

        Inputs are host fp32 tensors:
            si_input:  [N_tok, 449]
            si_trunk:  [N_tok, 384]
            zij_trunk: [N_tok, N_tok, 128]
            repr_x_pred: [N_tok, 3]   representative atom coords per token
            max_atom_per_token_mask: [N_tok * 23] broadcast of token_mask to atom slots
            use_zij_trunk_embedding: reference eval-mode flag (True -> keep zij_trunk)

        Returns:
            plddt_logits:                [N_atom, 50]
            experimentally_resolved_logits: [N_atom, 2]
            pae_logits:      [N_tok, N_tok, 64]
            pde_logits:      [N_tok, N_tok, 64]
            distogram_logits: [N_tok, N_tok, 64]
            si_conf:         [N_tok, 384]   (confidence Pairformer single, host-fp32)
            zij_conf:        [N_tok, N_tok, 128] (device z-path output)
        """
        import ttnn

        N = si_trunk.shape[0]

        # --- z-embedding (host fp32, AF3 Algorithm 31 lines 1-3) ---
        z = zij_trunk if use_zij_trunk_embedding else zij_trunk * 0.0
        z = (z
             + F.linear(si_input, self._g("pairformer_embedding.linear_i.weight")).unsqueeze(-2)
             + F.linear(si_input, self._g("pairformer_embedding.linear_j.weight")).unsqueeze(-3))
        dij = torch.sum((repr_x_pred[..., None, :] - repr_x_pred[..., None, :, :]) ** 2, dim=-1,
                        keepdim=True)  # [N, N, 1]
        oh = ((dij > self._squared_bins) & (dij < self._upper)).to(z.dtype)  # [N, N, no_bin]
        z = z + F.linear(oh, self._g("pairformer_embedding.linear_distance.weight"))

        # --- confidence Pairformer: device z-path + host-fp32 s-path ---
        to_dev = lambda x: ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=self.dev,
                                           dtype=ttnn.bfloat16)
        z_d = to_dev(z.unsqueeze(0))
        s = si_trunk.clone()
        zf = z
        for i, blk in enumerate(self.pf.blocks):
            u = blk.triangle_multiplication_start(z_d, None); z_d = ttnn.add_(z_d, u); ttnn.deallocate(u)
            u = blk.triangle_multiplication_end(z_d, None);   z_d = ttnn.add_(z_d, u); ttnn.deallocate(u)
            u = blk.triangle_attention_start(z_d, None);      z_d = ttnn.add_(z_d, u); ttnn.deallocate(u)
            u = blk.triangle_attention_end(z_d, None);        z_d = ttnn.add_(z_d, u); ttnn.deallocate(u)
            u = blk.transition_z(z_d);                        z_d = ttnn.add_(z_d, u); ttnn.deallocate(u)
            z_host = torch.Tensor(ttnn.to_torch(z_d)).float().reshape(N, N, _C_Z)
            s = self._host_s_block(s, z_host, i)
            zf = z_host
        s_single, zij_conf = s, zf

        # --- output heads (host fp32) ---
        # Distogram reads the TRUNK pair (reference: computed before the confidence
        # Pairformer), symmetrised as L(z) + L(z).T (no LayerNorm).
        dlog = F.linear(zij_trunk, self._g("distogram.linear.weight"))
        distogram_logits = dlog + dlog.transpose(-2, -3)

        pae_logits = F.linear(
            F.layer_norm(zij_conf, (_C_Z,)) * self._g("pae.layer_norm.weight") + self._bias("pae.layer_norm.bias"),
            self._g("pae.linear.weight"))

        plog = F.linear(
            F.layer_norm(zij_conf, (_C_Z,)) * self._g("pde.layer_norm.weight") + self._bias("pde.layer_norm.bias"),
            self._g("pde.linear.weight"))
        pde_logits = plog + plog.transpose(-2, -3)

        plddt_logits = self._atom_head(s_single, "plddt", max_atom_per_token_mask, 50)
        exp_resolved_logits = self._atom_head(s_single, "experimentally_resolved",
                                              max_atom_per_token_mask, 2)

        return {
            "plddt_logits": plddt_logits,
            "experimentally_resolved_logits": exp_resolved_logits,
            "pae_logits": pae_logits,
            "pde_logits": pde_logits,
            "distogram_logits": distogram_logits,
            "si_conf": s_single,
            "zij_conf": zij_conf,
        }

    def _atom_head(self, s_single, name, max_atom_per_token_mask, c_out):
        """pLDDT / experimentally_resolved: Linear(LN(s)) -> [N_tok, 23*c_out] -> reshape
        to [N_tok*23, c_out] -> masked_select to [N_atom, c_out]."""
        ln = F.layer_norm(s_single, (_C_S,)) * self._g(f"{name}.layer_norm.weight") \
            + self._bias(f"{name}.layer_norm.bias")
        logits = F.linear(ln, self._g(f"{name}.linear.weight"))  # [N_tok, 23*c_out]
        n_tok = s_single.shape[0]
        logits = logits.reshape(n_tok * _MAX_ATOMS_PER_TOKEN, c_out)
        return logits[max_atom_per_token_mask.bool()]  # [N_atom, c_out]
