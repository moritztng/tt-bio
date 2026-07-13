"""OpenFold3 confidence heads (AF3 Algorithm 31) on Tenstorrent.

Ports ``aux_heads`` (PAE / PDE / pLDDT / distogram / experimentally_resolved) to device,
reusing the gated 4-block confidence ``Pairformer`` (``aux_heads.pairformer_embedding.
pairformer_stack``) for the confidence trunk and the Protenix-v2 ``ConfidenceHead``
discipline: the z-embedding (``linear_i`` / ``linear_j`` outer-sum + ``linear_distance``
on binned squared representative-atom distances) and the five output heads (host
``LayerNorm`` + ``Linear``) run in fp32 on host, isolating the device confidence
Pairformer as the only bf16 stage.

The OF3 confidence Pairformer block layout is bit-identical to the trunk's
``pairformer_stack`` block (c_z=128, 4 tri-att heads, 16 attn-pair-bias heads,
c_hidden_pair_bias=24, c_s=384), so ``remap_pairformer_stack`` + ``Pairformer`` build it
with no new code. The output heads differ from Protenix's layout (OF3 pLDDT / resolved
are a flat ``[N_tok, 23 * c_out]`` linear -> ``masked_select`` to atoms; PDE / distogram
symmetrise as ``L(z) + L(z).T``; PAE is the plain ``L(z)``; distogram reads the trunk
pair, not the confidence-pairformer output), so they are ported here rather than reused.

Validated vs the real OF3 reference golden
(``~/of3_ref_out.pkl["intermediates"]["confidence_heads_real"]``, captured by
``scripts/of3_confidence_golden.py``); see ``tests/test_openfold3_confidence.py`` for
per-head PCC.
"""

import torch
import torch.nn.functional as F

from .tenstorrent import Pairformer, get_device
from .openfold3_weights import remap_pairformer_stack

# aux_heads.pairformer_embedding distance bins (config.model_config).
_MIN_BIN, _MAX_BIN, _NO_BIN, _INF = 3.25, 50.75, 39, 1e8
_MAX_ATOMS_PER_TOKEN = 23
# Confidence pairformer dims (config.model_config.heads.pairformer_embedding.pairformer).
_C_S, _C_Z, _C_S_INPUT = 384, 128, 449


class OF3ConfidenceHead:
    """OF3 AF3-family confidence heads on device.

    Args:
        aux_state_dict: ``aux_heads`` sub-state-dict (keys stripped of ``aux_heads.``).
        device: ttnn device.
        compute_kernel_config: HiFi4 + fp32 dest acc.
    """

    def __init__(self, aux_state_dict, device, compute_kernel_config):
        self._w = dict(aux_state_dict)
        self.dev = device
        self.compute_kernel_config = compute_kernel_config

        import re
        _pf_prefix = "pairformer_embedding.pairformer_stack"
        pf_sd = remap_pairformer_stack(self._w, prefix=_pf_prefix)
        _blk = re.compile(rf"^{re.escape(_pf_prefix)}\.blocks\.(\d+)\.")
        n_blocks = 1 + max(int(_blk.match(k).group(1)) for k in self._w if _blk.match(k))
        b0 = "pairformer_embedding.pairformer_stack.blocks.0.pair_stack."
        tri_att_n_heads = self._w[b0 + "tri_att_start.linear_z.weight"].shape[0]
        tri_att_head_dim = self._w[b0 + "tri_att_start.mha.linear_q.weight"].shape[0] // tri_att_n_heads
        apb0 = "pairformer_embedding.pairformer_stack.blocks.0.attn_pair_bias."
        att_n_heads = self._w[apb0 + "linear_z.weight"].shape[0]
        att_head_dim = self._w[apb0 + "mha.linear_q.weight"].shape[0] // att_n_heads
        self.pf = Pairformer(n_blocks, tri_att_head_dim, tri_att_n_heads,
                             att_head_dim, att_n_heads, True, pf_sd, compute_kernel_config)

        bins = torch.linspace(_MIN_BIN, _MAX_BIN, _NO_BIN, dtype=torch.float32)
        self._squared_bins = bins ** 2
        self._upper = torch.cat([self._squared_bins[1:], self._squared_bins.new_tensor([_INF])])

    def _g(self, k):
        return self._w[k].float()

    def _bias(self, k):
        return self._w[k].float() if k in self._w else 0.0

    def forward(self, si_input, si_trunk, zij_trunk, repr_x_pred,
                max_atom_per_token_mask, use_zij_trunk_embedding=True):
        """Confidence forward -> dict of head logits (host fp32).

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
            si_conf:         [N_tok, 384]   (confidence Pairformer single, for gating)
            zij_conf:        [N_tok, N_tok, 128]
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

        # --- confidence Pairformer (device bf16) ---
        to_dev = lambda x: ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=self.dev,
                                           dtype=ttnn.bfloat16)
        so, zo = self.pf(to_dev(si_trunk.unsqueeze(0)), to_dev(z.unsqueeze(0)))
        s_single = torch.Tensor(ttnn.to_torch(so)).float().reshape(N, _C_S)
        zf = torch.Tensor(ttnn.to_torch(zo)).float().reshape(N, N, _C_Z)
        ttnn.deallocate(so); ttnn.deallocate(zo)

        # --- output heads (host fp32) ---
        # Distogram reads the TRUNK pair (reference: computed before the confidence
        # Pairformer), symmetrised as L(z) + L(z).T (no LayerNorm).
        dlog = F.linear(zij_trunk, self._g("distogram.linear.weight"))
        distogram_logits = dlog + dlog.transpose(-2, -3)

        pae_logits = F.linear(
            F.layer_norm(zf, (_C_Z,)) * self._g("pae.layer_norm.weight") + self._bias("pae.layer_norm.bias"),
            self._g("pae.linear.weight"))

        plog = F.linear(
            F.layer_norm(zf, (_C_Z,)) * self._g("pde.layer_norm.weight") + self._bias("pde.layer_norm.bias"),
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
            "zij_conf": zf,
        }

    def _atom_head(self, s_single, name, max_atom_per_token_mask, c_out):
        """pLDDT / experimentally_resolved: Linear(LN(s)) -> [N_tok, 23*c_out] -> reshape
        to [N_tok*23, c_out] -> masked_select to [N_atom, c_out]."""
        ln = F.layer_norm(s_single, (_C_S,)) * self._g(f"{name}.layer_norm.weight") \
            + self._bias(f"{name}.layer_norm.bias")
        logits = F.linear(ln, self._g(f"{name}.linear.weight"))  # [N_tok, 23*c_out]
        n_tok = s_single.shape[0]
        logits = logits.reshape(n_tok * _MAX_ATOMS_PER_TOKEN, c_out)
        mask = max_atom_per_token_mask.bool()
        return logits[mask]  # [N_atom, c_out]
