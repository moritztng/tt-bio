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
import ttnn

from .tenstorrent import Pairformer, accurate_softmax_site
from .openfold3_weights import remap_pairformer_stack

# aux_heads.pairformer_embedding distance bins (config.model_config).
_MIN_BIN, _MAX_BIN, _NO_BIN, _INF = 3.25, 50.75, 39, 1e8
_MAX_ATOMS_PER_TOKEN = 23
# Confidence pairformer dims (config.model_config.heads.pairformer_embedding.pairformer
# + attn_pair_bias: c_s=384, c_z=128, c_s_input=449, 16 attn-pair-bias heads, head_dim 24).
_C_S, _C_Z, _C_S_INPUT = 384, 128, 449
_APB_HEADS, _APB_HEAD_DIM = 16, 24
_BLK = "pairformer_embedding.pairformer_stack.blocks.%d."
# Which s-track the host-tensor entry point runs. "host" is torch fp32 and is
# what inference gates against; "device" is the differentiable one. Training
# does not read this -- it calls forward_device directly.
_S_PATH_DEFAULT = "host"


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
                             att_head_dim, att_n_heads, True, pf_sd, compute_kernel_config,
                             scale_pair_bias=True, fp32_softmax=True,
                             accurate_softmax=accurate_softmax_site("openfold3.confidence"),
                             s_fp32_residual=True)
        self.n_blocks = n_blocks

        bins = torch.linspace(_MIN_BIN, _MAX_BIN, _NO_BIN, dtype=torch.float32)
        self._squared_bins = bins ** 2
        self._upper = torch.cat([self._squared_bins[1:], self._squared_bins.new_tensor([_INF])])

    # The host path reads every weight through these three, so setting `_dtype` to
    # torch.float64 runs it in float64 WITHOUT a second implementation. PROTOCOL SS3c
    # wants a float64 reference that is not another approximation; the cheapest way to
    # be sure of that is for the reference and the shipped path to be one function.
    _dtype = torch.float32

    def _g(self, k):
        return self._w[k].to(self._dtype)

    def _bias(self, k):
        return self._w[k].to(self._dtype) if k in self._w else 0.0

    def _bw(self, i, name):
        return self._w[(_BLK % i) + name].to(self._dtype)

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


    # ------------------------------------------------------------------
    # Device path. Everything from the trunk outputs to the head logits is a
    # ttnn verb, so a tape following those verbs carries a confidence-loss
    # gradient back into the trunk. The host path below is the same arithmetic
    # with the s-track in torch fp32; it is kept for inference and it SEVERS
    # the graph, which is why training does not get to choose it.
    # ------------------------------------------------------------------

    def _wd(self, key, transpose=True, dtype=None):
        """Upload one weight once (TILE), cached. ``transpose`` for a Linear."""
        dtype = dtype or ttnn.bfloat16
        cache = self.__dict__.setdefault("_wd_cache", {})
        v = cache.get((key, transpose, dtype))
        if v is None:
            if key not in self._w:
                return None
            w = self._w[key].float()
            v = ttnn.from_torch(w.t().contiguous() if transpose else w,
                                layout=ttnn.TILE_LAYOUT, device=self.dev, dtype=dtype)
            cache[(key, transpose, dtype)] = v
        return v

    def _lin(self, x, key):
        w = self._wd(key, dtype=x.dtype if x.dtype == ttnn.float32 else None)
        return ttnn.linear(x, w, compute_kernel_config=self.compute_kernel_config)

    def _ln(self, x, prefix, fp32=False):
        """One head LayerNorm. ``fp32`` casts the input and the gain first.

        The two atom heads need it and the pair heads do not, and the reason is the
        magnitude of what they normalise. ``si_trunk`` reaches the confidence head raw,
        at absmax 2.28e5 with a per-row std of 1.6e4, so the variance a LayerNorm sums
        is of order 3e8; in bf16 that sum keeps 8 mantissa bits and the reciprocal
        square root it feeds is what the pLDDT logits ARE. The pair track arrives at
        absmax 1.1e3 and has no such problem. Measured: bf16 throughout puts
        plddt_logits at 2.44e-01 relative L2 against the host fp32 path, over the
        PROTOCOL SS3d 5.0e-02 per-tensor bar, while the pair heads read 6.2e-03 (pae)
        and 3.3e-03 (distogram) on the same run.
        """
        dt = ttnn.float32 if fp32 else None
        if fp32 and x.dtype != ttnn.float32:
            x = ttnn.typecast(x, ttnn.float32)
        return ttnn.layer_norm(x, weight=self._wd(prefix + ".layer_norm.weight", False, dt),
                               bias=self._wd(prefix + ".layer_norm.bias", False, dt),
                               epsilon=1e-5, compute_kernel_config=self.compute_kernel_config)

    def distance_onehot(self, repr_x_pred):
        """The AF3 Algorithm 31 line-3 distance one-hot, host -> device.

        It is built on the host and uploaded as a CONSTANT, and that is a property of
        the loss rather than a shortcut: the confidence head is trained on a rolled-out
        structure that upstream detaches, so no gradient flows to ``repr_x_pred``. The
        one-hot is exact in bf16 (it is 0 and 1), so the upload costs no accuracy.
        """
        dij = torch.sum((repr_x_pred[..., None, :] - repr_x_pred[..., None, :, :]) ** 2,
                        dim=-1, keepdim=True)
        oh = ((dij > self._squared_bins) & (dij < self._upper)).float()
        return ttnn.from_torch(oh.unsqueeze(0), layout=ttnn.TILE_LAYOUT, device=self.dev,
                               dtype=ttnn.bfloat16)

    def forward_device(self, si_input_d, si_trunk_d, zij_trunk_d, oh_d,
                       use_zij_trunk_embedding=True):
        """Confidence forward with every tensor on device. The training entry point.

        Shapes are batched-by-one so the shipped ``PairformerLayer`` takes them unchanged:
        ``si_input_d`` [1, N, 449], ``si_trunk_d`` [1, N, 384], ``zij_trunk_d``
        [1, N, N, 128], ``oh_d`` [1, N, N, 39] from :meth:`distance_onehot`.

        Two outputs differ in LAYOUT from the host path, and neither loses information.
        ``plddt_logits`` and ``experimentally_resolved_logits`` come back as
        [1, N_tok, 23 * c_out] rather than gathered to [N_atom, c_out]: the gather is
        ``masked_select`` on a fixed per-batch mask, which on device is either a
        one-hot matmul the size of the token axis squared or a row-major round trip,
        and it buys nothing -- a loss that reads the padded layout with the atom mask
        as its weight computes the same number. :meth:`forward` does the gather for
        inference, where the [N_atom, c_out] shape is what the caller wants.
        """
        N = int(zij_trunk_d.shape[-2])
        pe = "pairformer_embedding."
        z = zij_trunk_d if use_zij_trunk_embedding else ttnn.multiply(zij_trunk_d, 0.0)
        # Out-of-place: zij_trunk_d is the trunk's own output and the distogram head
        # reads it again below. An in-place add here would corrupt both.
        z = ttnn.add(z, ttnn.reshape(self._lin(si_input_d, pe + "linear_i.weight"),
                                     (1, N, 1, _C_Z)))
        z = ttnn.add(z, ttnn.reshape(self._lin(si_input_d, pe + "linear_j.weight"),
                                     (1, 1, N, _C_Z)))
        z = ttnn.add(z, self._lin(oh_d, pe + "linear_distance.weight"))

        s, z = self.pf(si_trunk_d, z)

        dlog = self._lin(zij_trunk_d, "distogram.linear.weight")
        distogram_logits = ttnn.add(dlog, ttnn.permute(dlog, (0, 2, 1, 3)))
        pae_logits = self._lin(self._ln(z, "pae"), "pae.linear.weight")
        plog = self._lin(self._ln(z, "pde"), "pde.linear.weight")
        pde_logits = ttnn.add(plog, ttnn.permute(plog, (0, 2, 1, 3)))
        plddt_logits = self._lin(self._ln(s, "plddt", fp32=True), "plddt.linear.weight")
        resolved_logits = self._lin(self._ln(s, "experimentally_resolved", fp32=True),
                                    "experimentally_resolved.linear.weight")
        return {
            "plddt_logits": plddt_logits,
            "experimentally_resolved_logits": resolved_logits,
            "pae_logits": pae_logits,
            "pde_logits": pde_logits,
            "distogram_logits": distogram_logits,
            "si_conf": s,
            "zij_conf": z,
        }

    def forward(self, si_input, si_trunk, zij_trunk, repr_x_pred,
                max_atom_per_token_mask, use_zij_trunk_embedding=True,
                s_path=None, dtype=None):
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
        N = si_trunk.shape[0]
        self._dtype = dtype or torch.float32
        if (s_path or _S_PATH_DEFAULT) == "device":
            to_dev = lambda x, dt=ttnn.bfloat16: ttnn.from_torch(
                x.float().unsqueeze(0), layout=ttnn.TILE_LAYOUT, device=self.dev, dtype=dt)
            out = self.forward_device(
                to_dev(si_input), to_dev(si_trunk, ttnn.float32), to_dev(zij_trunk),
                self.distance_onehot(repr_x_pred),
                use_zij_trunk_embedding=use_zij_trunk_embedding)
            host = {k: torch.Tensor(ttnn.to_torch(v)).float() for k, v in out.items()}
            m = max_atom_per_token_mask.bool()
            for name, c_out in (("plddt_logits", 50), ("experimentally_resolved_logits", 2)):
                host[name] = host[name].reshape(N * _MAX_ATOMS_PER_TOKEN, c_out)[m]
            for name in ("pae_logits", "pde_logits", "distogram_logits", "zij_conf"):
                host[name] = host[name].reshape(N, N, -1)
            host["si_conf"] = host["si_conf"].reshape(N, _C_S)
            return host

        # --- z-embedding (host, AF3 Algorithm 31 lines 1-3) ---
        si_input, si_trunk = si_input.to(self._dtype), si_trunk.to(self._dtype)
        zij_trunk, repr_x_pred = zij_trunk.to(self._dtype), repr_x_pred.to(self._dtype)
        z = zij_trunk if use_zij_trunk_embedding else zij_trunk * 0.0
        z = (z
             + F.linear(si_input, self._g("pairformer_embedding.linear_i.weight")).unsqueeze(-2)
             + F.linear(si_input, self._g("pairformer_embedding.linear_j.weight")).unsqueeze(-3))
        dij = torch.sum((repr_x_pred[..., None, :] - repr_x_pred[..., None, :, :]) ** 2, dim=-1,
                        keepdim=True)  # [N, N, 1]
        oh = ((dij > self._squared_bins.to(dij.dtype)) &
              (dij < self._upper.to(dij.dtype))).to(z.dtype)  # [N, N, no_bin]
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
