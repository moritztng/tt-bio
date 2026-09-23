"""OpenFold3 device port for tt-bio.

Assembled from the shared ``tenstorrent`` primitives (the same Pairformer / MSA /
Triangle / DiT / AdaLN / AttentionPairBias building blocks Protenix-v2 and Boltz-2
reuse), driven by the OF3->tt-bio weight remap in ``tt_bio.openfold3_weights``.

OF3 is *not* a pure weight-remap onto ``protenix.Trunk``: the trunk hyperparameters
differ (OF3 c_z=128, no_heads_pair=4 vs Protenix-v2 c_z=256, no_heads_pair=8), so the
trunk is assembled from the shared primitives at OF3 dims rather than instantiated from
``protenix.py``. The DiffusionModule's core DiT/AtomTransformer dims (24 DiT blocks,
16 heads, head_dim=48, c_atom=128, c_atom_pair=16, NQ=32/NK=128/PAD_LEFT=48) match
Protenix-v2 and reuse its ``DiffusionModule`` via key remap.

Status (P6): the InputEmbedder *glue* leg (s_input -> s, z via the 5 weight-only
linears + outer-sum z) is PCC-gated on device (``tests/test_openfold3_input_embedder.py``).
The atom-encoder -> s_input leg, trunk assembly, template embedder, DiffusionModule and
confidence heads are landed incrementally in subsequent ticks, each PCC-gated vs the
golden in ``~/of3_ref_out.pkl`` exactly as the Pairformer / MSA legs were in P3-P5.
"""

from __future__ import annotations

import ttnn

from tt_bio.tenstorrent import Module, _dtype, device_dtype_override
from .eltwise_fusion import mask_add


class InputEmbedderGlue(Module):
    """OF3 ``InputEmbedderAllAtom`` glue: ``s_input -> (s, z)``.

    Implements the five weight-only linears (``linear_s``, ``linear_z_i``,
    ``linear_z_j``, ``linear_relpos``, ``linear_token_bonds``) plus the outer-sum pair
    representation

        z[i,j] = linear_z_i(s_input)[i] + linear_z_j(s_input)[j]
                 + linear_relpos(relpos)[i,j] + linear_token_bonds(token_bonds)[i,j]

    The atom-encoder leg (atom features -> ``s_input``) is gated separately; this module
    isolates the glue linears so the device linear precision is PCC-gated independently of
    the atom-transformer attention. All five linears are bias-free in the OF3 checkpoint.

    Args:
        state_dict: the ``input_embedder`` sub-dict (keys ``linear_*.weight``).
        compute_kernel_config: device compute kernel config (HiFi4 + fp32 dest acc).

    Inputs (device bf16):
        s_input:     [1, N_token, c_s_input=449]
        relpos:      [1, N_token, N_token, 139]  (OF3 ``relpos_complex`` feature)
        token_bonds: [1, N_token, N_token, 1]

    Outputs (device bf16):
        s: [1, N_token, c_s=384]
        z: [1, N_token, N_token, c_z=128]
    """

    def __init__(self, state_dict, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self.w_s = self.torch_to_tt("linear_s.weight")
        self.w_zi = self.torch_to_tt("linear_z_i.weight")
        self.w_zj = self.torch_to_tt("linear_z_j.weight")
        self.w_relpos = self.torch_to_tt("linear_relpos.weight")
        self.w_tb = self.torch_to_tt("linear_token_bonds.weight")

    def __call__(self, s_input, relpos, token_bonds):
        lin = self._lin
        s = lin(s_input, self.w_s)
        zi = lin(s_input, self.w_zi)
        zj = lin(s_input, self.w_zj)
        relpos_emb = lin(relpos, self.w_relpos)
        tb_emb = lin(token_bonds, self.w_tb)

        # Outer sum z[i,j] = zi[i] + zj[j]. ttnn add broadcasts a [1,N,1,c] operand over
        # dim -2 and a [1,1,N,c] operand over dim -3; seed with a zero [1,N,N,c] so both
        # single-dim broadcasts follow the same path as the pair-bias add in protenix.py.
        n = s_input.shape[-2]
        c_z = self.w_zj.shape[-1]
        z = ttnn.zeros((1, n, n, c_z), device=self.device, dtype=ttnn.bfloat16)
        z = ttnn.add(z, ttnn.unsqueeze(zi, -2))
        z = ttnn.add(z, ttnn.unsqueeze(zj, -3))
        z = ttnn.add(z, relpos_emb)
        z = ttnn.add(z, tb_emb)
        ttnn.deallocate(relpos_emb)
        ttnn.deallocate(tb_emb)
        return s, z


class AtomEncoderTokenHead(Module):
    """OF3 atom-encoder aggregation head: masked per-atom ``ql`` -> per-token ``ai``.

        ai = atom_to_token_mean @ relu(linear_q(ql * atom_mask))

    The input embedder's atom encoder ends here, and on the shipped path it ended on the
    host: ``openfold3_host_prep.run_input_atom_encoder`` brought ``ql`` back with a
    ``ttnn.to_torch`` and applied ``linear_q.0.weight`` in host float32, which is why that
    weight has no device gradient. ``OF3DiffusionModule._post_encoder`` performs the same
    two ops on its own ``linear_q.0.weight`` and keeps them on the card; this is that leg
    for the other encoder.

    Inputs (device):
        ql:                 [1, N_atom, c_atom=128]
        atom_mask_col:      [1, N_atom, 1]
        atom_to_token_mean: [1, N_token, N_atom]

    Output (device):
        ai: [1, N_token, c_s=384]
    """

    def __init__(self, state_dict, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self.w_q = self.torch_to_tt("0.weight")

    def __call__(self, ql, atom_mask_col, atom_to_token_mean):
        q = self._lin(ttnn.multiply(ql, atom_mask_col), self.w_q, activation="relu")
        ai = ttnn.matmul(atom_to_token_mean, q,
                         compute_kernel_config=self.compute_kernel_config)
        ttnn.deallocate(q)
        return ai


class RefAtomFeatureEmbedder(Module):
    """OF3 ``RefAtomFeatureEmbedder``: reference-conformer atom featurization.

    Single leg (-> ``cl``): five weight-only linears over per-atom reference features
    (``ref_pos`` 3->c, ``ref_charge`` arcsinh 1->c, ``ref_mask`` 1->c, ``ref_element``
    E->c, ``ref_atom_name_chars`` flattened 4*64->c) summed into the per-atom
    conditioning ``cl`` [N_atom, c_atom].

    Pair leg (-> ``plm``): three weight-only linears over the precomputed block inputs
    ``dlm`` (per-block q-k offsets [N_blk, N_q, N_k, 3]), ``vlm`` (same-residue+valid
    mask [N_blk, N_q, N_k, 1]) and ``inv_sq_dists`` ([N_blk, N_q, N_k, 1]):

        plm = linear_ref_offset(dlm)*vlm + linear_inv_sq_dists(inv_sq_dists)*vlm
              + linear_valid_mask(vlm)*vlm

    The block construction (``convert_single_rep_to_blocks`` + ``get_block_indices``) is
    mask-derived gather captured in the golden (``dlm``/``vlm``/``inv_sq_dists``), so this
    module isolates the device linear precision from the blocking logic -- the same
    discipline as the glue's golden relpos. All eight linears are bias-free in the OF3
    checkpoint.

    Inputs (device bf16):
        ref_pos, ref_charge, ref_mask, ref_element, ref_atom_chars: per-atom features
            with the shapes above (ref_charge/ref_mask unsqueezed to [N,1]; ref_atom_chars
            pre-flattened to [N, 256]; ref_charge passed through arcsinh on host).
        dlm:          [1, N_blk, N_q, N_k, 3]
        vlm:          [1, N_blk, N_q, N_k, 1]
        inv_sq_dists: [1, N_blk, N_q, N_k, 1]

    Outputs (device bf16):
        cl:  [1, N_atom, c_atom]
        plm: [1, N_blk, N_q, N_k, c_atom_pair]
    """

    def __init__(self, state_dict, compute_kernel_config):
        super().__init__(state_dict, compute_kernel_config)
        self.w_ref_pos = self.torch_to_tt("linear_ref_pos.weight")
        self.w_ref_charge = self.torch_to_tt("linear_ref_charge.weight")
        self.w_ref_mask = self.torch_to_tt("linear_ref_mask.weight")
        self.w_ref_element = self.torch_to_tt("linear_ref_element.weight")
        self.w_ref_chars = self.torch_to_tt("linear_ref_atom_chars.weight")
        self.w_ref_offset = self.torch_to_tt("linear_ref_offset.weight")
        self.w_inv_sq = self.torch_to_tt("linear_inv_sq_dists.weight")
        self.w_valid = self.torch_to_tt("linear_valid_mask.weight")

    def __call__(self, ref_pos, ref_charge, ref_mask, ref_element, ref_atom_chars,
                 dlm, vlm, inv_sq_dists):
        lin = self._lin
        # Single leg -> cl [1, N_atom, c_atom]
        cl = lin(ref_pos, self.w_ref_pos)
        cl = ttnn.add(cl, lin(ref_charge, self.w_ref_charge))
        cl = ttnn.add(cl, lin(ref_mask, self.w_ref_mask))
        cl = ttnn.add(cl, lin(ref_element, self.w_ref_element))
        cl = ttnn.add(cl, lin(ref_atom_chars, self.w_ref_chars))

        # Pair leg -> plm [1, N_blk, N_q, N_k, c_atom_pair]
        # Three legs all masked by vlm then summed: two of the multiplies fold into the
        # adds that consume them, so five eltwise ops become three.
        off = ttnn.multiply(lin(dlm, self.w_ref_offset), vlm)
        isd = lin(inv_sq_dists, self.w_inv_sq)
        vm = lin(vlm, self.w_valid)
        acc = mask_add(off, isd, vlm)
        ttnn.deallocate(off)
        ttnn.deallocate(isd)
        plm = mask_add(acc, vm, vlm)
        ttnn.deallocate(acc)
        ttnn.deallocate(vm)
        return cl, plm


class InputAtomEncoder(Module):
    """OF3 input embedder's ``AtomAttentionEncoder`` on the card: ref-atom features -> ``s_input``.

        cl, plm = RefAtomFeatureEmbedder(ref features)
        plm     = AtomPairUpdate(cl, plm)
        ql      = OF3AtomTransformer(cl, cl, plm)
        ai      = AtomEncoderTokenHead(ql)
        s_input = concat(ai, restype, profile, deletion_mean)

    ``openfold3_host_prep.run_input_atom_encoder`` is the same leg with the embedding, the pair
    update and the head on the host, which is what inference runs. This one keeps every weight
    on the card so a taped forward can differentiate them; the training step builds it, at fp32
    (``device_dtype_override``) so its s_input is as near exact as the host leg's.
    ``state_dict`` is the ``input_embedder.atom_attn_enc`` sub-dict.

    Inputs (device): the eight tensors of ``ref_atom_device_inputs`` padded to NP, at this
    module's dtype; the diffusion module's own blocking tensors from ``build_dm_device_aux``
    (``amc_d``, ``kidx_tt``, ``valid_d``, ``mb_d``, ``pm_d``, ``amc_na_d``, ``mean_d``), which
    describe the same atom windows, except ``atom_to_token_mean`` [1, N_token, N_atom], which
    should come at this module's dtype (its 1/n entries do not survive bf16);
    ``token_feats`` [1, N_token, 65] bf16. Returns bf16 s_input, the trunk's dtype.
    """

    def __init__(self, state_dict, compute_kernel_config):
        super().__init__({}, compute_kernel_config)
        self._act_dtype = _dtype(ttnn.bfloat16)
        from .openfold3_atom_transformer import OF3AtomTransformer
        from .openfold3_diffusion_module import AtomPairUpdate
        from .openfold3_weights import _sub

        self.ref_embed = RefAtomFeatureEmbedder(
            _sub(state_dict, "ref_atom_feature_embedder"), compute_kernel_config)
        self.pair_update = AtomPairUpdate(state_dict, compute_kernel_config)
        self.at = OF3AtomTransformer(_sub(state_dict, "atom_transformer"), compute_kernel_config)
        self.at.materialize_device_weights()
        self.head = AtomEncoderTokenHead(_sub(state_dict, "linear_q"), compute_kernel_config)

    def __call__(self, ref_in, atom_mask_col, key_block_idxs_tt, valid_mask, mask_bias,
                 pair_mask, atom_mask_col_na, atom_to_token_mean, token_feats, n_atom, NP, nb):
        # Run at the dtype the module was built at: `Module._lin` resolves its output dtype at
        # call time, and a bf16 output meeting an fp32 [.., 1] mask is D259's wrong multiply.
        # The 0/1 masks come at bf16 and are cast for the same reason.
        c = lambda t: t if t.dtype == self._act_dtype else ttnn.typecast(t, self._act_dtype)  # noqa: E731
        with device_dtype_override(self._act_dtype):
            cl, plm = self.ref_embed(*ref_in)
            plm = self.pair_update(cl, plm, key_block_idxs_tt, c(valid_mask), c(pair_mask),
                                   NP, nb)
            ql = self.at(cl, cl, plm, c(atom_mask_col), key_block_idxs_tt, c(valid_mask),
                         c(mask_bias), n_atom, NP, nb)
            ai = self.head(ql, c(atom_mask_col_na), c(atom_to_token_mean))
        if ai.dtype != token_feats.dtype:
            ai = ttnn.typecast(ai, token_feats.dtype)
        return ttnn.concat([ai, token_feats], dim=-1)
