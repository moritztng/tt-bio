"""OpenDDE StructuralTokenExpander on Tenstorrent.

OpenDDE (Aureka AI Research) is an AF3-family co-folding model whose entire
trunk / MSA / diffusion / confidence graph is Protenix-v2's, already ported in
``tt_bio.tenstorrent`` + ``tt_bio.protenix``. Its one novel compute block is
``StructuralTokenExpander``: it expands the residue-level trunk (``s_inputs``,
``s``, ``z``) onto the structural-token axis, adding role conditioning and
same-residue pair structure, before diffusion. The rest of the pipeline then
runs unchanged on the structural-token axis (the ttnn ops are axis-agnostic).

This module ports that one block; assembly reuses the Protenix-v2 stack verbatim
(see docs/opendde-port.md). The integer index gathers (parent, prev/next-parent
adjacency, role-pair-type maps) are precomputed host-side; only the split-MLP,
the 49 role-pair pair projections, and the bias adds run on device.
"""
import torch
import ttnn

from .protenix import _KeyedWeights
from .tenstorrent import get_device, CORE_GRID_MAIN

# opendde/data/tokenizer.py
STRUCTURAL_TOKEN_ROLES = {
    "atom": 0, "protein_bb": 1, "protein_sc": 2,
    "dna_bb": 3, "dna_base": 4, "rna_bb": 5, "rna_base": 6,
}
_BACKBONE = (STRUCTURAL_TOKEN_ROLES["protein_bb"],
             STRUCTURAL_TOKEN_ROLES["dna_bb"],
             STRUCTURAL_TOKEN_ROLES["rna_bb"])
_SIDECHAIN = STRUCTURAL_TOKEN_ROLES["protein_sc"]
_BASE = (STRUCTURAL_TOKEN_ROLES["dna_base"], STRUCTURAL_TOKEN_ROLES["rna_base"])


class StructuralTokenExpander(_KeyedWeights):
    """ttnn port of OpenDDE's residue->structural-token expander (opendde_v1:
    pair_projection_mode="full", 49 role-pair projections, chunked).

    Takes host-side residue-trunk tensors + the integer input-feature dict, and
    returns the expanded structural-token tensors as resident ttnn tensors:
    ``(s_inputs_struct, s_struct, z_struct, structural_pair_attn_bias)``.
    """

    def __init__(self, state_dict, compute_kernel_config, *, c_s=384, c_z=384,
                 c_s_inputs=449, n_roles=7, pair_chunk_size=128):
        self._w = {k: v for k, v in state_dict.items()}
        self.compute_kernel_config = compute_kernel_config
        self.c_s, self.c_z, self.c_s_inputs = c_s, c_z, c_s_inputs
        self.n_roles = n_roles
        self.pair_chunk_size = pair_chunk_size

    # --- host-side integer/mask features for a block of rows vs all columns.
    # Mirrors OpenDDE _build_structural_pair_context + _for_rows exactly; only
    # the terms consumed downstream (pair-init bias + attn bias) are kept. ---
    def _pair_features_rows(self, ifd, role, parent, row_index):
        Ns = role.shape[0]
        asym = ifd["asym_id"].long().index_select(0, parent)
        is_bb = (role == _BACKBONE[0]) | (role == _BACKBONE[1]) | (role == _BACKBONE[2])
        is_sc = role == _SIDECHAIN
        is_base = (role == _BASE[0]) | (role == _BASE[1])
        prev_parent = ifd.get("prev_parent_residue_idx")
        next_parent = ifd.get("next_parent_residue_idx")
        prev_parent = parent.new_full((Ns,), -1) if prev_parent is None else prev_parent.long()
        next_parent = parent.new_full((Ns,), -1) if next_parent is None else next_parent.long()

        ri = row_index
        rp = parent.index_select(0, ri)
        ra = asym.index_select(0, ri)
        r_bb = is_bb.index_select(0, ri)
        r_sc = is_sc.index_select(0, ri)
        r_base = is_base.index_select(0, ri)
        r_prev = prev_parent.index_select(0, ri)
        r_next = next_parent.index_select(0, ri)

        same_parent = rp[:, None] == parent[None, :]
        same_chain = ra[:, None] == asym[None, :]
        same_twin = same_parent & (
            (r_bb[:, None] & (is_sc[None, :] | is_base[None, :]))
            | (is_bb[None, :] & (r_sc[:, None] | r_base[:, None]))
        )
        prev_bb = r_bb[:, None] & is_bb[None, :] & same_chain & (r_prev[:, None] == parent[None, :])
        next_bb = r_bb[:, None] & is_bb[None, :] & same_chain & (r_next[:, None] == parent[None, :])

        clen = ri.numel()
        rpt = torch.full((clen, Ns), 7, dtype=torch.long)
        rpt[r_bb[:, None] & is_bb[None, :]] = 0
        rpt[r_bb[:, None] & is_sc[None, :]] = 1
        rpt[r_sc[:, None] & is_bb[None, :]] = 2
        rpt[r_sc[:, None] & is_sc[None, :]] = 3
        rpt[r_bb[:, None] & is_base[None, :]] = 4
        rpt[r_base[:, None] & is_bb[None, :]] = 5
        rpt[r_base[:, None] & is_base[None, :]] = 6
        return {
            "same_parent_residue": same_parent, "same_residue_twin": same_twin,
            "prev_bb_chain": prev_bb, "next_bb_chain": next_bb, "role_pair_type": rpt,
        }

    def _emb(self, name, idx):
        """Host gather of an embedding table (idx-shaped -> +last dim)."""
        w = self._w[name]
        return w.index_select(0, idx.reshape(-1)).reshape(*idx.shape, w.shape[-1])

    def _pair_init_bias(self, pf):
        """Sum of the five additive pair-init embeddings (host gather); float32."""
        b = self._emb("same_parent_embedding.weight", pf["same_parent_residue"].long())
        b = b + self._emb("same_residue_twin_embedding.weight", pf["same_residue_twin"].long())
        b = b + self._emb("prev_bb_chain_embedding.weight", pf["prev_bb_chain"].long())
        b = b + self._emb("next_bb_chain_embedding.weight", pf["next_bb_chain"].long())
        b = b + self._emb("role_pair_type_embedding.weight", pf["role_pair_type"])
        return b

    def _attn_bias(self, pf):
        """Scalar-weighted mask sum + role-pair-type bias -> ttnn (clen, Ns).
        Mask scaling by the (scalar) learned weights is host-side; the additive
        assembly runs on device."""
        w = self._w
        rpt = pf["role_pair_type"]
        role_pair_bias = w["attn_bias_role_pair_type"].index_select(0, rpt.reshape(-1)).reshape(rpt.shape)
        terms = [
            pf["same_parent_residue"].float() * float(w["attn_bias_same_parent"]),
            pf["same_residue_twin"].float() * float(w["attn_bias_same_residue_twin"]),
            pf["prev_bb_chain"].float() * float(w["attn_bias_prev_bb_chain"]),
            pf["next_bb_chain"].float() * float(w["attn_bias_next_bb_chain"]),
            role_pair_bias,
        ]
        ab = self._up(terms[0])
        for t in terms[1:]:
            ab = ttnn.add(ab, self._up(t))
        return ab

    def _pair_project_full(self, z_chunk_h, role, row_index):
        """delta[a,b] = W[role[a]*n+role[b]] @ z[a,b], full 49-projection mode.
        Rows are grouped by role-pair (host permute), each group is one device
        matmul, then scattered back via a device gather -- numerically identical
        to OpenDDE's per-(role_i,role_j) masked projection, reordered."""
        clen = row_index.numel()
        Ns = role.shape[0]
        C = self.c_z
        flat = z_chunk_h.reshape(clen * Ns, C)
        row_role = role.index_select(0, row_index)
        role_i = row_role[:, None].expand(clen, Ns).reshape(-1)
        role_j = role[None, :].expand(clen, Ns).reshape(-1)
        pidx = role_i * self.n_roles + role_j

        perm = torch.argsort(pidx, stable=True)
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(perm.numel())
        flat_sorted = flat.index_select(0, perm).contiguous()
        uniq, counts = torch.unique_consecutive(pidx.index_select(0, perm), return_counts=True)

        pieces = []
        off = 0
        for g, c in zip(uniq.tolist(), counts.tolist()):
            seg = self._up(flat_sorted[off:off + c].contiguous())
            out = self._lin(seg, "pair_block_proj.%d.weight" % g)
            pieces.append(ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT))
            off += c
        sorted_delta = pieces[0] if len(pieces) == 1 else ttnn.concat(pieces, dim=0)

        inv_idx = ttnn.from_torch(inv.reshape(1, -1).to(torch.int32),
                                  layout=ttnn.ROW_MAJOR_LAYOUT, device=get_device(),
                                  dtype=ttnn.uint32)
        flat_delta = ttnn.embedding(inv_idx, sorted_delta, layout=ttnn.ROW_MAJOR_LAYOUT,
                                    memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.to_layout(ttnn.reshape(flat_delta, (clen, Ns, C)), ttnn.TILE_LAYOUT)

    def __call__(self, ifd, s_inputs_res, s_res, z_res):
        parent = ifd["parent_residue_idx"].long()
        role = ifd["subtoken_role_id"].long()

        # --- single: gather parent rep (host) + role embedding, add on device ---
        s_inputs_struct = ttnn.add(
            self._up(s_inputs_res.index_select(0, parent).contiguous()),
            self._up(self._emb("single_input_role_embedding.weight", role)),
        )
        s_parent = self._up(s_res.index_select(0, parent).contiguous())
        mlp = self._ln(s_parent, "single_split_mlp.0.weight", "single_split_mlp.0.bias")
        mlp = self._lin(mlp, "single_split_mlp.1.weight")
        mlp = ttnn.silu(mlp)
        mlp = self._lin(mlp, "single_split_mlp.3.weight")
        s_struct = ttnn.add(ttnn.add(s_parent, mlp),
                            self._up(self._emb("single_role_embedding.weight", role)))

        # --- pair: chunked over rows (opendde_v1 pair_chunk_size) ---
        Ns = role.shape[0]
        chunk = min(self.pair_chunk_size or Ns, Ns)
        z_chunks, ab_chunks = [], []
        for start in range(0, Ns, chunk):
            end = min(start + chunk, Ns)
            row_index = torch.arange(start, end)
            pf = self._pair_features_rows(ifd, role, parent, row_index)
            row_parent = parent.index_select(0, row_index)
            z_chunk_h = z_res.index_select(0, row_parent).index_select(1, parent).contiguous()
            z_dev = self._up(z_chunk_h)
            z_dev = ttnn.add(z_dev, self._pair_project_full(z_chunk_h, role, row_index))
            z_dev = ttnn.add(z_dev, self._up(self._pair_init_bias(pf)))
            z_chunks.append(z_dev)
            ab_chunks.append(self._attn_bias(pf))
        z_struct = z_chunks[0] if len(z_chunks) == 1 else ttnn.concat(z_chunks, dim=-3)
        attn_bias = ab_chunks[0] if len(ab_chunks) == 1 else ttnn.concat(ab_chunks, dim=0)
        return s_inputs_struct, s_struct, z_struct, attn_bias


# ---------------------------------------------------------------------------
# Pipeline assembly + real-weight load (docs/opendde-port.md steps 2-3).
#
# OpenDDE's compute graph = Protenix-v2's trunk/MSA/template/diffusion/confidence
# (byte-identical checkpoint key names, verified 2026-07-12 against protenix-v2.pt:
# 0 keys missing) + this module's novel StructuralTokenExpander + a 4-block
# structural-token refiner (a reused PairformerStack). So the real-weight "remap"
# is mostly a routing split; the shared subtree feeds the existing Protenix stack
# unchanged, the expander keys match 1:1 under a prefix strip, and the refiner
# reuses the Protenix pairformer-block remap.
# ---------------------------------------------------------------------------

OPENDDE_REPO = "aurekaresearch/OpenDDE"

# Measured from opendde.pt (opendde_v1, 656M; config/model_base.py + weight shapes,
# 2026-07-12). NOTE these correct the earlier "dims match Protenix-v2 exactly" note:
# OpenDDE's pair channel is c_z=384 (not the tt-bio Protenix-v2 checkpoint's 256) and
# its triangle attention has 12 heads (not 8). c_s/c_s_inputs/MSA-depth do match.
OPENDDE_CONFIG = dict(
    c_s=384, c_z=384, c_s_inputs=449, n_roles=7, pair_chunk_size=128,
    pairformer_blocks=48, pairformer_tri_heads=12, pairformer_att_heads=16,
    msa_blocks=4,
    refiner_blocks=4, refiner_tri_heads=12, refiner_att_heads=8,
)


def load_opendde_checkpoint(path=None, *, abag=False):
    """Load an OpenDDE checkpoint to a flat ``{name: tensor}`` state_dict (``module.``
    prefix stripped, untrusted weights read with ``weights_only=True``). ``path=None``
    fetches from HF ``aurekaresearch/OpenDDE`` (``opendde.pt`` general, or
    ``opendde_abag.pt`` for the antibody-antigen checkpoint when ``abag=True``)."""
    import torch
    if path is None:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(OPENDDE_REPO, "opendde_abag.pt" if abag else "opendde.pt")
    ck = torch.load(path, map_location="cpu", weights_only=True)
    ck = ck.get("model", ck)
    return {k[len("module."):] if k.startswith("module.") else k: v for k, v in ck.items()}


def route_opendde_weights(state_dict):
    """Split an OpenDDE state_dict into the three subtrees the tt-bio assembly consumes,
    asserting full coverage (every checkpoint key is routed exactly once, no leftovers):

      - ``expander`` : ``structural_token_expander.*`` -> identity under the prefix strip
        (the names match ``StructuralTokenExpander``'s consumed keys 1:1).
      - ``refiner``  : ``structural_token_refiner.*`` -> N x ``remap_pairformer_block``
        (the reused Protenix-v2 pairformer-block remap; the refiner IS a PairformerStack).
      - ``shared``   : everything else -> the Protenix-v2-family graph, keys byte-identical
        to protenix-v2.pt, so it feeds the existing ``protenix.Protenix`` stack unchanged.

    Returns ``dict(expander=..., refiner=..., refiner_blocks=int, shared=...)``.
    """
    import re
    import tt_bio.protenix_weights as PW
    EP, RP = "structural_token_expander.", "structural_token_refiner."
    exp = {k[len(EP):]: v for k, v in state_dict.items() if k.startswith(EP)}
    ref_raw = {k: v for k, v in state_dict.items() if k.startswith(RP)}
    shared = {k: v for k, v in state_dict.items()
              if not k.startswith(EP) and not k.startswith(RP)}
    assert len(exp) + len(ref_raw) + len(shared) == len(state_dict), "routing dropped keys"

    nb = 1 + max(int(re.search(r"blocks\.(\d+)\.", k).group(1)) for k in ref_raw if "blocks." in k)
    refiner = {}
    for i in range(nb):
        pfx = f"{RP}blocks.{i}."
        blk = {k[len(pfx):]: v for k, v in ref_raw.items() if k.startswith(pfx)}
        for k, v in PW.remap_pairformer_block(blk).items():
            refiner[f"layers.{i}.{k}"] = v
    return dict(expander=exp, refiner=refiner, refiner_blocks=nb, shared=shared)


class OpenDDE:
    """OpenDDE co-folding on Tenstorrent: the Protenix-v2 trunk/diffusion/confidence stack
    (reused verbatim, ``tt_bio.protenix``) + the novel :class:`StructuralTokenExpander` +
    a 4-block structural-token refiner (a reused ``Pairformer``), on the structural-token
    axis. Ships co-folding only (no design/affinity) -- see docs/opendde-port.md.

    Assembly status (2026-07-12): real-weight routing and the novel expander->refiner seam
    are wired and run finite on-device with the REAL checkpoint
    (``scripts/opendde_assembly_verify.py``). The full residue->structure fold is gated on
    three still-pending prerequisites -- see :meth:`fold`."""

    def __init__(self, state_dict, compute_kernel_config, device=None):
        from .tenstorrent import get_device, Pairformer
        from .protenix import Protenix
        self.dev = device or get_device()
        self.compute_kernel_config = compute_kernel_config
        C = OPENDDE_CONFIG
        routed = route_opendde_weights(state_dict)
        self._shared = routed["shared"]         # Protenix-v2-family graph (for step-2 trunk/diffusion)
        # Shared Protenix-v2-family stack (input embedder, trunk, diffusion, confidence),
        # built at OpenDDE's c_z=384 (Trunk.__init__(c_z=...), docs/opendde-port.md item 2).
        # Reused verbatim -- no duplicated orchestration class.
        self._protenix = Protenix(
            self._shared, compute_kernel_config, self.dev, c_z=C["c_z"], msa_update_first=True)
        self.expander = StructuralTokenExpander(
            routed["expander"], compute_kernel_config, c_s=C["c_s"], c_z=C["c_z"],
            c_s_inputs=C["c_s_inputs"], n_roles=C["n_roles"], pair_chunk_size=C["pair_chunk_size"])
        self.refiner = Pairformer(
            routed["refiner_blocks"], C["c_z"] // C["refiner_tri_heads"], C["refiner_tri_heads"],
            C["c_s"] // C["refiner_att_heads"], C["refiner_att_heads"], True,
            routed["refiner"], compute_kernel_config)

    @classmethod
    def load_from_checkpoint(cls, path=None, *, abag=False, compute_kernel_config=None, device=None):
        """Fetch/load ``opendde.pt`` (or ``opendde_abag.pt`` when ``abag=True``) and build
        the model on ``device`` (card 0 by default)."""
        import ttnn
        from .tenstorrent import get_device
        dev = device or get_device()
        ckc = compute_kernel_config or ttnn.init_device_compute_kernel_config(
            dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
        return cls(load_opendde_checkpoint(path, abag=abag), ckc, dev)

    def expand_and_refine(self, ifd, s_inputs_res, s_res, z_res, *,
                          extra_attn_bias=True, return_attn_bias=False):
        """The novel seam (opendde/model/opendde.py forward): residue-trunk (s_inputs, s, z)
        -> structural-token (s_inputs, s, z). The expander produces the structural-token
        tensors + the additive pair attention bias; the 4-block refiner then refines (s, z),
        with that bias fed to the pair/triangle attention (matching OpenDDE's
        ``extra_attn_bias``). Returns ``(s_inputs_struct, s_struct, z_struct)`` as resident
        ttnn tensors. All inputs are host tensors / the integer feature dict, as for
        :meth:`StructuralTokenExpander.__call__`."""
        import ttnn
        s_inputs_st, s_st, z_st, attn_bias = self.expander(ifd, s_inputs_res, s_res, z_res)
        Ns = s_st.shape[0]
        z4 = ttnn.reshape(z_st, (1, Ns, Ns, self.expander.c_z))
        s3 = ttnn.reshape(s_st, (1, Ns, self.expander.c_s))
        bias = None
        if extra_attn_bias:
            bias = ttnn.reshape(attn_bias, (1, 1, Ns, Ns))
        s_ref, z_ref = self.refiner(s3, z4, extra_attn_bias=bias)
        result = (s_inputs_st, ttnn.reshape(s_ref, (Ns, self.expander.c_s)), z_ref)
        if return_attn_bias:
            return (*result, attn_bias)
        return result

    def fold(self, feats, *, n_step=20, n_cycles=2, seed=None, n_sample=1,
             return_confidence=False, progress_fn=None, trace=False, dump_fn=None):
        """First end-to-end residue->structure co-fold. feats: a tt_bio.protenix_data-style
        residue-token feature dict (as tt_bio.protenix.Protenix.fold consumes -- e.g.
        tt_bio.protenix_data.build_complex_features for a single protein chain).

        Pipeline (opendde/model/opendde.py get_pairformer_output -> expand_to_structural_tokens
        -> EDM diffusion, reusing the Protenix-v2 stack throughout):
          1. input embedder + trunk at the RESIDUE axis (self._protenix, c_z=384) -> s_inputs,
             s_trunk, z_trunk.
          2. expand_and_refine (the novel seam) -> structural-axis (s_inputs, s, z).
          3. diffusion pair conditioning + EDM sampler at the STRUCTURAL axis (atom broadcast
             via atom_to_structural_token_idx, not the residue atom_to_token_idx).

        Confidence + best-of-N selection reuses Protenix-v2's ConfidenceHead verbatim
        (self._protenix.confidence_head), called with the RESIDUE-axis s_inputs/s_trunk/
        z_trunk from step 1 and the ORIGINAL residue-level feats -- matching OpenDDE's own
        select_pair_output_branch(pair_output_space="residue"), which for the shipped config
        returns the pre-expansion residue tensors unchanged rather than pooling the
        structural-token pair back down (verified by reading opendde/model/opendde.py, not
        assumed). Confidence is independent of the structural-token diffusion axis, so no
        structural-token distogram-rep-atom machinery is needed here.

        --fast and multi-card fanout ride the existing Protenix-v2 machinery (the trunk
        reads the global fast flag; the predict scheduler fans targets across --devices),
        both verified for OpenDDE -- see docs/opendde-port.md. trace=True replays a
        captured ttnn trace of the shared denoise stream (lossless; faster on
        dispatch-bound diffusion, mirroring Protenix-v2.fold(trace=)); needs a device
        opened with a trace region (get_device(trace_region_size=1 << 30)). Returns
        coords (n_sample, N_atom, 3) host tensor; if
        return_confidence, returns (coords, conf) where conf is a dict (n_sample==1) or a
        list of dicts (n_sample>1), same shape as tt_bio.protenix.Protenix.fold.
        """
        import torch
        from .opendde_data import build_structural_token_features
        from .tenstorrent import get_device
        from .protenix import edm_sample

        if trace:
            import tt_bio.tenstorrent as _TTd
            if _TTd.trace_region_size() <= 0:
                raise ValueError(
                    "fold(trace=True) needs a device opened with a trace region; "
                    "call get_device(trace_region_size=1 << 30) before folding.")
        P = self._protenix
        tt = P._tt
        ifd = build_structural_token_features(feats)
        Ns = ifd["parent_residue_idx"].shape[0]

        fi = P._atom_feat_inputs(feats)
        N, NT, nb, nq, nk = fi["N"], fi["NT"], fi["nb"], fi["nq"], fi["nk"]
        mt, S = fi["mt"], fi["S"]
        Mmat = (S.t() / (S.t().sum(-1, keepdim=True) + 1e-6))
        dm = feats["deletion_mean"]; dm = dm.reshape(-1, 1) if dm.dim() == 1 else dm

        # 1) input embedder + trunk, residue axis (identical to Protenix.fold steps 1-3)
        s_inputs_tt = P.input_aae(
            tt(feats["ref_pos"]), tt(fi["ref_charge_asinh"]), tt(feats["ref_mask"].reshape(N, 1)),
            tt(fi["f_in"]), tt(fi["d"]), tt(fi["v"]), tt(fi["invd"]), mt, tt(Mmat),
            tt(feats["restype"]), tt(feats["profile"]), tt(dm))
        s_inputs = P._to_host(s_inputs_tt)[:NT]
        mt_dev = tt(mt.reshape(-1, 1).float())
        c_l = P._to_host(P.diff_feat.c_l(tt(feats["ref_pos"]), tt(fi["ref_charge_asinh"]),
                                         tt(feats["ref_mask"].reshape(N, 1)), tt(fi["f_in"])), (N, 128))
        p_lm = P._to_host(P.diff_feat.p_lm(tt(fi["d"]), tt(fi["v"]), tt(fi["invd"]), mt_dev), (nb, nq, nk, 16))
        relp = feats["relp"] if "relp" in feats else P._generate_relp(feats)
        s_trunk_tt, z_tt = P.trunk(feats, s_inputs, relp, feats["token_bonds"],
                                  n_cycles=n_cycles, progress_fn=progress_fn)
        s_trunk = P._to_host(s_trunk_tt, (NT, s_trunk_tt.shape[-1]))
        z_trunk = P._to_host(z_tt, (NT, NT, P.trunk.C_Z))

        # 2) the novel seam: residue -> structural-token axis
        s_inputs_st, s_st, z_st, structural_attn_bias = self.expand_and_refine(
            ifd, s_inputs, s_trunk, z_trunk, return_attn_bias=True)
        s_inputs_struct = P._to_host(s_inputs_st, (Ns, self.expander.c_s_inputs))
        s_struct = P._to_host(s_st, (Ns, self.expander.c_s))
        structural_attn_bias = P._to_host(structural_attn_bias, (Ns, Ns))

        # 3) diffusion pair conditioning + EDM sampler, structural-token axis
        parent = ifd["parent_residue_idx"]
        relp_struct = P._generate_relp({
            "asym_id": feats["asym_id"].index_select(0, parent),
            "residue_index": feats["residue_index"].index_select(0, parent),
            "entity_id": feats["entity_id"].index_select(0, parent),
            "sym_id": feats["sym_id"].index_select(0, parent),
            "token_index": ifd["structural_token_index"],
        })
        pair_z = P._diffusion_pair_cond(z_st, relp_struct).reshape(Ns, Ns, -1)
        a2s = ifd["atom_to_structural_token_idx"]
        S_struct = torch.zeros(N, Ns); S_struct[torch.arange(N), a2s] = 1.0
        p_lm = p_lm + P._plm_z_term(pair_z, a2s, nb, nq, nk)

        cond = {"s_trunk": s_struct, "s_inputs": s_inputs_struct, "pair_z": pair_z, "c_l": c_l,
                "p_lm": p_lm, "S": S_struct, "mask_trunked": mt.float(),
                "structural_pair_attn_bias": structural_attn_bias}
        if P.diffusion.device_dit:
            cond["dit_z"] = P.diffusion._dit_z_device(pair_z)
        else:
            cond["dit_biases"] = P.diffusion._dit_pair_biases(pair_z)
        coords = []
        for k in range(n_sample):
            sd_seed = None if seed is None else seed + k
            _df = (lambda step, x, _k=k: dump_fn(_k, step, x)) if dump_fn is not None else None
            coords.append(edm_sample(P.diffusion, cond, N, n_step=n_step, seed=sd_seed,
                                     trace=trace, progress_fn=progress_fn, dump_fn=_df)[0])
        coords = torch.stack(coords, 0)
        if return_confidence:
            # Residue-axis confidence (select_pair_output_branch(pair_output_space="residue")):
            # s_inputs/s_trunk/z_trunk are the step-1 pre-expansion tensors, `feats` the
            # original residue-level dict -- identical call shape to Protenix.fold's.
            confs = [P.confidence_head.confidence(s_inputs, s_trunk, z_trunk, coords[k], feats)
                     for k in range(n_sample)]
            return coords, (confs[0] if n_sample == 1 else confs)
        return coords
