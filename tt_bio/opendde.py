"""OpenDDE StructuralTokenExpander on Tenstorrent.

OpenDDE (Aureka AI Research) is an AF3-family co-folding model whose entire
trunk / MSA / diffusion / confidence graph is Protenix-v2's, already ported in
``tt_bio.tenstorrent`` + ``tt_bio.protenix``. Its one novel compute block is
``StructuralTokenExpander``: it expands the residue-level trunk (``s_inputs``,
``s``, ``z``) onto the structural-token axis, adding role conditioning and
same-residue pair structure, before diffusion. The rest of the pipeline then
runs unchanged on the structural-token axis (the ttnn ops are axis-agnostic).

This module ports that one block; assembly reuses the Protenix-v2 stack verbatim.
The integer index maps (parent, prev/next-parent
adjacency, role-pair-type) are precomputed host-side; the gathers themselves,
the split-MLP, the 49 role-pair pair projections, and the bias adds run on
device.
"""
import torch
import ttnn

from .protenix import _KeyedWeights
from .tenstorrent import _acc_concat, concat_host_bytes, get_device

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


# Keep the OpenDDE trunk->structural z_trunk host copy in the bf16 the device held instead of
# upcasting it to fp32 that both consumers immediately undo. Bit-exact; ON by default.
_SEAM_BF16 = True


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

    def _emb_tt(self, key):
        """Pair-init embedding table, uploaded once as fp32 TILE and cached.
        fp32 so the five-way sum below rounds once at the final fp32->bf16
        cast, bit-identical to the old host fp32 sum + from_torch upload."""
        cache = self.__dict__.setdefault("_wc", {})
        v = cache.get((key, "emb_fp32"))
        if v is None:
            v = ttnn.from_torch(self._w[key], layout=ttnn.TILE_LAYOUT,
                                device=get_device(), dtype=ttnn.float32)
            cache[(key, "emb_fp32")] = v
        return v

    def _pair_init_bias(self, pf):
        """Sum of the five additive pair-init embeddings, on device. Each table
        lookup is a where-chain row select in fp32 (pure selection, no
        arithmetic, so the row comes through exactly), the five results add in
        the host's order in fp32, and one fp32->bf16 cast rounds at the end --
        bit-identical to the old host fp32 gather/sum + from_torch upload
        (IEEE fp32 adds, both casts round-to-nearest-even). Per chunk only the
        five (clen,Ns,1) index grids upload instead of a (clen,Ns,c_z) fp32
        gather+sum+upload. (ttnn.embedding is bf16-only and ttnn.matmul rounds
        fp32 inputs to bf16, hence the where-chain.)

        The index grid uploads already shaped (clen, Ns, 1) so the where-chain
        result comes out (clen, Ns, c_z) directly. The first version of this
        uploaded it flat and reshaped the (1, clen*Ns, c_z) result at the end,
        which splits a TILE tensor's row axis on Ns=249 -- not a multiple of 32.
        That tensor reads back through ttnn.to_torch bit-exact and computes
        WRONG as an operand of the ttnn.add below, so every host-side check
        passed while the fold's pair track was corrupted (`6c3f5ecaf`; the
        isolated shapes do not reproduce it, only a fold does --
        perf/wh-correctness/pairbias_wherechain_probe.py)."""
        dev = get_device()
        C = self.c_z
        clen, Ns = pf["role_pair_type"].shape
        b = None
        for tkey, ikey, n in (("same_parent_embedding.weight", "same_parent_residue", 2),
                              ("same_residue_twin_embedding.weight", "same_residue_twin", 2),
                              ("prev_bb_chain_embedding.weight", "prev_bb_chain", 2),
                              ("next_bb_chain_embedding.weight", "next_bb_chain", 2),
                              ("role_pair_type_embedding.weight", "role_pair_type", 8)):
            tab = self._emb_tt(tkey)
            idx = ttnn.from_torch(pf[ikey].reshape(clen, Ns, 1).to(torch.int32),
                                  layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.uint32)
            g = ttnn.reshape(ttnn.slice(tab, [n - 1, 0], [n, C]), (1, 1, C))
            for k in range(n - 2, -1, -1):
                rowk = ttnn.reshape(ttnn.slice(tab, [k, 0], [k + 1, C]), (1, 1, C))
                g = ttnn.where(ttnn.eq(idx, k), rowk, g)
            b = g if b is None else ttnn.add(b, g)
        return ttnn.typecast(b, getattr(self, "dtype", ttnn.bfloat16))

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

    def _pair_project_full(self, z_chunk_dev, role, row_index):
        """delta[a,b] = W[role[a]*n+role[b]] @ z[a,b], full 49-projection mode.
        The z chunk arrives device-resident (bf16, row-major (clen*Ns, C)); the
        role-pair grouping is a device gather by the host-computed permutation,
        each group is one device matmul, then scattered back via a second
        device gather -- numerically identical to OpenDDE's per-(role_i,role_j)
        masked projection, reordered. Bit-exact vs the old host gather +
        per-group upload: both gathers move the same bf16 values."""
        clen = row_index.numel()
        Ns = role.shape[0]
        C = self.c_z
        row_role = role.index_select(0, row_index)
        role_i = row_role[:, None].expand(clen, Ns).reshape(-1)
        role_j = role[None, :].expand(clen, Ns).reshape(-1)
        pidx = role_i * self.n_roles + role_j

        perm = torch.argsort(pidx, stable=True)
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(perm.numel())
        uniq, counts = torch.unique_consecutive(pidx.index_select(0, perm), return_counts=True)

        dev = get_device()
        sorted_dev = ttnn.embedding(
            ttnn.from_torch(perm.reshape(1, -1).to(torch.int32),
                            layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, dtype=ttnn.uint32),
            z_chunk_dev, layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        sorted_dev = ttnn.reshape(sorted_dev, (clen * Ns, C))

        pieces = []
        off = 0
        for g, c in zip(uniq.tolist(), counts.tolist()):
            seg = ttnn.to_layout(ttnn.slice(sorted_dev, [off, 0], [off + c, C]),
                                 ttnn.TILE_LAYOUT)
            out = self._lin(seg, "pair_block_proj.%d.weight" % g)
            pieces.append(ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT))
            off += c
        ttnn.deallocate(sorted_dev)
        sorted_delta = pieces[0] if len(pieces) == 1 else ttnn.concat(pieces, dim=0)

        inv_idx = ttnn.from_torch(inv.reshape(1, -1).to(torch.int32),
                                  layout=ttnn.ROW_MAJOR_LAYOUT, device=dev,
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
        # z_res stays resident: upload once as bf16 and do both gathers on
        # device -- the (row_parent x parent) chunk gather below, and the
        # role-pair permutation inside _pair_project_full -- instead of a
        # (chunk,Ns,c_z) fp32 host gather + ~118 MB of per-chunk uploads.
        # Bit-exact: the gathers are pure movement and the fp32->bf16 cast is
        # elementwise, so it commutes with them.
        Nr = z_res.shape[0]
        z_flat = ttnn.from_torch(
            z_res.reshape(Nr * Nr, self.c_z), layout=ttnn.ROW_MAJOR_LAYOUT,
            device=get_device(), dtype=getattr(self, "dtype", ttnn.bfloat16))
        # Assemble z_struct on the host once it is large, and upload it as ONE allocation
        # after the row loop's transients are gone.
        #
        # This is placement, not footprint. Measured on the Galaxy at Ns=2113: the trunk hands
        # the seam a single 912.8 MiB/bank hole, and by the time this function returns the
        # largest hole is 208.7 MiB -- already the number the refiner is then refused at, before
        # it has allocated anything. The device concat asks for its [Ns,Ns,c_z] result while all
        # ceil(Ns/128) row chunks are still live beside it (~3.4 GB at Ns=2113), so the result
        # lands above them and its neighbours never coalesce. Accumulating on the host leaves at
        # most one chunk on device, so the upload takes the bottom of an intact hole and what
        # remains above it stays contiguous -- which is what the refiner's own pair tensor needs.
        #
        # Bit-exact: to_torch preserves the bf16 bytes, the concat is along the row axis so no
        # tile padding is crossed, and from_torch re-tilizes the same values. bf16 only, for the
        # reason _host_concat gives: a bf8 or fp32 round trip through torch bf16 would not be.
        host_z = (z_flat.dtype == ttnn.bfloat16
                  and Ns * Ns * self.c_z * 2 > concat_host_bytes())
        z_chunks, ab_chunks, live = [], [], []
        for start in range(0, Ns, chunk):
            end = min(start + chunk, Ns)
            row_index = torch.arange(start, end)
            pf = self._pair_features_rows(ifd, role, parent, row_index)
            row_parent = parent.index_select(0, row_index)
            gidx = (row_parent[:, None] * Nr + parent[None, :]).reshape(1, -1).to(torch.int32)
            z_dev = ttnn.embedding(
                ttnn.from_torch(gidx, layout=ttnn.ROW_MAJOR_LAYOUT, device=get_device(),
                                dtype=ttnn.uint32),
                z_flat, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            z_dev = ttnn.reshape(z_dev, ((end - start) * Ns, self.c_z))
            z_tile = ttnn.to_layout(
                ttnn.reshape(z_dev, (end - start, Ns, self.c_z)), ttnn.TILE_LAYOUT)
            z_tile = ttnn.add(z_tile, self._pair_project_full(z_dev, role, row_index))
            z_tile = ttnn.add(z_tile, self._pair_init_bias(pf))
            if host_z:
                z_chunks.append(ttnn.to_torch(z_tile))
                live.append(z_tile)      # freed together below, NOT one per iteration
            else:
                z_chunks.append(z_tile)
            ab_chunks.append(self._attn_bias(pf))
        # Free the row loop's residents before the upload, in one go at the end. Deallocating
        # each chunk inside the loop instead lets the next iteration's tensors reuse its memory,
        # and that -- not the host assembly, which is bit-identical (verified in a real fold,
        # scripts/probe_zstruct_assembly.py: 8 blocks, (977,977,384), equal=True) -- is what moved
        # 9i3p's structure. Freeing here keeps the loop's allocation pattern as it was and still
        # leaves the heap empty for the upload, which is the whole point of the change.
        for t in live:
            ttnn.deallocate(t)
        ttnn.deallocate(z_flat)
        z_struct = _acc_concat(z_chunks, -3, host_z)
        attn_bias = ab_chunks[0] if len(ab_chunks) == 1 else ttnn.concat(ab_chunks, dim=0)
        return s_inputs_struct, s_struct, z_struct, attn_bias


# ---------------------------------------------------------------------------
# Pipeline assembly + real-weight load.
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
        from tt_bio import weights
        path = weights.fetch("opendde-abag" if abag else "opendde")
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
    axis. Ships co-folding only (no design/affinity)."""

    def __init__(self, state_dict, compute_kernel_config, device=None):
        from .tenstorrent import get_device, Pairformer, accurate_softmax_site
        from .protenix import Protenix
        self.dev = device or get_device()
        self.compute_kernel_config = compute_kernel_config
        C = OPENDDE_CONFIG
        routed = route_opendde_weights(state_dict)
        self._shared = routed["shared"]         # Protenix-v2-family graph (for step-2 trunk/diffusion)
        # Shared Protenix-v2-family stack (input embedder, trunk, diffusion, confidence),
        # built at OpenDDE's c_z=384.
        # Reused verbatim -- no duplicated orchestration class. diffusion_fp32=False pins
        # OpenDDE to its own validated bf16 diffusion config regardless of Protenix-v2's
        # PROTENIX_DIFFUSION_FP32_DEVICE default (fp32 diffusion is >60x slower on OpenDDE's
        # atom-level tensors, see tt-bio-shared-diffusion-global-env-default-regression).
        # gated_move=True: E6 fires on 1048 of the fold's 1216 trimul channel moves at c_z=384
        # and is torch.equal to the sequence it replaces at both slice widths.
        #
        # OPENDDE_DIFFUSION_FP32=1 lifts the bf16 pin for an A/B. The pin is a perf decision,
        # so it needs an opt-out that does not also flip Protenix-v2 (which is what
        # PROTENIX_DIFFUSION_FP32_DEVICE would do -- tt-bio-shared-diffusion-global-env-default-regression).
        # Diagnostic only: fp32 here is >60x slower on OpenDDE's atom-level tensors.
        import os
        self._protenix = Protenix(
            self._shared, compute_kernel_config, self.dev, c_z=C["c_z"], msa_update_first=True,
            diffusion_fp32=os.environ.get("OPENDDE_DIFFUSION_FP32", "0") == "1", gated_move=True,
            softmax_scope="opendde")
        self.expander = StructuralTokenExpander(
            routed["expander"], compute_kernel_config, c_s=C["c_s"], c_z=C["c_z"],
            c_s_inputs=C["c_s_inputs"], n_roles=C["n_roles"], pair_chunk_size=C["pair_chunk_size"])
        self.refiner = Pairformer(
            routed["refiner_blocks"], C["c_z"] // C["refiner_tri_heads"], C["refiner_tri_heads"],
            C["c_s"] // C["refiner_att_heads"], C["refiner_att_heads"], True,
            routed["refiner"], compute_kernel_config, gated_move=True,
            accurate_softmax=accurate_softmax_site("opendde.refiner"))

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
        # Build the refiner trimuls' fused input-weight cache up front, so the first (timed)
        # call does not interleave the 96-tensor `_gp_cache` uploads with its compute.
        # Numerically inert, measured: a fold with this prewarm produces the same numbers as
        # one without.
        for blk in self.refiner.blocks:
            blk.triangle_multiplication_start.prewarm(Ns, 1)
            blk.triangle_multiplication_end.prewarm(Ns, 1)
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
             return_confidence=False, progress_fn=None, trace=False, dump_fn=None,
             max_parallel_samples=None):
        """End-to-end residue-to-structure co-fold. feats: a tt_bio.protenix_data-style
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
        both apply unchanged to OpenDDE. trace=True replays a
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
        from .protenix import DEFAULT_MAX_PARALLEL_SAMPLES, edm_sample

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
        # The z_trunk host copy stays bf16 when `_SEAM_BF16` is on. Its two consumers both
        # accept it: the expander re-uploads it as bf16 (`ttnn.from_torch(..., dtype=bfloat16)`,
        # a no-op cast), and the confidence head only ever adds it to fp32 tensors, where torch
        # promotes bf16 -> fp32 exactly. Deletes 402 MB of host fp32 writes per fold.
        z_trunk = P._to_host(z_tt, (NT, NT, P.trunk.C_Z), fp32=not _SEAM_BF16)
        # The residue-axis device tensors are never read from device again -- the
        # expander, diffusion conditioning and confidence all consume the host copies
        # above. Free them before the expander allocates the structural-scale pair
        # tensor (~1.9x the residue axis): on 12 GiB Wormhole parts their holes are
        # what the refiner's full-size concats squeeze into.
        for _t in (s_inputs_tt, s_trunk_tt, z_tt, mt_dev):
            ttnn.deallocate(_t)

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
        # The structural pair tensor's only consumer was the pair conditioning above
        # (confidence runs on the residue axis). Free it before the sampler stage: at
        # Ns=2113 it is 3.2 GiB the confidence pairformer will need back.
        ttnn.deallocate(z_st)
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
        # Multiplicity batching (see Protenix.fold): one batched trajectory when
        # P.diffusion.supports_multiplicity is on; else the per-sample loop (bit-exact).
        if n_sample > 1 and getattr(P.diffusion, "supports_multiplicity", False):
            _mps = DEFAULT_MAX_PARALLEL_SAMPLES if max_parallel_samples is None else max_parallel_samples
            _df = (lambda step, x: dump_fn(step, step, x)) if dump_fn is not None else None
            coords = edm_sample(P.diffusion, cond, N, n_step=n_step, multiplicity=n_sample,
                                 max_parallel_samples=_mps, seed=seed, trace=trace,
                                 progress_fn=progress_fn, dump_fn=_df)
        else:
            coords = []
            for k in range(n_sample):
                sd_seed = None if seed is None else seed + k
                _df = (lambda step, x, _k=k: dump_fn(_k, step, x)) if dump_fn is not None else None
                coords.append(edm_sample(P.diffusion, cond, N, n_step=n_step, seed=sd_seed,
                                         trace=trace, progress_fn=progress_fn, dump_fn=_df)[0])
            coords = torch.stack(coords, 0)
        # dit_z (LN(pair_z) uploaded for the on-device DiT) is sampler-only state; the
        # residue-axis confidence head never reads it. At Ns=2113 it is another ~1.1 GiB
        # the confidence pairformer needs back.
        if "dit_z" in cond:
            ttnn.deallocate(cond["dit_z"])
        if return_confidence:
            # Residue-axis confidence (select_pair_output_branch(pair_output_space="residue")):
            # s_inputs/s_trunk/z_trunk are the step-1 pre-expansion tensors, `feats` the
            # original residue-level dict -- identical call shape to Protenix.fold's.
            confs = [P.confidence_head.confidence(s_inputs, s_trunk, z_trunk, coords[k], feats)
                     for k in range(n_sample)]
            return coords, (confs[0] if n_sample == 1 else confs)
        return coords
