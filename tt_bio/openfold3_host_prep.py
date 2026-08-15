"""Host-side derivation of every OpenFold3 ``fold()`` input that the P11/P12 accuracy
path used to read from the ``~/of3_ref_out.pkl`` golden fixture.

Everything here is a pure function of ``build_openfold3_features()`` output plus the
checkpoint state dict -- no golden intermediates anywhere:

- ``derive_template_feat``: the template-embedder feature block from the raw
  ``template_*`` feature keys (the same construction
  ``scripts/of3_template_embedder_golden.py`` used to capture the fixture). For a
  template-free query this is the all-dummy block (restype one-hot 31, zeros elsewhere).
- ``derive_relpos``: the 139-d ``relpos_complex`` feature (input-embedder config
  ``max_relative_idx=32``, ``max_relative_chain=2``).
- ``derive_block_aux``: the atom-attention block/mask auxiliaries (``nb``, ``NP``,
  ``key_block_idxs``, ``invalid_mask``, ``mask_trunked``, ``npe_q/k_indices``,
  ``zij_mask``, ``atom_to_token_mean``, ``max_atom_per_token_mask``, ``ca_mask``) from
  ``atom_mask`` / ``atom_to_token_index`` / ``token_mask`` -- the same derivations the
  golden capture scripts ran once and froze into the pickle.
- ``ref_atom_embed``: host replica of OF3 ``RefAtomFeatureEmbedder`` (eight bias-free
  linears + the block construction), yielding the DM encoder's pre-noisy-position
  conditioning ``cl0``/``plm0`` and the input embedder's ``cl``/``plm``.
- ``run_input_atom_encoder``: the input embedder's atom-encoder leg (-> ``ai``):
  host ref-atom embedding + ``linear_l``/``linear_m``/``pair_mlp`` pair completion,
  then the PCC-gated device ``OF3AtomTransformer`` and the mean aggregation to tokens.

Validated tensor-by-tensor against the 1UBQ fixture by
``scripts/of3_host_prep_check.py`` (PCC / exact-match per tensor).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ._vendor.openfold3.core.utils.atom_attention_block_utils import (
    convert_single_rep_to_blocks,
    get_block_indices,
    get_pair_atom_block_mask,
    get_query_block_padding,
)
from ._vendor.openfold3.core.utils.relpos import relpos_complex
from .openfold3_weights import _sub

N_QUERY = 32
N_KEY = 128
MAX_RELATIVE_IDX = 32
MAX_RELATIVE_CHAIN = 2
ATOM_SLOTS_PER_TOKEN = 23


def derive_template_feat(features: dict) -> dict:
    """Template-embedder feature block from the raw ``template_*`` keys.

    Mirrors the fixture capture in ``scripts/of3_template_embedder_golden.py``:
    pair masks are the outer product of the per-token mask with itself, restricted to
    same-chain pairs; ``restype_ti/tj`` broadcast the per-token restype one-hot over
    the pair axes; the unit vector is split per component.
    """
    asym = features["asym_id"]
    mcm = (asym[:, None] == asym[None, :]).float()          # [N, N]
    pbm = features["template_pseudo_beta_mask"].float()      # [N_t, N]
    bbfm = features["template_backbone_frame_mask"].float()  # [N_t, N]
    restype = features["template_restype"].float()           # [N_t, N, 32]
    n_tok = restype.shape[-2]
    uv = features["template_unit_vector"].float()            # [N_t, N, N, 3]
    ux, uy, uz = uv.unbind(dim=-1)
    return {
        "distogram": features["template_distogram"].float(),
        "pseudo_beta_pair_mask": (pbm[..., None] * pbm[..., None, :])[..., None] * mcm[..., None],
        "restype_ti": restype[..., None, :].expand(*restype.shape[:-2], -1, n_tok, -1).contiguous(),
        "restype_tj": restype[..., None, :, :].expand(*restype.shape[:-2], n_tok, -1, -1).contiguous(),
        "unit_vec_x": ux[..., None],
        "unit_vec_y": uy[..., None],
        "unit_vec_z": uz[..., None],
        "backbone_frame_pair_mask": (bbfm[..., None] * bbfm[..., None, :])[..., None] * mcm[..., None],
    }


def dedup_template_slots(feat: dict) -> tuple[dict, list[int]]:
    """Drop template slots that repeat an earlier one, and say where each slot went.

    A query with fewer real templates than ``n_templates`` fills the rest of the slots
    from the same GAP/NaN precursor, so every derived feature is byte-identical across
    them and the pair stack runs the same two blocks on the same values several times.
    Returns the feature dict restricted to the distinct slots plus, per original slot,
    the row of that dict it maps to. Slots that genuinely differ are all kept, so a
    query with real templates takes the unchanged path.
    """
    nt = int(next(iter(feat.values())).shape[0])
    reps: list[int] = []          # original index of each distinct slot
    slot_index: list[int] = []    # per original slot, its row in the returned dict
    for t in range(nt):
        for j, r in enumerate(reps):
            if all(torch.equal(v[t], v[r]) for v in feat.values()):
                slot_index.append(j)
                break
        else:
            slot_index.append(len(reps))
            reps.append(t)
    if len(reps) == nt:
        return feat, slot_index
    return {k: v[reps].contiguous() for k, v in feat.items()}, slot_index


def derive_relpos(features: dict) -> torch.Tensor:
    """139-d ``relpos_complex`` feature, input-embedder clipping config."""
    return relpos_complex(
        batch=features,
        max_relative_idx=MAX_RELATIVE_IDX,
        max_relative_chain=MAX_RELATIVE_CHAIN,
    ).float()


def derive_block_aux(features: dict) -> dict:
    """Atom-attention block auxiliaries from ``atom_mask``/``atom_to_token_index``.

    Same derivations as the ``diffusion_module_xlout`` / ``atom_transformer`` golden
    captures (scripts/of3_diffusion_module_xlout_golden.py).
    """
    atom_mask = features["atom_mask"].float()
    atom_to_token_index = features["atom_to_token_index"].long()
    n_atom = int(atom_mask.shape[0])
    n_token = int(features["token_mask"].shape[0])
    nb = math.ceil(n_atom / N_QUERY)
    NP = nb * N_QUERY
    pad_right = get_query_block_padding(n_atom, N_QUERY)
    key_block_idxs, invalid_mask = get_block_indices(
        atom_mask=atom_mask, n_query=N_QUERY, n_key=N_KEY, device=torch.device("cpu"))
    mask_trunked = get_pair_atom_block_mask(
        atom_mask=atom_mask, num_blocks=nb, n_query=N_QUERY, n_key=N_KEY,
        pad_len_right_q=pad_right, key_block_idxs=key_block_idxs,
        invalid_mask=invalid_mask)
    npe_q_indices = F.pad(atom_to_token_index, (0, pad_right), value=0).reshape(nb, N_QUERY).long()
    npe_k_indices = torch.gather(
        atom_to_token_index.unsqueeze(0).expand(nb, n_atom), 1, key_block_idxs.long())
    zij_mask = ((~invalid_mask).float())[:, None, :].expand(nb, N_QUERY, N_KEY) * mask_trunked

    atom_to_token_mean = torch.zeros(n_token, n_atom)
    atom_to_token_mean[atom_to_token_index, torch.arange(n_atom)] = atom_mask
    atom_to_token_mean = atom_to_token_mean / atom_to_token_mean.sum(-1, keepdim=True).clamp_min(1.0)

    # broadcast_token_feat_to_atoms layout: per token, the first num_atoms_per_token
    # slots carry the token value, the remaining 23 - n slots are zero.
    napt = features["num_atoms_per_token"].long() * features["token_mask"].long()
    max_atom_per_token_mask = (
        (torch.arange(ATOM_SLOTS_PER_TOKEN)[None, :] < napt[:, None]).float()
        * features["token_mask"].float()[:, None]
    ).reshape(-1)

    atom_array = features["atom_array"]
    # Representative ("token center") atom per token for the confidence head: CA for
    # protein, C1' for nucleic acids — the upstream TOKEN_CENTER_ATOMS convention the
    # vendored tokenizer already annotates. A protein-only "== CA" mask leaves RNA/DNA
    # tokens with zero representatives and crashes the confidence head on an empty
    # one-hot. Fall back to CA only if the annotation is absent.
    if "token_center_atom" in atom_array.get_annotation_categories():
        ca_mask = torch.from_numpy(atom_array.token_center_atom).bool()
    else:
        ca_mask = torch.from_numpy(atom_array.atom_name == "CA").bool()

    return dict(
        atom_mask=atom_mask, atom_to_token_index=atom_to_token_index,
        n_atom=n_atom, n_token=n_token, nb=nb, NP=NP,
        key_block_idxs=key_block_idxs, invalid_mask=invalid_mask,
        mask_trunked=mask_trunked, npe_q_indices=npe_q_indices,
        npe_k_indices=npe_k_indices, zij_mask=zij_mask,
        atom_to_token_mean=atom_to_token_mean,
        max_atom_per_token_mask=max_atom_per_token_mask, ca_mask=ca_mask)


def _lin(x: torch.Tensor, w: dict, name: str) -> torch.Tensor:
    out = x @ w[f"{name}.weight"].t().float()
    bias = w.get(f"{name}.bias")
    if bias is not None:
        out = out + bias.float()
    return out


def ref_atom_embed(w: dict, features: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Host replica of OF3 ``RefAtomFeatureEmbedder`` (Algorithm 5, line 1-7).

    ``w`` is the encoder's ``ref_atom_feature_embedder`` sub-state-dict (all eight
    linears are bias-free in the OF3 checkpoint). Returns ``cl`` [n_atom, c_atom] and
    ``plm`` [nb, n_query, n_key, c_atom_pair].
    """
    ref_pos = features["ref_pos"].float()
    atom_mask = features["atom_mask"].float()
    cl = (
        _lin(ref_pos, w, "linear_ref_pos")
        + _lin(torch.arcsinh(features["ref_charge"].unsqueeze(-1).float()), w, "linear_ref_charge")
        + _lin(features["ref_mask"].unsqueeze(-1).float(), w, "linear_ref_mask")
        + _lin(features["ref_element"].float(), w, "linear_ref_element")
        + _lin(features["ref_atom_name_chars"].flatten(start_dim=-2).float(), w, "linear_ref_atom_chars")
    )
    d_l, d_m, pair_mask = convert_single_rep_to_blocks(
        ql=ref_pos, n_query=N_QUERY, n_key=N_KEY, atom_mask=atom_mask)
    v_l, v_m, _ = convert_single_rep_to_blocks(
        ql=features["ref_space_uid"].unsqueeze(-1).float(),
        n_query=N_QUERY, n_key=N_KEY, atom_mask=atom_mask)
    dlm = (d_l.unsqueeze(-2) - d_m.unsqueeze(-3)) * pair_mask.unsqueeze(-1)
    vlm = (v_l.unsqueeze(-2) == v_m.unsqueeze(-3)).float() * pair_mask.unsqueeze(-1)
    inv_sq_dists = 1.0 / (1.0 + (dlm ** 2).sum(dim=-1, keepdim=True))
    plm = (
        _lin(dlm, w, "linear_ref_offset") * vlm
        + _lin(inv_sq_dists, w, "linear_inv_sq_dists") * vlm
        + _lin(vlm, w, "linear_valid_mask") * vlm
    )
    return cl, plm


def run_input_atom_encoder(dev, compute_kernel_config, sd: dict, features: dict,
                           aux: dict) -> torch.Tensor:
    """Input-embedder atom-encoder leg -> ``ai`` [n_token, 384].

    Host: ref-atom embedding + pair-rep completion. Device: the PCC-gated
    ``OF3AtomTransformer`` (P7). Host: ``relu(linear_q(...))`` + mean aggregation to
    tokens via ``atom_to_token_mean``.
    """
    import ttnn

    from .openfold3_atom_transformer import OF3AtomTransformer

    enc = _sub(sd, "input_embedder.atom_attn_enc")
    atom_mask = aux["atom_mask"]
    n_atom, nb, NP = aux["n_atom"], aux["nb"], aux["NP"]

    cl, plm = ref_atom_embed(_sub(enc, "ref_atom_feature_embedder"), features)
    # Algorithm 5 lines 13-14 (pair completion), host replica.
    cl_l, cl_m, pair_mask = convert_single_rep_to_blocks(
        ql=cl, n_query=N_QUERY, n_key=N_KEY, atom_mask=atom_mask)
    cl_lm = (
        _lin(cl_l.relu().unsqueeze(-2), enc, "linear_l")
        + _lin(cl_m.relu().unsqueeze(-3), enc, "linear_m")
    ) * pair_mask.unsqueeze(-1)
    plm = plm + cl_lm
    h = plm
    for key in ("pair_mlp.1", "pair_mlp.3", "pair_mlp.5"):
        h = _lin(h.relu(), enc, key)
    plm = (plm + h) * pair_mask.unsqueeze(-1)

    at = OF3AtomTransformer(_sub(enc, "atom_transformer"), compute_kernel_config)

    def ft(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(x.float(), layout=layout, device=dev, dtype=dtype)

    a_pad = torch.zeros(1, NP, 128)
    a_pad[0, :n_atom] = cl
    atom_mask_col = torch.zeros(1, NP, 1)
    atom_mask_col[0, :n_atom, 0] = atom_mask
    valid = (~aux["invalid_mask"]).float().reshape(1, nb, N_KEY, 1)
    mask_bias = (1e9 * (aux["mask_trunked"] - 1)).reshape(1, nb, 1, N_QUERY, N_KEY)
    kidx = aux["key_block_idxs"].reshape(1, nb * N_KEY).to(torch.int32)
    kidx_tt = ttnn.from_torch(kidx, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev,
                              dtype=ttnn.uint32)
    ql_d = at(ft(a_pad), ft(a_pad.clone()), ft(plm.unsqueeze(0)), ft(atom_mask_col),
              kidx_tt, ft(valid), ft(mask_bias), n_atom, NP, nb)
    ql = ttnn.to_torch(ql_d).float().reshape(n_atom, 128)
    ttnn.deallocate(ql_d)

    lq_w = _sub(enc, "linear_q")["0.weight"]
    q = F.linear(ql * atom_mask[:, None], lq_w.float()).relu()
    ai = aux["atom_to_token_mean"] @ q
    return ai
