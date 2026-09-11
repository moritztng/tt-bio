"""Real template features for the Protenix / OpenDDE template embedder.

`protenix.py::Trunk._template` runs a real template embedder (a 2-block pairformer stack on
protenix-v2 and opendde) over five features: `template_aatype`, `template_distogram`,
`template_pseudo_beta_mask`, `template_unit_vector` and `template_backbone_frame_mask`. Until
this module it was only ever handed `protenix_data.dummy_template_features` — slot 0's aatype
all gap, everything else zero — so a `templates:` block in the input reached a validator and
then nothing. This builds those five from a template structure and an alignment.

The geometry is upstream's (bytedance/Protenix, `protenix/data/template/template_utils.py`,
`TemplateFeatures` + `Templates.as_protenix_dict`), reproduced rather than approximated,
because the embedder is trained on it:

* pseudo-beta is CB, or CA for glycine;
* the distogram is 39 bins with left edges `linspace(3.25, 50.75, 39)`, compared as SQUARED
  distances against squared edges, the last upper edge 1e8;
* the frame is CA-origin with e1 along C-CA, e2 the Gram-Schmidt component of N-CA, e3 =
  e1 x e2, and the unit vector is `normalise(R^T (CA_j - CA_i))` with epsilon 1e-6;
* both masks are the outer product of the per-residue atom mask;
* the distogram and the unit vector are masked by their own 2D mask, and computed over the
  WHOLE complex token axis, so a cross-chain pair is real whenever both ends have a template.

`tests/test_protenix_template.py` scores every one of those against upstream's own functions.

The alignment file is the same `templates:` npz OpenFold3 already takes, so one input works on
both stacks: `{f"{pdb_id}_{chain}": {"index": rank, "release_date": ..., "idx_map": (n, 2)}}`
with column 0 the 1-indexed query residue and column 1 the template residue id. Coordinates
come from `<pdb_id>.cif` in the shared template-structure cache, which the worker fetches from
RCSB exactly as it already does for OpenFold3.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

MAX_TEMPLATES = 4
GAP = 31                       # protenix STD_RESIDUES_WITH_GAP["-"]
#: N, CA and C build the backbone frame; the fourth slot is the pseudo-beta atom (CB, CA for
#: GLY). Everything the five template features read, so nothing else is carried around.
FRAME_ATOMS = ("N", "CA", "C")
_MIN_BIN, _MAX_BIN, _NUM_BINS = 3.25, 50.75, 39


def _resname_to_restype() -> dict[str, int]:
    """CCD 3-letter residue name -> protenix restype index, read off the featurizer's own
    one-letter order so the two cannot disagree."""
    from tt_bio.data import const
    from tt_bio.protenix_data import RESTYPE_ORDER

    return {const.prot_letter_to_token[letter]: i
            for i, letter in enumerate(RESTYPE_ORDER)}


def _dgram_breaks():
    lower = np.square(np.linspace(_MIN_BIN, _MAX_BIN, _NUM_BINS, dtype=np.float32))
    upper = np.empty_like(lower)
    upper[:-1] = lower[1:]
    upper[-1] = 1e8
    return lower, upper


def read_alignment_entries(path, max_templates: int = MAX_TEMPLATES):
    """The template entries of a `templates:` npz, best first.

    Returns [(pdb_id, chain_id, idx_map)] for at most ``max_templates`` entries, ordered by
    the alignment rank the file records, so which templates are used does not depend on dict
    order.
    """
    with np.load(str(path), allow_pickle=True) as z:
        entries = []
        for key in z.files:
            e = z[key]
            e = e.item() if getattr(e, "shape", None) == () else e
            pdb_id, _, chain = str(key).partition("_")
            idx_map = np.asarray(e["idx_map"], dtype=np.int64)
            idx_map = idx_map[(idx_map[:, 0] != -1) & (idx_map[:, 1] != -1)]
            entries.append((int(e.get("index", 0)), pdb_id, chain or "A", idx_map))
    entries.sort(key=lambda t: t[0])
    return [(p, c, m) for _i, p, c, m in entries[:max_templates]]


def _template_chain_atoms(cif_path, chain_id):
    """{res_id: (res_name, {atom_name: xyz})} for one chain of a template mmCIF.

    PDB-assigned numbering (label_asym_id / label_seq_id / label_comp_id / label_atom_id),
    which is what an alignment's template residue index refers to -- the same swap
    OpenFold3's own loader makes (`_vendor/openfold3/.../labels.py`). Author numbering is a
    different space entirely: on 1y57 it runs 82..533 while the alignment indexes 1..450, and
    reading it that way matched the query letter on 27 of 368 aligned residues instead of 279
    of 444. First model only, highest-occupancy copy of a duplicated atom.
    """
    import biotite.structure.io.pdbx as pdbx

    arr = pdbx.get_structure(pdbx.CIFFile.read(str(cif_path)), model=1,
                             use_author_fields=False, extra_fields=["occupancy"])
    arr = arr[arr.chain_id == chain_id]
    out: dict[int, tuple[str, dict]] = {}
    occ: dict[tuple[int, str], float] = {}
    for i in range(arr.array_length()):
        rid, name = int(arr.res_id[i]), str(arr.atom_name[i])
        o = float(arr.occupancy[i])
        if occ.get((rid, name), -1.0) >= o:
            continue
        occ[(rid, name)] = o
        rec = out.setdefault(rid, (str(arr.res_name[i]), {}))
        rec[1][name] = arr.coord[i]
    return out


def chain_template_arrays(query_len: int, entries, struct_dir):
    """Per-residue template arrays for one query chain.

    Returns ``(aatype (T, L) int64, pos (T, L, 4, 3) float32, mask (T, L, 4) float32)`` where
    the four atom slots are N, CA, C, pseudo-beta. An unaligned query residue, a template
    residue missing from the structure, or a residue missing one of those atoms is left as
    gap with a zero mask, which is what makes it contribute nothing.
    """
    three_to_idx = _resname_to_restype()

    T = len(entries)
    aatype = np.full((max(T, 1), query_len), GAP, dtype=np.int64)
    pos = np.zeros((max(T, 1), query_len, 4, 3), dtype=np.float32)
    mask = np.zeros((max(T, 1), query_len, 4), dtype=np.float32)
    for t, (pdb_id, chain_id, idx_map) in enumerate(entries):
        residues = _template_chain_atoms(Path(struct_dir) / f"{pdb_id}.cif", chain_id)
        for q_res, t_res in idx_map:
            q = int(q_res) - 1                       # query residue ids are 1-indexed
            if not (0 <= q < query_len):
                continue
            rec = residues.get(int(t_res))
            if rec is None:
                continue
            res_name, atoms = rec
            pb_name = "CA" if res_name == "GLY" else "CB"
            wanted = FRAME_ATOMS + (pb_name,)
            if not all(a in atoms for a in wanted):
                continue
            aatype[t, q] = three_to_idx.get(res_name, 20)     # UNK for a modified residue
            for k, a in enumerate(wanted):
                pos[t, q, k] = atoms[a]
                mask[t, q, k] = 1.0
    return aatype, pos, mask


def pair_features(aatype, pos, mask):
    """The four pairwise template features from per-token N/CA/C/pseudo-beta arrays.

    ``pos`` is (T, N, 4, 3) and ``mask`` (T, N, 4) over the WHOLE complex token axis, matching
    upstream, which computes these on the merged residue axis rather than per chain.
    """
    T, N = aatype.shape
    lower, upper = _dgram_breaks()
    dgram = np.zeros((T, N, N, _NUM_BINS), dtype=np.float32)
    pb_mask2 = np.zeros((T, N, N), dtype=np.float32)
    unit_vec = np.zeros((T, N, N, 3), dtype=np.float32)
    bb_mask2 = np.zeros((T, N, N), dtype=np.float32)
    eps = 1e-6
    for t in range(T):
        m = mask[t]
        p = pos[t] * m[..., None]
        pb, pbm = p[:, 3], m[:, 3]
        pbm2 = pbm[:, None] * pbm[None, :]
        diff = pb[:, None, :] - pb[None, :, :]
        d2 = np.einsum("ijk,ijk->ij", diff, diff)[..., None]
        dgram[t] = ((d2 > lower) & (d2 < upper)).astype(np.float32) * pbm2[..., None]
        pb_mask2[t] = pbm2

        n_pos, ca_pos, c_pos = p[:, 0], p[:, 1], p[:, 2]
        fm = (m[:, 0] * m[:, 1] * m[:, 2]).astype(np.float32)
        v1, v2 = c_pos - ca_pos, n_pos - ca_pos
        e1 = v1 / (np.sqrt(np.einsum("ij,ij->i", v1, v1))[:, None] + eps)
        e2 = v2 - np.einsum("ij,ij->i", v2, e1)[:, None] * e1
        e2 = e2 / (np.sqrt(np.einsum("ij,ij->i", e2, e2))[:, None] + eps)
        e3 = np.cross(e1, e2)
        R = np.stack([e1, e2, e3], axis=-1)
        d = ca_pos[None, :, :] - ca_pos[:, None, :]
        uv = np.einsum("ilk,ijl->ijk", R, d)
        uv = uv / (np.sqrt(np.einsum("ijk,ijk->ij", uv, uv))[..., None] + eps)
        fm2 = fm[:, None] * fm[None, :]
        unit_vec[t] = uv * fm2[..., None]
        bb_mask2[t] = fm2
    return {
        "template_aatype": torch.from_numpy(aatype),
        "template_distogram": torch.from_numpy(dgram),
        "template_pseudo_beta_mask": torch.from_numpy(pb_mask2),
        "template_unit_vector": torch.from_numpy(unit_vec),
        "template_backbone_frame_mask": torch.from_numpy(bb_mask2),
    }


def complex_template_features(chain_blocks, n_token: int, max_templates: int = MAX_TEMPLATES):
    """Assemble per-chain template arrays onto the complex token axis.

    ``chain_blocks`` is one ``(token_offset, msa_col, aatype, pos, mask)`` per chain that has
    templates, where ``msa_col`` is the chain's token -> query-residue map (-1 for a token
    with no residue column of its own: a ligand atom, a modified residue's atoms). A run with
    no template block at all is left to ``dummy_template_features``.

    The slot fill matches upstream's assembly line, which pads each chain's own template
    stack to ``max_templates`` with zeros: a column with no template in that slot reads gap in
    slot 0 and ALA(0) above it. The masks are zero either way, so geometry contributes
    nothing, but the aatype one-hot is a real input to the embedder and gap is not ALA.
    """
    aatype = np.zeros((max_templates, n_token), dtype=np.int64)
    aatype[0] = GAP
    pos = np.zeros((max_templates, n_token, 4, 3), dtype=np.float32)
    mask = np.zeros((max_templates, n_token, 4), dtype=np.float32)
    for off, msa_col, c_aatype, c_pos, c_mask in chain_blocks:
        cols = np.asarray(msa_col)
        tok = off + np.nonzero(cols >= 0)[0]
        res = cols[cols >= 0]
        t = min(max_templates, c_aatype.shape[0])
        aatype[:t, tok] = c_aatype[:t][:, res]
        aatype[t:, tok] = 0                      # this chain's unused slots are the zero pad
        pos[:t, tok] = c_pos[:t][:, res]
        mask[:t, tok] = c_mask[:t][:, res]
    return pair_features(aatype, pos, mask)
