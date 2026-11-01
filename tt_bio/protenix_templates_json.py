"""Template features for an upstream Protenix / Mol-1 JSON input, built exactly as upstream builds them.

``upstream_template_features`` is ``mol.data.template.template_featurizer.InferenceTemplateFeaturizer
.make_template_feature`` with the AtomArray replaced by what tt-bio's featurizer already knows (per-token asym index
and residue number, per-chain atom count), as ``tt_bio.protenix_msa`` does for the MSA. Hit parsing (.a3m from
hmmsearch, .hhr, inline JSON mmCIF), the release-date filter (2021-09-30), the obsolete-entry map, kalign
realignment and the feature assembly are the vendored upstream code in ``tt_bio._vendor.mol1``.

Upstream only reads templates under ``--use_template`` (default off); the caller decides that.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from tt_bio import weights
from tt_bio._vendor.mol1.constants_full import DNA_CHAIN, LIGAND_CHAIN_TYPES, PROTEIN_CHAIN, RNA_CHAIN
from tt_bio._vendor.mol1.helpers import load_json_cached
from tt_bio._vendor.mol1.msa_utils import map_to_standard
from tt_bio._vendor.mol1.template_featurizer import (HHRParser, HmmsearchA3MParser, TemplateFeatureAssemblyLine,
                                                     TemplateHitFeaturizer)

_CTYPE = {"protein": PROTEIN_CHAIN, "rna": RNA_CHAIN, "dna": DNA_CHAIN}
MAX_TEMPLATE_DATE = "2021-09-30"      # upstream InferenceDataset
MODEL_KEYS = ("template_aatype", "template_distogram", "template_pseudo_beta_mask", "template_unit_vector",
              "template_backbone_frame_mask")   # what the Protenix template embedder reads


def hit_featurizer(mol_root: str | None = None, fetch_remote: bool = True) -> TemplateHitFeaturizer:
    """Upstream's inference ``TemplateHitFeaturizer``: 4 hits, 20 candidates, date cut-off 2021-09-30, the release
    dates and obsolete map from ``<mol_root>/common``, mmCIFs from ``<mol_root>/mmcif`` (fetched from PDBe on
    demand when missing, upstream's default). A ``<mol_root>`` without upstream's ``common/`` files gets tt-bio's
    copies (weights rows ``mol1-release-dates``, ``mol1-obsolete``), downloaded on first use unless offline."""
    root = Path(mol_root or os.environ.get("MOL_ROOT_DIR") or weights.cache_root() / "mol1").expanduser()

    def common(key: str) -> str:
        p = root / weights.ARTIFACTS[key].filename
        return str(p if p.exists() or not fetch_remote else weights.fetch(key, quiet=True))

    return TemplateHitFeaturizer(
        mmcif_dir=str(root / "mmcif"), template_cache_dir="", max_hits=4, max_template_date=MAX_TEMPLATE_DATE,
        release_dates_path=common("mol1-release-dates"), obsolete_pdbs_path=common("mol1-obsolete"),
        _shuffle_top_k_prefiltered=None, _max_template_candidates_num=20, fetch_remote=fetch_remote)


def chain_templates(t_path: str, seq: str, featurizer: TemplateHitFeaturizer) -> list:
    """The template feature dicts upstream builds for one protein chain from its ``templatesPath``."""
    if t_path.endswith(".json"):
        return featurizer.parse_json_templates(load_json_cached(t_path), seq).features
    content = Path(t_path).read_text()
    if t_path.endswith(".hhr"):
        hits = HHRParser.parse(hhr_string=content)
    elif t_path.endswith(".a3m"):
        hits = HmmsearchA3MParser.parse(query_seq=seq, a3m_str=content, skip_first=False)
    else:
        raise ValueError(f"Unsupported template format: {t_path}")
    if hits and not featurizer._release_dates:
        # upstream drops every hit as "date unknown" and folds template-free without a word
        raise RuntimeError(f"{t_path}: no PDB release dates loaded, so every template hit would be dropped; "
                           "put upstream's common/release_date_cache.json under $MOL_ROOT_DIR or allow the download")
    result, _ = featurizer.get_templates(sequence_uid=seq, query_sequence=seq, hits=hits, max_template_date=None)
    return result.features


def upstream_template_features(chains, token_asym: torch.Tensor, token_res: torch.Tensor, templates: dict,
                               featurizer: TemplateHitFeaturizer | None = None) -> dict:
    """Per-token template features for the whole complex.

    ``chains`` is ``tt_bio.protenix_msa.ChainMSA``-like (chain_id, entity, mol_type, sequence, n_atoms) in token
    order; ``templates`` maps a protein chain id to its ``templatesPath``. Copies of one entity share their hits, as
    upstream builds them once per entity."""
    per_entity, meta = {}, {}
    for aid, c in enumerate(chains):
        ctype = _CTYPE.get(c.mol_type, LIGAND_CHAIN_TYPES)
        seq = "X" * c.n_atoms if ctype == LIGAND_CHAIN_TYPES else c.sequence
        t_path = templates.get(c.chain_id, "")
        if t_path and ctype != PROTEIN_CHAIN:
            raise ValueError(f"chain {c.chain_id}: only protein templates are supported")
        if c.entity not in per_entity:
            if t_path:
                featurizer = featurizer or hit_featurizer()
                per_entity[c.entity] = chain_templates(t_path, seq, featurizer)
            else:
                per_entity[c.entity] = []
        meta[aid] = {"entity_id": c.entity, "chain_id": c.chain_id, "sequence": seq,
                     "chain_entity_type": ctype, "templates": per_entity[c.entity]}
    std = map_to_standard(np.asarray(token_asym), np.asarray(token_res), meta)
    f = TemplateFeatureAssemblyLine(max_templates=4).assemble(meta, std).as_mol_dict()
    out = {}
    for k in MODEL_KEYS:
        t = torch.from_numpy(np.asarray(f[k]))
        out[k] = t.long() if k == "template_aatype" else t.float()
    return out


def features_from_sidecar(feats: dict, chains: list, side: dict, use_template: bool,
                          featurizer: TemplateHitFeaturizer | None = None) -> dict:
    """Template features for an input staged from upstream JSON: the job's ``templatesPath`` hits under
    ``use_template``, otherwise upstream's no-template features (which are not tt-bio's dummy ones: upstream pads
    slots 1-3 with aatype 0, not gap)."""
    from tt_bio.protenix_msa import ChainMSA

    asym, res, ent = feats["asym_id"].long(), feats["residue_index"].long(), feats["entity_id"].long()
    atom_asym = asym[feats["atom_to_token_idx"].long()]
    cm = [ChainMSA(cid, int(ent[asym == a][0]), mt, seq, n_atoms=int((atom_asym == a).sum()))
          for a, (cid, seq, _spec, mt, _mods) in enumerate(chains)]
    return upstream_template_features(cm, asym, res, side.get("templates", {}) if use_template else {}, featurizer)
