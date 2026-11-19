"""Regenerate <case>.npz: upstream OpenDDE 1.2.0's own inference features for every case in cases.py.

Run with the upstream package importable and its common/ data under $OPENDDE_ROOT_DIR:
    OPENDDE_ROOT_DIR=... python tests/data/opendde_featurizer/make_fixtures.py
Templates off and no MSA, as `tt-bio predict --model opendde` folds an input without them."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import cases  # noqa: E402
from opendde.data.inference.infer_dataloader import dict_to_tensor  # noqa: E402
from opendde.data.inference.json_to_feature import SampleDictToFeatures  # noqa: E402
from opendde.data.template.template_featurizer import InferenceTemplateFeaturizer  # noqa: E402

KEYS = ["parent_residue_idx", "subtoken_role_id", "twin_token_idx", "prev_parent_residue_idx",
        "next_parent_residue_idx", "atom_to_structural_token_idx", "atom_to_structural_tokatom_idx",
        "template_aatype", "template_distogram", "template_pseudo_beta_mask", "template_unit_vector",
        "template_backbone_frame_mask", "residue_index", "asym_id", "entity_id", "sym_id", "restype",
        "token_bonds", "ref_mask", "ref_element", "ref_charge", "ref_atom_name_chars", "ref_space_uid",
        "atom_to_token_idx", "ref_pos"]

for name, (seqs, _) in cases.CASES.items():
    sample = {"name": name, "sequences": seqs}
    if name in cases.COVALENT_BONDS:
        sample["covalent_bonds"] = cases.COVALENT_BONDS[name]
    feats, atoms, _ = SampleDictToFeatures(sample).get_feature_dict()
    feats.update(dict_to_tensor(InferenceTemplateFeaturizer.make_template_feature(
        bioassembly=seqs, atom_array=atoms, use_template=False)))
    out = {k: feats[k].numpy() for k in KEYS}
    for k in ("restype", "ref_element", "ref_atom_name_chars"):   # one-hots, stored as indices
        out[k] = out[k].argmax(-1).astype(np.int16)
    np.savez_compressed(HERE / f"{name}.npz", **out)
    print(name, {k: v.shape for k, v in out.items() if k.startswith("template_a") or k == "ref_pos"})
