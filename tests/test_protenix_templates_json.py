"""tt_bio.protenix_templates_json: upstream's template pipeline on tt-bio chains. CPU only, no kalign (JSON
templates carry their own alignment) and no network."""
import json

import pytest
import torch

pytest.importorskip("Bio")
from tt_bio.protenix_msa import ChainMSA  # noqa: E402
from tt_bio.protenix_templates_json import hit_featurizer, upstream_template_features  # noqa: E402

_BB = {"N": (-1.46, 0.0, 0.0), "CA": (0.0, 0.0, 0.0), "C": (0.55, 1.42, 0.0), "O": (1.2, 1.7, 1.0),
       "CB": (-0.53, -0.78, -1.2)}


def _mmcif(resnames):
    rows, k = [], 0
    for i, rn in enumerate(resnames, start=1):
        for name, (x, y, z) in _BB.items():
            if rn == "GLY" and name == "CB":
                continue
            k += 1
            rows.append(f"ATOM {k} {name[0]} {name} . {rn} A 1 {i} ? {x + 3.8 * i:.3f} {y:.3f} {z:.3f} 1.00 0.00 "
                        f"{i} {rn} A {name} 1")
    head = ("data_t\nloop_\n_atom_site.group_PDB\n_atom_site.id\n_atom_site.type_symbol\n_atom_site.label_atom_id\n"
            "_atom_site.label_alt_id\n_atom_site.label_comp_id\n_atom_site.label_asym_id\n_atom_site.label_entity_id\n"
            "_atom_site.label_seq_id\n_atom_site.pdbx_PDB_ins_code\n_atom_site.Cartn_x\n_atom_site.Cartn_y\n"
            "_atom_site.Cartn_z\n_atom_site.occupancy\n_atom_site.B_iso_or_equiv\n_atom_site.auth_seq_id\n"
            "_atom_site.auth_comp_id\n_atom_site.auth_asym_id\n_atom_site.auth_atom_id\n_atom_site.pdbx_PDB_model_num\n")
    return head + "\n".join(rows) + "\n"


def test_json_template_reaches_the_aligned_tokens_only(tmp_path):
    q = "MKTAYG"
    tpl = [{"mmcif": _mmcif(["LYS", "THR", "ALA"]), "queryIndices": [1, 2, 3], "templateIndices": [0, 1, 2]}]
    p = tmp_path / "t.json"
    p.write_text(json.dumps(tpl))
    lig = 2
    chains = [ChainMSA("A", 0, "protein", q), ChainMSA("B", 1, "ligand", "", n_atoms=lig)]
    asym = torch.tensor([0] * len(q) + [1] * lig)
    res = torch.cat([torch.arange(1, len(q) + 1), torch.ones(lig, dtype=torch.long)])
    f = upstream_template_features(chains, asym, res, {"A": str(p)},
                                   featurizer=hit_featurizer(str(tmp_path), fetch_remote=False))
    n = len(q) + lig
    assert f["template_aatype"].shape == (4, n) and f["template_distogram"].shape == (4, n, n, 39)
    # template 0 covers query residues 2-4 (K, T, A) and is gap elsewhere; upstream pads slots 1-3 with zeros
    assert f["template_aatype"][0, 1:4].tolist() == [11, 16, 0]
    assert (f["template_aatype"][0, [0, 4, 5, 6, 7]] == 31).all() and (f["template_aatype"][1:] == 0).all()
    assert float(f["template_pseudo_beta_mask"][1:].sum()) == 0
    pb = f["template_pseudo_beta_mask"][0]
    assert pb[1:4, 1:4].all() and pb.sum() == 9
    bb = f["template_backbone_frame_mask"][0]
    assert bb[1:4, 1:4].all() and bb.sum() == 9


def test_no_templates_is_one_gap_row_then_zero_padding():
    """Upstream with no hits: slot 0 is an all-gap template, slots 1-3 are zero padding. tt-bio's
    dummy_template_features puts gap (31) in all four, so the two differ in template_aatype."""
    q = "MKT"
    f = upstream_template_features([ChainMSA("A", 0, "protein", q)], torch.zeros(3, dtype=torch.long),
                                   torch.arange(1, 4), {})
    assert (f["template_aatype"][0] == 31).all() and (f["template_aatype"][1:] == 0).all()
    assert float(f["template_distogram"].abs().sum()) == 0


def test_sidecar_templates_follow_use_template(tmp_path):
    pytest.importorskip("rdkit")
    from tt_bio.protenix_data import build_complex_features
    from tt_bio.protenix_templates_json import features_from_sidecar

    q = "MKTAYG"
    p = tmp_path / "t.json"
    p.write_text(json.dumps([{"mmcif": _mmcif(["LYS", "THR", "ALA"]), "queryIndices": [1, 2, 3],
                              "templateIndices": [0, 1, 2]}]))
    feats = build_complex_features([(q, None, "protein")])
    chains = [("A", q, None, "protein", None)]
    side = {"templates": {"A": str(p)}}
    fz = hit_featurizer(str(tmp_path), fetch_remote=False)
    on = features_from_sidecar(feats, chains, side, True, fz)
    off = features_from_sidecar(feats, chains, side, False, fz)
    assert float(on["template_pseudo_beta_mask"].sum()) == 9 and float(off["template_pseudo_beta_mask"].sum()) == 0
    assert set(on) == {k for k in feats if k.startswith("template_")}
