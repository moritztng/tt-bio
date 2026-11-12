"""Mol-1-constraint: the folded token-bond projection reproduces upstream's three encoders, inputs parse by
upstream's rules, and conditions land on the right tokens. Card-free."""
import pytest
import torch

from tt_bio import mol1_constraint as mc

CZ = 16


def _conditioner(gen):
    r = lambda *s: torch.randn(*s, generator=gen, dtype=torch.float64)
    sd = {"linear_no_bias_token_bond.weight": r(CZ, 1)}
    for enc in ("", "exclusion."):
        sd[mc.PREFIX + enc + "pair_orientation_embedding"] = r(2, CZ)
        sd[mc.PREFIX + enc + "output_projection.weight"] = r(CZ, CZ)
    sd[mc.PREFIX + "contact.contact_embedding"] = r(CZ)
    sd[mc.PREFIX + "contact.output_projection.weight"] = r(CZ, CZ)
    return sd


def _upstream(sd, n, site, excl, pairs):
    """Upstream mol/constraint/encoders.py forward, float64: (residue mask, partner mask) per site/exclusion."""
    out = torch.zeros(n, n, CZ, dtype=torch.float64)
    for enc, roles in (("", site), ("exclusion.", excl)):
        if roles is None:
            continue
        res, partner = (x.double() for x in roles)
        e = sd[mc.PREFIX + enc + "pair_orientation_embedding"]
        upd = ((partner[:, None] * res[None, :])[..., None] * e[0]
               + (res[:, None] * partner[None, :])[..., None] * e[1])
        out += upd @ sd[mc.PREFIX + enc + "output_projection.weight"].T
    v = sd[mc.PREFIX + "contact.output_projection.weight"] @ sd[mc.PREFIX + "contact.contact_embedding"]
    for i, j in pairs:
        out[i, j] += v
        out[j, i] += v
    return out


def test_fold_reproduces_upstream_encoders():
    gen = torch.Generator().manual_seed(0)
    sd = _conditioner(gen)
    ref_sd = {k: v.clone() for k, v in sd.items()}
    assert mc.fold_weights(sd)
    assert not any(k.startswith(mc.PREFIX) for k in sd)
    w = sd["linear_no_bias_token_bond.weight"]
    assert w.shape == (CZ, 1 + mc.CHANNELS)
    assert torch.equal(w[:, :1], ref_sd["linear_no_bias_token_bond.weight"])
    n = 12
    site = (torch.zeros(n, dtype=torch.bool), torch.zeros(n, dtype=torch.bool))
    site[0][[1, 3]] = True
    site[1][8:] = True
    excl = (torch.zeros(n, dtype=torch.bool), torch.zeros(n, dtype=torch.bool))
    excl[0][5] = True
    excl[1][8:] = True
    pairs = [(2, 9), (0, 4)]
    ch = torch.zeros(n, n, mc.CHANNELS, dtype=torch.float64)
    for k, (res, partner) in ((0, site), (2, excl)):
        ch[..., k] = (partner[:, None] & res[None, :]).double()
        ch[..., k + 1] = (res[:, None] & partner[None, :]).double()
    for i, j in pairs:
        ch[i, j, 4] = ch[j, i, 4] = 1
    got = ch @ w[:, 1:].T
    torch.testing.assert_close(got, _upstream(ref_sd, n, site, excl, pairs), rtol=1e-12, atol=1e-12)


def test_base_checkpoint_untouched():
    sd = {"linear_no_bias_token_bond.weight": torch.ones(CZ, 1)}
    assert not mc.fold_weights(sd)
    assert sd["linear_no_bias_token_bond.weight"].shape == (CZ, 1)


def test_parse_normalizes_like_upstream():
    got = mc.parse({"site": {"source_chains": ["A"], "partner_chains": ["B"],
                             "residue_ids": [["A", 12], ["A", 8], ["A", 12]]},
                    "contact": {"pairs": [[["B", 15], ["A", 8]], [["A", 8], ["B", 15]]]},
                    "exclusion": {"source_chains": ["A"], "partner_chains": ["B"], "residue_ids": []}})
    assert got == {"site": {"source_chains": ["A"], "partner_chains": ["B"], "residue_ids": [["A", 8], ["A", 12]]},
                   "contact": {"pairs": [[["A", 8], ["B", 15]]]}}
    assert mc.parse(None) is None and mc.parse({}) is None
    assert mc.parse({"site": {"residue_ids": []}}) is None


@pytest.mark.parametrize("bad", [
    {"pocket": {}},
    {"site": {"source_chains": ["A"], "partner_chains": ["B"]}},
    {"site": {"source_chains": ["A"], "partner_chains": ["A"], "residue_ids": [["A", 1]]}},
    {"site": {"source_chains": ["A"], "partner_chains": ["B"], "residue_ids": [["C", 1]]}},
    {"site": {"source_chains": ["A"], "partner_chains": ["B"], "residue_ids": [["A", 0]]}},
    {"contact": {"pairs": [], "token_pairs": []}},
    {"contact": {"token_pairs": [[3, 3]]}},
    {"site": {"source_chains": ["A"], "partner_chains": ["B", "C"], "residue_ids": [["A", 2]]},
     "exclusion": {"source_chains": ["A"], "partner_chains": ["C"], "residue_ids": [["A", 2]]}},
])
def test_parse_rejects(bad):
    with pytest.raises(ValueError):
        mc.parse(bad)


def test_site_and_exclusion_on_different_partners_allowed():
    assert mc.parse({"site": {"source_chains": ["A"], "partner_chains": ["B"], "residue_ids": [["A", 2]]},
                     "exclusion": {"source_chains": ["A"], "partner_chains": ["C"], "residue_ids": [["A", 2]]}})


# chain A protein (4 res), chain B protein with residue 2 modified into 3 atom tokens, chain L a 2-atom ligand
PLACE = {"A": {"start": 0, "mt": "protein", "n": 4, "tokens": 4, "res_tok": [0, 1, 2, 3], "res_atoms": {}},
         "B": {"start": 4, "mt": "protein", "n": 3, "tokens": 5, "res_tok": [0, 1, 4], "res_atoms": {2: {"N": 1, "CA": 2, "C": 3}}},
         "L": {"start": 9, "mt": "ligand", "n": 1, "tokens": 2, "res_tok": None, "res_atoms": {}}}


def test_pair_channels_place_tokens():
    cons = mc.parse({"site": {"source_chains": ["A"], "partner_chains": ["L"], "residue_ids": [["A", 2]]},
                     "exclusion": {"source_chains": ["B"], "partner_chains": ["L"], "residue_ids": [["B", 2]]},
                     "contact": {"pairs": [[["A", 4], ["B", 3]]]}})
    ch = mc.pair_channels(cons, PLACE, 11)
    assert ch.shape == (11, 11, mc.CHANNELS)
    assert ch[..., 0].nonzero().tolist() == [[9, 1], [10, 1]]
    assert ch[..., 1].nonzero().tolist() == [[1, 9], [1, 10]]
    assert ch[..., 2].nonzero().tolist() == [[i, j] for i in (9, 10) for j in (5, 6, 7)]
    assert torch.equal(ch[..., 3], ch[..., 2].T)
    assert ch[..., 4].nonzero().tolist() == [[3, 8], [8, 3]]


@pytest.mark.parametrize("cons", [
    {"site": {"source_chains": ["L"], "partner_chains": ["A"], "residue_ids": [["L", 1]]}},  # not protein
    {"site": {"source_chains": ["A"], "partner_chains": ["Z"], "residue_ids": [["A", 1]]}},  # missing chain
    {"site": {"source_chains": ["A"], "partner_chains": ["B"], "residue_ids": [["A", 5]]}},  # past the end
    {"contact": {"pairs": [[["A", 1], ["B", 2]]]}},                                           # modified: several tokens
    {"contact": {"token_pairs": [[0, 11]]}},                                                  # off the token axis
    {"exclusion": {"source_chains": ["A"], "partner_chains": ["L"], "residue_ids": [["A", 1]]},
     "contact": {"token_pairs": [[0, 9]]}},                                                   # inside the exclusion
])
def test_pair_channels_rejects(cons):
    with pytest.raises(ValueError):
        mc.pair_channels(mc.parse(cons), PLACE, 11)


def test_upstream_json_reaches_the_yaml_reader(tmp_path):
    import json
    import yaml
    from tt_bio import protenix_json
    from tt_bio.main import _read_bio_constraints, _read_mol1_constraints
    cons = {"site": {"source_chains": ["A"], "partner_chains": ["B"], "residue_ids": [["A", 3], ["A", 1]]},
            "contact": {"token_pairs": [[0, 9]]}}
    src = tmp_path / "in.json"
    src.write_text(json.dumps([{"name": "x", "constraints": cons, "sequences": [
        {"proteinChain": {"sequence": "MKVLA", "count": 1}}, {"ligand": {"ligand": "CCD_ATP", "count": 1}}]}]))
    doc = protenix_json.read_jobs(src)[0].doc
    yml = tmp_path / "x.yaml"
    yml.write_text(yaml.safe_dump(doc))
    assert _read_mol1_constraints(yml) == mc.parse(cons)
    assert _read_bio_constraints(yml) == []


def test_boltz_contact_still_refused(tmp_path):
    import click
    from tt_bio.main import _read_bio_constraints, _read_mol1_constraints
    yml = tmp_path / "x.yaml"
    yml.write_text("sequences: []\nconstraints:\n- contact: {token1: [A, 1], token2: [B, 2], max_distance: 6}\n")
    assert _read_mol1_constraints(yml) is None
    with pytest.raises(click.ClickException):
        _read_bio_constraints(yml)
