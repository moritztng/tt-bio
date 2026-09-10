# Validate the tt_bio.protenix_data token-level featurizer (offline protein case)
# exactly reproduces the v2 reference token features. Gated on the golden feat dict.
import os, pickle
import pytest
import torch

_GOLD = os.path.expanduser("~/protenix_ref_out.pkl")
# Only the two golden-scored tests need the pickle. The charge and OXT contracts below are
# checked against the CCD and against the featurizer's own invariants, so they always run.
# The golden contains no histidine, and a suite that skipped wholesale is how HIS ND1's
# formal charge stayed missing while the tests reported 20/20 residue types.
needs_gold = pytest.mark.skipif(not os.path.exists(_GOLD), reason="needs ~/protenix_ref_out.pkl")


@needs_gold
def test_protein_token_features_exact():
    from tt_bio.protenix_data import protein_token_features, aatype_from_sequence, RESTYPE_ORDER
    ie = pickle.load(open(_GOLD, "rb"))["intermediates"]["input_embedder"]["in"][0]
    aatype = ie["restype"].argmax(-1)
    f = protein_token_features(aatype)
    for k in ["restype", "profile", "deletion_mean", "msa", "has_deletion", "deletion_value",
              "token_bonds", "asym_id", "entity_id", "sym_id", "residue_index", "token_index"]:
        assert f[k].shape == ie[k].shape, f"{k} shape {tuple(f[k].shape)} != {tuple(ie[k].shape)}"
        assert torch.equal(f[k].float(), ie[k].float()), f"{k} mismatch"
    # aatype_from_sequence round-trips the standard restype order
    seq = "".join(RESTYPE_ORDER[i] for i in aatype.tolist() if i < len(RESTYPE_ORDER))
    rt = aatype_from_sequence(seq)
    assert torch.equal(rt, torch.tensor([i for i in aatype.tolist() if i < len(RESTYPE_ORDER)]))


@needs_gold
def test_protein_atom_metadata_exact():
    """Atom-level metadata (element/charge/name/mask/indices) reproduces the v2 reference
    exactly (incl. C-terminal OXT and ARG/LYS formal charges). ref_pos is a stochastic
    conformer (not bit-matched) so it is excluded here."""
    from tt_bio.protenix_data import protein_atom_features, RESTYPE_ORDER
    from tt_bio.data import const
    ie = pickle.load(open(_GOLD, "rb"))["intermediates"]["input_embedder"]["in"][0]
    aatype = ie["restype"].argmax(-1); a2t = ie["atom_to_token_idx"].long(); ref_pos = ie["ref_pos"].float()
    l2r = {v: k for k, v in const.prot_token_to_letter.items()}
    conf = {}
    for t in range(len(aatype)):
        res = l2r[RESTYPE_ORDER[int(aatype[t])]]
        if res not in conf:
            idx = (a2t == t).nonzero().flatten()
            conf[res] = ref_pos[idx][:len(const.ref_atoms[res])].clone()
    f = protein_atom_features(aatype, conf)
    for k in ["ref_element", "ref_charge", "ref_atom_name_chars", "ref_mask",
              "atom_to_token_idx", "ref_space_uid"]:
        assert f[k].shape == ie[k].shape, f"{k} shape {tuple(f[k].shape)} != {tuple(ie[k].shape)}"
        assert torch.equal(f[k].float(), ie[k].float()), f"{k} mismatch"


# ---------------------------------------------------------------------------
# The CCD formal-charge table, over every standard residue instead of the ones a golden
# happens to contain. And OXT, which is per residue rather than always the last one.
# ---------------------------------------------------------------------------

# The PDB chemical component dictionary's `_chem_comp_atom.charge` column, read out of
# components.cif for the 20 standard residues plus UNK and MSE. Exactly three atoms are
# non-zero. The carboxylate side chains (ASP, GLU) and the ionisable HIS NE2 are not among
# them: the CCD's ideal components are the protonated forms, so the charge sits on ARG NH2,
# LYS NZ and HIS ND1 and nowhere else.
_CCD_CHARGED_ATOMS = {("ARG", "NH2"): 1.0, ("LYS", "NZ"): 1.0, ("HIS", "ND1"): 1.0}


def _paf(oxt=None):
    """protein_atom_features over a chain carrying every standard residue once."""
    from tt_bio.protenix_data import (RESTYPE_ORDER, aatype_from_sequence,
                                      load_ref_conformers, protein_atom_features,
                                      restype_to_resname)
    aatype = aatype_from_sequence(RESTYPE_ORDER)
    res_names = restype_to_resname(aatype)
    return protein_atom_features(aatype, load_ref_conformers(), oxt=oxt), res_names


def _atom_names(feats):
    chars = feats["ref_atom_name_chars"].reshape(-1, 4, 64).argmax(-1) + 32
    return ["".join(chr(c) for c in row).strip() for row in chars.tolist()]


def test_formal_charge_table_is_ccd_complete():
    from tt_bio.protenix_data import _FORMAL_CHARGE
    assert _FORMAL_CHARGE == _CCD_CHARGED_ATOMS


def test_ref_charge_covers_every_standard_residue():
    """ref_charge is non-zero at exactly the three CCD-charged atoms on a chain that holds
    all 20 residue types, which is the coverage the golden pickle cannot give."""
    f, res_names = _paf()
    names = _atom_names(f)
    a2t = f["atom_to_token_idx"].tolist()
    charged = {(res_names[a2t[i]], names[i]): float(f["ref_charge"][i])
               for i in range(len(names)) if float(f["ref_charge"][i]) != 0.0}
    assert charged == _CCD_CHARGED_ATOMS


def test_oxt_default_matches_explicit_last_residue():
    """oxt=None, the sequence-input default, is byte-identical to spelling out 'last
    residue only', so no existing caller moves."""
    a, res_names = _paf()
    b, _ = _paf([False] * (len(res_names) - 1) + [True])
    assert a.keys() == b.keys()
    for k in a:
        assert torch.equal(a[k], b[k]), f"{k} moved"


def test_oxt_none_anywhere_drops_the_atom():
    """A chain cropped mid-sequence has no C-terminus, so no residue carries an OXT."""
    from tt_bio.data import const
    _, res_names = _paf()
    heavy = sum(len(const.ref_atoms[r]) for r in res_names)
    assert _paf([False] * len(res_names))[0]["ref_charge"].shape[0] == heavy
    assert _paf()[0]["ref_charge"].shape[0] == heavy + 1


def test_oxt_can_sit_on_any_residue():
    """The OXT rides the residue the structure file says it does, not the last one."""
    from tt_bio.data import const
    _, res_names = _paf()
    f, _ = _paf([i == 3 for i in range(len(res_names))])
    off = sum(len(const.ref_atoms[r]) for r in res_names[:4])   # end of residue 3's block
    assert _atom_names(f)[off] == "OXT"
    assert f["atom_to_token_idx"][off].item() == 3


def test_build_complex_features_oxt_is_per_chain():
    """build_complex_features routes one oxt list per chain, and oxt=None reproduces the
    old behaviour on every chain."""
    from tt_bio.protenix_data import build_complex_features
    chains = [("ARNDH", None, "protein"), ("KHIL", None, "protein")]
    base = build_complex_features(chains, chain_ids=["A", "B"])
    same = build_complex_features(chains, chain_ids=["A", "B"],
                                  oxt=[[False] * 4 + [True], [False] * 3 + [True]])
    for k, v in base.items():
        if isinstance(v, torch.Tensor):
            assert torch.equal(v.float(), same[k].float()), f"{k} moved"
    cropped = build_complex_features(chains, chain_ids=["A", "B"],
                                     oxt=[[False] * 5, [False] * 3 + [True]])
    assert cropped["ref_charge"].shape[0] == base["ref_charge"].shape[0] - 1


# --- `modifications:` ---------------------------------------------------------------------
# A modified residue is tokenized PER ATOM from its CCD component (AF3 SI 2.6), which is why
# a chain with one gains tokens but not residues. Before this, build_complex_features had no
# argument for the key at all and every Protenix/OpenDDE fold quietly returned the standard
# residue.

_MSEQ = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK"      # residue 13 is SER
_SEP = [{"position": 13, "ccd": "SEP"}]


def _feats(mods=None):
    from tt_bio.protenix_data import build_complex_features
    return build_complex_features([(_MSEQ, None, "protein")], chain_ids=["A"],
                                  modifications=None if mods is None else [mods])


def test_no_modification_is_byte_identical_to_the_per_residue_path():
    """The control. Every existing fold goes through the new code path, so it has to be a
    no-op when the key is absent."""
    plain, explicit_none = _feats(), _feats(None)
    for k in plain:
        assert torch.equal(plain[k], explicit_none[k]), k


def test_a_modified_residue_becomes_atom_tokens_in_one_residue_frame():
    plain, mod = _feats(), _feats(_SEP)
    # SER 6 heavy atoms -> SEP 10 (OXT is a leaving atom mid-chain): +4 atoms, +9 tokens
    assert plain["ref_pos"].shape[0] == 418 and mod["ref_pos"].shape[0] == 422
    assert plain["restype"].shape[0] == 53 and mod["restype"].shape[0] == 62
    assert int(plain["ref_space_uid"].max()) == int(mod["ref_space_uid"].max()) == 52
    assert mod["residue_index"].tolist().count(13) == 10
    assert (mod["restype"].argmax(-1)[mod["residue_index"] == 13] == 20).all()  # UNK per atom


def test_the_modified_residue_carries_the_ccd_element_and_the_backbone_link():
    plain, mod = _feats(), _feats(_SEP)
    z = lambda f: (f["ref_element"].argmax(-1) + 1)
    assert (z(plain) == 15).sum() == 0 and (z(mod) == 15).sum() == 1   # the SEP phosphorus
    assert (z(mod) == 8).sum() - (z(plain) == 8).sum() == 3            # O1P, O2P, O3P
    names = ["".join(chr(c + 32) for c in ch).strip()
             for ch in mod["ref_atom_name_chars"].argmax(-1).tolist()]
    a2t = mod["atom_to_token_idx"].tolist()
    tok = {names[i]: a2t[i] for i in range(len(a2t)) if int(mod["residue_index"][a2t[i]]) == 13}
    assert "OXT" not in tok, "a mid-chain residue kept the leaving carboxylate oxygen"
    tb = mod["token_bonds"]
    assert tb[tok["N"] - 1, tok["N"]] == 1.0, "no bond to the preceding residue"
    assert tb[tok["C"], max(tok.values()) + 1] == 1.0, "no bond to the following residue"
    assert tb[tok["N"] - 1, max(tok.values()) + 1] == 0.0, "residues 12 and 14 bonded directly"


def test_the_alignment_follows_the_tokens():
    """The MSA is per residue and the chain is now per token in one place, so the columns
    have to be re-placed or every column after the modification reads the wrong residue."""
    from tt_bio.protenix_data import MSA_GAP_IDX, build_complex_features
    a3m = ">q\n" + _MSEQ + "\n>h\n" + _MSEQ.replace("K", "R") + "\n"
    plain = build_complex_features([(_MSEQ, a3m, "protein")], chain_ids=["A"])
    mod = build_complex_features([(_MSEQ, a3m, "protein")], chain_ids=["A"],
                                 modifications=[_SEP])
    atom_cols = (mod["residue_index"] == 13).nonzero().flatten().tolist()
    keep = [t for t in range(mod["msa"].shape[1]) if t not in atom_cols]
    assert torch.equal(mod["msa"][:, keep],
                       plain["msa"][:, [t for t in range(53) if t != 12]])
    assert (mod["msa"][0, atom_cols] == 20).all()            # query row: the token's UNK
    assert (mod["msa"][1:, atom_cols] == MSA_GAP_IDX).all()  # every other row: gap


def test_an_unknown_ccd_code_is_named_not_guessed():
    with pytest.raises(ValueError, match="not a CCD component"):
        _feats([{"position": 13, "ccd": "ZZZZ"}])


def test_a_bond_constraint_onto_a_modified_residue_resolves_by_atom_name():
    from tt_bio.protenix_data import build_complex_features
    f = build_complex_features([(_MSEQ, None, "protein")], chain_ids=["A"],
                               modifications=[_SEP],
                               bonds=[(("A", 13, "O3P"), ("A", 40, "CA"))])
    assert f["token_bonds"].sum() > 0
    with pytest.raises(ValueError, match="modified residue"):
        build_complex_features([(_MSEQ, None, "protein")], chain_ids=["A"],
                               modifications=[_SEP],
                               bonds=[(("A", 13, "NOPE"), ("A", 40, "CA"))])
