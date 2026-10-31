"""tt_bio.protenix_msa: upstream's MSA assembly on tt-bio chains. CPU only. Exactness vs upstream's own featurizer
is pinned by tests/test_mol1_features.py once the feature goldens exist; this pins the adapter's shape rules."""
import torch

from tt_bio.protenix_msa import ChainMSA, upstream_msa_features

A = "MKTAYIAKQR"
B = "GSHMLLE"


def test_heterodimer_pairs_and_ligand_gets_query_row():
    pa = f">101\n{A}\n>UniRef100_X_9606/\n{A[:-1]}W\n"
    pb = f">101\n{B}\n>UniRef100_Y_9606/\n{B[:-1]}K\n"
    chains = [ChainMSA("A", 0, "protein", A, paired=pa, unpaired=f">q\n{A}\n>u\nMKTAYIAKQa-\n"),
              ChainMSA("B", 1, "protein", B, paired=pb),
              ChainMSA("C", 2, "ligand", "", n_atoms=3)]
    asym = torch.tensor([0] * len(A) + [1] * len(B) + [2] * 3)
    res = torch.cat([torch.arange(1, len(A) + 1), torch.arange(1, len(B) + 1), torch.ones(3, dtype=torch.long)])
    f = upstream_msa_features(chains, asym, res)
    n_tok = len(A) + len(B) + 3
    assert f["msa"].shape[1] == n_tok and f["profile"].shape == (n_tok, 32) and f["deletion_mean"].shape == (n_tok,)
    # row 0 is the query everywhere; the paired homolog row carries both chains' hits
    assert f["msa"][0, len(A) - 1] == 1 and f["msa"][1, len(A) - 1] == 17           # R query, W hit (AF order)
    assert f["msa"][1, len(A) + len(B) - 1] == 11                                  # K hit on chain B, same row
    lig = f["msa"][:, len(A) + len(B):]
    assert (lig == 20).all()          # ligand tokens: UNK query broadcast into otherwise all-gap rows


def test_monomer_without_alignment_is_query_rows():
    # upstream keeps the (query-only) paired block ahead of the unpaired one even without pairing: 2 query rows
    f = upstream_msa_features([ChainMSA("A", 0, "protein", A)], torch.zeros(len(A), dtype=torch.long),
                              torch.arange(1, len(A) + 1))
    assert f["msa"].shape == (2, len(A)) and (f["msa"][0] == f["msa"][1]).all()
    assert float(f["has_deletion"].sum()) == 0


def test_sidecar_features_on_built_complex():
    import os

    import pytest

    from tt_bio.protenix_data import build_complex_features
    from tt_bio.protenix_msa import features_from_sidecar

    mols = os.path.expanduser("~/.boltz/mols")
    if not os.path.isdir(mols):
        pytest.skip("CCD mols dir missing")
    chains = [("A", A, None, "protein", None), ("B", A, None, "protein", None), ("C", "CCD_NAG", None, "ligand", None)]
    feats = build_complex_features([(s, None, mt) for _c, s, _p, mt, _m in chains], mol_dir=mols)
    u = f">q\n{A}\n>h1\nMKTAYIAKQW\n"
    side = {"paired_msa": {}, "unpaired_msa": {"A": u, "B": u}, "templates": {}, "seeds": None}
    f = features_from_sidecar(feats, chains, side)
    n_tok = feats["restype"].shape[0]
    assert f["msa"].shape[1] == n_tok and f["profile"].shape == (n_tok, 32)
    assert (f["msa"][:, 2 * len(A):] == 20).all()          # 14 NAG atom tokens: UNK query row broadcast
