# Unit tests for the Protenix-v2 MSA featurizer (tt_bio.protenix_data.protein_msa_features).
# Self-contained (synthetic a3m, no device/checkpoint) -- checks the protenix MSA conventions:
# AF token order, gap=31, lowercase insertions -> deletions, profile = column frequency,
# deletion_value = arctan(d/3)*2/pi, row dedup, and the not-aligned fallback.
import math

import torch

from tt_bio.protenix_data import build_protein_features, protein_msa_features

# query ACDEF; s1 has an insertion ('a') before C; s2 has a gap at col 2; s3 duplicates query
A3M = ">q\nACDEF\n>s1\nAaCDEF\n>s2\nAC-EF\n>s3\nACDEF\n"
QUERY = "ACDEF"  # A=0, C=4, D=3, E=6, F=13 in AF restype order


def test_msa_features_conventions():
    f = protein_msa_features(A3M, QUERY)
    # 4 rows -> 3 after de-duplicating the query-identical s3
    assert f["msa"].shape == (3, 5), f["msa"].shape
    assert f["msa"][0].tolist() == [0, 4, 3, 6, 13]            # query row == query aatype
    assert f["msa"][2].tolist() == [0, 4, 31, 6, 13]           # s2 gap -> token 31
    # insertion before C in s1 -> deletion at column 1 only
    assert f["has_deletion"][1].tolist() == [0, 1, 0, 0, 0]
    assert abs(float(f["deletion_value"][1, 1]) - math.atan(1 / 3) * 2 / math.pi) < 1e-6
    # profile: per-column frequency over 32 classes, sums to 1; col2 is gap in 1/3 of rows
    assert torch.allclose(f["profile"].sum(-1), torch.ones(5), atol=1e-5)
    assert abs(float(f["profile"][2, 31]) - 1 / 3) < 1e-6
    assert abs(float(f["profile"][0, 0]) - 1.0) < 1e-6           # col0 is A in all rows


def test_msa_not_aligned_falls_back():
    # a3m query length (5) != requested query length (7) -> None (caller folds single-seq)
    assert protein_msa_features(A3M, "ACDEFGH") is None


def test_build_features_msa_vs_single():
    single = build_protein_features(QUERY)
    withmsa = build_protein_features(QUERY, a3m=A3M)
    assert single["msa"].shape[0] == 1
    assert withmsa["msa"].shape[0] == 3
    # query row of the MSA equals the single-sequence msa (the query itself)
    assert withmsa["msa"][0].tolist() == single["msa"][0].tolist()


def _rows(n, k=5):
    """n distinct alignment rows over ACDEF, query first."""
    aa, out = "ACDEFGHIKLMNPQRSTVWY", ["ACDEF"]
    i = 0
    while len(out) < n:
        s = "".join(aa[(i // 20 ** j) % 20] for j in range(k))
        i += 1
        if s not in out:
            out.append(s)
    return out


def test_pool_crops_to_upstream_rows_after_the_profile(monkeypatch):
    # Upstream assembles at most MSA_POOL_ROWS rows and reads nothing past them; the profile is
    # taken over the chain's whole alignment first, as upstream takes it.
    import tt_bio.protenix_data as PD
    monkeypatch.setattr(PD, "MSA_POOL_ROWS", 6)
    rows = _rows(9)
    a3m = "".join(f">{i}\n{s}\n" for i, s in enumerate(rows))
    f = build_protein_features(QUERY, a3m=a3m)
    whole = protein_msa_features(a3m, QUERY)
    assert f["msa"].shape[0] == 6
    assert torch.equal(f["msa"], whole["msa"][:6])
    assert torch.equal(f["profile"], whole["profile"])


def test_pool_keeps_half_paired_then_fills_unpaired(monkeypatch):
    import tt_bio.protenix_data as PD
    monkeypatch.setattr(PD, "MSA_POOL_ROWS", 8)
    rows = _rows(12)
    a3m = "".join(f">{i}\n{s}\n" for i, s in enumerate(rows))
    f = PD.build_complex_features([(QUERY, a3m), (QUERY, a3m)], paired_a3ms=[a3m, a3m])
    # 4 paired rows counting the query (which sits in the unpaired block), 4 unpaired: 3 + 5.
    assert f["msa"].shape[0] == 8
    paired_only = f["msa"][:3, :5]
    assert paired_only.tolist() == protein_msa_features(a3m, QUERY)["msa"][1:4].tolist()
