"""The conf_bins lever: softmax(pae) reduced to its bin-vector products answers every confidence
reading the host softmax does (pTM / ipTM, the chain reductions, the PAE matrix)."""
import torch

from tt_bio.openfold3_fold import _BinProj, _pae_bin_vectors
from tt_bio.protenix import ConfidenceHead


def _readings(probs, logits, asym, has_frame):
    centers = (torch.arange(64, dtype=torch.float32) + 0.5) * 0.5
    ptm, iptm = ConfidenceHead._ptm_iptm(logits, asym, has_frame=has_frame, probs=probs)
    chain = ConfidenceHead._chain_confidence(logits, asym, has_frame=has_frame, probs=probs)
    return ptm, iptm, chain, probs @ centers


def test_bin_projection_answers_every_reading():
    torch.manual_seed(0)
    n = 70
    asym = torch.tensor([1] * 30 + [2] * 25 + [3] * 15)
    has_frame = torch.ones(n, dtype=torch.bool); has_frame[5] = False
    logits = torch.randn(n, n, 64) * 3
    probs = torch.softmax(logits, -1)
    vecs = _pae_bin_vectors(n, asym)
    proj = probs @ torch.stack(vecs, -1)
    ref = _readings(probs, logits, asym, has_frame)
    got = _readings(_BinProj(proj, vecs), torch.empty(n, n, 64, device="meta"), asym, has_frame)
    assert abs(ref[0] - got[0]) < 1e-5 and abs(ref[1] - got[1]) < 1e-5
    for k in ("chain_ptm", "chain_iptm"):
        assert max(abs(x - y) for x, y in zip(ref[2][k], got[2][k])) < 1e-5
    assert torch.allclose(ref[3], got[3], atol=1e-4)


def test_single_chain_needs_only_the_global_count():
    vecs = _pae_bin_vectors(40, torch.ones(40))
    assert len(vecs) == 2
