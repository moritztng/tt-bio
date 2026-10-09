"""PaeBins: the reduced PAE form every pTM / ipTM / chain key reads, against the logits form."""
import torch

from tt_bio.protenix import ConfidenceHead, PaeBins, tm_per_bin

CENTERS = (torch.arange(64, dtype=torch.float32) + 0.5) * 0.5


def _case(sizes, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = sum(sizes)
    logits = torch.randn(n, n, 64, generator=g) * 3
    asym = torch.cat([torch.full((k,), i + 1) for i, k in enumerate(sizes)])
    return logits, asym


def _reduced(logits, asym):
    """What PaeBins.on_device hands back: only the columns PaeBins.counts asked for, no probs."""
    probs = torch.softmax(logits, -1)
    counts = PaeBins.counts(asym, logits.shape[0])
    return PaeBins(CENTERS, pae=probs @ CENTERS,
                   etm={n: probs @ tm_per_bin(n, CENTERS) for n in counts})


def test_reduced_bins_give_the_logits_keys():
    for sizes in ([40], [30, 12], [25, 9, 30], [5, 5, 50, 7]):
        logits, asym = _case(sizes)
        bins = _reduced(logits, asym)
        frame = torch.ones(logits.shape[0], dtype=torch.bool)
        frame[3] = False
        for hf in (None, frame):
            assert (ConfidenceHead._ptm_iptm(logits, asym, has_frame=hf)
                    == ConfidenceHead._ptm_iptm(None, asym, has_frame=hf, bins=bins))
            assert (ConfidenceHead._chain_confidence(logits, asym, has_frame=hf)
                    == ConfidenceHead._chain_confidence(None, asym, has_frame=hf, bins=bins))


def test_counts_cover_the_global_chain_and_pair_normalisations():
    _, asym = _case([30, 12, 4])
    assert PaeBins.counts(asym, 46) == [19, 30, 34, 42, 46]
    assert PaeBins.counts(None, 10) == [19]


def test_a_count_not_reduced_is_an_error_not_a_guess():
    logits, asym = _case([30, 12])
    bins = _reduced(logits, asym)
    try:
        bins.etm(25)
    except KeyError:
        return
    raise AssertionError("an unreduced count must raise")
