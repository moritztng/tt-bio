"""DiffDock's receptor embedding: the conventions on CPU, the card's numbers on a card.

Three separate claims, three separate tests:

* the chain splitting, the stripping of ``<cls>``/``<eos>`` and the 1022-residue refusal are
  pure host logic and are checked with a stub encoder, no weights and no device;
* the host reference (HF ``EsmForMaskedLM`` hidden state) runs when the 650M is cached;
* the card's embedding is graded against that reference per residue, and needs a Blackhole
  card plus ``TT_VISIBLE_DEVICES``.

The bar on the last one is stated in the test rather than inherited: ESM-2 650M is a
33-layer encoder and tt-bio runs it with an fp32 stream, so the reference-vs-card PCC is an
accumulation-order question, not a representational one.
"""

import os

import pytest

torch = pytest.importorskip("torch")

from tt_bio.diffdock import esm as dfd_esm
from tt_bio.diffdock.config import ESM2_DIM, ESM2_TRUNCATION

SEQ_A = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
SEQ_B = "GVNDNEEGFFSARGHRPLDKKREEAPSLRPAPPPISGGGYR"


def _stub(seq):
    """A deterministic stand-in: row 0 is <cls>, row L+1 is <eos>, residue i is i+1."""
    rows = torch.arange(len(seq) + 2, dtype=torch.float32).unsqueeze(1)
    return rows.expand(-1, ESM2_DIM).unsqueeze(0).contiguous()


def test_split_chains():
    assert dfd_esm.split_chains(f"{SEQ_A}:{SEQ_B}") == [SEQ_A, SEQ_B]
    assert dfd_esm.split_chains(SEQ_A) == [SEQ_A]


def test_specials_are_stripped_and_chains_stay_in_order():
    out = dfd_esm.receptor_embeddings([SEQ_A, SEQ_B], encode=_stub)
    assert [t.shape for t in out] == [(len(SEQ_A), ESM2_DIM), (len(SEQ_B), ESM2_DIM)]
    # <cls> (value 0) and <eos> (value L+1) are gone; residues keep upstream's 1..L rows.
    assert torch.equal(out[0][:, 0], torch.arange(1, len(SEQ_A) + 1, dtype=torch.float32))
    assert torch.equal(out[1][:, 0], torch.arange(1, len(SEQ_B) + 1, dtype=torch.float32))

    block = dfd_esm.receptor_x_columns([SEQ_A, SEQ_B], encode=_stub)
    assert block.shape == (len(SEQ_A) + len(SEQ_B), ESM2_DIM)


def test_a_chain_past_the_truncation_is_refused_not_truncated():
    with pytest.raises(ValueError, match="truncation"):
        dfd_esm.receptor_embeddings(["A" * (ESM2_TRUNCATION + 1)], encode=_stub)
    dfd_esm.receptor_embeddings(["A" * ESM2_TRUNCATION], encode=_stub)   # the edge is fine


def test_a_wrong_row_count_is_caught():
    with pytest.raises(ValueError, match="rows for a"):
        dfd_esm.receptor_embeddings([SEQ_A], encode=lambda s: _stub(s)[:, :-1])


def _esm2_cached():
    try:
        from huggingface_hub import try_to_load_from_cache

        from tt_bio.diffdock.config import ESM2_MODEL
        from tt_bio.weights import hf_revision
        got = try_to_load_from_cache(ESM2_MODEL, "config.json", revision=hf_revision(ESM2_MODEL))
        return isinstance(got, str)
    except Exception:
        return False


@pytest.mark.skipif(not _esm2_cached(), reason="ESM-2 650M not in the local HF cache")
def test_host_reference_shape():
    out = dfd_esm.receptor_embeddings([SEQ_A], on_device=False)
    assert out[0].shape == (len(SEQ_A), ESM2_DIM)
    assert torch.isfinite(out[0]).all()


@pytest.mark.skipif(not os.environ.get("TT_VISIBLE_DEVICES"),
                    reason="needs a leased Tenstorrent card")
@pytest.mark.skipif(not _esm2_cached(), reason="ESM-2 650M not in the local HF cache")
def test_card_embedding_matches_the_host_reference():
    """Per-residue PCC of the card's ESM-2 against the host's, on two chains.

    Bar: PCC >= 0.999 on every residue row and a relative Frobenius error under 2 %. PCC is
    the check that catches a layout or masking bug. The Frobenius bar is measured, not
    reasoned: the residual stream is fp32 but the projections read bf16, and over 33 blocks
    that lands at 0.97 % (33 aa), 1.27 % (41 aa) and 1.29 % (500 aa) on a Blackhole chip,
    worst residue PCC 0.99986 on the two test chains (a random 500 aa chain, not tested here,
    reaches 0.99894). The card output is bit-identical to the
    encoder as it was inside nesso1_input before the move, so this is what Nesso-1 already
    ships, not a regression.
    """
    ref = dfd_esm.receptor_embeddings([SEQ_A, SEQ_B], on_device=False)
    got = dfd_esm.receptor_embeddings([SEQ_A, SEQ_B], on_device=True)
    for i, (r, g) in enumerate(zip(ref, got)):
        assert r.shape == g.shape
        rel = torch.linalg.norm(g - r) / torch.linalg.norm(r)
        pcc = torch.stack([
            torch.corrcoef(torch.stack([r[j], g[j]]))[0, 1] for j in range(r.shape[0])
        ])
        assert rel < 0.02, f"chain {i}: relative Frobenius {rel:.4f}"
        assert pcc.min() > 0.999, f"chain {i}: worst residue PCC {pcc.min():.6f}"
