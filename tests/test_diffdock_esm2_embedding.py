"""The receptor embedding is upstream's, and it is computed once.

Two claims the card test (``test_diffdock_esm2.py::test_card_embedding_matches_the_host
_reference``) does not make, both gradeable without a card:

* **it is upstream's embedding.** Upstream reads fair-esm's ``representations[33]``; tt-bio
  reads HF's last hidden state through its own shared ESM-2. Same weights, same final layer
  norm, two loaders. This grades our host path against upstream's own cached rows for a
  PoseBusters receptor, which is what the card is then graded against in turn.
* **it is computed once.** The encoder is built once per process and each distinct sequence
  is embedded once, so a screen does not reload 650 M parameters per ligand and the
  confidence graph reuses the score graph's rows.
"""

import pytest

torch = pytest.importorskip("torch")

from pathlib import Path

from tt_bio.diffdock import esm as dfd_esm
from tt_bio.diffdock.config import ESM2_DIM

SEQ_A = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
SEQ_B = "GVNDNEEGFFSARGHRPLDKKREEAPSLRPAPPPISGGGYR"

# dfd-ref's reference environment cached upstream's fair-esm embeddings while dumping the
# graph goldens (scripts/diffdock_port/dump_graph_golden.py), one file per complex.
LM_CACHE = Path("/home/moritz/scratch/dfd/data/lm_cache")
SEL = Path("/home/moritz/scratch/dfd/data/posebusters_dl/sel")
COMPLEX = "7BJ6_TVK"


def _counting(width=ESM2_DIM):
    calls = []

    def encode(seq):
        calls.append(seq)
        rows = torch.arange(len(seq) + 2, dtype=torch.float32).unsqueeze(1)
        return rows.expand(-1, width).unsqueeze(0).contiguous()

    return encode, calls


# ------------------------------------------------------------------ it is computed once

def test_memoize_embeds_each_sequence_once():
    encode, calls = _counting()
    cached = dfd_esm.memoize(encode)
    first = cached(SEQ_A)
    assert cached(SEQ_A) is first          # the same tensor back, not a recompute
    cached(SEQ_B)
    cached(SEQ_A)
    assert calls == [SEQ_A, SEQ_B]


def test_the_cache_is_bounded_and_drops_the_oldest_sequence():
    encode, calls = _counting()
    cached = dfd_esm.memoize(encode, keep=2)
    cached(SEQ_A), cached(SEQ_B), cached("MKV")
    assert list(cached.cache) == [SEQ_B, "MKV"]
    cached(SEQ_A)                           # evicted, so this one is recomputed
    assert calls == [SEQ_A, SEQ_B, "MKV", SEQ_A]


def test_the_encoder_is_built_once_and_a_second_run_hits_the_cache(monkeypatch):
    """What a screen pays: one encoder build, one embedding per distinct chain."""
    builds = []
    encode, calls = _counting()

    def fake_cpu_encoder(model_name=None, cache_dir=None):
        builds.append(model_name)
        return encode

    monkeypatch.setattr(dfd_esm, "cpu_encoder", fake_cpu_encoder)
    monkeypatch.setattr(dfd_esm, "_ENCODERS", {})

    for _ in range(3):
        out = dfd_esm.receptor_embeddings([SEQ_A, SEQ_B], on_device=False)
        assert [t.shape for t in out] == [(len(SEQ_A), ESM2_DIM), (len(SEQ_B), ESM2_DIM)]
    assert builds == [dfd_esm.ESM2_MODEL], "the 650M was loaded more than once"
    assert calls == [SEQ_A, SEQ_B], f"the same chain was embedded {len(calls)} times"


def test_the_card_and_the_host_do_not_share_cache_entries(monkeypatch):
    """They agree to ~1.3 % relative Frobenius, not to the bit, so one run must not mix them."""
    host, host_calls = _counting()
    card, card_calls = _counting()
    monkeypatch.setattr(dfd_esm, "cpu_encoder", lambda *a, **k: host)
    monkeypatch.setattr(dfd_esm, "_ENCODERS", {})
    monkeypatch.setitem(dfd_esm._ENCODERS, (dfd_esm.ESM2_MODEL, True, None),
                        dfd_esm.memoize(card))

    dfd_esm.receptor_embeddings([SEQ_A], on_device=False)
    dfd_esm.receptor_embeddings([SEQ_A], on_device=True)
    assert host_calls == [SEQ_A] and card_calls == [SEQ_A]


# ------------------------------------------------------------- it is upstream's embedding

def _have_fixture():
    return (LM_CACHE / f"{COMPLEX}.pt").exists() and (SEL / COMPLEX).exists()


@pytest.mark.skipif(not _have_fixture(),
                    reason=f"missing {LM_CACHE}/{COMPLEX}.pt; regenerate with "
                           "scripts/diffdock_port/dump_graph_golden.py in the reference env")
def test_tt_embedding_matches_fair_esm():
    """tt-bio's ESM-2 rows against upstream's fair-esm rows, per residue.

    Bar: worst-residue PCC >= 0.9999 and relative Frobenius < 1e-4. Not a reasoned bar and
    not a loose one -- these are the same weights through two loaders in float32, so the
    only difference allowed is accumulation order. Measured on %s chain 0 (92 residues):
    worst residue PCC 0.9999998, relative Frobenius 1.4e-6, max absolute 4.3e-6.

    The card is then graded against this host path in test_diffdock_esm2.py, so the two
    tests chain: upstream -> host here, host -> card there.
    """ % COMPLEX
    from tt_bio.diffdock import input as dfd_input

    up = torch.load(LM_CACHE / f"{COMPLEX}.pt", map_location="cpu", weights_only=False)[0]
    chains = dfd_input.chain_sequences_for_embedding(SEL / COMPLEX / f"{COMPLEX}_protein.pdb")
    got = dfd_esm.receptor_embeddings(chains, on_device=False)
    assert len(got) == len(up)
    for j, (u, g) in enumerate(zip(up, got)):
        u = u.float()
        assert u.shape == g.shape
        rel = torch.linalg.norm(g - u) / torch.linalg.norm(u)
        pcc = torch.stack([torch.corrcoef(torch.stack([u[i], g[i]]))[0, 1]
                           for i in range(u.shape[0])])
        assert rel < 1e-4, f"chain {j}: relative Frobenius {rel:.2e}"
        assert pcc.min() > 0.9999, f"chain {j}: worst residue PCC {pcc.min():.7f}"
