"""DiffDock's receptor embedding, on the card, from tt-bio's existing ESM-2.

DiffDock conditions every receptor residue on the final hidden state of
``esm2_t33_650M_UR50D`` (``utils/inference_utils.py``), 1280 columns concatenated onto the
residue features. tt-bio already runs that checkpoint on Tenstorrent for Nesso-1, so this
module is an adapter, not an implementation: it calls :func:`tt_bio.saprot.esm2_encoder` and
applies DiffDock's three conventions on top.

1. **One call per chain.** Upstream splits a protein on ``:`` and embeds each chain on its
   own, then concatenates in chain order. A chain is not conditioned on its neighbours.
2. **Specials stripped.** Upstream keeps ``representations[33][i, 1:len+1]``, so the ``<cls>``
   and ``<eos>`` rows are dropped and the result has exactly one row per residue.
3. **1022 residues per chain.** ``compute_ESM_embeddings`` passes
   ``truncation_seq_length=1022`` to the batch converter. Past that, upstream's embedding is
   shorter than the receptor's residue count and the concatenation that builds ``receptor.x``
   raises a shape error several functions later. We raise here instead, with the chain named.

fair-esm's ``representations[33]`` is taken **after** ``emb_layer_norm_after`` (ESM2.forward
applies the final layer norm before recording the last layer), which is the same tensor HF's
``EsmModel`` returns as its last hidden state and the same one ``tt_bio.saprot`` returns as
``emb``. The three agree by construction; the parity gate measures how closely.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Iterable

import torch

from tt_bio.diffdock.config import ESM2_DIM, ESM2_MODEL, ESM2_TRUNCATION

#: distinct sequences kept per encoder; a dock needs one receptor, a screen a few.
CACHED_SEQUENCES = 8

_ENCODERS: dict[tuple, Callable[[str], torch.Tensor]] = {}


def split_chains(sequence: str) -> list[str]:
    """Upstream's ``protein_sequence.split(':')``, one entry per chain."""
    return sequence.split(":")


def _check_length(seq: str, label: str) -> None:
    if len(seq) > ESM2_TRUNCATION:
        raise ValueError(
            f"{label}: {len(seq)} residues, over ESM-2's {ESM2_TRUNCATION}-residue truncation. "
            f"Upstream DiffDock truncates the embedding here and then fails when it "
            f"concatenates {len(seq)} residue features with {ESM2_TRUNCATION} embedding rows; "
            f"split the chain or drop the receptor rather than embedding a prefix.")


def cpu_encoder(model_name: str = ESM2_MODEL, cache_dir=None) -> Callable[[str], torch.Tensor]:
    """The host reference: HF ``EsmModel``'s last hidden state, ``[1, L+2, 1280]``.

    This is what the card's output is graded against when no GPU reference is at hand, and
    what runs when tt-bio is asked for DiffDock embeddings with no device available.
    """
    from transformers import AutoTokenizer, EsmForMaskedLM

    from tt_bio.weights import hf_revision

    rev = hf_revision(model_name)
    kw = dict(revision=rev, cache_dir=None if cache_dir is None else str(cache_dir))
    tok = AutoTokenizer.from_pretrained(model_name, **kw)
    model = EsmForMaskedLM.from_pretrained(model_name, **kw).eval()

    def encode(seq: str) -> torch.Tensor:
        with torch.no_grad():
            out = model(tok(seq, return_tensors="pt")["input_ids"], output_hidden_states=True)
        return out.hidden_states[-1].float().contiguous()

    return encode


def memoize(encode: Callable[[str], torch.Tensor], *, keep: int = CACHED_SEQUENCES):
    """The same encoder with each distinct sequence embedded once, newest *keep* kept.

    A screen docks thousands of ligands against a handful of receptors, and the confidence
    graph needs the same rows the score graph did, so the embedding is the one part of a dock
    that is recomputed for nothing. The bound matters: a chain is ``L x 1280`` float32, about
    1 MB per 200 residues, and a screen over hundreds of distinct receptors would otherwise
    grow without end.
    """
    store: "OrderedDict[str, torch.Tensor]" = OrderedDict()

    def cached(seq: str) -> torch.Tensor:
        got = store.pop(seq, None)
        if got is None:
            got = encode(seq)
        store[seq] = got
        while len(store) > keep:
            store.popitem(last=False)
        return got

    cached.cache = store          # the cache-hit test reads this, nothing else does
    return cached


def encoder(*, on_device: bool = True, model_name: str = ESM2_MODEL, cache_dir=None):
    """``seq -> [1, L+2, 1280]``, on the card by default, on the host when asked.

    Built once per process per (checkpoint, path, cache dir) and memoized: 650 M parameters
    are not reloaded per complex, and a card is not reopened per complex. The card and the
    host keep separate entries on purpose -- they are the same weights to about 1.3 %
    relative Frobenius, not to the bit, and a run must not silently mix the two.
    """
    key = (model_name, on_device, None if cache_dir is None else str(cache_dir))
    enc = _ENCODERS.get(key)
    if enc is None:
        if on_device:
            from tt_bio.saprot import esm2_encoder

            base = esm2_encoder(model_name, cache_dir)
        else:
            base = cpu_encoder(model_name, cache_dir)
        enc = _ENCODERS[key] = memoize(base)
    return enc


def receptor_embeddings(
    chains: Iterable[str],
    *,
    on_device: bool = True,
    encode: Callable[[str], torch.Tensor] | None = None,
    model_name: str = ESM2_MODEL,
    cache_dir=None,
) -> list[torch.Tensor]:
    """One ``[L_i, 1280]`` float32 tensor per chain, in chain order.

    *encode* lets a caller pass an encoder it already holds (the card model is loaded once
    per process, and a screen embeds the same receptor for every ligand).
    """
    chains = list(chains)
    for i, seq in enumerate(chains):
        _check_length(seq, f"chain {i}")
    if encode is None:
        encode = encoder(on_device=on_device, model_name=model_name, cache_dir=cache_dir)
    out = []
    for seq in chains:
        emb = encode(seq)
        if emb.dim() == 3:
            emb = emb[0]
        if emb.shape[0] != len(seq) + 2:
            raise ValueError(
                f"ESM-2 returned {emb.shape[0]} rows for a {len(seq)}-residue chain; "
                f"expected {len(seq) + 2} (<cls> + residues + <eos>)")
        if emb.shape[1] != ESM2_DIM:
            raise ValueError(f"ESM-2 returned width {emb.shape[1]}, expected {ESM2_DIM}")
        out.append(emb[1:len(seq) + 1].float().contiguous())
    return out


def receptor_x_columns(chains: Iterable[str], **kw) -> torch.Tensor:
    """The ``[sum(L_i), 1280]`` block DiffDock concatenates onto ``data['receptor'].x``."""
    return torch.cat(receptor_embeddings(chains, **kw), dim=0)
