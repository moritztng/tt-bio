"""ESMC's eager forward honours want_logits: the embed path asks for no logits and gets None, not a NameError."""
from __future__ import annotations

import types

import pytest
import torch

from tt_bio import esmc


@pytest.mark.parametrize("want", [True, False])
def test_eager_forward_want_logits(want, monkeypatch):
    seen = {}

    def module(tokens, mask, kv, want_logits=True):
        seen["want"] = want_logits
        return ("logits" if want_logits else None), "emb"

    monkeypatch.setattr(esmc.ttnn, "from_torch", lambda t, **kw: t)
    m = object.__new__(esmc.ESMC)
    m.__dict__.update(trace=False, tt_device=None, module=module, _to_torch=lambda t: t)
    logits, emb = m._dispatch(torch.zeros(1, 32, dtype=torch.int64), None, None, want)
    assert seen["want"] is want
    assert logits == ("logits" if want else None) and emb == "emb"
