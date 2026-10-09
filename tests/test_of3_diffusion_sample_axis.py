"""The OpenFold3 diffusion module's sample axis: one encoder, DiT and decoder call for S structures.

`OF3DiffusionModule.__call__(..., samples=[(si, rl_noisy, xl_noisy, t), ...])` stacks the
samples on the leading dim of the atom encoder, the 24-block token DiT and the atom decoder.
Only the per-sample token aggregation and the EDM output scaling loop over the samples.

No device here and no ttnn op: the leaves are stubbed and what runs is the orchestration --
how many calls a batch makes, which slice each structure's EDM step reads, and whether the
S=1 arm is the same program the per-replicate loop was. The arithmetic is device work and is
gated on a card by the DiT golden (`test_openfold3_diffusion_transformer.py`).
"""

from __future__ import annotations

import pytest

ttnn = pytest.importorskip("ttnn")

from tt_bio import openfold3_diffusion_module as M  # noqa: E402
from tt_bio.openfold3_diffusion_transformer import OF3DiffusionTransformer  # noqa: E402


class _Stub:
    """A stand-in tensor: a name and a shape, and nothing a device would have to allocate."""

    def __init__(self, name, s=1):
        self.name, self.s = name, s

    def __getitem__(self, sl):
        assert isinstance(sl, slice) and sl.stop == sl.start + 1
        return _Stub(f"{self.name}[{sl.start}]")

    def __repr__(self):                                                # pragma: no cover
        return f"_Stub({self.name!r}, s={self.s})"


class _Recorder:
    """An OF3DiffusionModule with every leaf replaced, so only `_denoise_samples` runs."""

    def __init__(self):
        self.dit_calls, self.enc_calls, self.dec_calls, self.edm = [], [], [], []
        self.npe = type("npe", (), {"ql_at_np": lambda _s, cl, rl: _Stub(f"ql({rl.name})", rl.s)})()

    def enc_at(self, ql_pad, *a, **kw):
        self.enc_calls.append(ql_pad.name)
        return _Stub(f"enc({ql_pad.name})", ql_pad.s)

    def _pre_dit(self, q, si, *a, **kw):
        return _Stub(f"ai({q.name},{si.name})"), None

    def _decode(self, a, ql, *rest):
        self.dec_calls.append((a.name, ql.name))
        return _Stub(f"dec({a.name})", a.s)

    def _edm(self, xl, rl_update, _mask, t, _sigma):
        self.edm.append((xl.name, rl_update.name, t))
        return _Stub(f"out({xl.name})")

    denoise = M.OF3DiffusionModule._denoise_samples


def _run(rec, S):
    def dit(a, s, *rest, **kw):
        rec.dit_calls.append((a.name, s.name, a.s))
        return _Stub("dit_out", s=a.s)

    rec.dit = dit
    samples = [(_Stub(f"si{k}"), _Stub(f"rl{k}"), _Stub(f"xl{k}"), 1.0 + k) for k in range(S)]
    # cl_pad, plm, zij, then the nine shared masks/maps, then the five extents, sigma, cache.
    out = rec.denoise(samples, _Stub("cl"), _Stub("plm"), _Stub("zij"), *[None] * 9,
                      *[None] * 5, 16.0, {})
    return out, samples


@pytest.fixture(autouse=True)
def _stack(monkeypatch):
    """The two ttnn calls `_denoise_samples` makes itself; record one, no-op the other."""
    monkeypatch.setattr(M, "stack_samples",
                        lambda parts: _Stub("+".join(p.name for p in parts), s=len(parts)))
    monkeypatch.setattr(M.ttnn, "deallocate", lambda t, *a, **k: None)


def test_encoder_dit_and_decoder_run_once_for_the_whole_batch():
    rec = _Recorder()
    out, _ = _run(rec, 4)
    assert rec.enc_calls == ["ql(rl0+rl1+rl2+rl3)"], "the encoder is being called per replicate"
    assert len(rec.dit_calls) == 1 and rec.dit_calls[0][2] == 4, "the DiT did not see S=4"
    assert len(rec.dec_calls) == 1, "the decoder is being called per replicate"
    assert len(out) == 4


def test_every_structure_reads_its_own_slice_and_its_own_sigma():
    rec = _Recorder()
    _, samples = _run(rec, 3)
    a_name, s_name, _ = rec.dit_calls[0]
    enc = "enc(ql(rl0+rl1+rl2))"
    assert a_name == "+".join(f"ai({enc}[{k}],si{k})" for k in range(3))
    assert s_name == "si0+si1+si2"
    assert rec.dec_calls == [("dit_out", enc)], "the decoder must read the batched encoder output"
    for k, (_si, _rl, _xl, t) in enumerate(samples):
        xl, rl_update, got_t = rec.edm[k]
        assert (xl, rl_update) == (f"xl{k}", f"dec(dit_out)[{k}]"), f"structure {k} read the wrong slice"
        assert got_t == t, "the EDM output scaling is per structure and must keep its sigma"


def test_one_sample_is_the_per_replicate_loop_with_no_stack_and_no_slice():
    """The A/B control. At S=1 the batched route must BE the loop, not resemble it."""
    rec = _Recorder()
    out, _ = _run(rec, 1)
    a_name, s_name, s = rec.dit_calls[0]
    assert s == 1 and "+" not in a_name and "+" not in s_name and "[" not in a_name
    assert rec.edm[0][1] == "dec(dit_out)", "S=1 took a slice it does not need"
    assert len(out) == 1


def test_the_transformer_declares_the_capability():
    """The gate every AF3-family sampler on this fleet checks before it batches."""
    assert OF3DiffusionTransformer.supports_multiplicity is True
