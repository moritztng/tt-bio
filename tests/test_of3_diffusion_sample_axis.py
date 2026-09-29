"""The OpenFold3 diffusion module's sample axis: one token-DiT call for S noised structures.

`OF3DiffusionModule.__call__(..., samples=[(si, rl_noisy, xl_noisy, t), ...])` runs the
atom-level encoder and decoder once per structure and the 24-block token DiT ONCE for all of
them. The DiT is where the FLOPs and the dispatches are, the atom-level stages carry 5-D
tensors a sample axis would make 6-D, so that is the split.

No device here and no ttnn op: the leaves are stubbed and what runs is the orchestration --
how many DiT calls a batch makes, which slice each structure's decoder reads, and whether the
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
        self.dit_calls, self.post_dit, self.enc_calls = [], [], []
        self.npe = type("npe", (), {"ql_at_np": lambda _s, cl, rl: _Stub(f"ql({rl.name})")})()
        self.dit = type("dit", (), {"supports_multiplicity": True})()

    def enc_at(self, ql_pad, *a, **kw):
        self.enc_calls.append(ql_pad.name)
        return _Stub(f"enc({ql_pad.name})")

    def _pre_dit(self, q, si, *a, **kw):
        return _Stub(f"ai({q.name},{si.name})"), None

    def _post_dit(self, ai, ql, *a, **kw):
        # Counting back from the end: cache, _return_intermediates, sigma_data, t.
        self.post_dit.append((ai.name, ql.name, a[-4]))
        return _Stub(f"xl({ai.name})")

    denoise = M.OF3DiffusionModule._denoise_samples


def _run(rec, S):
    def dit(a, s, *rest, **kw):
        rec.dit_calls.append((a.name, s.name, a.s))
        return _Stub("dit_out", s=a.s)

    rec.dit.__call__ = dit
    rec.dit = type("dit", (), {"supports_multiplicity": True, "__call__":
                               staticmethod(lambda *a, **k: dit(*a, **k))})()
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


def test_the_dit_runs_once_for_the_whole_batch():
    rec = _Recorder()
    out, _ = _run(rec, 4)
    assert len(rec.dit_calls) == 1, "the DiT is being called per replicate, so nothing batched"
    assert rec.dit_calls[0][2] == 4, "the DiT did not see a sample axis of 4"
    # The atom-level stages stay at batch 1 and still run once per structure: that is the
    # deliberate half of the split, not an oversight.
    assert len(rec.enc_calls) == 4
    assert len(out) == 4


def test_every_structure_reads_its_own_slice_and_its_own_sigma():
    rec = _Recorder()
    _, samples = _run(rec, 3)
    a_name, s_name, _ = rec.dit_calls[0]
    assert a_name == "ai(enc(ql(rl0)),si0)+ai(enc(ql(rl1)),si1)+ai(enc(ql(rl2)),si2)"
    assert s_name == "si0+si1+si2"
    for k, (_si, _rl, _xl, t) in enumerate(samples):
        ai, ql, got_t = rec.post_dit[k]
        assert ai == f"dit_out[{k}]", f"structure {k} read slice {ai}"
        assert ql == f"enc(ql(rl{k}))", f"structure {k}'s decoder got the wrong encoder output"
        assert got_t == t, "the EDM output scaling is per structure and must keep its sigma"


def test_one_sample_is_the_per_replicate_loop_with_no_stack_and_no_slice():
    """The A/B control. At S=1 the batched route must BE the loop, not resemble it."""
    rec = _Recorder()
    out, _ = _run(rec, 1)
    a_name, s_name, s = rec.dit_calls[0]
    assert s == 1 and "+" not in a_name and "+" not in s_name
    assert rec.post_dit[0][0] == "dit_out", "S=1 took a slice it does not need"
    assert len(out) == 1


def test_the_transformer_declares_the_capability():
    """The gate every AF3-family sampler on this fleet checks before it batches."""
    assert OF3DiffusionTransformer.supports_multiplicity is True
