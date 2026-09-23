"""The diffusion sample axis narrows on a DRAM refusal instead of ending the fold.

Host-only: the device denoise is replaced by a per-sample deterministic function, so what runs is
the samplers' own RNG, chunking and refusal handling. With such a function every width must give
every sample the same bits, because the samplers draw all noise over the whole sample axis before
chunking. That is the property the fallback relies on: a narrower width changes device numerics
at most, never which noise a sample gets.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tt_bio import boltz2  # noqa: E402
from tt_bio.protenix import edm_sample  # noqa: E402
from tt_bio.sample_chunks import denoise_in_chunks, resolve_sample_chunk_width  # noqa: E402

# Verbatim from the boltz2 eal_b12 x5 fold on main (1728 tokens), its first diffusion step.
REFUSAL = ("Out of Memory: Not enough space to allocate 1274019840 B DRAM buffer across 12 banks, "
           "where each bank needs to store 106168320 B, but bank size is 1073741792 B (allocated: "
           "852732192 B, free: 221009600 B, largest free block: 92528640 B)")


def _per_sample(x):
    """Deterministic in each sample alone, so any change of chunking leaves it bit-identical."""
    return 0.5 * x + 0.01 * x.pow(2).sum(-1, keepdim=True).sqrt()


class _Refuse:
    """Refuses (with the allocator's own message) any chunk wider than `fits`."""

    def __init__(self, fits):
        self.fits, self.widths, self.resets = fits, [], 0

    def run(self, chunk, width):
        self.widths.append(width)
        if width > self.fits:
            raise RuntimeError(REFUSAL)
        return _per_sample(chunk)

    def reset(self):
        self.resets += 1


def test_a_chunk_that_fits_is_left_alone():
    x = torch.randn(10, 7, 3)
    r = _Refuse(fits=5)
    out, width = denoise_in_chunks(x, 5, r.run, reset=r.reset)
    assert width == 5 and r.widths == [5, 5] and r.resets == 0
    assert torch.equal(out, _per_sample(x))


def test_a_refusal_halves_the_width_and_reruns_only_the_refused_chunk():
    x = torch.randn(25, 7, 3)
    r = _Refuse(fits=2)
    out, width = denoise_in_chunks(x, 5, r.run, reset=r.reset)
    assert width == 2
    assert r.widths[:2] == [5, 2] and set(r.widths[1:]) == {2} and r.resets == 1
    assert torch.equal(out, _per_sample(x))
    # The next step starts at the width this one settled at: one refusal per trajectory.
    r.widths.clear()
    denoise_in_chunks(x, width, r.run, reset=r.reset)
    assert set(r.widths) == {2} and r.resets == 1


def test_it_narrows_down_to_one_sample_and_then_raises():
    x = torch.randn(5, 7, 3)
    out, width = denoise_in_chunks(x, 5, _Refuse(fits=1).run)
    assert width == 1 and torch.equal(out, _per_sample(x))
    with pytest.raises(RuntimeError, match="Out of Memory"):
        denoise_in_chunks(x, 5, _Refuse(fits=0).run)


def test_a_batch_that_cannot_be_split_raises_the_refusal():
    with pytest.raises(RuntimeError, match="Out of Memory"):
        denoise_in_chunks(torch.randn(4, 7, 3), 4, _Refuse(fits=1).run, narrowest=4)


def test_anything_but_an_allocator_refusal_propagates():
    def broken(chunk, width):
        raise RuntimeError("TT_FATAL: circular buffers clash with L1 buffers")
    with pytest.raises(RuntimeError, match="circular"):
        denoise_in_chunks(torch.randn(5, 7, 3), 5, broken)


def test_the_width_rule_is_torch_chunks_partition():
    """Protenix chunked with `torch.chunk(ceil(M / mps))` before it shared the rule, so the
    shared rule must give the same partition or every protenix-family digest moves."""
    for m in range(1, 60):
        for mps in range(1, 12):
            n = -(-m // mps)
            sizes = [c.numel() for c in torch.arange(m).chunk(n)]
            w = resolve_sample_chunk_width(m, mps)
            assert sizes == [min(w, m - s) for s in range(0, m, w)], (m, mps)


class _Stub:
    denoise_traced = None

    def __init__(self, fits=None):
        self.fits = fits

    def denoise(self, x, t_hat, cond):
        if self.fits is not None and x.shape[0] > self.fits:
            raise RuntimeError(REFUSAL)
        return _per_sample(x)


@pytest.mark.parametrize("seeds", [False, True])
def test_protenix_samples_do_not_depend_on_the_width(seeds):
    m = 7
    kw = dict(multiplicity=m, n_step=6, seed=3,
              member_seeds=[3 + k for k in range(m)] if seeds else None)
    ref = edm_sample(_Stub(), {}, 11, max_parallel_samples=m, **kw)
    for w in (1, 2, 3, 5):
        assert torch.equal(edm_sample(_Stub(), {}, 11, max_parallel_samples=w, **kw), ref), w
    assert torch.equal(edm_sample(_Stub(fits=2), {}, 11, max_parallel_samples=5, **kw), ref)


def test_protenix_multi_target_batch_is_never_split():
    with pytest.raises(RuntimeError, match="Out of Memory"):
        edm_sample(_Stub(fits=2), {"_members": 4}, 11, multiplicity=4, max_parallel_samples=4,
                   member_seeds=[0] * 4, n_step=3)


class _Net(torch.nn.Module):
    def __init__(self, **_kw):
        super().__init__()


def _boltz2_sampler(monkeypatch, fits=None):
    monkeypatch.setattr(boltz2, "_get_pytorch_modules", lambda: (_Net, None, None, None))
    ad = boltz2.AtomDiffusion(score_model_args={"token_s": 384})
    widths = []

    def net(r, sigma, network_condition_kwargs):
        widths.append(r.shape[0])
        assert network_condition_kwargs["multiplicity"] == r.shape[0]
        if fits is not None and r.shape[0] > fits:
            raise RuntimeError(REFUSAL)
        return _per_sample(r)

    ad.preconditioned_network_forward = net
    return ad, widths


def _boltz2_sample(ad, m, mps):
    torch.manual_seed(0)
    steering = {"fk_steering": False, "physical_guidance_update": False,
                "contact_guidance_update": False}
    return ad.sample(atom_mask=torch.ones(1, 11), num_sampling_steps=6, multiplicity=m,
                     max_parallel_samples=mps, steering_args=steering)["sample_atom_coords"]


def test_boltz2_samples_do_not_depend_on_the_width(monkeypatch):
    ad, _ = _boltz2_sampler(monkeypatch)
    ref = _boltz2_sample(ad, 7, 7)
    for w in (1, 2, 3, 5):
        assert torch.equal(_boltz2_sample(ad, 7, w), ref), w
    ad, widths = _boltz2_sampler(monkeypatch, fits=2)
    assert torch.equal(_boltz2_sample(ad, 7, 5), ref)
    # 5 is rebalanced to 4 for 7 samples, refused, then 2 for the rest of the trajectory.
    assert widths[0] == 4 and set(widths[1:]) == {2}, widths


def test_every_batched_sampler_goes_through_the_one_helper():
    """Counted in the source, so a sampler loop that grows its own copy is caught here."""
    calls = {p.name: len(re.findall(r"\bdenoise_in_chunks\(", p.read_text()))
             for p in (ROOT / "tt_bio").rglob("*.py") if p.name != "sample_chunks.py"}
    assert {k: v for k, v in calls.items() if v} == {"boltz2.py": 1, "protenix.py": 1}
    # OpenDDE, OpenDDE-AbAg and PXDesign sample through protenix.edm_sample.
    for rel in ("opendde.py", "pxdesign/model.py"):
        assert "edm_sample(" in (ROOT / "tt_bio" / rel).read_text(), rel
