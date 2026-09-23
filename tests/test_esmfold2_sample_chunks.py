"""The esmfold2 sampler builds the molecule's conditioning once per fold, not once per sample chunk.

Re-uploading the fp32 z_trunk per chunk (one L^2 x 256 x 4 B contiguous request) is what failed
esmfold2-fast at its 1152-token cap with two or more samples on Wormhole: the first chunk had
fragmented DRAM, so the second chunk's upload found no block large enough.
"""

import torch

from tt_bio import esmfold2_runtime as R


class _Head:
    def __init__(self, oom_at=None):
        self.calls, self.oom_at = [], oom_at

    def prepare(self, *args):
        self.calls.append(("prepare",))

    def draw(self, steps, seed, multiplicity):
        self.calls.append(("draw", seed, multiplicity))
        if (seed, multiplicity) == self.oom_at:
            self.oom_at = None
            raise RuntimeError("Out of Memory: not enough space to allocate")
        return torch.full((multiplicity, 4, 3), float(seed))

    def release(self):
        self.calls.append(("release",))


def _sample(head, n, monkeypatch, budget):
    monkeypatch.setenv("TT_ESMFOLD2_DIFFUSION_BUDGET", str(budget))
    L = 8
    z = torch.zeros(1, L, L, 4)
    s = torch.zeros(1, L, 4)
    ref = torch.zeros(1, 4, 3)
    return R._StructureHeadAdapter(head).sample(
        z_trunk=z, s_inputs=s, relative_position_encoding=z, ref_pos=ref,
        ref_charge=torch.zeros(1, 4), ref_mask=torch.ones(1, 4), ref_element=ref,
        ref_atom_name_chars=ref, ref_space_uid=torch.zeros(1, 4), tok_idx=torch.zeros(1, 4),
        num_diffusion_samples=n, num_sampling_steps=6, seed=10)["sample_atom_coords"]


def test_one_prepare_for_every_chunk(monkeypatch):
    head = _Head()
    out = _sample(head, 5, monkeypatch, budget=2 * 64)  # 2 samples per chunk at L=8
    assert head.calls == [("prepare",), ("draw", 10, 2), ("draw", 12, 2), ("draw", 14, 1),
                          ("release",)]
    assert out.shape == (5, 4, 3)
    assert out[:, 0, 0].tolist() == [10, 10, 12, 12, 14]


def test_oom_halves_without_rebuilding(monkeypatch):
    head = _Head(oom_at=(12, 2))
    out = _sample(head, 5, monkeypatch, budget=2 * 64)
    assert [c for c in head.calls if c[0] != "draw"] == [("prepare",), ("release",)]
    assert [c[1:] for c in head.calls if c[0] == "draw"] == [(10, 2), (12, 2), (12, 1),
                                                             (13, 1), (14, 1)]
    assert out.shape == (5, 4, 3)


def test_release_runs_when_a_draw_fails(monkeypatch):
    head = _Head(oom_at=(10, 1))
    try:
        _sample(head, 3, monkeypatch, budget=64)
    except RuntimeError:
        pass
    assert head.calls[-1] == ("release",)
