"""Which rollout `tt-bio dock` runs: the card's own when it can take the batch, the stepped one otherwise."""
from types import SimpleNamespace

import pytest
import torch

from tt_bio.diffdock import run, sampling
from tt_bio.diffdock.job import DockOptions

P, N = 4, 5
REFUSAL = ("Not enough space to allocate 4831838208 B DRAM buffer across 8 banks, where each "
           "bank needs to store 603979776 B, but bank size is 4278190016 B (allocated: "
           "3579670528 B, free: 698519488 B, largest free block: 504088512 B)")


class Net:
    """A score network that records which of its two rollout members was used."""

    config = SimpleNamespace(no_torsion=True, crop_beyond=None)

    def __init__(self, refuse=False):
        self.calls, self.refuse = [], refuse

    def score_fn(self, graph):
        def score(step):
            self.calls.append("score_fn")
            b = step.lig_pos.shape[0]
            return torch.zeros(b, 3), torch.zeros(b, 3), torch.zeros(0)
        return score

    def rollout(self, graph, start, cfg, bounds, noise):
        self.calls.append("rollout")
        if self.refuse:
            raise RuntimeError(REFUSAL)
        return sampling.Rollout(poses=start + 1, failed=torch.zeros(P, dtype=torch.bool))


def roll(net, **opts):
    options = DockOptions(diffusion_samples=P, **opts)
    graph = SimpleNamespace(ligand={"mask_rotate": []})
    draws = []

    def draw():
        draws.append(1)
        g = torch.Generator().manual_seed(0)
        return torch.randn(P, N, 3, generator=g), sampling.Noise(seed=0)

    cfg = run._sampling_config(options)
    bounds = sampling.SigmaBounds(tr_sigma_min=0.1, tr_sigma_max=19.0, rot_sigma_min=0.03,
                                  rot_sigma_max=1.55, tor_sigma_min=0.0314, tor_sigma_max=3.14)
    out = run._roll_out(net, graph, draw, cfg, bounds, options, torch.zeros(0, 2, dtype=torch.long))
    return out, draws


def test_the_whole_batch_runs_on_the_card():
    net = Net()
    out, draws = roll(net)
    assert net.calls == ["rollout"] and len(draws) == 1
    assert out.poses.shape == (P, N, 3)


@pytest.mark.parametrize("opts", [{"max_parallel_samples": P - 1}, {"save_trajectory": True}])
def test_a_narrower_batch_or_a_trajectory_is_stepped(opts):
    net = Net()
    out, _ = roll(net, **opts)
    assert set(net.calls) == {"score_fn"} and len(net.calls) >= 3
    assert bool(torch.isfinite(out.poses).all())


def test_a_refused_batch_is_stepped_from_a_fresh_draw():
    net = Net(refuse=True)
    out, draws = roll(net)
    assert net.calls[0] == "rollout" and set(net.calls[1:]) == {"score_fn"}
    assert len(draws) == 2
    stepped, _ = roll(Net(), save_trajectory=True)
    assert torch.equal(out.poses, stepped.poses)


def test_anything_but_a_refusal_propagates():
    class Broken(Net):
        def rollout(self, *a):
            raise ValueError("not a refusal")
    with pytest.raises(ValueError):
        roll(Broken())


def test_a_host_network_is_stepped():
    class Host(Net):
        rollout = None
    net = Host()
    roll(net)
    assert set(net.calls) == {"score_fn"}
