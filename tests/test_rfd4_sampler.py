"""RFD4's flow-matching sampler: the released recipe, the RNG, and batch invariance.

Host only, no card, no weights. The exact grade against upstream's own loop lives in
`perf/rfd4/sampler/upstream_parity.py`, which needs the reference environment
(`state/rfd4/REF.md`) because `rfproteina` is only importable there; it is bit-exact on the
60-step trajectory, both partial modes and inverse folding, and `state/rfd4/parity/sampler.json`
records it. What is kept here is everything that can run without upstream installed: the released
schedule's own numbers, the recycle accounting, and the two properties the quality grade rests on,
that a design does not change with the batch it rides in or the chunk width it ran at.
"""
import pytest
import torch

from tt_bio.rfd4 import sampler as S


#: An allocator refusal in the allocator's own words, which is what `is_alloc_refusal` reads.
REFUSAL = ("Not enough space to allocate 1048576 B DRAM buffer across 12 banks, where each bank "
           "needs to store 87392 B, but bank size is 1073741824 B (allocated: 1073741824 B, "
           "free: 0 B, largest free block: 0 B)")


def masks(n_designs=3, n_tokens=48):
    padding = torch.ones(n_designs, n_tokens, dtype=torch.bool)
    padding[:, -6:] = False
    latent = padding.clone()
    latent[:, 20:24] = False  # tokens with no local latent, a ligand say
    return {S.BB_CA: padding, S.LOCAL_LATENTS: latent}


def groups(n_designs=3, n_tokens=48):
    molecule = torch.zeros(n_designs, n_tokens, dtype=torch.long)
    molecule[:, n_tokens // 2:] = 1
    return {"molecule_iid": molecule}


class Net:
    """A fixed linear field, so a trajectory is a deterministic function of the sampler alone."""

    def __init__(self, dtype=torch.float32):
        g = torch.Generator().manual_seed(11)
        self.w = {m: torch.randn(d, d, generator=g, dtype=dtype) / d ** 0.5
                  for m, d in ((S.BB_CA, 3), (S.LOCAL_LATENTS, 32))}
        self.calls = []

    def __call__(self, x_t, t, *, recycle_depth, step):
        self.calls.append((step, recycle_depth, x_t[S.BB_CA].shape[0]))
        return {m: {"v": x_t[m] @ self.w[m]} for m in x_t}


def short(nsteps=24):
    """A short trajectory: the sampler's properties do not depend on the step count."""
    return S.replace(S.RELEASED, nsteps=nsteps, n_recycle_schedule=((nsteps, 1),))


def run(spec, *, n_designs=3, seed=4, network=None, **kw):
    n_tokens = 48
    m = masks(n_designs, n_tokens)
    return S.generate(spec, network or Net(), n_designs=n_designs, n_tokens=n_tokens, seed=seed,
                      emit_masks=m, groups=groups(n_designs, n_tokens), **kw)


# ------------------------------------------------------------------ the released recipe


def test_the_released_recipe_is_60_steps_and_180_trunk_passes():
    assert S.RELEASED.nsteps == 60
    assert S.trunk_passes(S.RELEASED) == 180
    depths = S.recycle_depths(S.RELEASED, 60)
    assert len(depths) == 60
    assert depths[:5] == (6,) * 5 and depths[-10:] == (1,) * 10


@pytest.mark.parametrize("name,passes", [("thinking_low", 120), ("thinking_medium", 180),
                                         ("thinking_high", 300), ("medium/flat", 180),
                                         ("medium/midpeak", 150), ("high/midpeak", 180)])
def test_every_shipped_recycle_strategy_costs_what_the_docs_say(name, passes):
    assert S.trunk_passes(S.recycle_strategy(name)) == passes


def test_a_partial_run_takes_the_tail_of_the_schedule():
    full = S.recycle_depths(S.RELEASED, 60)
    tail = S.recycle_depths(S.RELEASED, 24, full_nsteps=60)
    assert tail == full[-24:]


def test_the_tsr_schedule_starts_above_zero_and_ends_at_one():
    ts, gt = S.sample_schedule(S.RELEASED, 60)
    for mode in S.RELEASED.data_modes:
        assert ts[mode].shape == (61,) and gt[mode].shape == (60,)
        assert 0.0 < float(ts[mode][0]) < 0.02, "vf_tsr divides by t: the schedule cannot start at 0"
        assert float(ts[mode][-1]) == pytest.approx(1.0)
        assert torch.all(ts[mode][1:] > ts[mode][:-1])
        assert torch.all(torch.isfinite(gt[mode])) and torch.all(gt[mode] >= 0)


def test_the_tail_is_sampled_at_twice_the_density():
    """10 tail steps at 2x: the last ten intervals are half the width they would otherwise be."""
    ts, _ = S.sample_schedule(S.RELEASED, 60)
    flat = S.schedule_times("log", 60, p1=2.0, tsr_safe=True)
    dt_tail = (ts[S.BB_CA][-11:][1:] - ts[S.BB_CA][-11:][:-1]).mean()
    assert float(dt_tail) < float((flat[-11:][1:] - flat[-11:][:-1]).mean())


def test_the_noise_schedule_is_rescaled_by_the_tsr_ratio():
    ts, gt = S.sample_schedule(S.RELEASED, 60)
    m = S.RELEASED.modalities[S.BB_CA]
    raw = S.noise_injection(ts[S.BB_CA][:-1], m.gt_mode, m.gt_p, m.gt_clamp)
    ratio = S.tsr_ratio(ts[S.BB_CA][:-1], m.tsr_k, m.tsr_sigma)
    assert torch.allclose(gt[S.BB_CA], raw / ratio ** 2)
    assert float(ratio.max()) > 1.5, "the whole point of TSR is that the ratio moves"


def test_an_ode_noise_schedule_is_identically_zero():
    assert torch.all(S.noise_injection(torch.linspace(0.01, 0.99, 20), "ode", 1.0) == 0)


# ------------------------------------------------------------------ the loop


def test_a_trajectory_calls_the_network_once_a_step_with_the_scheduled_depth():
    net = Net()
    out = run(S.RELEASED, network=net)
    assert len(net.calls) == 60
    assert tuple(d for _, d, _ in net.calls) == S.recycle_depths(S.RELEASED, 60)
    assert out.trunk_passes == 180


def test_the_loop_refuses_a_recycle_schedule_that_does_not_cover_the_steps():
    with pytest.raises(ValueError, match="expands to"):
        S.recycle_depths(S.recycle_strategy("medium/flat"), 50)


def test_nothing_leaks_onto_a_token_the_modality_does_not_emit():
    out = run(short())
    m = masks()
    for mode in S.RELEASED.data_modes:
        assert torch.all(out.x[mode][~m[mode]] == 0)


def test_the_state_stays_finite_through_the_released_recipe():
    out = run(S.RELEASED)
    for mode in S.RELEASED.data_modes:
        assert torch.all(torch.isfinite(out.x[mode]))


# ------------------------------------------------------------------ RNG


def test_a_design_is_the_same_whatever_else_is_in_the_batch():
    one = run(short(), n_designs=1)
    three = run(short(), n_designs=3)
    for mode in S.RELEASED.data_modes:
        assert torch.equal(one.x[mode][0], three.x[mode][0])


def test_two_designs_in_one_batch_are_not_copies():
    out = run(short())
    assert float((out.x[S.BB_CA][0] - out.x[S.BB_CA][1]).abs().max()) > 1e-3


def test_the_same_seed_gives_the_same_designs_and_another_seed_does_not():
    a, b, c = run(short(), seed=4), run(short(), seed=4), run(short(), seed=5)
    assert torch.equal(a.x[S.BB_CA], b.x[S.BB_CA])
    assert not torch.equal(a.x[S.BB_CA], c.x[S.BB_CA])


def test_upstream_rng_mode_follows_the_ambient_stream():
    """rng='upstream' is what the exact comparison runs; it reads the global generator."""
    torch.manual_seed(1)
    a = run(short(), rng_mode="upstream")
    torch.manual_seed(1)
    b = run(short(), rng_mode="upstream")
    torch.manual_seed(2)
    c = run(short(), rng_mode="upstream")
    assert torch.equal(a.x[S.BB_CA], b.x[S.BB_CA])
    assert not torch.equal(a.x[S.BB_CA], c.x[S.BB_CA])


def test_a_design_seed_does_not_depend_on_the_machine_or_the_batch():
    assert S.design_seed(7, 0) != S.design_seed(7, 1)
    assert S.design_seed(7, 1) == S.design_seed(7, 1)
    assert S.design_seed(8, 0) != S.design_seed(7, 0)


# ------------------------------------------------------------------ batching


@pytest.mark.parametrize("width", [1, 2, 3, 5])
def test_the_chunk_width_does_not_change_a_design(width):
    whole = run(short(), n_designs=5, network=Net())
    net = Net()
    chunked = run(short(), n_designs=5, network=S.ChunkedNetwork(net, short(), max_parallel_samples=width))
    assert max(n for _, _, n in net.calls) <= width
    for mode in S.RELEASED.data_modes:
        assert torch.equal(whole.x[mode], chunked.x[mode])


def test_a_refused_chunk_halves_the_width_and_reruns():
    """The allocator path: the first wide call refuses, the narrower rerun stands."""
    net = Net()
    refused = {"n": 0}

    def flaky(x_t, t, *, recycle_depth, step):
        if x_t[S.BB_CA].shape[0] > 2 and refused["n"] < 1:
            refused["n"] += 1
            raise RuntimeError(REFUSAL)
        return net(x_t, t, recycle_depth=recycle_depth, step=step)

    out = run(short(nsteps=24), n_designs=4, network=S.ChunkedNetwork(flaky, short(24), max_parallel_samples=4))
    whole = run(short(nsteps=24), n_designs=4, network=Net())
    assert refused["n"] == 1
    for mode in S.RELEASED.data_modes:
        assert torch.equal(out.x[mode], whole.x[mode])


# ------------------------------------------------------------------ lanes


class StagedNet(Net):
    """`Net` split into prepare / launch / collect / complete, logging the order the driver calls them in."""

    def __init__(self):
        super().__init__()
        self.log = []

    def prepare(self, x_t, t, *, recycle_depth, step):
        self.log.append(("prepare", step))
        return x_t, t, recycle_depth, step

    def launch(self, prepared):
        self.log.append(("launch", prepared[3]))
        return prepared

    def collect(self, launched):
        self.log.append(("collect", launched[3]))
        return launched

    def complete(self, collected):
        x_t, t, recycle_depth, step = collected
        self.log.append(("complete", step))
        return self(x_t, t, recycle_depth=recycle_depth, step=step)


def _same(a, b):
    for mode in S.RELEASED.data_modes:
        assert torch.equal(a.x[mode], b.x[mode])
        assert torch.equal(a.x_1_pred[mode], b.x_1_pred[mode])


@pytest.mark.parametrize("width", [1, 2, 3])
def test_lanes_give_the_designs_the_chunker_gives(width):
    chunked = run(short(), n_designs=5, network=S.ChunkedNetwork(Net(), short(), max_parallel_samples=width))
    net = StagedNet()
    laned = run(short(), n_designs=5, network=net, lane_width=width)
    _same(laned, chunked)
    assert max(n for _, _, n in net.calls) <= width


def test_the_next_lane_launches_before_the_host_does_anything():
    """One call in flight, and every readback is followed straight away by the next launch, so the
    card never waits on host work (the heads, the step, the next inputs) that can run under it."""
    net = StagedNet()
    run(short(), n_designs=2, network=net, lane_width=1)
    kinds = [k for k, _ in net.log]
    in_flight = 0
    for k in kinds:
        in_flight += {"launch": 1, "collect": -1}.get(k, 0)
        assert in_flight <= 1
    for i, k in enumerate(kinds[:-2]):
        if k == "collect":
            assert kinds[i + 1:i + 3] == ["launch", "complete"], kinds[i - 2:i + 4]
    assert kinds.count("launch") == 2 * short().nsteps


def test_a_partial_start_runs_in_lanes_too():
    n_tokens = 48
    x_1 = {S.BB_CA: torch.randn(3, n_tokens, 3), S.LOCAL_LATENTS: torch.randn(3, n_tokens, 32)}
    start = S.resolve_partial_start(S.RELEASED, partial_trajectory_fraction_steps=0.4)
    kw = dict(n_designs=3, n_tokens=n_tokens, seed=4, emit_masks=masks(), groups=groups(), start=start, x_1=x_1)
    _same(S.generate(S.RELEASED, StagedNet(), lane_width=1, **kw), S.generate(S.RELEASED, Net(), **kw))


def test_lanes_refuse_the_upstream_stream():
    with pytest.raises(ValueError, match="per_design"):
        run(short(), n_designs=2, network=StagedNet(), lane_width=1, rng_mode="upstream")


# ------------------------------------------------------------------ partial diffusion


def test_a_fraction_runs_that_fraction_of_the_steps_from_the_schedules_tail():
    start = S.resolve_partial_start(S.RELEASED, partial_trajectory_fraction_steps=0.4)
    assert start.nsteps == 24 and start.from_fraction
    ts, _ = S.sample_schedule(S.RELEASED, 60)
    assert start.t[S.BB_CA] == pytest.approx(float(ts[S.BB_CA][36]))

    net = Net()
    n_tokens = 48
    x_1 = {S.BB_CA: torch.randn(3, n_tokens, 3), S.LOCAL_LATENTS: torch.randn(3, n_tokens, 32)}
    out = S.generate(S.RELEASED, net, n_designs=3, n_tokens=n_tokens, seed=4, emit_masks=masks(),
                     groups=groups(), start=start, x_1=x_1)
    assert len(net.calls) == 24
    assert tuple(d for _, d, _ in net.calls) == S.recycle_depths(S.RELEASED, 24, full_nsteps=60)


def test_a_partial_start_at_or_above_one_is_refused():
    with pytest.raises(ValueError, match="must be < 1"):
        S.resolve_partial_start(S.RELEASED, partial_t_backbone=1.0, partial_t_latent=0.5)


def test_one_partial_time_alone_is_refused_unless_it_is_coupled():
    with pytest.raises(ValueError, match="couple_partial_t"):
        S.resolve_partial_start(S.RELEASED, partial_t_backbone=0.5)
    coupled = S.resolve_partial_start(S.RELEASED, partial_t_backbone=0.5, couple_partial_t=True)
    assert set(coupled.t) == set(S.RELEASED.data_modes)
    assert coupled.t[S.LOCAL_LATENTS] != coupled.t[S.BB_CA], "the two schedules are not the same curve"


def test_inverse_folding_excludes_the_partial_arguments():
    with pytest.raises(ValueError, match="mutually exclusive"):
        S.resolve_partial_start(S.RELEASED, inverse_folding=True, partial_t_backbone=0.5)


def test_inverse_folding_returns_the_input_backbone_and_designs_the_latents():
    n_tokens = 48
    start = S.resolve_partial_start(S.RELEASED, inverse_folding=True)
    assert start.t == {S.BB_CA: 0.0, S.LOCAL_LATENTS: 0.0}
    x_1 = {S.BB_CA: torch.randn(3, n_tokens, 3) * 2, S.LOCAL_LATENTS: torch.zeros(3, n_tokens, 32)}
    out = S.generate(short(), Net(), n_designs=3, n_tokens=n_tokens, seed=4, emit_masks=masks(),
                     groups=groups(), start=S.replace(start, nsteps=short().nsteps), x_1=x_1)
    m = masks()
    expected = x_1[S.BB_CA] * m[S.BB_CA][..., None]
    assert torch.allclose(out.x[S.BB_CA], expected), "the backbone is fixed, that is what folding backwards means"
    assert float(out.x[S.LOCAL_LATENTS].abs().max()) > 0


# ------------------------------------------------------------------ trajectory saving


def test_save_trajectory_every_snapshots_on_the_cadence_and_always_the_last_step():
    out = run(short(24), save_trajectory_every=5)
    steps = [snap["step"] for snap in out.trajectory]
    assert steps == [0, 5, 10, 15, 20, 23], "the final step is written whatever the cadence"
    for snap in out.trajectory:
        assert set(snap["x_1"]) == set(S.RELEASED.data_modes)
        assert snap["x_1"][S.BB_CA].shape == (3, 48, 3)
        assert "x_t" not in snap, "the noisy state is opt-in"


def test_no_trajectory_is_kept_unless_it_is_asked_for():
    assert run(short(24)).trajectory == []


def test_dump_noisy_trajectories_adds_the_noisy_state_beside_the_clean_prediction():
    out = run(short(20), save_trajectory_every=4, dump_noisy_trajectories=True)
    final = out.trajectory[-1]
    assert final["step"] == 19
    for mode in S.RELEASED.data_modes:
        assert final["x_t"][mode].shape == final["x_1"][mode].shape
        # x_t is the state the step landed on, x_1 the clean prediction it was made from: at a
        # mid-trajectory step those are not the same tensor.
        assert not torch.equal(final["x_t"][mode], final["x_1"][mode])


def test_a_snapshot_is_the_state_the_trajectory_actually_ended_on():
    out = run(short(20), save_trajectory_every=19, dump_noisy_trajectories=True)
    for mode in S.RELEASED.data_modes:
        assert torch.equal(out.trajectory[-1]["x_t"][mode], out.x[mode])
        assert torch.equal(out.trajectory[-1]["x_1"][mode], out.x_1_pred[mode])
