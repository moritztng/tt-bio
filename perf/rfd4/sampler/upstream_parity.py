"""Grade tt-bio's RFD4 sampler against upstream's own loop, on fixed seeds.

Runs under the reference environment (`state/rfd4/REF.md`), which is the only place
`rfproteina` is importable:

    /home/moritz/rfd4-ref-env/venv/bin/python perf/rfd4/sampler/upstream_parity.py

The network is a stub: a fixed random linear map of (x_t, t) that both sides call identically.
That is the point. A sampler test must grade the schedule, the update rule and the draw order,
and a stub isolates them from the trunk, which is `rfd4-trunk`'s row and its own parity record.
The stub is deterministic and shared, so any difference in the trajectory is this module's.

Legs:
  schedules   every shipped schedule and noise-injection mode, tt-bio vs upstream, float64
  full        60-step vf_tsr generation from pure noise, both modalities
  partial     a fractional restart (40 %) and an explicit coupled partial_t
  inverse     inverse folding, resampled forward marginal
  recycle     the recycle depth of every step, every shipped strategy, and the pass count
  rng         per-design streams: batch-size and chunk-width invariance (tt-bio only; upstream
              has no such property, which is why we do not inherit it)
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]


def _load_sampler():
    """Import tt_bio/rfd4/sampler.py by path: the reference venv has no tt-bio installed."""
    path = REPO / "tt_bio" / "rfd4" / "sampler.py"
    spec = importlib.util.spec_from_file_location("rfd4_sampler", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["rfd4_sampler"] = module
    spec.loader.exec_module(module)
    return module


S = _load_sampler()

from omegaconf import OmegaConf  # noqa: E402
from rfproteina.flow_matching import product_space_flow_matcher as U  # noqa: E402

N_TOKENS = 48
N_DESIGNS = 3
SEED = 20261021
LATENT_DIM = 32


def upstream_cfg():
    """The released flow-matcher constructor args, as upstream composes them."""
    return OmegaConf.create({
        "loss": {"signed_volume_weight": 0.0},
        "product_flowmatcher": {
            "bb_ca": {
                "zero_com_noise": True, "guidance_enabled": True, "dim": 3,
                "stochastic_centering_mode": "x_0", "stochastic_centering_scale": 0.2,
                "stochastic_centering_scope": "per_group",
                "stochastic_centering_group_keys": ["molecule_iid"],
            },
            "local_latents": {
                "zero_com_noise": False, "guidance_enabled": True, "dim": LATENT_DIM,
                "stochastic_centering_mode": "none", "stochastic_centering_scale": None,
            },
        },
    })


def upstream_sampling_args():
    """`configs/generation/inference_base.yaml`'s `model:` block."""
    return OmegaConf.create({
        "bb_ca": {
            "schedule": {"mode": "log", "p": 2.0, "tail_steps": 10, "tail_step_multiplier": 2},
            "gt": {"mode": "1/t", "p": 1.0, "clamp_val": None},
            "simulation_step_params": {
                "sampling_mode": "vf_tsr", "sc_scale_noise": 1.0, "t_lim_ode": 0.95,
                "t_lim_ode_below": 0.02, "center_every_step": False, "tsr_k": 16.0, "tsr_sigma": 3.0,
            },
        },
        "local_latents": {
            "schedule": {"mode": "power", "p": 2.0, "tail_steps": 10, "tail_step_multiplier": 2},
            "gt": {"mode": "tan", "p": 1.0, "clamp_val": None},
            "simulation_step_params": {
                "sampling_mode": "vf_tsr", "sc_scale_noise": 1.0, "t_lim_ode": 0.95,
                "t_lim_ode_below": 0.02, "center_every_step": False, "tsr_k": 4.0, "tsr_sigma": 12.0,
            },
        },
    })


def make_batch(device="cpu", dtype=torch.float32):
    """A synthetic batch carrying exactly the keys the sampler path reads.

    Two molecules (so per-group centering has more than one group), a padded tail, a context
    region that emits neither modality, and tokens without a local latent (a ligand, say).
    """
    padding = torch.ones(N_DESIGNS, N_TOKENS, dtype=torch.bool)
    padding[:, -6:] = False
    context = torch.zeros(N_DESIGNS, N_TOKENS, dtype=torch.bool)
    context[:, :5] = True
    supports = torch.ones(N_DESIGNS, N_TOKENS, dtype=torch.bool)
    supports[:, 20:24] = False
    molecule_iid = torch.zeros(N_DESIGNS, N_TOKENS, dtype=torch.long)
    molecule_iid[:, N_TOKENS // 2:] = 1
    coord_cond = torch.zeros(N_DESIGNS, N_TOKENS, dtype=torch.bool)
    coord_cond[:, :5] = True

    g = torch.Generator().manual_seed(7)
    x_1 = {
        "bb_ca": torch.randn(N_DESIGNS, N_TOKENS, 3, generator=g, dtype=dtype) * 1.5,
        "local_latents": torch.randn(N_DESIGNS, N_TOKENS, LATENT_DIM, generator=g, dtype=dtype),
    }
    return {
        "padding_mask": padding.to(device),
        "is_in_context_conditioning": context.to(device),
        "supports_local_latent": supports.to(device),
        "molecule_iid": molecule_iid.to(device),
        "chain_iid": molecule_iid.to(device),
        "pn_unit_iid": molecule_iid.to(device),
        "coord_cond_mask": coord_cond.to(device),
        "coords": x_1["bb_ca"].to(device),
        "x_1": {k: v.to(device) for k, v in x_1.items()},
    }


class StubNet:
    """A fixed, cheap, deterministic stand-in for the trunk.

    ``v = W_mode x_t + b_mode * sin(pi t)``, with W and b drawn once from a fixed seed. Nonlinear
    in t, coupling across channels, bounded: enough that a wrong schedule or a wrong step shows
    up in the trajectory instead of cancelling.
    """

    def __init__(self, dims, dtype=torch.float32):
        g = torch.Generator().manual_seed(11)
        self.w = {m: torch.randn(d, d, generator=g, dtype=dtype) / (d ** 0.5) for m, d in dims.items()}
        self.b = {m: torch.randn(d, generator=g, dtype=dtype) * 0.1 for m, d in dims.items()}
        self.calls: list[tuple[int, int]] = []

    def v(self, mode, x_t, t):
        import math
        return x_t @ self.w[mode] + self.b[mode] * math.sin(math.pi * float(t))

    # tt-bio side
    def __call__(self, x_t, t, *, recycle_depth, step):
        self.calls.append((step, recycle_depth))
        return {m: {"v": self.v(m, x_t[m], t[m])} for m in x_t}

    # upstream side: predict_for_sampling(batch, mode=..., recycle_depth=...)
    def upstream(self, batch, mode="full", recycle_depth=0):
        self.calls.append((len(self.calls), recycle_depth))
        return {m: {"v": self.v(m, batch["x_t"][m], batch["t"][m].flatten()[0])} for m in batch["x_t"]}


def rel_err(a, b):
    a, b = a.double(), b.double()
    denom = b.abs().max().clamp(min=1e-12)
    return float((a - b).abs().max() / denom)


def tt_spec(nsteps=60, schedule=S.RECYCLE_SCHEDULES["medium/front_loaded"]):
    return S.replace(S.RELEASED, nsteps=nsteps, n_recycle_schedule=schedule)


def emit_masks(batch):
    from rfproteina.model_masks import compute_flow_emit_masks
    return compute_flow_emit_masks(batch)


def groups_of(batch):
    return {k: batch[k] for k in ("chain_iid", "pn_unit_iid", "molecule_iid")}


# ---------------------------------------------------------------- legs


def leg_schedules():
    """Every schedule and gt mode, tt-bio vs upstream, over float64."""
    rows = []
    for mode, p1 in (("uniform", None), ("power", 2.0), ("log", 2.0)):
        for tsr_safe in (False, True):
            for tail in ((0, 1), (10, 2), (5, 3)):
                fn = U.get_schedule_tsr_safe if tsr_safe else U.get_schedule
                ref = fn(mode=mode, nsteps=60, p1=p1, tail_steps=tail[0], tail_step_multiplier=tail[1])
                got = S.schedule_times(mode, 60, p1=p1, tsr_safe=tsr_safe,
                                       tail_steps=tail[0], tail_step_multiplier=tail[1])
                rows.append({"name": f"schedule:{mode}:tsr_safe={tsr_safe}:tail={tail}",
                             "rel_err": rel_err(got, ref), "ref": "fp32"})
    t = torch.linspace(0.001, 0.999, 60)
    for gt_mode in ("1/t", "1-t/t", "tan", "ode"):
        for p in (1.0, 2.0):
            ref = U.get_gt(t=t, mode=gt_mode, param=p, clamp_val=None)
            got = S.noise_injection(t, gt_mode, p, None)
            rows.append({"name": f"gt:{gt_mode}:p={p}", "rel_err": rel_err(got, ref), "ref": "fp32"})
    for k, sigma in ((16.0, 3.0), (4.0, 12.0)):
        ref = U.compute_tsr_ratio(t, k, sigma)
        got = S.tsr_ratio(t, k, sigma)
        rows.append({"name": f"tsr_ratio:k={k}:sigma={sigma}", "rel_err": rel_err(got, ref), "ref": "fp32"})

    fm = U.ProductSpaceFlowMatcher(upstream_cfg())
    ref_ts, ref_gt = fm.sample_schedule(nsteps=60, sampling_model_args=upstream_sampling_args())
    ts, gt = S.sample_schedule(S.RELEASED, 60)
    for dm in ("bb_ca", "local_latents"):
        rows.append({"name": f"sample_schedule:{dm}:ts", "rel_err": rel_err(ts[dm], ref_ts[dm]), "ref": "fp32"})
        rows.append({"name": f"sample_schedule:{dm}:gt", "rel_err": rel_err(gt[dm], ref_gt[dm]), "ref": "fp32"})
    return rows


def leg_recycle():
    """Recycle depths per step, every shipped strategy, plus the partial tail."""
    rows = []
    from rfproteina.proteina import Proteina

    class _Nn:
        recycle_enabled = True

    for name, sched in S.RECYCLE_SCHEDULES.items():
        model = Proteina.__new__(Proteina)
        model.inf_cfg = OmegaConf.create({"n_recycle_schedule": [list(b) for b in sched], "n_recycle": 0})
        model.nn = _Nn()
        for nsteps, full in ((60, None), (24, 60), (1, 60)):
            ref = model.recycle_depths(nsteps, full_nsteps=full)
            got = S.recycle_depths(tt_spec(schedule=sched), nsteps, full_nsteps=full)
            rows.append({"name": f"recycle:{name}:nsteps={nsteps}", "exact": tuple(ref) == tuple(got),
                         "rel_err": 0.0 if tuple(ref) == tuple(got) else 1.0, "ref": "fp32"})
    passes = S.trunk_passes(tt_spec())
    rows.append({"name": "trunk_passes:medium/front_loaded", "value": passes,
                 "rel_err": 0.0 if passes == 180 else 1.0, "ref": "fp32"})
    return rows


def _upstream_full(batch, net, nsteps):
    fm = U.ProductSpaceFlowMatcher(upstream_cfg())
    model_cfg = upstream_sampling_args()

    class _Nn:
        recycle_enabled = True

    from rfproteina.proteina import Proteina
    model = Proteina.__new__(Proteina)
    model.inf_cfg = OmegaConf.create({
        "n_recycle_schedule": [list(b) for b in S.RECYCLE_SCHEDULES["medium/front_loaded"]], "n_recycle": 0})
    model.nn = _Nn()
    depths = model.recycle_depths(nsteps, full_nsteps=60)
    torch.manual_seed(SEED)
    x, _ = fm.full_simulation(
        batch=batch, predict_for_sampling=net.upstream, recycle_depths=depths, nsteps=nsteps,
        nsamples=N_DESIGNS, n=N_TOKENS, sampling_model_args=model_cfg, device=torch.device("cpu"),
    )
    return x, depths


def leg_full():
    """60 steps of vf_tsr from pure noise: the released recipe's own loop."""
    batch = make_batch()
    net = StubNet({"bb_ca": 3, "local_latents": LATENT_DIM})
    ref_x, ref_depths = _upstream_full(dict(batch), net, 60)

    net.calls.clear()
    torch.manual_seed(SEED)
    out = S.generate(tt_spec(), net, n_designs=N_DESIGNS, n_tokens=N_TOKENS, seed=SEED,
                     emit_masks=emit_masks(batch), rng_mode="upstream",
                     coord_cond_mask=batch["coord_cond_mask"], groups=groups_of(batch))
    rows = [{"name": f"full:{dm}", "rel_err": rel_err(out.x[dm], ref_x[dm]), "ref": "fp32"}
            for dm in ("bb_ca", "local_latents")]
    rows.append({"name": "full:recycle_depths", "rel_err": 0.0 if tuple(d for _, d in net.calls) == tuple(ref_depths) else 1.0,
                 "ref": "fp32"})
    rows.append({"name": "full:trunk_passes", "value": out.trunk_passes,
                 "rel_err": 0.0 if out.trunk_passes == 180 else 1.0, "ref": "fp32"})
    return rows


def _upstream_partial(batch, net, *, frac=None, pt_backbone=None, pt_latent=None,
                      couple=False, inverse=False):
    from rfproteina.proteina import Proteina
    from rfproteina.search import vanilla

    fm = U.ProductSpaceFlowMatcher(upstream_cfg())
    model_cfg = upstream_sampling_args()
    cfg_args = OmegaConf.create({
        "nsteps": 60, "partial_trajectory_fraction_steps": frac, "partial_t_backbone": pt_backbone,
        "partial_t_latent": pt_latent, "couple_partial_t": couple, "inverse_folding": inverse,
        "inverse_folding_method": "resampled_forward_marginal",
    })
    partial_t = vanilla.resolve_partial_t(cfg_args, fm, model_cfg)
    actual = round(frac * 60) if frac is not None else 60

    class _Nn:
        recycle_enabled = True

    model = Proteina.__new__(Proteina)
    model.inf_cfg = OmegaConf.create({
        "n_recycle_schedule": [list(b) for b in S.RECYCLE_SCHEDULES["medium/front_loaded"]], "n_recycle": 0})
    model.nn = _Nn()
    depths = model.recycle_depths(actual, full_nsteps=60)

    torch.manual_seed(SEED)
    batched_t = {dm: torch.full((N_DESIGNS,), v) for dm, v in partial_t.items()}
    batch = fm.corrupt_batch(batch, t=batched_t)
    target = batch["x_1"]["bb_ca"].clone() if inverse else None
    ts, gt = vanilla.build_partial_schedule(partial_t_dict=partial_t, nsteps=actual, fm=fm,
                                            sampling_model_args=model_cfg,
                                            full_nsteps=60 if frac is not None else None)
    x, _, _ = fm.partial_simulation(
        batch=batch, x=batch["x_t"], predict_for_sampling=net.upstream, recycle_depths=depths,
        nsteps=actual, ts=ts, gt=gt,
        simulation_step_params={dm: model_cfg[dm]["simulation_step_params"] for dm in fm.data_modes},
        device=torch.device("cpu"), resample_target_backbone=inverse, target_backbone=target,
    )
    return x, partial_t


def _tt_partial(batch, net, **kw):
    start = S.resolve_partial_start(tt_spec(), **kw)
    torch.manual_seed(SEED)
    out = S.generate(tt_spec(), net, n_designs=N_DESIGNS, n_tokens=N_TOKENS, seed=SEED,
                     emit_masks=emit_masks(batch), rng_mode="upstream", start=start,
                     x_1={k: v.clone() for k, v in batch["x_1"].items()},
                     coord_cond_mask=batch["coord_cond_mask"], groups=groups_of(batch))
    return out, start


def leg_partial():
    rows = []
    cases = [
        ("partial:frac=0.4", {"frac": 0.4}, {"partial_trajectory_fraction_steps": 0.4}),
        ("partial:frac=1.0", {"frac": 1.0}, {"partial_trajectory_fraction_steps": 1.0}),
        ("partial:coupled_bb=0.5", {"pt_backbone": 0.5, "couple": True},
         {"partial_t_backbone": 0.5, "couple_partial_t": True}),
        ("partial:explicit", {"pt_backbone": 0.3, "pt_latent": 0.6},
         {"partial_t_backbone": 0.3, "partial_t_latent": 0.6}),
        ("inverse_folding", {"inverse": True}, {"inverse_folding": True}),
    ]
    for name, up_kw, tt_kw in cases:
        batch = make_batch()
        net = StubNet({"bb_ca": 3, "local_latents": LATENT_DIM})
        ref_x, ref_t = _upstream_partial(dict(batch), net, **up_kw)
        out, start = _tt_partial(batch, net, **tt_kw)
        t_err = max(abs(start.t[dm] - ref_t[dm]) for dm in ref_t)
        rows.append({"name": f"{name}:t", "rel_err": t_err, "ref": "fp32"})
        for dm in ("bb_ca", "local_latents"):
            rows.append({"name": f"{name}:{dm}", "rel_err": rel_err(out.x[dm], ref_x[dm]), "ref": "fp32"})
    return rows


def leg_rng():
    """Per-design streams: a design's trajectory must not depend on the batch it rode in."""
    batch = make_batch()
    masks = emit_masks(batch)
    rows = []

    def run(n_designs, keep):
        sub = {k: (v[:n_designs] if torch.is_tensor(v) else v) for k, v in batch.items()}
        sub_masks = {m: masks[m][:n_designs] for m in masks}
        net = StubNet({"bb_ca": 3, "local_latents": LATENT_DIM})
        out = S.generate(tt_spec(nsteps=24, schedule=((24, 1),)), net, n_designs=n_designs,
                         n_tokens=N_TOKENS, seed=SEED, emit_masks=sub_masks, rng_mode="per_design",
                         coord_cond_mask=sub["coord_cond_mask"], groups=groups_of(sub))
        return {dm: out.x[dm][keep] for dm in out.x}

    one = run(1, slice(0, 1))
    three = run(3, slice(0, 1))
    for dm in ("bb_ca", "local_latents"):
        rows.append({"name": f"rng:batch_invariance:{dm}", "rel_err": rel_err(three[dm], one[dm]), "ref": "exact"})
    again = run(3, slice(0, 1))
    for dm in ("bb_ca", "local_latents"):
        rows.append({"name": f"rng:repeatable:{dm}", "rel_err": rel_err(again[dm], three[dm]), "ref": "exact"})
    # Different designs must not be copies of each other.
    net = StubNet({"bb_ca": 3, "local_latents": LATENT_DIM})
    out = S.generate(tt_spec(nsteps=24, schedule=((24, 1),)), net, n_designs=3, n_tokens=N_TOKENS,
                     seed=SEED, emit_masks=masks, rng_mode="per_design",
                     coord_cond_mask=batch["coord_cond_mask"], groups=groups_of(batch))
    spread = float((out.x["bb_ca"][0] - out.x["bb_ca"][1]).abs().max())
    rows.append({"name": "rng:designs_differ", "rel_err": 0.0 if spread > 1e-3 else 1.0,
                 "value": spread, "ref": "exact"})
    return rows


def main():
    legs = {"schedules": leg_schedules, "recycle": leg_recycle, "full": leg_full,
            "partial": leg_partial, "rng": leg_rng}
    only = sys.argv[1:] or list(legs)
    modules = []
    for name in only:
        rows = legs[name]()
        modules.extend(rows)
        worst = max(r["rel_err"] for r in rows)
        print(f"{name:10s} {len(rows):3d} checks  max rel_err {worst:.3e}")
        for r in rows:
            if r["rel_err"] > 1e-6:
                print(f"    {r['name']}: {r['rel_err']:.3e}")
    out = {"modules": modules, "max_rel_err": max(r["rel_err"] for r in modules)}
    print(json.dumps({"max_rel_err": out["max_rel_err"], "checks": len(modules)}))
    Path(REPO / "perf/rfd4/sampler/parity_raw.json").write_text(json.dumps(out, indent=1))
    return 0 if out["max_rel_err"] <= 1e-6 else 1


if __name__ == "__main__":
    sys.exit(main())
