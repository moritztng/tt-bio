#!/usr/bin/env python3
"""RF3's structural-accuracy cell: absolute-Angstrom R/D/X against a measured floor.

Every other model in `docs/implementation-parity.md` is scored one way, an absolute RMSD
against the reference's own run-to-run spread. RF3's row records ceiling-relative
activation ratios instead, so the first question anyone asks about it ("is it as accurate
as the others") has no answer. This produces the comparable number.

    R = reference vs reference across seeds       the spread the model itself carries
    D = device vs device across seeds
    X = device vs reference at a matched seed     the port's error
    floor = max(mean R, mean D); the leg is inside the floor when
    mean X <= floor + max(std R, std D)

That is `pharma_parity.noise_floor_verdict`, the function every legacy R/D/X row in the
parity table was produced with, called here rather than reimplemented. The distance is
`boltz2_affinity_parity._kabsch_rmsd`, the gate's own superposition, for the same reason.

Two things a diffusion port gets wrong if it is careless, both already paid for here:

* Cross-RNG comparison. X only means something when both arms see the same noise, so the
  reference rollout RECORDS its draws and the device rollout REPLAYS them, and the RNG is
  re-seeded immediately before each sampler entry rather than once up front. R and D are
  the opposite by construction: different seeds, different draws, which is what makes
  them a floor.
* Kabsch. Superposition is not transitive, so every pair is superposed on itself and no
  shared reference frame exists for the answer to depend on. The helper is identity-tested
  at startup (a rigid copy must score 0), because an inverted Kabsch reports a large
  phantom RMSD on exactly this kind of comparison and nothing else catches it.

The arm is the shipped one: `tt-bio predict --model rf3` defaults. 10 recycles, 50
sampling steps, one diffusion sample, HiFi4 + fp32_dest_acc + packer_l1_acc, MSA attached.
Each seed's coordinates are cached under --work, so a run that dies part-way resumes
instead of starting over.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from boltz2_affinity_parity import _kabsch_rmsd  # noqa: E402  the gate's own superposition
from pharma_parity import noise_floor_verdict  # noqa: E402  the gate's own floor verdict
sys.path.insert(0, str(Path(__file__).resolve().parent))
from arms import ARMS, ROUTE, SCOPE, apply_arm  # noqa: E402

import tt_bio  # noqa: E402

# The env installs tt_bio editable against the SHARED checkout, and running a script puts
# scripts/ on sys.path[0] rather than the repo root, so an unqualified run silently scores
# the shared tree instead of this worktree. Refuse rather than report a number about code
# that is not the code under test.
if Path(tt_bio.__file__).resolve().parent.parent != REPO:
    raise SystemExit(
        f"tt_bio resolves to {tt_bio.__file__}, not {REPO}. Re-run with "
        f"PYTHONPATH={REPO}")

DEFAULT_CKPT = os.path.expanduser(
    "~/.cache/tt-bio/rf3/rf3_foundry_01_24_latest_remapped.ckpt")


def kabsch_selftest() -> float:
    """A rigidly transformed copy must score 0. One line, and it catches an inverted
    Kabsch, which otherwise reads as a large port error on displaced-but-identical
    structures while still scoring near-aligned pairs about right."""
    rng = np.random.default_rng(0)
    P = rng.normal(size=(64, 3)) * 12.0
    th = 0.734
    rot = np.array([[np.cos(th), -np.sin(th), 0.0],
                    [np.sin(th), np.cos(th), 0.0],
                    [0.0, 0.0, 1.0]])
    Q = (rot @ P.T).T + np.array([31.0, -7.5, 104.0])
    v = _kabsch_rmsd(P, Q)
    if not v < 1e-6:
        raise SystemExit(f"Kabsch identity test failed: {v} A on a rigid copy")
    return float(v)


def draws_sha(values) -> str:
    h = hashlib.sha256()
    for v in values:
        h.update(np.ascontiguousarray(v.detach().cpu().numpy(), dtype=np.float32).tobytes())
    return h.hexdigest()[:16]


def fixture_dir(name: str) -> Path:
    roots = [REPO / "scripts/rf3_port/parity_artifacts",
             REPO / "scripts/rf3_port/size_ladder"]
    d = next((r / name for r in roots if (r / name).is_dir()), None)
    if d is None:
        raise SystemExit(f"{name}: not under any of {[str(r) for r in roots]}")
    return d


def featurize_seed(fixture: str, recycles: int, seed: int):
    from tt_bio.rf3.featurize import featurize
    d = fixture_dir(fixture)
    inp = next((n for n in ("input.json", "input.cif") if (d / n).exists()), None)
    if inp is None:
        raise SystemExit(f"{d}: no input.json or input.cif")
    prev = os.getcwd()
    os.chdir(d)
    try:
        # diffusion_batch_size=1: one sample per rollout, so there is no confidence
        # ranking between samples and X compares a trajectory to its own shared-draws
        # counterpart. A 5-sample batch would put a rank-0 coin flip inside the metric
        # (see scripts/parity_sample_matrix.py).
        return featurize(inp, n_recycles=recycles, diffusion_batch_size=1, seed=seed)[0]
    finally:
        os.chdir(prev)


def ref_trunk_bf16(net, f, recycles: int):
    """The reference trunk under bf16 autocast, which is the arm the port is scored
    against: both sides run bf16 where the reference does. Autocast casts msa_stack and
    the template features IN PLACE, so the reference gets a cloned dict and the device
    arm keeps the original."""
    from collections import deque
    ff = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in f.items()}
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        ro = deque(net.trunk_forward_with_recycling(f=ff, n_recycles=recycles),
                   maxlen=1).pop()
    return {k: v.float() for k, v in ro.items()}


def host_sampler(steps: int):
    """The sampler without a card. Same class and the same `sigma_data` the device model
    would construct it with, read off `RF3.__init__` rather than copied, so the schedule
    constant cannot drift between the two entry points."""
    import inspect
    from tt_bio.rf3.model import RF3
    from tt_bio.rf3.sampler import DiffusionSampler
    sd = inspect.signature(RF3.__init__).parameters["sigma_data"].default
    return DiffusionSampler(num_timesteps=steps, sigma_data=sd)


def harvest_draws(sampler, seed: int, n_atom: int):
    """The draw stream, without running a denoiser.

    Every draw the sampler consumes has a fixed shape and none of them depends on what the
    denoiser returned: one `normal (1, L, 3)` before the loop, then `rand (1)` x3,
    `normal (1, 1, 3)` and `normal X.shape` per step. So a stub denoiser reproduces the
    stream exactly, and an arm re-run at an already-scored fixture needs no CPU reference.
    The hash is asserted against the cached reference's, so a schedule change aborts rather
    than silently scoring a cross-RNG comparison.
    """
    from tt_bio.rf3.sampler import Draws
    torch.manual_seed(seed)
    _, rec = sampler.sample(lambda x_noisy, t: torch.zeros_like(x_noisy),
                            torch.zeros(1, n_atom, 3), 1, draws=Draws())
    return rec


def run_seed(args, net, tt, seed: int, work: Path, ref_cache: Path | None = None) -> dict:
    """One seed: features, both trunks, the reference rollout that records the draws, and
    the device rollout that replays them.

    The reference half is arm-independent -- it is CPU torch and never sees a flag -- so
    when `ref_cache` already holds this seed the reference is loaded instead of recomputed
    and only the draws are re-harvested (and hash-checked) to feed the replay. `tt is None`
    is the reference-only mode: no card, nothing device-side, and the npz it writes has no
    `dev` key, which is how a later device run recognises it as a reference cache.
    """
    from tt_bio.rf3.sampler import Draws

    cache = work / f"seed{seed}.npz"
    if cache.exists():
        z = np.load(cache)
        if "dev" in z.files or tt is None:
            return {k: z[k] for k in z.files} | {"cached": True}

    t0 = time.time()
    out = featurize_seed(args.fixture, args.recycles, seed)
    f = out["feats"]
    rep = out.get("ground_truth", {}).get("rep_atom_idxs")
    if rep is None:
        raise SystemExit("no rep_atom_idxs in the capture: nothing to take CA-RMSD over")
    rep_idx = np.asarray(rep).reshape(-1).astype(int)
    n_atom = int(f["ref_pos"].shape[-2])
    feat_s = time.time() - t0

    coord = torch.zeros(1, n_atom, 3)
    sampler = tt.sampler if tt is not None else host_sampler(args.steps)
    cached_ref = ref_cache / f"seed{seed}.npz" if ref_cache is not None else None

    if cached_ref is not None and cached_ref.exists():
        z = np.load(cached_ref)
        ref_np, sha = z["ref"], str(z["shastr"][0])
        ref_trunk_s = 0.0
        t0 = time.time()
        rec = harvest_draws(sampler, seed, n_atom)
        ref_roll_s = time.time() - t0
        got = draws_sha(rec.values)
        if got != sha:
            raise SystemExit(
                f"seed {seed}: harvested draws {got} != cached reference {sha}; the "
                "schedule or the draw order moved, so the cached reference cannot be "
                "replayed")
        ref_src = f"cache:{cached_ref}"
    else:
        t0 = time.time()
        ro = ref_trunk_bf16(net, f, args.recycles)
        ref_trunk_s = time.time() - t0

        def ref_denoise(x_noisy, t):
            with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
                return net.diffusion_module(
                    X_noisy_L=x_noisy, t=t, f=f,
                    S_inputs_I=ro["S_inputs_I"], S_trunk_I=ro["S_I"],
                    Z_trunk_II=ro["Z_II"]).float()

        # Re-seed at SAMPLER ENTRY. A single up-front manual_seed is not enough: the two
        # trunks can consume the stream differently, which desyncs the draws before the
        # rollout starts (memory diffusion-port-parity-shared-draws, 2026-07-23 addendum).
        # The shipped path does the same thing, torch.manual_seed(seed) right before
        # predict() (worker.py).
        torch.manual_seed(seed)
        t0 = time.time()
        ref_x, rec = sampler.sample(ref_denoise, coord, 1, draws=Draws())
        ref_roll_s = time.time() - t0
        sha = draws_sha(rec.values)
        ref_np = ref_x[0].detach().cpu().numpy().astype(np.float64)
        ref_src = "computed"

    if tt is None:
        rec_arr = {
            "ref": ref_np, "rep_idx": rep_idx,
            "timing": np.array([feat_s, ref_trunk_s, ref_roll_s, 0.0]),
            "sha": np.array([int(sha[:8], 16)], dtype=np.int64),
            "shastr": np.array([sha]), "refsrc": np.array([ref_src]),
        }
        work.mkdir(parents=True, exist_ok=True)
        np.savez(cache, **rec_arr)
        return rec_arr | {"cached": False}

    torch.manual_seed(seed)
    t0 = time.time()
    dev = tt.predict(f, n_recycles=args.recycles, diffusion_batch_size=1,
                     rep_atom_idxs=rep, coord_to_be_noised=coord,
                     draws=Draws(list(rec.values)))
    dev_s = time.time() - t0
    replay = dev["draws"]
    if not replay.exhausted():
        raise SystemExit(f"seed {seed}: the device rollout left recorded draws unused; "
                         "the two call sequences differ")
    if draws_sha(replay.values) != sha:
        raise SystemExit(f"seed {seed}: replayed draws hash {draws_sha(replay.values)} "
                         f"!= recorded {sha}; the arms did not share noise")

    rec_arr = {
        "ref": ref_np,
        "dev": dev["X_L"][0].detach().cpu().numpy().astype(np.float64),
        "rep_idx": rep_idx,
        "timing": np.array([feat_s, ref_trunk_s, ref_roll_s, dev_s]),
        "sha": np.array([int(sha[:8], 16)], dtype=np.int64),
        "shastr": np.array([sha]), "refsrc": np.array([ref_src]),
    }
    work.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **rec_arr)
    return rec_arr | {"cached": False}


def score(per_seed: dict, seeds: list[int]) -> dict:
    """R, D and X per metric, then the gate's own floor verdict on each."""
    metrics = {}
    for name, sel in (("kabsch_rmsd", "ca"), ("allatom_rmsd", "all")):
        def coords(seed, arm):
            c = per_seed[seed][arm]
            return c[per_seed[seed]["rep_idx"]] if sel == "ca" else c

        R = [_kabsch_rmsd(coords(a, "ref"), coords(b, "ref"))
             for a, b in itertools.combinations(seeds, 2)]
        D = [_kabsch_rmsd(coords(a, "dev"), coords(b, "dev"))
             for a, b in itertools.combinations(seeds, 2)]
        X = [_kabsch_rmsd(coords(s, "dev"), coords(s, "ref")) for s in seeds]
        v = noise_floor_verdict(X, R, D, name)
        v["X_per_seed"] = {str(s): round(x, 4) for s, x in zip(seeds, X)}
        v["R_pairs"] = [round(x, 4) for x in R]
        v["D_pairs"] = [round(x, 4) for x in D]
        metrics[name] = v
    return metrics


def resolved_flags() -> dict:
    from tt_bio import tenstorrent as tts
    from tt_bio.rf3.remap import PAIRFORMER_FLAGS
    return {
        "pairformer_flags": {k: (v if isinstance(v, (bool, int, float, str)) or v is None
                                 else repr(v)) for k, v in PAIRFORMER_FLAGS.items()},
        "triatt_fused_hifi": bool(tts._TRIATT_FUSED_HIFI),
        "boltz2_fp32_softmax": bool(tts._FP32_SOFTMAX),
        # Which route the arm actually took, counted by the code rather than argued from
        # the flags: TRIATT_FUSED_HIFI_STATS separates served from too_short, and
        # FP32_SOFTMAX_STATS["calls"] is 0 for an arm that left the materialised path.
        "triatt_fused_hifi_stats": dict(tts.TRIATT_FUSED_HIFI_STATS),
        "fp32_softmax_stats": dict(tts.FP32_SOFTMAX_STATS),
        "sdpa_k_chunk_stats": list(tts.SDPA_K_CHUNK_STATS),
        "env": {k: v for k, v in sorted(os.environ.items())
                if k.startswith(("TT_BIO", "TT_METAL", "TT_VISIBLE"))},
    }


def probe(args, net) -> int:
    """Cost, not accuracy: one featurization, one trunk, one denoiser call. This is what
    sizes the real run, so the seed count is a measured decision and not a guess."""
    t0 = time.time()
    out = featurize_seed(args.fixture, args.recycles, 0)
    f = out["feats"]
    feat_s = time.time() - t0
    rep = np.asarray(out["ground_truth"]["rep_atom_idxs"]).reshape(-1)
    n_atom = int(f["ref_pos"].shape[-2])

    t0 = time.time()
    ro = ref_trunk_bf16(net, f, args.recycles)
    trunk_s = time.time() - t0

    x = torch.randn(1, n_atom, 3) * 16.0
    t = torch.tensor([16.0])
    t0 = time.time()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        net.diffusion_module(X_noisy_L=x, t=t, f=f, S_inputs_I=ro["S_inputs_I"],
                            S_trunk_I=ro["S_I"], Z_trunk_II=ro["Z_II"])
    call_s = time.time() - t0

    rep_out = {
        "fixture": args.fixture, "recycles": args.recycles, "steps": args.steps,
        "n_atom": n_atom, "n_token": int(ro["S_I"].shape[-2]), "n_rep_atom": int(rep.size),
        "featurize_s": round(feat_s, 1),
        "ref_trunk_s": round(trunk_s, 1),
        "ref_denoiser_call_s": round(call_s, 2),
        "ref_rollout_s_projected": round(call_s * (args.steps - 1), 1),
        "ref_seed_s_projected": round(feat_s + trunk_s + call_s * (args.steps - 1), 1),
    }
    print(json.dumps(rep_out, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(rep_out, indent=2) + "\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--fixture", default="ubq_76")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--recycles", type=int, default=10,
                    help="shipped default for rf3 (main._resolve_recycling_steps)")
    ap.add_argument("--steps", type=int, default=50,
                    help="shipped default for rf3 (main._resolve_sampling_steps); the "
                         "schedule holds `steps` entries and the rollout runs steps-1 "
                         "denoise calls")
    ap.add_argument("--work", default=None,
                    help="per-seed coordinate cache; defaults to "
                         "perf/rf3/results/accuracy_<fixture>")
    ap.add_argument("--out", default=None)
    ap.add_argument("--probe", action="store_true",
                    help="cost only: one featurization, one trunk, one denoiser call, no "
                         "device and no rollout")
    ap.add_argument("--rescore", action="store_true",
                    help="score the cached seeds and exit; no model, no card")
    ap.add_argument("--arm", default="a0", choices=sorted(ARMS),
                    help="triangle-attention route; see scripts/rf3_port/arms.py")
    ap.add_argument("--one-k-chunk", action="store_true",
                    help="force TriangleAttention(tri_att_one_k_chunk=True) on every instance "
                         "the model constructs, so the fused SDPA spans the key length in ONE k "
                         "chunk. Reachable only on --arm a2/the TT_BIO_TRIATT_FUSED_HIFI route, "
                         "and a provable no-op at any size whose PADDED length is at or below "
                         "the shipped k_chunk -- which both anchors are (76 and 117 aa both pad "
                         "to 128, where _sdpa_chunks_shipped already returns k_chunk = 128). "
                         "Kept so that no-op is measured rather than asserted.")
    ap.add_argument("--ref-cache", default=None,
                    help="where the arm-independent reference coordinates live; defaults "
                         "to a0's work dir, so a new arm pays only its device rollout")
    ap.add_argument("--ref-only", action="store_true",
                    help="reference half only: no card, no device, no arm. Writes a npz "
                         "with no `dev` key, which a later device run reads as a cache")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    # a0's dir keeps its bare name so the committed caches and --rescore are unchanged, and
    # it doubles as every other arm's reference cache: R is arm-independent by construction.
    ref_work = Path(args.ref_cache) if args.ref_cache else \
        REPO / "perf/rf3/results" / f"accuracy_{args.fixture}"
    suffix = "" if args.arm == "a0" or args.ref_only else f"_{args.arm}"
    work = Path(args.work) if args.work else \
        REPO / "perf/rf3/results" / f"accuracy_{args.fixture}{suffix}"
    self_test = kabsch_selftest()

    report = {"arm": args.arm, "arm_route": ROUTE[args.arm], "arm_scope": SCOPE[args.arm],
              "ref_cache": str(ref_work),
              "fixture": args.fixture, "seeds": seeds, "recycles": args.recycles,
              "steps": args.steps, "diffusion_samples": 1,
              "reference": "tt_bio._vendor.rf3 (upstream foundry), CPU torch, bf16 autocast",
              "kabsch_selftest_A": self_test,
              "metric_convention": "pharma_parity.noise_floor_verdict; floor = "
                                   "max(mean R, mean D); pairwise self-superposition, "
                                   "no shared frame",
              "work": str(work)}

    if args.rescore:
        per_seed = {}
        for s in seeds:
            z = np.load(work / f"seed{s}.npz")
            per_seed[s] = {k: z[k] for k in z.files}
        report["metrics"] = score(per_seed, seeds)
        report["timing_s"] = {str(s): [round(float(x), 1)
                                       for x in per_seed[s]["timing"]] for s in seeds}
        report["draws_sha"] = {str(s): str(per_seed[s]["shastr"][0]) for s in seeds}
        print(json.dumps(report, indent=2))
        if args.out:
            Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
        return 0

    from tt_bio.rf3.weights import load_reference
    t0 = time.time()
    net, cfg = load_reference(args.ckpt, num_steps=args.steps)
    report["ref_load_s"] = round(time.time() - t0, 1)

    if args.probe:
        return probe(args, net)

    if args.ref_only:
        for s in seeds:
            r = run_seed(args, net, None, s, work, ref_cache=None)
            print(f"[ref seed {s}] {'cached' if r.get('cached') else 'ran'} "
                  f"{[round(float(x), 1) for x in r['timing']]}", flush=True)
        report["ref_only"] = True
        print(json.dumps(report, indent=2))
        if args.out:
            Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
        return 0

    # Before rf3_model.load: both flags are read at construction, not at call time.
    report["arm_applied"] = apply_arm(args.arm)
    print(json.dumps(report["arm_applied"], indent=2), flush=True)

    # Patch the CLASS, not an instance: the kwarg is read in __init__, and one class patch
    # covers the 48-block trunk stack, the template embedder and the confidence head without
    # a post-hoc module walk having to rediscover where they are. Every kwarg on the signature
    # has a default and PairformerLayer passes them by keyword, so forcing one by name cannot
    # displace a positional.
    triatt_built = {"n": 0}
    report["one_k_chunk"] = args.one_k_chunk
    if args.one_k_chunk:
        import tt_bio.tenstorrent as _T
        _orig_init = _T.TriangleAttention.__init__

        def _init(self, *ar, **kw):
            kw["tri_att_one_k_chunk"] = True
            triatt_built["n"] += 1
            _orig_init(self, *ar, **kw)

        _T.TriangleAttention.__init__ = _init

    import ttnn
    from tt_bio.rf3 import model as rf3_model
    from tt_bio.tenstorrent import get_device

    dev = get_device()
    # The shipped compute kernel config, verbatim from worker.py's rf3 branch.
    kcfg = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    t0 = time.time()
    tt = rf3_model.load(args.ckpt, kcfg, num_timesteps=args.steps,
                        with_confidence="confidence_head" in cfg)
    report["port_load_s"] = round(time.time() - t0, 1)

    per_seed = {}
    for s in seeds:
        per_seed[s] = run_seed(args, net, tt, s, work, ref_cache=ref_work)
        print(f"[seed {s}] {'cached' if per_seed[s].get('cached') else 'ran'} "
              f"{[round(float(x), 1) for x in per_seed[s]['timing']]}", flush=True)

    report["metrics"] = score(per_seed, seeds)
    report["timing_s"] = {str(s): [round(float(x), 1) for x in per_seed[s]["timing"]]
                          for s in seeds}
    report["draws_sha"] = {str(s): str(per_seed[s]["shastr"][0]) for s in seeds}
    # The device coordinates themselves, hashed. draws_sha above is the RNG stream, which is
    # arm-independent by construction, so it cannot witness a kernel change: two arms that
    # differ agree on it. This is what an A/B on a bit-exact-or-not lever actually compares.
    report["dev_sha"] = {
        str(s): hashlib.sha256(
            np.ascontiguousarray(per_seed[s]["dev"], dtype=np.float64).tobytes()
        ).hexdigest()[:16] for s in seeds}
    import tt_bio.tenstorrent as _TT
    # A silent L1 refusal falls through to exactly the baseline config, so the served
    # (q_chunk, k_chunk, kv_buffer_factor) is the only proof the arm is the arm.
    report["triatt_fused_hifi_picks"] = {f"{q}x{k}": list(v) for (q, k), v
                                         in sorted(_TT.TRIATT_FUSED_HIFI_PICKS.items())}
    report["triatt_fused_hifi_stats"] = dict(_TT.TRIATT_FUSED_HIFI_STATS)
    report["triatt_instances_forced_one_k_chunk"] = triatt_built["n"]
    report["flags"] = resolved_flags()
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
