"""The upstream side of the G3 curve leg: run a :class:`CurvePlan` through upstream's own step.

Same model, same loss, same optimizers (Muon + Adam, upstream's `build_optimizers` and its
gradient clipping) as `rfd4-train`, but the data order, the per-step `t`/noise seeds and the
recycle depth come from the plan instead of from the dataloader and the process RNG. That is
what makes the two sides comparable: upstream's own loop decides its order from its sampler,
which moves with any config change, so a TT run could never be matched against it.

Three uses, one script:

    --make-plan PLAN --steps N --crop C      scan the dataset, write the plan both sides read
    --plan PLAN --out LOG --side gpu_a       run it, appending one JSONL line a step
    --plan PLAN --out LOG --side gpu_b --seed-offset 1    the same plan at a second seed

`--side gpu_b` is the seed control the bar needs. It differs from `gpu_a` only in the weights'
init draw and the per-step seeds (`--seed-offset`), never in the data order.

Grade with `perf/r4t/grade_parity.py --ours OURS --gpu-a A --gpu-b B`.

    RFD4_REPO=~/rfd4-upstream/RFD4-Proteina DATA_PATH=~/r4t/objective/data HF_HOME=~/r4t/gpu-baseline/hf \\
      ~/rfd4-cpu-venv/bin/python perf/r4t/curve_upstream.py --plan plan.json --out gpu_a.jsonl --side gpu_a
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from loss_parity_reference import HF_REPO, HF_REV, build_model, flat, load_batch  # noqa: E402
from tt_bio.train.curve import CurveLog, CurvePlan  # noqa: E402
from tt_bio.train.parity import GradientCoverage, zero_or_missing  # noqa: E402

SCHEMA = "rfproteina.datapipes.schema.proteina.ProteinaDataSample"


def example_seed(plan_seed: int, example_id: str) -> int:
    """The plan's seed for one example's pipeline draw. By id, so order never changes a crop."""
    return (plan_seed ^ int(hashlib.sha256(example_id.encode()).hexdigest()[:8], 16)) % (2**31)


def _upstream_sha(repo: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def dataset_index(cfg, min_tokens: int):
    """example_id -> dataset index, for every system upstream's pipeline accepts."""
    from hydra.utils import instantiate
    ds = instantiate(cfg.datasets.train.test_dataset.dataset)
    index = {}
    for i in range(len(ds)):
        try:
            s = ds[i]
        except Exception as e:
            print(f"skip {i}: {type(e).__name__}: {str(e)[:100]}", file=sys.stderr)
            continue
        index[str(s["example_id"])] = i
    return ds, index


def collate(ds, i, crop, k, seed=None):
    """One example through upstream's pipeline, at a seed the plan fixes.

    Upstream's design-task transform draws from the GLOBAL numpy RNG while the example is being
    built (`transforms/design_task.py`: coverage fraction, which chains and residues are
    conditioned, which motif, which tip atoms), so two processes running the same plan see
    different conditioning unless that RNG is pinned here. Measured 2026-11-26: without this,
    three legs of the same 20-step plan disagreed on step 0's loss by up to 7 %, and one of them
    lost the insertion supervision entirely, which the G2 guard then refused. Seeding per example
    id (not per step) keeps the cache sound: an id always yields the same crop and conditioning.
    """
    import random

    import numpy as np
    import torch
    from rfproteina.datapipes.collate import collate_batch

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
    b = collate_batch([ds[i]], schema=SCHEMA, fixed_dimensions={"I": crop},
                      keep_atom_array=False, num_replicates=k)
    return b["feats"] if "feats" in b else b


def make_plan(cfg, args) -> CurvePlan:
    """Choose the data order once, from the ids the pipeline actually yields at this crop."""
    import torch
    ds, index = dataset_index(cfg, args.min_tokens)
    wanted = [e for e in args.ids.split(",") if e] or list(index)
    missing = [e for e in wanted if e not in index]
    if missing:
        raise SystemExit(f"--ids names examples the pipeline does not yield: {missing}")
    ids = []
    for eid in wanted:
        i = index[eid]
        feats = collate(ds, i, args.crop, args.exploration_k, seed=example_seed(args.seed, eid))
        tokens = int(feats["padding_mask"][0].sum())
        if tokens < args.min_tokens:
            print(f"skip {eid}: {tokens} tokens < {args.min_tokens}", file=sys.stderr)
            continue
        ids.append(eid)
        if args.max_ids and len(ids) >= args.max_ids:
            break
    if not ids:
        raise SystemExit("no example survived the pipeline at this crop")
    del torch
    exempt = tuple(tuple(line.split("=", 1)) for line in args.exempt.splitlines() if line.strip())
    return CurvePlan(steps=args.steps, crop=args.crop, example_ids=tuple(ids),
                     batch_size=args.batch_size, exploration_k=args.exploration_k,
                     warmup=args.warmup, seed=args.seed, exempt=exempt, note=args.note)


def to_device(x, device):
    """Move a (nested) batch to the network's device; non-tensors pass through."""
    import torch
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    return x


def corrupt(model, batch, spec, k):
    """The network input of one plan step: upstream's clean sample, then `t` and noise at the
    step's seeds. Weight-independent, so the TT leg can take it from a dump (`--dump-dir`) and
    see the tensors this script trains on, not a re-derivation of them."""
    import torch
    fm = model.fm
    batch["coords_nm"] = batch["coords"] / 10.0
    batch["coord_mask"] = batch["atom37_mask"]
    batch["strict_feats"] = True
    with torch.no_grad():
        batch = model.add_clean_samples(batch)
        torch.manual_seed(spec.t_seed)
        t = fm.sample_t(shape=(batch["padding_mask"].shape[0] // k,),
                        device=batch["padding_mask"].device)
        torch.manual_seed(spec.noise_seed)
        return fm.corrupt_batch(batch, t={dm: v.repeat_interleave(k) for dm, v in t.items()})


def train_step(model, batch, spec, k, optimizers, schedulers, clip_val, device):
    """Upstream's training step, with the plan's t, noise and recycle depth.

    The clean sample, t and the noise are drawn on the CPU and only then moved to `device`:
    CUDA's generator gives different numbers for the same seed, so drawing on the GPU would
    hand the GPU legs a different corruption than the TT leg draws from the same plan.
    """
    import torch
    from rfproteina.utils.torch import minimum_loss_attempt_indices

    fm = model.fm
    batch = to_device(corrupt(model, batch, spec, k), device)
    nn_out = model.call_nn(batch, n_recycle=spec.n_recycle)
    losses = fm.compute_loss(batch=batch, nn_out=nn_out)
    names = model.cfg_exp.training.get("exploration_losses", fm.data_modes)
    win = minimum_loss_attempt_indices(losses=losses, loss_names=list(names),
                                       attempts_per_example=k)
    losses = {n: v[win] for n, v in losses.items()}
    terms = {n: v.mean() for n, v in fm.optimization_terms(losses).items()}
    loss = sum(terms.values())
    loss.backward()
    dead = zero_or_missing({n: p.grad for n, p in model.nn.named_parameters() if p.requires_grad})
    for opt in optimizers:
        torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], clip_val)
        opt.step()
        opt.zero_grad(set_to_none=True)
    for sch in schedulers:
        sch.step()
    return float(loss.detach()), {n: float(v.detach()) for n, v in terms.items()}, dead


def dump(model, ds, index, plan, args):
    """Write each plan step's network input as `step<NNNNN>.npz` in `load_batch`'s format: the
    corrupted batch, the torch RNG state `call_nn` starts from, and the step's recycle depth.

    The TT leg trains on these files, so both sides see the same tensors by construction rather
    than by two implementations of the plan agreeing. `--check-dump` reruns step 0 from its file
    and must land on this script's own step-0 loss (no update has happened before step 0)."""
    import numpy as np
    import torch
    out = Path(args.dump_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(json.dumps(asdict(plan), indent=1))
    n = args.dump_steps or plan.steps
    cache = {}
    for i in range(n):
        spec = plan.step(i)
        eid = spec.example_ids[0]                 # batch_size 1 in the plans we run today
        if eid not in cache:
            cache[eid] = collate(ds, index[eid], plan.crop, plan.exploration_k,
                                 seed=example_seed(plan.seed, eid))
        b = corrupt(model, {k: (v.clone() if torch.is_tensor(v) else v) for k, v in cache[eid].items()},
                    spec, plan.exploration_k)
        np.savez_compressed(out / f"step{i:05d}.npz", **{f"in/{k}": v for k, v in flat(b).items()},
                            rng_state=torch.get_rng_state().numpy(), winners=np.array([-1]),
                            depth=np.int64(spec.n_recycle), seed=np.int64(spec.noise_seed),
                            plan_digest=np.array(plan.digest()), example_id=np.array(eid))
    print(f"dumped {n} steps of plan {plan.digest()} to {out}")


def check_dump(path: Path, log: Path) -> int:
    """Step 0 from its dump, at the released weights, against the step-0 loss a run of the same
    plan logged. Exit 0 when they agree to the replay floor, 1 otherwise, 2 when the log ran
    a different plan."""
    import numpy as np
    import torch
    from rfproteina.utils.torch import minimum_loss_attempt_indices
    from loss_parity_reference import REPLAY_TOL
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    _, model = build_model(Path(os.environ["RFD4_REPO"]).expanduser(), torch.float32)
    fm = model.fm
    batch, meta, _, _ = load_batch(path / "step00000.npz", torch.float32)
    k = json.loads((path / "plan.json").read_text())["exploration_k"]
    torch.set_rng_state(meta["rng_state"])
    with torch.no_grad():
        losses = fm.compute_loss(batch=batch, nn_out=model.call_nn(batch, n_recycle=meta["depth"]))
        names = model.cfg_exp.training.get("exploration_losses", fm.data_modes)
        win = minimum_loss_attempt_indices(losses=losses, loss_names=list(names), attempts_per_example=k)
        loss = float(sum(v.mean() for v in fm.optimization_terms({n: v[win] for n, v in losses.items()}).values()))
    lines = [json.loads(ln) for ln in log.read_text().splitlines() if ln.strip()]
    dumped = str(np.load(path / "step00000.npz")["plan_digest"])
    if lines[0].get("plan_digest") != dumped:
        print(f"REFUSED: {log} ran plan {lines[0].get('plan_digest')}, the dump is plan {dumped}")
        return 2
    logged = next(r["loss"] for r in lines if r.get("step") == 0)
    rel = abs(loss - logged) / abs(logged)
    ok = rel <= REPLAY_TOL
    print(json.dumps({"dump_step0_loss": loss, "logged_step0_loss": logged, "rel": rel,
                      "tol": REPLAY_TOL, "verdict": "MATCH" if ok else "MISMATCH"}))
    return 0 if ok else 1


def run(args):
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    device = torch.device(args.device)
    if device.type == "cuda":
        # fp32 legs stay fp32: TF32 would silently drop the matmuls to a 10-bit mantissa.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    repo = Path(os.environ["RFD4_REPO"]).expanduser()
    plan = CurvePlan.from_dict(json.loads(Path(args.plan).read_text()))
    cfg, model = build_model(repo, torch.float32)
    model = model.to(device)
    if getattr(model, "autoencoder", None) is not None:
        # The frozen AE only produces the clean sample, and corrupt() runs add_clean_samples on
        # the CPU batch, so moving it to cuda with the rest crashes the leg at step 0
        # (r4t-gpu-baseline, 2026-11-26). It stays on the host; nothing else changes.
        model.autoencoder.to("cpu")
    ds, index = dataset_index(cfg, 0)
    missing = [e for e in plan.example_ids if e not in index]
    if missing:
        raise SystemExit(f"the plan names examples this dataset does not have: {missing[:4]}")

    if args.dump_dir:
        return dump(model, ds, index, plan, args)

    torch.manual_seed(plan.seed + args.seed_offset)
    opts = model.configure_optimizers()
    optimizers, schedulers = opts if isinstance(opts, tuple) else (opts, [])
    optimizers = optimizers if isinstance(optimizers, list) else [optimizers]
    clip_val = float(model.cfg_exp.opt.gradient_clip_val)
    model.train()

    # The plan's exemptions, not the run's: a parameter this id set cannot drive is named with
    # upstream's reason in the plan, before the run, and refusing is the default for everything else.
    guard = GradientCoverage([n for n, p in model.nn.named_parameters() if p.requires_grad],
                             exempt=dict(plan.exempt), where=f"{args.side} curve leg: ")
    meta = {"upstream_sha": _upstream_sha(repo), "hf": f"{HF_REPO}@{HF_REV}",
            "seed_offset": args.seed_offset, "data_seed": "plan.seed ^ sha256(example_id)", "dtype": "float32",
            "device": (f"cuda:{torch.cuda.get_device_name(device)}" if device.type == "cuda" else str(device)),
            "tf32": bool(device.type == "cuda" and torch.backends.cuda.matmul.allow_tf32),
            "threads": torch.get_num_threads(), "sha": args.sha, "host": socket.gethostname(),
            "optimizers": [type(o).__name__ for o in optimizers], "clip": clip_val,
            # Which feature each exempt creator index is, so an exemption's reason is checkable
            # against upstream's code rather than taken on the writer's word.
            "feat_creators": [type(c).__name__ for c in
                              getattr(getattr(model.nn, "init_repr_factory", None),
                                      "feat_creators", [])]}
    cache = {}
    with CurveLog(args.out, plan, args.side, meta=meta) as log:
        for i in range(plan.steps):
            spec = plan.step(i)
            t0 = time.time()
            feats = {}
            for eid in spec.example_ids:          # batch_size 1 in the plans we run today
                if eid not in cache:
                    cache[eid] = collate(ds, index[eid], plan.crop, plan.exploration_k,
                                         seed=example_seed(plan.seed, eid))
                feats = {k: (v.clone() if torch.is_tensor(v) else v)
                         for k, v in cache[eid].items()}
            loss, terms, dead = train_step(model, feats, spec, plan.exploration_k,
                                           optimizers, schedulers, clip_val, device)
            # G2 rides along: the depth-1 steps train neither cycle projection, so they are not
            # observed (state/r4t/rows/accuracy.md; counting them refuses ~1 healthy run in 150).
            if spec.n_recycle >= 2 and not guard.done:
                guard.observe(dead)
            log.record(i, loss, terms=terms, seconds=round(time.time() - t0, 2),
                       dead=len(dead), example=spec.example_ids[0],
                       **({"peak_gb": round(torch.cuda.max_memory_allocated(device) / 2**30, 2)}
                          if device.type == "cuda" else {}))
            print(f"step {i} loss {loss:.4f} depth {spec.n_recycle} dead {len(dead)} "
                  f"{time.time() - t0:.1f}s", flush=True)
    print(f"wrote {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plan", required=True, help="plan JSON to run, or to write with --make-plan")
    ap.add_argument("--make-plan", action="store_true")
    ap.add_argument("--out", help="curve log to append to")
    ap.add_argument("--side", choices=("ours", "gpu_a", "gpu_b"), default="gpu_a")
    ap.add_argument("--seed-offset", type=int, default=0,
                    help="second seed for the gpu_b control; it moves the step's t and noise, "
                         "never the data (the pipeline draw is pinned per example id)")
    ap.add_argument("--device", default="cpu", help="where the network runs (cpu, cuda); data, t and "
                    "noise are always drawn on the CPU so every side draws the same numbers")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--exploration-k", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--seed", type=lambda v: int(v, 0), default=0x47330000,
                    help="the plan's seed: per-step t/noise and the per-example pipeline draw")
    ap.add_argument("--min-tokens", type=int, default=32)
    ap.add_argument("--max-ids", type=int, default=0, help="cap the plan's id list (0: all)")
    ap.add_argument("--ids", default="", help="comma-separated example ids for the plan, in "
                    "order, instead of every id the pipeline yields")
    ap.add_argument("--exempt", default="", help="name=reason pairs, one a line (reasons carry "
                    "punctuation, so lines, not a separator char): parameters this plan's data "
                    "cannot drive (see CurvePlan.exempt)")
    ap.add_argument("--note", default="")
    ap.add_argument("--dump-dir", default="", help="with --plan: write every step's network input "
                    "here for the TT leg instead of training (see dump())")
    ap.add_argument("--dump-steps", type=int, default=0, help="dump only the first N steps (0: all)")
    ap.add_argument("--check-dump", default="", help="LOG: rerun step 0 from --dump-dir and match "
                    "the step-0 loss LOG recorded for the same plan")
    ap.add_argument("--sha", default=os.environ.get("TT_BIO_SHA", ""))
    a = ap.parse_args()
    if a.make_plan:
        import torch  # noqa: F401  (upstream's stack needs it imported before hydra)
        repo = Path(os.environ["RFD4_REPO"]).expanduser()
        from hydra import compose, initialize_config_dir
        sys.path.insert(0, str(repo / "src"))
        with initialize_config_dir(config_dir=str(repo / "configs"), version_base=None):
            cfg = compose(config_name="training_proteina",
                          overrides=["experiment=tutorial/finetune_test_dataset", "+single=true",
                                     "+nolog=true"])
        plan = make_plan(cfg, a)
        Path(a.plan).write_text(json.dumps(asdict(plan), indent=1))
        print(f"{a.plan}: {plan.steps} steps, crop {plan.crop}, {len(plan.example_ids)} ids, "
              f"digest {plan.digest()}, graded={plan.graded}")
        return
    if a.check_dump:
        sys.exit(check_dump(Path(a.dump_dir), Path(a.check_dump)))
    if not a.out and not a.dump_dir:
        raise SystemExit("--out is required when running a plan")
    run(a)


if __name__ == "__main__":
    main()
