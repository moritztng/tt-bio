#!/usr/bin/env python3
"""Nesso-1's lever census, and what the caught L1 refusal at 532 tokens actually costs.

At 532 tokens the bf16 arm prints one `TT_THROW: Statically allocated circular buffers in
program N clash with L1 buffers` and carries on. Nothing in the reported number says which op
refused or what the fold ran instead, and `tt-bio affinity` folds in the process you launch it
from, so the subprocess census in `scripts/lever_census.py` has nothing to attach to. This
drives the same census table in-process against `tt_bio.nesso1.Nesso1.predict`, and adds two
things the release-gate arm cannot ask for:

  * `--trace-clash` wraps every ttnn entry point, so a throw names its op and its tt-bio call
    site even though something downstream catches it. That is how the 532-token clash was
    pinned to `_pair_proj_linear`'s L1-destination leg at the trimul out-projection.
  * the fused triangle-attention SDPA's plan terms per (shape, q_chunk), so a decline on
    `fill_preconditions` names the clause instead of just the count.

Usage (one device context per process, card pinned by the caller):
  TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=... NESSO_CACHE=... <env>/bin/python \
      scripts/nesso1_port/l1_probe.py --rung aa512 --repeats 3 --trace-clash
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tt_bio.nesso1_input import CLI_PREDICT_ARGS, collate, prepare  # noqa: E402

FEAT_SEED = 20260820

SCALARS = (
    "affinity_pred_value",
    "affinity_pred_value1",
    "affinity_pred_value2",
    "affinity_logits_binary",
    "affinity_probability_binary",
    "entropy_pp",
    "entropy_pl",
    "entropy_ll",
    "entropy_crop_pp",
    "entropy_crop_pl",
    "entropy_crop_ll",
)

# Everything triatt_sdpa.sdpa asserts before it will build the descriptor, so a decline on
# `fill_preconditions` can name its clause.
FILL_TERMS = ("nh_per_core", "q_per_core", "bcast_batch", "use_padded_mask", "NKH", "NVH", "NQH")


def census():
    """`scripts/lever_census.py` loaded as a module: one lever table, two drivers."""
    spec = importlib.util.spec_from_file_location(
        "_lever_census", REPO / "scripts" / "lever_census.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def featurize(yaml_path: Path, scratch: Path) -> dict:
    ds, _, failed = prepare(yaml_path, scratch, ccd_pkl=None, num_workers=0, esm_cache=None)
    if failed:
        raise SystemExit(f"preprocessing failed for {failed}")
    torch.manual_seed(FEAT_SEED)
    item = ds[0]
    if item.get("exception"):
        raise SystemExit(f"featurizer raised on {yaml_path.name}")
    return collate(item)


def trace_clashes(needle: str = "clash with L1 buffers") -> list:
    """Wrap every ttnn entry point so a throw names its op and its tt-bio call site.

    The clash is caught somewhere -- that is the point of the probe -- so it never reaches a
    traceback. Wrapping the raising side instead of hunting the catching side finds it in one
    fold no matter which of the repo's L1 fallbacks swallowed it.
    """
    import inspect

    import ttnn

    hits: list = []

    def wrap(mod, name, fn, label):
        def wrapped(*a, **kw):
            try:
                return fn(*a, **kw)
            except Exception as exc:                                     # noqa: BLE001
                if needle in str(exc):
                    frames = [f"{Path(f.filename).name}:{f.lineno} {f.name}"
                              for f in traceback.extract_stack()[:-1] if "tt_bio" in f.filename]
                    hits.append({"op": label, "site": frames[-6:]})
                raise
        try:
            setattr(mod, name, wrapped)
        except Exception:                                                # noqa: BLE001
            pass

    for mod, prefix in ((ttnn, "ttnn"), (ttnn.experimental, "ttnn.experimental"),
                        (ttnn.transformer, "ttnn.transformer")):
        for name in dir(mod):
            if name.startswith("_"):
                continue
            fn = getattr(mod, name, None)
            if not callable(fn) or inspect.isclass(fn) or inspect.ismodule(fn):
                continue
            wrap(mod, name, fn, f"{prefix}.{name}")
    return hits


def instrument_triatt():
    """Every fused-SDPA attempt: (Sq, Sk, q_chunk, k_chunk) -> served, or the clause it refused."""
    from tt_bio import sdpa_generic as SG
    from tt_bio import triatt_sdpa as T

    calls: dict = {}
    inner = T.sdpa
    plan_inner = SG.plan
    last_plan: dict = {}

    def plan_wrapped(*a, **kw):
        p = plan_inner(*a, **kw)
        last_plan.clear()
        last_plan.update({t: p[t] for t in FILL_TERMS if t in p})
        last_plan["mask_cb_tiles"] = p["k_num_chunks"] * p["Sq_chunk_t"] * p["Sk_chunk_t"]
        last_plan["stock_mask_cb_tiles"] = 2 * p["Sq_chunk_t"] * p["Sk_chunk_t"]
        last_plan["batch_per_core"] = p["batch_per_core"]
        return p

    SG.plan = plan_wrapped
    T.SG.plan = plan_wrapped

    def wrapped(q, k, v, bias, scale, q_chunk, k_chunk, ckc_default=None):
        rejects_before = dict(T.REJECTS)
        last_plan.clear()
        out = inner(q, k, v, bias, scale, q_chunk, k_chunk, ckc_default)
        key = (int(q.shape[2]), int(k.shape[2]), int(q_chunk), int(k_chunk))
        row = calls.setdefault(key, {"served": 0, "declined": 0, "reasons": {}})
        if last_plan and "plan" not in row:
            row["plan"] = dict(last_plan)
            row["mask_shape"] = [int(d) for d in bias.shape]
            row["q_shape"] = [int(d) for d in q.shape]
        if out is not None:
            row["served"] += 1
        else:
            row["declined"] += 1
            for r, n in T.REJECTS.items():
                if n > rejects_before.get(r, 0):
                    row["reasons"][r[0]] = row["reasons"].get(r[0], 0) + 1
        return out

    T.sdpa = wrapped
    return calls


def triatt_report(calls: dict) -> dict:
    from tt_bio import tenstorrent as TT
    from tt_bio import triatt_sdpa as T

    return {
        "enabled": T._ENABLED,
        "q_split": T._Q_SPLIT,
        "served": T.STATS[0],
        "declined": T.STATS[1],
        "pm_over_l1": sorted(map(list, T._PM_OVER_L1)),
        "sdpa_q_chunk_over_l1": sorted(map(list, TT._SDPA_Q_CHUNK_OVER_L1)),
        "calls": {f"Sq{a}_Sk{b}_q{c}_k{d}": v for (a, b, c, d), v in sorted(calls.items())},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", default="aa512")
    ap.add_argument("--inputs", type=Path, default=REPO / "perf/nesso1/inputs/ladder")
    ap.add_argument("--scratch", type=Path, default=Path("~/scratch/nesso1/ladder").expanduser())
    ap.add_argument("--weights", default="recursionpharma/nesso")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", type=Path, default=REPO / "perf/nesso1/l1_probe")
    ap.add_argument("--trace-clash", action="store_true",
                    help="wrap every ttnn op so an L1 clash names its op and call site")
    args = ap.parse_args()

    arm = "pm_on" if os.environ.get("TT_BIO_TRIATT_PERSISTENT_MASK", "1") == "1" else "pm_off"
    yamls = sorted((args.inputs / args.rung).glob("*.yaml"))
    if len(yamls) != 1:
        raise SystemExit(f"expected one yaml in {args.inputs / args.rung}")
    feats = featurize(yamls[0], args.scratch / args.rung)
    n_tokens = int(feats["token_pad_mask"].shape[-1])

    lc = census()
    lc._install_wraps()
    calls = instrument_triatt()
    clashes = trace_clashes() if args.trace_clash else None

    from tt_bio.nesso1 import Nesso1

    model = Nesso1.from_pretrained(
        args.weights, use_tenstorrent=True, trunk_fp32=False, affinity_fp32=True)
    model.use_kernels = False
    model.predict_args.update(CLI_PREDICT_ARGS)

    times, runs = [], []
    for rep in range(args.repeats):
        t0 = time.perf_counter()
        with torch.no_grad():
            pred = model.predict(feats)
        times.append(time.perf_counter() - t0)
        runs.append({k: float(pred[k].reshape(-1)[0]) for k in SCALARS if k in pred})
        print(f"  {arm} rep{rep}: {times[-1]:.3f}s", flush=True)

    levers = lc._snapshot_process()
    dark = sorted(f for f, r in levers.items()
                  if r["served"] == 0 and (r["declined"] or 0) > 0)
    report = {
        "gate": "nesso1_l1_probe",
        "arm": arm,
        "rung": args.rung,
        "n_tokens": n_tokens,
        "trunk": "bf16",
        "affinity": "fp32",
        "host": os.uname().nodename,
        "card": os.environ.get("TT_VISIBLE_DEVICES"),
        "grid": lc._compute_grid(),
        "wall_s": times,
        "warm_wall_s": min(times[1:]) if len(times) > 1 else times[0],
        "scalars": runs[0],
        "scalars_all_reps_equal": all(r == runs[0] for r in runs),
        "levers": levers,
        "dark_levers": dark,
        "triatt": triatt_report(calls),
        "clashes": clashes,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.rung}_{arm}.json"
    path.write_text(json.dumps(report, indent=2, default=str) + "\n")
    if clashes:
        print(json.dumps(clashes, indent=2))
    print(f"dark levers: {', '.join(dark) or 'none'}")
    print(f"PAIR_PROJ_L1_OUT: {levers.get('PAIR_PROJ_L1_OUT')}")
    print(f"TRIATT_PERSISTENT_MASK: {levers.get('TRIATT_PERSISTENT_MASK')}")
    print(f"warm {report['warm_wall_s']:.3f}s -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
