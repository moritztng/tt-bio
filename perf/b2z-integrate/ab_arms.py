#!/usr/bin/env python3
"""b2z: measure an arbitrary set of Boltz-2 levers, and their combination, as one paired A/B.

The b2z swarm screens levers on Wormhole; the published cell is a Blackhole p300c. Nothing a child
finds is a result until it is re-taken here, on qb2, against its own incumbent in the same process.
This is that instrument, and it is deliberately generic: a lever is named on the command line, not
compiled in, so a winner that lands at 03:00 can be measured without editing a harness first.

    ab_arms.py --out r.json --cifdir cif \
        --arm 'sdpa=tt_bio.tenstorrent._B2_CUSTOM_SDPA=1' \
        --arm 'bfp8=tt_bio.tenstorrent._PAIR_PROJ_BFP8=1' \
        --arm 'all=tt_bio.tenstorrent._B2_CUSTOM_SDPA=1,tt_bio.tenstorrent._PAIR_PROJ_BFP8=1' \
        --combined all --reps 5

Protocol constants (3 recycles, 200 steps, 1 sample, seed 0, diffusion_trace off) are IMPORTED from
`perf/b2x-flag-levers/ab_flag_levers.py` rather than copied, so this harness cannot silently drift
off the cell's protocol. The CIF directory layout it writes is the one
`perf/b2x-flag-levers/score298.py` and `domain_split.py` already read, so parity scoring is those
scripts unchanged.

Design, and why each piece is there:

* **Paired and interleaved.** `base` runs immediately before every arm, inside one process, one
  device open. The headline is the median of the per-pair ratios, not a ratio of two medians taken
  hours apart. Absolute seconds are reported but mean nothing off an idle benchlocked box.
* **A/A floor from the same data.** `base` appears once per arm per rep, so splitting its folds by
  position gives the session's own noise floor with no extra folds. A ratio inside the floor is not
  a result.
* **Defaults are snapshotted and restored between arms.** Every attribute any arm names is captured
  at startup and rewritten before each fold. Lever leakage between arms is the failure this
  prevents; it is invisible in the output when it happens.
* **No-observable-effect guard.** An arm that is inside the A/A floor AND writes base's CIF digest
  did not apply. That is reported as NO-OBSERVABLE-EFFECT, not as "1.00x, no win" -- the two look
  identical in a table and mean opposite things.
* **Additivity is measured, never summed.** With `--combined`, the harness prints the sum of the
  separate savings, the measured combined saving, and the discount between them.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import shutil
import socket
import statistics as st
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

_SPEC = importlib.util.spec_from_file_location(
    "_b2x_flag_levers", REPO / "perf" / "b2x-flag-levers" / "ab_flag_levers.py")
_B2X = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_B2X)          # import-safe: the file guards on __main__

FIX = REPO / "perf" / "size512" / "fixtures"
build_cfg, _seed_msa = _B2X.build_cfg, _B2X._seed_msa

OUT: dict = {}
OUT_PATH: Path | None = None


def dump() -> None:
    if OUT_PATH:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(OUT, indent=1))


# ---------------------------------------------------------------- lever parsing

def parse_value(s: str):
    """`1` -> 1, `1.5` -> 1.5, `true`/`false` -> bool, `none` -> None, anything else -> str."""
    low = s.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


# Protocol constants that may be moved per-arm. Deliberately a short allowlist: these change what
# the model COMPUTES, not how fast it computes it, so each one needs an accuracy argument of its
# own and none of them may be flipped as a default without Moritz. `sampling_steps` is here because
# b2z-sampler-steps measured the 200->50 quality case on 130 folds and the TIME half was left as a
# projection off the closed BH decomposition; this is what turns it into a paired measurement.
CFG_LEVERS = {"sampling_steps", "recycling_steps", "diffusion_samples"}


def parse_arm(spec: str) -> tuple[str, list[tuple[str, str, object]]]:
    """`name=mod.attr=v,env:VAR=v` -> (name, [(kind, target, value), ...]).

    kind is 'attr' (a module global, the form every tt-bio lever actually takes), 'env', or
    'cfg' (a protocol constant in the run config, e.g. `cfg:sampling_steps=50`).

    'cfg' exists because the protocol constants are baked into cfg by build_cfg BEFORE any arm
    runs, so setting the module global per-arm would silently do nothing -- every arm would fold
    at the same step count and the A/B would read 1.00x for a lever that is really worth 1.28x.
    """
    if "=" not in spec:
        return spec.strip(), []
    name, rest = spec.split("=", 1)
    levers = []
    for part in rest.split(","):
        part = part.strip()
        if not part:
            continue
        if part.lower().startswith("cfg:"):
            key, _, val = part[4:].partition("=")
            key = key.strip()
            if key not in CFG_LEVERS:
                raise SystemExit(f"cfg:{key} is not a protocol constant this harness can move; "
                                 f"known: {sorted(CFG_LEVERS)}")
            levers.append(("cfg", key, parse_value(val)))
            continue
        if part.lower().startswith("env:"):
            var, _, val = part[4:].partition("=")
            levers.append(("env", var.strip(), str(parse_value(val))))
            continue
        target, _, val = part.rpartition("=")
        if not target:
            raise SystemExit(f"lever {part!r} in arm {name!r} is not <mod.attr>=<value>")
        levers.append(("attr", target.strip(), parse_value(val)))
    return name.strip(), levers


def _set_cfg(cfg: dict, key: str, val, model=None) -> None:
    """Set a protocol constant everywhere it is actually read.

    THREE places, and the third is the one that matters. `build_cfg` stores
    sampling_steps/recycling_steps/diffusion_samples at the cfg top level and again inside
    cfg["predict_args"]. The boltz-2 path reads NEITHER per fold: `predict_one` calls
    `self.model.predict_step(batch)` with no protocol arguments, and `predict_step` reads
    `self.predict_args["sampling_steps"]` off the MODEL, populated once at `load_model(cfg)`.

    So mutating cfg alone is a silent no-op -- every arm folds at the shipped 200 steps while the
    report says 50, and the A/B reads 1.00x for a lever worth ~1.28x. That is exactly what this
    arm's first run did: 12 folds, base and arm identical to three decimals with the same CIF
    digest. The observed-step assertion in fold() is what stops that recurring silently.
    """
    if key in cfg:
        cfg[key] = val
    pa = cfg.get("predict_args")
    if isinstance(pa, dict) and key in pa:
        pa[key] = val
    mpa = getattr(model, "predict_args", None)
    if isinstance(mpa, dict) and key in mpa:
        mpa[key] = val


def split_target(dotted: str) -> tuple[str, str]:
    mod, _, attr = dotted.rpartition(".")
    if not mod:
        raise SystemExit(f"{dotted!r} needs a module path, e.g. tt_bio.tenstorrent._FLAG")
    return mod, attr


# ---------------------------------------------------------------- main

def summarize(warm: list[dict], arm_names: list[str], order_arms: list[str],
              combined: str | None) -> tuple[dict, dict]:
    """Turn the fold log into (summary, A/A floor). Pure: no device, no globals, unit-testable.

    `warm` is the timed folds in the order they ran, warmups already dropped.
    """
    def med(vals):
        vals = [v for v in vals if v is not None]
        return round(st.median(vals), 4) if vals else None

    summ: dict[str, dict] = {}
    for arm in arm_names:
        v = [r for r in warm if r["arm"] == arm]
        summ[arm] = {
            "n": len(v),
            "fold_s": med([r["fold_s"] for r in v]),
            "folds": [r["fold_s"] for r in v],
            "sampler_s": med([r["stages_s"].get("sampler") for r in v]),
            "ms_per_step": med([r["sampler_ms_per_step"] for r in v]),
            "trunk_s": med([r["prepare_and_trunk_s"] for r in v]),
            "confidence_s": med([r["stages_s"].get("confidence") for r in v]),
            "shape": v[0]["diffusion_shape"] if v else None,
            "cif_sha256": sorted({r["cif_sha256"] for r in v}),
            "plddt": sorted({round(float(r["plddt"]), 6) for r in v}),
        }

    # A/A floor: base folds split by their position in the rep, exactly the structure the arms are
    # compared through. With k arms there are k base folds per rep, so this is the spread the
    # session can resolve, not an estimate of it.
    bf = [r["fold_s"] for r in warm if r["arm"] == "base"]
    k = max(1, len(order_arms))
    halves = [m for m in (med(bf[j::k]) for j in range(k)) if m]
    floor = {
        "base_folds": bf,
        "position_medians_s": halves,
        "fold_AA_ratio": round(max(halves) / min(halves), 5) if len(halves) > 1 else None,
        "fold_spread_pct": round(100 * (max(bf) - min(bf)) / st.median(bf), 2) if bf else None,
    }
    aa = floor["fold_AA_ratio"] or 1.0

    # The headline: the median of the per-pair ratios. Each arm's fold is divided by the base fold
    # that ran immediately before it, so a drift in the box during the run cancels instead of
    # landing in one arm.
    base_digests = set(summ["base"]["cif_sha256"])
    for arm in order_arms:
        pairs = []
        for i, r in enumerate(warm):
            if r["arm"] == arm and i and warm[i - 1]["arm"] == "base":
                pairs.append(warm[i - 1]["fold_s"] / r["fold_s"])
        summ[arm]["paired_ratios"] = [round(x, 5) for x in pairs]
        summ[arm]["paired_speedup"] = round(st.median(pairs), 5) if pairs else None
        summ[arm]["median_speedup"] = (round(summ["base"]["fold_s"] / summ[arm]["fold_s"], 5)
                                       if summ[arm]["fold_s"] else None)
        summ[arm]["fold_saved_s"] = round(summ["base"]["fold_s"] - summ[arm]["fold_s"], 4)
        sp_ = summ[arm]["paired_speedup"] or 1.0
        same_cif = set(summ[arm]["cif_sha256"]) == base_digests
        summ[arm]["bit_exact_vs_base"] = same_cif
        if 1 / aa <= sp_ <= aa:
            summ[arm]["flag"] = (
                "NO-OBSERVABLE-EFFECT: inside the A/A floor and bit-identical to base. Check the "
                "lever applied before reading this as 1.00x." if same_cif else
                "INSIDE-AA-FLOOR: changes the answer but not the time")

    if combined:
        parts = [a for a in order_arms if a != combined]
        summed = round(sum(summ[a]["fold_saved_s"] for a in parts), 4)
        got = summ[combined]["fold_saved_s"]
        summ["additivity"] = {
            "parts": parts,
            "sum_of_separate_saved_s": summed,
            "measured_combined_saved_s": got,
            "discount": round(got / summed, 4) if summed else None,
            "note": "measured, not projected: the discount is an output of this run",
        }
    return summ, floor


def main() -> int:
    global OUT_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cifdir", type=Path, required=True)
    ap.add_argument("--arm", action="append", default=[], metavar="NAME=LEVER[,LEVER]",
                    help="repeatable. 'base' is implicit and is the incumbent.")
    ap.add_argument("--combined", default=None,
                    help="the arm that is every other arm at once; enables the additivity block")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--steps", type=int, default=_B2X.SAMPLING_STEPS)
    ap.add_argument("--recycles", type=int, default=_B2X.RECYCLING_STEPS)
    ap.add_argument("--fixture", default="cdk2x2_512")
    ap.add_argument("--control", default="cdk2x2_298",
                    help="the monomeric parity control; 'none' to skip it")
    ap.add_argument("--dry-run", action="store_true",
                    help="parse the arms, print the plan, open no device")
    args = ap.parse_args()

    arms: dict[str, list] = {"base": []}
    order_arms: list[str] = []
    for spec in args.arm:
        name, levers = parse_arm(spec)
        if name == "base":
            raise SystemExit("'base' is the implicit incumbent and cannot be redefined")
        if name in arms:
            raise SystemExit(f"duplicate arm {name!r}")
        arms[name] = levers
        order_arms.append(name)
    if not order_arms:
        raise SystemExit("no --arm given: there is nothing to measure against base")
    if args.combined and args.combined not in arms:
        raise SystemExit(f"--combined {args.combined!r} is not one of the arms")

    # base immediately before every arm: one pair per arm per rep, and >=2 base folds per rep so
    # the A/A floor falls out of the same data.
    order = [x for a in order_arms for x in ("base", a)]
    n_folds = len(arms) + args.reps * len(order)
    plan = {"arms": {k: [list(x) for x in v] for k, v in arms.items()},
            "order": order, "reps": args.reps, "folds_incl_warmup": n_folds,
            "combined": args.combined, "fixture": args.fixture, "control": args.control}
    print("[plan]", json.dumps(plan, indent=1), flush=True)
    if args.dry_run:
        OUT["plan"] = plan
        OUT["dry_run"] = True
        dump()
        return 0

    _B2X.SAMPLING_STEPS, _B2X.RECYCLING_STEPS = args.steps, args.recycles
    OUT_PATH = args.out
    OUT["plan"] = plan

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    from tt_bio.tenstorrent import get_device
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts
    from tt_bio import esmfold2 as _E
    _E.set_progress(lambda *a, **k: None)
    import tt_bio as _TB
    assert Path(_TB.__file__).resolve().is_relative_to(REPO), (
        f"imported tt_bio from {_TB.__file__}, not this worktree -- the venv's installed package "
        "would score a different tree (memory parity-gate-scores-installed-package-not-checkout)")

    # Resolve every lever site now, snapshot the shipped default, and refuse to run if the
    # environment already pins one: an env pin makes every arm the same arm, silently.
    sites: dict[str, object] = {}
    for name, levers in arms.items():
        for kind, target, _v in levers:
            if kind == "cfg":
                sites.setdefault(f"cfg:{target}", None)   # default filled in once cfg exists
                continue
            if kind == "env":
                if target in os.environ:
                    raise SystemExit(f"{target} is pinned in the environment; arm {name!r} sets it "
                                     "in-process and the pin would win")
                sites.setdefault(f"env:{target}", None)
                continue
            mod, attr = split_target(target)
            m = importlib.import_module(mod)
            if not hasattr(m, attr):
                raise SystemExit(f"{target} does not exist on this tree (arm {name!r})")
            sites[target] = getattr(m, attr)
    # cfg: sites cannot be resolved yet -- build_cfg has not run -- so print them separately once
    # they exist rather than printing a placeholder None that reads as "the default is None".
    print("[defaults]", json.dumps(
        {k: repr(v) for k, v in sites.items() if not k.startswith("cfg:")}, indent=1), flush=True)
    _pending_cfg = [k for k in sites if k.startswith("cfg:")]
    if _pending_cfg:
        print("[defaults] cfg levers resolved after build_cfg: "
              + ", ".join(sorted(_pending_cfg)), flush=True)

    dev = get_device()
    try:
        g = dev.compute_with_storage_grid_size()
        grid = [g.x, g.y]
    except Exception:
        grid = None
    OUT["env"] = {
        "host": socket.gethostname(),
        "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
        "lease_cards": os.environ.get("TT_BIO_LEASE_CARDS"),
        "grid": grid, "torch": torch.__version__,
        "arch": str(getattr(dev, "arch", lambda: "?")()),
        "tt_bio_file": _TB.__file__,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "loadavg": os.getloadavg(),
        "shipped_defaults": {k: repr(v) for k, v in sites.items()},
        "protocol": {"recycling_steps": args.recycles, "sampling_steps": args.steps,
                     "diffusion_samples": _B2X.DIFFUSION_SAMPLES, "seed": _B2X.SEED,
                     "diffusion_trace": False, "fixture": args.fixture,
                     "control": args.control},
    }
    try:
        import importlib.metadata as _md
        OUT["env"]["ttnn"] = _md.version("ttnn")
    except Exception:
        pass
    dump()

    work = Path(tempfile.mkdtemp(prefix="b2z-integrate-"))
    struct_dir = work / "out"; struct_dir.mkdir(parents=True)
    msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
    targets = [args.fixture] + ([] if args.control == "none" else [args.control])
    for name in targets:
        _seed_msa(FIX / f"{name}.yaml", (FIX / f"{name}.a3m").read_text(), msa_dir)

    cfg = build_cfg(msa_dir, struct_dir)
    for key in list(sites):
        if key.startswith("cfg:"):
            k = key[4:]
            if k not in cfg:
                raise SystemExit(f"cfg:{k} is not in this tree's run config: {sorted(cfg)}")
            sites[key] = cfg[k]
    OUT["cfg_defaults"] = {k[4:]: sites[k] for k in sites if k.startswith("cfg:")}
    if OUT["cfg_defaults"]:
        print("[defaults:cfg]", json.dumps(OUT["cfg_defaults"], indent=1), flush=True)
    _ensure_local_artifacts(cfg)
    t_load = time.perf_counter()
    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("b2z-integrate", cfg)
    OUT["model_load_s"] = round(time.perf_counter() - t_load, 3)
    dump()

    diff_mod = state.model.structure_module.score_model
    seen: dict = {}
    _orig_pop = diff_mod._populate_diffusion_cache

    def _pop(*a, **k):
        r = _orig_pop(*a, **k)
        seen["shape"] = tuple(int(x) for x in r)
        return r
    diff_mod._populate_diffusion_cache = _pop

    def apply_arm(name: str) -> None:
        """Restore every site to its shipped default, then apply this arm's overrides."""
        for key, default in sites.items():
            if key.startswith("cfg:"):
                _set_cfg(cfg, key[4:], default, state.model)
            elif key.startswith("env:"):
                os.environ.pop(key[4:], None)
            else:
                mod, attr = split_target(key)
                setattr(importlib.import_module(mod), attr, default)
        for kind, target, val in arms[name]:
            if kind == "cfg":
                _set_cfg(cfg, target, val, state.model)
            elif kind == "env":
                os.environ[target] = val
            else:
                mod, attr = split_target(target)
                setattr(importlib.import_module(mod), attr, val)

    def fold(arm: str, target: Path, keep: Path | None = None) -> dict:
        apply_arm(arm)
        try:
            diff_mod.reset_static_cache()
        except Exception:
            pass
        splitter = Splitter()
        state.pfn = splitter
        state.model.progress_fn = splitter
        for p in struct_dir.glob("*"):
            p.unlink() if p.is_file() else shutil.rmtree(p)
        seen.pop("shape", None)
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        metrics, _b, _f = state.predict_one(target, cfg)
        ttnn.synchronize_device(dev)
        wall = time.perf_counter() - t0
        stages = splitter.close()
        cifs = sorted(struct_dir.glob("*.cif"))
        assert cifs, "no CIF written"
        ps = None
        if len(splitter.loop) > 2:
            ps = round(1e3 * (splitter.loop[-1][1] - splitter.loop[0][1]) /
                       (splitter.loop[-1][0] - splitter.loop[0][0]), 4)
        # Did the fold run the protocol it was asked for? splitter.loop counts real progress
        # callbacks out of the diffusion loop, so this is OBSERVED, not the requested number read
        # back to itself. A protocol lever that never reaches the live model is otherwise
        # invisible: the arm folds at the default, the ratio reads 1.00x, and a broken instrument
        # is indistinguishable from a measured negative result.
        want_steps = int(cfg.get("sampling_steps") or 0)
        seen_steps = (splitter.loop[-1][0] + 1) if splitter.loop else 0
        if want_steps and seen_steps and abs(seen_steps - want_steps) > 1:
            raise SystemExit(
                f"arm {arm!r} asked for {want_steps} sampling steps but the diffusion loop ran "
                f"{seen_steps}. The protocol lever did not reach the live model -- "
                f"model.predict_args is what predict_step reads. Refusing to report a ratio "
                f"from a fold that did not run the protocol it claims.")
        steps_observed = seen_steps
        if keep:
            keep.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cifs[0], keep / cifs[0].name)
        return {
            "arm": arm, "target": target.stem,
            "diffusion_shape": seen.get("shape"),
            "fold_s": round(wall, 3),
            "prepare_and_trunk_s": round(splitter.marks[0][1] - t0, 4),
            "stages_s": stages,
            "sampler_ms_per_step": ps,
            "steps_observed": steps_observed,
            "plddt": metrics.get("complex_plddt", metrics.get("plddt")),
            "cif_sha256": hashlib.sha256(cifs[0].read_bytes()).hexdigest(),
            "loadavg1": round(os.getloadavg()[0], 2),
        }

    class Splitter:
        """Stage marks from the model's own progress emits, device-synchronised."""

        def __init__(self):
            self.marks, self.loop, self.n_diff = [], [], 0

        def _mark(self, label):
            ttnn.synchronize_device(dev)
            self.marks.append((label, time.perf_counter()))

        def __call__(self, stage=None, step=0, total=0, *a, **k):
            if stage == "diffusion":
                self.n_diff += 1
                if self.n_diff == 1:
                    self._mark("diffusion_conditioning")
                elif self.n_diff == 2:
                    self._mark("sampler")
                    self.loop.append((step, self.marks[-1][1]))
                else:
                    self.loop.append((step, time.perf_counter()))
            elif stage == "confidence":
                self._mark("confidence")

        def close(self):
            self._mark("end")
            return {lab: round(t1 - t0, 4)
                    for (lab, t0), (_l, t1) in zip(self.marks, self.marks[1:])}

    tgt = FIX / f"{args.fixture}.yaml"
    runs: list[dict] = []

    print("[warmup] one fold per arm, discarded (program cache, per-arm shapes)", flush=True)
    for arm in arms:
        r = fold(arm, tgt); r["warmup"] = True; runs.append(r)
        print(f"  warm {arm:12s} {r['fold_s']:7.3f}s shape={r['diffusion_shape']} "
              f"cif {r['cif_sha256'][:16]}", flush=True)
        OUT["timed"] = runs; dump()

    kept: dict[str, int] = {}
    for i in range(args.reps):
        for arm in order:
            n = kept[arm] = kept.get(arm, -1) + 1
            r = fold(arm, tgt, keep=args.cifdir / f"512_{arm}_{n}")
            r["warmup"] = False; r["rep"] = i; r["slot"] = n
            runs.append(r)
            print(f"  rep{i} {arm:12s} {r['fold_s']:7.3f}s "
                  f"sampler {r['stages_s'].get('sampler')} {r['sampler_ms_per_step']}ms/step "
                  f"plddt {r['plddt']} cif {r['cif_sha256'][:16]} load {r['loadavg1']}", flush=True)
            OUT["timed"] = runs; dump()

    warm = [r for r in runs if not r["warmup"]]
    summ, floor = summarize(warm, list(arms), order_arms, args.combined)
    OUT["summary"] = summ
    OUT["AA_floor"] = floor
    dump()
    print("\n[AA floor]", json.dumps(floor), flush=True)
    print("[summary]", json.dumps(summ, indent=1), flush=True)

    # ---- parity control ----------------------------------------------------
    if args.control != "none":
        ctl = []
        print(f"\n[control] {args.control}", flush=True)
        ctgt = FIX / f"{args.control}.yaml"
        for arm in list(arms) + ["base"]:
            tag = f"{arm}_{sum(1 for c in ctl if c['arm'] == arm)}"
            r = fold(arm, ctgt, keep=args.cifdir / f"298_{tag}")
            r["tag"] = tag
            ctl.append(r)
            print(f"  {tag:14s} {r['fold_s']:7.3f}s shape={r['diffusion_shape']} "
                  f"plddt {r['plddt']} cif {r['cif_sha256'][:16]}", flush=True)
            OUT["control"] = ctl; dump()

    OUT["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    dump()
    print("\nwrote", OUT_PATH, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
