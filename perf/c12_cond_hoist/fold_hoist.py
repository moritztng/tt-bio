#!/usr/bin/env python3
"""The folds and the fold A/B that flipping TT_BIO_DIT_COND_HOIST on by default needs.

Same protocol and the same scorer as `perf/b2z2_snorm/fold_snorm.py`, which this is a copy of with
the `BOLTZ2_ADALN_SHARED_SNORM` arm removed: that lever was NO-GO'd as a strict subset of this one
and its code is gone from main, so an arm that set it would set an attribute nothing reads.

Why the numbers are re-taken rather than quoted. The only accuracy screen on record for this lever
(`state/concluded/roof-difftx-arith-efficiency`, 0.1233 A) ran on pc card 0, which
`state/pc-card0-down` bars for parity, and the fault there is size-dependent and was silent at the
two sizes carrying the number. The fold ratio was re-measured on qb2 card 0 by
`state/concluded/b2z2-adaln-snorm-ship` (1.01503x) but that row declined to flip the default, so
this row owns both readings and takes them itself.

The lever is a MODULE-LEVEL gate read at call time (`tt_bio.tenstorrent._B2_DIT_COND_HOIST`), so the
arm is selected by setting that attribute per fold and the env var is asserted absent: a pinned
value would be baked into the module at import and would serve every arm.

Arms, in ONE process so they share a device, a program cache and a model load:

    base      _B2_DIT_COND_HOIST = False, each of the 24 layers takes its own conditioning path
    hoist     _B2_DIT_COND_HOIST = True, one layer_norm + two concatenated matmuls for the stack
    default   the checkout untouched. Its digest must equal the arm the default is supposed to
              select, or the default is not what this run measured.

`hoist_norms` counts the hoisted stacks a fold actually took, so an arm whose lever never fired is
distinguishable from an arm whose lever fired and was swallowed.

    fold_hoist.py --out <json> --cifdir <dir> [--plan base:0,1,2,3;hoist:0,1,2,3]
"""
import argparse
import hashlib
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
sys.path.insert(0, str(REPO / "perf" / "b2x-flag-levers"))

import ab_flag_levers as AB  # noqa: E402  -- the fixtures, cfg and MSA seeding, unmodified

FLAG_HOIST = "TT_BIO_DIT_COND_HOIST"
ATTR_HOIST = "_B2_DIT_COND_HOIST"

# arm -> the value of `_B2_DIT_COND_HOIST`, or None for "touch nothing". `default` tests the
# SHIPPED DEFAULT rather than a value this driver sets: flipping a default and then measuring an arm
# that overrides it proves nothing about the default.
ARMS = {"base": False, "hoist": True, "default": None}


def check_arms(arms, T):
    """Refuse to measure a lever this checkout does not have.

    Setting an attribute nothing reads succeeds silently, so both arms would run the same code and
    score as a null. Check the reader exists before folding anything.
    """
    for arm in arms:
        assert arm in ARMS, f"unknown arm {arm}"
    assert hasattr(T, ATTR_HOIST), (
        f"this checkout has no tt_bio.tenstorrent.{ATTR_HOIST}; both arms would be identical")
    assert hasattr(T, "DiffusionTransformer"), "no DiffusionTransformer to gate"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cifdir", type=Path, required=True)
    ap.add_argument("--plan", default="base:0,1,2,3;hoist:0,1,2,3")
    ap.add_argument("--sizes", default="512")
    ap.add_argument("--steps", type=int, default=AB.SAMPLING_STEPS)
    ap.add_argument("--recycles", type=int, default=AB.RECYCLING_STEPS)
    ap.add_argument("--timing-reps", type=int, default=0,
                    help="run the TIMING protocol instead of the seed plan: this many reps of "
                         "--timing-arms at one seed, in one process")
    ap.add_argument("--block-timing", action="store_true",
                    help="wall each token-level DiffusionTransformer call. Inserts two syncs per "
                         "block, so the "
                         "fold times it produces are NOT publishable as absolute numbers")
    ap.add_argument("--timing-arms", default="base,hoist,base",
                    help="arms per rep, in order. base at two positions gives the A/A floor")
    ap.add_argument("--timing-seed", type=int, default=0)
    args = ap.parse_args()

    plan = {}
    for part in args.plan.split(";"):
        arm, _, seeds = part.partition(":")
        assert arm in ARMS, f"unknown arm {arm}"
        plan[arm] = [int(s) for s in seeds.split(",")]
    sizes = args.sizes.split(",")

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio.tenstorrent as T
    from tt_bio.tenstorrent import get_device
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts
    from tt_bio import esmfold2 as _E
    _E.set_progress(lambda *a, **k: None)
    import tt_bio as _TB
    assert Path(_TB.__file__).resolve().is_relative_to(REPO), (
        f"imported tt_bio from {_TB.__file__}, not this worktree")
    import tt_bio.boltz2 as B2
    assert FLAG_HOIST not in os.environ, (
        f"{FLAG_HOIST} may not be pinned; the arm is set per fold")
    check_arms(list(plan) + args.timing_arms.split(","), T)
    SHIPPED = bool(getattr(T, ATTR_HOIST))

    AB.SAMPLING_STEPS, AB.RECYCLING_STEPS = args.steps, args.recycles
    dev = get_device()
    g = dev.compute_with_storage_grid_size()
    out = {"doc": __doc__, "env": {
        "host": socket.gethostname(), "grid": [g.x, g.y], "arch": str(dev.arch()),
        "torch": torch.__version__,
        "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
        "tt_bio_file": _TB.__file__,
        "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "loadavg": os.getloadavg(),
        "protocol": {"recycling_steps": args.recycles, "sampling_steps": args.steps,
                     "plan": args.plan, "diffusion_trace": False},
    }, "runs": []}
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def dump():
        args.out.write_text(json.dumps(out, indent=1))

    dump()

    work = Path(tempfile.mkdtemp(prefix="b2z2-hoist-"))
    struct_dir = work / "out"; struct_dir.mkdir(parents=True)
    msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
    for name in ("cdk2x2_512", "cdk2x2_298"):
        AB._seed_msa(AB.FIX / f"{name}.yaml", (AB.FIX / f"{name}.a3m").read_text(), msa_dir)

    cfg = AB.build_cfg(msa_dir, struct_dir)
    _ensure_local_artifacts(cfg)
    t0 = time.perf_counter()
    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("b2z2-hoist", cfg)
    out["model_load_s"] = round(time.perf_counter() - t0, 3)
    dump()

    # Count the hoisted stacks a fold takes, at the site that decides to take one. A digest that
    # does not move is then a lever that fired and did nothing, not a lever that never fired.
    fired = {"h": 0}
    _dt_call = T.DiffusionTransformer.__call__

    def _counted(self, *a, **kw):
        if getattr(T, ATTR_HOIST) and not self.atom_level:
            fired["h"] += 1
        return _dt_call(self, *a, **kw)

    T.DiffusionTransformer.__call__ = _counted

    def fold(arm, seed, target, keep):
        ho = SHIPPED if ARMS[arm] is None else ARMS[arm]
        setattr(T, ATTR_HOIST, ho)
        fired["h"] = 0
        cfg["seed"] = seed
        try:
            state.model.structure_module.score_model.reset_static_cache()
        except Exception:
            pass
        for p in struct_dir.glob("*"):
            p.unlink() if p.is_file() else shutil.rmtree(p)
        ttnn.synchronize_device(dev)
        # Load at both ends of the fold, not once at the start. qb2 is shared and a co-tenant
        # that starts mid-fold is exactly the case a single reading misses -- the same blind spot
        # benchlock's acquire-time check has. A bracket is only usable if BOTH ends were quiet.
        load0 = os.getloadavg()[0]
        t = time.perf_counter()
        metrics, _b, _f = state.predict_one(target, cfg)
        ttnn.synchronize_device(dev)
        wall = time.perf_counter() - t
        load1 = os.getloadavg()[0]
        cifs = sorted(struct_dir.glob("*.cif"))
        assert cifs, "no CIF written"
        keep.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cifs[0], keep / cifs[0].name)
        body = (keep / cifs[0].name).read_bytes()
        return {"arm": arm, "seed": seed, "target": target.stem, "fold_s": round(wall, 3),
                "load0": round(load0, 2), "load1": round(load1, 2),
                "sha256": hashlib.sha256(body).hexdigest()[:16],
                "cond_hoist": bool(getattr(T, ATTR_HOIST)), "hoist_norms": fired["h"],
                "plddt": round(float(metrics.get("plddt", metrics.get("confidence_score", 0))), 6)}

    if args.timing_reps:
        return timing(args, out, dump, fold, T, ttnn, dev)

    first = next(iter(plan))
    for size in sizes:
        target = AB.FIX / f"cdk2x2_{size}.yaml"
        order = []
        for seed in sorted({s for v in plan.values() for s in v}):
            order += [(arm, seed) for arm in plan if seed in plan[arm]]
        order.append((first, plan[first][0]))             # the A/A repeat, last
        seen = set()
        for arm, seed in order:
            tag = f"{arm}-s{seed}" + ("_r1" if (arm, seed) in seen else "")
            seen.add((arm, seed))
            r = fold(arm, seed, target, args.cifdir / f"{size}_{tag}")
            r["tag"] = tag
            out["runs"].append(r)
            print(f"  {size} {tag:12s} {r['fold_s']:7.3f}s sha={r['sha256']} "
                  f"plddt={r['plddt']}", flush=True)
            dump()

    aa = [r for r in out["runs"] if r["tag"].startswith(f"{first}-s{plan[first][0]}")]
    out["aa_floor_identical"] = {
        size: len({r["sha256"] for r in aa if r["target"].endswith(size)}) == 1 for size in sizes}
    dump()
    print(json.dumps(out["aa_floor_identical"], indent=1))
    return 0


def timing(args, out, dump, fold, T, ttnn, dev):
    """The fold A/B, on the same protocol the rest of the campaign times on.

    One process, one device open, arms interleaved inside every rep so a drift across the run
    cannot land on one arm. `base` runs at two positions in each rep, so the A/A floor is a
    measurement of this run rather than a number quoted from another one, and the cold fold is
    discarded. The DiffusionTransformer wall is recorded next to the fold wall: a null on the fold with
    a moved block is a lever that fired and was swallowed, which is a different finding from a
    lever that never fired at all. The wall is on the token-level `DiffusionTransformer`, which is
    where every op this lever deletes or regroups lives.
    """
    wall = {"n": 0, "s": 0.0, "atom_n": 0}
    block = getattr(T, "DiffusionTransformer", None) if args.block_timing else None
    if block is not None:
        inner = block.__call__

        def timed(self, *a, **kw):
            # Only the TOKEN-level stack carries the lever: `not self.atom_level` is the guard the
            # hoist itself reads. The atom-level stacks are counted, not walled, because syncing
            # around work the lever cannot touch would only add a constant to both arms.
            if self.atom_level:
                wall["atom_n"] += 1
                return inner(self, *a, **kw)
            ttnn.synchronize_device(dev)
            t = time.perf_counter()
            r = inner(self, *a, **kw)
            ttnn.synchronize_device(dev)
            wall["n"] += 1
            wall["s"] += time.perf_counter() - t
            return r

        block.__call__ = timed

    size = args.sizes.split(",")[0]
    target = AB.FIX / f"cdk2x2_{size}.yaml"
    arms = args.timing_arms.split(",")
    out["timing"] = {"arms": arms, "reps": args.timing_reps, "seed": args.timing_seed,
                     "block_timed": layer is not None}
    cif = Path(tempfile.mkdtemp(prefix="b2z2-hoist-timing-"))

    for rep in range(-1, args.timing_reps):                # rep -1 is the cold fold, discarded
        for pos, arm in enumerate(arms if rep >= 0 else arms[:1]):
            wall["n"] = 0
            wall["s"] = 0.0
            wall["atom_n"] = 0
            r = fold(arm, args.timing_seed, target, cif / f"{rep}_{pos}_{arm}")
            r.update(rep=rep, pos=pos, cold=rep < 0,
                     block_s=round(wall["s"], 4), block_n=wall["n"],
                     atom_block_n=wall["atom_n"])
            out["runs"].append(r)
            print(f"  rep{rep:<2d} {arm:7s} pos{pos} fold {r['fold_s']:7.3f}s  "
                  f"load {r['load0']:5.2f}->{r['load1']:5.2f} sha={r['sha256']}", flush=True)
            dump()

    warm = [r for r in out["runs"] if not r["cold"]]
    by = {a: [r["fold_s"] for r in warm if r["arm"] == a] for a in set(arms)}
    blk = {a: [r["block_s"] for r in warm if r["arm"] == a] for a in set(arms)}
    base = st.median(by["base"])
    summary = {}
    for arm in sorted(by):
        v = by[arm]
        row = {"n": len(v), "median_fold_s": round(st.median(v), 3),
               "fold_ratio_vs_base": round(base / st.median(v), 5),
               "spread_pct": round(100 * (max(v) - min(v)) / st.median(v), 2)}
        # any(), not truth: with --block-timing off every block_s is 0.0, and a non-empty list
        # of zeros is truthy, so the ratio below would divide 0.0 by 0.0 and lose the summary
        # after every fold had already been paid for.
        if any(blk[arm]) and any(blk["base"]):
            row["median_block_s"] = round(st.median(blk[arm]), 4)
            row["block_ratio_vs_base"] = round(st.median(blk["base"]) / st.median(blk[arm]), 5)
            # The syncs the wall inserts add the same constant to both arms, so they attenuate the
            # RATIO but leave the DIFFERENCE alone. The difference is what was pre-registered.
            row["block_delta_s_vs_base"] = round(st.median(blk["base"]) - st.median(blk[arm]), 4)
            row["block_spread_pct"] = round(
                100 * (max(blk[arm]) - min(blk[arm])) / st.median(blk[arm]), 2)
        summary[arm] = row
        print(f"{arm:8s} n={row['n']} fold {row['median_fold_s']:.3f}s "
              f"{row['fold_ratio_vs_base']:.5f}x  block {row.get('median_block_s')}s "
              f"{row.get('block_ratio_vs_base')}x  spread {row['spread_pct']:.2f}%", flush=True)

    p0 = [r["fold_s"] for r in warm if r["arm"] == "base" and r["pos"] == 0]
    pn = [r["fold_s"] for r in warm if r["arm"] == "base" and r["pos"] != 0]
    if p0 and pn:
        summary["aa_floor"] = round(st.median(p0) / st.median(pn), 5)
        print(f"A/A floor (base pos0 / base pos-last): {summary['aa_floor']:.5f}x", flush=True)
    b0 = [r["block_s"] for r in warm if r["arm"] == "base" and r["pos"] == 0]
    bn = [r["block_s"] for r in warm if r["arm"] == "base" and r["pos"] != 0]
    if any(b0) and any(bn):
        # The A/A the verdict is read against: same configuration, two positions, SAME arm list and
        # SAME session as the lever. A block ratio inside this floor is not a result.
        summary["aa_floor_block"] = round(st.median(b0) / st.median(bn), 5)
        summary["aa_floor_block_delta_s"] = round(st.median(b0) - st.median(bn), 4)
        print(f"A/A floor BLOCK (base pos0 / base pos-last): "
              f"{summary['aa_floor_block']:.5f}x  "
              f"delta {summary['aa_floor_block_delta_s']:+.4f}s", flush=True)
    summary["block_witness"] = {
        "block_n": sorted({r["block_n"] for r in warm}),
        "atom_block_n": sorted({r.get("atom_block_n", 0) for r in warm}),
        "hoist_norms_by_arm": {a: sorted({r["hoist_norms"] for r in warm if r["arm"] == a})
                               for a in sorted(by)},
    }
    print(json.dumps(summary["block_witness"]), flush=True)
    summary["identical_across_arms"] = len({r["sha256"] for r in warm}) == 1
    out["timing"]["summary"] = summary
    dump()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
