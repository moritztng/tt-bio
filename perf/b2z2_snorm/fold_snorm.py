#!/usr/bin/env python3
"""The folds a per-pseudo-domain + CA-lDDT reading of BOLTZ2_ADALN_SHARED_SNORM needs.

Same protocol and the same scorer as `fold_cond.py` (`perf/b2z2_fusebias/score.py`, arithmetic
unmodified). One thing differs: this lever is a MODULE-LEVEL gate read once at import
(`tt_bio.tenstorrent._B2_ADALN_SHARED_SNORM`), not a per-call env lookup, so the arm is selected by
setting that attribute per fold and the env var is asserted absent -- a pinned value would be baked
into the module at import and would serve every arm.

This row exists to resolve a contradiction. `state/b2z2-step-layernorm-fusion.md` measured this
lever alone on WORMHOLE at 0.279-0.349 A per pseudo-domain; `state/b2z2-bh-union-step.md` measured
the same lever alone on BLACKHOLE at 0.70391 A (seed 0) and 0.25836 A (seed 1) -- two seeds, a 2.7x
spread, and only the larger one reached the merge ladder. Two seeds cannot resolve a statistic that
spreads that far, so this row runs four per arm, on the composed baseline main actually ships today:
TT_BIO_ATOM_SHIFT_GATHER and TT_BIO_DEVICE_CONDITIONING are both merged and default-on, and neither
was on main when either of those two measurements was taken.

The reading with teeth is CA-lDDT against the experimental structure 1HCL, not per-domain RMSD
against the seed floor: `b2z2-union-land` showed RMSD-against-the-seed-floor clears an arm that has
lost 0.07 CA-lDDT.

Arms, in ONE process so they share a device, a program cache and a model load:

    base      _B2_ADALN_SHARED_SNORM = False, one layer_norm per AdaLN
    snorm     _B2_ADALN_SHARED_SNORM = True, one layer_norm per DiffusionTransformer stack
    default   the checkout untouched. Its digest must equal the arm the default is supposed to
              select, or the default is not what this run measured.

`shared_norms` counts the shared norms a fold actually took, so an arm whose lever never fired is
distinguishable from an arm whose lever fired and was swallowed.

    fold_snorm.py --out <json> --cifdir <dir> [--plan base:0,1,2,3;snorm:0,1,2,3]
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

FLAG = "BOLTZ2_ADALN_SHARED_SNORM"
ATTR = "_B2_ADALN_SHARED_SNORM"
FLAG_HOIST = "TT_BIO_DIT_COND_HOIST"
ATTR_HOIST = "_B2_DIT_COND_HOIST"

# arm -> (shared-snorm, cond-hoist), or None for "touch nothing". `default` tests the SHIPPED
# DEFAULT rather than a value this driver sets: flipping a default and then measuring an arm that
# overrides it proves nothing about the default.
#
# `hoist` is main TT_BIO_DIT_COND_HOIST, added here 2026-09-14 because main landed it while this
# row was measuring and it deletes the SAME 48 layer_norms per step that `snorm` does, plus 144
# projections. The two are never on together: a tree carrying both would be measuring one lever
# through the other.
ARMS = {"base": (False, False), "snorm": (True, False), "hoist": (False, True), "default": None}


def check_arms(arms, T):
    """Refuse to measure a lever this checkout does not have.

    Setting an attribute nothing reads succeeds silently, so both arms would run the same code and
    score as a null. Check the reader exists before folding anything.
    """
    for arm in arms:
        assert arm in ARMS, f"unknown arm {arm}"
    assert hasattr(T, ATTR), (
        f"this checkout has no tt_bio.tenstorrent.{ATTR}; both arms would be identical")
    assert hasattr(T, ATTR_HOIST), (
        f"this checkout has no tt_bio.tenstorrent.{ATTR_HOIST}; the hoist arm would be a null")
    assert hasattr(T, "DiffusionTransformer"), "no DiffusionTransformer to gate"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cifdir", type=Path, required=True)
    ap.add_argument("--plan", default="base:0,1,2,3;snorm:0,1,2,3")
    ap.add_argument("--sizes", default="512")
    ap.add_argument("--steps", type=int, default=AB.SAMPLING_STEPS)
    ap.add_argument("--recycles", type=int, default=AB.RECYCLING_STEPS)
    ap.add_argument("--timing-reps", type=int, default=0,
                    help="run the TIMING protocol instead of the seed plan: this many reps of "
                         "--timing-arms at one seed, in one process")
    ap.add_argument("--block-timing", action="store_true",
                    help="wall each PairformerLayer call. Inserts two syncs per block, so the "
                         "fold times it produces are NOT publishable as absolute numbers")
    ap.add_argument("--timing-arms", default="base,snorm,base",
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
    assert FLAG not in os.environ, f"{FLAG} may not be pinned; the arm is set per fold"
    assert FLAG_HOIST not in os.environ, f"{FLAG_HOIST} may not be pinned; set per fold"
    check_arms(list(plan) + args.timing_arms.split(","), T)
    SHIPPED = (bool(getattr(T, ATTR)), bool(getattr(T, ATTR_HOIST)))

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

    work = Path(tempfile.mkdtemp(prefix="b2z2-snorm-"))
    struct_dir = work / "out"; struct_dir.mkdir(parents=True)
    msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
    for name in ("cdk2x2_512", "cdk2x2_298"):
        AB._seed_msa(AB.FIX / f"{name}.yaml", (AB.FIX / f"{name}.a3m").read_text(), msa_dir)

    cfg = AB.build_cfg(msa_dir, struct_dir)
    _ensure_local_artifacts(cfg)
    t0 = time.perf_counter()
    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("b2z2-snorm", cfg)
    out["model_load_s"] = round(time.perf_counter() - t0, 3)
    dump()

    # Count the shared norms a fold takes, at the site that decides to take one. A digest that
    # does not move is then a lever that fired and did nothing, not a lever that never fired.
    fired = {"n": 0, "h": 0}
    _dt_call = T.DiffusionTransformer.__call__

    def _counted(self, *a, **kw):
        if getattr(T, ATTR_HOIST) and not self.atom_level:
            fired["h"] += 1
        elif getattr(T, ATTR) and self.shared_s_norm:
            fired["n"] += 1
        return _dt_call(self, *a, **kw)

    T.DiffusionTransformer.__call__ = _counted

    def fold(arm, seed, target, keep):
        sn, ho = SHIPPED if ARMS[arm] is None else ARMS[arm]
        setattr(T, ATTR, sn)
        setattr(T, ATTR_HOIST, ho)
        fired["n"] = fired["h"] = 0
        cfg["seed"] = seed
        try:
            state.model.structure_module.score_model.reset_static_cache()
        except Exception:
            pass
        for p in struct_dir.glob("*"):
            p.unlink() if p.is_file() else shutil.rmtree(p)
        ttnn.synchronize_device(dev)
        t = time.perf_counter()
        metrics, _b, _f = state.predict_one(target, cfg)
        ttnn.synchronize_device(dev)
        wall = time.perf_counter() - t
        cifs = sorted(struct_dir.glob("*.cif"))
        assert cifs, "no CIF written"
        keep.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cifs[0], keep / cifs[0].name)
        body = (keep / cifs[0].name).read_bytes()
        return {"arm": arm, "seed": seed, "target": target.stem, "fold_s": round(wall, 3),
                "sha256": hashlib.sha256(body).hexdigest()[:16],
                "shared_snorm": bool(getattr(T, ATTR)), "shared_norms": fired["n"],
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
    discarded. The PairformerLayer wall is recorded next to the fold wall: a null on the fold with
    a moved block is a lever that fired and was swallowed, which is a different finding from a
    lever that never fired at all.
    """
    wall = {"n": 0, "s": 0.0}
    layer = getattr(T, "PairformerLayer", None) if args.block_timing else None
    if layer is not None:
        inner = layer.__call__

        def timed(self, *a, **kw):
            ttnn.synchronize_device(dev)
            t = time.perf_counter()
            r = inner(self, *a, **kw)
            ttnn.synchronize_device(dev)
            wall["n"] += 1
            wall["s"] += time.perf_counter() - t
            return r

        layer.__call__ = timed

    size = args.sizes.split(",")[0]
    target = AB.FIX / f"cdk2x2_{size}.yaml"
    arms = args.timing_arms.split(",")
    out["timing"] = {"arms": arms, "reps": args.timing_reps, "seed": args.timing_seed,
                     "block_timed": layer is not None}
    cif = Path(tempfile.mkdtemp(prefix="b2z2-snorm-timing-"))

    for rep in range(-1, args.timing_reps):                # rep -1 is the cold fold, discarded
        for pos, arm in enumerate(arms if rep >= 0 else arms[:1]):
            wall["n"] = 0
            wall["s"] = 0.0
            r = fold(arm, args.timing_seed, target, cif / f"{rep}_{pos}_{arm}")
            r.update(rep=rep, pos=pos, cold=rep < 0,
                     block_s=round(wall["s"], 4), block_n=wall["n"])
            out["runs"].append(r)
            print(f"  rep{rep:<2d} {arm:7s} pos{pos} fold {r['fold_s']:7.3f}s  "
                  f"block {r['block_s']:7.3f}s/{r['block_n']:<4d} sha={r['sha256']}", flush=True)
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
            row["median_block_s"] = round(st.median(blk[arm]), 3)
            row["block_ratio_vs_base"] = round(st.median(blk["base"]) / st.median(blk[arm]), 5)
        summary[arm] = row
        print(f"{arm:8s} n={row['n']} fold {row['median_fold_s']:.3f}s "
              f"{row['fold_ratio_vs_base']:.5f}x  block {row.get('median_block_s')}s "
              f"{row.get('block_ratio_vs_base')}x  spread {row['spread_pct']:.2f}%", flush=True)

    p0 = [r["fold_s"] for r in warm if r["arm"] == "base" and r["pos"] == 0]
    pn = [r["fold_s"] for r in warm if r["arm"] == "base" and r["pos"] != 0]
    if p0 and pn:
        summary["aa_floor"] = round(st.median(p0) / st.median(pn), 5)
        print(f"A/A floor (base pos0 / base pos-last): {summary['aa_floor']:.5f}x", flush=True)
    summary["identical_across_arms"] = len({r["sha256"] for r in warm}) == 1
    out["timing"]["summary"] = summary
    dump()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
