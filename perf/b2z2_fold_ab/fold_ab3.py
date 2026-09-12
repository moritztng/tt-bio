#!/usr/bin/env python3
"""Paired interleaved fold A/B for the three read-deleting trunk levers, plus their A/A floor.

`b2z2-trunk-byte-round2` measured 1.05106x on ONE PairformerLayer and located 0.497 s/fold. This
runs the fold. Arms, all in one process so they share a program cache and a card state:

    base    TT_BIO_TRIMUL_FUSED_GOUT / TRIATT_FUSED_QKVG / TRIATT_FUSED_QKVGB all off
    all3    all three on
    qkvg    only the qkv+gate fusion -- a 1.01459x block lever against all3's 1.05106x, so the
            two points test whether fold gain is proportional to block gain over a 3.5x range

`qkvgb` cannot be an arm on its own: `qkvgb_heads` requires `_QKVG_ENABLED`, because the bias tile
rides the qkv+gate pass rather than replacing it.

The arms alternate base/all3/qkvg every rep, so a drift in the card or the host cannot land on one
arm. `os.getloadavg()` is read INSIDE every rep -- whglx runs several rows at once and a one-shot
check before the loop is blind to a neighbour that starts mid-run. One warmup fold per arm is
discarded so no rep is a program-cache miss.

The A/A floor is the 95 % band of the SAME statistic the ratio is quoted with (median of reps),
resampled from the base arm's own reps. A per-position floor beside a median-of-reps ratio
understates it ~2.5x.

Parity is the fold's CIF, not a block output: `tt_bio.reference` zero-initialises 23 of a
PairformerLayer's weights including both trimuls' `p_out`, so a block comparison against it passes
for an arm that computes nothing. `--negctl` runs one extra all3 fold with the fused bias output
scaled by 1+2**-10 and asserts the sha CHANGES, which is what makes the equal shas evidence.

  fold_ab3.py --out o.json --cifdir cifs --reps 7 [--negctl] [--size 512]
"""
import argparse
import hashlib
import json
import os
import random
import shutil
import socket
import statistics
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "perf" / "b2x-flag-levers"))

import ab_flag_levers as AB  # noqa: E402  -- fixtures, cfg and MSA seeding, unmodified

ARMS = {"base": (), "all3": ("gout", "qkvg", "qkvgb"), "qkvg": ("qkvg",)}


def aa_band(vals, iters=20000, seed=0):
    """95 % band of the median-ratio estimator under random splits of one arm's own reps.

    Same statistic as the reported ratio, so the floor is the floor of what is claimed.
    """
    rng = random.Random(seed)
    n = len(vals)
    if n < 4:
        return None
    half = n // 2
    r = []
    for _ in range(iters):
        v = vals[:]
        rng.shuffle(v)
        a, b = statistics.median(v[:half]), statistics.median(v[half:half * 2])
        r.append(max(a / b, b / a))
    r.sort()
    return {"p50": r[len(r) // 2], "p95": r[int(0.95 * len(r))], "n": n}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cifdir", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--size", default="512")
    ap.add_argument("--steps", type=int, default=AB.SAMPLING_STEPS)
    ap.add_argument("--recycles", type=int, default=AB.RECYCLING_STEPS)
    ap.add_argument("--negctl", action="store_true")
    args = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    from tt_bio.tenstorrent import get_device
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts
    from tt_bio import esmfold2 as _E
    from tt_bio import tenstorrent as tt
    from tt_bio import triatt_qkv as K
    from tt_bio import reblock_permute as RP
    _E.set_progress(lambda *a, **k: None)
    import tt_bio as _TB
    assert Path(_TB.__file__).resolve().is_relative_to(REPO), (
        f"imported tt_bio from {_TB.__file__}, not this worktree")
    assert not (set(os.environ) & {"TT_BIO_TRIMUL_FUSED_GOUT", "TT_BIO_TRIATT_FUSED_QKVG",
                                  "TT_BIO_TRIATT_FUSED_QKVGB"}), \
        "no lever may be pinned in the environment; the arms are set in-process"
    assert os.environ.get("TT_METAL_DEVICE_PROFILER") is None, \
        "profiler env present -- a watcher-armed run cannot produce a ratio"

    SET = {"gout": tt.set_trimul_fused_gout,
           "qkvg": lambda on: setattr(K, "_QKVG_ENABLED", bool(on)),
           "qkvgb": lambda on: setattr(K, "_QKVGB_ENABLED", bool(on))}

    def census():
        return {"trimul_gout": list(tt.TRIMUL_GOUT_STATS), "qkvg": list(K.QKVG_STATS),
                "qkvgb": list(K.QKVGB_STATS), "qkv_heads": list(K.STATS),
                "tail": list(K.TAIL_STATS), "trimul_tail_f1": list(tt._trimul_tail.STATS),
                "reblock_gated": int(RP.STATS_GATED[0])}

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
        "trace_region": os.environ.get("TT_BIO_TRACE_REGION_SIZE"),
        "protocol": {"recycling_steps": args.recycles, "sampling_steps": args.steps,
                     "seed": AB.SEED, "size": args.size},
    }, "arms": list(ARMS), "reps": args.reps, "runs": []}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    dump = lambda: args.out.write_text(json.dumps(out, indent=1))
    dump()

    work = Path(tempfile.mkdtemp(prefix="b2z2-foldab-"))
    struct_dir = work / "out"; struct_dir.mkdir(parents=True)
    msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
    name = f"cdk2x2_{args.size}"
    AB._seed_msa(AB.FIX / f"{name}.yaml", (AB.FIX / f"{name}.a3m").read_text(), msa_dir)
    cfg = AB.build_cfg(msa_dir, struct_dir)
    _ensure_local_artifacts(cfg)
    t0 = time.perf_counter()
    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("b2z2-fold-ab", cfg)
    out["model_load_s"] = round(time.perf_counter() - t0, 3)
    dump()
    target = AB.FIX / f"{name}.yaml"

    def fold(arm, keep, tag=""):
        for lv, fn in SET.items():
            fn(lv in ARMS[arm])
        try:
            state.model.structure_module.score_model.reset_static_cache()
        except Exception:
            pass
        for p in struct_dir.glob("*"):
            p.unlink() if p.is_file() else shutil.rmtree(p)
        c0 = census()
        load = round(os.getloadavg()[0], 2)
        ttnn.synchronize_device(dev)
        t = time.perf_counter()
        metrics, _b, _f = state.predict_one(target, cfg)
        ttnn.synchronize_device(dev)
        wall = time.perf_counter() - t
        c1 = census()
        cifs = sorted(struct_dir.glob("*.cif"))
        assert cifs, "no CIF written"
        keep.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cifs[0], keep / cifs[0].name)
        body = (keep / cifs[0].name).read_bytes()
        return {"arm": arm + tag, "fold_s": round(wall, 4), "loadavg": load,
                "sha256": hashlib.sha256(body).hexdigest()[:16],
                "plddt": float(metrics.get("plddt", metrics.get("confidence_score", 0))),
                "census_delta": {k: [c1[k][i] - c0[k][i] for i in range(len(c1[k]))]
                                 if isinstance(c1[k], list) else c1[k] - c0[k] for k in c1}}

    for arm in ARMS:                               # compile every arm before anything is timed
        r = fold(arm, args.cifdir / f"warm_{arm}"); r["warmup"] = True
        out["runs"].append(r)
        print(f"  warm {arm:5s} {r['fold_s']:8.3f}s sha={r['sha256']} "
              f"census={r['census_delta']}", flush=True)
        dump()

    for i in range(args.reps):
        for arm in ARMS:
            r = fold(arm, args.cifdir / f"{arm}_{i}"); r["rep"] = i
            out["runs"].append(r)
            print(f"  rep{i} {arm:5s} {r['fold_s']:8.3f}s sha={r['sha256']} "
                  f"plddt={r['plddt']:.6f} load={r['loadavg']}", flush=True)
            dump()

    if args.negctl:
        # The negative control the brief demands: make the FUSED path compute something else and
        # show the CIF sha moves. Without this, equal shas are not evidence.
        real = K.qkvgb_heads
        def broken(*a, **kw):
            r = real(*a, **kw)
            if r is None:
                return None
            qkv, gate, bias = r
            return qkv, gate, ttnn.multiply(bias, 1.0 + 2 ** -10)
        K.qkvgb_heads = broken
        try:
            r = fold("all3", args.cifdir / "negctl", tag="_negctl")
            r["negative_control"] = True
            out["runs"].append(r)
            print(f"  negctl     {r['fold_s']:8.3f}s sha={r['sha256']}", flush=True)
        finally:
            K.qkvgb_heads = real

    warm = [r for r in out["runs"] if not r.get("warmup") and not r.get("negative_control")]
    med, series = {}, {}
    for arm in ARMS:
        series[arm] = [r["fold_s"] for r in warm if r["arm"] == arm]
        med[arm] = statistics.median(series[arm])
    out["median_fold_s"] = med
    out["series_s"] = series
    out["ratio_vs_base"] = {a: med["base"] / med[a] for a in ARMS if a != "base"}
    out["seconds_saved"] = {a: med["base"] - med[a] for a in ARMS if a != "base"}
    out["aa_floor"] = aa_band(series["base"])
    gains = {a: med["base"] / med[a] - 1.0 for a in ARMS if a != "base"}
    out["gain_ratio_all3_over_qkvg"] = (gains["all3"] / gains["qkvg"]) if gains.get("qkvg") else None
    shas = {a: sorted({r["sha256"] for r in warm if r["arm"] == a}) for a in ARMS}
    out["sha_per_arm"] = shas
    out["bit_exact_across_arms"] = len({s for v in shas.values() for s in v}) == 1
    neg = [r for r in out["runs"] if r.get("negative_control")]
    if neg:
        out["negative_control_breaks"] = neg[0]["sha256"] not in shas["all3"]
        out["negative_control_sha"] = neg[0]["sha256"]
    cen = {a: sorted({json.dumps(r["census_delta"], sort_keys=True)
                      for r in warm if r["arm"] == a}) for a in ARMS}
    out["census_flat_per_arm"] = {a: len(v) == 1 for a, v in cen.items()}
    out["census_per_arm"] = {a: json.loads(v[0]) for a, v in cen.items() if v}
    out["loadavg_range"] = [min(r["loadavg"] for r in warm), max(r["loadavg"] for r in warm)]
    dump()
    print(json.dumps({k: out[k] for k in (
        "median_fold_s", "ratio_vs_base", "seconds_saved", "aa_floor",
        "gain_ratio_all3_over_qkvg", "bit_exact_across_arms", "negative_control_breaks",
        "census_flat_per_arm", "census_per_arm", "loadavg_range") if k in out}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
