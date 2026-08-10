#!/usr/bin/env python3
"""AbAg-XM deep-N analysis core (state doc abag-xm-deepn-saturation-fullpanel, PHASE 3).

Assembles per-(model, target, N) sample pools and computes the saturation curves:
oracle (max DockQ over the pool) and user (DockQ at the pool's argmax of the model's own
selector -- confidence_score, pLDDT for esmfold2) at DockQ thresholds 0.23/0.49/0.80.

Pools merge every chunk of a (target, rung): the deep-N user sees all N samples and picks by
confidence, so chunk-boundary ranks are re-derived over the pool, never taken per-chunk.

Data arms (this script: deepn + tier_a; overlays bolt on later):
  deepn   ~/abag_xm/deepn/<model>/<target>_n<N>[_c<j>]/  results.json all_runs + labels.json
  tier_a  ~/abag_xm/tier_a/<model_dir>/<prefix>_results_<target>/results.json + labels dir

Writes deepn/analysis_curves.json and prints the curve table. CPU-only.
"""
import argparse, json, os, sys
from pathlib import Path

BASE = Path.home() / "abag_xm" / "deepn"
TIER_A = Path.home() / "abag_xm" / "tier_a"
FRONTIER_A = Path.home() / "abag_xm" / "frontier" / "A"      # opendde, m=200, 11 targets
SATURATION = Path.home() / "abag_xm" / "saturation"          # 3 models, N=1000, 16 targets
PHASE0 = BASE / "phase0"                                     # galaxy p2 parquets (N=16)
GALAXY64 = BASE / "galaxy"                                   # harvested galaxy deep-N folds
CHUNK_SAMPLES = 64                                           # samples per _c<j> chunk dir
# PHASE 0 verdict: galaxy overlay is statistically consistent for boltz2 + opendde ONLY.
GALAXY_OK = {"boltz2": "_boltz2", "opendde-abag": ""}
# Galaxy N>=64 arm: models licensed to contribute curve points. boltz2/opendde licensed by
# PHASE 0; protenix-v2/esmfold2 join ONLY after their N=64 cross-hardware gate verdict.
# Galaxy-spine license per model (N=64 cross-hardware gate, phase0_n64_gate.py):
# boltz2/opendde PHASE-0-consistent; esmfold2 LICENSED at N=64 (2026-08-03); protenix-v2
# LICENSED 2026-08-04 by gate amendment: the pre-registered partition null is misspecified
# for the same-seed paired design (per-index cross-arch perturbations cancel under random
# partition but accumulate under the hardware split, inflating exceed_q95 at zero bias).
# Root-cause (scripts/abag_xm/xhw_same_seed_pairing.py): MSA byte-identical, same-seed
# noise streams identical (per-index DockQ spr 0.78, ptm 0.84, n=960), zero signed oracle
# bias (med -0.003, wilcoxon p=0.30); residual = chaotic amplification of WH/BH
# reduction-order numerics on borderline multi-basin targets, reproduced on-galaxy by the
# mps 1->5 control (~0.014 oracle wobble, same magnitude). Moritz 2026-08-04 policy: px
# runs on Galaxy.
GALAXY64_OK = {"boltz2", "opendde-abag", "esmfold2", "protenix-v2"}
# N=16 ARK restatement (opt-in, DEEPN_N16_ARK=1): the galaxy p2 N=16 structures re-labeled
# with the ARK-interface scorer, removing the global-vs-ARK flavor gap from the curve's
# first rung. boltz2: replaces the parquet rung (flavor gap decisive, -0.087). esmfold2:
# adds the rung (galaxy spine LICENSED at N=64). opendde's p2 structures were not retained
# (parquet rung stays, flavor-flagged; gap inconclusive). protenix-v2 joins 2026-08-04 with
# the galaxy-spine license (gate amendment above); its restated rung was validated by the
# flavor-clean recheck (n=152, med|d|=0.030, zero directional bias).
N16_ARK = BASE / "n16_ark"
N16_ARK_OK = {"boltz2", "esmfold2", "protenix-v2"}
# Per-model target exclusions for the restated N=16 rung. esmfold2 9loz/9w14: the
# p2-era galaxy pipeline mis-folded the complex on the IDENTICAL input sequence --
# all 16 samples ptm ~0.55-0.61 vs 0.92 for the current pipeline (tier_a), DockQ
# ~0.03 vs ~0.89. Scorer-independent (the model's own ptm condemns the fold); the
# other 161 esm targets match tier_a at median delta-ptm +0.003. A pipeline artifact
# is not model behavior, so the rung drops these two targets.
N16_ARK_EXCLUDE = {"esmfold2": {"9loz", "9w14"}}
# Galaxy-spine pipeline-artifact exclusions (scorer-independent; the model's own ptm
# condemns the fold -- the pass-75 exclusion rule). opendde 9sbb: galaxy N=64 basin
# ptm 0.67-0.70 vs tier_a 0.91, current-pipeline probe reproduces 0.914 -- the p2-era
# galaxy pipeline mis-folded the complex on the identical input. The seed-nested
# ladder makes the chunked rungs share the condemned fold setup (c0 IS the n64 seed
# block), so the exclusion spans all od galaxy rungs. A pipeline artifact is not
# model behavior. Full-panel scan (xhw_galaxy_tiera_scan.py, pass 75) confirmed the
# set: 9sbb is od's only galaxy-worse>0.2 outlier; bz 9v1h is tail-luck (ptm-clean).
GALAXY_EXCLUDE = {"opendde-abag": {"9sbb"}}
# The four large targets held out of the panel for device DRAM capacity (an engineering
# boundary, never a scoring or biology decision). Window p32 folds them at N=512 on the
# OOM-fixed engine, so they exist at 512 only, and on a different engine commit than the
# rest of the panel. Two reasons they stay out of the primary curve by default: including
# them would move the 512 rung's target set away from 256's, confounding the one question
# the campaign asks (does the oracle-vs-delivered gap keep widening from 256 to 512), and
# it would mix two engine commits inside a single rung mean. DEEPN_P32_EXT=1 folds them in
# for the separately-reported large-target extension cohort.
#
# Model-scoped, and it must stay that way: boltz2 folded all four targets through p27/p28,
# so they are already in its published 64/128/256 rungs and excluding them there would move
# a frozen number (analysis_curves.pre-n512.json). Only the cells p32 creates are new --
# opendde-abag never folded any of the four on Wormhole, px/esm only lacked 9j4c.
P32_EXTENSION = {"opendde-abag": {"9i3p", "9j4c", "9ivj", "9q7y"},
                 "protenix-v2": {"9j4c"},
                 "esmfold2": {"9j4c"}}
P32_EXT_ON = os.environ.get("DEEPN_P32_EXT") == "1"
GALAXY_NOTE = "galaxy N=16 uses global_dockq (mean over native interfaces), not the " \
              "ARK-interface DockQ of the qb1 arms; PHASE 0 measured the flavors " \
              "statistically equivalent for these two models."
THR = (0.23, 0.49, 0.80)
THR_KEY = {t: str(t).replace(".", "") for t in THR}
MODELS = {"opendde-abag": ("opendde", "opendde_abag", "confidence_score"),
          "protenix-v2": ("protenix", "protenix_v2", "confidence_score"),
          "boltz2": ("boltz2", "boltz2", "confidence_score"),
          "esmfold2": ("esmfold2", "esmfold2", "plddt")}


def pool_fold(results_json: Path, labels_json: Path, sel_key: str):
    """One fold -> list of (selector, dockq) joined by rank. None on any gap."""
    try:
        runs = json.loads(results_json.read_text())[0].get("all_runs", [])
        labs = json.loads(labels_json.read_text()).get("samples", [])
    except Exception:
        return None
    conf = {}
    for r in runs:
        v = r.get(sel_key)
        if v is not None:
            conf[int(r["rank"])] = float(v)
    dockq = {}
    for s in labs:
        d = s.get("dockq")
        if isinstance(d, dict) and d.get("dockq") is not None:
            dockq[int(s["rank"])] = float(d["dockq"])
    ranks = sorted(set(conf) & set(dockq))
    if not ranks:
        return None
    return [(conf[r], dockq[r]) for r in ranks]


def deepn_pools(model: str):
    """(target, rung) -> pooled [(sel, dockq)] across chunks, plus wall_s."""
    prefix, _md, sel = MODELS[model]
    out = {}
    mdir = BASE / prefix
    if not mdir.is_dir():
        return out
    walls = {}
    pj = BASE / "progress.jsonl"
    if pj.exists():
        for line in pj.read_text().splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("model") == model and r.get("status") == "ok":
                k = (r["target"], r["rung"])
                walls[k] = walls.get(k, 0.0) + r["wall_s"]
    for out_dir in sorted(mdir.iterdir()):
        if not out_dir.is_dir():
            continue
        name = out_dir.name  # <target>_n<N>[_c<j>]
        try:
            t, rest = name.split("_n")
            rung = int(rest.split("_c")[0])
        except ValueError:
            continue
        pool = pool_fold(out_dir / f"{prefix}_results_{t}" / "results.json",
                         out_dir / "labels.json", sel)
        if pool is None:
            continue
        k = (t, rung)
        out.setdefault(k, []).extend(pool)
    for k in out:
        out[k] = {"pool": out[k], "wall_s": walls.get(k)}
    return out


def tiera_pools(model: str):
    prefix, md, sel = MODELS[model]
    out = {}
    lab_dir = TIER_A / "labels"
    for rj in sorted(TIER_A.glob(f"{md}/{prefix}_results_*/results.json")):
        t = rj.parent.name.split("_results_")[-1]
        pool = pool_fold(rj, lab_dir / f"{md}_{t}.json", sel)
        if pool:
            out[(t, 50)] = {"pool": pool, "wall_s": None}
    return out


def overlay_pools(model: str):
    """frontier A (N=200, opendde) + saturation-depth (N=1000, 3 models), chunks pooled.

    The two arms stay SEPARATE (they are different N points); within an arm, a chunked
    target's pieces pool together. N keys are the MEASURED pool sizes per arm."""
    prefix, _md, sel = MODELS[model]
    out = {}
    arms = []
    if model == "opendde-abag" and FRONTIER_A.is_dir():
        arms.append(FRONTIER_A)
    sat_dir = SATURATION / prefix
    if sat_dir.is_dir():
        arms.append(sat_dir)
    for root in arms:
        per_target = {}
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue
            t = d.name.split("_c")[0]
            pool = pool_fold(d / f"{prefix}_results_{t}" / "results.json",
                             d / "labels.json", sel)
            if pool is None:
                continue
            per_target.setdefault(t, []).extend(pool)
        for t, pool in per_target.items():
            out[(t, len(pool))] = {"pool": pool, "wall_s": None}
    return out


def galaxy_pools(model: str):
    """WH-Galaxy p2 N=16 overlay for the PHASE-0-consistent models (GT-overlap targets)."""
    if model not in GALAXY_OK or not (PHASE0 / f"abag_xm_scaling{GALAXY_OK[model]}_samples.parquet").exists():
        return {}
    import pandas as pd
    _prefix, _md, sel = MODELS[model]
    df = pd.read_parquet(PHASE0 / f"abag_xm_scaling{GALAXY_OK[model]}_samples.parquet")
    df = df[(df.status == "ok") & (df.n_samples == 16)]
    out = {}
    for t, g in df.groupby("target"):
        g = g.sort_values("rank")
        pool = []
        for _i, r in g.iterrows():
            c = r["plddt"] if sel == "plddt" else r["confidence_score"]
            d = r["global_dockq"]
            if c is not None and d is not None and d == d:
                pool.append((float(c), float(d)))
        if pool:
            out[(t, len(pool))] = {"pool": pool, "wall_s": None}
    return out


def _reuse_skip(model: str):
    """(target, rung, chunk) keys of NESTED linked chunks (reused_chunks.*.jsonl).

    Linked records carry their source fold's real seconds for provenance, but those
    seconds were paid at the source rung/window and are counted there via the source's
    own record (empirically: od 21du's 1214s sat at both rung 64 and rung 256 c0).
    Skipping them in the walls sum makes per-rung card_h the marginal ladder cost --
    the denominator of the pre-registered marginal-oracle-per-card-second metric.

    Only the NESTED half is free. The ladder is seed-nested, so at rung R the chunks
    covering the first R/2 samples ARE the previous rung's chunks and were billed
    there; chunks past R/2 are work that exists only because of R, wherever it was
    paid. Skipping a link purely because it is a link drops the latter: 68 records
    on this window are p28-window folds of 512-only chunks 4-7 that p31 hardlinked
    rather than re-folded, and billing them to nobody understates the 512 step."""
    skip = set()
    for f in sorted(GALAXY64.glob("reused_chunks.*.jsonl")):
        for line in f.read_text().splitlines():
            if not line.startswith("{"):
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("model") != model:
                continue
            c = r.get("chunk")
            # chunk j covers samples [64j, 64j+64); nested iff it sits below R/2.
            # An unchunked link is a whole-rung reuse, which is nested by definition.
            if c is None or c * CHUNK_SAMPLES < r["rung"] // 2:
                skip.add((r["target"], r["rung"], c))
    return skip


def galaxy64_pools(model: str):
    """Harvested WH-Galaxy deep-N folds (the PHASE 2 campaign spine, N>=64).

    Same pool shape as deepn_pools, rooted at BASE/galaxy/<prefix>/<target>_n<N>[_c<j>],
    walls from BASE/galaxy/fleet_results.jsonl. In assembly this arm is applied LAST with
    replacement semantics: where galaxy and qb1 both hold a (target, N) pool, the galaxy
    pool defines the curve point (qb1 deep-N is the pilot overlay; cross-hardware deltas
    are a separate gate analysis, not pooled into the curve)."""
    if model not in GALAXY64_OK:
        return {}
    prefix, _md, sel = MODELS[model]
    out = {}
    mdir_root = GALAXY64 / prefix
    if not mdir_root.is_dir():
        return out
    walls = {}
    fj = GALAXY64 / "fleet_results.jsonl"
    skip = _reuse_skip(model)
    if fj.exists():
        # One entry per chunk, not per record. The p31 window dispatched 106 chunks
        # to two chips at once (adjacent records, same target/chunk, different `umd`,
        # seconds within 1 s), and a bare sum bills the rung for both. The chunk dirs
        # all hold exactly 64 CIFs, so the pools are unaffected -- this is a cost
        # denominator artifact only, and it is confined to the 512 rung (64 and 256
        # have zero duplicate keys, so deduping cannot move a published number).
        per_chunk = {}
        for line in fj.read_text().splitlines():
            if not line.startswith("{"):
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("model") == model and r.get("rc") == 0 and r.get("cifs", 0) > 0:
                ck = (r["target"], r["rung"], r.get("chunk"))
                if ck in skip:
                    continue
                per_chunk[ck] = max(per_chunk.get(ck, 0.0), r["seconds"])
        for (t, rung, _c), s in per_chunk.items():
            k = (t, rung)
            walls[k] = walls.get(k, 0.0) + s
    meta = {}
    for out_dir in sorted(mdir_root.iterdir()):
        if not out_dir.is_dir():
            continue
        name = out_dir.name  # <target>_n<N>[_c<j>]
        try:
            t, rest = name.split("_n")
            rung = int(rest.split("_c")[0])
        except ValueError:
            continue
        if t in GALAXY_EXCLUDE.get(model, ()):
            continue
        if not P32_EXT_ON and t in P32_EXTENSION.get(model, ()):
            continue
        chunk = None
        if "_c" in rest:
            try:
                chunk = int(rest.split("_c")[1])
            except ValueError:
                chunk = None
        pool = pool_fold(out_dir / f"{prefix}_results_{t}" / "results.json",
                         out_dir / "labels.json", sel)
        if pool is None:
            continue
        k = (t, rung)
        out.setdefault(k, []).extend(pool)
        m = meta.setdefault(k, {"chunks": set(), "plain": 0})
        if chunk is None:
            m["plain"] += 1
        else:
            m["chunks"].add(chunk)
    # Rung-completeness gate: a chunked rung (N>=256 -> N/64 chunks) contributes only
    # when every chunk is present -- a 2-of-4 pool is a 128-sample oracle mislabeled
    # as N=256. Unchunked single-fold rungs (n64) are complete by construction.
    for k in list(out):
        m = meta[k]
        if m["plain"] and not m["chunks"]:
            continue
        if len(m["chunks"]) < max(1, k[1] // 64):
            del out[k]
    for k in out:
        out[k] = {"pool": out[k], "wall_s": walls.get(k)}
    return out


def n16ark_pools(model: str):
    """ARK-flavor re-label of the galaxy p2 N=16 structures (PHASE 3 restatement arm).

    Rooted at BASE/n16_ark/<prefix>/<target>_n16 (single-fold rung, complete by
    construction; labels land via the deepn labeler on that base). Opt-in via
    DEEPN_N16_ARK=1; applied LAST so a restated (target, 16) pool replaces the
    global_dockq parquet pool. Seconds were paid in the p2 window, so wall_s is
    None here (the p2 cost table carries them)."""
    if model not in N16_ARK_OK:
        return {}
    prefix, _md, sel = MODELS[model]
    out = {}
    mdir = N16_ARK / prefix
    if not mdir.is_dir():
        return out
    for out_dir in sorted(mdir.iterdir()):
        if not out_dir.is_dir():
            continue
        name = out_dir.name  # <target>_n16
        try:
            t, rung = name.split("_n")
            if int(rung) != 16:
                continue
        except ValueError:
            continue
        if t in N16_ARK_EXCLUDE.get(model, ()):
            continue
        pool = pool_fold(out_dir / f"{prefix}_results_{t}" / "results.json",
                         out_dir / "labels.json", sel)
        if pool is None:
            continue
        out[(t, 16)] = {"pool": pool, "wall_s": None}
    return out


def curve_points(pools):
    """Aggregate a {(target, N): {pool}} map into per-N curve points."""
    by_n = {}
    for (t, n), d in pools.items():
        pool = d["pool"]
        oracle = max(v for _c, v in pool)
        user = max(pool, key=lambda x: x[0])[1]
        by_n.setdefault(n, []).append({"target": t, "oracle": oracle, "user": user,
                                       "wall_s": d["wall_s"]})
    pts = {}
    for n, rows in sorted(by_n.items()):
        nt = len(rows)
        pts[n] = {"n_targets": nt,
                  "oracle_mean": sum(r["oracle"] for r in rows) / nt,
                  "user_mean": sum(r["user"] for r in rows) / nt,
                  **{f"oracle_ge_{THR_KEY[t]}": sum(1 for r in rows if r["oracle"] >= t) / nt
                     for t in THR},
                  **{f"user_ge_{THR_KEY[t]}": sum(1 for r in rows if r["user"] >= t) / nt
                     for t in THR},
                  "card_h": sum(r["wall_s"] for r in rows if r["wall_s"]) / 3600}
    return pts


def oracle_of(pool):
    return max(v for _c, v in pool)


def subsample_oracle_curve(pool, ms, rng, b=200):
    """E[max dockq over an m-subset] per m -- the within-fold i.i.d. saturation curve."""
    import numpy as np
    vals = np.array([v for _c, v in pool])
    out = {}
    for m in ms:
        if m > len(vals):
            continue
        out[m] = float(np.mean([vals[rng.choice(len(vals), m, replace=False)].max()
                                for _ in range(b)]))
    return out


def seed_noise_floor(pool, n, rng, b=200):
    """Median |oracle(A) - oracle(B)| over b disjoint n+n splits of the pool.

    The pre-registered stop-rule floor: the gain per doubling is compared against the
    oracle difference two same-size disjoint draws produce from seed noise alone.
    """
    import numpy as np
    if len(pool) < 2 * n:
        return None
    vals = np.array([v for _c, v in pool])
    ds = []
    for _ in range(b):
        i = rng.choice(len(vals), 2 * n, replace=False)
        ds.append(abs(vals[i[:n]].max() - vals[i[n:]].max()))
    return float(np.median(ds))


def paired_boot(ci_rows, b=20000, seed=20260802):
    """Paired bootstrap over targets: mean oracle/user per N with 95 pct CIs.

    ci_rows: {N: [(oracle, user), ...]} over a COMMON target set; one resample index
    vector serves every N (and every model caller shares the seed), keeping comparisons
    paired across rungs and models.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    out = {}
    for n, rows in sorted(ci_rows.items()):
        arr = np.array(rows)
        nt = len(arr)
        idx = rng.integers(0, nt, (b, nt))
        om = arr[idx, 0].mean(axis=1)
        um = arr[idx, 1].mean(axis=1)
        out[n] = {"oracle_mean": [float(np.quantile(om, q)) for q in (0.025, 0.5, 0.975)],
                  "user_mean": [float(np.quantile(um, q)) for q in (0.025, 0.5, 0.975)]}
    return out


def _ci_row(pool):
    """Per-target CI metrics for one pool: oracle, user, threshold indicators."""
    o = oracle_of(pool)
    u = max(pool, key=lambda x: x[0])[1]
    return (o, u) + tuple(1.0 if o >= t else 0.0 for t in THR)


def paired_gain_boot(rows_lo, rows_hi, b=20000, seed=20260802):
    """CI of the per-metric mean GAIN (hi rung minus lo rung) over a common target set.

    rows_lo/rows_hi: aligned _ci_row tuples per target. One resample index vector
    serves both rungs, so the gain is paired; every model caller shares the seed.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    lo = np.array(rows_lo, dtype=float)
    hi = np.array(rows_hi, dtype=float)
    nt = len(lo)
    idx = rng.integers(0, nt, (b, nt))
    keys = ("oracle", "user", "ge_0.23", "ge_0.49", "ge_0.80")
    out = {}
    for c, k in enumerate(keys):
        g = hi[idx, c].mean(axis=1) - lo[idx, c].mean(axis=1)
        out[k] = [float(np.quantile(g, q)) for q in (0.025, 0.5, 0.975)]
    return out


def deep_stats(pools, model):
    """Stop-rule + exhaustion inputs from each target's LARGEST pool."""
    import numpy as np
    rng = np.random.default_rng(20260802)
    # Campaign-spine cap: the exhaustion verdict is read at the campaign's top
    # POWERED rung (>= 20 targets, the within-fold common-set rule), not at a
    # sparse legacy tail. Without the cap, e.g. px's 3-target N=500 saturation-
    # depth remnant becomes "top rung", its targets get judged at 500, and the
    # within-fold curve mixes target sets past the campaign cap (pass-400 note).
    rung_nt = {}
    for _t, n in pools:
        rung_nt[n] = rung_nt.get(n, 0) + 1
    powered = [n for n, nt in rung_nt.items() if nt >= 20]
    cap = max(powered) if powered else None
    biggest = {}
    for (t, n), d in pools.items():
        if cap is not None and n > cap:
            continue
        if t not in biggest or n > biggest[t][0]:
            biggest[t] = (n, d["pool"])
    per_m, per_floor, solvable = {}, {}, {str(thr): 0 for thr in THR}
    for t, (n, pool) in sorted(biggest.items()):
        for m, v in subsample_oracle_curve(pool, (16, 32, 50, 64, 100, 128, 200, 256,
                                                  400, 512), rng).items():
            per_m.setdefault(m, []).append(v)
        for k in (16, 25, 32, 50, 64, 128, 256):
            f = seed_noise_floor(pool, k, rng)
            if f is not None:
                per_floor.setdefault(k, []).append(f)
        o = oracle_of(pool)
        for thr in THR:
            if o >= thr:
                solvable[str(thr)] += 1
    # Common-set within-fold curve: the per-m table above mixes target sets across
    # m (deep-m entries are overlay-heavy hard targets, so the raw curve can
    # invert). Pick the largest grid depth D with >= 20 targets at len(pool) >= D
    # and report the curve at all m <= D over that fixed set.
    MGRID = (16, 32, 50, 64, 100, 128, 200, 256, 400, 512)
    depth_ok = [m for m in MGRID
                if sum(1 for _n, p in biggest.values() if len(p) >= m) >= 20]
    common = None
    if depth_ok:
        D = max(depth_ok)
        cset = [biggest[t][1] for t in sorted(biggest) if len(biggest[t][1]) >= D]
        cc, cf = {}, {}
        for p in cset:
            for m, v in subsample_oracle_curve(p, [m for m in MGRID if m <= D],
                                             rng).items():
                cc.setdefault(m, []).append(v)
            for m in MGRID:
                if 2 * m > D:
                    break
                f = seed_noise_floor(p, m, rng)
                if f is not None:
                    cf.setdefault(m, []).append(f)
        common = {"depth": D, "n_targets": len(cset),
                  "curve": {m: float(np.mean(v)) for m, v in sorted(cc.items())},
                  "floor_med": {m: float(np.median(v)) for m, v in sorted(cf.items())}}
    top = cap if cap is not None else max(n for _t, n in pools)
    return {"top_rung": top, "n_targets": len(biggest),
            "within_fold_oracle_curve": {m: float(np.mean(v)) for m, v in sorted(per_m.items())},
            "within_fold_nt": {m: len(v) for m, v in sorted(per_m.items())},
            "within_fold_common": common,
            "seed_noise_floor_med": {k: float(np.median(v)) for k, v in sorted(per_floor.items())},
            "solvable_at_top": solvable}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, choices=sorted(MODELS))
    ap.add_argument("--deep", action="store_true",
                    help="add stop-rule floors, within-fold curves, paired-bootstrap CIs")
    ap.add_argument("--out", default=str(BASE / "analysis_curves.json"))
    a = ap.parse_args()
    models = [a.model] if a.model else sorted(MODELS)
    report = {}
    ark16 = []
    for model in models:
        pools = tiera_pools(model) | deepn_pools(model) | overlay_pools(model) \
            | galaxy_pools(model)
        pools.update(galaxy64_pools(model))  # campaign spine wins key collisions
        if os.environ.get("DEEPN_N16_ARK") == "1":
            ark = n16ark_pools(model)
            if any(n == 16 for _t, n in ark):
                ark16.append(model)
            pools.update(ark)  # ARK restatement wins the N=16 rung
        pts = curve_points(pools)
        report[model] = pts
        print(f"\n=== {model} ===")
        print(f"{'N':>5} {'nt':>4} {'oracle':>7} {'user':>7} "
              + " ".join(f"{'o>='+str(t):>8} {'u>='+str(t):>8}" for t in THR) + f" {'card-h':>8}")
        for n, p in pts.items():
            row = " ".join(f"{p['oracle_ge_' + THR_KEY[t]]:>8.3f} "
                           f"{p['user_ge_' + THR_KEY[t]]:>8.3f}" for t in THR)
            print(f"{n:>5} {p['n_targets']:>4} {p['oracle_mean']:>7.4f} {p['user_mean']:>7.4f} "
                  + row + f" {p['card_h']:>8.1f}")
        if a.deep:
            ns = sorted(pts)
            common = sorted({t for t, _n in pools if all((t, n) in pools for n in ns)})
            if len(ns) >= 2 and common:
                ci = paired_boot({n: [(max(v for _c, v in pools[(t, n)]["pool"]),
                                       max(pools[(t, n)]["pool"], key=lambda x: x[0])[1])
                                      for t in common] for n in ns})
                report[model + "__paired_ci"] = {"common_targets": len(common), "ci": ci}
                print(f"  paired CI over {len(common)} common targets:")
                for n in ns:
                    o, u = ci[n]["oracle_mean"], ci[n]["user_mean"]
                    print(f"    N={n:<5} oracle {o[1]:.4f} [{o[0]:.4f},{o[2]:.4f}] "
                          f"user {u[1]:.4f} [{u[0]:.4f},{u[2]:.4f}]")
            # Pairwise adjacent-rung gain CIs -- the stop-rule comparator. The
            # all-rungs intersection above degenerates to zero once sparse overlay
            # rungs (200/500/1000) join the curve, so the knee test reads these
            # per-pair gains (normalize by `doublings` for gain-per-doubling).
            # Adjacency runs over the POWERED spine (>= POWER_MIN targets): sparse
            # legacy arms (od 200/500/1000, bz 1000, px 500) must not interpose
            # between campaign rungs -- else e.g. od's stop-rule pair 64->256
            # (and later 256->512) is never emitted and the knee test silently
            # reads tiny-n legacy noise instead. Sparse rungs stay in the ladder
            # table above; they just don't break the chain.
            import math
            POWER_MIN = 50
            ns_powered = [n for n in ns if pts[n]["n_targets"] >= POWER_MIN]
            gains = {}
            for lo, hi in zip(ns_powered, ns_powered[1:]):
                both = sorted({t for t, _n in pools
                               if (t, lo) in pools and (t, hi) in pools})
                if not both:
                    continue
                g = paired_gain_boot([_ci_row(pools[(t, lo)]["pool"]) for t in both],
                                     [_ci_row(pools[(t, hi)]["pool"]) for t in both])
                gains[f"{lo}->{hi}"] = {"common_targets": len(both),
                                        "doublings": round(math.log2(hi / lo), 4),
                                        # below ~8 targets a bootstrap CI is a
                                        # coarse discrete point mass -- never let
                                        # it clear the stop rule
                                        "degenerate": len(both) < 8,
                                        "gain_ci": g}
                # Pre-registered marginal-oracle-per-1000-card-seconds: gain CI
                # midpoint over what the step to `hi` actually cost (hours -> 1000 s
                # = x3.6). Skipped when the pair has no measured cost basis
                # (ARK/tier_a rungs carry wall_s=None -> card_h 0.0).
                #
                # The denominator is rung hi's own cost alone, NOT hi minus lo.
                # _reuse_skip drops the NESTED linked chunks from the walls sum, so
                # rung hi's wall_s IS the marginal cost of arriving at that rung: at
                # 512 it counts chunks 4-7 only, because 0-3 are hardlinks of the 256
                # rung. Subtracting lo would then bill the step for samples it never
                # re-folded. Chunks 4-7 count whether they were folded in this window
                # or hardlinked from an earlier one that already folded them at 512. It is also right for a non-nested pair (50->64 re-folds
                # the whole rung, so the whole rung is the cost).
                #
                # This was masked until now: every pair that ever emitted the metric
                # had card_h[lo] == 0.0 exactly, so the delta happened to equal
                # card_h[hi]. 256->512 is the first pair with real wall data on both
                # sides, where the delta goes small or negative and the campaign's
                # cost headline would be inflated or silently dropped. It is also the
                # first pair whose two rungs sit on different panels, which is what
                # exposed the units defect fixed below.
                #
                # The denominator is PER-TARGET, restricted to `both`. It used to be
                # pts[hi]["card_h"], a SUM over rung hi's whole panel, divided into a
                # gain that is a MEAN over `both`. Mixing an extensive numerator basis
                # with an intensive one makes the metric scale as 1/n_targets[hi], so
                # two pairs on different panels are not comparable: on the partial
                # opendde run it read 1e-05 at 128->256 (269.7 card-h / 153 targets)
                # against 3e-05 at 256->512 (130.6 / 58), i.e. the second doubling
                # looked 3x cheaper per unit gain. Per target it is 1.72e-03 against
                # 1.55e-03: the 512 step is about 10 pct DEARER, the opposite call.
                # The panels can never be made to match (the 512 panel tops out near
                # 137 against rung 256's 153), so per-target is the only form that is
                # ever quotable. Sum-over-targets gain / sum-over-targets cost gives
                # the same number, which is the check that the units now close.
                w_hi = [pools[(t, hi)]["wall_s"] for t in both]
                w_lo = [pools[(t, lo)]["wall_s"] for t in both]
                if all(w for w in w_hi):
                    h_hi = sum(w_hi) / len(both) / 3600
                    gains[f"{lo}->{hi}"]["cost_h_per_target"] = round(h_hi, 4)
                    if h_hi > 0:
                        gains[f"{lo}->{hi}"]["marginal_oracle_per_1000cs"] = \
                            round(g["oracle"][1] / (h_hi * 3.6), 6)
                    # Same target set at the lower rung, so "did the second 256
                    # samples cost more than the first 256" is answerable without the
                    # panel-composition confound (rung 512's panel is overlay-heavy
                    # hard targets, which are slower per sample for reasons that have
                    # nothing to do with the doubling).
                    if all(w for w in w_lo):
                        gains[f"{lo}->{hi}"]["cost_h_per_target_lo_same_panel"] = \
                            round(sum(w_lo) / len(both) / 3600, 4)
            if gains:
                report[model + "__pairwise_gain_ci"] = gains
                print("  pairwise adjacent-rung gain CIs (stop-rule basis):")
                for pair, d in gains.items():
                    o = d["gain_ci"]["oracle"]
                    marg = d.get("marginal_oracle_per_1000cs")
                    cph = d.get("cost_h_per_target")
                    print(f"    {pair:<12} nt={d['common_targets']:<4} "
                          f"oracle gain {o[1]:+.4f} [{o[0]:+.4f},{o[2]:+.4f}]"
                          + (f"  marginal {marg:+.6f}/1000 card-s per target "
                             f"({cph:.2f} card-h/target)" if marg is not None else ""))
            ds = deep_stats(pools, model)
            report[model + "__deep"] = ds
            print(f"  within-fold oracle curve (top rung N={ds['top_rung']}, "
                  f"{ds['n_targets']} targets):")
            for m, v in ds["within_fold_oracle_curve"].items():
                fl = ds["seed_noise_floor_med"].get(str(m)) or ds["seed_noise_floor_med"].get(m)
                print(f"    m={m:<4} E[oracle]={v:.4f}" + (f"  floor={fl:.4f}" if fl else ""))
            print(f"  solvable at top rung: {ds['solvable_at_top']}")
    if os.environ.get("DEEPN_N16_ARK") == "1":
        # models whose N=16 rung is the ARK restatement (same flavor as the qb1 arms);
        # the datasheet keys its cross-flavor flags/prose off this
        report["n16_ark_models"] = sorted(ark16)
    Path(a.out).write_text(json.dumps(report, indent=1))
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
