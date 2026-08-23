#!/usr/bin/env python3
"""Score `TT_BIO_TRIATT_FUSED_HIFI` on RF3's distogram across SEVERAL REAL TARGETS at one rung.

`disto_score.py` is the single-target instrument that produced the FLOOR in
state/fused-sdpa-adopt.md. It reads one fixture family (`cdk2x2_*`, CDK2 tandem-repeated) against
one crystal (1HCL). That answered "does the lever lose accuracy on this target"; it cannot answer
"does it lose accuracy", and above 298 aa the fixture is a chimera whose inter-domain geometry has
no ground truth at all. This scorer takes the targets built by `build_targets.py` -- real single
chains, real crystals, one rung so every fold runs the same padded length and therefore the same
fused k_chunk -- and pools them.

The unit of independence is the TARGET, not the seed. Seeds resample one protein's own trunk; two
targets are two different proteins. So the headline interval is a cluster bootstrap over targets
with seeds nested inside, and the seed axis only sharpens each target's own estimate.

Three gates run per target before its margin is allowed into the pool, each of which has already
cost this campaign a wrong answer once:

  DARK     `triatt_fused_hifi_stats` in the fused arm must show served > 0 and declined ==
           too_short == 0. §1d of the state doc scored two OpenFold3 anchors that were an A/A
           because the kernel silently declined every call. A margin of zero from a lever that
           never ran is not evidence of safety.
  VOID     the shipped arm's |rho| must reach 0.80. Below that the distogram head is not tracking
           the crystal and the margin measures nothing. Pre-registered in PLAN2 §1 Step 3.
  A/A      the two arms must not be byte-identical. If they are, the fold did not read the flag.

Verdict rule, fixed here BEFORE any fold runs, with constants that do not depend on n:

  delta = 0.0020 rho, the smallest loss worth declining the lever for. Anchored to two measured
  numbers, neither from this run: the 298 aa loss already rejected is 0.00852, and the shipped
  arm's own cross-seed rho SD is 0.0024 at 298 aa and 0.0010 at 512 aa. delta sits 4.3x below the
  rejected effect and at about the shipped arm's own seed noise.

  FLOOR       cross-target 95 % CI upper bound < 0 AND >= 3/4 of targets one-signed negative
  GO          cross-target 95 % CI lower bound > -delta          (non-inferiority at delta)
  UNRESOLVED  otherwise. The deliverable is then the CI lower bound: the loss is at most that.

The old rule (`threshold = -(shipped seed range)/4`) is NOT reused. A range grows with n, so that
threshold loosens as seeds are added -- adding folds would have made adoption easier rather than
the estimate sharper, which is not a property a pre-registered bar may have.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ARMS = ["def", "hifi"]
DMIN, DMAX = 4.0, 30.0
LONG_RANGE = 24
CONTACT = 8.0
BLOCK = 10
NBOOT = 2000
NBOOT_CLUSTER = 20000
RHO_FLOOR = 0.80
DELTA = 0.0020
RNG = np.random.default_rng(0)
# One positive target must not veto a decisive interval, and one negative target must not
# carry a FLOOR. Three quarters, rounded up.
MIN_NEG = lambda k: -(-3 * k // 4)


def ranks(v):
    o = np.argsort(v, kind="stable")
    r = np.empty(len(v), dtype=np.float64)
    r[o] = np.arange(len(v), dtype=np.float64)
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    ra -= ra.mean()
    rb -= rb.mean()
    return float(ra @ rb / np.sqrt((ra @ ra) * (rb @ rb)))


def expected_bin(logits):
    """E[b] over the last axis. RF3 emits bare logits and its 65 bin edges are not in the repo,
    so every metric here is monotone in distance by construction and never needs them."""
    z = logits - logits.max(-1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(-1, keepdims=True)
    return p @ np.arange(logits.shape[-1], dtype=np.float64)


def score_target(name, tdir, gt):
    """One target: paired rho margins per seed, plus the three gates. None if a gate fires."""
    folds = {arm: sorted((tdir / arm).glob("f*_seed*")) for arm in ARMS}
    if not folds["def"] or not folds["hifi"]:
        return {"name": name, "verdict": "MISSING", "reason": "no folds for one of the arms"}
    seeds = [int(p.name.split("seed")[1]) for p in folds["def"]]
    assert seeds == [int(p.name.split("seed")[1]) for p in folds["hifi"]], f"{name}: arm seeds differ"

    # --- DARK gate, off the fold record rather than off an assumption
    fj = json.loads((tdir / "hifi" / "fold.json").read_text())
    st = [f["triatt_fused_hifi_stats"] for f in fj["folds"]]
    served = sum(s["served"] for s in st)
    bad = sum(s["declined"] + s["too_short"] for s in st)
    if served == 0 or bad:
        return {"name": name, "verdict": "DARK", "triatt_stats": st,
                "reason": f"fused arm served {served}, declined+too_short {bad}"}

    E, shas = {}, {}
    for arm in ARMS:
        E[arm], shas[arm] = [], []
        for p in folds[arm]:
            lg = np.load(p / "distogram.npy")
            assert lg.ndim == 4 and lg.shape[0] == 1 and lg.shape[1] == lg.shape[2], lg.shape
            E[arm].append(expected_bin(lg[0].astype(np.float64)))
            shas[arm].append(hashlib.sha256(lg.tobytes()).hexdigest())
    if shas["def"] == shas["hifi"]:
        return {"name": name, "verdict": "AA", "reason": "both arms produced identical distograms"}

    n_tok = E["def"][0].shape[0]
    n_res = gt["n_res"]
    assert n_tok == n_res, \
        f"{name}: {n_tok} distogram tokens vs {n_res} sequence residues -- token map is not 1:1"

    # fold token i <-> entity position i+1 <-> mmCIF label_seq_id i+1, exactly, no alignment
    idx = np.array(sorted(int(k) for k in gt["ca"]), dtype=int)
    ti = idx - 1
    gx = np.array([gt["ca"][str(i)] for i in idx], dtype=np.float64)
    L = len(idx)
    D = np.linalg.norm(gx[:, None] - gx[None], axis=-1)
    iu, ju = np.triu_indices(L, 1)
    keep = (D[iu, ju] >= DMIN) & (D[iu, ju] <= DMAX)
    pi, pj = iu[keep], ju[keep]
    dtrue = D[pi, pj]

    Efull = {arm: [e[np.ix_(ti, ti)] for e in E[arm]] for arm in ARMS}
    rho = {arm: [spearman(m[pi, pj], dtrue) for m in Efull[arm]] for arm in ARMS}
    r0 = rho["def"]
    if abs(float(np.mean(r0))) < RHO_FLOOR:
        return {"name": name, "verdict": "VOID", "rho_shipped": r0,
                "reason": f"|rho| shipped {np.mean(r0):.4f} < {RHO_FLOOR}"}
    near_is_low = float(np.mean(r0)) > 0

    margin = [rho["hifi"][k] - rho["def"][k] for k in range(len(seeds))]

    # within-target residue-block bootstrap, as in disto_score.py
    blocks = [np.arange(s, min(s + BLOCK, L)) for s in range(0, L, BLOCK)]
    boot = np.empty(NBOOT)
    for t in range(NBOOT):
        sel = np.concatenate([blocks[k] for k in RNG.integers(0, len(blocks), len(blocks))])
        bi, bj = np.triu_indices(len(sel), 1)
        si, sj = sel[bi], sel[bj]
        dt = D[si, sj]
        m = (dt >= DMIN) & (dt <= DMAX)
        si, sj, dt = si[m], sj[m], dt[m]
        boot[t] = np.mean([spearman(Efull["hifi"][k][si, sj], dt)
                           - spearman(Efull["def"][k][si, sj], dt) for k in range(len(seeds))])
    lo, hi = np.percentile(boot, [2.5, 97.5])

    # corroborator: top-L/5 long-range contact precision, ranked by E[b]
    topk = max(1, L // 5)
    lr = np.abs(pi - pj) >= LONG_RANGE
    prec = {}
    for arm in ARMS:
        vals = []
        for m in Efull[arm]:
            v = m[pi, pj][lr]
            order = np.argsort(v if near_is_low else -v, kind="stable")[:topk]
            vals.append(float((dtrue[lr][order] < CONTACT).mean()))
        prec[arm] = vals

    conf = {}
    for arm in ARMS:
        j = json.loads((tdir / arm / "fold.json").read_text())
        conf[arm] = {"plddt": [f["plddt"] for f in j["folds"]],
                     "ptm": [f["ptm"] for f in j["folds"]],
                     "fold_s": [f["fold_s"] for f in j["folds"]]}

    return {"name": name, "verdict": "SCORED", "pdb": gt["pdb"], "n_res": n_res,
            "n_scored_residues": L, "n_pairs": int(len(dtrue)), "seeds": seeds,
            "served_per_fold": [s["served"] for s in st],
            "rho": {a: [round(v, 6) for v in rho[a]] for a in ARMS},
            "rho_shipped_mean": round(float(np.mean(r0)), 6),
            "rho_shipped_sd": round(float(np.std(r0, ddof=1)) if len(r0) > 1 else 0.0, 6),
            "margin_per_seed": [round(v, 6) for v in margin],
            "margin_mean": round(float(np.mean(margin)), 6),
            "margin_block_ci95": [round(float(lo), 6), round(float(hi), 6)],
            "near_is_low_bin": bool(near_is_low),
            "contact_precision_topL5": {a: [round(v, 6) for v in prec[a]] for a in ARMS},
            "confidence": conf}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", type=int, required=True)
    ap.add_argument("--dir", type=Path, default=None, help="default perf/fused_sdpa/disto_multi/<rung>")
    ap.add_argument("--targets", type=Path, default=HERE / "targets")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    root = a.dir or HERE / "disto_multi" / str(a.rung)
    out = a.out or HERE / f"disto_multi_{a.rung}.json"

    names = sorted(p.name for p in root.iterdir() if p.is_dir())
    assert names, f"no target directories under {root}"
    rep = {"rung": a.rung, "delta": DELTA, "targets": {}}
    scored = []
    for n in names:
        gt = json.loads((a.targets / f"{n}.gt.json").read_text())
        assert gt["padded"] == a.rung, \
            f"{n} pads to {gt['padded']}, not rung {a.rung} -- mixing rungs mixes k_chunks"
        r = score_target(n, root / n, gt)
        rep["targets"][n] = r
        v = r["verdict"]
        if v != "SCORED":
            print(f"  {n:<12} {v}: {r.get('reason')}")
            continue
        scored.append(r)
        print(f"  {n:<12} rho {r['rho_shipped_mean']:.5f}  margin {r['margin_mean']:+.5f}  "
              f"block CI [{r['margin_block_ci95'][0]:+.5f}, {r['margin_block_ci95'][1]:+.5f}]  "
              f"served/fold {r['served_per_fold'][0]}")

    K = len(scored)
    assert K >= 2, f"only {K} target(s) survived the gates; a cross-target verdict needs >= 2"
    per_t = np.array([r["margin_mean"] for r in scored])
    all_m = np.array([m for r in scored for m in r["margin_per_seed"]])

    # cluster bootstrap: targets are the independent unit, seeds nested
    cb = np.empty(NBOOT_CLUSTER)
    per_seed = [np.array(r["margin_per_seed"]) for r in scored]
    for t in range(NBOOT_CLUSTER):
        pick = RNG.integers(0, K, K)
        cb[t] = np.mean([per_seed[k][RNG.integers(0, len(per_seed[k]), len(per_seed[k]))].mean()
                         for k in pick])
    clo, chi = np.percentile(cb, [2.5, 97.5])
    mean = float(per_t.mean())
    neg_t = int((per_t < 0).sum())
    neg_f = int((all_m < 0).sum())

    if chi < 0 and neg_t >= MIN_NEG(K):
        verdict = "FLOOR"
    elif clo > -DELTA:
        verdict = "GO"
    else:
        verdict = "UNRESOLVED"

    print(f"\nrung {a.rung}: {K} targets scored, {len(all_m)} paired folds")
    print(f"  per-target mean margins {np.round(per_t, 5).tolist()}")
    print(f"  cross-target mean {mean:+.5f}   one-signed negative {neg_t}/{K} targets, "
          f"{neg_f}/{len(all_m)} folds")
    print(f"  cluster bootstrap 95% CI [{clo:+.5f}, {chi:+.5f}]   delta {-DELTA:+.5f}")
    print(f"  VERDICT {verdict}")
    if verdict == "UNRESOLVED":
        print(f"  bound: the loss at rung {a.rung} is at most {-clo:.5f} rho at 95% confidence")

    rep["pooled"] = {"n_targets": K, "n_folds": int(len(all_m)),
                     "per_target_mean_margin": [round(float(x), 6) for x in per_t],
                     "cross_target_mean": round(mean, 6),
                     "cluster_boot_ci95": [round(float(clo), 6), round(float(chi), 6)],
                     "targets_negative": neg_t, "min_negative_for_floor": MIN_NEG(K),
                     "folds_negative": neg_f,
                     "verdict": verdict}
    out.write_text(json.dumps(rep, indent=1) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
