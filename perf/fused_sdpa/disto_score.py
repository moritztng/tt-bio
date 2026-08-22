#!/usr/bin/env python3
"""Score the two triangle-attention arms on RF3's distogram, which carries no sampler noise.

Global CA RMSD on cdk2x2_298 reports which basin the sampler drew, not how accurate the kernel is
(state/fused-sdpa-adopt.md §0: the basins are shared across arms and both arms visit the same
alternate basin). The distogram is `linear(z + z.T)` at rf3/model.py:324, computed BEFORE
`sampler.sample` at :338, so it is a direct linear readout of the trunk pair representation --
which is where triangle attention lives -- and it is bit-identical at any sampling-step count
(proven byte-for-byte: perf/fused_sdpa/disto/298/proof50 at 50 steps == disto/298/def at 5).

Two edge-free metrics. RF3's 65 bin edges are not in the repo (the head emits bare logits and
nothing consumes them), and both metrics below are monotone in distance by construction, so
nothing here needs them:

  PRIMARY   Spearman rho between the expected bin index E[b] = sum_b b*softmax(logits)_b and the
            1HCL crystal CA-CA distance, over pairs with true distance in [4, 30] A. The 4 A floor
            drops sequence-adjacent pairs (CA i,i+1 is ~3.8 A in every structure, so they inflate
            rho identically for both arms and dilute the signal); 30 A is a no-op guard past any
            plausible bin range. The SIGN of rho fixes which end of the bin axis is near, and
            abs(rho) >= 0.8 on the shipped arm is the instrument validation -- if it fails, the
            head is not doing what this assumes and the step is void.

  SECONDARY top-L/5 long-range contact precision, L = scored residues, pairs |i-j| >= 24 (the CASP
            long-range band), against the crystal's own contacts at CA-CA < 8 A. Ranked two ways
            that need no edges: by E[b] toward the near end, and by cumulative probability over
            the near bins, with the near/far bin cut CALIBRATED ON THE SHIPPED ARM and then applied
            unchanged to both, so the calibration cannot favour either.

CI: residue-BLOCK bootstrap, blocks of 10 consecutive residues, because neighbouring residues'
pair errors are correlated and an i.i.d. residue bootstrap would understate the interval.

cdk2x2_512 is CDK2 followed by its own residues 1-214, so every pair set is restricted to
within-segment pairs via of3_score_ref.GT_SEGMENTS
(memory `cdk2x2-chimeric-fixture-cannot-score-non-bit-exact-parity`).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from of3_score_ref import ca_map, GT_SEGMENTS      # noqa: E402

ARMS = ["def", "hifi"]
DMIN, DMAX = 4.0, 30.0
LONG_RANGE = 24
CONTACT = 8.0
BLOCK = 10
NBOOT = 2000
RNG = np.random.default_rng(0)


def ranks(v):
    """Ordinal ranks. Distances and E[b] are continuous floats, so ties are negligible."""
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
    """E[b] over the last axis, in a numerically safe softmax."""
    z = logits - logits.max(-1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(-1, keepdims=True)
    b = np.arange(logits.shape[-1], dtype=np.float64)
    return p @ b, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--dir", type=Path, default=None, help="perf/fused_sdpa/disto/<size>")
    ap.add_argument("--gt", type=Path, default=HERE / "cifs" / "1hcl.cif")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    d = a.dir or HERE / "disto" / str(a.size)
    out_path = a.out or HERE / f"disto_score_{a.size}.json"

    gt = ca_map(a.gt)
    folds = {arm: sorted((d / arm).glob("f*_seed*")) for arm in ARMS}
    seeds = [int(p.name.split("seed")[1]) for p in folds["def"]]
    assert seeds == [int(p.name.split("seed")[1]) for p in folds["hifi"]], "arm seeds differ"
    print(f"size {a.size}  seeds {seeds}  arms {ARMS}")

    # distograms, and the fold's own residue numbering
    E = {arm: [] for arm in ARMS}
    P = {arm: [] for arm in ARMS}
    for arm in ARMS:
        for p in folds[arm]:
            lg = np.load(p / "distogram.npy")
            assert lg.shape[0] == 1 and lg.shape[1] == lg.shape[2], lg.shape
            e, pr = expected_bin(lg[0].astype(np.float64))
            E[arm].append(e)
            P[arm].append(pr)
    n_tok = E["def"][0].shape[0]
    cif = next(folds["def"][0].glob("*.cif"))
    fold_ca = ca_map(cif)
    res_ids = sorted(fold_ca)
    assert len(res_ids) == n_tok, \
        f"{len(res_ids)} CA residues vs {n_tok} distogram tokens -- token/residue map unclear"
    tok_of = {r: i for i, r in enumerate(res_ids)}   # token index of a fold residue id
    n_bins = P["def"][0].shape[-1]
    print(f"tokens {n_tok}  bins {n_bins}  cif {cif.name}")

    report = {"size": a.size, "seeds": seeds, "n_tokens": n_tok, "n_bins": n_bins,
              "sampling_steps": json.loads((d / "def" / "fold.json").read_text())["sampling_steps"],
              "diffusion_samples": json.loads(
                  (d / "def" / "fold.json").read_text())["diffusion_samples"],
              "segments": {}}

    for label, pairs in GT_SEGMENTS[a.size].items():
        use = [(f, g) for f, g in pairs if f in fold_ca and g in gt]
        bad = [(f, g) for f, g in use if fold_ca[f][0] != gt[g][0]]
        assert not bad, f"identity mismatch vs 1HCL {bad[:5]}"
        L = len(use)
        ti = np.array([tok_of[f] for f, _ in use])           # token index per scored residue
        gx = np.array([gt[g][1] for _, g in use])            # crystal CA coords
        D = np.linalg.norm(gx[:, None] - gx[None], axis=-1)  # true CA-CA, within segment only
        iu, ju = np.triu_indices(L, 1)
        keep = (D[iu, ju] >= DMIN) & (D[iu, ju] <= DMAX)
        pi, pj = iu[keep], ju[keep]
        dtrue = D[pi, pj]
        print(f"\n=== {a.size} aa [{label}] ===  {L} scored residues, "
              f"{len(dtrue)} within-segment pairs in [{DMIN},{DMAX}] A")

        # E[b] on the scored token subset, per arm per seed
        Eseg = {arm: [e[np.ix_(ti, ti)][pi, pj] for e in E[arm]] for arm in ARMS}

        # --- PRIMARY: Spearman, and the instrument validation
        rho = {arm: [spearman(v, dtrue) for v in Eseg[arm]] for arm in ARMS}
        r0 = rho["def"]
        print(f"  rho(shipped) per seed {np.round(r0,5).tolist()}   "
              f"spread {max(r0)-min(r0):.5f}")
        print(f"  rho(fused)   per seed {np.round(rho['hifi'],5).tolist()}   "
              f"spread {max(rho['hifi'])-min(rho['hifi']):.5f}")
        # Pre-registered void condition (PLAN2 §1 Step 3): if the shipped arm cannot track the
        # crystal, the head is not doing what this step assumes. Report and stop for this segment
        # rather than crashing, so the void itself is banked as a result -- but do NOT let a void
        # segment contribute a verdict.
        if abs(np.mean(r0)) < 0.8:
            print(f"  INSTRUMENT VOID: abs(rho) on the shipped arm is {np.mean(r0):.4f} < 0.8. "
                  f"No verdict from this segment.")
            print(f"  (direction only, NOT a verdict: fused - shipped mean "
                  f"{np.mean(rho['hifi']) - np.mean(r0):+.5f}, shipped seed spread "
                  f"{max(r0)-min(r0):.5f})")
            report["segments"][label] = {
                "n_residues": L, "n_pairs": int(len(dtrue)),
                "rho": {arm: [round(v, 6) for v in rho[arm]] for arm in ARMS},
                "rho_shipped_mean": round(float(np.mean(r0)), 6),
                "rho_shipped_spread": round(float(max(r0) - min(r0)), 6),
                "rho_margin_mean_DIRECTION_ONLY": round(
                    float(np.mean(rho["hifi"]) - np.mean(r0)), 6),
                "verdict": "VOID",
                "void_reason": f"abs(rho) shipped {np.mean(r0):.4f} < 0.8 pre-registered floor",
            }
            continue
        near_is_low = np.mean(r0) > 0     # rho > 0 => bin index rises with distance
        print(f"  instrument OK: abs(rho)={abs(np.mean(r0)):.4f} >= 0.8; "
              f"near end of the bin axis is {'LOW' if near_is_low else 'HIGH'} indices")
        margin_per_seed = [rho["hifi"][k] - rho["def"][k] for k in range(len(seeds))]
        print(f"  rho margin (fused - shipped) per seed "
              f"{np.round(margin_per_seed,5).tolist()}   mean {np.mean(margin_per_seed):+.5f}")

        # --- residue-BLOCK bootstrap on the seed-averaged rho margin
        blocks = [np.arange(s, min(s + BLOCK, L)) for s in range(0, L, BLOCK)]
        # dense per-arm per-seed E[b] over the scored residues, for fast re-pairing
        Efull = {arm: [e[np.ix_(ti, ti)] for e in E[arm]] for arm in ARMS}
        boot = np.empty(NBOOT)
        for t in range(NBOOT):
            sel = np.concatenate([blocks[k] for k in
                                  RNG.integers(0, len(blocks), len(blocks))])
            bi, bj = np.triu_indices(len(sel), 1)
            si, sj = sel[bi], sel[bj]
            dt = D[si, sj]
            m = (dt >= DMIN) & (dt <= DMAX)
            si, sj, dt = si[m], sj[m], dt[m]
            mk = [spearman(Efull["hifi"][k][si, sj], dt)
                  - spearman(Efull["def"][k][si, sj], dt) for k in range(len(seeds))]
            boot[t] = np.mean(mk)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"  block bootstrap ({len(blocks)} blocks of <={BLOCK}, {NBOOT} resamples) "
              f"95% CI on the mean rho margin: [{lo:+.5f}, {hi:+.5f}]")

        # --- pre-registered rule: margin -0.005, tightened to a quarter of the shipped spread
        #     if that spread is under 0.02 (PLAN2 §1 Step 3, fixed before this ran)
        spread = max(r0) - min(r0)
        thr = -0.005 if spread >= 0.02 else -spread / 4
        verdict = "ADOPT-side" if lo > thr else "FLOOR-side"
        print(f"  RULE: shipped rho spread {spread:.5f} -> threshold {thr:+.6f}; "
              f"CI lower bound {lo:+.5f}  =>  {verdict}")

        # --- SECONDARY: top-L/5 long-range contact precision, two edge-free rankings
        topk = max(1, L // 5)
        lr = np.abs(pi - pj) >= LONG_RANGE
        # (a) rank by E[b] toward the near end
        prec_e = {}
        for arm in ARMS:
            vals = []
            for v in Eseg[arm]:
                sc = v[lr] if near_is_low else -v[lr]
                order = np.argsort(sc, kind="stable")[:topk]
                vals.append(float((dtrue[lr][order] < CONTACT).mean()))
            prec_e[arm] = vals
        # (b) rank by cumulative probability over the near bins, cut calibrated on the SHIPPED arm
        #     as the E[b] value whose pairs straddle CONTACT, then applied unchanged to both arms
        e0 = Eseg["def"][0]
        cut = float(np.interp(CONTACT, np.sort(dtrue), np.sort(e0) if near_is_low
                              else np.sort(e0)[::-1]))
        bcut = int(round(cut))
        prec_p = {}
        for arm in ARMS:
            vals = []
            for pr in P[arm]:
                sub = pr[np.ix_(ti, ti)][pi, pj]
                pnear = (sub[:, :bcut].sum(-1) if near_is_low else sub[:, bcut:].sum(-1))
                order = np.argsort(-pnear[lr], kind="stable")[:topk]
                vals.append(float((dtrue[lr][order] < CONTACT).mean()))
            prec_p[arm] = vals
        print(f"  top-{topk} long-range (|i-j|>={LONG_RANGE}) contact precision, "
              f"{lr.sum()} candidate pairs")
        print(f"    ranked by E[b]      shipped {np.round(prec_e['def'],4).tolist()}  "
              f"fused {np.round(prec_e['hifi'],4).tolist()}")
        print(f"    ranked by P(bin<{bcut})  shipped {np.round(prec_p['def'],4).tolist()}  "
              f"fused {np.round(prec_p['hifi'],4).tolist()}")

        report["segments"][label] = {
            "n_residues": L, "n_pairs": int(len(dtrue)),
            "rho": {arm: [round(v, 6) for v in rho[arm]] for arm in ARMS},
            "rho_shipped_spread": round(spread, 6),
            "rho_margin_per_seed": [round(v, 6) for v in margin_per_seed],
            "rho_margin_mean": round(float(np.mean(margin_per_seed)), 6),
            "rho_margin_boot_ci95": [round(float(lo), 6), round(float(hi), 6)],
            "threshold": round(thr, 6), "verdict": verdict,
            "near_is_low_bin": bool(near_is_low),
            "topk": topk, "n_long_range_pairs": int(lr.sum()), "bin_cut": bcut,
            "contact_precision_by_Eb": {arm: [round(v, 6) for v in prec_e[arm]] for arm in ARMS},
            "contact_precision_by_Pnear": {arm: [round(v, 6) for v in prec_p[arm]]
                                           for arm in ARMS},
        }

    out_path.write_text(json.dumps(report, indent=1) + "\n")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
