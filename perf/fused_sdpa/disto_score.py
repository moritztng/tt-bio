#!/usr/bin/env python3
"""Score the two triangle-attention arms on the distogram, which carries no sampler noise.

Global CA RMSD on cdk2x2_298 reports which basin the sampler drew, not how accurate the kernel is
(state/fused-sdpa-adopt.md §0: the basins are shared across arms and both arms visit the same
alternate basin). The distogram is `linear(z + z.T)` at rf3/model.py:324, computed BEFORE
`sampler.sample` at :338, so it is a direct linear readout of the trunk pair representation --
which is where triangle attention lives -- and it is bit-identical at any sampling-step count
(proven byte-for-byte: perf/fused_sdpa/disto/298/proof50 at 50 steps == disto/298/def at 5).
OpenFold3's distogram is the same kind of readout, built from `zij_trunk` at
openfold3_confidence.py:196-197, and the same proof holds for it (disto_of3/298/proof200).

Two modes:

  --size 298|512    RF3 on cdk2x2, scored against 1HCL through of3_score_ref.ca_map. This is the
                    path that produced the committed disto_score_298.json / _512.json and the
                    FLOOR verdict; it is frozen and must re-run byte-identical.
  --anchor <name>   OpenFold3 on one of the anchors in ANCHORS below, scored against that anchor's
                    own crystal through of3_score_ref.ca_map_chains, which keys on the chain as
                    well as the residue number. cdk2x2 cannot serve OF3: it folds it at plDDT
                    0.516 and rho 0.7375, below the pre-registered 0.8 floor, because its 35-row
                    MSA is RF3's fixture and not an OF3 anchor.

Two edge-free metrics. Neither model's bin edges are needed: RF3 emits bare logits and its 65
edges are not in the repo, and OF3's edges are `linspace(_MIN_BIN, _MAX_BIN, _NO_BIN)`, so
expected distance is an affine function of expected bin index and Spearman is invariant under it.

  PRIMARY   Spearman rho between the expected bin index E[b] = sum_b b*softmax(logits)_b and the
            crystal CA-CA distance, over pairs with true distance in [4, 30] A. The 4 A floor
            drops sequence-adjacent pairs (CA i,i+1 is ~3.8 A in every structure, so they inflate
            rho identically for both arms and dilute the signal); 30 A is a no-op guard past any
            plausible bin range. The SIGN of rho fixes which end of the bin axis is near, and
            abs(rho) >= 0.8 on the shipped arm is the instrument validation -- if it fails, the
            head is not doing what this assumes and the segment is void.

  SECONDARY top-L/5 long-range contact precision, L = scored residues, pairs |i-j| >= 24 (the CASP
            long-range band), against the crystal's own contacts at CA-CA < 8 A. Ranked two ways
            that need no edges: by E[b] toward the near end, and by cumulative probability over
            the near bins, with the near/far bin cut CALIBRATED ON THE SHIPPED ARM and then applied
            unchanged to both, so the calibration cannot favour either. On a cross-chain segment
            every pair is inter-chain, so the |i-j| filter is not applied there.

CI: residue-BLOCK bootstrap, blocks of 10 consecutive residues, because neighbouring residues'
pair errors are correlated and an i.i.d. residue bootstrap would understate the interval. Blocks
are built within a chain, never across one, so a cross-chain segment resamples each side of the
interface on its own.

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
from of3_score_ref import ca_map, ca_map_chains, GT_SEGMENTS      # noqa: E402

# Default is the pair the committed RF3 scorings were produced with, so `--size 298/512` with no
# `--arms` re-runs byte-identical (disto_score_regress.py proves it). `--arms BASE,TEST` scores any
# other pair out of the same fold tree: the margin is always TEST - BASE, the instrument validation
# and the threshold are always read off BASE, so which arm is the reference is explicit rather than
# positional luck.
ARMS = ["def", "hifi"]
DMIN, DMAX = 4.0, 30.0
LONG_RANGE = 24
CONTACT = 8.0
BLOCK = 10
NBOOT = 2000
RNG = np.random.default_rng(0)

# OpenFold3 anchors. `chains` is the fixture's own per-chain length, asserted against the fold's
# CA census. Every residue range below was read off the downloaded crystal, and in all three cases
# the fold-to-crystal map is the identity on residue number, so a segment's pair list is (k, k):
#   1AO6  2.5 A   chains A and B are two copies of HSA, each resolving label_seq_id 5..582 of the
#                 585-aa construct. Score chain A only; B is a second lattice copy.
#   1A8Q  1.75 A  all 274 residues resolved, label_seq_id 1..274, sequence identical to the
#                 fixture.
#   9BK6  2.0 A   chain A resolves 3..103 of 104 (MSG at 1-3 partly unresolved), chain B all 60.
ANCHORS = {
    "hsa_585": dict(
        gt="1ao6.cif", fixture="of3_hsa_585", chains={"A": 585},
        segments={"A": ("within", [("A", i) for i in range(5, 583)])}),
    "1a8q_274": dict(
        gt="1a8q.cif", fixture="of3_1a8q_274", chains={"A": 274},
        segments={"A": ("within", [("A", i) for i in range(1, 275)])}),
    "9bk6_164": dict(
        gt="9bk6.cif", fixture="9bk6", chains={"A": 104, "B": 60},
        segments={"A": ("within", [("A", i) for i in range(3, 104)]),
                  "B": ("within", [("B", i) for i in range(1, 61)]),
                  "AB": ("cross", ([("A", i) for i in range(3, 104)],
                                   [("B", i) for i in range(1, 61)]))},
        # 60 residues is 6 bootstrap blocks, too few for a 95% interval (PLAN3 §3)
        report_only={"B"}),
}


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


def pair_index(parts, mode):
    """Index pairs into a scored-residue list: the upper triangle, or the A-B cross product."""
    if mode == "cross":
        ai = np.where(parts == 0)[0]
        bi = np.where(parts == 1)[0]
        return np.repeat(ai, len(bi)), np.tile(bi, len(ai))
    return np.triu_indices(len(parts), 1)


def blocks_of(parts):
    """Bootstrap blocks of <=BLOCK consecutive residues, grouped by chain.

    Returns one list of blocks per chain. Resampling within each group rather than from the pooled
    list is what keeps a cross-chain segment scoreable: a pooled draw can land entirely inside one
    chain, leave zero inter-chain pairs, and return NaN. For a single-chain segment there is one
    group and the draw is identical to a pooled one, RNG consumption included.
    """
    return [[idx[s:s + BLOCK] for s in range(0, len(idx), BLOCK)]
            for idx in (np.where(parts == p)[0] for p in np.unique(parts))]


def resample(groups, rng):
    """One bootstrap resample: len(g) blocks drawn with replacement from each chain's own blocks."""
    return np.concatenate([g[k] for g in groups
                           for k in rng.integers(0, len(g), len(g))])


def resolve_segments(a):
    """(gt map, {label: (mode, [(fold_key, gt_key)...])}, fold-key mapper, dir, out path)."""
    if a.size is not None:
        d = a.dir or HERE / "disto" / str(a.size)
        out = a.out or HERE / f"disto_score_{a.size}.json"
        segs = {k: ("within", v) for k, v in GT_SEGMENTS[a.size].items()}
        return ca_map(a.gt or HERE / "cifs" / "1hcl.cif"), segs, ca_map, d, out, None
    an = ANCHORS[a.anchor]
    d = a.dir or HERE / "anchor" / an["fixture"]
    out = a.out or HERE / f"disto_of3_{a.anchor}.json"
    segs = {}
    for label, (mode, keys) in an["segments"].items():
        if mode == "cross":
            segs[label] = (mode, ([(k, k) for k in keys[0]], [(k, k) for k in keys[1]]))
        else:
            segs[label] = (mode, [(k, k) for k in keys])
    return (ca_map_chains(a.gt or HERE / "cifs" / an["gt"]), segs, ca_map_chains, d, out, an)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=None, help="RF3 cdk2x2: 298 or 512")
    ap.add_argument("--anchor", choices=sorted(ANCHORS), default=None,
                    help="OpenFold3 anchor, scored against its own crystal")
    ap.add_argument("--dir", type=Path, default=None,
                    help="dir holding <arm>/f<i>_seed<n>/ for both arms")
    ap.add_argument("--arms", default=",".join(ARMS),
                    help="BASE,TEST -- subdirectory names under --dir. The margin is TEST - BASE "
                         "and the threshold comes off BASE's own seed spread.")
    ap.add_argument("--gt", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    assert (a.size is None) != (a.anchor is None), "give exactly one of --size / --anchor"
    ARMS[:] = [x.strip() for x in a.arms.split(",") if x.strip()]
    assert len(ARMS) == 2 and len(set(ARMS)) == 2, f"--arms wants two distinct names, got {ARMS}"
    BASE, TEST = ARMS
    gt, SEGMENTS, cmap, d, out_path, an = resolve_segments(a)
    tag = f"size {a.size}" if a.size is not None else f"anchor {a.anchor}"

    folds = {arm: sorted((d / arm).glob("f*_seed*")) for arm in ARMS}
    all_seeds = [int(p.name.split("seed")[1]) for p in folds[BASE]]
    assert all_seeds == [int(p.name.split("seed")[1]) for p in folds[TEST]], "arm seeds differ"
    # A repeated seed is an A/A control, not a second sample. Score the first occurrence of each
    # and check the repeat against it; including it double-counts one draw, and
    # state/fused-sdpa-adopt.md §0 records that turning a -0.00059 lDDT margin into a -0.00791
    # with a CI excluding zero.
    keep, seen = [], {}
    for i, s in enumerate(all_seeds):
        if s in seen:
            continue
        seen[s] = i
        keep.append(i)
    repeats = [(i, seen[s]) for i, s in enumerate(all_seeds) if i not in keep]
    seeds = [all_seeds[i] for i in keep]
    print(f"{tag}  seeds {seeds}  arms {ARMS}"
          + (f"  (A/A controls dropped: {[all_seeds[i] for i, _ in repeats]})" if repeats else ""))

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
    for arm in ARMS:
        for i, j in repeats:
            assert np.array_equal(np.load(folds[arm][i] / "distogram.npy"),
                                  np.load(folds[arm][j] / "distogram.npy")), \
                f"{arm}: A/A control {folds[arm][i].name} does not reproduce {folds[arm][j].name}"
        E[arm] = [E[arm][i] for i in keep]
        P[arm] = [P[arm][i] for i in keep]
    n_tok = E[BASE][0].shape[0]
    cif = next(folds[BASE][0].glob("*.cif"))
    fold_ca = cmap(cif)
    census = None
    if an is not None:
        # Do not assume the fold writes the crystal's asym ids. Take the fold's chains in order,
        # map them onto the fixture's chains in order, and assert the CA counts.
        seen = sorted({k[0] for k in fold_ca})
        want = list(an["chains"])
        census = {c: sum(1 for k in fold_ca if k[0] == c) for c in seen}
        assert len(seen) == len(want), f"fold chains {census} vs fixture {an['chains']}"
        remap = dict(zip(seen, want))
        fold_ca = {(remap[c], i): v for (c, i), v in fold_ca.items()}
        got = {c: sum(1 for k in fold_ca if k[0] == c) for c in want}
        assert got == an["chains"], f"fold CA census {got} vs fixture {an['chains']}"
        print(f"fold chains {census} -> {remap}, CA census {got} matches the fixture")
    res_ids = sorted(fold_ca)
    assert len(res_ids) == n_tok, \
        f"{len(res_ids)} CA residues vs {n_tok} distogram tokens -- token/residue map unclear"
    tok_of = {r: i for i, r in enumerate(res_ids)}   # token index of a fold residue id
    n_bins = P[BASE][0].shape[-1]
    print(f"tokens {n_tok}  bins {n_bins}  cif {cif.name}")

    report = {"size": a.size, "seeds": seeds, "n_tokens": n_tok, "n_bins": n_bins,
              "aa_control_seeds": [all_seeds[i] for i, _ in repeats],
              "sampling_steps": json.loads((d / BASE / "fold.json").read_text())["sampling_steps"],
              "diffusion_samples": json.loads(
                  (d / BASE / "fold.json").read_text())["diffusion_samples"],
              "segments": {}}
    if an is not None:
        report["anchor"] = a.anchor
        report["gt"] = an["gt"]
        report["fold_chain_census"] = census

    for label, (mode, pairs) in SEGMENTS.items():
        if mode == "cross":
            left = [(f, g) for f, g in pairs[0] if f in fold_ca and g in gt]
            right = [(f, g) for f, g in pairs[1] if f in fold_ca and g in gt]
            use = left + right
            parts = np.array([0] * len(left) + [1] * len(right))
        else:
            use = [(f, g) for f, g in pairs if f in fold_ca and g in gt]
            parts = np.zeros(len(use), dtype=int)
        bad = [(f, g) for f, g in use if fold_ca[f][0] != gt[g][0]]
        assert not bad, f"identity mismatch vs the crystal {bad[:5]}"
        L = len(use)
        ti = np.array([tok_of[f] for f, _ in use])           # token index per scored residue
        gx = np.array([gt[g][1] for _, g in use])            # crystal CA coords
        D = np.linalg.norm(gx[:, None] - gx[None], axis=-1)  # true CA-CA, within segment only
        iu, ju = pair_index(parts, mode)
        keep = (D[iu, ju] >= DMIN) & (D[iu, ju] <= DMAX)
        pi, pj = iu[keep], ju[keep]
        dtrue = D[pi, pj]
        print(f"\n=== {tag} [{label}] ===  {L} scored residues, "
              f"{len(dtrue)} {mode}-segment pairs in [{DMIN},{DMAX}] A")

        # E[b] on the scored token subset, per arm per seed
        Eseg = {arm: [e[np.ix_(ti, ti)][pi, pj] for e in E[arm]] for arm in ARMS}

        # --- PRIMARY: Spearman, and the instrument validation
        rho = {arm: [spearman(v, dtrue) for v in Eseg[arm]] for arm in ARMS}
        r0 = rho[BASE]
        print(f"  rho({BASE}) per seed {np.round(r0,5).tolist()}   "
              f"spread {max(r0)-min(r0):.5f}")
        print(f"  rho({TEST}) per seed {np.round(rho[TEST],5).tolist()}   "
              f"spread {max(rho[TEST])-min(rho[TEST]):.5f}")
        # Pre-registered void condition (PLAN2 §1 Step 3): if the shipped arm cannot track the
        # crystal, the head is not doing what this step assumes. Report and stop for this segment
        # rather than crashing, so the void itself is banked as a result -- but do NOT let a void
        # segment contribute a verdict.
        if abs(np.mean(r0)) < 0.8:
            print(f"  INSTRUMENT VOID: abs(rho) on the shipped arm is {np.mean(r0):.4f} < 0.8. "
                  f"No verdict from this segment.")
            print(f"  (direction only, NOT a verdict: {TEST} - {BASE} mean "
                  f"{np.mean(rho[TEST]) - np.mean(r0):+.5f}, {BASE} seed spread "
                  f"{max(r0)-min(r0):.5f})")
            report["segments"][label] = {
                "n_residues": L, "n_pairs": int(len(dtrue)),
                "rho": {arm: [round(v, 6) for v in rho[arm]] for arm in ARMS},
                "rho_shipped_mean": round(float(np.mean(r0)), 6),
                "rho_shipped_spread": round(float(max(r0) - min(r0)), 6),
                "rho_margin_mean_DIRECTION_ONLY": round(
                    float(np.mean(rho[TEST]) - np.mean(r0)), 6),
                "verdict": "VOID",
                "void_reason": f"abs(rho) shipped {np.mean(r0):.4f} < 0.8 pre-registered floor",
            }
            continue
        near_is_low = np.mean(r0) > 0     # rho > 0 => bin index rises with distance
        print(f"  instrument OK: abs(rho)={abs(np.mean(r0)):.4f} >= 0.8; "
              f"near end of the bin axis is {'LOW' if near_is_low else 'HIGH'} indices")
        margin_per_seed = [rho[TEST][k] - rho[BASE][k] for k in range(len(seeds))]
        print(f"  rho margin ({TEST} - {BASE}) per seed "
              f"{np.round(margin_per_seed,5).tolist()}   mean {np.mean(margin_per_seed):+.5f}")

        # --- residue-BLOCK bootstrap on the seed-averaged rho margin
        groups = blocks_of(parts)
        n_blocks = sum(len(g) for g in groups)
        # dense per-arm per-seed E[b] over the scored residues, for fast re-pairing
        Efull = {arm: [e[np.ix_(ti, ti)] for e in E[arm]] for arm in ARMS}
        boot = np.empty(NBOOT)
        for t in range(NBOOT):
            sel = resample(groups, RNG)
            bi, bj = pair_index(parts[sel], mode)
            si, sj = sel[bi], sel[bj]
            dt = D[si, sj]
            m = (dt >= DMIN) & (dt <= DMAX)
            si, sj, dt = si[m], sj[m], dt[m]
            mk = [spearman(Efull[TEST][k][si, sj], dt)
                  - spearman(Efull[BASE][k][si, sj], dt) for k in range(len(seeds))]
            boot[t] = np.mean(mk)
        assert np.isfinite(boot).all(), \
            f"{label}: {int((~np.isfinite(boot)).sum())} of {NBOOT} resamples had no scoreable pairs"
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"  block bootstrap ({n_blocks} blocks of <={BLOCK}, {NBOOT} resamples) "
              f"95% CI on the mean rho margin: [{lo:+.5f}, {hi:+.5f}]")

        # --- pre-registered rule. RF3 (--size): margin -0.005, tightened to a quarter of the
        #     shipped spread if that spread is under 0.02 (PLAN2 §1 Step 3). OF3 (--anchor):
        #     T = min(0.005, max(0.002, spread/4)) (PLAN3 §3), the same -0.005 ceiling so OF3 is
        #     not held to a looser bar than RF3, with a floor at 0.002 because that is the
        #     smallest shipped-arm seed spread this instrument has ever produced (RF3 512 copy1,
        #     0.00191) and a threshold below it would assert resolution never demonstrated.
        spread = max(r0) - min(r0)
        if an is None:
            thr = -0.005 if spread >= 0.02 else -spread / 4
        else:
            thr = -min(0.005, max(0.002, spread / 4))
        verdict = "ADOPT-side" if lo > thr else "FLOOR-side"
        print(f"  RULE: {BASE} rho spread {spread:.5f} -> threshold {thr:+.6f}; "
              f"CI lower bound {lo:+.5f}  =>  {verdict}")

        # --- SECONDARY: top-L/5 long-range contact precision, two edge-free rankings
        topk = max(1, L // 5)
        lr = (np.ones(len(pi), dtype=bool) if mode == "cross"
              else np.abs(pi - pj) >= LONG_RANGE)
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
        e0 = Eseg[BASE][0]
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
        band = "inter-chain" if mode == "cross" else f"|i-j|>={LONG_RANGE}"
        print(f"  top-{topk} long-range ({band}) contact precision, "
              f"{lr.sum()} candidate pairs")
        print(f"    ranked by E[b]      {BASE} {np.round(prec_e[BASE],4).tolist()}  "
              f"{TEST} {np.round(prec_e[TEST],4).tolist()}")
        print(f"    ranked by P(bin<{bcut})  {BASE} {np.round(prec_p[BASE],4).tolist()}  "
              f"{TEST} {np.round(prec_p[TEST],4).tolist()}")

        report["segments"][label] = {
            "n_residues": L, "n_pairs": int(len(dtrue)),
            "gated": bool(an is None or label not in an.get("report_only", set())),
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
