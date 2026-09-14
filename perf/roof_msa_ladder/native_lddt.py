#!/usr/bin/env python3
"""CA-lDDT against the experimental structure 1HCL for both `TT_BIO_MSA_LADDER` arms.

`roof-msa-ladder` left the lever OFF on one reading: paired same-seed all-atom RMSD between the
512-rung and 1024-rung arms is 1.32206 A at 512 aa against a 0.60 A bar. RMSD between two runs of
a diffusion sampler moves when the trajectory moves, whether or not the answer got worse, so the
reading that decides it is distance to the experimental answer. That is what this computes.

Nothing here is a new scorer. `perf/b2z2_fusebias/score.py` already owns the 1HCL apparatus --
`native()` (per-pseudo-domain CA-lDDT and CA RMSD via `of3_score_ref.GT_SEGMENTS`/`ca_map` and
`basin_lddt.lddt_per_residue`) and `pair()` (arm-against-arm, all-atom and CA, Kabsch in fp64).
Both are imported and called unmodified. This file only maps the ladder leg's flat CIF layout
(`<size>_<tag>_<arm>_<target>.cif`) onto them and lines the numbers up.

    python3 perf/roof_msa_ladder/native_lddt.py --out perf/roof_msa_ladder/native_lddt.json \
        [--anchor-score /tmp/k10_score.json]

`--anchor-score` is `perf/k10_anchor/out/score.json` from `wk/k10-p1-accuracy-anchor`, which
carries the TT stack's own 4-seed CA-lDDT spread and the fp32-vs-bf16 control. Both are quoted
beside the deficit; neither is recomputed here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "perf" / "b2z2_fusebias"))
from score import load, native, pair, GT, spread                        # noqa: E402
sys.path.insert(0, str(REPO / "perf" / "fused_sdpa"))
from of3_score_ref import ca_map, GT_SEGMENTS                           # noqa: E402
from basin_lddt import lddt_per_residue                                 # noqa: E402

HERE = Path(__file__).resolve().parent
# arm "0" is TT_BIO_MSA_LADDER=0, the shipped single 1024 rung; arm "1" is the ladder.
ARM = {"0": "off", "1": "on"}
# Which directory holds which size, straight off what the leg wrote.
SESSIONS = {"s1": HERE / "cif", "s2": HERE / "cif_s2"}


def per_residue(cif: Path, size: int, gt: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per-residue CA-lDDT against 1HCL, keyed by pseudo-domain, plus the matched CA coordinates.

    One global lDDT per arm is n=1 and this leg holds a single seed, so the residues carry the
    statistics the seeds cannot. Same `lddt_per_residue` the global number comes out of -- it
    already returns the per-residue vector and `score.native()` throws it away.
    """
    pm = ca_map(cif)
    out = {}
    for name, pairs in GT_SEGMENTS[size].items():
        use = [(q, r) for q, r in pairs if q in pm and r in gt]
        A = np.array([pm[q][1] for q, _ in use])
        B = np.array([gt[r][1] for _, r in use])
        out[name] = (lddt_per_residue(A, B)[0], A, B)
    return out


def block_bootstrap(d: np.ndarray, block: int = 10, n: int = 20000, seed: int = 0) -> dict:
    """Mean of a per-residue delta with a moving-block CI. lDDT is a neighbourhood metric on a
    chain, so adjacent residues are correlated and an i.i.d. interval would be too narrow."""
    rng = np.random.default_rng(seed)
    m = len(d)
    nb = int(np.ceil(m / block))
    starts = rng.integers(0, max(m - block + 1, 1), size=(n, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n, -1)[:, :m] % m
    means = d[idx].mean(1)
    return {"mean": round(float(d.mean()), 5),
            "ci95": [round(float(np.quantile(means, 0.025)), 5),
                     round(float(np.quantile(means, 0.975)), 5)],
            "frac_residues_improved": round(float((d > 0).mean()), 4), "n_residues": m}


def folds(size: str) -> dict[tuple[str, str, str], Path]:
    """{(session, tag, arm): cif}. The leg kept every fold, not one per arm, so the A/A floor
    below is measured rather than assumed."""
    out = {}
    for sess, d in SESSIONS.items():
        for f in sorted(d.glob(f"{size}_*")):
            _size, tag, arm = f.name.split("_")[:3]
            out[(sess, tag, arm)] = f
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "native_lddt.json")
    ap.add_argument("--anchor-score", type=Path,
                    help="perf/k10_anchor/out/score.json from wk/k10-p1-accuracy-anchor")
    a = ap.parse_args()

    gt = ca_map(GT)
    rep = {"gt": GT.name, "arms": ARM, "sizes": {}}

    for size in ("298", "512"):
        F = folds(size)
        if not F:
            continue
        n = int(size)
        split = 298 if n > 298 else n
        sec = {"n_folds": len(F), "domains": list(GT_SEGMENTS[n])}

        # A/A control first: every fold of an arm must be byte-identical, or no arm number below
        # is readable. The leg measured 0.00000 A of run-to-run spread; this re-checks the bytes.
        sha = {k: hashlib.sha256(v.read_bytes()).hexdigest()[:16] for k, v in F.items()}
        per_arm = {arm: sorted({s for k, s in sha.items() if k[2] == arm}) for arm in ARM}
        sec["arm_bitexact"] = {ARM[k]: len(v) == 1 for k, v in per_arm.items()}
        sec["arm_sha256"] = {ARM[k]: v for k, v in per_arm.items()}
        assert all(sec["arm_bitexact"].values()), f"an arm is not self-consistent at {size}: {sha}"

        one = {arm: next(v for k, v in sorted(F.items()) if k[2] == arm) for arm in ARM}
        sec["scored_cif"] = {ARM[k]: str(v.relative_to(REPO)) for k, v in one.items()}

        # (1) the reading that decides it: distance to the experimental answer, per pseudo-domain.
        sec["native"] = {ARM[arm]: native(one[arm], n, gt) for arm in ARM}
        sec["deficit_on_minus_off"] = {
            dom: {m: round(sec["native"]["on"][dom][m] - sec["native"]["off"][dom][m], 5)
                  for m in ("lddt_ca", "ca_rmsd_A")}
            for dom in GT_SEGMENTS[n]}

        # (2) instrument control: reproduce the leg's arm-to-arm RMSD through a different parser
        # and superposition than rmsd.py's gemmi one. 1.32206 A / 0.24539 A is the known answer.
        S = {arm: load(one[arm], split) for arm in ARM}
        sec["arm_vs_arm"] = pair(S["1"], S["0"], split)

        # (3) the per-residue reading, because this leg has one seed and 294+210 residues. Sign
        # convention throughout: positive = the ladder is CLOSER to 1HCL than the shipped arm.
        pr = {arm: per_residue(one[arm], n, gt) for arm in ARM}
        sec["per_residue"] = {dom: block_bootstrap(pr["1"][dom][0] - pr["0"][dom][0])
                              for dom in GT_SEGMENTS[n]}

        # (4) negative control: the metric has to be able to see a move of the size this lever
        # makes. Displace the ladder arm's CAs by a random walk of the measured arm-to-arm CA
        # RMSD and confirm lDDT drops far past anything reported above.
        rng = np.random.default_rng(1234)
        sec["negative_control"] = {}
        for dom in GT_SEGMENTS[n]:
            _p, A, B = pr["1"][dom]
            step = float(sec["arm_vs_arm"]["whole_ca_A"])
            noise = rng.normal(0.0, step / np.sqrt(3.0), size=A.shape)
            sec["negative_control"][dom] = {
                "displacement_A": round(step, 5),
                "lddt_ca": round(lddt_per_residue(A + noise, B)[1], 5)}
        rep["sizes"][size] = sec

    if a.anchor_score and a.anchor_score.is_file():
        anc = json.loads(a.anchor_score.read_text())
        for size, sec in rep["sizes"].items():
            ns = anc["sizes"][size].get("native_summary", {})
            # The TT stack's own CA-lDDT spread across 4 seeds: the floor any lDDT move is read
            # against. And the arithmetic-only control: upstream fp32 vs its shipped bf16-mixed.
            sec["seed_floor_lddt_ca_ttmain_4seeds"] = {
                dom: ns.get(f"ttmain {dom} lddt_ca") for dom in sec["domains"]}
            sec["bf16_control_lddt_ca_gpuref_vs_bf16"] = {
                dom: round(ns[f"gpurefbf16 {dom} lddt_ca"]["mean"]
                           - ns[f"gpuref {dom} lddt_ca"]["mean"], 5)
                for dom in sec["domains"] if f"gpuref {dom} lddt_ca" in ns}
            # Cross-leg control. The OFF arm is the shipped stack at seed 0, which the anchor
            # already folded on its own card in its own session as `ttmain-s0`. If this leg's OFF
            # arm is that structure, the two legs are directly comparable and neither drifted.
            # The reference class this lever belongs to: a coordinate-moving lever read at ONE
            # seed. `ttbase` -> `ttmain` is the anchor's own such lever (device conditioning,
            # 0.49519 A of coordinates, 0.00001 lDDT averaged over 4 seeds). Its per-seed lDDT
            # deltas are the spread a single-seed lDDT reading of a lever carries.
            nat = anc["sizes"][size]["native"]
            sec["single_seed_lever_reference_ttmain_minus_ttbase"] = {
                dom: spread([round(nat[f"ttmain-s{k}"][dom]["lddt_ca"]
                                   - nat[f"ttbase-s{k}"][dom]["lddt_ca"], 5)
                             for k in range(4)])
                for dom in sec["domains"]}
            ref = anc["sizes"][size]["native"].get("ttmain-s0", {})
            sec["cross_leg_off_vs_anchor_ttmain_s0"] = {
                dom: {m: round(sec["native"]["off"][dom][m] - ref[dom][m], 6)
                      for m in ("lddt_ca", "ca_rmsd_A")}
                for dom in sec["domains"] if dom in ref}

    a.out.write_text(json.dumps(rep, indent=1) + "\n")

    for size, sec in rep["sizes"].items():
        print(f"\n=== {size} aa === folds {sec['n_folds']}  per-arm bit-exact "
              f"{sec['arm_bitexact']}")
        for dom in sec["domains"]:
            off, on = sec["native"]["off"][dom], sec["native"]["on"][dom]
            d = sec["deficit_on_minus_off"][dom]
            print(f"  {dom:20s} n_ca {off['n_ca']:4d}  CA-lDDT off {off['lddt_ca']:.5f}  "
                  f"on {on['lddt_ca']:.5f}  deficit {d['lddt_ca']:+.5f}")
            print(f"  {'':20s}            CA RMSD off {off['ca_rmsd_A']:.5f} A  "
                  f"on {on['ca_rmsd_A']:.5f} A  deficit {d['ca_rmsd_A']:+.5f} A")
            fl = sec.get("seed_floor_lddt_ca_ttmain_4seeds", {}).get(dom)
            if fl:
                print(f"  {'':20s}            seed floor (TT, 4 seeds) lDDT "
                      f"{fl['min']:.5f}-{fl['max']:.5f}, width {fl['max'] - fl['min']:.5f}"
                      f"  | bf16 control "
                      f"{sec['bf16_control_lddt_ca_gpuref_vs_bf16'][dom]:+.5f}")
            pr = sec["per_residue"][dom]
            print(f"  {'':20s}            per-residue delta {pr['mean']:+.5f} "
                  f"CI95 [{pr['ci95'][0]:+.5f}, {pr['ci95'][1]:+.5f}]  "
                  f"{pr['frac_residues_improved']:.1%} of {pr['n_residues']} residues improved")
            nc = sec["negative_control"][dom]
            print(f"  {'':20s}            neg control: displace {nc['displacement_A']:.3f} A "
                  f"-> lDDT {nc['lddt_ca']:.5f}")
        for dom, v in sec.get("single_seed_lever_reference_ttmain_minus_ttbase", {}).items():
            print(f"  reference class {dom:20s} a known 0.495 A lever, per-seed lDDT delta "
                  f"{v['min']:+.5f} to {v['max']:+.5f} (mean {v['mean']:+.5f}, n={v['n']})")
        xl = sec.get("cross_leg_off_vs_anchor_ttmain_s0")
        if xl:
            print(f"  cross-leg: off arm minus anchor ttmain-s0 = "
                  + "  ".join(f"{d} lDDT {v['lddt_ca']:+.6f}" for d, v in xl.items()))
        p = sec["arm_vs_arm"]
        print(f"  arm-to-arm (control)  all-atom {p['whole_all_atom_A']:.5f} A  "
              f"CA {p['whole_ca_A']:.5f} A  lDDT(on vs off) {p['lddt_ca']:.5f}")
    print("\nwrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
