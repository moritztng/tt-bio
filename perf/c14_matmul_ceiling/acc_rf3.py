#!/usr/bin/env python3
"""Does `TT_BIO_MM_SHORT_M_BW` cost RF3 accuracy? Scored against the experimental answer.

RF3 reaches this lever: `rf3/token_dit.py` imports the shared `DiffusionTransformer`, whose
`ConditionedTransitionBlock._proj` is the flag's only call site, and a 117 aa RF3 fold serves
4,419 of 4,419 calls with the flag on (`fire2.json`). So the lever is not Boltz-2-exclusive and
owes a cross-model accuracy reading.

The metric is CA-lDDT against PDB 7ROA, not RMSD against the seed scatter. That choice is
measured, not stylistic: `state/b2z2-union-land.md:66` records a structural-RMSD-vs-seed-scatter
screen clearing `TT_BIO_UNFUSED_SILU` on Protenix-v2 (2.969 / 3.443 A moved against a
3.233 / 3.743 A floor) while CA-lDDT against the experimental structure separated the arms
completely. lDDT is superposition-free and local, so it cannot be saturated by a hinge or by
which basin the sampler drew.

Residue correspondence is aligned, not assumed. 7ROA resolves 115 residues numbered 3..134 while
the fold delivers 117 numbered 1..117: two Met are absent from the crystal, so matching by
`label_seq_id` pairs 98 residues of which only 58 agree in identity. This aligns the two
sequences (Needleman-Wunsch, identity scoring) and asserts every matched pair is identical.

    acc_rf3.py --out <json> --cifs <dir> --seeds 0,1,2,3 --arms base,bw
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "perf" / "other512"), str(REPO / "perf" / "fused_sdpa")]
from of3_score_ref import AA3, ca_map                                     # noqa: E402
from basin_lddt import lddt_per_residue                                   # noqa: E402

GT = REPO / "examples" / "ground_truth_structures" / "prot.cif"
DATA = REPO / "examples" / "prot.yaml"
PY = os.environ.get("C14_PY", "/home/ttuser/tt-bio-dev/env/bin/python3")
ARMS = {"base": 0, "bw": 1}
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306}


def seq_of(m: dict) -> tuple[str, list[int]]:
    ks = sorted(m)
    return "".join(AA3.get(m[k][0], "X") for k in ks), ks


def align(a: str, b: str) -> list[tuple[int, int]]:
    """Needleman-Wunsch on identity, gap -1. Returns matched index pairs (i in a, j in b)."""
    n, m = len(a), len(b)
    s = np.zeros((n + 1, m + 1))
    s[:, 0] = -np.arange(n + 1)
    s[0, :] = -np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            s[i, j] = max(s[i - 1, j - 1] + (1.0 if a[i - 1] == b[j - 1] else -1.0),
                          s[i - 1, j] - 1.0, s[i, j - 1] - 1.0)
    out, i, j = [], n, m
    while i > 0 and j > 0:
        d = s[i - 1, j - 1] + (1.0 if a[i - 1] == b[j - 1] else -1.0)
        if s[i, j] == d:
            out.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif s[i, j] == s[i - 1, j] - 1.0:
            i -= 1
        else:
            j -= 1
    return out[::-1]


def kabsch_rmsd(p: np.ndarray, q: np.ndarray) -> float:
    p = p - p.mean(0)
    q = q - q.mean(0)
    u, _, vt = np.linalg.svd(p.T @ q)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return float(np.sqrt((((p @ r.T) - q) ** 2).sum(1).mean()))


def fold(arm: str, seed: int, cifs: Path, msa_dir: Path, stats: Path, tag: str) -> dict:
    out = cifs / f"{tag}_out"
    if out.exists():
        shutil.rmtree(out)
    env = dict(os.environ)
    env.update({"TT_BIO_MM_SHORT_M_BW": str(ARMS[arm]),
                "C14_MM_STATS_OUT": str(stats.parent / f"{stats.name}.{tag}"),
                "PYTHONPATH": str(REPO / "perf" / "c14_matmul_ceiling" / "statshim")})
    cmd = [PY, "-m", "tt_bio.main", "predict", str(DATA), "--model", "rf3",
           "--seed", str(seed), "--out_dir", str(out), "--msa_dir", str(msa_dir),
           "--msa_cache_only"]
    t0 = time.perf_counter()
    p = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    cif = out / "rf3_results_prot" / "structures" / "prot.cif"
    if p.returncode != 0 or not cif.exists():
        print(p.stdout[-2000:], p.stderr[-2000:], flush=True)
        raise SystemExit(f"fold {tag} failed rc={p.returncode}")
    keep = cifs / f"{tag}.cif"
    shutil.copy(cif, keep)
    res = json.loads((out / "rf3_results_prot" / "results.json").read_text())[0]
    served = declined = 0
    for f in sorted(stats.parent.glob(f"{stats.name}.{tag}.*")):
        r = json.loads(f.read_text())
        for g in r["groups"]:
            served += g["served"]
            declined += g["calls"] - g["served"]
    shutil.rmtree(out)
    return {"arm": arm, "seed": seed, "tag": tag, "wall_s": round(dt, 2),
            "plddt": res["plddt"], "ptm": res["ptm"], "runtime_s": res["runtime_s"],
            "served": served, "declined": declined,
            "cif_sha": hashlib.sha256(keep.read_bytes()).hexdigest()[:16], "cif": str(keep)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cifs", type=Path, required=True)
    ap.add_argument("--seeds", default="0,1,2,3")
    ap.add_argument("--arms", default="base,bw")
    ap.add_argument("--msa-dir", type=Path, required=True)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    arms = a.arms.split(",")
    a.cifs.mkdir(parents=True, exist_ok=True)
    stats = a.cifs / "stats.json"

    runs = []
    for s in seeds:                      # interleaved, arm order reversed on odd seeds
        order = arms if s % 2 == 0 else arms[::-1]
        for arm in order:
            r = fold(arm, s, a.cifs, a.msa_dir, stats, f"{arm}_s{s}")
            print(f"  {r['tag']:>10}  plDDT {r['plddt']:.6f}  {r['runtime_s']:5.1f}s  "
                  f"served {r['served']}  declined {r['declined']}  {r['cif_sha']}", flush=True)
            runs.append(r)
    # A/A: the base arm at seed 0 again, so a zero here proves the pipeline is deterministic
    runs.append(fold("base", seeds[0], a.cifs, a.msa_dir, stats, f"base_s{seeds[0]}_aa"))
    print(f"  {runs[-1]['tag']}  plDDT {runs[-1]['plddt']:.6f}  {runs[-1]['cif_sha']}", flush=True)

    gt = ca_map(GT)
    gseq, gkeys = seq_of(gt)
    scored = {}
    for r in runs:
        pm = ca_map(Path(r["cif"]))
        pseq, pkeys = seq_of(pm)
        pairs = align(pseq, gseq)
        assert all(pseq[i] == gseq[j] for i, j in pairs), "alignment matched non-identical residues"
        p = np.array([pm[pkeys[i]][1] for i, _ in pairs])
        g = np.array([gt[gkeys[j]][1] for _, j in pairs])
        per, glob = lddt_per_residue(p, g)
        scored[r["tag"]] = {**r, "n_matched": len(pairs), "ca_lddt_gt": float(glob),
                            "ca_rmsd_gt": kabsch_rmsd(p, g), "per_res": [float(v) for v in per]}

    out = {"target": "7ROA (examples/prot.yaml)", "model": "rf3", "gt": str(GT),
           "seeds": seeds, "arms": arms, "host": os.uname().nodename,
           "card": os.environ.get("TT_VISIBLE_DEVICES"), "runs": scored}

    def col(arm, key):
        return [scored[f"{arm}_s{s}"][key] for s in seeds]

    print(f"\nCA-lDDT against 7ROA, {len(scored[f'{arms[0]}_s{seeds[0]}']['per_res'])} matched CA\n")
    for arm in arms:
        v = col(arm, "ca_lddt_gt")
        print(f"  {arm:>5}  " + "  ".join(f"s{s} {x:.6f}" for s, x in zip(seeds, v))
              + f"   mean {st.mean(v):.6f}  sd {st.stdev(v):.6f}")
    summary = {}
    if len(arms) == 2 and len(seeds) >= 2:
        b, t = col(arms[0], "ca_lddt_gt"), col(arms[1], "ca_lddt_gt")
        d = [y - x for x, y in zip(b, t)]
        md, sd = st.mean(d), st.stdev(d)
        ci = T95[len(d) - 1] * sd / len(d) ** 0.5
        sep = max(b) < min(t) or max(t) < min(b)
        summary = {"lddt_base_mean": st.mean(b), "lddt_base_sd": st.stdev(b),
                   "lddt_arm_mean": st.mean(t), "lddt_arm_sd": st.stdev(t),
                   "paired_delta_mean": md, "paired_delta_sd": sd, "paired_ci95": ci,
                   "resolved": abs(md) > ci, "rank_separated": bool(sep),
                   "base_seed_spread": max(b) - min(b),
                   "aa_lddt_delta": scored[f"base_s{seeds[0]}_aa"]["ca_lddt_gt"]
                   - scored[f"base_s{seeds[0]}"]["ca_lddt_gt"],
                   "aa_bit_exact": (scored[f"base_s{seeds[0]}_aa"]["cif_sha"]
                                    == scored[f"base_s{seeds[0]}"]["cif_sha"])}
        print(f"\n  paired delta (bw - base) {md:+.6f}  sd {sd:.6f}  95% CI +/-{ci:.6f}  "
              f"resolved {summary['resolved']}  rank-separated {sep}")
        print(f"  base across-seed spread {max(b) - min(b):.6f}   A/A delta "
              f"{summary['aa_lddt_delta']:+.6f}  bit-exact {summary['aa_bit_exact']}")
        for arm in arms:
            v = col(arm, "ca_rmsd_gt")
            print(f"  CA-RMSD vs GT {arm:>5}: " + "  ".join(f"{x:.4f}" for x in v))
        for arm in arms:
            v = col(arm, "plddt")
            print(f"  plDDT        {arm:>5}: " + "  ".join(f"{x:.6f}" for x in v))
    out["summary"] = summary
    a.out.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
