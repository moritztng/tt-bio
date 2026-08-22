#!/usr/bin/env python3
"""Controls and corroborators for the OpenFold3 fused-SDPA anchors.

The verdict is carried by the distogram Spearman rho in disto_score.py --anchor. Nothing here can
set it. This file does two other jobs, both pre-registered in state/fused-sdpa-adopt_PLAN3.md §3:

CONTROLS, which must pass before any margin is read. A failed control voids the anchor.
  1. A/A determinism -- seeds are 0,1,2,3,4,0 and the repeated seed-0 fold must reproduce the
     first one's CIF sha256 AND its distogram sha256, per arm. f5 is a control and is excluded
     from every estimate: state/fused-sdpa-adopt.md §0 records including a repeated seed turning a
     -0.00059 lDDT margin into -0.00791 with a CI excluding zero.
  2. The lever served -- every fused fold has served > 0, declined == 0, too_short == 0, and every
     shipped fold has served == 0. The counter is cumulative in a process, so read per-fold deltas.
  3. The seed is live -- seeds 1-4 move the CIF sha. On a one-row-MSA fixture the trunk may be
     seed-invariant; that is a finding to report, not a failure.

CORROBORATORS, which corroborate a structural number or count for nothing.
  1a8q  mean CA Kabsch RMSD to 1A8Q over all 274 residues, against upstream's < 2.0 A bound and
        our own committed 0.23-0.26 A range. This is also the anchor's acceptance check: outside
        that band the template is not being consumed and the anchor is not the one
        docs/openfold3-upstream-suite.md measured.
  hsa   CA-lDDT to 1AO6 chain A (Mariani et al., Bioinformatics 29(21):2722, 15 A radius,
        0.5/1/2/4 A thresholds), reusing basin_lddt.lddt_per_residue unchanged. HSA is three
        domains on flexible hinges, so global RMSD is reported too but labelled hinge-sensitive.
  9bk6  plDDT / pTM / ipTM against the OF3 CPU reference's own five-seed spread.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "perf" / "other512"))
sys.path.insert(0, str(HERE))
from cif_rmsd import kabsch_rmsd                       # noqa: E402
from of3_score_ref import ca_map_chains                # noqa: E402
from basin_lddt import lddt_per_residue                # noqa: E402
from disto_score import ANCHORS, ARMS                  # noqa: E402

# The OF3 CPU reference's own five-seed spread on 9bk6, read off the committed
# ref-fixtures seed*/results.json. A device delta smaller than this says nothing.
REF_9BK6 = {"plddt_spread": 2.616, "ptm_spread": 0.03086, "iptm_spread": 0.03539}


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", choices=sorted(ANCHORS), required=True)
    ap.add_argument("--dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    an = ANCHORS[a.anchor]
    d = a.dir or HERE / "anchor" / an["fixture"]
    out_path = a.out or HERE / f"corr_of3_{a.anchor}.json"

    gt = ca_map_chains(HERE / "cifs" / an["gt"])
    rep: dict = {"anchor": a.anchor, "gt": an["gt"], "controls": {}, "arms": {}}
    ok = True

    for arm in ARMS:
        fj = json.loads((d / arm / "fold.json").read_text())
        rows = fj["folds"]
        served = [r["triatt_fused_hifi_stats"]["served"] for r in rows]
        decl = [r["triatt_fused_hifi_stats"]["declined"] for r in rows]
        short = [r["triatt_fused_hifi_stats"]["too_short"] for r in rows]
        per_fold = [served[0]] + [served[i] - served[i - 1] for i in range(1, len(served))]
        shas = [list(r["cif_sha256"].values())[0] for r in rows]
        dsha = [sha_file(d / arm / f"f{i}_seed{r['seed']}" / "distogram.npy")
                for i, r in enumerate(rows)]

        # control 1: the A/A repeat
        aa_cif = shas[0] == shas[-1]
        aa_dis = dsha[0] == dsha[-1]
        # control 2: the lever
        want_served = arm == "hifi"
        lever = (all(v > 0 for v in per_fold) if want_served else all(v == 0 for v in served)) \
            and max(decl) == 0 and max(short) == 0
        # control 3: the seed is live
        live = len(set(shas[:5])) > 1

        rep["controls"][arm] = {
            "aa_cif_identical": bool(aa_cif), "aa_distogram_identical": bool(aa_dis),
            "cif_sha": shas, "distogram_sha": dsha,
            "triatt_served_per_fold": per_fold, "declined": max(decl), "too_short": max(short),
            "lever_ok": bool(lever), "seed_live": bool(live),
            "env_flags": fj["env_flags"], "n_msa": fj["n_msa"],
            "recycling_steps": fj["recycling_steps"], "sampling_steps": fj["sampling_steps"],
        }
        ok &= aa_cif and aa_dis and lever

        # corroborators, over the five SCORED seeds only -- f5 is the control and is dropped
        scored = rows[:5]
        block: dict = {"plddt": [r["plddt"] for r in scored], "ptm": [r["ptm"] for r in scored],
                       "fold_s": [r["fold_s"] for r in scored]}
        if scored[0].get("iptm") is not None:
            block["iptm"] = [r["iptm"] for r in scored]

        rmsd, lddt = [], []
        for i, r in enumerate(scored):
            cif = next((d / arm / f"f{i}_seed{r['seed']}").glob("*.cif"))
            fm = ca_map_chains(cif)
            seen = sorted({k[0] for k in fm})
            remap = dict(zip(seen, list(an["chains"])))
            fm = {(remap[c], j): v for (c, j), v in fm.items()}
            # score chain A only: 1AO6 chain B is a second lattice copy, and 9bk6's B is scored
            # by the distogram segments rather than by a global superposition
            keys = [k for k in an["segments"]["A"][1] if k in fm and k in gt]
            bad = [k for k in keys if fm[k][0] != gt[k][0]]
            assert not bad, f"{arm} f{i}: residue identity mismatch vs {an['gt']}: {bad[:5]}"
            Pm = np.array([fm[k][1] for k in keys])
            Q = np.array([gt[k][1] for k in keys])
            rmsd.append(kabsch_rmsd(Pm, Q))
            lddt.append(lddt_per_residue(Pm, Q)[1])
        block["n_scored_ca"] = len(keys)
        block["ca_rmsd_A"] = [round(v, 4) for v in rmsd]
        block["ca_rmsd_mean_A"] = round(float(np.mean(rmsd)), 4)
        block["ca_lddt"] = [round(v, 5) for v in lddt]
        block["ca_lddt_mean"] = round(float(np.mean(lddt)), 5)
        rep["arms"][arm] = block

    da = rep["arms"]["def"]
    ha = rep["arms"]["hifi"]
    rep["margins"] = {
        "ca_rmsd_A": round(ha["ca_rmsd_mean_A"] - da["ca_rmsd_mean_A"], 4),
        "ca_lddt": round(ha["ca_lddt_mean"] - da["ca_lddt_mean"], 5),
        "plddt": round(float(np.mean(ha["plddt"]) - np.mean(da["plddt"])), 5),
        "ptm": round(float(np.mean(ha["ptm"]) - np.mean(da["ptm"])), 5),
        "shipped_rmsd_seed_range_A": round(max(da["ca_rmsd_A"]) - min(da["ca_rmsd_A"]), 4),
    }

    if a.anchor == "1a8q_274":
        # the anchor acceptance check from PLAN3 §4 Step 3
        m = da["ca_rmsd_mean_A"]
        rep["anchor_acceptance"] = {
            "band_A": [0.23, 0.26], "shipped_mean_A": m, "pass": bool(0.23 <= m <= 0.26),
            "note": "outside the band the self-template is not being consumed and this is not "
                    "the anchor docs/openfold3-upstream-suite.md measured",
        }
        rep["upstream_bound_A"] = 2.0
    if a.anchor == "9bk6_164":
        rep["reference_own_spread"] = REF_9BK6

    rep["controls_pass"] = bool(ok)
    out_path.write_text(json.dumps(rep, indent=1) + "\n")

    print(f"=== {a.anchor} ===  controls {'PASS' if ok else 'FAIL'}")
    for arm in ARMS:
        c = rep["controls"][arm]
        print(f"  [{arm}] A/A cif {c['aa_cif_identical']} distogram {c['aa_distogram_identical']}"
              f"  served/fold {c['triatt_served_per_fold']} declined {c['declined']} "
              f"too_short {c['too_short']}  seed_live {c['seed_live']}")
    for arm in ARMS:
        b = rep["arms"][arm]
        print(f"  [{arm}] over {b['n_scored_ca']} CA: RMSD {b['ca_rmsd_A']} "
              f"mean {b['ca_rmsd_mean_A']:.4f} A   lDDT mean {b['ca_lddt_mean']:.5f}   "
              f"plDDT {np.round(b['plddt'],4).tolist()}")
    print(f"  margins (fused - shipped): {rep['margins']}")
    if "anchor_acceptance" in rep:
        print(f"  anchor acceptance: {rep['anchor_acceptance']}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
