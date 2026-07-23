"""Stage 2 final analysis: combine Stage-1 OpenDDE ground truth with the Stage-2
Protenix-v2 + Boltz-2 fold campaign to compute the three trust signals and the
GO/NO-GO consensus-vs-native-ipTM decision metric.

Ground-truth label per target: OpenDDE Stage-1 top-pose global DockQ vs native
(re-scored from the Stage-1 CIFs recovered from qb2, DockQ==2.1.3 via
~/.opendde_dockq_venv) -- medium-correct iff DockQ >= 0.49.

Run on qb1 (CPU only, no device needed) once scripts/abag_pilot_stage2_fold.py's
progress.jsonl has all 24 (target, model) rows:
    ~/.opendde_dockq_venv/bin/python3 scripts/abag_pilot_stage2_analyze.py
"""
import json, subprocess, sys, itertools

ROOT = "/home/ttuser/.coworker/wt/flagship-abag-consensus-pilot-stage2"
GT = f"{ROOT}/examples/ground_truth_structures"
OUT_BASE = "/tmp/abag_stage2"
STAGE1_CIF_DIR = f"{OUT_BASE}/opendde_stage1_cifs"
PROGRESS = f"{OUT_BASE}/progress.jsonl"
TARGETS = ["9ck4", "9d3j", "9i5n", "9m72", "9obn", "22ps", "9yio", "9ncy", "9w14", "9gfr", "9udq", "9jkr"]
DOCKQ_PY = "scripts/opendde_dockq.py"  # run with cwd=ROOT
MEDIUM_BAR = 0.49
COVERAGE_BAR = 0.30
PRECISION_BAR = 0.85

def dockq_pair(cif_a, cif_b):
    r = subprocess.run([sys.executable, DOCKQ_PY, cif_a, cif_b], cwd=ROOT,
                        capture_output=True, text=True)
    if r.returncode not in (0, 2):
        return None
    try:
        return json.loads(r.stdout)["global_dockq"]
    except Exception:
        return None

def load_stage1():
    stage1 = {}
    for t in TARGETS:
        d = json.load(open(f"{STAGE1_CIF_DIR}/{t}_dockq.json"))
        conf = json.load(open(f"{ROOT}/docs/implementation-parity-data/abag-pilot-stage1/{t}_results.json"))
        c = conf[0] if isinstance(conf, list) else conf
        stage1[t] = {"cif": f"{STAGE1_CIF_DIR}/{t}_abag.cif", "dockq": d["global_dockq"],
                     "iptm": c.get("iptm"), "ptm": c.get("ptm")}
    return stage1

def load_stage2():
    rows = {}
    for line in open(PROGRESS):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        rows[(r["target"], r["model"])] = r
    return rows

def top_pose_info(rec):
    if rec is None or rec.get("status") != "ok" or not rec.get("sample_dockq"):
        return None
    sd = rec["sample_dockq"][0]
    conf = rec.get("confidence", {})
    return {"dockq": sd.get("global_dockq"), "iptm": conf.get("iptm"),
            "ptm": conf.get("ptm"), "confidence_score": conf.get("confidence_score")}

def intra_model_consistency(rec):
    """Mean pairwise DockQ among the (up to) 5 sampled poses -- a real, measured
    seed-to-seed agreement number, not a proxy. Reuses sample_dockq's already-scored
    per-sample vs-native DockQ is NOT what we want here; we need pose-vs-pose, so this
    recomputes pairwise DockQ between the sample CIFs directly."""
    if rec is None or rec.get("status") != "ok":
        return None
    n = rec.get("n_samples_scored", 0)
    if n < 2:
        return None
    model = rec["model"]
    prefix = {"boltz2": "boltz2", "protenix-v2": "protenix"}[model]
    target = rec["target"]
    tid = f"{target}_abag"
    struct_dir = f"{OUT_BASE}/{model.replace('-', '_')}/{prefix}_results_{tid}/structures"
    files = [f"{struct_dir}/{tid}.cif"] + [f"{struct_dir}/{tid}_model_{i}.cif" for i in range(1, n)]
    pairs = list(itertools.combinations(files, 2))
    vals = [v for v in (dockq_pair(a, b) for a, b in pairs) if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None

def main():
    stage1 = load_stage1()
    stage2 = load_stage2()
    missing = [(t, m) for t in TARGETS for m in ("boltz2", "protenix-v2") if (t, m) not in stage2]
    table = []
    for t in TARGETS:
        s1 = stage1[t]
        b2 = top_pose_info(stage2.get((t, "boltz2")))
        pv2 = top_pose_info(stage2.get((t, "protenix-v2")))
        label = 1 if s1["dockq"] >= MEDIUM_BAR else 0
        # cross-model agreement: pairwise top-pose DockQ among the 3 models that
        # actually folded (available pairs only -- missing models just shrink n_pairs)
        cifs = {"opendde": s1["cif"]}
        b2_rec, pv2_rec = stage2.get((t, "boltz2")), stage2.get((t, "protenix-v2"))
        if b2 is not None:
            cifs["boltz2"] = f"{OUT_BASE}/boltz2/boltz2_results_{t}_abag/structures/{t}_abag.cif"
        if pv2 is not None:
            cifs["protenix-v2"] = f"{OUT_BASE}/protenix_v2/protenix_results_{t}_abag/structures/{t}_abag.cif"
        pair_dockqs = {}
        for (na, ca), (nb, cb) in itertools.combinations(cifs.items(), 2):
            pair_dockqs[f"{na}-{nb}"] = dockq_pair(ca, cb)
        vals = [v for v in pair_dockqs.values() if v is not None]
        consensus = round(sum(vals) / len(vals), 4) if vals else None
        table.append({
            "target": t, "label_medium_correct": label, "opendde_dockq": round(s1["dockq"], 4),
            "opendde_iptm": s1["iptm"],
            "boltz2_dockq": b2["dockq"] if b2 else None, "boltz2_iptm": b2["iptm"] if b2 else None,
            "protenix_v2_dockq": pv2["dockq"] if pv2 else None, "protenix_v2_iptm": pv2["iptm"] if pv2 else None,
            "pair_dockqs": pair_dockqs, "consensus_score": consensus,
            "boltz2_seed_consistency": intra_model_consistency(b2_rec),
            "protenix_v2_seed_consistency": intra_model_consistency(pv2_rec),
        })

    n = len(table)
    k = max(1, round(COVERAGE_BAR * n))  # top-k for >=30% coverage at N=12 -> k=4
    by_consensus = sorted([r for r in table if r["consensus_score"] is not None],
                           key=lambda r: -r["consensus_score"])[:k]
    by_native_iptm = sorted([r for r in table if r["opendde_iptm"] is not None],
                            key=lambda r: -r["opendde_iptm"])[:k]
    prec_consensus = sum(r["label_medium_correct"] for r in by_consensus) / len(by_consensus) if by_consensus else 0
    prec_native = sum(r["label_medium_correct"] for r in by_native_iptm) / len(by_native_iptm) if by_native_iptm else 0
    cov_consensus = len(by_consensus) / n
    cov_native = len(by_native_iptm) / n

    go = (prec_consensus >= PRECISION_BAR and cov_consensus >= COVERAGE_BAR
          and prec_consensus > prec_native)
    verdict = "GO" if go else "NO-GO"

    out = {"n_targets": n, "k_tier": k, "table": table,
           "precision@consensus-tier": round(prec_consensus, 4),
           "coverage@consensus-tier": round(cov_consensus, 4),
           "precision@native-ipTM-tier": round(prec_native, 4),
           "coverage@native-ipTM-tier": round(cov_native, 4),
           "consensus_tier_targets": [r["target"] for r in by_consensus],
           "native_iptm_tier_targets": [r["target"] for r in by_native_iptm],
           "verdict": verdict, "missing_folds": missing}
    json.dump(out, open(f"{OUT_BASE}/stage2_final.json", "w"), indent=2)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
