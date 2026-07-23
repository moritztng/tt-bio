"""Stage 3 final analysis: fold in OpenDDE-abag's OWN 5-sample self-consistency (the
missing piece from Stage 1/2) and report precision@coverage for every trust signal at
FOUR coverage points (25/33/50/67%), not the single 33% point that produced an
artificial tie in the Stage 2 pilot.

Ground-truth label per target is refreshed to use OpenDDE's best-of-5 top-pose DockQ
(Stage 3), a more robust estimate than Stage 1's single-seed label.

Known limitation (documented, not silently dropped): cross-model consensus_score's
opendde-vs-{boltz2,protenix-v2} pairwise components still use OpenDDE's OLD Stage-1
single-seed CIF (the Stage-2 host's ephemeral /tmp CIFs for boltz2/protenix-v2 no
longer exist and the host was unreachable this pass) -- the boltz2-vs-protenix-v2
pairwise component and all seed-consistency / native-ipTM numbers are fully fresh.

Run on pc (CPU only):
    /home/moritz/.opendde_dockq_venv/bin/python3 scripts/abag_pilot_stage3_analyze.py
"""
import json, itertools

ROOT = "/home/moritz/.coworker/wt/flagship-abag-trust-signal-rethink"
STAGE3_BASE = "/tmp/abag_stage3"
STAGE3_PROGRESS = f"{STAGE3_BASE}/progress.jsonl"
STAGE2_FINAL = f"{ROOT}/docs/implementation-parity-data/abag-pilot-stage2-final.json"
TARGETS = ["9ck4", "9d3j", "9i5n", "9m72", "9obn", "22ps", "9yio", "9ncy", "9w14", "9gfr", "9udq", "9jkr"]
MEDIUM_BAR = 0.49
COVERAGE_POINTS = [0.25, 0.33, 0.50, 0.67]

def load_stage2():
    d = json.load(open(STAGE2_FINAL))
    return {row["target"]: row for row in d["table"]}

def load_stage3():
    rows = {}
    for line in open(STAGE3_PROGRESS):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        rows[r["target"]] = r
    return rows

def opendde_top_pose(rec):
    if rec is None or rec.get("status") != "ok" or not rec.get("sample_dockq"):
        return None, None
    sd = rec["sample_dockq"][0]
    conf = rec.get("confidence", {})
    return sd.get("global_dockq"), conf.get("iptm")

def opendde_seed_consistency(rec, target):
    if rec is None or rec.get("status") != "ok":
        return None
    n = rec.get("n_samples_scored", 0)
    if n < 2:
        return None
    tid = f"{target}_abag"
    struct_dir = f"{STAGE3_BASE}/opendde_abag/opendde_results_{tid}/structures"
    files = [f"{struct_dir}/{tid}.cif"] + [f"{struct_dir}/{tid}_model_{i}.cif" for i in range(1, n)]
    import subprocess, sys, os
    dockq_py = os.path.expanduser("~/.opendde_dockq_venv/bin/python3")
    pairs = list(itertools.combinations(files, 2))
    vals = []
    for a, b in pairs:
        r = subprocess.run([dockq_py, "scripts/opendde_dockq.py", a, b], cwd=ROOT,
                            capture_output=True, text=True)
        if r.returncode not in (0, 2):
            continue
        try:
            vals.append(json.loads(r.stdout)["global_dockq"])
        except Exception:
            pass
    return round(sum(vals) / len(vals), 4) if vals else None

def precision_at_coverage(table, signal_key, n, coverage_points):
    ranked = sorted([row for row in table if row.get(signal_key) is not None],
                     key=lambda r: r[signal_key], reverse=True)
    out = {}
    for c in coverage_points:
        k = max(1, round(c * n))
        tier = ranked[:k]
        if not tier:
            out[c] = None
            continue
        prec = sum(1 for r in tier if r["label_medium_correct"]) / len(tier)
        out[c] = {"k": k, "precision": round(prec, 3),
                   "targets": [r["target"] for r in tier]}
    return out

def main():
    stage2 = load_stage2()
    stage3 = load_stage3()
    table = []
    missing = []
    for t in TARGETS:
        s2 = stage2.get(t)
        s3 = stage3.get(t)
        if s2 is None:
            missing.append(t)
            continue
        opendde_dockq, opendde_iptm = opendde_top_pose(s3)
        if opendde_dockq is None:
            missing.append(t)
            continue
        opendde_sc = opendde_seed_consistency(s3, t)
        label = 1 if opendde_dockq >= MEDIUM_BAR else 0
        pair = s2["pair_dockqs"]
        # consensus: boltz2-protenix-v2 pair is fresh; opendde pairs reuse Stage-1
        # 1-seed OpenDDE CIF comparisons (documented limitation, see module docstring)
        consensus = round((pair["opendde-boltz2"] + pair["opendde-protenix-v2"] +
                            pair["boltz2-protenix-v2"]) / 3, 4)
        row = {
            "target": t,
            "label_medium_correct": label,
            "opendde_dockq_5sample": opendde_dockq,
            "opendde_iptm": opendde_iptm,
            "opendde_seed_consistency": opendde_sc,
            "boltz2_dockq": s2["boltz2_dockq"], "boltz2_iptm": s2["boltz2_iptm"],
            "protenix_v2_dockq": s2["protenix_v2_dockq"], "protenix_v2_iptm": s2["protenix_v2_iptm"],
            "boltz2_seed_consistency": s2["boltz2_seed_consistency"],
            "protenix_v2_seed_consistency": s2["protenix_v2_seed_consistency"],
            "consensus_score_stale_opendde_pair": consensus,
        }
        table.append(row)

    signals = ["opendde_iptm", "boltz2_iptm", "protenix_v2_iptm",
               "consensus_score_stale_opendde_pair",
               "opendde_seed_consistency", "boltz2_seed_consistency",
               "protenix_v2_seed_consistency"]
    n = len(table)
    n_correct = sum(r["label_medium_correct"] for r in table)
    result = {
        "n_targets": n, "n_medium_correct": n_correct, "missing_targets": missing,
        "table": table,
        "precision_by_signal_and_coverage": {
            s: precision_at_coverage(table, s, n, COVERAGE_POINTS) for s in signals
        },
    }
    out_path = f"{ROOT}/docs/implementation-parity-data/abag-pilot-stage3-final.json"
    json.dump(result, open(out_path, "w"), indent=2)
    print(f"wrote {out_path}")
    print(f"n_targets={n} n_medium_correct={n_correct} missing={missing}")
    for s in signals:
        print(f"\n=== {s} ===")
        for c in COVERAGE_POINTS:
            v = result["precision_by_signal_and_coverage"][s][c]
            if v is None:
                print(f"  cov={c}: no data")
            else:
                print(f"  cov={c} (top {v['k']}/{n}): precision={v['precision']} targets={v['targets']}")

if __name__ == "__main__":
    main()
