"""Arm 1 + Arm 2 analysis: combine the original Stage-3 N=12 abag targets with the
new Tier-A N-expansion targets (up to 12 more, /tmp/abag_expansion/progress.jsonl) into
one N<=24 table, compute seed self-consistency for the new targets (DockQ isn't
enough -- need pairwise sample agreement, same recipe as Stage 3), compute the
literature-baseline signals (AntiConf/pDockQ2/ipSAE) for every target that has a
--write_pae dump (all new-N-expansion folds; the original 12 predate --write_pae and
are NOT included in the baseline comparison -- flagged, not silently averaged in),
and report AUC for every signal at whatever N is available when this is run (partial
runs are fine -- this script is designed to be re-run as the campaign progresses).

Run on pc (needs the opendde_dockq venv for DockQ + gemmi for pDockQ2/ipSAE):
    /home/moritz/.opendde_dockq_venv/bin/python3 scripts/abag_expansion_analyze.py
"""
import itertools, json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
STAGE3_FINAL = f"{ROOT}/docs/implementation-parity-data/abag-pilot-stage3-final.json"
EXP_BASE = "/tmp/abag_expansion"
EXP_PROGRESS = f"{EXP_BASE}/progress.jsonl"
EXP_YAML_DIR = f"{ROOT}/examples/abag_pilot_expansion"
MEDIUM_BAR = 0.49
DOCKQ_PYTHON = os.path.expanduser("~/.opendde_dockq_venv/bin/python3")

MODEL_KEYS = {"opendde-abag": "opendde", "boltz2": "boltz2", "protenix-v2": "protenix_v2"}
RESULT_DIR_PREFIX = {"opendde-abag": "opendde", "boltz2": "boltz2", "protenix-v2": "protenix"}


def load_original_12():
    d = json.load(open(STAGE3_FINAL))
    return {row["target"]: row for row in d["table"]}


def load_expansion_progress():
    rows = {}  # (target, model) -> record
    if not os.path.exists(EXP_PROGRESS):
        return rows
    for line in open(EXP_PROGRESS):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        rows[(r["target"], r["model"])] = r
    return rows


def seed_consistency(rec, target, model):
    if rec is None or rec.get("status") != "ok":
        return None
    n = rec.get("n_samples_scored", 0)
    if n < 2:
        return None
    tid = f"{target}_abag"
    out_dir = f"{EXP_BASE}/{model.replace('-', '_')}"
    struct_dir = f"{out_dir}/{RESULT_DIR_PREFIX[model]}_results_{tid}/structures"
    files = [f"{struct_dir}/{tid}.cif"] + [f"{struct_dir}/{tid}_model_{i}.cif" for i in range(1, n)]
    files = [f for f in files if os.path.exists(f)]
    if len(files) < 2:
        return None
    vals = []
    for a, b in itertools.combinations(files, 2):
        r = subprocess.run([DOCKQ_PYTHON, "scripts/opendde_dockq.py", a, b], cwd=ROOT,
                            capture_output=True, text=True)
        if r.returncode not in (0, 2):
            continue
        try:
            vals.append(json.loads(r.stdout)["global_dockq"])
        except Exception:
            pass
    return round(sum(vals) / len(vals), 4) if vals else None


def baseline_metrics(rec, target, model):
    """AntiConf/pDockQ2/ipSAE from the --write_pae dump, if present."""
    if rec is None or rec.get("status") != "ok":
        return None
    cif = rec.get("winner_cif")
    pae_npz = rec.get("pae_npz")
    if not cif or not os.path.exists(cif) or not pae_npz or not os.path.exists(pae_npz):
        return None
    yaml_path = f"{EXP_YAML_DIR}/{target}_abag.yaml"
    ptm = rec.get("confidence", {}).get("ptm")
    if ptm is None:
        return None
    from abag_pae_metrics import compute_metrics
    try:
        return compute_metrics(cif, pae_npz, yaml_path, ptm)
    except Exception as e:
        return {"error": str(e)}


def build_expansion_rows():
    prog = load_expansion_progress()
    rows = {}
    targets = sorted({t for t, _m in prog.keys()})
    for target in targets:
        row = {"target": target, "source": "expansion"}
        opendde_rec = prog.get((target, "opendde-abag"))
        if opendde_rec is None or opendde_rec.get("status") != "ok" or not opendde_rec.get("sample_dockq"):
            continue  # need OpenDDE's own top-pose DockQ for the correctness label
        top_dockq = opendde_rec["sample_dockq"][0].get("global_dockq")
        if top_dockq is None:
            continue
        row["opendde_dockq_5sample"] = top_dockq
        row["label_medium_correct"] = int(top_dockq >= MEDIUM_BAR)
        row["opendde_iptm"] = opendde_rec.get("confidence", {}).get("iptm")
        row["opendde_seed_consistency"] = seed_consistency(opendde_rec, target, "opendde-abag")
        bm = baseline_metrics(opendde_rec, target, "opendde-abag")
        if bm and "error" not in bm:
            row["opendde_pdockq2"] = bm["pdockq2"]
            row["opendde_ipsae"] = bm["ipsae"]
            row["opendde_anticonf"] = bm["anticonf"]

        for model, key in [("boltz2", "boltz2"), ("protenix-v2", "protenix_v2")]:
            rec = prog.get((target, model))
            if rec is None or rec.get("status") != "ok":
                continue
            row[f"{key}_dockq"] = rec.get("sample_dockq", [{}])[0].get("global_dockq") if rec.get("sample_dockq") else None
            row[f"{key}_iptm"] = rec.get("confidence", {}).get("iptm")
            row[f"{key}_seed_consistency"] = seed_consistency(rec, target, model)
            bm = baseline_metrics(rec, target, model)
            if bm and "error" not in bm:
                row[f"{key}_pdockq2"] = bm["pdockq2"]
                row[f"{key}_ipsae"] = bm["ipsae"]
                row[f"{key}_anticonf"] = bm["anticonf"]
        rows[target] = row
    return rows


def auc(table, signal_key):
    correct = [r[signal_key] for r in table if r["label_medium_correct"] == 1 and r.get(signal_key) is not None]
    incorrect = [r[signal_key] for r in table if r["label_medium_correct"] == 0 and r.get(signal_key) is not None]
    if not correct or not incorrect:
        return None
    pairs = len(correct) * len(incorrect)
    wins = sum(1.0 if c > w else 0.5 if c == w else 0.0 for c in correct for w in incorrect)
    return round(wins / pairs, 4)


def main():
    original = load_original_12()
    expansion = build_expansion_rows()
    print(f"original targets loaded: {len(original)}")
    print(f"expansion targets fully scored so far: {len(expansion)} / 12 "
          f"({sorted(expansion.keys())})")

    combined = list(original.values()) + list(expansion.values())
    n = len(combined)
    n_correct = sum(r["label_medium_correct"] for r in combined)
    print(f"\ncombined N = {n} ({n_correct} medium-correct, {n - n_correct} incorrect)")

    print("\n=== AUC across combined N (Arm 1: does it survive past N=12?) ===")
    for label, key in [("opendde native ipTM", "opendde_iptm"),
                       ("opendde seed self-consistency", "opendde_seed_consistency"),
                       ("boltz2 native ipTM", "boltz2_iptm"),
                       ("boltz2 seed self-consistency", "boltz2_seed_consistency"),
                       ("protenix-v2 native ipTM", "protenix_v2_iptm"),
                       ("protenix-v2 seed self-consistency", "protenix_v2_seed_consistency")]:
        n_avail = sum(1 for r in combined if r.get(key) is not None)
        print(f"  {label:38s} AUC={auc(combined, key)}  (n_available={n_avail}/{n})")

    exp_only = list(expansion.values())
    n_exp = len(exp_only)
    print(f"\n=== Arm 2: literature-baseline head-to-head, N-expansion subset only (N={n_exp}, "
          "needs --write_pae, not available for the original 12) ===")
    for label, key in [("protenix-v2 native ipTM", "protenix_v2_iptm"),
                       ("opendde native ipTM", "opendde_iptm"),
                       ("opendde AntiConf", "opendde_anticonf"),
                       ("opendde pDockQ2", "opendde_pdockq2"),
                       ("opendde ipSAE", "opendde_ipsae"),
                       ("protenix-v2 AntiConf", "protenix_v2_anticonf"),
                       ("protenix-v2 pDockQ2", "protenix_v2_pdockq2"),
                       ("protenix-v2 ipSAE", "protenix_v2_ipsae")]:
        n_avail = sum(1 for r in exp_only if r.get(key) is not None)
        print(f"  {label:38s} AUC={auc(exp_only, key)}  (n_available={n_avail}/{n_exp})")

    out = {"n_original": len(original), "n_expansion_scored": len(expansion),
           "n_combined": n, "n_correct": n_correct,
           "combined_table": combined, "expansion_table": exp_only}
    out_path = f"{ROOT}/docs/implementation-parity-data/abag-pilot-expansion-progress.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
