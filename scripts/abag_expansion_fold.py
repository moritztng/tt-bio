"""N-expansion fold campaign: fold Tier-A verified 2026ARK-AB targets (beyond the
original Stage-3 N=12) across all 3 models at 5 diffusion samples, with --write_pae so
AntiConf/pDockQ2/ipSAE can be computed from the same folds (see
docs/implementation-parity-data/abag-n-expansion-candidates.json for provenance).

Parameterized so the same script runs unmodified on any host/card -- pass a target
subset per invocation to fan across cards. Appends one JSON line per (target, model)
to progress.jsonl so partial progress survives a restart.

Usage:
    nohup env TT_VISIBLE_DEVICES=<card> python3 scripts/abag_expansion_fold.py \
        --targets 9dsg,9fte,9j4c --device <card> \
        > /tmp/abag_expansion/campaign_<card>.log 2>&1 &
"""
import argparse, json, os, subprocess, sys, time

ALL_TARGETS = ["9dsg", "9fte", "9j4c", "9k6j", "9loe", "9lof", "9log", "9kwy",
               "21tw", "9lp1", "9jno", "9loz"]
MODELS = ["opendde-abag", "boltz2", "protenix-v2"]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_BASE = "/tmp/abag_expansion"
GT = f"{ROOT}/examples/ground_truth_structures"
YAML_DIR = f"{ROOT}/examples/abag_pilot_expansion"
MSA_DIR = f"{OUT_BASE}/msa_cache"
PROGRESS = f"{OUT_BASE}/progress.jsonl"
DOCKQ_PYTHON = os.path.expanduser("~/.opendde_dockq_venv/bin/python3")

RESULT_DIR_PREFIX = {"opendde-abag": "opendde", "boltz2": "boltz2", "protenix-v2": "protenix"}


def done_pairs():
    seen = set()
    if os.path.exists(PROGRESS):
        for line in open(PROGRESS):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                seen.add((r["target"], r["model"]))
            except Exception:
                pass
    return seen


def dockq(cif, native):
    r = subprocess.run([DOCKQ_PYTHON, "scripts/opendde_dockq.py", cif, native],
                        capture_output=True, text=True, cwd=ROOT)
    if r.returncode not in (0, 2):
        return {"error": f"rc={r.returncode}", "stderr": r.stderr[-300:]}
    try:
        d = json.loads(r.stdout)
        return {"global_dockq": d["global_dockq"], "n_interfaces": d["n_interfaces"],
                "interfaces": d["interfaces"]}
    except Exception as e:
        return {"error": str(e), "stdout": r.stdout[-300:]}


def sample_cifs(struct_dir, tid):
    winner = f"{struct_dir}/{tid}.cif"
    files = [winner] if os.path.exists(winner) else []
    i = 1
    while os.path.exists(f"{struct_dir}/{tid}_model_{i}.cif"):
        files.append(f"{struct_dir}/{tid}_model_{i}.cif")
        i += 1
    return files


def fold_one(target, model, device):
    tid = f"{target}_abag"
    out_dir = f"{OUT_BASE}/{model.replace('-', '_')}"
    result_dir = f"{out_dir}/{RESULT_DIR_PREFIX[model]}_results_{tid}"
    yaml = f"{YAML_DIR}/{target}_abag.yaml"
    native = f"{GT}/{target}.cif"
    t0 = time.time()
    r = subprocess.run(
        [sys.executable, "-m", "tt_bio.main", "predict", yaml, "--model", model,
         "--out_dir", out_dir, "--diffusion_samples", "5", "--msa_dir", MSA_DIR,
         "--seed", "42", "--override", "--write_pae"],
        cwd=ROOT, capture_output=True, text=True,
        env={**os.environ, "TT_VISIBLE_DEVICES": str(device), "PYTHONPATH": ROOT},
    )
    wall_s = time.time() - t0
    rec = {"target": target, "model": model, "wall_s": round(wall_s, 1)}
    rjson = f"{result_dir}/results.json"
    if r.returncode != 0 or not os.path.exists(rjson):
        rec["status"] = "fold_failed"
        rec["stderr"] = r.stderr[-2000:]
        return rec
    results = json.load(open(rjson))
    entry = results[0] if isinstance(results, list) else results
    rec["status"] = "ok"
    rec["confidence"] = {k: entry.get(k) for k in
                          ("confidence_score", "ptm", "iptm", "protein_iptm", "complex_plddt", "runtime_s")}
    rec["all_runs"] = entry.get("all_runs")
    struct_dir = f"{result_dir}/structures"
    cifs = sample_cifs(struct_dir, tid)
    rec["n_samples_scored"] = len(cifs)
    rec["sample_dockq"] = [dockq(c, native) for c in cifs]
    rec["pae_npz"] = f"{struct_dir}/{tid}_pae.npz"
    rec["winner_cif"] = cifs[0] if cifs else None
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default=",".join(ALL_TARGETS), help="comma-separated subset")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--device", type=int, default=0)
    a = ap.parse_args()
    targets = a.targets.split(",")
    models = a.models.split(",")
    os.makedirs(OUT_BASE, exist_ok=True)
    skip = done_pairs()
    for target in targets:
        for model in models:
            if (target, model) in skip:
                print(f"[skip] {target} {model} already in progress.jsonl", flush=True)
                continue
            print(f"[start] {target} {model} {time.strftime('%H:%M:%S')}", flush=True)
            rec = fold_one(target, model, a.device)
            with open(PROGRESS, "a") as fp:
                fp.write(json.dumps(rec) + "\n")
            top_dockq = None
            if rec.get("sample_dockq"):
                top_dockq = rec["sample_dockq"][0].get("global_dockq")
            print(f"[done]  {target} {model} status={rec['status']} wall_s={rec.get('wall_s')} "
                  f"top_dockq={top_dockq}", flush=True)
    print("CAMPAIGN COMPLETE", flush=True)


if __name__ == "__main__":
    main()
