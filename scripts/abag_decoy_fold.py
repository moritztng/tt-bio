"""Arm 3 (specificity frontier) decoy fold campaign: fold the 8 non-cognate
antibody/antigen pairs built by abag_decoy_build.py across all 3 models at 5 diffusion
samples, --write_pae on. No DockQ is computed (a decoy pair has no native complex to
score against by construction) -- only the model's own confidence outputs (ptm/iptm/
plddt) and the PAE dump, which is exactly what's being tested: does any trust signal
correctly assign LOW confidence to a pair that should not bind.

    nohup env TT_VISIBLE_DEVICES=<card> python3 scripts/abag_decoy_fold.py \
        --device <card> > /tmp/abag_decoys/campaign.log 2>&1 &
"""
import argparse, json, os, subprocess, sys, time

DECOYS = ["decoy_9ck4ab_9i5nag", "decoy_9i5nab_9m72ag", "decoy_9m72ab_22psag",
          "decoy_22psab_9obnag", "decoy_9obnab_9gfrag", "decoy_9gfrab_9udqag",
          "decoy_9udqab_9jkrag", "decoy_9jkrab_9ck4ag"]
MODELS = ["opendde-abag", "boltz2", "protenix-v2"]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_BASE = "/tmp/abag_decoys"
YAML_DIR = f"{ROOT}/examples/abag_pilot_decoys"
MSA_DIR = f"{OUT_BASE}/msa_cache"
PROGRESS = f"{OUT_BASE}/progress.jsonl"

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


def sample_cifs(struct_dir, tid):
    winner = f"{struct_dir}/{tid}.cif"
    files = [winner] if os.path.exists(winner) else []
    i = 1
    while os.path.exists(f"{struct_dir}/{tid}_model_{i}.cif"):
        files.append(f"{struct_dir}/{tid}_model_{i}.cif")
        i += 1
    return files


def fold_one(decoy_id, model, device):
    out_dir = f"{OUT_BASE}/{model.replace('-', '_')}"
    result_dir = f"{out_dir}/{RESULT_DIR_PREFIX[model]}_results_{decoy_id}"
    yaml = f"{YAML_DIR}/{decoy_id}.yaml"
    t0 = time.time()
    r = subprocess.run(
        [sys.executable, "-m", "tt_bio.main", "predict", yaml, "--model", model,
         "--out_dir", out_dir, "--diffusion_samples", "5", "--msa_dir", MSA_DIR,
         "--seed", "42", "--override", "--write_pae"],
        cwd=ROOT, capture_output=True, text=True,
        env={**os.environ, "TT_VISIBLE_DEVICES": str(device), "PYTHONPATH": ROOT},
    )
    wall_s = time.time() - t0
    rec = {"target": decoy_id, "model": model, "wall_s": round(wall_s, 1)}
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
    cifs = sample_cifs(struct_dir, decoy_id)
    rec["n_samples"] = len(cifs)
    rec["pae_npz"] = f"{struct_dir}/{decoy_id}_pae.npz"
    rec["winner_cif"] = cifs[0] if cifs else None
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default=",".join(DECOYS))
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--device", type=int, default=0)
    a = ap.parse_args()
    targets = a.targets.split(",")
    models = a.models.split(",")
    os.makedirs(OUT_BASE, exist_ok=True)
    skip = done_pairs()
    for decoy_id in targets:
        for model in models:
            if (decoy_id, model) in skip:
                print(f"[skip] {decoy_id} {model} already in progress.jsonl", flush=True)
                continue
            print(f"[start] {decoy_id} {model} {time.strftime('%H:%M:%S')}", flush=True)
            rec = fold_one(decoy_id, model, a.device)
            with open(PROGRESS, "a") as fp:
                fp.write(json.dumps(rec) + "\n")
            print(f"[done]  {decoy_id} {model} status={rec['status']} wall_s={rec.get('wall_s')} "
                  f"iptm={rec.get('confidence', {}).get('iptm')}", flush=True)
    print("CAMPAIGN COMPLETE", flush=True)


if __name__ == "__main__":
    main()
