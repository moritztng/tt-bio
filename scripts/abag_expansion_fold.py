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
import argparse, json, os, time

from abag_campaign_lib import (CONFIDENCE_KEYS, ROOT, RESULT_DIR_PREFIX,
                               dockq, done_pairs, run_predict, sample_cifs)

ALL_TARGETS = ["9dsg", "9fte", "9j4c", "9k6j", "9loe", "9lof", "9log", "9kwy",
               "21tw", "9lp1", "9jno", "9loz"]
MODELS = ["opendde-abag", "boltz2", "protenix-v2"]
OUT_BASE = "/tmp/abag_expansion"
GT = f"{ROOT}/examples/ground_truth_structures"
YAML_DIR = f"{ROOT}/examples/abag_pilot_expansion"
MSA_DIR = f"{OUT_BASE}/msa_cache"
PROGRESS = f"{OUT_BASE}/progress.jsonl"


def fold_one(target, model, device):
    tid = f"{target}_abag"
    out_dir = f"{OUT_BASE}/{model.replace('-', '_')}"
    status, payload, wall_s = run_predict(
        tid, f"{YAML_DIR}/{target}_abag.yaml", model, out_dir, MSA_DIR,
        device, extra_args=("--write_pae",))
    rec = {"target": target, "model": model, "wall_s": wall_s}
    if status != "ok":
        rec["status"] = status
        rec["stderr"] = payload
        return rec
    rec["status"] = "ok"
    rec["confidence"] = {k: payload.get(k) for k in CONFIDENCE_KEYS}
    rec["all_runs"] = payload.get("all_runs")
    struct_dir = f"{out_dir}/{RESULT_DIR_PREFIX[model]}_results_{tid}/structures"
    cifs = sample_cifs(struct_dir, tid)
    rec["n_samples_scored"] = len(cifs)
    rec["sample_dockq"] = [dockq(c, f"{GT}/{target}.cif") for c in cifs]
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
    skip = done_pairs(PROGRESS)
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
