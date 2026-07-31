"""Arm 3 (specificity frontier) decoy fold campaign: fold the 8 non-cognate
antibody/antigen pairs built by abag_decoy_build.py across all 3 models at 5 diffusion
samples, --write_pae on. No DockQ is computed (a decoy pair has no native complex to
score against by construction) -- only the model's own confidence outputs (ptm/iptm/
plddt) and the PAE dump, which is exactly what's being tested: does any trust signal
correctly assign LOW confidence to a pair that should not bind.

    nohup env TT_VISIBLE_DEVICES=<card> python3 scripts/abag_decoy_fold.py \
        --device <card> > /tmp/abag_decoys/campaign.log 2>&1 &
"""
import argparse, json, os, time

from abag_campaign_lib import (CONFIDENCE_KEYS, ROOT, RESULT_DIR_PREFIX,
                               done_pairs, run_predict, sample_cifs)

DECOYS = ["decoy_9ck4ab_9i5nag", "decoy_9i5nab_9m72ag", "decoy_9m72ab_22psag",
          "decoy_22psab_9obnag", "decoy_9obnab_9gfrag", "decoy_9gfrab_9udqag",
          "decoy_9udqab_9jkrag", "decoy_9jkrab_9ck4ag"]
MODELS = ["opendde-abag", "boltz2", "protenix-v2"]
OUT_BASE = "/tmp/abag_decoys"
YAML_DIR = f"{ROOT}/examples/abag_pilot_decoys"
MSA_DIR = f"{OUT_BASE}/msa_cache"
PROGRESS = f"{OUT_BASE}/progress.jsonl"


def fold_one(decoy_id, model, device):
    out_dir = f"{OUT_BASE}/{model.replace('-', '_')}"
    status, payload, wall_s = run_predict(
        decoy_id, f"{YAML_DIR}/{decoy_id}.yaml", model, out_dir, MSA_DIR,
        device, extra_args=("--write_pae",))
    rec = {"target": decoy_id, "model": model, "wall_s": wall_s}
    if status != "ok":
        rec["status"] = status
        rec["stderr"] = payload
        return rec
    rec["status"] = "ok"
    rec["confidence"] = {k: payload.get(k) for k in CONFIDENCE_KEYS}
    rec["all_runs"] = payload.get("all_runs")
    struct_dir = f"{out_dir}/{RESULT_DIR_PREFIX[model]}_results_{decoy_id}/structures"
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
    skip = done_pairs(PROGRESS)
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
