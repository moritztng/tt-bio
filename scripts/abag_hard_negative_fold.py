"""Hard-negative specificity fold campaign: 3 real, PDB-verified antibody Fabs
(REGN10933/casirivimab, LY-CoV555/bamlanivimab, S309/sotrovimab) vs WT SARS-CoV-2 RBD
and single-point escape mutants with well-documented experimental binding outcomes
(K417N, E484K). See docs/implementation-parity-data/abag-hard-negative-manifest.json
for provenance/rationale of each pair.

Usage:
    nohup env TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:flagship-abag-hard-negative-specificity \
        python3 scripts/abag_hard_negative_fold.py > /tmp/abag_hard_neg/campaign.log 2>&1 &
"""
import json, os, time

from abag_campaign_lib import (CONFIDENCE_KEYS, ROOT, RESULT_DIR_PREFIX,
                               done_pairs, run_predict, sample_cifs)

TARGETS = ["regn10933_wt", "regn10933_k417n", "lycov555_wt", "lycov555_e484k",
           "s309_wt", "s309_e484k", "s309_k417n"]
MODELS = ["opendde-abag"]
OUT_BASE = "/tmp/abag_hard_neg"
YAML_DIR = f"{ROOT}/examples/abag_hard_negative"
MSA_DIR = f"{OUT_BASE}/msa_cache"
PROGRESS = f"{OUT_BASE}/progress.jsonl"


def fold_one(target, model, device):
    out_dir = f"{OUT_BASE}/{model.replace('-', '_')}"
    status, payload, wall_s = run_predict(
        target, f"{YAML_DIR}/{target}.yaml", model, out_dir, MSA_DIR, device,
        extra_args=("--write_pae",))
    rec = {"target": target, "model": model, "wall_s": wall_s}
    if status != "ok":
        rec["status"] = status
        rec["stderr"] = payload
        return rec
    rec["status"] = "ok"
    rec["confidence"] = {k: payload.get(k) for k in CONFIDENCE_KEYS}
    struct_dir = f"{out_dir}/{RESULT_DIR_PREFIX[model]}_results_{target}/structures"
    cifs = sample_cifs(struct_dir, target)
    rec["n_samples_scored"] = len(cifs)
    rec["winner_cif"] = cifs[0] if cifs else None
    return rec


def main():
    os.makedirs(OUT_BASE, exist_ok=True)
    device = int(os.environ.get("TT_VISIBLE_DEVICES", "0"))
    skip = done_pairs(PROGRESS)
    for target in TARGETS:
        for model in MODELS:
            if (target, model) in skip:
                print(f"[skip] {target} {model} already ok", flush=True)
                continue
            print(f"[start] {target} {model} {time.strftime('%H:%M:%S')}", flush=True)
            rec = fold_one(target, model, device)
            with open(PROGRESS, "a") as fp:
                fp.write(json.dumps(rec) + "\n")
            print(f"[done]  {target} {model} status={rec['status']} wall_s={rec.get('wall_s')} "
                  f"iptm={rec.get('confidence', {}).get('iptm')}", flush=True)
    print("CAMPAIGN COMPLETE", flush=True)


if __name__ == "__main__":
    main()
