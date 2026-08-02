"""Stage 3 fold campaign: OpenDDE-abag, 5 diffusion samples, on the 12 verified
2026ARK-AB abag targets -- the missing piece from Stage 1 (which only folded 1 seed).
Gives OpenDDE its own measured seed self-consistency, comparable to the Protenix-v2 /
Boltz-2 numbers already in docs/implementation-parity-data/abag-pilot-stage2-final.json.

Scores every sample (not just the winner) against the native CIF. Appends one JSON
line per target to progress.jsonl so partial progress survives a restart.

Usage:
    nohup env TT_VISIBLE_DEVICES=<card> python3 scripts/abag_pilot_stage3_opendde_fold.py \
        --device <card> > /tmp/abag_stage3/campaign.log 2>&1 &
"""
import argparse, json, os, time

from abag_campaign_lib import (CONFIDENCE_KEYS, ROOT, RESULT_DIR_PREFIX,
                               dockq, done_pairs, run_predict, sample_cifs)

TARGETS = ["9ck4", "9d3j", "9i5n", "9m72", "9obn", "22ps", "9yio", "9ncy", "9w14", "9gfr", "9udq", "9jkr"]
MODEL = "opendde-abag"
OUT_BASE = "/tmp/abag_stage3"
GT = f"{ROOT}/examples/ground_truth_structures"
MSA_DIR = f"{OUT_BASE}/msa_cache"
PROGRESS = f"{OUT_BASE}/progress.jsonl"


def fold_one(target, device):
    tid = f"{target}_abag"
    out_dir = f"{OUT_BASE}/opendde_abag"
    status, payload, wall_s = run_predict(
        tid, f"{ROOT}/examples/abag_pilot/{target}_abag.yaml", MODEL, out_dir,
        MSA_DIR, device)
    rec = {"target": target, "model": MODEL, "wall_s": wall_s}
    if status != "ok":
        rec["status"] = status
        rec["stderr"] = payload
        return rec
    rec["status"] = "ok"
    rec["confidence"] = {k: payload.get(k) for k in CONFIDENCE_KEYS}
    rec["all_runs"] = payload.get("all_runs")
    struct_dir = f"{out_dir}/{RESULT_DIR_PREFIX[MODEL]}_results_{tid}/structures"
    cifs = sample_cifs(struct_dir, tid)
    rec["n_samples_scored"] = len(cifs)
    rec["sample_dockq"] = [dockq(c, f"{GT}/{target}.cif") for c in cifs]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int,
                    default=int(os.environ.get("TT_VISIBLE_DEVICES", "0").split(",")[0]))
    a = ap.parse_args()
    os.makedirs(OUT_BASE, exist_ok=True)
    skip = done_pairs(PROGRESS)
    for target in TARGETS:
        if (target, MODEL) in skip:
            print(f"[skip] {target} already in progress.jsonl", flush=True)
            continue
        print(f"[start] {target} {time.strftime('%H:%M:%S')}", flush=True)
        rec = fold_one(target, a.device)
        with open(PROGRESS, "a") as fp:
            fp.write(json.dumps(rec) + "\n")
        top_dockq = None
        if rec.get("sample_dockq"):
            top_dockq = rec["sample_dockq"][0].get("global_dockq")
        print(f"[done]  {target} status={rec['status']} wall_s={rec.get('wall_s')} "
              f"top_dockq={top_dockq}", flush=True)
    print("CAMPAIGN COMPLETE", flush=True)


if __name__ == "__main__":
    main()
