"""Stage 3 fold campaign: OpenDDE-abag, 5 diffusion samples, on the 12 verified
2026ARK-AB abag targets -- the missing piece from Stage 1 (which only folded 1 seed).
Gives OpenDDE its own measured seed self-consistency, comparable to the Protenix-v2 /
Boltz-2 numbers already in docs/implementation-parity-data/abag-pilot-stage2-final.json.

Scores every sample (not just the winner) against the native CIF. Appends one JSON
line per target to progress.jsonl so partial progress survives a restart.

Run on pc (single Blackhole card 0, TT_VISIBLE_DEVICES=0):
    nohup env TT_VISIBLE_DEVICES=0 python3 scripts/abag_pilot_stage3_opendde_fold.py \
        > /tmp/abag_stage3/campaign.log 2>&1 &
"""
import json, os, subprocess, sys, time

TARGETS = ["9ck4", "9d3j", "9i5n", "9m72", "9obn", "22ps", "9yio", "9ncy", "9w14", "9gfr", "9udq", "9jkr"]
MODEL = "opendde-abag"
ROOT = "/home/moritz/.coworker/wt/flagship-abag-trust-signal-rethink"
OUT_BASE = "/tmp/abag_stage3"
GT = f"{ROOT}/examples/ground_truth_structures"
MSA_DIR = f"{OUT_BASE}/msa_cache"
PROGRESS = f"{OUT_BASE}/progress.jsonl"
DOCKQ_PYTHON = os.path.expanduser("~/.opendde_dockq_venv/bin/python3")

def done_targets():
    seen = set()
    if os.path.exists(PROGRESS):
        for line in open(PROGRESS):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                seen.add(r["target"])
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

def fold_one(target):
    tid = f"{target}_abag"
    out_dir = f"{OUT_BASE}/opendde_abag"
    result_dir = f"{out_dir}/opendde_results_{tid}"
    yaml = f"{ROOT}/examples/abag_pilot/{target}_abag.yaml"
    native = f"{GT}/{target}.cif"
    t0 = time.time()
    r = subprocess.run(
        ["/home/moritz/tt-bio/env/bin/python3", "-m", "tt_bio.main", "predict", yaml, "--model", MODEL,
         "--out_dir", out_dir, "--diffusion_samples", "5", "--msa_dir", MSA_DIR,
         "--seed", "42", "--override"],
        cwd=ROOT, capture_output=True, text=True,
        env={**os.environ, "TT_VISIBLE_DEVICES": "0", "PYTHONPATH": ROOT},
    )
    wall_s = time.time() - t0
    rec = {"target": target, "model": MODEL, "wall_s": round(wall_s, 1)}
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
    return rec

def main():
    os.makedirs(OUT_BASE, exist_ok=True)
    skip = done_targets()
    for target in TARGETS:
        if target in skip:
            print(f"[skip] {target} already in progress.jsonl", flush=True)
            continue
        print(f"[start] {target} {time.strftime('%H:%M:%S')}", flush=True)
        rec = fold_one(target)
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
