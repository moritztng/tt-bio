"""Hard-negative specificity fold campaign: 3 real, PDB-verified antibody Fabs
(REGN10933/casirivimab, LY-CoV555/bamlanivimab, S309/sotrovimab) vs WT SARS-CoV-2 RBD
and single-point escape mutants with well-documented experimental binding outcomes
(K417N, E484K). See docs/implementation-parity-data/abag-hard-negative-manifest.json
for provenance/rationale of each pair.

Usage:
    nohup env TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:flagship-abag-hard-negative-specificity \
        python3 scripts/abag_hard_negative_fold.py > /tmp/abag_hard_neg/campaign.log 2>&1 &
"""
import json, os, signal, subprocess, sys, time

FOLD_TIMEOUT_S = 1200
TARGETS = ["regn10933_wt", "regn10933_k417n", "lycov555_wt", "lycov555_e484k",
           "s309_wt", "s309_e484k", "s309_k417n"]
MODELS = ["opendde-abag"]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_BASE = "/tmp/abag_hard_neg"
YAML_DIR = f"{ROOT}/examples/abag_hard_negative"
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
                if r.get("status") == "ok":
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


def fold_one(target, model, device):
    tid = target
    out_dir = f"{OUT_BASE}/{model.replace('-', '_')}"
    result_dir = f"{out_dir}/{RESULT_DIR_PREFIX[model]}_results_{tid}"
    yaml = f"{YAML_DIR}/{target}.yaml"
    t0 = time.time()
    proc = subprocess.Popen(
        [sys.executable, "-m", "tt_bio.main", "predict", yaml, "--model", model,
         "--out_dir", out_dir, "--diffusion_samples", "5", "--msa_dir", MSA_DIR,
         "--seed", "42", "--override", "--write_pae"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        env={**os.environ, "TT_VISIBLE_DEVICES": str(device), "PYTHONPATH": ROOT},
        start_new_session=True,
    )
    timed_out = False
    try:
        out, _ = proc.communicate(timeout=FOLD_TIMEOUT_S)
        returncode = proc.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        out, _ = proc.communicate()
        returncode = -9

    wall_s = time.time() - t0
    rec = {"target": target, "model": model, "wall_s": round(wall_s, 1)}
    rjson = f"{result_dir}/results.json"
    if timed_out:
        rec["status"] = "timed_out"
        rec["stderr"] = f"killed after {FOLD_TIMEOUT_S}s; tail: {(out or '')[-1000:]}"
        return rec
    if returncode != 0 or not os.path.exists(rjson):
        rec["status"] = "fold_failed"
        rec["stderr"] = (out or "")[-2000:]
        return rec
    results = json.load(open(rjson))
    entry = results[0] if isinstance(results, list) else results
    if entry.get("status") == "failed":
        rec["status"] = "fold_failed"
        rec["stderr"] = f"inner status=failed: {json.dumps(entry)[-1000:]}"
        return rec
    rec["status"] = "ok"
    rec["confidence"] = {k: entry.get(k) for k in
                          ("confidence_score", "ptm", "iptm", "protein_iptm", "complex_plddt", "runtime_s")}
    struct_dir = f"{result_dir}/structures"
    cifs = sample_cifs(struct_dir, tid)
    rec["n_samples_scored"] = len(cifs)
    rec["winner_cif"] = cifs[0] if cifs else None
    return rec


def main():
    os.makedirs(OUT_BASE, exist_ok=True)
    device = int(os.environ.get("TT_VISIBLE_DEVICES", "0"))
    skip = done_pairs()
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
