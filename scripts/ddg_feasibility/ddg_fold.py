"""Fold WT+mutant AB-Bind complexes (1VFB, 3HFM) with opendde-abag for the
ddG-on-predicted-structures feasibility test. See ~/.coworker/state/
flagship-abag-hard-negative-specificity.md for context."""
import json, os, signal, subprocess, sys, time

FOLD_TIMEOUT_S = 1200
ROOT = "/home/ttuser/.coworker/wt/flagship-abag-hard-negative-specificity"
YAML_DIR = "/tmp/abbind/yaml"
OUT_BASE = "/tmp/abbind/folds"
MSA_DIR = f"{OUT_BASE}/msa_cache"
PROGRESS = f"{OUT_BASE}/progress.jsonl"
MODEL = "opendde-abag"


def done_tags():
    seen = set()
    if os.path.exists(PROGRESS):
        for line in open(PROGRESS):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if r.get("status") == "ok":
                    seen.add(r["tag"])
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


def fold_one(tag, device):
    out_dir = f"{OUT_BASE}/{tag}"
    result_dir = f"{out_dir}/opendde_results_{tag}"
    yaml = f"{YAML_DIR}/{tag}.yaml"
    t0 = time.time()
    proc = subprocess.Popen(
        [sys.executable, "-m", "tt_bio.main", "predict", yaml, "--model", MODEL,
         "--out_dir", out_dir, "--diffusion_samples", "1", "--msa_dir", MSA_DIR,
         "--sampling_steps", "50", "--fast",
         "--seed", "42", "--override", "--write_pae"],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        env={**os.environ, "TT_VISIBLE_DEVICES": str(device),
             "TT_BIO_LEASE_HOLDER": "worker:flagship-abag-hard-negative-specificity",
             "PYTHONPATH": ROOT},
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
    rec = {"tag": tag, "wall_s": round(wall_s, 1)}
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
    cifs = sample_cifs(struct_dir, tag)
    rec["n_samples_scored"] = len(cifs)
    rec["winner_cif"] = cifs[0] if cifs else None
    return rec


def main():
    os.makedirs(OUT_BASE, exist_ok=True)
    device = int(sys.argv[2]) if len(sys.argv) > 2 else int(os.environ.get("TT_VISIBLE_DEVICES", "0"))
    manifest = json.load(open(f"{YAML_DIR}/manifest.json"))
    tags = [f"{m['complex']}_{m['tag']}" for m in manifest]
    if len(sys.argv) > 1 and sys.argv[1] != "all":
        tags = [sys.argv[1]]
    skip = done_tags()
    for tag in tags:
        if tag in skip:
            print(f"[skip] {tag} already ok", flush=True)
            continue
        print(f"[start] {tag} {time.strftime('%H:%M:%S')}", flush=True)
        rec = fold_one(tag, device)
        with open(PROGRESS, "a") as fp:
            fp.write(json.dumps(rec) + "\n")
        print(f"[done]  {tag} status={rec['status']} wall_s={rec.get('wall_s')} "
              f"iptm={rec.get('confidence', {}).get('iptm')}", flush=True)
    print("CAMPAIGN COMPLETE", flush=True)


if __name__ == "__main__":
    main()
