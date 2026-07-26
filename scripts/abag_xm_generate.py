"""AbAg-XM Tier-A generation: fold every 2026ARK-AB target (164) across the three
generators (protenix-v2, opendde-abag, boltz2) at 50 diffusion samples with
--write_pae, fully offline against the local ColabFold DB. One JSON line per
(target, model) is appended to progress.jsonl so partial progress survives a
restart; a (target, model) already recorded status=ok is skipped.

Fan across cards by passing a target subset per invocation (one card per
invocation, one device context per process):

    TT_VISIBLE_DEVICES=<card> TT_BIO_LEASE_HOLDER=worker:abag-xm-crossmodel-ranking-dataset-p3 \
    PYTHONPATH=$PWD python3 scripts/abag_xm_generate.py --targets 9w14,9j4c --device <card>

Per-sample DockQ against the ARK interface is NOT computed here -- it is a CPU label
(Phase 4), not device work; this phase only produces the 50 CIFs + per-sample PAE
artifacts that Phase 4 scores. The failed-results cross-check (results.json status
must be "ok" AND all_runs/n_cifs must equal n_samples) is enforced so a job whose
results.json says failed is never recorded ok (opendde-abag-paired-msa-offline-gap).
"""
import argparse, json, os, signal, subprocess, sys, time
from pathlib import Path
# colabfold_search (invoked by tt_bio for uncached MSAs) needs localcolabfold's
# own mmseqs (v18, supports --prefilter-mode); the apt mmseqs (v13) exits 1.
_lc_bin = str(Path.home() / "localcolabfold" / ".pixi" / "envs" / "default" / "bin")
if Path(_lc_bin).exists():
    os.environ["PATH"] = _lc_bin + os.pathsep + os.environ.get("PATH", "")

ROOT = Path(__file__).resolve().parent.parent
# Persistent (never /tmp -- qb1 clears it, losing hours of fold work). All tiers/hosts
# share one MSA cache (keyed by sequence hash) and one offline DB.
OUT_BASE = Path.home() / "abag_xm" / "tier_a"
MSA_DIR = Path.home() / "abag_xm" / "msa_cache"
MSA_DB_PATH = Path.home() / ".boltz" / "msa_db"
YAML_DIR = ROOT / "examples" / "abag_xm"
GT = ROOT / "examples" / "ground_truth_structures"
PROGRESS = OUT_BASE / "progress.jsonl"
MANIFEST = ROOT / "docs" / "implementation-parity-data" / "abag-xm-targets.parquet"

MODELS = ["protenix-v2", "opendde-abag", "boltz2"]
RESULT_PREFIX = {"protenix-v2": "protenix", "opendde-abag": "opendde",
                 "opendde": "opendde", "boltz2": "boltz2"}
# Tier-A config: 50 samples, seed grid. boltz2 uses mps=5 (the fixed, memory-bounded
# chunking); protenix/opendde use their own (correct) sampler at the same mps=5.
N_SAMPLES = 50
SEED = 42
FOLD_TIMEOUT_S = 3600  # 50 samples on a 1095-res target can approach ~50 min; give 60.
MPS = 5  # max_parallel_samples -- honours the cap after the 3b fix (boltz2) / always did (protenix)


def all_targets():
    """164 target ids from the manifest, in manifest order (stable across relaunches)."""
    try:
        import pandas as pd
        df = pd.read_parquet(MANIFEST)
        return df["pdb_id"].tolist()
    except Exception:
        # fall back to the on-disk fold YAMLs if pandas/manifest unavailable
        return sorted(p.stem for p in YAML_DIR.glob("*.yaml"))


def done_pairs():
    seen = set()
    if PROGRESS.exists():
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


def _dir_bytes(p):
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def fold_one(target, model, device, n_samples=N_SAMPLES, mps=MPS,
             fold_timeout_s=FOLD_TIMEOUT_S):
    out_dir = OUT_BASE / model.replace("-", "_")
    result_dir = out_dir / f"{RESULT_PREFIX[model]}_results_{target}"
    yaml = YAML_DIR / f"{target}.yaml"
    native = GT / f"{target}.cif"
    if not yaml.exists():
        return {"target": target, "model": model, "status": "no_yaml",
                "yaml": str(yaml)}
    # Input sanity: reject malformed YAMLs (empty sequence / missing chain)
    # BEFORE folding, so a bad YAML fails fast instead of producing silent
    # garbage (the Phase 1 regression that folded antibody-only structures for
    # 22ps/9hv9/9nw4/9udq whose antigen chain was empty).
    try:
        import yaml as _yaml
        _doc = _yaml.safe_load(yaml.read_text())
        _seqs = _doc.get("sequences", []) if _doc else []
        _bad = []
        for _s in _seqs:
            _p = _s.get("protein", {}) if isinstance(_s, dict) else {}
            _sid = _p.get("id"); _seq = _p.get("sequence", "") or ""
            if _sid in ("A", "H", "L") and len(_seq) < 5:
                _bad.append(f"{_sid}={len(_seq)}")
        if _bad or not _seqs:
            return {"target": target, "model": model, "status": "bad_yaml",
                    "yaml": str(yaml), "reason": "empty/short sequence: " + ",".join(_bad)}
    except Exception as _e:
        return {"target": target, "model": model, "status": "bad_yaml",
                "yaml": str(yaml), "reason": f"parse error: {_e}"}
    cmd = [sys.executable, "-m", "tt_bio.main", "predict", str(yaml),
           "--model", model, "--out_dir", str(out_dir),
           "--diffusion_samples", str(n_samples), "--max_parallel_samples", str(mps),
           "--msa_dir", str(MSA_DIR), "--msa_db_path", str(MSA_DB_PATH),
           "--seed", str(SEED), "--override", "--write_pae"]
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            env={**os.environ, "TT_VISIBLE_DEVICES": str(device),
                                 "PYTHONPATH": str(ROOT),
                                 "TT_BIO_LEASE_HOLDER": os.environ.get(
                                     "TT_BIO_LEASE_HOLDER",
                                     "worker:abag-xm-crossmodel-ranking-dataset-p3")},
                            start_new_session=True)
    timed_out = False
    try:
        out, _ = proc.communicate(timeout=fold_timeout_s)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        out, _ = proc.communicate()
        rc = -9
    wall_s = time.time() - t0
    rec = {"target": target, "model": model, "wall_s": round(wall_s, 1),
           "device": device, "n_samples": n_samples, "mps": mps}
    if timed_out:
        rec["status"] = "timed_out"
        rec["stderr"] = f"killed after {fold_timeout_s}s (process group); tail: {(out or '')[-1500:]}"
        return rec
    rjson = result_dir / "results.json"
    if rc != 0 or not rjson.exists():
        rec["status"] = "fold_failed"
        rec["rc"] = rc
        rec["stderr"] = (out or "")[-2000:]
        return rec
    try:
        results = json.load(open(rjson))
        entry = results[0] if isinstance(results, list) else results
    except Exception as e:
        rec["status"] = "bad_results_json"
        rec["stderr"] = f"{e}; tail: {(out or '')[-1000:]}"
        return rec
    struct_dir = result_dir / "structures"
    # Naming: the top-ranked sample (rank 0) is written as <tid>.cif (the "winner", no
    # _model_0 suffix); ranks >= 1 are <tid>_model_<i>.cif. Per-sample PAEs are
    # <tid>_model_<i>_pae.npz for ALL ranks (incl. 0) + a backwards-compat <tid>_pae.npz
    # (top-ranked) which the glob below deliberately excludes so paes == n_samples.
    cif_winner = struct_dir / f"{target}.cif"
    cifs = ([cif_winner] if cif_winner.exists() else []) + \
        sorted(struct_dir.glob(f"{target}_model_*.cif"))
    paes = sorted(struct_dir.glob(f"{target}_model_*_pae.npz"))
    # FAILED-RESULTS CROSS-CHECK: results.json status must be "ok" and the sample
    # count must match on both all_runs and the written CIF/PAE artifacts. A job
    # whose results.json says failed (or silently dropped samples) is never "ok".
    n_runs = len(entry.get("all_runs") or [])
    ok = (entry.get("status") == "ok"
          and n_runs == n_samples
          and len(cifs) == n_samples
          and len(paes) == n_samples)
    if not ok:
        rec["status"] = "incomplete"
        rec["results_status"] = entry.get("status")
        rec["n_runs"] = n_runs
        rec["n_cifs"] = len(cifs)
        rec["n_paes"] = len(paes)
        rec["stderr"] = (out or "")[-1500:]
        return rec
    rec["status"] = "ok"
    rec["confidence"] = {k: entry.get(k) for k in
                         ("confidence_score", "ptm", "iptm", "protein_iptm",
                          "complex_plddt", "runtime_s")}
    rec["n_cifs"] = len(cifs)
    rec["n_paes"] = len(paes)
    rec["bytes_structures"] = _dir_bytes(struct_dir)
    rec["result_dir"] = str(result_dir)
    rec["native_present"] = native.exists()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default=None,
                    help="comma-separated subset; default = all 164 from the manifest")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--n_samples", type=int, default=N_SAMPLES)
    ap.add_argument("--mps", type=int, default=MPS)
    ap.add_argument("--timeout", type=int, default=FOLD_TIMEOUT_S,
                    help="per-fold wall-clock timeout in seconds (default %(default)d)")
    a = ap.parse_args()
    targets = a.targets.split(",") if a.targets else all_targets()
    models = a.models.split(",")
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    MSA_DIR.mkdir(parents=True, exist_ok=True)
    skip = done_pairs()
    print(f"[harness] device={a.device} targets={len(targets)} models={models} "
          f"n_samples={a.n_samples} mps={a.mps} skip={len(skip)}", flush=True)
    for target in targets:
        for model in models:
            if (target, model) in skip:
                print(f"[skip] {target} {model} already ok", flush=True)
                continue
            print(f"[start] {target} {model} {time.strftime('%H:%M:%S')}", flush=True)
            rec = fold_one(target, model, a.device, a.n_samples, a.mps,
                           fold_timeout_s=a.timeout)
            with open(PROGRESS, "a") as fp:
                fp.write(json.dumps(rec) + "\n")
            print(f"[done]  {target} {model} status={rec['status']} "
                  f"wall_s={rec.get('wall_s')} n_cifs={rec.get('n_cifs')} "
                  f"n_paes={rec.get('n_paes')}", flush=True)
    print("CAMPAIGN SLICE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
