"""Cost probe for the AbAg-XM sampling budget: split a co-fold's wall time into the
per-target fixed cost (MSA + feature prep + trunk) and the marginal cost of each extra
diffusion sample, and record how many bytes a sample costs on disk.

The whole feasibility of a deep-sampling dataset rests on that split: N samples come from
ONE trunk pass, so a design that assumes N independent folds overestimates cost by a large
factor. Run one instance per card, one model each.

Usage (from the worktree, per ALWAYS-ON device rules):
    TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:abag-xm-crossmodel-ranking-dataset \
    PYTHONPATH=$PWD python3 scripts/abag_xm_cost_probe.py \
        --model protenix-v2 --device 0 --target 9w14 --samples 1,1,16
"""
import argparse, json, os, shutil, signal, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULT_PREFIX = {"opendde-abag": "opendde", "opendde": "opendde", "boltz2": "boltz2",
                 "protenix-v2": "protenix", "protenix": "protenix"}


def dir_bytes(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def run_once(model, device, target, yaml_path, n_samples, msa_dir, out_base, timeout_s,
             write_pae):
    tid = yaml_path.stem
    out_dir = out_base / f"{model.replace('-', '_')}_n{n_samples}_{int(time.time())}"
    cmd = [sys.executable, "-m", "tt_bio.main", "predict", str(yaml_path),
           "--model", model, "--out_dir", str(out_dir),
           "--diffusion_samples", str(n_samples), "--msa_dir", str(msa_dir),
           "--seed", "42", "--override"]
    if write_pae:
        cmd.append("--write_pae")
    env = {**os.environ, "TT_VISIBLE_DEVICES": str(device), "PYTHONPATH": str(ROOT),
           "TT_BIO_LEASE_HOLDER": os.environ.get(
               "TT_BIO_LEASE_HOLDER", "worker:abag-xm-crossmodel-ranking-dataset")}
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, env=env,
                            start_new_session=True)
    try:
        out, _ = proc.communicate(timeout=timeout_s)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        out, _ = proc.communicate()
        rc = -9
    wall_s = time.time() - t0

    rec = {"model": model, "target": target, "n_samples": n_samples,
           "wall_s": round(wall_s, 1), "device": device, "write_pae": write_pae,
           "out_dir": str(out_dir)}
    rjson = out_base.glob(f"{out_dir.name}/{RESULT_PREFIX[model]}_results_{tid}/results.json")
    rjson = next(rjson, None)
    if rc != 0 or rjson is None:
        rec["status"] = "failed" if rc != 0 else "no_results_json"
        rec["rc"] = rc
        rec["tail"] = (out or "")[-1500:]
        return rec
    res = json.load(open(rjson))
    entry = res[0] if isinstance(res, list) else res
    rec["status"] = "ok"
    rec["runtime_s"] = entry.get("runtime_s")
    rec["n_runs_reported"] = len(entry.get("all_runs") or [])
    struct_dir = rjson.parent / "structures"
    rec["cifs"] = len(list(struct_dir.glob("*.cif")))
    rec["bytes_structures"] = dir_bytes(struct_dir)
    pae = sorted(struct_dir.glob("*_pae.npz"))
    rec["bytes_pae"] = sum(f.stat().st_size for f in pae)
    rec["bytes_total"] = dir_bytes(rjson.parent)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", type=int, required=True)
    ap.add_argument("--target", default="9w14")
    ap.add_argument("--yaml", default=None, help="defaults to examples/abag_pilot/<target>_abag.yaml")
    ap.add_argument("--samples", default="1,1,16", help="comma list of --diffusion_samples values, in order")
    ap.add_argument("--msa_dir", default=str(Path.home() / "abag_xm" / "msa_cache"),
                    help="PERSISTENT (never /tmp — qb1 clears it, losing hours of MSA work)")
    ap.add_argument("--fresh_msa", action="store_true",
                    help="wipe this target's cache first so run 1 measures the MSA cost too")
    ap.add_argument("--out_base", default=str(Path.home() / "abag_xm" / "cost_probe"))
    ap.add_argument("--jsonl", default=None)
    ap.add_argument("--timeout_s", type=int, default=2400)
    ap.add_argument("--no_pae", action="store_true")
    a = ap.parse_args()

    yaml_path = Path(a.yaml) if a.yaml else ROOT / "examples/abag_pilot" / f"{a.target}_abag.yaml"
    msa_dir = Path(a.msa_dir).expanduser()
    out_base = Path(a.out_base).expanduser()
    out_base.mkdir(parents=True, exist_ok=True)
    jsonl = Path(a.jsonl) if a.jsonl else out_base / "cost_probe.jsonl"
    if a.fresh_msa and msa_dir.exists():
        shutil.rmtree(msa_dir)
    msa_dir.mkdir(parents=True, exist_ok=True)

    for i, n in enumerate([int(x) for x in a.samples.split(",")]):
        rec = run_once(a.model, a.device, a.target, yaml_path, n, msa_dir, out_base,
                       a.timeout_s, not a.no_pae)
        rec["run_index"] = i
        rec["msa_cold"] = bool(a.fresh_msa and i == 0)
        with open(jsonl, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
