"""Phase 3b perf-hunt probe: measure the Protenix-v2/OpenDDE denoise *trace* win on the
campaign target, and verify it is numerically lossless across the full 200-step
trajectory (the two-gate rule: isolated PCC is not enough — compare per-sample PAE
multisets between trace-off and trace-on, same seed).

Why single_sequence: qb2 has no local ColabFold DB, and the diffusion sampling loop
(the thing being traced) is MSA-independent — MSA only feeds the once-per-fold trunk
conditioning (cond), built before the per-sample loop. So the trace win % measured
single_sequence transfers to the real (MSA-fed) fold, and the lossless check is valid
because the denoise math is identical.

Also discharges the owed Phase-2 acceptance leg: capture `ss -tnp` for the predict pid
during a fold and confirm zero outbound connections (offline-MSA acceptance, except
single_sequence here has no MSA at all, so this proves the predict path itself makes no
network calls during sampling).

Usage (from the worktree, per ALWAYS-ON device rules):
    TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:abag-xm-crossmodel-ranking-dataset-p3 \
    PYTHONPATH=$PWD python3 scripts/abag_xm_trace_probe.py --device 0 --target 9w14
"""
import argparse, hashlib, json, os, signal, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULT_PREFIX = {"opendde-abag": "opendde", "opendde": "opendde",
                 "protenix-v2": "protenix", "protenix": "protenix"}


def _pae_files(rjson_parent, tid):
    sdir = rjson_parent / "structures"
    return sorted(sdir.glob(f"{tid}_model_*_pae.npz"))


def _pae_hashes(pae_files):
    return [hashlib.sha256(f.read_bytes()).hexdigest() for f in pae_files]


def run_leg(model, device, target, yaml_path, n_samples, out_dir, trace, timeout_s):
    tid = yaml_path.stem
    leg = "trace_on" if trace else "trace_off"
    out_dir = out_dir / f"{model.replace('-', '_')}_n{n_samples}_{leg}_{int(time.time())}"
    out_dir.mkdir(parents=True, exist_ok=True)
    base = [sys.executable, "-m", "tt_bio.main", "predict", str(yaml_path),
            "--model", model, "--out_dir", str(out_dir),
            "--diffusion_samples", str(n_samples), "--seed", "42",
            "--single_sequence", "--write_pae", "--override"]
    cmd = base + (["--trace"] if trace else [])
    env = {**os.environ, "TT_VISIBLE_DEVICES": str(device), "PYTHONPATH": str(ROOT),
           "TT_BIO_LEASE_HOLDER": os.environ.get(
               "TT_BIO_LEASE_HOLDER", "worker:abag-xm-crossmodel-ranking-dataset-p3")}
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, env=env,
                            start_new_session=True)
    pid = proc.pid
    ss_samples = []
    # Sample ss -tnp a few times mid-run while the fold is in flight.
    while proc.poll() is None:
        try:
            out, _ = proc.communicate(timeout=15)
            break
        except subprocess.TimeoutExpired:
            try:
                ss = subprocess.run(
                    ["ss", "-tnp"], capture_output=True, text=True, timeout=5).stdout
                # established connections owned by this process group
                lines = [ln for ln in ss.splitlines()
                         if f"pid={pid}" in ln]
                ss_samples.append({
                    "t_s": round(time.time() - t0, 1),
                    "n_estab": len(lines),
                    "lines": lines[:3]})
            except Exception as e:
                ss_samples.append({"t_s": round(time.time() - t0, 1), "err": str(e)})
            if time.time() - t0 > timeout_s:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                break
    try:
        out, _ = proc.communicate(timeout=5)
    except Exception:
        out = ""
    wall_s = time.time() - t0
    rc = proc.returncode

    rec = {"model": model, "target": target, "n_samples": n_samples, "trace": trace,
           "wall_s": round(wall_s, 1), "device": device, "leg": leg, "rc": rc,
           "out_dir": str(out_dir)}
    rjson = next(out_dir.glob(
        f"{RESULT_PREFIX[model]}_results_{tid}/results.json"), None)
    if rc != 0 or rjson is None:
        rec["status"] = "failed" if rc != 0 else "no_results_json"
        rec["tail"] = (out or "")[-2000:]
        rec["ss_samples"] = ss_samples
        return rec
    res = json.load(open(rjson))
    entry = res[0] if isinstance(res, list) else res
    rec["status"] = "ok"
    rec["runtime_s"] = entry.get("runtime_s")
    paes = _pae_files(rjson.parent, tid)
    rec["n_pae_files"] = len(paes)
    rec["pae_hashes"] = _pae_hashes(paes)
    rec["ss_samples"] = ss_samples
    rec["outbound_connections_seen"] = any(s.get("n_estab", 0) > 0 for s in ss_samples)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="protenix-v2")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--target", default="9w14")
    ap.add_argument("--yaml", default=None)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--out_base",
                    default=str(Path.home() / "abag_xm" / "trace_probe"))
    ap.add_argument("--timeout_s", type=int, default=2400)
    a = ap.parse_args()

    yaml_path = Path(a.yaml) if a.yaml else ROOT / "examples/abag_xm" / f"{a.target}.yaml"
    out_base = Path(a.out_base).expanduser()
    out_base.mkdir(parents=True, exist_ok=True)
    jsonl = out_base / "trace_probe.jsonl"

    rec_off = run_leg(a.model, a.device, a.target, yaml_path, a.samples,
                      out_base, False, a.timeout_s)
    with open(jsonl, "a") as fh:
        fh.write(json.dumps(rec_off) + "\n")
    print(json.dumps(rec_off), flush=True)
    if rec_off.get("status") != "ok":
        print("Leg A (trace_off) failed; aborting before trace_on leg.", flush=True)
        return

    rec_on = run_leg(a.model, a.device, a.target, yaml_path, a.samples,
                     out_base, True, a.timeout_s)
    with open(jsonl, "a") as fh:
        fh.write(json.dumps(rec_on) + "\n")
    print(json.dumps(rec_on), flush=True)

    h_off = sorted(rec_off.get("pae_hashes", []))
    h_on = sorted(rec_on.get("pae_hashes", []))
    verdict = {
        "pae_count_match": len(h_off) == len(h_on) == a.samples,
        "pae_multiset_identical": h_off == h_on,
        "wall_off_s": rec_off.get("wall_s"),
        "wall_on_s": rec_on.get("wall_s"),
        "delta_s": round((rec_off.get("wall_s", 0) - rec_on.get("wall_s", 0)), 1),
        "speedup_pct": round(
            (1 - rec_on.get("wall_s", 0) / rec_off.get("wall_s", 1)) * 100, 1)
            if rec_off.get("wall_s") else None,
        "outbound_connections_off": rec_off.get("outbound_connections_seen"),
        "outbound_connections_on": rec_on.get("outbound_connections_seen"),
    }
    with open(jsonl, "a") as fh:
        fh.write(json.dumps({"verdict": verdict}) + "\n")
    print(json.dumps({"verdict": verdict}), flush=True)


if __name__ == "__main__":
    main()
