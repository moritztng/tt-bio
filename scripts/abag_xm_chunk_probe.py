"""Phase 3a: bounded chunk-size throughput measurement.

Same target/model, --max_parallel_samples in {5,10,16} at --diffusion_samples 16,
warm offline MSA. Records wall_s/s_sample and checks chunking is numerically
inert (the multiset of per-sample PAE matrices is identical across chunk sizes
for a fixed seed grid). Cap: 3 runs, one card.
"""
import argparse, json, os, shutil, signal, subprocess, sys, time, hashlib
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def run_chunk(model, device, yaml_path, n, chunk, msa_dir, msa_db_path, out_base, seed=42):
    out_dir = out_base / f"chunk{chunk}_n{n}_{int(time.time())}"
    cmd = [sys.executable, "-m", "tt_bio.main", "predict", str(yaml_path),
           "--model", model, "--out_dir", str(out_dir),
           "--diffusion_samples", str(n), "--max_parallel_samples", str(chunk),
           "--msa_dir", str(msa_dir), "--msa_db_path", str(msa_db_path),
           "--seed", str(seed), "--write_pae", "--override"]
    env = {**os.environ, "TT_VISIBLE_DEVICES": str(device),
           "PYTHONPATH": str(ROOT),
           "TT_BIO_LEASE_HOLDER": os.environ.get(
               "TT_BIO_LEASE_HOLDER", "worker:abag-xm-crossmodel-ranking-dataset-p2")}
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, env=env,
                            start_new_session=True)
    try:
        out, _ = proc.communicate(timeout=2400)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        out, _ = proc.communicate()
        rc = -9
    wall_s = time.time() - t0
    rec = {"model": model, "n": n, "chunk": chunk, "wall_s": round(wall_s, 1),
           "rc": rc, "out_dir": str(out_dir)}
    # locate structures
    sdirs = list(Path(out_dir).glob("*_results_*/structures"))
    if not sdirs or rc != 0:
        rec["status"] = "failed"
        rec["tail"] = (out or "")[-1200:]
        return rec, None
    sdir = sdirs[0]
    paes = sorted(sdir.glob("*_model_*_pae.npz"))
    cifs = sorted(sdir.glob("*.cif"))
    rec["cifs"] = len(cifs)
    rec["paes"] = len(paes)
    rec["s_sample"] = round(wall_s / n, 2)
    # canonical signature of the PAE multiset: sorted list of sha256 of each matrix
    sigs = []
    for p in paes:
        a = np.load(p)["pae"].astype(np.float32)
        sigs.append(hashlib.sha256(a.tobytes()).hexdigest())
    sigs.sort()
    rec["pae_multiset_hash"] = hashlib.sha256("|".join(sigs).encode()).hexdigest()
    rec["status"] = "ok"
    return rec, sigs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="protenix-v2")
    ap.add_argument("--device", type=int, required=True)
    ap.add_argument("--target", default="9w14")
    ap.add_argument("--yaml", default=None)
    ap.add_argument("--chunks", default="5,10,16")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--msa_dir", default=str(Path.home() / "abag_xm" / "msa_cache"))
    ap.add_argument("--msa_db_path", default=str(Path.home() / ".boltz" / "msa_db"))
    ap.add_argument("--out_base", default=str(Path.home() / "abag_xm" / "chunk_probe"))
    ap.add_argument("--jsonl", default=None)
    a = ap.parse_args()
    yaml_path = Path(a.yaml) if a.yaml else ROOT / "examples/abag_xm" / f"{a.target}.yaml"
    out_base = Path(a.out_base).expanduser(); out_base.mkdir(parents=True, exist_ok=True)
    jsonl = Path(a.jsonl) if a.jsonl else out_base / "chunk_probe.jsonl"
    chunks = [int(x) for x in a.chunks.split(",")]
    sigs_by_chunk = {}
    for chunk in chunks:
        rec, sigs = run_chunk(a.model, a.device, yaml_path, a.n, chunk,
                             a.msa_dir, a.msa_db_path, out_base)
        sigs_by_chunk[chunk] = sigs
        with open(jsonl, "a") as fh: fh.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
    # bit-identical check: multiset of per-sample PAEs identical across chunk sizes
    base = sigs_by_chunk.get(chunks[0])
    print("\n=== BIT-IDENTICAL CHECK (per-sample PAE multiset vs chunk=%d ) ===" % chunks[0])
    for chunk in chunks:
        s = sigs_by_chunk.get(chunk)
        ok = (s is not None and base is not None and s == base)
        print(f"chunk={chunk:2d}: bit_identical={ok}  (n_paes={len(s) if s else 0})")
    # table
    print("\n=== THROUGHPUT TABLE ===")
    print("chunk | wall_s | s/sample")
    for line in open(jsonl):
        r = json.loads(line)
        if r.get("status") == "ok":
            print("%5d | %6.1f | %7.2f" % (r["chunk"], r["wall_s"], r["s_sample"]))


if __name__ == "__main__":
    main()
