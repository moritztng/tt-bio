#!/usr/bin/env python3
"""On/off bit-exactness for the batched program config across the OpenFold3 parity-leg targets.

Why this and not scripts/full_parity_gate.py's openfold3 legs: those legs are legacy R/D/X, a
device SELF-CONSISTENCY floor measured across 5 seeds. A program-config swap that changed the
numerics would change them the same way every seed, so R/D/X would not move -- it is the wrong
instrument for this question, and at ~5 min/fold x 35 folds it costs 3 hours of card to answer
it weakly. What actually settles it is the same test used for protenix-v2, opendde and prot300:
fold the identical target twice, once with TT_BIO_BATCHED_MATMUL=1 and once with 0, and compare
the output byte for byte. Identical CIF sha over N targets is a direct statement that the config
changed nothing, per target.

One process per arm: the kill switch is read once at tt_bio.tenstorrent import.
"""
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGETS = ["examples/8hel_nomsa.yaml", "examples/7xi5_tmpl.yaml", "examples/7xi5_notmpl.yaml",
           "examples/9bk6.yaml", "examples/ubq.yaml", "examples/prot.yaml"]


def fold(target, arm, out_root, steps, py):
    out = out_root / f"{Path(target).stem}_{arm}"
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["TT_BIO_BATCHED_MATMUL"] = "1" if arm == "on" else "0"
    cmd = [py, "-m", "tt_bio.main", "predict", target, "--model", "openfold3",
           "--sampling_steps", str(steps), "--diffusion_samples", "1", "--seed", "0",
           "--single_sequence", "--out_dir", str(out)]
    t0 = time.monotonic()
    p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=1500)
    wall = time.monotonic() - t0
    cifs = sorted(out.rglob("*.cif"))
    if p.returncode != 0 or not cifs:
        tail = (p.stderr or p.stdout or "")[-300:].replace("\n", " ")
        return {"arm": arm, "rc": p.returncode, "wall_s": round(wall, 1), "err": tail}
    h = hashlib.sha256(b"".join(c.read_bytes() for c in cifs)).hexdigest()[:16]
    return {"arm": arm, "rc": 0, "wall_s": round(wall, 1), "cif_sha256_16": h,
            "n_cif": len(cifs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--python", default=str(Path.home() / "tt-bio-dev/env/bin/python3"))
    ap.add_argument("--targets", nargs="*", default=TARGETS)
    a = ap.parse_args()

    out_root = ROOT / "perf/ktiles/of3_parity_ab"
    rows = []
    for t in a.targets:
        r = {"target": t}
        for arm in ("on", "off"):
            r[arm] = fold(t, arm, out_root, a.steps, a.python)
        on, off = r["on"], r["off"]
        r["verdict"] = ("ERROR" if on.get("rc") or off.get("rc") else
                        "BIT-EXACT" if on["cif_sha256_16"] == off["cif_sha256_16"] else "DIFFERS")
        rows.append(r)
        print(f"{t:<32} {r['verdict']:<10} on={on.get('cif_sha256_16') or on.get('err','')[:40]} "
              f"off={off.get('cif_sha256_16') or off.get('err','')[:40]} "
              f"({on.get('wall_s')}s/{off.get('wall_s')}s)", flush=True)
        Path(a.out).write_text(json.dumps({"steps": a.steps, "rows": rows}, indent=1))

    ok = [r for r in rows if r["verdict"] == "BIT-EXACT"]
    print(f"\n{len(ok)}/{len(rows)} targets bit-exact")
    sys.exit(0 if len(ok) == len(rows) else 1)


main()
