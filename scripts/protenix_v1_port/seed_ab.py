#!/usr/bin/env python3
"""Does --seed actually reach the Protenix sampler? A/A first, then A/B.

Same seed twice measures the floor: on pc card 0 that floor is NOT zero
(pc-card0-512aa-fold-nondeterminism), so a bit-exact verdict cannot rest here and the A/A run
is what makes the A/B number readable. Different seeds must beat it by orders of magnitude.

Reports RMSD after a Kabsch superposition, because the sampler's coordinate frame is arbitrary:
a raw coordinate delta would report the frame, not the structure.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def read_struct(path):
    xs = []
    for line in Path(path).read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            xs.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return np.asarray(xs)


def read_cif(path):
    xs = []
    hdr, cols = [], None
    for line in Path(path).read_text().splitlines():
        s = line.strip()
        if s.startswith("_atom_site."):
            hdr.append(s.split(".", 1)[1])
            continue
        if hdr and s and not s.startswith(("#", "loop_", "_")):
            f = s.split()
            if len(f) < len(hdr):
                continue
            if cols is None:
                cols = [hdr.index(k) for k in ("Cartn_x", "Cartn_y", "Cartn_z")]
            xs.append([float(f[c]) for c in cols])
        elif hdr and (s.startswith("#") or not s):
            if xs:
                break
    return np.asarray(xs)


def rmsd(a, b):
    """Kabsch-superposed RMSD. Both arms come off the same target, so the atom order matches."""
    n = min(len(a), len(b))
    a, b = a[:n] - a[:n].mean(0), b[:n] - b[:n].mean(0)
    u, _s, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1.0, 1.0, d]) @ vt
    return float(np.sqrt((((a @ r) - b) ** 2).sum(1).mean()))


def run(cli, target, seed, out, steps, model):
    cmd = cli + ["predict", target, "--model", model, "--single_sequence",
                 "--sampling_steps", str(steps), "--diffusion_samples", "1",
                 "--seed", str(seed), "--out_dir", out]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if r.returncode != 0:
        sys.exit("fold failed (seed %s):\n%s\n%s" % (seed, r.stdout[-2000:], r.stderr[-2000:]))
    hits = sorted(Path(out).rglob("*.cif")) + sorted(Path(out).rglob("*.pdb"))
    if not hits:
        sys.exit("no structure written under " + out)
    p = hits[0]
    return read_cif(p) if p.suffix == ".cif" else read_struct(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="examples/multimer.yaml")
    ap.add_argument("--model", default="protenix-v1")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=3, help="A/A control runs (>=3)")
    ap.add_argument("--out", default="/tmp/pv1/seed")
    ap.add_argument("--python", default="python3")
    args = ap.parse_args()

    cli = [args.python, "-m", "tt_bio.main"]
    base = Path(args.out)

    aa = [run(cli, args.target, 7, str(base / f"a{i}"), args.steps, args.model)
          for i in range(args.repeats)]
    aa_deltas = [rmsd(aa[0], aa[i]) for i in range(1, len(aa))]
    b = run(cli, args.target, 12345, str(base / "b"), args.steps, args.model)
    ab = rmsd(aa[0], b)

    a_a = max(aa_deltas) if aa_deltas else 0.0
    res = {"model": args.model, "target": args.target, "steps": args.steps,
           "repeats": args.repeats, "a_a_deltas": aa_deltas, "a_a_delta": a_a,
           "a_b_delta": ab, "ratio": (ab / a_a) if a_a else float("inf")}
    print(json.dumps(res, indent=2))
    Path(args.out + "/seed_ab.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
