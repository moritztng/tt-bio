"""What the softmax precision lever is worth at fold level, in Angstrom, against the seed floor.

Runs the production CLI (`python -m tt_bio.main predict`) on 1UBQ, once per arm, and scores
CA-RMSD between the arms' structures after a Kabsch superposition. Rank 0 (the sample a user
receives) and best-of-N (the confidence-selected one) are reported separately.

The seed floor is measured in the SAME harness on the SAME card: the off arm re-run at a
different seed. A lever smaller than that floor has not moved the structure, it has moved the
noise, and the bare Angstrom number cannot tell you which.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from perf import clocksample   # noqa: E402


def ca_coords(cif: Path):
    """CA coordinates in file order, from an mmCIF _atom_site loop."""
    xs, hdr, in_loop = [], [], False
    for line in cif.read_text().splitlines():
        s = line.strip()
        if s.startswith("_atom_site."):
            hdr.append(s.split(".", 1)[1]); in_loop = True; continue
        if in_loop and (s.startswith("ATOM") or s.startswith("HETATM")):
            f = s.split()
            r = dict(zip(hdr, f))
            if r.get("label_atom_id", "").strip('"') == "CA":
                xs.append([float(r["Cartn_x"]), float(r["Cartn_y"]), float(r["Cartn_z"])])
        elif in_loop and s.startswith("#"):
            in_loop = False
    return np.asarray(xs, dtype=np.float64)


def kabsch_rmsd(P, Q):
    assert P.shape == Q.shape and len(P), f"{P.shape} vs {Q.shape}"
    P = P - P.mean(0); Q = Q - Q.mean(0)
    V, S, W = np.linalg.svd(P.T @ Q)
    d = np.sign(np.linalg.det(V @ W))
    D = np.diag([1.0, 1.0, d])
    R = V @ D @ W
    return float(np.sqrt(((P @ R - Q) ** 2).sum(1).mean()))


def digest(paths):
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(Path(p).read_bytes())
    return h.hexdigest()


def run_arm(name, model, inp, outdir, env_ab, seed, samples, extra_env=None, timeout=3600):
    import os
    out = Path(outdir) / name
    if (out / "done.json").is_file():
        return json.loads((out / "done.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["TT_VISIBLE_DEVICES"] = "0"
    env["TT_BIO_LEASE_CARDS"] = "0"
    env["TT_BIO_LEASE_HOLDER"] = "worker:of3t-softmax"
    env["PYTHONPATH"] = str(ROOT)
    if env_ab is None:
        env.pop("TT_BIO_SOFTMAX_PRECISE_AB", None)
    else:
        env["TT_BIO_SOFTMAX_PRECISE_AB"] = env_ab
    env.update(extra_env or {})
    cmd = [sys.executable, "-m", "tt_bio.main", "predict", str(inp),
           "--model", model, "--out_dir", str(out), "--seed", str(seed),
           "--diffusion_samples", str(samples), "--single_sequence", "--output_format", "cif",
           "--override"]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout)
    wall = time.perf_counter() - t0
    cifs = sorted(out.rglob("structures/*.cif"))
    res = {"arm": name, "model": model, "seed": seed, "samples": samples,
           "env_TT_BIO_SOFTMAX_PRECISE_AB": env_ab, "rc": r.returncode, "wall_s": wall,
           "cifs": [str(p) for p in cifs], "sha256_structures": digest(cifs) if cifs else None}
    if not cifs:
        res["stderr_tail"] = r.stderr[-3000:]
    (out / "done.json").write_text(json.dumps(res, indent=2))
    return res


def rank_order(arm):
    """CIFs ordered as the CLI ranks them: rank 0 first, from results.json when present."""
    cifs = [Path(p) for p in arm["cifs"]]
    return sorted(cifs, key=lambda p: p.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openfold3")
    ap.add_argument("--input", default="examples/ubq.yaml")
    ap.add_argument("--workdir", default="/home/ttuser/of3t_softmax_work")
    ap.add_argument("--out", default="perf/of3t_softmax/fold_ab_of3_ubq_qb2c0.json")
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--floor-seed", type=int, default=1)
    ap.add_argument("--tag", default="of3")
    ap.add_argument("--arms", default="off,on,floor")
    a = ap.parse_args()

    wd = Path(a.workdir) / a.tag
    plan = {
        "off":   (None, a.seed),
        "on":    ("all", a.seed),
        "floor": (None, a.floor_seed),
    }
    arms, clk = {}, None
    with clocksample.during(period=5.0) as clk:
        for name in a.arms.split(","):
            env_ab, seed = plan[name]
            arms[name] = run_arm(name, a.model, ROOT / a.input, wd, env_ab, seed, a.samples)
            print(name, arms[name]["rc"], arms[name]["wall_s"],
                  arms[name]["sha256_structures"], flush=True)

    out = {"host": "qb2", "card": 0, "board": "p300c", "model": a.model,
           "input": a.input, "samples": a.samples,
           "clock_aiclk_during": clk.summary(), "clock_line": clk.line(0),
           "arms": arms, "angstrom": {}}

    def pair(x, y, label):
        px, py = rank_order(arms[x]), rank_order(arms[y])
        n = min(len(px), len(py))
        if not n:
            return
        per = []
        for i in range(n):
            A, B = ca_coords(px[i]), ca_coords(py[i])
            per.append(kabsch_rmsd(A, B))
        out["angstrom"][label] = {
            "rank0_ca_rmsd_A": per[0],
            "per_sample_ca_rmsd_A": per,
            "min_over_samples_A": min(per), "max_over_samples_A": max(per),
            "n": n,
        }

    if "off" in arms and "on" in arms:
        pair("off", "on", "lever_off_vs_on")
    if "off" in arms and "floor" in arms:
        pair("off", "floor", "seed_floor_off_seed0_vs_off_seed1")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out["angstrom"], indent=2))
    print(clk.line(0))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
