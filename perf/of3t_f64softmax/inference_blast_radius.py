#!/usr/bin/env python3
"""D63: what the host float64 softmax path does to a shipped fold, measured by digest.

Three arms per model, all on the same card, same seed, same fixture:

  base      the tree BEFORE this row's commit, so the path is not in the source at all
  off       this row's tree with the path present and its default (off at every site)
  on        this row's tree with TT_BIO_HOST_F64_SOFTMAX_AB=all

`off` must be byte-identical to `base`: that is the blast radius, and it is zero or it is not.
`on` must DIFFER, and it is the control that keeps the first claim from being vacuous -- a digest
that cannot see the softmax would report byte-identity whatever the path did.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PY = "/home/ttuser/tt-bio-dev/env/bin/python"


def digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def fold(tree: Path, model: str, fixture: Path, out: Path, env_extra: dict, card: str):
    env = dict(os.environ)
    env.update({"TT_VISIBLE_DEVICES": card, "TT_BIO_LEASE_CARDS": card,
                "TT_BIO_LEASE_HOLDER": "worker:of3t-f64softmax",
                "PYTHONPATH": str(tree)})
    env.update(env_extra)
    for k in ("TT_BIO_HOST_F64_SOFTMAX_AB",):
        if k not in env_extra:
            env.pop(k, None)
    cmd = [PY, "-m", "tt_bio.main", "predict", str(fixture), "--model", model,
           "--single_sequence", "--sampling_steps", "6", "--diffusion_samples", "1",
           "--seed", "0", "--out_dir", str(out)]
    t = time.time()
    r = subprocess.run(cmd, cwd=str(tree), env=env, capture_output=True, text=True)
    cifs = sorted(out.rglob("*.cif"))
    return {"rc": r.returncode, "seconds": round(time.time() - t, 1),
            "cifs": {c.name: digest(c) for c in cifs},
            "tail": r.stdout[-1500:] if r.returncode else (r.stdout + r.stderr)[-3000:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--base-tree", required=True)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--card", default="3")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    wd = Path(a.workdir)
    wd.mkdir(parents=True, exist_ok=True)
    arms = {
        "base": (Path(a.base_tree), {}),
        "off": (Path(a.tree), {}),
        "on": (Path(a.tree), {"TT_BIO_HOST_F64_SOFTMAX_AB": "all"}),
    }
    res = {}
    for name, (tree, env) in arms.items():
        out = wd / ("out_%s_%s" % (a.model, name))
        res[name] = fold(tree, a.model, Path(a.fixture), out, env, a.card)
        print("[%s] rc=%d %.1fs %s" % (name, res[name]["rc"], res[name]["seconds"],
                                       res[name]["cifs"]), flush=True)
        if res[name]["rc"]:
            print(res[name]["tail"], flush=True)

    same = res["base"]["cifs"] == res["off"]["cifs"] and bool(res["off"]["cifs"])
    moved = res["on"]["cifs"] != res["off"]["cifs"] and bool(res["on"]["cifs"])
    rep = {"model": a.model, "fixture": a.fixture, "card": a.card, "arms": res,
           "off_is_byte_identical_to_base": same,
           "on_moves_the_digest": moved,
           "verdict": "PASS" if (same and moved) else "FAIL"}
    print(json.dumps({k: v for k, v in rep.items() if k != "arms"}, indent=1), flush=True)
    if a.out:
        json.dump(rep, open(a.out, "w"), indent=1, sort_keys=True)
        print("wrote", a.out)
    return 0 if rep["verdict"] == "PASS" else 3


if __name__ == "__main__":
    sys.exit(main())
