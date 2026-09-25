#!/usr/bin/env python3
"""A46 clause 4 for J2: prove an inference fold cannot feel the fused softmax backward.

Inference is a hard stop, not a bar, and the softmax sites are on the SHARED triangle path
that serves Boltz-2, OF3T, BC2 and RFD3. So the claim owed is byte-identity against the tree
this branch forked from, on every model that executes the site.

This row's claim is stronger than D63's and the arms say so. D63 changed a FORWARD, so its
`on` arm had to move the digest. `softmax_bw` is reached only from `bw` closures -- proved by
AST in `perf/land_standing/renorm_reach.py`, whose `call_count` this branch takes from 1 to 4 --
and an inference fold builds no tape, so BOTH of this row's arms must be byte-identical to base:

  base     the fork point, `origin/main`, where the seam does not exist
  off      this branch, TT_BIO_SOFTMAX_BW_FUSED at its default
  on       this branch, TT_BIO_SOFTMAX_BW_FUSED=1, the fused verb armed everywhere

Three equal digests prove nothing on their own -- a digest blind to the softmax would report
byte-identity whatever happened -- so the run carries its own control:

  control  base tree, TT_BIO_ACCURATE_SOFTMAX_AB=all, an unrelated lever on the same softmax
           in the FORWARD. It must MOVE the digest, or this fixture cannot see a softmax and
           the other three arms are vacuous.

THE CONTROL WAS ITSELF WRONG ONCE, WHICH IS WHY IT SAYS WHICH LEVER. This arm first used
`TT_BIO_HOST_F64_SOFTMAX_AB=all`, and that lever cannot move an inference fold BY DESIGN --
`tenstorrent.host_f64_softmax_site` is gated on a tape being open, and its own docstring says
so: "the gate also needs a tape open, so the variable cannot move an inference fold whatever
it is set to." A control tape-gated exactly like the thing it controls for would have read
`control_moves_the_digest: false` and failed the run for a reason that has nothing to do with
this branch. `accurate_softmax_site` is a plain `_site_flag` with no tape gate and it swaps the
forward softmax for the 5-op accurate chain at `openfold3.trunk`, `.template`, `.msa` and
`.confidence`, so on an openfold3 fold it reaches the forward and moves the digest.

The model is therefore openfold3 and not boltz2: the control lever has to be wired to sites the
fixture actually executes, and these four are openfold3's.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def fold(py, tree, model, fixture, out, env_extra, card, steps):
    env = dict(os.environ)
    env.update({"TT_VISIBLE_DEVICES": card, "TT_BIO_LEASE_CARDS": card,
                "TT_BIO_LEASE_HOLDER": "worker:of3t-softbw", "PYTHONPATH": str(tree)})
    # every arm's levers are set explicitly, so an inherited one cannot decide an arm
    for k in ("TT_BIO_SOFTMAX_BW_FUSED", "TT_BIO_ACCURATE_SOFTMAX_AB",
              "TT_BIO_HOST_F64_SOFTMAX_AB"):
        env.pop(k, None)
    env.update(env_extra)
    cmd = [py, "-m", "tt_bio.main", "predict", str(fixture), "--model", model,
           "--single_sequence", "--sampling_steps", str(steps), "--diffusion_samples", "1",
           "--seed", "0", "--out_dir", str(out)]
    t = time.time()
    r = subprocess.run(cmd, cwd=str(tree), env=env, capture_output=True, text=True)
    cifs = sorted(out.rglob("*.cif"))
    return {"rc": r.returncode, "seconds": round(time.time() - t, 1),
            "cifs": {c.name: digest(c) for c in cifs},
            "tail": (r.stdout + r.stderr)[-2500:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--base-tree", required=True)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--py", default="/home/moritz/tt-bio/env/bin/python3")
    ap.add_argument("--card", default="0")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    wd = Path(a.workdir)
    wd.mkdir(parents=True, exist_ok=True)
    arms = {
        "base": (Path(a.base_tree), {}),
        "off": (Path(a.tree), {}),
        "on": (Path(a.tree), {"TT_BIO_SOFTMAX_BW_FUSED": "1"}),
        "control": (Path(a.base_tree), {"TT_BIO_ACCURATE_SOFTMAX_AB": "all"}),
    }
    res = {}
    for name, (tree, env) in arms.items():
        out = wd / ("out_%s_%s" % (a.model, name))
        res[name] = fold(a.py, tree, a.model, Path(a.fixture), out, env, a.card, a.steps)
        print("[%s] rc=%d %.1fs %s" % (name, res[name]["rc"], res[name]["seconds"],
                                       res[name]["cifs"]), flush=True)
        if res[name]["rc"]:
            print(res[name]["tail"], flush=True)

    base = res["base"]["cifs"]
    held = bool(base) and all(res[k]["cifs"] == base for k in ("off", "on"))
    sees = bool(res["control"]["cifs"]) and res["control"]["cifs"] != base
    rep = {"model": a.model, "fixture": a.fixture, "card": a.card, "arms": res,
           "both_arms_byte_identical_to_base": held,
           "control_moves_the_digest": sees,
           "verdict": "PASS" if (held and sees) else "FAIL"}
    print(json.dumps({k: v for k, v in rep.items() if k != "arms"}, indent=1), flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(rep, open(a.out, "w"), indent=1, sort_keys=True)
        print("wrote", a.out)
    return 0 if rep["verdict"] == "PASS" else 3


if __name__ == "__main__":
    sys.exit(main())
