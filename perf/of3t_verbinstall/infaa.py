#!/usr/bin/env python3
"""of3t-verbinstall D3: the A/A floor for the inference hard stop, with the census that counts.

Two folds per model on one card, byte-compared, with a per-PID census taken INSIDE the folding
process (D236: four published A/Bs counted softmax calls in the launcher and read zero). The
census reports three things, and the third is the one this row's lever needs:

  HOST_F64_SOFTMAX_STATS   the site selector's counters, the other route's reach
  autograd_imported        whether `tt_bio.autograd` is in the folding process's sys.modules
                           at all. For a lever with no environment variable, "the training
                           module was never imported" is a stronger statement than "the
                           counter stayed at zero", and it is the one that matches the claim.
  EXACT_SOFTMAX_STATS      this row's counters, read WITHOUT importing the module -- if it is
                           absent from sys.modules there is nothing to read, which is the pass.

A zero is evidence only once the same counter has been seen non-zero under a condition this
row controls. That control is not here and is not a fold: it is `EXACT_SOFTMAX_PKG_HF3.json`,
the training arm on the same tree, where these counters are large.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SITE = '''import atexit, json, sys
def _dump():
    out = {"pid": __import__("os").getpid()}
    try:
        from tt_bio.tenstorrent import HOST_F64_SOFTMAX_STATS as s
        out["HOST_F64_SOFTMAX_STATS"] = dict(s)
    except Exception as e:
        out["HOST_F64_SOFTMAX_STATS"] = "unavailable: %s" % e
    # Deliberately sys.modules and not an import: importing the training package here would
    # create the very thing this census exists to prove absent.
    mod = sys.modules.get("tt_bio.autograd")
    out["autograd_imported"] = mod is not None
    out["EXACT_SOFTMAX_STATS"] = dict(mod.EXACT_SOFTMAX_STATS) if mod is not None else None
    out["exact_softmax_installed"] = bool(mod.exact_softmax_installed()) if mod is not None else False
    print("VERBINSTALL_CENSUS " + json.dumps(out), flush=True)
atexit.register(_dump)
'''


def digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def fold(py, tree, model, fixture, out: Path, card, env_extra):
    # A fresh out_dir per fold, and this is not hygiene: `tt-bio predict` returns in ~3 s when
    # its --out_dir already holds the answer, so a reused workdir re-digests the PREVIOUS
    # fold's .cif and reports a byte-identical A/A for folds that never happened.
    subprocess.run(["rm", "-rf", str(out)], check=False)
    sc = out.parent / ("_sc_%s" % out.name)
    sc.mkdir(parents=True, exist_ok=True)
    (sc / "sitecustomize.py").write_text(SITE)
    env = dict(os.environ)
    env.update({"TT_VISIBLE_DEVICES": card, "TT_BIO_LEASE_CARDS": card,
                "TT_BIO_LEASE_HOLDER": "worker:of3t-verbinstall",
                "PYTHONPATH": str(sc) + os.pathsep + str(tree)})
    env.update(env_extra)
    cmd = [py, "-m", "tt_bio.main", "predict", str(fixture), "--model", model,
           "--single_sequence", "--sampling_steps", "6", "--diffusion_samples", "1",
           "--seed", "0", "--out_dir", str(out)]
    t = time.time()
    r = subprocess.run(cmd, cwd=str(tree), env=env, capture_output=True, text=True)
    secs = time.time() - t
    # EVERY line, not the last one. `sitecustomize` is inherited through PYTHONPATH, so a fold
    # that spawns a worker emits one census per process and keeping only the last one reports
    # whichever process happened to exit last -- usually the launcher, which folds nothing.
    # That is D236 exactly, and it is why the artifacts this row builds on read zero in every
    # arm including the adversarial one.
    census = [json.loads(line.split(" ", 1)[1]) for line in r.stdout.splitlines()
              if line.startswith("VERBINSTALL_CENSUS ")]
    cifs = sorted(out.rglob("*.cif"))
    return {"rc": r.returncode, "seconds": round(secs, 2), "census": census,
            "cifs": {c.name: digest(c) for c in cifs},
            "tail": (r.stdout + r.stderr)[-2500:] if r.returncode else ""}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--python", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--card", required=True)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--selector", default="", help="value for TT_BIO_HOST_F64_SOFTMAX_AB; the "
                    "adversarial arm sets it to `all`, which is every construction site")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    wd = Path(a.workdir)
    env_extra = {"TT_BIO_HOST_F64_SOFTMAX_AB": a.selector} if a.selector else {}
    runs = [fold(a.python, Path(a.tree), a.model, Path(a.fixture),
                 wd / ("rep%d" % i), a.card, env_extra) for i in range(a.reps)]

    ok = all(r["rc"] == 0 for r in runs)
    cifs = [r["cifs"] for r in runs]
    identical = ok and len(cifs) > 1 and all(c == cifs[0] and c for c in cifs)
    imported = [p.get("autograd_imported") for r in runs for p in r["census"]]
    report = {
        "what": "A/A floor for one model: two folds, byte-compared, census taken in the "
                "folding process. `autograd_imported` False in every arm is the inference "
                "hard stop for a lever that has no environment variable to set.",
        "model": a.model, "fixture": a.fixture, "card": a.card,
        "selector_TT_BIO_HOST_F64_SOFTMAX_AB": a.selector or "<unset>",
        "all_folds_succeeded": ok,
        "AA_byte_identical": identical,
        "cif_digests": cifs,
        "census_processes_per_fold": [len(r["census"]) for r in runs],
        "autograd_imported_per_process": imported,
        "training_lever_absent_in_every_fold": all(i is False for i in imported),
        "seconds_per_fold": [r["seconds"] for r in runs],
        "census_per_fold": [r["census"] for r in runs],
        "tails": [r["tail"] for r in runs if r["tail"]],
    }
    print("INFAA " + json.dumps({k: v for k, v in report.items() if k != "census_per_fold"}))
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
