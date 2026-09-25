#!/usr/bin/env python3
"""The OFF arm's gradient against a FLOAT64 reference, per tensor, with the ON arm beside it.

Device leg. Thin on purpose: `perf/bcx_afgrad/afgrad.py` already builds the float64 reference
this campaign grades against, so this runs ITS two commands once per arm in ONE process and
diffs the readings. Rewriting the reference would mean grading against a second one.

  vjp    per-block VJP, teacher-forced block by block on the same bf16-rounded inputs, with the
         float64, float32 and bfloat16 reference arms. This is the PER-TENSOR half: every
         Evoformer and extra-MSA block boundary is a tensor with a path, so the worst one can
         be named.
  stack  the whole-stack dL/dlogits against float64, which is the number the campaign's other
         rows quote, plus the finite-difference and permutation controls.

Both arms run against the SAME float64 reference and are never graded against each other. BC2
does not train weights, so "parameter path" here is the block boundary path -- `evo[i].msa`,
`evo[i].pair`, `extra[i].pair` -- and the logits, which is what the loop differentiates.

    grade.py --card 0 --n 288 --out out/
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
import types

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(ROOT), str(ROOT / "perf" / "bcx_afgrad"), str(ROOT / "perf" / "bcx_stack")):
    if p not in sys.path:
        sys.path.insert(0, p)

import afgrad as A                                                     # noqa: E402
from tt_bio import autograd as ag                                      # noqa: E402


def ns(**kw):
    base = dict(params=A.DEFAULT_PARAMS, card=int(os.environ.get("TT_VISIBLE_DEVICES", "0")),
                n=288, extra=4, evo=48, blocks=None, controls_all=False, controls_only=False,
                seed=0, eps="1e-1,3e-2,1e-2,3e-3,1e-3", ckpt=False, msa_mask=False,
                ns="288", ks="1,2,4", stacks="evo,extra", steps=30, warm=3, out="time.json",
                tag="", threads=8)
    base.update(kw)
    return types.SimpleNamespace(**base)


def run(cmd, on, args, tag):
    """One afgrad command under one arm, with the exact counters read around it.

    `afgrad.cmd_stack` builds its filename from n/extra/evo and not from --tag, so three arms
    in one process would overwrite each other. Each arm gets its own output directory instead.
    """
    A.OUT = HERE / "out" / tag
    A.OUT.mkdir(parents=True, exist_ok=True)
    sm0, ln0 = dict(ag.EXACT_SOFTMAX_STATS), dict(ag.EXACT_LAYER_NORM_STATS)
    t0 = time.time()
    with ag.exact_training(on):
        {"vjp": A.cmd_vjp, "stack": A.cmd_stack}[cmd](args)
    return {"cmd": cmd, "exact": on, "seconds": round(time.time() - t0, 2),
            "sm": {k: ag.EXACT_SOFTMAX_STATS[k] - sm0[k] for k in sm0},
            "ln": {k: ag.EXACT_LAYER_NORM_STATS[k] - ln0[k] for k in ln0},
            "artifact": str(A.OUT)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", type=int, default=int(os.environ.get("TT_VISIBLE_DEVICES", "0")))
    ap.add_argument("--n", type=int, default=288)
    ap.add_argument("--evo", type=int, default=48)
    ap.add_argument("--extra", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cmds", default="stack,vjp")
    ap.add_argument("--out", default=str(HERE / "GRADE.json"))
    a = ap.parse_args()

    log = []
    # ON first, then OFF, then ON again: the repeat is the control that says the process did not
    # drift between the arms.
    for cmd in a.cmds.split(","):
        for on, tag in ((True, "on"), (False, "off"), (True, "on2")):
            args = ns(card=a.card, n=a.n, evo=a.evo, extra=a.extra, seed=a.seed,
                      tag=f"exact_{tag}_n{a.n}")
            log.append({**run(cmd, on, args, tag), "tag": tag})
            print(json.dumps(log[-1]), flush=True)
    blob = {"host": os.uname().nodename, "card": a.card, "n": a.n,
            "when_utc": time.strftime("%FT%TZ", time.gmtime()),
            "loadavg": os.getloadavg(), "runs": log,
            "note": "per-arm numbers live in afgrad's own artifacts, keyed by --tag"}
    pathlib.Path(a.out).write_text(json.dumps(blob, indent=1))
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
