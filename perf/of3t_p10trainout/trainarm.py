#!/usr/bin/env python3
"""One arm of the OF3T training-outcome grade: N real training steps, curve flushed per step.

The question this row answers is whether the device-native trunk -- bf16 softmax and layer norm
on the card -- trains the same model as the exact reference does. That is not a tensor question
and `perf/of3t_stepfloor/fullstep.py` cannot answer it: its ground truth is the model's own
prediction plus noise, so its loss value and gradient direction are meaningless by construction.
It times a step. It does not grade one.

So the arm runs the shipped training loop, `tt_bio.train.recipes.train_loop`, on real deposited
structures, and the only difference between arm A and arm B is `exact_training`.

Two preconditions are asserted here rather than remembered, and both are recorded in the
artifact:

1. `ce0f78d60` is an ANCESTOR of the tree. It is the slot-ordering fix: without it `rebind()`
   writes AdamW's new weight into a dict the forward never reads, and 270 diffusion weights
   train on nothing from the second step on. A cherry-pick does not satisfy this -- `4afedc223`,
   `05023a7aa` and `3bb070f9e` are three shas carrying the same two edits and the ancestor test
   sees none of them -- so the branch is merged, not the commit re-applied.

2. `exact_training_ops()` is read INSIDE the run and recorded. `--exact on` must show both ops,
   `--exact off` must show none. The switch is a context manager with no environment variable,
   so an arm that believed its own argument and never looked would be the whole experiment.

`params_with_grad == 2,944` is necessary but NOT sufficient as of `42a664fa6`: the AdamW
write-skip drops writes that round away, so fewer handles move and a tree WITHOUT the slot fix
now reads 2,944 while still mis-slotting. Precondition 1 is what actually covers it.

    trainarm.py --corpus <dir> --checkpoint <ckpt> --steps N --exact off --out <json>
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from perf.clocksample import during                                   # noqa: E402

SLOT_FIX = "ce0f78d60"


def _git(*args):
    return subprocess.run(["git", "-C", str(REPO), *args],
                          capture_output=True, text=True).stdout.strip()


def _precondition_slot_fix():
    """The slot fix by ancestry. Refuses the arm rather than annotating it."""
    ok = subprocess.run(["git", "-C", str(REPO), "merge-base", "--is-ancestor",
                         SLOT_FIX, "HEAD"]).returncode == 0
    if not ok:
        raise SystemExit(
            f"{SLOT_FIX} is not an ancestor of HEAD ({_git('rev-parse', '--short', 'HEAD')}). "
            f"Without it 270 diffusion weights are written where the forward never reads them "
            f"and this arm trains a different model. Merge origin/wk/of3t-p10axis; a "
            f"cherry-pick makes a new sha and does not satisfy this check")
    return {"commit": SLOT_FIX, "is_ancestor_of_head": True,
            "head": _git("rev-parse", "HEAD"),
            "head_short": _git("rev-parse", "--short", "HEAD"),
            "dirty": bool(_git("status", "--porcelain"))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, type=Path,
                    help="the featurised corpus featurise.py dumped")
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--exact", choices=("on", "off"), required=True,
                    help="on: the float64 host softmax and layer norm, the reference arm. "
                         "off: the card's own, which is what ships")
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0, help="reaches the batch ORDER; both arms "
                                                        "of one comparison share it")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup-steps", type=int, default=0,
                    help="train_loop defaults to 1000, which at N~30 leaves the rate at ~9e-6 "
                         "and both arms would agree by not training. 0 runs at lr from step 0")
    ap.add_argument("--global-batch", type=int, default=1)
    ap.add_argument("--rollout", type=int, default=20)
    ap.add_argument("--num-cycles", type=int, default=1)
    ap.add_argument("--checkpoint-every", type=int, default=10)
    ap.add_argument("--out-dir", required=True, type=Path, help="checkpoints and provenance")
    ap.add_argument("--curve", type=Path, help="jsonl, one line per step, flushed as it runs")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    rec = {"argv": sys.argv[1:], "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "precondition_slot_fix": _precondition_slot_fix(), "steps": []}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out_dir.mkdir(parents=True, exist_ok=True)

    def dump():
        a.out.write_text(json.dumps(rec, indent=2, default=str) + "\n")

    dump()
    curve = open(a.curve, "a", buffering=1) if a.curve else None
    t0 = time.perf_counter()
    with during() as clk:
        try:
            from tt_bio import autograd as ag
            from tt_bio.train import openfold3 as of3
            from tt_bio.train.recipes import train_loop

            manifest = a.corpus / "MANIFEST.json"
            if manifest.exists():
                rec["corpus"] = json.loads(manifest.read_text())
            with ag.exact_training(a.exact == "on"):
                # Read the switch from the MECHANISM, not from the argument.
                rec["exact_ops"] = list(ag.exact_training_ops())
                expected = 2 if a.exact == "on" else 0
                if len(rec["exact_ops"]) != expected:
                    raise SystemExit(f"--exact {a.exact} but exact_training_ops() is "
                                     f"{rec['exact_ops']}; the arm is not the arm it claims")
                fwd, ds = of3.adapter(a.corpus, checkpoint=a.checkpoint,
                                      rollout=a.rollout, num_cycles=a.num_cycles, seed=a.seed)
                rec["dataset"] = {"n": len(ds), "files": [p.name for p in ds.paths]}
                dump()

                def on_step(row):
                    row = {**row, "wall_s": round(time.perf_counter() - t0, 3),
                           "aiclk": clk.summary().get(0)}
                    rec["steps"].append(row)
                    if curve:
                        curve.write(json.dumps(row, default=str) + "\n")
                    print(f"[{row['wall_s']:8.1f}s] step {row['step']:3d}  "
                          f"loss {row['loss']:.6f}  lr {row['lr']:.3e}  "
                          f"|g| {row['grad_norm']}", flush=True)
                    dump()

                run = train_loop(fwd, ds, out_dir=a.out_dir, global_batch=a.global_batch,
                                 steps=a.steps, train="weights", seed=a.seed, lr=a.lr,
                                 warmup_steps=a.warmup_steps,
                                 checkpoint_every=a.checkpoint_every, on_step=on_step)
            rec["displacement"] = run["displacement"]
            rec["provenance"] = run["provenance"].as_dict()
            rec["params_trained"] = len(run["params"])
            rec["ok"] = True
        except BaseException as exc:                                   # noqa: BLE001
            import traceback
            rec["ok"] = False
            rec["error"] = f"{type(exc).__name__}: {exc}"
            rec["traceback"] = traceback.format_exc()
            raise
        finally:
            rec["wall_s"] = round(time.perf_counter() - t0, 3)
            rec["aiclk"] = clk.summary()
            rec["aiclk_line"] = clk.line()
            rec["host_load"] = clk.load
            dump()
            if curve:
                curve.close()
    print(rec["aiclk_line"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
