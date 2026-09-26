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


def _evaluate(fwd, corpus, stage, seed, tokens=None):
    """`af3_loss` on every target of a held-out corpus, with the weights as they are now.

    Run on the SHIPPED inference path -- no hook, no tape, the device`s own softmax and layer
    norm -- for both arms. That is one instrument, identical across the comparison, and it is
    the path a user actually gets; scoring arm A with its own arithmetic and arm B with a
    different one would make the metric a property of the instrument instead of the weights.
    """
    from tt_bio import autograd as ag
    from tt_bio.train import objectives
    from tt_bio.train.losses import of3_loss_weights
    from tt_bio.train.openfold3 import OpenFold3Dataset

    ds = OpenFold3Dataset(corpus, tokens=tokens)
    row = objectives.objective("af3")
    weights = of3_loss_weights(stage)
    prev_seed, fwd.seed = fwd.seed, seed
    out = []
    try:
        with ag.no_grad():
            for i in range(len(ds)):
                data = ds.batch([i])
                t0 = time.perf_counter()
                outputs = fwd(data)
                host = {k: (v.value if hasattr(v, "value") else v)
                        for k, v in outputs.items()}
                from tt_bio.train.tensors import to_host
                loss, terms, _ = row(data, {k: to_host(v) for k, v in host.items()},
                                     weights=weights)
                out.append({"index": i, "pdb_id": data.get("pdb_id"),
                            "loss": float(loss), "s": round(time.perf_counter() - t0, 3),
                            "breakdown": {k: (v.get("contribution")
                                              if isinstance(v, dict) else v)
                                          for k, v in terms.items()}})
                print(f"    [eval] {i} {data.get('pdb_id')} loss {float(loss):.6f} "
                      f"{out[-1]['s']:.1f}s", flush=True)
    finally:
        fwd.seed = prev_seed
    mean = sum(o["loss"] for o in out) / len(out) if out else float("nan")
    return {"mean_loss": mean, "n": len(out), "seed": seed, "stage": stage, "targets": out}


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
    ap.add_argument("--displacement-band", default="",
                    help="lo,hi for the step control. The shipped band is (0.9, 1.1) and a "
                         "device-resident bf16 arm can sit under it honestly: at 2 steps and "
                         "lr 3e-4 the ratio read 0.7468, the master moving 2.9485 where the "
                         "weight the forward reads moved 2.2021. The ratio is recorded "
                         "whatever the band, and both arms carry the same one")
    ap.add_argument("--stage", default="initial_training",
                    help="the loss weights the objective scores with, train and eval alike")
    ap.add_argument("--exact-scope", choices=("all", "trunk"), default="all",
                    help="where --exact on is allowed to reach. trunk keeps it to the taped "
                         "trunk, the section the campaign tensor clause was measured on, and "
                         "runs the diffusion half device-native in BOTH arms. The instrument "
                         "cannot run the diffusion decoder today: it norms a buffer pad_dim "
                         "has already deallocated, four arms dead at ~400 s")
    ap.add_argument("--rollout", type=int, default=20)
    ap.add_argument("--num-cycles", type=int, default=1)
    ap.add_argument("--checkpoint-every", type=int, default=10)
    ap.add_argument("--out-dir", required=True, type=Path, help="checkpoints and provenance")
    ap.add_argument("--eval-corpus", type=Path,
                    help="the HELD-OUT corpus. Scored once before the first step and once "
                         "after the last, in this process, with the trained weights still on "
                         "the card -- no checkpoint round trip to get wrong. The before score "
                         "is the movement control: two arms that never moved agree perfectly, "
                         "so a grade is only readable if the arm moved further than its floor")
    ap.add_argument("--eval-seed", type=int, default=20260926,
                    help="fixes the diffusion draw of the evaluation, so the metric is a "
                         "function of the weights alone")
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
                                      rollout=a.rollout, num_cycles=a.num_cycles, seed=a.seed,
                                      exact_scope=a.exact_scope)
                rec["exact_scope"] = a.exact_scope
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

                band = (tuple(float(x) for x in a.displacement_band.split(","))
                        if a.displacement_band else None)
                rec["displacement_band"] = band
                if a.eval_corpus:
                    print("[eval] before the first step", flush=True)
                    rec["eval_before"] = _evaluate(fwd, a.eval_corpus, a.stage, a.eval_seed)
                    dump()
                run = train_loop(fwd, ds, out_dir=a.out_dir, global_batch=a.global_batch,
                                 steps=a.steps, train="weights", seed=a.seed, lr=a.lr,
                                 warmup_steps=a.warmup_steps,
                                 checkpoint_every=a.checkpoint_every, on_step=on_step,
                                 displacement_band=band)
            if a.eval_corpus:
                print("[eval] after the last step", flush=True)
                rec["eval_after"] = _evaluate(fwd, a.eval_corpus, a.stage, a.eval_seed)
                rec["eval_delta"] = (rec["eval_after"]["mean_loss"]
                                     - rec["eval_before"]["mean_loss"])
                dump()
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
