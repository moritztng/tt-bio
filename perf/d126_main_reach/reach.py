#!/usr/bin/env python3
"""Does main's SHIPPED training recipe train past step 1?

D126's claim -- "a training run computes one gradient and then exactly zero forever" -- was
measured on `wk/of3t` and restated as `main`. This answers it on main BY EXECUTION, on a real
card, through `tt_bio.train.finetune`, which is main's only training route.

Nothing here edits shipped code. Every reading is a probe installed from outside:

* `recipes.backward` is wrapped to count the tape the backward actually reached. It is patched
  on `recipes`, not on `autograd`, because `recipes.py` did `from ..autograd import backward`
  and so holds its own reference.
* `AdamW.step` is wrapped to read, per step, how many of the parameter set's leaves carry a
  gradient, how many the identity-keyed `_PARAMS` registry still resolves, and how far the
  master and the device weight have travelled.
* `autograd._taped_linear` is wrapped to record the handle the FORWARD reads for each weight.
  That is the one reading `check_displacement` cannot take: displacement compares the master
  against `t.value`, so a forward left behind on a stale handle passes it.

The model is `perf/train_d_dp/model.py`, the same real-ttnn trunk the launcher benchmark uses:
it routes through `tt_bio.ops.linear`, so the census finds its sites and the tape records a
genuine backward. Sized down here because the question is reach, not throughput.
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def aiclk():
    """Every chip's AICLK right now, in MHz.

    Read from `smbus_telem.AICLK`, which is a HEX STRING (`'0x320'` = 800 MHz) and not a
    number: a sampler that types the field before coercing it drops every sample and then
    reports an instrument failure. `board_info.aiclk` is absent on this build.
    """
    try:
        out = subprocess.run([os.path.expanduser("~/.local/bin/tt-smi"), "-s"],
                             capture_output=True, text=True, timeout=60).stdout
        d = json.loads(out)
        return [int(str(c["smbus_telem"]["AICLK"]).strip(), 16) for c in d["device_info"]]
    except Exception as e:
        return f"unavailable: {e}"


class ClockSampler(threading.Thread):
    """AICLK sampled DURING the work. A clock read before the run is not the run's clock."""

    daemon = True

    def __init__(self, period=0.25):
        super().__init__()
        self.period, self.samples = period, []
        self._done = threading.Event()

    def run(self):
        while not self._done.wait(self.period):
            c = aiclk()
            if isinstance(c, list):
                self.samples.append(c)

    def stop(self):
        self._done.set()
        self.join(timeout=10)
        return self.samples

    def summary(self, card=0):
        v = [s[card] for s in self.samples if len(s) > card]
        if not v:
            return {"samples": 0}
        return {"samples": len(v), "min": min(v), "max": max(v),
                "median": sorted(v)[len(v) // 2]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--channels", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--examples", type=int, default=16)
    ap.add_argument("--train", default="adapters", choices=("adapters", "weights"))
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume-from", default=None,
                    help="checkpoint dir to load into the parameter set before stepping")
    ap.add_argument("--break-rekey", action="store_true",
                    help="negative control: strand the registry AND the model's own handle, "
                         "so the forward is left on the pre-step weights")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default=str(ROOT / "perf" / "d126_main_reach" / "out"))
    a = ap.parse_args()

    import numpy as np
    from tt_bio import train
    from tt_bio.train import recipes, optim, lora
    import tt_bio.autograd as ag
    from perf.train_d_dp import model as M

    tag = a.tag or f"{a.train}_s{a.steps}"
    probe, cur, prev_handles, written = [], {}, {}, {}

    # ---- the tape the backward reached -------------------------------------------------
    _real_backward = recipes.backward

    def backward_probe(roots, seeds=None):
        rs = [roots] if isinstance(roots, ag.Tensor) else list(roots)
        order = ag._reverse_topo(rs)
        cur["tape_order"] = len(order)
        cur["tape_nodes"] = sum(1 for t in order if t.node is not None)
        cur["tape_roots"] = len(rs)
        return _real_backward(roots, seeds)

    recipes.backward = backward_probe

    # ---- the handle the FORWARD reads for each weight ----------------------------------
    _real_taped_linear = ag._taped_linear

    def taped_linear_probe(shipped, args, kwargs):
        w = args[1] if len(args) >= 2 else kwargs.get("w")
        if isinstance(w, ag.Tensor):
            cur.setdefault("fwd_leaf_value_ids", []).append(id(w.value))
            cur.setdefault("fwd_leaf_ids", []).append(id(w))
        else:
            cur["fwd_raw_weights"] = cur.get("fwd_raw_weights", 0) + 1
        return _real_taped_linear(shipped, args, kwargs)

    ag._taped_linear = taped_linear_probe
    ag._TAPED["linear"] = taped_linear_probe

    # ---- per-step optimizer reading ----------------------------------------------------
    _real_step = optim.AdamW.step

    def step_probe(self, *, replicas=None):
        e = {"step_index": self.steps + 1, "leaves": len(self.params)}
        e["leaves_with_grad"] = sum(1 for t in self.params.values() if t.grad is not None)
        # The D126 reach metric itself. `parameter_for(raw)` on wk/of3t is exactly this
        # lookup; on main the accessor does not exist, so the lookup is spelled out.
        e["registry_size"] = len(ag._PARAMS)
        e["registry_resolves_leaf"] = sum(
            1 for t in self.params.values() if ag._PARAMS.get(id(t.value)) is t)
        # Which handle the forward that just ran actually read. Scored against what the
        # optimizer WROTE at the end of the previous step, not against `t.value` now: a
        # comparison to the current slot is true by construction even when something has
        # since put the leaf back on a stale handle, which is exactly the failure being
        # tested. `written` is empty on step 1, so the column reads 0/N there by definition.
        fwd = set(cur.get("fwd_leaf_value_ids", []))
        e["fwd_weight_handles"] = len(fwd)
        e["fwd_handles_are_current_leaf_values"] = len(
            fwd & {id(t.value) for t in self.params.values()})
        e["fwd_handles_are_last_written"] = len(fwd & set(written.values())) if written else 0
        e["last_written"] = len(written)
        e["fwd_raw_weights"] = cur.get("fwd_raw_weights", 0)
        e["tape_order"] = cur.get("tape_order")
        e["tape_nodes"] = cur.get("tape_nodes")

        r = _real_step(self, replicas=replicas)
        # Captured HERE, before the break control can touch anything: `written` has to mean
        # "the handle the optimizer wrote". Recording it after the revert would record the
        # stale handle, the next forward would match it, and the control could not fail.
        for n, t in self.params.items():
            written[n] = id(t.value)

        e["grad_norm"] = self.last_grad_norm
        e["lr"] = self.last_lr
        e["clip"] = self.last_clip
        d = self.displacement()
        e["master_disp"] = d["master"]
        e["device_disp"] = d["device"]
        e["disp_ratio"] = d["ratio"]
        e["registry_resolves_leaf_after_step"] = sum(
            1 for t in self.params.values() if ag._PARAMS.get(id(t.value)) is t)
        e["per_param_master_step"] = {
            n: v["master_step"] for n, v in (self.last_report or {}).items()}
        e["stepped_params"] = len(self.last_report or {})
        e["master_sha"] = int(sum(float(np.abs(v).sum()) for v in self.master.values()) * 1e6)

        if a.break_rekey:
            # Negative control. Put every leaf back on the handle it held BEFORE this step,
            # which is what a run whose updates never reach the forward looks like. The
            # master still moves; the weight the forward reads does not. Held in a dict
            # beside the tensors rather than on them: `ag.Tensor` has `__slots__`, so an
            # attribute probe would not attach at all.
            for n, t in self.params.items():
                if n in prev_handles:
                    t.value = prev_handles[n]
        for n, t in self.params.items():
            prev_handles[n] = t.value

        probe.append(e)
        cur.clear()
        return r

    optim.AdamW.step = step_probe

    # ---- resume, the Tier-2 way -------------------------------------------------------
    # The shipped recipe has no `resume` argument, so a resumed run is a user calling
    # `checkpoint.load_adapter` on the parameter set the census just built and then running
    # the loop. Injected at exactly that seam: `trainable` is patched on `recipes`, which
    # imported it by name, so the SHIPPED loop below runs on resumed parameters.
    resume_info = {}
    if a.resume_from:
        from tt_bio.train.checkpoint import load_adapter
        _real_trainable = recipes.trainable

        def trainable_resume(forward, cfg, device, *args, **kw):
            installed, params = _real_trainable(forward, cfg, device, *args, **kw)
            before = {n: id(t.value) for n, t in params.items()}
            resume_info["meta"] = load_adapter(a.resume_from, params, device)
            resume_info["names"] = sorted(params)
            resume_info["handles_replaced"] = sum(
                1 for n, t in params.items() if id(t.value) != before[n])
            resume_info["registry_resolves_after_load"] = sum(
                1 for t in params.values() if ag._PARAMS.get(id(t.value)) is t)
            return installed, params

        recipes.trainable = trainable_resume

    clk_before = aiclk()
    sampler = ClockSampler()
    sampler.start()
    t0 = time.perf_counter()
    err = None
    run = None
    try:
        run = train.finetune(
            M.Trunk(a.tokens, a.channels, a.blocks, seed=a.seed),
            M.Dataset(a.examples, a.tokens, a.channels, seed=a.seed),
            out_dir=str(Path(a.out) / tag / "ckpt"),
            global_batch=1, steps=a.steps, objective="af3", weights={"mse": 1.0},
            train=a.train, seed=a.seed, lr=a.lr, warmup_steps=2,
            checkpoint_every=max(a.steps, 1), tokens=256,
            lora=lora.LoraConfig(rank=8) if a.train == "adapters" else None)
    except BaseException as ex:                     # recorded, not swallowed: a raise IS a reading
        err = f"{type(ex).__name__}: {ex}"
    wall = time.perf_counter() - t0
    sampler.stop()
    clk_during = sampler.summary(card=0)
    clk_after = aiclk()

    rec = {
        "tag": tag, "argv": vars(a), "wall_s": wall,
        "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                  capture_output=True, text=True).stdout.strip(),
        "has_parameter_for": hasattr(ag, "parameter_for"),
        "value_is_property": isinstance(getattr(ag.Tensor, "value", None), property),
        "aiclk_before": clk_before, "aiclk_during_card0": clk_during,
        "aiclk_after": clk_after,
        "error": err, "probe": probe, "resume": resume_info,
    }
    if run is not None:
        rec["history"] = run.history
        rec["displacement"] = run.displacement
        rec["sites"] = sorted(run.params)
        rec["provenance"] = run.provenance.as_dict()
    outdir = Path(a.out) / tag
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "reach.json").write_text(json.dumps(rec, indent=2, default=str) + "\n")

    print(f"=== {tag}  sha {rec['git_sha'][:9]}  parameter_for={rec['has_parameter_for']}  "
          f"value_is_property={rec['value_is_property']}")
    print(f"    aiclk during card0 {clk_during}, before {clk_before} after "
          f"{clk_after}, wall {wall:.1f}s")
    hdr = ("step", "grad_norm", "tape_nodes", "leaves", "w/grad", "resolves",
           "fwd_stepped", "master_disp", "dev_disp", "ratio")
    print("    " + " ".join(f"{h:>12}" for h in hdr))
    for e in probe:
        print("    " + " ".join(f"{v:>12}" for v in (
            e["step_index"], f"{e['grad_norm']:.6e}", e["tape_nodes"], e["leaves"],
            e["leaves_with_grad"],
            f"{e['registry_resolves_leaf']}/{e['leaves']}",
            f"{e['fwd_handles_are_last_written']}/{e['fwd_weight_handles']}",
            f"{e['master_disp']:.4e}", f"{e['device_disp']:.4e}", f"{e['disp_ratio']:.4f}")))
    if resume_info:
        print(f"    resumed from {a.resume_from}: {resume_info['handles_replaced']}/"
              f"{len(resume_info['names'])} handles replaced by the load, registry resolves "
              f"{resume_info['registry_resolves_after_load']} after it")
    if err:
        print(f"    RAISED: {err}")
    if run is not None:
        print(f"    {run}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
