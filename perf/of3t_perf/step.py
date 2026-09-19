#!/usr/bin/env python3
"""Where an OpenFold3 training step spends its time, stage by stage, before any lever.

PROTOCOL sequencing: a per-stage breakdown needs a structurally COMPLETE taped step, not a
verified-correct gradient, so this runs now. No lever may be proposed from it until
`of3t-equivalence` has passed.

WHAT THEIR STEP IS, read out of upstream by `their_step_shape.py` and not assumed:

  trunk        num_cycles = num_recycles + 1, and in TRAINING num_recycles is drawn per step
               from U{0..3} (model.py, `synced_generator.integers`). Only the FINAL cycle
               carries a gradient; the earlier ones run under no_grad. So a training step's
               trunk is a no_grad prefix of random length followed by one taped cycle, and a
               step time quoted without the drawn cycle count is a sample from a distribution.
  mini rollout 20 diffusion steps, entirely under `torch.no_grad()`, feeding the confidence
               heads. No tape is involved, so our shipped sampler runs it as-is and this
               harness prices it exactly rather than excluding it.
  diffusion    `_train_diffusion` differentiates no_samples noised structures: 48 in
               initial_training, 32 in finetune_1 and finetune_2, skipped entirely in
               finetune_3 (`train_confidence_only`). This one IS excluded, and the reason is
               recorded rather than averaged in: `perf/of3t_perf/verb_census.json` shows 5
               `ttnn.embedding` gathers with no tape entry, three of them in
               `openfold3_diffusion_module.py`, so the diffusion module cannot go under a tape
               until `of3t-tape` lands their backward.

THE INPUT IS THE SHIPPED ONE. Rather than reimplement `worker.py`'s OF3 prep -- which this row
may not edit and must not fork -- the harness intercepts a real `predict_one` on a real fixture
at two points and keeps the device tensors the shipped pipeline built: `OF3Trunk.__call__` on
the way past (recorded, then handed straight to the real trunk so the fold carries on) and
`OF3SampleDiffusion.__call__`, where it stops. There is no second featurisation to drift.

THE WEIGHTS ARE DECLARED, and this is the instrument's own failure mode. `tape()` recognises a
trainable weight by the IDENTITY of its raw handle (`autograd.parameter`), so a taped forward
whose weights were never declared still runs, still returns, and its backward walks a graph
with no weight leaves in it -- a fast number for work nobody asked for. `perf/ptx_integrate`
hit exactly that ("the module is in the forward without being in the step -- which is what the
first run of this harness did"). So the weights are walked off the shipped trunk off the shipped module, declared, and the backward reports how many of them RECEIVED a
gradient out of how many were declared. If that ratio is 0 the stage is marked INVALID rather
than reported.

THE BACKWARD IS SEEDED SYNTHETICALLY, deliberately. Our loss-weight table does not cover OF3
(LEDGER R2: `train/losses.py:63` holds Protenix constants under "pretrain"/"finetune"), so a
composed objective here would be an invention. A unit gradient on `z_trunk` prices the trunk's
backward, which is what a cost breakdown needs; it makes no claim about the objective.

THE ROLLOUT IS AN n-LADDER, not a bracket. A single 20-step timing charges the rollout's fixed
cost (conditioning, the pair branch, the DiT bias cache -- all loop invariants the sampler
hoists) to the 20 steps, and `synced-bracket-inflates-op-level-fixed-cost` is the fleet's
standing lesson that a bracketed total is not a per-unit rate. The ladder fits
seconds = fixed + n * per_step over several rung lengths and reports both.

    step.py --tokens 384 --reps 3 --out perf/of3t_perf/stages_384_qb2c1.json
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "gpu_vs_tt"))

from perf.clocksample import during                      # noqa: E402

FIXTURES = REPO / "perf" / "size512" / "fixtures"

# Their mini rollout is 20 steps (`their_step_shape.py`). The ladder brackets it rather than
# extrapolating past it: the fit is only used inside the range it was measured over.
LADDER = (4, 12, 20)


class _Captured(Exception):
    """Raised out of the intercepted sampler once the step's inputs are all in hand."""


def capture(tokens, out):
    """Run the shipped OF3 pipeline to the sampler, keeping the trunk's and sampler's inputs.

    Everything upstream -- the query parse, the MSA resolve, the featurizer, the input atom
    encoder, the token bucketing and the input glue -- is the production path. The
    interception is two attribute swaps in this process; nothing in the repo changes.
    """
    import tt_baseline as B
    from tt_bio.openfold3_trunk import OF3Trunk
    from tt_bio.openfold3_sample_diffusion import OF3SampleDiffusion

    held = {}
    real_trunk, real_sampler = OF3Trunk.__call__, OF3SampleDiffusion.__call__

    def trunk_intercept(self, *args, **kwargs):
        # Record and CARRY ON: the sampler's inputs are downstream of a real trunk output, so
        # stopping here would leave half the step uncaptured.
        held["trunk"] = (self, args, kwargs)
        t0 = time.perf_counter()
        r = real_trunk(self, *args, **kwargs)
        held["capture_trunk_s"] = time.perf_counter() - t0
        held["capture_trunk_cycles"] = int(self.num_cycles)
        return r

    def sampler_intercept(self, *args, **kwargs):
        held["sampler"] = (self, args, kwargs)
        raise _Captured

    t0 = time.perf_counter()
    tgt = FIXTURES / f"cdk2x2_{tokens}.yaml"
    a3m = FIXTURES / f"cdk2x2_{tokens}.a3m"
    if not tgt.is_file():
        raise SystemExit(f"no fixture {tgt}; sizes are "
                         f"{sorted(p.stem.split('_')[1] for p in FIXTURES.glob('*.yaml'))}")
    one_fold, meta = B.build_fold("openfold3", REPO / f".msa_of3t_{tokens}", tgt, a3m)[:2]
    out["build_fold_s"] = round(time.perf_counter() - t0, 2)

    OF3Trunk.__call__, OF3SampleDiffusion.__call__ = trunk_intercept, sampler_intercept
    t0 = time.perf_counter()
    try:
        one_fold()
    except _Captured:
        pass
    finally:
        OF3Trunk.__call__, OF3SampleDiffusion.__call__ = real_trunk, real_sampler
    out["prep_to_sampler_s"] = round(time.perf_counter() - t0, 2)
    if "trunk" not in held:
        raise SystemExit("the fold never reached OF3Trunk.__call__; nothing to time")
    out["capture"] = {"trunk_forward_s": round(held.get("capture_trunk_s", 0.0), 3),
                      "trunk_cycles": held.get("capture_trunk_cycles"),
                      "reached_sampler": "sampler" in held}
    return held, meta


# How deep the weight walk goes. `perf/ptx_integrate/step.py`'s `device_weights` is the same
# walk and would be the thing to import, but it caps at depth 4, which is one short of OF3:
# `trunk.pairformer.blocks[i].tri_mul_out.linear_a_g` is depth 5 and the whole pairformer
# stack -- the bulk of the trunk's weights -- sits below the cap. That row's file is not this
# row's to edit, so the depth lives here and `walk_depth_reached` is reported beside the count
# so a walk that bottomed out is visible instead of just small.
WALK_DEPTH = 8


def walk_weights(obj, prefix="", seen=None, out=None, depth=0, stat=None):
    """Every device weight tensor reachable from a shipped module, by attribute path.

    Walked rather than listed: the parameter set comes from the model, so a module that gains
    a tensor next week is in it. A weight that is in the forward but NOT in the trainable set
    is invisible -- the backward simply does less work and reports a smaller number.
    """
    import ttnn
    out = {} if out is None else out
    seen = set() if seen is None else seen
    stat = {"max_depth": 0, "truncated": 0} if stat is None else stat
    stat["max_depth"] = max(stat["max_depth"], depth)
    if depth > WALK_DEPTH:
        stat["truncated"] += 1
        return out, stat
    if id(obj) in seen:
        return out, stat
    seen.add(id(obj))
    items = obj.items() if isinstance(obj, dict) else (
        list(enumerate(obj)) if isinstance(obj, (list, tuple)) else
        vars(obj).items() if hasattr(obj, "__dict__") else [])
    for k, v in items:
        name = f"{prefix}{k}"
        if isinstance(v, ttnn.Tensor):
            out[name] = (obj, k, v)
        elif isinstance(v, (list, tuple, dict)) or hasattr(v, "__dict__"):
            walk_weights(v, name + ".", seen, out, depth + 1, stat)
    return out, stat


def declare_weights(trunk, out):
    """Declare every device weight the shipped trunk holds a tape leaf.

    `tape()` recognises a trainable weight by the IDENTITY of its raw handle, so this is the
    difference between a backward that computes the trunk's weight gradients and one that
    walks activations to the inputs and stops. The count is reported, and so is a structural
    check that the walk actually descended into the pairformer stack -- a walk that stopped at
    the top level would declare a handful of glue weights, run, and report a small number.
    """
    from tt_bio import autograd as ag
    found, stat = walk_weights(trunk, prefix="trunk.")
    params = {n: ag.parameter(t) for n, (_o, _k, t) in sorted(found.items())}
    deep = sorted(n for n in params if ".pairformer." in n)
    out["params"] = {"declared": len(params),
                     "elements": sum(_numel(t.value) for t in params.values()),
                     "by_rank": _rank_hist(params),
                     "walk_depth_reached": stat["max_depth"],
                     "walk_truncated_at_limit": stat["truncated"],
                     "pairformer_weights": len(deep),
                     "deepest_name": max(params, key=lambda n: n.count(".")) if params else None}
    if not deep:
        out["params"]["WARNING"] = ("the walk found no pairformer weight -- the trunk's bulk "
                                    "is not in the trainable set and every backward number "
                                    "below is about a smaller model than the one that ran")
    print(f"[weights] declared {len(params)} tensors, "
          f"{out['params']['elements'] / 1e6:.1f}M elements, "
          f"{len(deep)} in the pairformer stack, walk depth {stat['max_depth']}", flush=True)
    return params


def _numel(t):
    n = 1
    for d in t.shape:
        n *= int(d)
    return n


def _rank_hist(params):
    h = {}
    for t in params.values():
        h[str(len(t.value.shape))] = h.get(str(len(t.value.shape)), 0) + 1
    return h


def cycle_once(trunk, held, cycles, taped):
    """One trunk forward at a pinned cycle count, taped or not.

    `num_cycles` is an instance attribute the shipped trunk reads on every call, so pinning it
    here is how the harness holds the draw fixed. Their training samples it per step; a timing
    run that let it vary would report the draw's variance as the measurement's.

    Both arms start from the same captured inputs, so the taped cycle does not consume the
    prefix's recycled state. Shapes and kernels are identical either way, which is what a cost
    breakdown reads; the numerical coupling is `of3t-equivalence`'s business, not this file's.
    """
    from tt_bio import autograd as ag
    self_, args, kwargs = held["trunk"]
    trunk.num_cycles = cycles
    ctx = ag.tape() if taped else ag.no_grad()
    with ctx:
        return trunk(*args, **kwargs)


def rollout_ladder(held, dev, out, rungs=LADDER):
    """Time their 20-step mini rollout, and separate its fixed cost from its per-step cost.

    Their mini rollout runs entirely under `torch.no_grad()`, so it needs no tape at all and
    our shipped sampler runs it unmodified. The sampler's loop is `for tau in
    range(len(t_list))`, so a rung is the captured call with the five per-step lists truncated
    -- the same kernels, the same conditioning, fewer steps.
    """
    import ttnn
    from tt_bio import autograd as ag
    if "sampler" not in held:
        return {"note": "the fold never reached the sampler; rollout not priced"}
    self_, args, kwargs = held["sampler"]
    args = list(args)
    # The five per-step artefacts (rots / trans / noise / t / c_tau) are python lists of one
    # entry per step; `noise_schedule` is a torch tensor of n+1 and the body never reads it.
    # So the rungs are "every list argument, truncated", found by type rather than by index --
    # a positional index here would go stale the first time the signature gains an argument.
    list_idx = [i for i, v in enumerate(args) if isinstance(v, (list, tuple))]
    full = max((len(args[i]) for i in list_idx), default=0)
    if full == 0:
        return {"note": "the sampler call carries no per-step list; rollout not priced"}
    dead = [i for i, v in enumerate(args)
            if hasattr(v, "is_allocated") and not v.is_allocated()]
    if dead:
        return {"note": f"captured sampler inputs {dead} were freed on the way out; "
                        f"rollout not priced"}
    rows = []
    for n in rungs:
        if n > full:
            continue
        rung = list(args)
        for i in list_idx:
            v = args[i]
            rung[i] = v[:n] if len(v) == full else v
        t0 = time.perf_counter()
        with ag.no_grad():
            xl = self_(*rung, **{k: v for k, v in kwargs.items() if k != "progress_fn"})
        ttnn.synchronize_device(dev)
        s = time.perf_counter() - t0
        if hasattr(xl, "is_allocated") and xl.is_allocated():
            ttnn.deallocate(xl)
        rows.append({"steps": n, "s": round(s, 3)})
        print(f"  [rollout] {n:>3} steps  {s:7.3f}s", flush=True)
    r = {"rungs": rows, "full_schedule_steps": full}
    if len(rows) >= 2:
        (n0, s0), (n1, s1) = (rows[0]["steps"], rows[0]["s"]), (rows[-1]["steps"], rows[-1]["s"])
        per = (s1 - s0) / (n1 - n0)
        r["per_step_s"] = round(per, 4)
        r["fixed_s"] = round(s0 - per * n0, 3)
        r["their_mini_rollout_20_s"] = round(r["fixed_s"] + 20 * per, 3)
    return r


def _dram(dev):
    import ttnn
    mv = ttnn.get_memory_view(dev, ttnn.BufferType.DRAM)
    return int(mv.total_bytes_allocated_per_bank) * int(mv.num_banks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=384,
                    help="their training crops are 384 / 640 / 768 and nothing larger")
    ap.add_argument("--cycles", type=int, default=4,
                    help="trunk cycles to PIN. Their training draws this from U{1..4} per "
                         "step; pin it so the timing is not reading the draw")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--no-rollout", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("perf/of3t_perf/stages.json"))
    a = ap.parse_args()

    out = {"doc": __doc__.split("\n\n")[0], "argv": sys.argv[1:], "env": {
        "host": socket.gethostname(),
        "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
        "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "loadavg": os.getloadavg()},
        "their_step": {"per_rank_batch": 1, "crop": a.tokens,
                       "trunk_cycles_pinned": a.cycles,
                       "grad_on_final_cycle_only": True,
                       "mini_rollout_steps": 20,
                       "diffusion_half": "EXCLUDED: 5 ttnn.embedding gathers have no tape "
                                         "entry (perf/of3t_perf/verb_census.json)"}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    dump = lambda: a.out.write_text(json.dumps(out, indent=1, default=str))
    dump()

    with during() as clk:
        try:
            import torch
            import ttnn
            from tt_bio import autograd as ag
            from tt_bio.tenstorrent import get_device

            held, meta = capture(a.tokens, out)
            trunk = held["trunk"][0]
            dev = get_device()
            out["env"]["arch"] = str(dev.arch())
            out["dram"] = {"after_prep": _dram(dev)}
            params = declare_weights(trunk, out)
            dump()

            stages = []
            for rep in range(a.reps):
                row = {"rep": rep}
                for t in params.values():
                    t.grad = None
                # --- forward, no_grad prefix: cycles 0..n-2, which is what their loop runs
                # under torch.no_grad(). At cycles=1 there is no prefix and this is 0 by
                # construction, which is also a real training draw (num_recycles=0).
                if a.cycles > 1:
                    t0 = time.perf_counter()
                    cycle_once(trunk, held, a.cycles - 1, taped=False)
                    ttnn.synchronize_device(dev)
                    row["forward_nograd_prefix_s"] = time.perf_counter() - t0
                    ag.release_pins()
                else:
                    row["forward_nograd_prefix_s"] = 0.0

                # --- forward, the one taped cycle
                t0 = time.perf_counter()
                s, z = cycle_once(trunk, held, 1, taped=True)
                ttnn.synchronize_device(dev)
                row["forward_taped_cycle_s"] = time.perf_counter() - t0
                row["dram_after_taped_forward"] = _dram(dev)

                # --- download: the trunk leaving the device, which their step also pays
                t0 = time.perf_counter()
                zr = ag._unwrap(z) if isinstance(z, ag.Tensor) else z
                _ = ttnn.to_torch(zr)
                row["download_s"] = time.perf_counter() - t0

                # --- losses: not priced here. Our weight table does not cover OF3 (LEDGER R2),
                # so the backward is seeded synthetically instead of against an invented row.
                row["losses_s"] = None
                row["losses_note"] = "SKIPPED: no OF3 row in train/losses.py (LEDGER R2)"

                # --- host_backward: building the seed
                t0 = time.perf_counter()
                seed = ttnn.from_torch(torch.ones(tuple(int(d) for d in zr.shape)),
                                       layout=ttnn.TILE_LAYOUT, device=dev,
                                       dtype=zr.dtype)
                row["host_backward_s"] = time.perf_counter() - t0

                # --- device_backward, with the instrument's own integrity check beside it.
                row["root_was_taped"] = taped_root = isinstance(z, ag.Tensor)
                if taped_root:
                    row["tape_nodes_from_root"] = len(ag._reverse_topo([z]))
                    t0 = time.perf_counter()
                    ag.backward([z], [seed])
                    ttnn.synchronize_device(dev)
                    row["device_backward_s"] = time.perf_counter() - t0
                else:
                    row["tape_nodes_from_root"] = 0
                    row["device_backward_s"] = 0.0
                got = sum(1 for t in params.values() if getattr(t, "grad", None) is not None)
                row["params_with_grad"] = f"{got} of {len(params)}"
                row["backward_valid"] = bool(taped_root and got)
                if not row["backward_valid"]:
                    row["backward_note"] = (
                        "INVALID -- the backward reached no declared weight, so this is the "
                        "cost of an empty graph, not of a training step's backward")
                row["dram_after_backward"] = _dram(dev)
                ag.release_pins()

                row["trunk_step_s"] = (row["forward_nograd_prefix_s"]
                                       + row["forward_taped_cycle_s"] + row["download_s"]
                                       + row["host_backward_s"] + row["device_backward_s"])
                stages.append(row)
                out["stages"] = stages
                print(f"[rep {rep}] prefix {row['forward_nograd_prefix_s']:.2f}s  "
                      f"taped-cycle {row['forward_taped_cycle_s']:.2f}s  "
                      f"download {row['download_s']:.2f}s  "
                      f"backward {row['device_backward_s']:.2f}s "
                      f"({row['params_with_grad']} weights, "
                      f"{row['tape_nodes_from_root']} nodes)  "
                      f"= {row['trunk_step_s']:.2f}s", flush=True)
                dump()

            if not a.no_rollout:
                out["rollout"] = rollout_ladder(held, dev, out)
                dump()
        except Exception:
            out["error"] = traceback.format_exc()
            print(out["error"], flush=True)
        out["clock"] = clk.summary() if hasattr(clk, "summary") else None
    dump()
    print("WROTE", a.out)
    return 0 if "error" not in out else 1


if __name__ == "__main__":
    sys.exit(main())
