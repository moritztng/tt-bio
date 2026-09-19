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
               heads.
  diffusion    `_train_diffusion` differentiates no_samples noised structures: 48 in
               initial_training, 32 in finetune_1 and finetune_2, skipped entirely in
               finetune_3 (`train_confidence_only`).

This harness prices the TRUNK half, which is the part that runs on our stack today, and it
prices it against that structure: a no_grad prefix and one taped cycle, timed apart. The
diffusion half is excluded and the reason is recorded rather than averaged in --
`perf/of3t_perf/verb_census.json` shows 5 `ttnn.embedding` gathers with no tape entry, three
of them in `openfold3_diffusion_module.py`, so the diffusion module cannot go under a tape
until `of3t-tape` lands their backward. A total that quietly omitted it would be the wrong
number reported confidently.

THE INPUT IS THE SHIPPED ONE. Rather than reimplement `worker.py`'s OF3 prep -- which this row
may not edit and must not fork -- the harness intercepts `OF3Trunk.__call__` during a real
`predict_one` on a real fixture and keeps the device tensors the shipped pipeline built. There
is no second featurisation to drift.

THE BACKWARD IS SEEDED SYNTHETICALLY, deliberately. Our loss-weight table does not cover OF3
(LEDGER R2: `train/losses.py:63` holds Protenix constants under "pretrain"/"finetune"), so a
composed objective here would be an invention. A unit gradient on `z_trunk` prices the trunk's
backward, which is what a cost breakdown needs; it makes no claim about the objective.

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

from perf.clocksample import during  # noqa: E402

FIXTURES = REPO / "perf" / "size512" / "fixtures"


class _Captured(Exception):
    """Raised out of the intercepted trunk once its inputs are in hand."""


def capture_trunk_inputs(tokens, out):
    """Run the shipped OF3 pipeline until it calls the trunk, and keep the trunk's arguments.

    Everything upstream of the trunk -- the query parse, the MSA resolve, the featurizer, the
    input atom encoder, the token bucketing and the input glue -- is the production path. The
    interception is one attribute swap in this process; nothing in the repo changes.
    """
    import tt_baseline as B
    from tt_bio.openfold3_trunk import OF3Trunk

    held = {}
    real_call = OF3Trunk.__call__

    def intercept(self, *args, **kwargs):
        held["self"], held["args"], held["kwargs"] = self, args, kwargs
        raise _Captured

    t0 = time.perf_counter()
    tgt = FIXTURES / f"cdk2x2_{tokens}.yaml"
    a3m = FIXTURES / f"cdk2x2_{tokens}.a3m"
    if not tgt.is_file():
        raise SystemExit(f"no fixture {tgt}; sizes are "
                         f"{sorted(p.stem.split('_')[1] for p in FIXTURES.glob('*.yaml'))}")
    one_fold, meta = B.build_fold("openfold3", REPO / f".msa_of3t_{tokens}", tgt, a3m)[:2]
    out["build_fold_s"] = round(time.perf_counter() - t0, 2)

    OF3Trunk.__call__ = intercept
    t0 = time.perf_counter()
    try:
        one_fold()
    except _Captured:
        pass
    finally:
        OF3Trunk.__call__ = real_call
    out["prep_to_trunk_s"] = round(time.perf_counter() - t0, 2)
    if "self" not in held:
        raise SystemExit("the fold never reached OF3Trunk.__call__; nothing to time")
    return held, meta


def cycle_once(trunk, held, cycles, taped):
    """One trunk forward at a pinned cycle count, taped or not.

    `num_cycles` is an instance attribute the shipped trunk reads on every call, so pinning it
    here is how the harness holds the draw fixed. Their training samples it per step; a timing
    run that let it vary would report the draw's variance as the measurement's.
    """
    from tt_bio import autograd as ag
    trunk.num_cycles = cycles
    ctx = ag.tape() if taped else ag.no_grad()
    with ctx:
        return trunk(*held["args"], **held["kwargs"])


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
                       "diffusion_half": "EXCLUDED: 5 ttnn.embedding gathers have no tape "
                                         "entry (perf/of3t_perf/verb_census.json)"}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    dump = lambda: a.out.write_text(json.dumps(out, indent=1, default=str))
    dump()

    with during() as clk:
        try:
            import ttnn
            from tt_bio import autograd as ag
            from tt_bio.tenstorrent import get_device

            held, meta = capture_trunk_inputs(a.tokens, out)
            trunk = held["self"]
            dev = get_device()
            out["env"]["arch"] = str(dev.arch())
            out["dram"] = {"after_prep": _dram(dev)}
            dump()

            stages = []
            for rep in range(a.reps):
                row = {"rep": rep}
                # --- forward, no_grad prefix: cycles 0..n-2, which is what their loop runs
                # under torch.no_grad(). At cycles=1 there is no prefix and this is 0 by
                # construction, which is also a real training draw (num_recycles=0).
                if a.cycles > 1:
                    t0 = time.perf_counter()
                    s, z = cycle_once(trunk, held, a.cycles - 1, taped=False)
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
                _ = ttnn.to_torch(ag._unwrap(z) if isinstance(z, ag.Tensor) else z)
                row["download_s"] = time.perf_counter() - t0

                # --- losses: not priced here. Our weight table does not cover OF3 (LEDGER R2),
                # so the backward is seeded synthetically instead of against an invented row.
                row["losses_s"] = None
                row["losses_note"] = "SKIPPED: no OF3 row in train/losses.py (LEDGER R2)"

                # --- host_backward: building the seed
                t0 = time.perf_counter()
                import torch
                zr = ag._unwrap(z) if isinstance(z, ag.Tensor) else z
                seed = ttnn.from_torch(torch.ones(tuple(int(d) for d in zr.shape)),
                                       layout=ttnn.TILE_LAYOUT, device=dev,
                                       dtype=ttnn.bfloat16)
                row["host_backward_s"] = time.perf_counter() - t0

                # --- device_backward
                t0 = time.perf_counter()
                taped_root = isinstance(z, ag.Tensor)
                if taped_root:
                    ag.backward([z], [seed])
                    ttnn.synchronize_device(dev)
                row["device_backward_s"] = time.perf_counter() - t0
                row["root_was_taped"] = taped_root
                row["dram_after_backward"] = _dram(dev)
                ag.release_pins()

                row["step_s"] = (row["forward_nograd_prefix_s"]
                                 + row["forward_taped_cycle_s"] + row["download_s"]
                                 + row["host_backward_s"] + row["device_backward_s"])
                stages.append(row)
                out["stages"] = stages
                print(f"[rep {rep}] prefix {row['forward_nograd_prefix_s']:.2f}s  "
                      f"taped-cycle {row['forward_taped_cycle_s']:.2f}s  "
                      f"download {row['download_s']:.2f}s  "
                      f"backward {row['device_backward_s']:.2f}s  "
                      f"= {row['step_s']:.2f}s  taped={taped_root}", flush=True)
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
