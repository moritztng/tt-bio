#!/usr/bin/env python3
"""Float64 grade of BindCraft 2's gradient round, one arm per process, for any architecture.

`perf/bcp_evo/stack_grade.py` graded the round's levers on Blackhole: `perf/bcx_afgrad` `stack
--ckpt`, the whole 4 + 48 block stack checkpointed under one tape from sequence logits, dL/dlogits
against float64 autograd on the same stack, a float64 finite difference along the device gradient,
repeat, zero-seed and permuted controls, all inside `bindcraft2.fast_round()`. It toggled one
lever in-process. This runs the same grade with whatever the environment arms, so an arm is a set
of the levers' own env vars and nothing in the process has to know which lever it is grading:

    TT_VISIBLE_DEVICES=5 python perf/spd/bc2_stack_grade.py --arm base
    TT_VISIBLE_DEVICES=5 TT_BIO_PAIR_MM=1 ... python perf/spd/bc2_stack_grade.py --arm bhlev

`fast_round` reads every lever's env var when it arms, and a named env var wins over the Wormhole
default it would otherwise keep. An arm passes when its device-vs-float64 rel L2 is no worse than
the base arm's on the same chip, the finite difference agrees, and the controls hold.
"""
import argparse
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from perf.bcx_afgrad import afgrad as A  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True)
ap.add_argument("--n", type=int, default=128)
ap.add_argument("--params", default=A.DEFAULT_PARAMS)
ap.add_argument("--threads", type=int, default=8)
a = ap.parse_args()
# One directory per arm, so arms can run side by side on one tree without renaming each other's output.
A.OUT = A.OUT / f"spd_bc2_{a.arm}"

# afgrad's own main() sets this; cmd_stack does not, and an unthrottled float64 graph took 13 cores of
# a shared Galaxy host whose other rows' folds are host-sensitive.
A.torch.set_num_threads(a.threads)
from tt_bio.main import ensure_p300_mesh_descriptor  # noqa: E402
ensure_p300_mesh_descriptor()
from tt_bio import bindcraft2  # noqa: E402

ns = argparse.Namespace(params=a.params, card=int(os.environ.get("TT_VISIBLE_DEVICES", "0")), n=a.n,
                        extra=4, evo=48, seed=0, controls_only=False, eps="1e-1,3e-2,1e-2,3e-3,1e-3",
                        ckpt=True, memory="fast", msa_mask=False, threads=a.threads)
stem = f"stack_n{a.n}_e4_v48_ckpt"
with bindcraft2.fast_round() as armed:
    print(f"== arm {a.arm} armed {json.dumps(armed, default=str)}", flush=True)
    A.cmd_stack(ns)
    for ext in (".json", ".pt"):
        src = A.OUT / (stem + ext)
        if src.exists():
            src.rename(A.OUT / f"{stem}_spd_bc2_{a.arm}{ext}")
d = json.loads((A.OUT / f"{stem}_spd_bc2_{a.arm}.json").read_text())
d["armed"] = {k: str(v) for k, v in armed.items()}
d["env"] = {k: v for k, v in sorted(os.environ.items()) if k.startswith("TT_BIO_")}
(A.OUT / f"{stem}_spd_bc2_{a.arm}.json").write_text(json.dumps(d, indent=1, default=str))
print(f"== arm {a.arm} device_vs_f64 {d['device_vs_f64']} loss dev {d['device']['loss']} "
      f"f64 {d['f64']['loss']} fd {d.get('fd_best', {}).get('ratio_fd_over_device')} "
      f"timing {d['device']['timing']}", flush=True)
