"""What the gradient all-reduce costs a replicate-sharded step, priced without a card.

The chips produce the numbers; the collective itself is /dev/shm and a numpy sum on the host
(`tt_bio.train.launcher.ProcessAxis`), so it measures honestly with no device open and no
AICLK to record. What it answers is the one term `of3t-p10dp`'s N=2 arithmetic left as "plus
the all-reduce", and whether naming only the replicate half of the parameter set is worth the
ordering constraint it imposes (`replicate_dp.reduce_grads`, `names=`).

Payload sizes are OpenFold3 crop 384, read out of this harness's own banked artifacts rather
than guessed: 381,302,188 declared elements over 3152 tensors
(`perf/of3t_stepfloor/out/step_rekey_384.json`), of which the trunk walk is 165,180,992 over
2531 (`out/d164_probeoff_384.json`). The two sets are disjoint -- 896 diffusion tensors found,
275 dropped as duplicates internal to that walk, 896-275 = 621 = the artifact's
`diffusion_weights` -- so the replicate half is 216,121,196 elements over 621 tensors.

    python3 perf/of3t_p10dp/reduce_cost.py
"""
import json
import multiprocessing as mp
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO)
from tt_bio.train.launcher import ProcessAxis          # noqa: E402
from tt_bio.train.mesh import Mesh                     # noqa: E402
from tt_bio.train import replicate_dp as rdp           # noqa: E402

TRUNK_ELEMENTS, DECLARED_ELEMENTS = 165_180_992, 381_302_188
DIFFUSION_ELEMENTS = DECLARED_ELEMENTS - TRUNK_ELEMENTS
DIFFUSION_TENSORS, DECLARED_TENSORS = 621, 3152
REPS = 5
OUT = Path(__file__).resolve().parent / "out" / "reduce_cost.json"


class _P:
    """Just enough of a parameter for `reduce_grads`: a gradient and a shape off the handle."""
    def __init__(self, g):
        self.grad = g

    @property
    def shape(self):
        return (0,) if self.grad is None else self.grad.shape


def _host():
    with open("/proc/meminfo") as f:
        mem = next(l for l in f if l.startswith("MemAvailable")).split()[1]
    return {"loadavg": [round(x, 2) for x in os.getloadavg()],
            "mem_available_mb": int(mem) // 1024}


def _rank(rank, world, run, elements, ntensors, q):
    axis = ProcessAxis(name="dp", device_ids=tuple(range(world)),
                       mesh=Mesh({"dp": list(range(world))}), dp_rank=rank, run=run)
    per = elements // ntensors
    names = [f"p{i:05d}" for i in range(ntensors)]
    params = {n: _P(None) for n in names}
    ts = []
    for _ in range(REPS):
        for n in names:                       # rebuilt per rep: reduce_grads writes the sum back
            params[n].grad = np.full(per, rank + 1.0, np.float32)
        t0 = time.perf_counter()
        moved = rdp.reduce_grads(axis, params)
        ts.append(round(time.perf_counter() - t0, 3))
    q.put((rank, ts, moved))


def run(world, elements, ntensors, label, results):
    ctx = mp.get_context("spawn")
    with tempfile.TemporaryDirectory(dir="/dev/shm") as run_dir:
        q = ctx.Queue()
        ps = [ctx.Process(target=_rank, args=(r, world, run_dir, elements, ntensors, q))
              for r in range(world)]
        for p in ps:
            p.start()
        got = dict((r, (ts, moved)) for r, ts, moved in
                   (q.get(timeout=900) for _ in range(world)))
        for p in ps:
            p.join(timeout=60)
            assert p.exitcode == 0, f"rank exited {p.exitcode}"
    # A collective costs what the SLOWEST rank pays. Taking the mean would report the fastest
    # rank's step as if the barrier were free.
    slow = [max(got[r][0][i] for r in got) for i in range(REPS)]
    warm = sorted(slow[1:])
    row = {"arm": label, "world": world, "elements": elements, "tensors": ntensors,
           "mb_returned": round(got[0][1] / 1e6, 1), "warm_reps": len(warm),
           "median_s": warm[len(warm) // 2], "min_s": warm[0], "max_s": warm[-1],
           "cold_s": slow[0], "per_rank_s": {str(r): got[r][0] for r in got},
           "host": _host()}
    results.append(row)
    # Flushed here and not after the sweep: an arm that dies later keeps what it measured.
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"doc": __doc__.split("\n\n")[0], "arms": results}, indent=1))
    print(f"{label:34s} world {world}  median {row['median_s']:6.2f} s  "
          f"min {row['min_s']:6.2f}  max {row['max_s']:6.2f}   "
          f"{row['mb_returned']:7.1f} MB", flush=True)
    return row["median_s"]


if __name__ == "__main__":
    print(f"host {os.uname().nodename}  {_host()}\nwarm {REPS - 1} of {REPS} reps, "
          f"slowest rank per rep, no device opened\n")
    results = []
    for world in (2, 4):
        whole = run(world, DECLARED_ELEMENTS, DECLARED_TENSORS,
                    "whole census 381.3 M el", results)
        half = run(world, DIFFUSION_ELEMENTS, DIFFUSION_TENSORS,
                   "replicate half 216.1 M el", results)
        print(f"  -> world {world}: naming the replicate half is {whole:.2f} -> {half:.2f} s, "
              f"{whole - half:.2f} s a step off ({100 * (whole - half) / whole:.1f} %)\n",
              flush=True)
    print(f"artifact {OUT}")
