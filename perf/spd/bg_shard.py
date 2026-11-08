"""bg_shard.py SPEC OUT N [N ...] : where a served BoltzGen shard's wall goes, fixed versus per design.

Runs the shard the way the JapanFold worker does (tt_bio.worker._execute_design_job_inprocess): one long-lived process,
one open device, `run_command` called once per shard with the full protein-anything pipeline. Each N is one shard of N
designs, seeded 0; the second and later shards are the served case (the worker process is warm). Every progress event is
stamped, and the host-side suspects of a per-shard fixed cost (checkpoint torch.load, model build, state-dict load, the
mid-run checkpoint swap, the per-design device driver builds, ttnn.from_torch) are timed cumulatively per stage.
Only class / module attributes are patched: the analysis stage pickles its task.
Writes OUT/shards.jsonl, one line per shard: {"i", "n", "wall", "stages": [{name, start, end, events, timers}]}.
"""
import functools
import json
import os
import sys
import time
from pathlib import Path

import torch
import ttnn

from tt_bio.boltzgen import progress as P
from tt_bio.boltzgen.cli import boltzgen as C
from tt_bio.boltzgen.model.models import boltz as B
from tt_bio.boltzgen.task.predict import predict as PR

REC = []

_w = P.FileReporter._w
def _stamped(self, **ev):
    ev["ts"] = time.time()
    if ev.get("t") == "start":
        REC.append({"name": ev["name"], "start": ev["ts"], "events": [], "timers": {}})
    elif REC:
        REC[-1]["events"].append([ev.get("t"), ev.get("kind"), ev.get("n"), round(ev["ts"] - REC[-1]["start"], 3)])
        if ev.get("t") == "done":
            REC[-1]["end"] = ev["ts"]
    _w(self, **ev)
P.FileReporter._w = _stamped


def _timed(owner, attr, label):
    f = getattr(owner, attr)
    @functools.wraps(f)
    def g(*a, **kw):
        t = time.time()
        try:
            return f(*a, **kw)
        finally:
            if REC:
                c = REC[-1]["timers"].setdefault(label, [0, 0.0, []])
                dt = time.time() - t
                c[0] += 1; c[1] += dt
                if len(c[2]) < 16 and dt > 0.05:
                    c[2].append([round(t - REC[-1]["start"], 2), round(dt, 2)])
    setattr(owner, attr, g)


_timed(torch, "load", "torch.load")
_timed(PR, "load_boltz_checkpoint", "load_boltz_checkpoint")
_timed(ttnn, "from_torch", "ttnn.from_torch")
for _m in ("load_checkpoint_weights", "load_state_dict", "setup_for_inference", "predict_step",
           "_tt_trunk_module", "_tt_zinit_module"):
    _timed(B.Boltz, _m, _m)


def main():
    # BoltzGen's featurizer is a spawn-started process, so this module must import without side effects beyond the
    # patches above.
    spec, out, ns = sys.argv[1], Path(sys.argv[2]), [int(x) for x in sys.argv[3:]]
    out.mkdir(parents=True, exist_ok=True)
    for i, n in enumerate(ns):
        REC.clear()
        os.environ["BOLTZGEN_PROGRESS_FILE"] = str(out / f"progress{i}.jsonl")
        argv = ["run", spec, "--output", str(out / f"shard{i}"), "--num_designs", str(n),
                "--protocol", "protein-anything", "--seed", "0"]
        t0 = time.time()
        C.run_command(C.build_parser().parse_args(argv))
        wall = time.time() - t0
        for r in REC:
            for c in r["timers"].values():
                c[1] = round(c[1], 3)
        with open(out / "shards.jsonl", "a") as f:
            f.write(json.dumps({"i": i, "n": n, "t0": t0, "wall": round(wall, 2), "stages": REC[:]}) + "\n")
        print("BGSHARD", json.dumps({"i": i, "n": n, "wall": round(wall, 1),
              "stages": {r["name"]: round(r.get("end", time.time()) - r["start"], 1) for r in REC}}), flush=True)


if __name__ == "__main__":
    main()
