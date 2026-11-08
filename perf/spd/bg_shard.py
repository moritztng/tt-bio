"""bg_shard.py SPEC OUT N [N ...] : where a served BoltzGen shard's wall goes, fixed versus per design.

Runs the shard the way the JapanFold worker does (tt_bio.worker._execute_design_job_inprocess): one process, one open
device, `run_command` called once per shard with the full protein-anything pipeline. Each N is one shard of N designs,
seeded 0. Every progress event (stage start/done, per-design step) is stamped, and each step's model build
(`_instantiate`) and `task.run` are timed apart, so a fit over N separates the per-shard fixed cost from per-design work.
Writes OUT/shards.jsonl, one line per shard: {"n", "wall", "stages": [{name, build, run, first_step, events}]}.
"""
import json
import os
import sys
import time
from pathlib import Path

from tt_bio.boltzgen import progress as P
from tt_bio.boltzgen.cli import boltzgen as C

REC = []

_w = P.FileReporter._w
def _stamped(self, **ev):
    ev["ts"] = time.time()
    if ev.get("t") == "start":
        REC.append({"name": ev["name"], "start": ev["ts"], "events": []})
    elif REC:
        REC[-1]["events"].append([ev.get("t"), ev.get("kind"), ev.get("n"), round(ev["ts"] - REC[-1]["start"], 3)])
    _w(self, **ev)
P.FileReporter._w = _stamped

_inst = C._instantiate
def _timed_instantiate(config):
    t = time.time(); task = _inst(config)
    if REC: REC[-1]["build"] = round(time.time() - t, 3)
    run = task.run
    def timed_run(cfg):
        t = time.time(); run(cfg)
        if REC: REC[-1]["run"] = round(time.time() - t, 3)
    task.run = timed_run
    return task
C._instantiate = _timed_instantiate

def main():
    # BoltzGen's featurizer is a spawn-started process, so this module must import without side effects.
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
            steps = [e[3] for e in r["events"] if e[0] == "step"]
            r["first_step"] = steps[0] if steps else None
        line = {"i": i, "n": n, "wall": round(wall, 2), "stages": REC[:]}
        with open(out / "shards.jsonl", "a") as f:
            f.write(json.dumps(line) + "\n")
        print("BGSHARD", json.dumps({"i": i, "n": n, "wall": round(wall, 1),
              "stages": {r["name"]: [r.get("build"), r.get("run"), r.get("first_step")] for r in REC}}), flush=True)


if __name__ == "__main__":
    main()
