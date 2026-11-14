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
from tt_bio.boltzgen.task.predict import writer as W
from tt_bio.boltzgen.model.modules import confidence as CF
from tt_bio import tenstorrent as TT

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
                if len(c[2]) < 64 and dt > 0.01:
                    c[2].append([round(t - REC[-1]["start"], 2), round(dt, 2)])
    setattr(owner, attr, g)


_timed(torch, "load", "torch.load")
# The refold tail (diffusion done -> next design): the confidence module per call (the outer sequential call plus one per
# sample), its TT pairformer (ends in a synced to_torch), its host heads, every device download, and the writers (which
# run on a helper thread, so they overlap the next design).
_timed(CF.ConfidenceModule, "forward", "conf.forward")
_timed(CF.ConfidenceHeads, "forward", "conf.heads")
_timed(TT.PairformerModule, "forward", "tt.Pairformer.forward")
_timed(ttnn, "to_torch", "ttnn.to_torch")
_timed(W.FoldingWriter, "write", "FoldingWriter.write")
_timed(W.DesignWriter, "write", "DesignWriter.write")
_timed(PR, "load_boltz_checkpoint", "load_boltz_checkpoint")
_timed(ttnn, "from_torch", "ttnn.from_torch")
# The per-stage prelude (stage start -> checkpoint load, ~2.8 s a stage on WH .111): config instantiate (data module,
# tokenizer, featurizer, writer), the bg_featproc spawn and its first reply, the dataloader, the writers' finalize.
from tt_bio.boltzgen.task.predict import data_from_generated as DG, data_from_yaml as DY
_timed(C, "_instantiate", "instantiate")
_timed(C, "seed_everything", "seed_everything")
_timed(PR, "quiet_startup", "quiet_startup")
_timed(PR._FeatureProcess, "__init__", "featproc.spawn")
_timed(PR._FeatureProcess, "result", "featproc.result")
for _c in (DG.FromGeneratedDataModule, DY.FromYamlDataModule):
    _timed(_c, "__init__", f"{_c.__name__}.__init__")
    _timed(_c, "predict_dataloader", f"{_c.__name__}.predict_dataloader")
for _c in (W.FoldingWriter, W.DesignWriter):
    if hasattr(_c, "finalize"):
        _timed(_c, "finalize", f"{_c.__name__}.finalize")
# Does building the model draw from torch's global stream? If it does, a resident model changes every later draw
# (the designs) unless the cache replays the build's draws.
_lbc = PR.load_boltz_checkpoint
def _lbc_rng(*a, **kw):
    before = torch.random.get_rng_state().clone()
    try:
        return _lbc(*a, **kw)
    finally:
        if REC:
            REC[-1]["timers"]["load.rng_moved"] = [1, float(not torch.equal(before, torch.random.get_rng_state())), []]
PR.load_boltz_checkpoint = _lbc_rng
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
