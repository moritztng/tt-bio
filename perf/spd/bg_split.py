"""bg_split.py DESIGN_ARGS... : where one BoltzGen design's wall goes, stage by stage, on the code a served request runs.

Runs `tt-bio design` in-process (same argv as perf/bmk/design_cell.py) and stamps each design at its stage boundaries:
featurize (the dataloader's next item, serial: predict sets num_workers 0), trunk (predict_step start to the trunk's
last progress event), cond (trunk end to the first diffusion step), diffusion (first step to the end of the last; `ms_step`
per step), post (last step to predict_step's return) and write (DesignWriter.write). The device is
synchronised at each boundary only (a dozen per design), so the stages add up to the design wall within noise.
The writer runs on a thread beside the next design's predict, so each design's stamps are snapshotted at predict_step's
return and `wall`/`featurize` are measured from the previous predict's end; `write` is the overlapped writer time.
The trunk stage is split further into pre / static / loop / down / after (see the stamps on TrunkModule below).

One line per design on stdout, `BGSPLIT {json}`, also appended to $BGSPLIT_OUT when set.

    BGSPLIT_OUT=split.jsonl python perf/spd/bg_split.py design perf/dsfix/fixtures/bg_R3.yaml --model boltzgen \
        --steps design --num_designs 6 --out_dir OUT --debug --log [--fast]
"""
import json
import os
import runpy
import sys
import time
from collections import deque

import ttnn

from tt_bio import tenstorrent as T
from tt_bio.boltzgen.model.models import boltz as B
from tt_bio.boltzgen.model.modules import diffusion as D
from tt_bio.boltzgen.task.predict import writer as W

OUT = os.environ.get("BGSPLIT_OUT")
EV = {}
DONE = deque()
LAST = {"end": time.perf_counter(), "n": 0}


def _sync():
    try:
        ttnn.synchronize_device(T.get_device())
    except Exception:
        pass


def _progress(orig):
    def run(kind, step=0, total=0):
        if (kind == "trunk" and step == total) or (kind == "diffusion" and step in (0, total)):
            _sync()
            EV.setdefault((kind, step), time.perf_counter())
            if kind == "diffusion":
                EV["steps"] = total
        return orig(kind, step, total)
    return run


B._emit_progress = _progress(B._emit_progress)
D._emit_progress = _progress(D._emit_progress)

_predict = B.Boltz.predict_step


def predict_step(self, *a, **k):
    EV.clear()
    t0 = time.perf_counter()
    EV["t0"] = t0
    out = _predict(self, *a, **k)
    _sync()
    EV["t1"] = time.perf_counter()
    DONE.append(dict(EV))
    return out


B.Boltz.predict_step = predict_step


def _stamp(name, fn):
    """Wrap `fn` so a synced stamp lands in EV under `name` (and `name`+"0" at entry) each call."""
    def run(*a, **k):
        _sync()
        EV[name + "0"] = time.perf_counter()
        out = fn(*a, **k)
        _sync()
        EV[name] = time.perf_counter()
        return out
    return run


# Inside the trunk stage: `tf` is TrunkModule.forward (static build + uploads, the recycle loop, s/z download),
# `st` its static build, `it` each recycle iteration (the last one's end closes the loop).
T.TrunkModule.forward = _stamp("tf", T.TrunkModule.forward)
T.TrunkModule._build_static = _stamp("st", T.TrunkModule._build_static)
T.TrunkModule._iteration = _stamp("it", T.TrunkModule._iteration)
_write = W.DesignWriter.write


def write(self, *a, **k):
    t = time.perf_counter()
    out = _write(self, *a, **k)
    t2 = time.perf_counter()
    ev = DONE.popleft()
    n = ev.get("steps", 0)
    trunk_end = max((v for kk, v in ev.items() if isinstance(kk, tuple) and kk[0] == "trunk"), default=None)
    d1, dn = ev.get(("diffusion", 0)), ev.get(("diffusion", n))
    LAST["n"] += 1
    rec = {"design": LAST["n"], "wall": round(ev["t1"] - LAST["end"], 3), "featurize": round(ev["t0"] - LAST["end"], 3),
           "write": round(t2 - t, 3), "predict": round(ev["t1"] - ev["t0"], 3), "steps": n}
    if trunk_end and d1 and dn and n:
        rec.update(trunk=round(trunk_end - ev["t0"], 3), cond=round(d1 - trunk_end, 3), diffusion=round(dn - d1, 3),
                   ms_step=round(1e3 * (dn - d1) / n, 2), post=round(ev["t1"] - dn, 3))
    if "tf" in ev and "st" in ev and "it" in ev:
        # pre: predict start to the resident trunk (host input embedder + z_init); static: masks + uploads;
        # loop: the recycle iterations; down: s/z to the host; after: distogram etc. up to the trunk's end event.
        rec.update(pre=round(ev["tf0"] - ev["t0"], 3), static=round(ev["st"] - ev["st0"], 3),
                   loop=round(ev["it"] - ev["st"], 3), down=round(ev["tf"] - ev["it"], 3),
                   after=round(trunk_end - ev["tf"], 3) if trunk_end else None)
    LAST["end"] = ev["t1"]
    line = "BGSPLIT " + json.dumps(rec)
    print(line, flush=True)
    if OUT:
        with open(OUT, "a") as f:
            f.write(line[8:] + "\n")
    return out


W.DesignWriter.write = write

sys.argv = ["tt-bio"] + sys.argv[1:]
runpy.run_module("tt_bio.main", run_name="__main__")
