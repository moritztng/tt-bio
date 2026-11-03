"""Fold a fixed panel of inputs on one chip under one lever arm, for a per-case A/B grade (`panel_grade.py`).

    python perf/mol1/fair/panel.py --chip 0 --arm a --out DIR --seeds 42,66 IN.json [IN.json ...]

Loads the model once, the way run.py does (`tt-bio predict` config, `_WorkerState.load_model`, `predict_one`), then
folds every input at every seed with its directory as cwd, so relative MSA paths resolve. Arms change the size set
on the serving path in-process, never by a harness lever list:
  base  the tree as it is
  a     `SIZE_LEVERS_MIN_TOKENS` 0: every input takes `size_levers` (MOL1_BH_512 on Blackhole)
  b     arm a with `diffusion_bf16` added to that set
One line per fold to DIR/runs.jsonl: wall, tokens, coords sha256, AICLK sampled during the fold. CIFs under
DIR/<input stem>/s<seed>/.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "perf"))
sys.path.insert(0, str(ROOT / "perf" / "mol1" / "fair"))

import clocksample  # noqa: E402
from run import run_config, seeded  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--chip", type=int, required=True)
    ap.add_argument("--arm", choices=("base", "a", "b"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="42,66")
    ap.add_argument("--model", default="mol1")
    a = ap.parse_args()
    os.environ["TT_VISIBLE_DEVICES"] = os.environ["TT_BIO_LEASE_CARDS"] = str(a.chip)
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import torch
    from tt_bio import protenix as P, worker as W
    if a.arm in ("a", "b"):
        P.SIZE_LEVERS_MIN_TOKENS = 0
    if a.arm == "b":
        P.MOL1_BH_512 = P.MOL1_BH_512 | {"diffusion_bf16"}
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "runs.jsonl", "a")
    inputs = [Path(p).resolve() for p in a.inputs]
    state = None
    for inp in inputs:
        os.chdir(inp.parent)
        d = out / inp.stem
        cfg, jobs = run_config(a.model, inp, d)
        W._ensure_local_artifacts(cfg)
        if state is None:
            state = W._WorkerState("tenstorrent")
            t0 = time.time()
            state.load_model(cfg)
            state.bind_run("mol1-panel", cfg)
            m = state.model
            log.write(json.dumps({"kind": "load", "arm": a.arm, "wall_s": round(time.time() - t0, 2),
                                  "levers": sorted(getattr(m, "_levers", ())),
                                  "size_levers": sorted(getattr(m, "_size_levers", ())),
                                  "size_min_tokens": P.SIZE_LEVERS_MIN_TOKENS}) + "\n")
            log.flush()
            coords, fold = {}, m.fold

            def counted_fold(*args, **kw):
                r = fold(*args, **kw)
                c = r[0] if isinstance(r, tuple) else r
                coords["c"] = c.detach().float().cpu()
                return r
            m.fold = counted_fold
        for s in (int(x) for x in a.seeds.split(",")):
            sdir = d / f"s{s}"
            sdir.mkdir(parents=True, exist_ok=True)
            coords.clear()
            err, metrics = None, {}
            with clocksample.during(period=1.0) as clk:
                t0 = time.time()
                try:
                    metrics, _, _ = state.predict_one(seeded(jobs[0], s), dict(cfg, seed=s, struct_dir=str(sdir)))
                except Exception as exc:  # noqa: BLE001 -- a failing fold is a result
                    err = f"{type(exc).__name__}: {str(exc)[:600]}"
                wall = time.time() - t0
            c = coords.get("c")
            log.write(json.dumps({"kind": "fold", "arm": a.arm, "input": inp.stem, "seed": s, "wall_s": round(wall, 3),
                                  "tokens": metrics.get("n_tokens"), "err": err,
                                  "finite": bool(c is not None and torch.isfinite(c).all()),
                                  "coords_sha256": hashlib.sha256(c.numpy().tobytes()).hexdigest() if c is not None else None,
                                  "plddt": metrics.get("plddt"), "aiclk": clk.summary(),
                                  "struct_dir": str(sdir), "t": time.strftime("%FT%TZ", time.gmtime())}) + "\n")
            log.flush()


if __name__ == "__main__":
    main()
