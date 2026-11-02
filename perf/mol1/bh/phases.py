"""Where a Mol-1 prediction's wall time goes, per phase, on one chip.

    cd state/mol1/inputs && TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 python <tt-bio>/perf/mol1/bh/phases.py \
        --input mol1-256.json --out <dir> [--rounds 3]

Loads the model the way perf/mol1/fair/run.py does (the serving path, `predict_one`), then wraps each phase with a
device sync on both sides: host featurisation (`_protenix_inputs`), trunk (`_trunk_cond`), the confidence head's
sample-invariant start, diffusion (`edm_sample`), the per-sample confidence head (`_confidences`) and the output
write (`_protenix_emit`). The syncs serialise phases that otherwise overlap, so the sum can exceed the fold's wall
time by the overlap; `wall_s` is the prediction as a user sees it. AICLK is sampled during every round.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # perf/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fair"))

import clocksample  # noqa: E402
from run import run_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="mol1")
    ap.add_argument("--rounds", type=int, default=3, help="warm rounds after the cold one")
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--cli-extra", nargs=argparse.REMAINDER, default=[])
    a = ap.parse_args()

    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    from tt_bio import protenix as P
    from tt_bio import worker as W

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg, jobs = run_config(a.model, Path(a.input).resolve(), out, a.cli_extra)
    W._ensure_local_artifacts(cfg)
    state = W._WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("mol1-phases", cfg)
    m = state.model
    dev = m.diffusion.dev
    times = {}

    def timed(name, fn):
        def wrap(*args, **kw):
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            r = fn(*args, **kw)
            ttnn.synchronize_device(dev)
            times[name] = times.get(name, 0.0) + time.perf_counter() - t0
            return r
        return wrap

    for name in ("_trunk_cond", "_confidence_start", "_confidences"):
        setattr(m, name, timed(name.strip("_"), getattr(m, name)))
    P.edm_sample = timed("diffusion", P.edm_sample)
    for name, label in (("_protenix_inputs", "featurise"), ("_protenix_emit", "write")):
        setattr(state, name, timed(label, getattr(state, name)))

    with open(out / "phases.jsonl", "a") as f:
        for i in range(1 + a.rounds):
            times.clear()
            sdir = out / f"r{i}"
            sdir.mkdir(exist_ok=True)
            with clocksample.during(period=1.0) as clk:
                t0 = time.perf_counter()
                metrics, _, _ = state.predict_one(jobs[0], dict(cfg, seed=a.seed + i, struct_dir=str(sdir)))
                wall = time.perf_counter() - t0
            line = {"kind": "cold" if i == 0 else "warm", "input": Path(a.input).name,
                    "tokens": metrics.get("n_tokens"), "seed": a.seed + i, "wall_s": round(wall, 3),
                    "phases_s": {k: round(v, 3) for k, v in times.items()},
                    "other_s": round(wall - sum(times.values()), 3),
                    "aiclk": clk.summary().get(0), "t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            print(json.dumps(line), flush=True)
            f.write(json.dumps(line) + "\n")


if __name__ == "__main__":
    main()
