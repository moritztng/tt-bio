"""One RFD4 design end to end on a card, timed, with the clock sampled during it.

    TT_VISIBLE_DEVICES=<card> python perf/rfd4/e2e/design.py --spec <spec.yaml> --weights <bundle> \
        --out <dir> [--seed 0] [--designs 2]

The recipe is the released `thinking_medium`, 60 steps and 180 trunk passes, and nothing here can
shorten it: the engine takes its step count from the recipe and the sampler refuses a depth list of
the wrong length. Each design writes its CIF and one JSON line to `<out>/runs.jsonl`:

* `kind: cold` is the engine build (device open, weight upload), reported apart, never amortised.
* The first design pays kernel compilation; it is recorded but is not the warm number.
* Every later design is warm. Its wall time is split into the trunk (device, synced once per
  network call when the trunk hands s back) and the host (features, sampler, decode, write).

AICLK is sampled on a thread for the life of each design (`perf/clocksample.py`). Below ~1200 MHz
the line is flagged `clock_artifact`, because on Blackhole the clock sets the time.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # perf/ for clocksample

import torch

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()

import clocksample  # noqa: E402

CLOCK_FLOOR_MHZ = 1200


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--designs", type=int, default=2, help="sequential single designs; the first is the compile")
    ap.add_argument("--rng", default="upstream", choices=("upstream", "per_design"),
                    help="upstream: same noise as capture.py --reseed-at-sampler at this seed (paired grade)")
    a = ap.parse_args()

    from tt_bio import rfd4
    from tt_bio.rfd4 import engine as E

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "runs.jsonl", "a")
    sha = os.popen("git rev-parse HEAD").read().strip()
    card = os.environ.get("TT_VISIBLE_DEVICES")

    def emit(rec):
        rec.update(sha=sha, card=card, host=os.uname().nodename, t=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), flush=True)

    t0 = time.perf_counter()
    designs = rfd4.prepare_examples(a.spec, out / "inputs")
    features, atom_array = rfd4.featurize(designs[0].cif, seed=a.seed)
    t_feat = time.perf_counter() - t0
    n_tokens = int(features["padding_mask"].shape[-1])

    with clocksample.during() as clk:
        t0 = time.perf_counter()
        engine = E.DesignEngine(a.weights, recipe="thinking_medium")
        t_build = time.perf_counter() - t0
    emit({"kind": "cold", "what": "engine build (device open + weight upload)", "wall_s": round(t_build, 2),
          "featurize_s": round(t_feat, 2), "aiclk": clk.summary()})

    # Time the trunk calls from inside the network so the host share is measured, not inferred.
    # The patch is by name: a tree that reaches the trunk another way leaves trunk_s empty, and then
    # the design is fine and only the instrument failed. Say which, rather than dying in the writer.
    trunk_s = []
    inner = engine.trunk.run_bucketed

    def timed(*args, **kwargs):
        t = time.perf_counter()
        r = inner(*args, **kwargs)
        trunk_s.append(time.perf_counter() - t)
        return r

    engine.trunk.run_bucketed = timed

    for i in range(a.designs):
        trunk_s.clear()
        with clocksample.during(period=1.0) as clk:
            t0 = time.perf_counter()
            output = engine.design(features, num_designs=1, nsteps=engine.spec.nsteps, seed=a.seed, rng=a.rng)
            t_design = time.perf_counter() - t0
            t1 = time.perf_counter()
            path = rfd4.write_design(output, features, atom_array,
                                     out / f"{designs[0].example_id}_s{a.seed}_r{i}_generated.cif")
            t_write = time.perf_counter() - t1
        summ = clk.summary()
        mhz = _median_mhz(summ)
        emit({"kind": "compile" if i == 0 else "warm", "size": n_tokens, "designs": 1, "batch": 1,
              "recipe": "thinking_medium", "steps": engine.spec.nsteps, "network_calls": len(trunk_s),
              "trunk_passes": int(sum(_depths(engine))), "seed": a.seed, "rng": a.rng,
              "wall_s": round(t_design + t_write, 3), "design_s": round(t_design, 3), "write_s": round(t_write, 3),
              "trunk_s": round(sum(trunk_s), 3) if trunk_s else None,
              "host_s": round(t_design - sum(trunk_s), 3) if trunk_s else None,
              "trunk_call_s": ({"min": round(min(trunk_s), 4), "max": round(max(trunk_s), 4)}
                               if trunk_s else None),
              "instrument_attached": bool(trunk_s),
              "aiclk": summ, "aiclk_mhz": mhz, "clock_artifact": bool(mhz is not None and mhz < CLOCK_FLOOR_MHZ),
              "cif": str(path)})
        if not trunk_s:
            print("INSTRUMENT: engine.trunk.run_bucketed was never called through the timing patch; "
                  "the design itself completed and wrote %s. Re-point the patch before reading any "
                  "trunk_s from this tree." % path, flush=True)
            raise SystemExit(3)


def _depths(engine):
    from tt_bio.rfd4 import sampler
    return sampler.recycle_depths(engine.spec, engine.spec.nsteps)


def _median_mhz(summary):
    try:
        dev = summary.get(0) or summary.get("0") or next(iter(summary.values()))
        return dev.get("median")
    except Exception:
        return None


if __name__ == "__main__":
    main()
