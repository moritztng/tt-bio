"""One chip of a scaling run: load and compile once, then run a batch each time the driver says so.

    TT_VISIBLE_DEVICES=<card> python perf/rfd4/scaling/worker.py --spec S --weights W --batch B [--lane-width L]

Reads commands on stdin, one per line, and answers each with one JSON line on stdout:

    run SEED   one full batch of B finished designs, timed exactly like perf/rfd4/fair/run.py's warm line
               (spec parse, featurisation, all 60 steps of thinking_medium, decode, every CIF written)
    quit       exit

The first line out is {"ready": ...} once the device is open and the compile batch has run. Wall-clock
stamps are time.time() so the driver can line batches on different chips up against each other.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()


def cpu_s():
    """CPU seconds this process (every thread) has used so far."""
    t = os.times()
    return t.user + t.system


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--lane-width", type=int)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from tt_bio import rfd4
    from tt_bio.rfd4 import engine as E

    out = sys.stdout
    sys.stdout = sys.stderr  # the engine's own prints must not corrupt the protocol
    card = os.environ.get("TT_VISIBLE_DEVICES")
    root = Path(a.out) / f"chip{card}"

    t0 = time.perf_counter()
    engine = E.DesignEngine(a.weights, recipe="thinking_medium")
    engine.max_parallel_samples = lane = min(a.batch, a.lane_width or a.batch)
    t_load = time.perf_counter() - t0

    def batch(seed, phase):
        d = root / f"{phase}-s{seed}"
        c0, w0, t0 = cpu_s(), time.time(), time.perf_counter()
        ex = rfd4.prepare_examples(a.spec, d / "inputs")[0]
        features, atom_array = rfd4.featurize(ex.cif, seed=seed)
        t_feat = time.perf_counter() - t0
        o = engine.design(features, num_designs=a.batch, nsteps=engine.spec.nsteps, seed=seed, rng="per_design")
        t_design = time.perf_counter() - t0 - t_feat
        for k in range(a.batch):
            rfd4.write_design(o, features, atom_array, d / f"{ex.example_id}_s{seed}_d{k}.cif", row=k)
        wall = time.perf_counter() - t0
        return {"card": card, "phase": phase, "seed": seed, "designs": a.batch, "batch": a.batch,
                "lane_width": lane, "width_achieved": engine.chunk_width,
                "size": int(features["padding_mask"].shape[-1]), "t_start": w0, "t_end": time.time(),
                "wall_s": round(wall, 3), "designs_per_s": round(a.batch / wall, 5),
                "featurize_s": round(t_feat, 3), "design_s": round(t_design, 3),
                "write_s": round(wall - t_feat - t_design, 3), "cpu_s": round(cpu_s() - c0, 2)}

    def say(rec):
        out.write(json.dumps(rec) + "\n")
        out.flush()

    first = batch(0, "compile")
    say({"ready": True, "card": card, "pid": os.getpid(), "load_s": round(t_load, 2), "compile": first})
    for line in sys.stdin:
        cmd = line.split()
        if not cmd or cmd[0] == "quit":
            break
        try:
            say(batch(int(cmd[1]), "measure"))
        except Exception as exc:  # noqa: BLE001 -- the driver records it and stops the run
            say({"card": card, "error": f"{type(exc).__name__}: {str(exc)[:600]}"})


if __name__ == "__main__":
    main()
