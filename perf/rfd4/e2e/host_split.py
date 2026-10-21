"""Where the host half of an RFD4 design goes, with the clock and the host load beside it.

    TT_VISIBLE_DEVICES=<card> python perf/rfd4/e2e/host_split.py --spec <spec.yaml> --weights <bundle> \
        --out <dir> [--seed 0] [--designs 4] [--profile]

`design.py` splits a design into trunk and host. This splits the host further, with synced timers:

* glue_s: upstream's feature factories, conditioning transitions and heads (`FlowGlue`), per step
* prep_s: `run_bucketed` minus the trunk's device time: padding, masking, upload, the s readback
* device_s: `Trunk.__call__`, synchronized before and after, so it is the card's time alone
* sampler_s: everything else inside `engine.design` before decode (the integrator, chunking)
* decode_s, write_s: the autoencoder's decoder and upstream's CIF writer

The sync around the trunk call costs one device round trip per step, 60 per design; device_s is
still within a few ms of the unsynced trunk time `design.py` reports. `--profile` adds one more warm
design under cProfile and writes the top of it, which is how the write and the glue are attributed
to functions; its wall is not a measurement.
"""
import argparse
import cProfile
import io
import json
import os
import pstats
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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
    ap.add_argument("--designs", type=int, default=4)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    import ttnn
    from tt_bio import rfd4
    from tt_bio.rfd4 import engine as E
    from tt_bio.rfd4.trunk import Trunk

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "split.jsonl", "a")
    sha = os.popen("git rev-parse --short HEAD").read().strip()

    designs = rfd4.prepare_examples(a.spec, out / "inputs")
    t0 = time.perf_counter()
    features, atom_array = rfd4.featurize(designs[0].cif, seed=a.seed)
    t_feat = time.perf_counter() - t0
    engine = E.DesignEngine(a.weights, recipe="thinking_medium")
    dev = engine._device

    acc = defaultdict(float)

    def timed(name, fn, sync=False):
        def wrapper(*args, **kwargs):
            if sync:
                ttnn.synchronize_device(dev)
            t = time.perf_counter()
            r = fn(*args, **kwargs)
            if sync:
                ttnn.synchronize_device(dev)
            acc[name] += time.perf_counter() - t
            return r
        return wrapper

    g = engine.glue
    g.init_repr_factory.forward = timed("glue_init_s", g.init_repr_factory.forward)
    g.pair_repr_builder.forward = timed("glue_pair_s", g.pair_repr_builder.forward)
    g.conditioning = timed("glue_cond_s", g.conditioning)
    g.heads = timed("glue_heads_s", g.heads)
    engine.trunk.run_bucketed = timed("bucketed_s", engine.trunk.run_bucketed)
    call = Trunk.__call__
    Trunk.__call__ = timed("device_s", lambda self, *x, **k: call(self, *x, **k), sync=True)
    engine.decode = timed("decode_s", engine.decode)

    def one(i, kind):
        acc.clear()
        load0 = os.getloadavg()[0]
        with clocksample.during(period=1.0) as clk:
            t0 = time.perf_counter()
            output = engine.design(features, num_designs=1, nsteps=engine.spec.nsteps, seed=a.seed, rng="upstream")
            t_design = time.perf_counter() - t0
            t1 = time.perf_counter()
            path = rfd4.write_design(output, features, atom_array,
                                     out / f"{designs[0].example_id}_s{a.seed}_{a.tag}{kind}{i}.cif")
            t_write = time.perf_counter() - t1
        summ = clk.summary()
        mhz = (summ.get(0) or summ.get("0") or next(iter(summ.values()), {}) or {}).get("median") if summ else None
        glue = sum(v for k, v in acc.items() if k.startswith("glue_"))
        rec = {"kind": kind, "i": i, "tag": a.tag, "size": int(features["padding_mask"].shape[-1]),
               "design_s": round(t_design, 3), "write_s": round(t_write, 3),
               "wall_s": round(t_design + t_write, 3),
               "device_s": round(acc["device_s"], 3),
               "prep_s": round(acc["bucketed_s"] - acc["device_s"], 3),
               "glue_s": round(glue, 3),
               **{k: round(v, 3) for k, v in sorted(acc.items()) if k.startswith("glue_")},
               "decode_s": round(acc["decode_s"], 3),
               "sampler_s": round(t_design - acc["bucketed_s"] - glue - acc["decode_s"], 3),
               "host_s": round(t_design - acc["device_s"], 3),
               "featurize_s": round(t_feat, 3),
               "aiclk_mhz": mhz, "aiclk": summ,
               "clock_artifact": bool(mhz is not None and mhz < CLOCK_FLOOR_MHZ),
               "load1": round(load0, 2), "load1_end": round(os.getloadavg()[0], 2), "nproc": os.cpu_count(),
               "sha": sha, "card": os.environ.get("TT_VISIBLE_DEVICES"),
               "t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "cif": str(path)}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps({k: v for k, v in rec.items() if k != "aiclk"}), flush=True)

    for i in range(a.designs):
        one(i, "compile" if i == 0 else "warm")

    if a.profile:
        prof = cProfile.Profile()
        prof.enable()
        one(0, "profiled")
        prof.disable()
        s = io.StringIO()
        pstats.Stats(prof, stream=s).sort_stats("cumulative").print_stats(80)
        pstats.Stats(prof, stream=s).sort_stats("tottime").print_stats(50)
        (out / f"profile_{a.tag}.txt").write_text(s.getvalue())


if __name__ == "__main__":
    main()
