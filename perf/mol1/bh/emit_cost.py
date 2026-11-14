"""What the write phase (`_protenix_emit`: rank, structures, upstream layout) costs per Mol-1 fold, and its inputs.

    cd <inputs> && python <tt-bio>/perf/mol1/bh/emit_cost.py --input mol1-256.json --chip 1 --out <dir> [--folds 2]

Folds through the serving path like perf/mol1/fair/run.py and times every `_protenix_emit` call with the device idle
(the fold has returned). The last fold's emit runs under cProfile (<out>/emit.prof, top lines in emit.txt) and its
arguments are pickled to <out>/emit_args.pt, so a write-path change can be A/B'd on the CPU with `--replay` and no
card: `python emit_cost.py --replay <out>/emit_args.pt --repeat 3`.
"""
import argparse
import cProfile
import importlib.util
import io
import json
import os
import pstats
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()


def replay(args_pt, repeat, out):
    import torch
    from tt_bio import worker as W
    a = torch.load(args_pt, weights_only=False)
    cfg = dict(a["cfg"], struct_dir=str(out))
    Path(out).mkdir(parents=True, exist_ok=True)
    for i in range(repeat):
        t = time.perf_counter()
        W._WorkerState._protenix_emit(None, a["path"], cfg, a["feats"], a["chains"], a["chain_specs"], a["coords"],
                                      a["confs"])
        print(json.dumps({"replay": i, "emit_s": round(time.perf_counter() - t, 3)}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input")
    ap.add_argument("--chip", type=int)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="mol1")
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--folds", type=int, default=2)
    ap.add_argument("--replay", help="emit_args.pt from an earlier run: time the write on the CPU, no device")
    ap.add_argument("--repeat", type=int, default=3)
    a = ap.parse_args()
    if a.replay:
        return replay(a.replay, a.repeat, a.out)
    os.environ["TT_VISIBLE_DEVICES"] = os.environ["TT_BIO_LEASE_CARDS"] = str(a.chip)
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import torch
    from tt_bio import worker as W
    spec = importlib.util.spec_from_file_location("mol1_fair_run", HERE.parents[1] / "fair" / "run.py")
    fair = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fair)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg, jobs = fair.run_config(a.model, Path(a.input).resolve(), out, [])
    W._ensure_local_artifacts(cfg)
    state = W._WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("mol1-bh-prof-emit", cfg)

    orig, last = W._WorkerState._protenix_emit, {"on": False, "s": None}

    def emit(self, path, cfg, feats, chains, chain_specs, coords, confs):
        if last["on"]:
            torch.save({"path": path, "cfg": cfg, "feats": feats, "chains": chains, "chain_specs": chain_specs,
                        "coords": coords, "confs": confs}, out / "emit_args.pt")
            prof = cProfile.Profile()
            t = time.perf_counter()
            r = prof.runcall(orig, self, path, cfg, feats, chains, chain_specs, coords, confs)
            last["s"] = time.perf_counter() - t
            prof.dump_stats(out / "emit.prof")
            buf = io.StringIO()
            pstats.Stats(prof, stream=buf).sort_stats("cumulative").print_stats(40)
            (out / "emit.txt").write_text(buf.getvalue())
            return r
        t = time.perf_counter()
        r = orig(self, path, cfg, feats, chains, chain_specs, coords, confs)
        last["s"] = time.perf_counter() - t
        return r

    W._WorkerState._protenix_emit = emit
    log = open(out / "runs.jsonl", "a")
    for i in range(a.folds):
        last["on"] = i == a.folds - 1
        sdir = out / f"r{i}"
        sdir.mkdir(exist_ok=True)
        t = time.time()
        state.predict_one(jobs[0], dict(cfg, seed=a.seed, struct_dir=str(sdir)))
        r = {"fold": i, "wall_s": round(time.time() - t, 2), "emit_s": round(last["s"], 3),
             "profiled": last["on"], "input": a.input}
        log.write(json.dumps(r) + "\n"); log.flush(); print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
