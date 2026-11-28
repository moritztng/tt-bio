"""Per-ligand cost of a screen, pass after pass, on the shipped dock path.

WORKLOAD-S docks the same 64 ligands in three timed batches and the batches disagree by more
than the protocol's 10 %. This runs a short list of ligands several times in one process
through `tt_bio.diffdock.run.dock` (the function `tt-bio dock` calls) and writes, per ligand
and pass: wall, seconds in the rollout's score calls with each step's wall, seconds in the
confidence pass, seconds inside the device layers, the program-cache entry count, the bytes
the graph caches hold on the card, the pose-chunk width, host RSS and load. Poses are kept per
pass so the same seed can be compared across passes.

    python shapes_repro.py --csv screen72.csv --names 5S8I_2LY,6M2B_EZO --passes 3 \
        --checkpoints <dir with score_model/ and confidence_model/> --out repro.json
"""
import argparse
import json
import os
import resource
import shutil
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True)
ap.add_argument("--names", required=True, help="comma-separated complex names, in run order")
ap.add_argument("--passes", type=int, default=3)
ap.add_argument("--edge-tp", default="tile", choices=["gemm", "tile"])
ap.add_argument("--grid-ratio", type=float, default=None,
                help="row grid for the confidence shapes (tt_bio.diffdock.graph.GRID_RATIO); "
                     "1.0 is the pad-to-32 the port used before the grid, the A/B's control")
ap.add_argument("--samples", type=int, default=None,
                help="poses per complex (default: the recipe's 10). Fewer is a cheaper A/B "
                     "instrument, not a perf number: the ranking pass is per pose")
ap.add_argument("--seed", type=int, default=None,
                help="sampler seed. Leave unset for a screen's own behaviour (upstream draws "
                     "from the global stream, so every visit gets different poses and therefore "
                     "different shapes); set it to compare two arms on identical poses")
ap.add_argument("--checkpoints", default=None)
ap.add_argument("--out", required=True)
ap.add_argument("--profile", default=None,
                help="complex name to run once more under cProfile after the passes")
args = ap.parse_args()

from tt_bio.diffdock import graph as dfd_graph
from tt_bio.diffdock import run as dock_run
from tt_bio.diffdock.job import DockOptions, read_csv_jobs

if args.grid_ratio is not None:
    dfd_graph.GRID_RATIO = args.grid_ratio

by_name = {j.name: j for j in read_csv_jobs(Path(args.csv))}
jobs = [by_name[n] for n in args.names.split(",")]
kw = {"device": "tt", "edge_tp": args.edge_tp}
if args.samples is not None:
    kw["diffusion_samples"] = args.samples
if args.seed is not None:
    kw["seed"] = args.seed
if args.checkpoints:
    kw["checkpoint"] = f"{args.checkpoints}/score_model"
    kw["confidence_checkpoint"] = f"{args.checkpoints}/confidence_model"
options = DockOptions(**kw)
model = dock_run.load_model(options)

clock = {"score": 0.0, "confidence": 0.0, "steps": [], "widths": []}
inner_score_fn, inner_confidence = model.score_fn, model.confidence


def score_fn(graph):
    fn = inner_score_fn(graph)

    def timed(step):
        t = time.perf_counter()
        out = fn(step)
        dt = time.perf_counter() - t
        clock["score"] += dt
        clock["steps"].append(round(dt, 3))
        clock["widths"].append(int(step.lig_pos.shape[0]))
        return out

    return timed


def confidence(graph, poses):
    t = time.perf_counter()
    out = inner_confidence(graph, poses)
    clock["confidence"] += time.perf_counter() - t
    return out


model.score_fn, model.confidence = score_fn, confidence
mods = [*model.convs.values(), *model.confidence_convs.values()]
caches = {id(m.graphs): m.graphs for m in mods}.values()


def aiclk():
    chip = os.environ.get("TT_VISIBLE_DEVICES", "0").split(",")[0]
    try:
        return int(Path(f"/sys/class/tenstorrent/tenstorrent!{chip}/tt_aiclk").read_text())
    except (OSError, ValueError):
        return None


def one(job, out_dir):
    clock.update(score=0.0, confidence=0.0, steps=[], widths=[])
    dev0 = sum(m.secs for m in mods)
    t = time.perf_counter()
    dock_run.dock([job], options, out_dir, model=model)
    wall = time.perf_counter() - t
    return {"name": job.name, "wall_s": round(wall, 3), "score_s": round(clock["score"], 3),
            "confidence_s": round(clock["confidence"], 3),
            "device_layer_s": round(sum(m.secs for m in mods) - dev0, 3),
            "steps_s": clock["steps"], "pose_widths": sorted(set(clock["widths"])),
            "program_cache_entries": int(model.device.num_program_cache_entries()),
            "graph_cache_bytes": sum(c.nbytes() for c in caches),
            "rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024,
            "load1": os.getloadavg()[0], "aiclk_mhz": aiclk(),
            "utc": time.strftime("%FT%TZ", time.gmtime())}


out = Path(args.out)
rows = []
for p in range(args.passes):
    for job in jobs:
        d = out.with_suffix("") / f"pass{p}"
        shutil.rmtree(d / job.name, ignore_errors=True)
        rows.append({"pass": p, **one(job, d)})
        print(json.dumps({k: v for k, v in rows[-1].items() if k != "steps_s"}), flush=True)
        out.write_text(json.dumps({"edge_tp": args.edge_tp, "samples": options.diffusion_samples,
                                   "seed": options.seed, "grid_ratio": dfd_graph.GRID_RATIO,
                                   "rows": rows}, indent=1))

if args.profile:
    import cProfile
    import io
    import pstats
    pr = cProfile.Profile()
    pr.enable()
    one(by_name[args.profile], out.with_suffix("") / "profile")
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(60)
    out.with_suffix(".prof.txt").write_text(s.getvalue())
