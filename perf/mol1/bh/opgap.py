"""Device op timeline of the Pairformer blocks and the diffusion step of a Mol-1 fold, and their walls.

    # attribution: Tracy build, eager diffusion, a short fold (the per-block and per-step ops do not change)
    python -m tracy -r --no-op-info-cache -o <prof> --op-support-count 40000 -- \
        <tt-bio>/perf/mol1/bh/opgap.py prof --input mol1-256.json --out <dir>
    # walls: the shipped wheel, the fold as served (traced diffusion), each trunk block synced on both sides
    python <tt-bio>/perf/mol1/bh/opgap.py wall --input mol1-256.json --out <dir> --step-probe 20
    # host: each trunk block synced before it, host return and device drain timed apart, some blocks under cProfile
    python <tt-bio>/perf/mol1/bh/opgap.py host --input mol1-256.json --out <dir> --cprofile pf48/b10/
    # diff: the fold as served, the warm fold's whole diffusion sampler under cProfile (the per-step host round trip)
    python <tt-bio>/perf/mol1/bh/opgap.py diff --input mol1-256.json --out <dir> --recycles 1

Both modes fold once cold and once warm in one process and record only the warm fold. `prof` enqueues a fence op
(a multiply on a [1,1,32,224] tensor nobody else makes) before and after every Pairformer block and every denoise
call, without a sync, so the queue in between runs exactly as in a fold; `regions.json` lists the fences in
order for perf/mol1/bh/opgap_table.py to cut the ops report with. `wall` syncs around every trunk Pairformer
block of the last cycle and replays the captured diffusion trace back to back (device time per step), the
numbers the kernel sums are compared against. `host` splits each synced block into the time the Python call
takes to return (host enqueue, queue stalls included) and the drain after it, and runs every block whose tag
starts with --cprofile under cProfile (host_blocks.json, cprofile.txt). `diff` runs the warm fold's diffusion
sampler under cProfile (cprofile.txt). AICLK is sampled during both folds.
"""
import argparse
import cProfile
import json
import pstats
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # perf/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fair"))

import clocksample  # noqa: E402
from run import run_config  # noqa: E402

FENCE_SHAPE = (1, 1, 32, 224)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("prof", "wall", "host", "diff"))
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--recycles", type=int, default=None, help="default: 1 for prof, the model's own for wall")
    ap.add_argument("--steps", type=int, default=None, help="default: 2 for prof, the protocol's 200 for wall")
    ap.add_argument("--step-probe", type=int, default=20)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--cprofile", default="pf48/b10/", help="host mode: tag prefix of the blocks run under cProfile")
    a = ap.parse_args()

    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import torch
    import ttnn
    from tt_bio import protenix as P
    from tt_bio import tenstorrent as T
    from tt_bio import worker as W

    prof = a.mode == "prof"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg, jobs = run_config("mol1", Path(a.input).resolve(), out)
    cfg.update(recycling_steps=1 if prof else cfg.get("recycling_steps"),
               sampling_steps={"prof": 2, "host": 2}.get(a.mode, 200))
    if a.recycles is not None:
        cfg["recycling_steps"] = a.recycles
    if a.steps is not None:
        cfg["sampling_steps"] = a.steps
    if prof:
        cfg["trace"] = False   # the profiler reads eager ops; the traced step's wall comes from `wall`
    W._ensure_local_artifacts(cfg)
    state = W._WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("mol1-opgap", cfg)
    m = state.model
    dev = m.diffusion.dev
    fence_t = ttnn.from_torch(torch.zeros(FENCE_SHAPE), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    rec = {"on": False, "stack": "?", "last_cycle_from": None}
    regions, walls, extra, host_blocks = [], [], {}, []
    cprof = cProfile.Profile()

    def fence(label):
        if not rec["on"]:
            return
        if prof:
            ttnn.deallocate(ttnn.multiply(fence_t, 1.0))
            regions.append(label)
        else:
            ttnn.synchronize_device(dev)
            regions.append((label, time.perf_counter()))

    stack_call = T.Pairformer.__call__

    def stack(self, s, z, *args, **kw):
        prev = rec["stack"]
        rec["stack"] = f"pf{len(self.blocks)}"
        rec["block"] = 0
        try:
            return stack_call(self, s, z, *args, **kw)
        finally:
            rec["stack"] = prev
    T.Pairformer.__call__ = stack

    layer_call = T.PairformerLayer.__call__

    def layer(self, s, z, *args, **kw):
        i = rec.get("block", 0)
        rec["block"] = i + 1
        tag = f"{rec['stack']}/b{i}/c{z.shape[-1]}/n{z.shape[-2]}"
        if a.mode == "host" and rec["on"]:
            ttnn.synchronize_device(dev)
            hot = tag.startswith(a.cprofile)
            t0 = time.perf_counter()
            if hot:
                cprof.enable()
            r = layer_call(self, s, z, *args, **kw)
            if hot:
                cprof.disable()
            t1 = time.perf_counter()
            ttnn.synchronize_device(dev)
            host_blocks.append({"block": tag, "host_ms": round(1000 * (t1 - t0), 3),
                                "drain_ms": round(1000 * (time.perf_counter() - t1), 3), "cprofile": hot})
            return r
        fence("start " + tag)
        r = layer_call(self, s, z, *args, **kw)
        fence("end " + tag)
        return r
    T.PairformerLayer.__call__ = layer

    if prof:
        denoise = m.diffusion.denoise

        def fenced_denoise(x_noisy, t_hat, cond):
            fence(f"start denoise/M{x_noisy.shape[0]}")
            r = denoise(x_noisy, t_hat, cond)
            fence(f"end denoise/M{x_noisy.shape[0]}")
            return r
        m.diffusion.denoise = fenced_denoise
    else:
        sampled = P.edm_sample

        def probed(*args, **kw):
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            if a.mode == "diff" and rec["on"]:
                cprof.enable()
            x = sampled(*args, **kw)
            cprof.disable()
            extra["diffusion_s"] = time.perf_counter() - t0
            tr = getattr(m.diffusion, "_trace", None)
            if rec["on"] and tr is not None and a.step_probe:
                ttnn.synchronize_device(dev)
                t0 = time.perf_counter()
                for _ in range(a.step_probe):
                    ttnn.execute_trace(dev, tr["tid"], cq_id=0, blocking=False)
                ttnn.synchronize_device(dev)
                extra["traced_step_ms"] = 1000 * (time.perf_counter() - t0) / a.step_probe
                t0 = time.perf_counter()
                for _ in range(a.step_probe):
                    ttnn.execute_trace(dev, tr["tid"], cq_id=0, blocking=False)
                    ttnn.to_torch(tr["out"])
                extra["traced_step_read_ms"] = 1000 * (time.perf_counter() - t0) / a.step_probe
            return x
        P.edm_sample = probed

    for i in range(2):
        rec["on"] = i == 1
        regions.clear()
        if i == 1 and prof and hasattr(ttnn, "ReadDeviceProfiler"):
            ttnn.synchronize_device(dev)
            ttnn.ReadDeviceProfiler(dev)   # the cold fold's markers out, so the warm fold has the whole buffer
        sdir = out / f"r{i}"
        sdir.mkdir(exist_ok=True)
        with clocksample.during(period=1.0) as clk:
            t0 = time.perf_counter()
            metrics, _, _ = state.predict_one(jobs[0], dict(cfg, seed=a.seed + i, struct_dir=str(sdir)))
            ttnn.synchronize_device(dev)
            wall = time.perf_counter() - t0
        line = {"mode": a.mode, "kind": "cold" if i == 0 else "warm", "input": Path(a.input).name,
                "tokens": metrics.get("n_tokens"), "wall_s": round(wall, 3),
                "recycles": cfg.get("recycling_steps"), "steps": cfg["sampling_steps"],
                "samples": cfg.get("diffusion_samples"), "trace": cfg.get("trace"),
                "aiclk": clk.summary().get(0), **{k: round(v, 4) for k, v in extra.items()},
                "t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        print(json.dumps(line), flush=True)
        with open(out / "runs.jsonl", "a") as f:
            f.write(json.dumps(line) + "\n")
    if prof:
        (out / "regions.json").write_text(json.dumps(regions))
    elif a.mode in ("host", "diff"):
        if a.mode == "host":
            (out / "host_blocks.json").write_text(json.dumps(host_blocks))
        with open(out / "cprofile.txt", "w") as f:
            for key in ("tottime", "cumulative"):
                f.write(f"==== sorted by {key}\n")
                pstats.Stats(cprof, stream=f).sort_stats(key).print_stats(60)
    else:
        # Block walls: each block between its own two synced fences.
        blocks = []
        for (lab0, t0), (lab1, t1) in zip(regions, regions[1:]):
            if lab0.startswith("start ") and lab1 == "end " + lab0[6:]:
                blocks.append({"block": lab0[6:], "ms": round(1000 * (t1 - t0), 3)})
        (out / "block_walls.json").write_text(json.dumps(blocks))


if __name__ == "__main__":
    main()
