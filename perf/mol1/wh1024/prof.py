"""Where a warm Mol-1 fold spends its time, per op family, on one chip, through the serving path.

    cd <inputs> && python <tt-bio>/perf/mol1/wh1024/prof.py --input mol1-1024.json --chip 25 --out <dir>

Loads the model the way perf/mol1/fair/run.py does (`tt-bio predict` builds the run config, `_WorkerState`
loads, `predict_one` folds) and runs three folds in one process: a cold fold, a warm fold timed with nothing
installed (the s/pred, AICLK sampled during it), then a warm fold with every op family wrapped in a
device-synchronised timer (attribution only: the syncs drain the queue, so its wall is not the s/pred).

Families wrap the class or function that owns them, so nesting carries the context: a TriangleMultiplication
inside the MSA module's pair stack is charged to `msa_module>...>trimul`, the trunk Pairformer's to
`pairformer>...>trimul`. Host transfers (`ttnn.from_device`, `to_torch`, device `from_torch`) are their own
families. Nothing is timed while a trace is being captured. A call a DRAM refusal unwinds through carries `err`. `<out>/sections.jsonl` has one line per call
(path, seconds, shape), `<out>/summary.json` the per-path totals with self time, `<out>/runs.jsonl` the folds.
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))  # perf/, for clocksample
import clocksample  # noqa: E402

_spec = importlib.util.spec_from_file_location("mol1_fair_run", HERE.parents[1] / "fair" / "run.py")
fair = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fair)

# (module, attribute path, family). Methods wrap on their class, functions in every tt_bio module that holds them.
TARGETS = [
    ("tt_bio.protenix", "Protenix.fold", "fold"),
    ("tt_bio.protenix", "Protenix._trunk_stage", "trunk_stage"),
    ("tt_bio.protenix", "Trunk.__call__", "trunk"),
    ("tt_bio.protenix", "Trunk._template", "template"),
    ("tt_bio.protenix", "Trunk._msa", "msa_module"),
    ("tt_bio.protenix", "_KeyedWeights._opm_from_host", "opm_from_host"),
    ("tt_bio.protenix", "Protenix._diffusion_pair_cond", "pair_cond"),
    ("tt_bio.protenix", "edm_sample", "diffusion"),
    ("tt_bio.protenix", "Protenix._confidence_start", "confidence_start"),
    ("tt_bio.protenix", "Protenix._confidences", "confidence"),
    ("tt_bio.tenstorrent", "Pairformer.__call__", "pairformer"),
    ("tt_bio.tenstorrent", "PairformerLayer.__call__", "pf_layer"),
    ("tt_bio.tenstorrent", "TriangleMultiplication.__call__", "trimul"),
    ("tt_bio.tenstorrent", "TriangleAttention.__call__", "triattn"),
    ("tt_bio.tenstorrent", "Transition.__call__", "transition"),
    ("tt_bio.tenstorrent", "AttentionPairBias.__call__", "attn_pair_bias"),
    ("tt_bio.tenstorrent", "OuterProductMean.__call__", "opm"),
    ("tt_bio.tenstorrent", "PairWeightedAveraging.__call__", "pwa"),
    ("tt_bio.tenstorrent", "msa_update_chunks", "msa_update_chunks"),
    ("tt_bio.tenstorrent", "msa_embed", "msa_embed"),
    ("tt_bio.tenstorrent", "host_unpark", "host_unpark"),
    ("ttnn", "from_device", "xfer_from_device"),
    ("ttnn", "to_torch", "xfer_to_torch"),
    ("ttnn", "from_torch", "xfer_from_torch"),
]


class Sections:
    def __init__(self, out):
        self.on, self.stack, self.out = False, [], out
        self.calls = defaultdict(lambda: [0, 0.0, 0.0])  # path -> [n, inclusive s, child s]
        self.f = None

    def sync(self):
        from tt_bio import tenstorrent as T
        import ttnn
        if getattr(T, "_device", None) is not None:
            ttnn.synchronize_device(T._device)

    def wrap(self, fam, fn):
        S = self

        def inner(*a, **k):
            from tt_bio import tenstorrent as T
            if not S.on or T.CAPTURING[0]:
                return fn(*a, **k)
            if fam.startswith("xfer_from_torch") and k.get("device") is None:
                return fn(*a, **k)
            name = fam
            if fam == "pf_layer":
                name = "pf_mixer" if getattr(a[0], "triangle_attention_start", 1) is None else "pf_block"
            S.sync()
            S.stack.append([name, 0.0])
            t0 = time.perf_counter()
            err = None
            try:
                return fn(*a, **k)
            except BaseException as exc:
                err = f"{type(exc).__name__}: {str(exc)[:200]}"
                raise
            finally:
                S.sync()
                dt = time.perf_counter() - t0
                _, child = S.stack.pop()
                path = ">".join(n for n, _ in S.stack) + (">" if S.stack else "") + name
                c = S.calls[path]
                c[0] += 1; c[1] += dt; c[2] += child
                if S.stack:
                    S.stack[-1][1] += dt
                shp = next((tuple(x.shape) for x in a[1:] if hasattr(x, "shape")), None)
                S.f.write(json.dumps({"path": path, "s": round(dt, 5), "shape": shp, **({"err": err} if err else {})})
                          + "\n")
        return inner

    def install(self):
        import ttnn
        for mod, attr, fam in TARGETS:
            m = sys.modules.get(mod) or __import__(mod, fromlist=["_"])
            owner, _, name = attr.rpartition(".")
            if owner:
                cls = getattr(m, owner)
                setattr(cls, name, self.wrap(fam, getattr(cls, name)))
                continue
            orig = getattr(m, name)
            w = self.wrap(fam, orig)
            homes = [ttnn] if mod == "ttnn" else [v for k, v in sys.modules.items() if k.startswith("tt_bio") and v]
            for h in homes:
                if getattr(h, name, None) is orig:
                    setattr(h, name, w)

    def summary(self):
        rows = {p: {"n": n, "s": round(s, 3), "self_s": round(s - ch, 3)} for p, (n, s, ch) in self.calls.items()}
        return dict(sorted(rows.items(), key=lambda kv: -kv[1]["s"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--chip", required=True, type=int)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="mol1")
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--skip-timed", action="store_true", help="attribution fold only (after the cold one)")
    ap.add_argument("--sections-first", action="store_true",
                    help="attribution fold only, as the first fold: the on-disk kernel cache makes it read like a warm "
                         "one (1024: cold 1275 s, warm 1323 s), and it halves the time a recall can cut it short")
    ap.add_argument("--trace-cold", action="store_true",
                    help="DRAM trace from the cold fold on, and DRAM in use before each fold: what a fold leaves behind")
    ap.add_argument("--peak-tensors", action="store_true",
                    help="no trace: sample a tag only when it reaches a new peak, and list the device tensors alive "
                         "then (a gc walk per sample, too slow for every sample)")
    a = ap.parse_args()
    os.environ["TT_VISIBLE_DEVICES"] = os.environ["TT_BIO_LEASE_CARDS"] = str(a.chip)
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import torch
    from tt_bio import worker as W

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "runs.jsonl", "a")
    emit = lambda r: (log.write(json.dumps(r) + "\n"), log.flush(), print(json.dumps(r), flush=True))
    cfg, jobs = fair.run_config(a.model, Path(a.input).resolve(), out, [])
    W._ensure_local_artifacts(cfg)
    state = W._WorkerState("tenstorrent")
    t0 = time.time()
    state.load_model(cfg)
    state.bind_run("mol1-wh1024-prof", cfg)
    emit({"phase": "load", "wall_s": round(time.time() - t0, 2), "work": {k: cfg.get(k) for k in fair.WORK},
          "sha": fair.git("rev-parse", "HEAD"), "chip": a.chip, "input": a.input})
    S = Sections(out)
    S.install()
    plan = ["sections"] if a.sections_first else ["cold"] + ([] if a.skip_timed else ["timed"]) + ["sections"]
    for i, kind in enumerate(plan):
        S.on = kind == "sections"
        if S.on:
            S.f = open(out / "sections.jsonl", "w", buffering=1)  # line-buffered: a recalled chip leaves the cycles it ran
        if S.on or a.trace_cold:
            os.environ["TT_BIO_DRAM_PEAK"] = str(out / "dram_peak.txt")  # read per call; drains like the syncs
            if a.peak_tensors:
                os.environ["TT_BIO_DRAM_PEAK_TENSORS"] = "1"
            else:
                os.environ["TT_BIO_DRAM_PEAK_TRACE"] = "1"  # every sample, so a refusal is placed in its cycle
        if a.trace_cold:
            import gc
            import ttnn
            from tt_bio.tenstorrent import dram_free, get_device
            gc.collect()
            mv = ttnn.get_memory_view(get_device(), ttnn.BufferType.DRAM)
            emit({"phase": f"before-{kind}", "dram_allocated_per_bank": int(mv.total_bytes_allocated_per_bank),
                  "dram_largest_free_B": dram_free()})
        sdir = out / f"r{i}-{kind}"
        sdir.mkdir(parents=True, exist_ok=True)
        with clocksample.during(period=1.0) as clk:
            t = time.time()
            try:
                metrics, _, _ = state.predict_one(jobs[0], dict(cfg, seed=a.seed, struct_dir=str(sdir)))
            except RuntimeError as exc:
                if not a.trace_cold:
                    raise
                emit({"phase": kind, "round": i, "wall_s": round(time.time() - t, 2), "err": str(exc)[:600]})
                return
            wall = time.time() - t
        cs = clk.summary()
        emit({"phase": kind, "round": i, "wall_s": round(wall, 2), "tokens": metrics.get("n_tokens"),
              "aiclk": cs, "aiclk_line": clk.line(0), "plddt": metrics.get("plddt"), "ptm": metrics.get("ptm")})
        if S.on:
            S.f.close()
            (out / "summary.json").write_text(json.dumps({"wall_s": round(wall, 2), "aiclk": cs,
                                                          "paths": S.summary()}, indent=1))


if __name__ == "__main__":
    main()
