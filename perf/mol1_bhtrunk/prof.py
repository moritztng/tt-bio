"""Trunk op-family attribution for Mol-1 on one Blackhole card: a real serving-path fold with a device sync around
every trunk sub-op, so each family's seconds are wall time the device spent on it.

    cd <inputs> && TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 python perf/mol1_bhtrunk/prof.py \
        --input mol1-512.json --out <dir> [--warm 2]

Rounds: one cold fold (discarded), one bare warm fold (the honest wall), then --warm instrumented folds. The syncs
remove host/device overlap, so instrumented totals run above the bare wall; read the families as shares and as
seconds of device work, and time levers with perf/mol1/fair/run.py, never with this. Keys are class:variant@width,
inclusive (TRUNK includes its blocks). AICLK sampled during every fold.
"""
import argparse
import collections
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mol1" / "fair"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--seed", type=int, default=101)
    a = ap.parse_args()
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    import clocksample
    from run import run_config
    from tt_bio import worker as W, tenstorrent as T, protenix as P

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    acc = collections.defaultdict(lambda: [0.0, 0])
    on = [False]

    def sync():
        ttnn.synchronize_device(T.get_device())

    def width(args):
        for x in args:
            if hasattr(x, "shape") and len(x.shape):
                return f"{x.shape[-1]}x{x.shape[-2]}" if len(x.shape) >= 2 else str(x.shape[-1])
        return "?"

    def wrap(cls, meth, name, variant=lambda self, args: ""):
        f = getattr(cls, meth)

        def g(self, *args, **kw):
            if not on[0] or T.CAPTURING[0]:  # no sync inside a trace capture (TT_FATAL)
                return f(self, *args, **kw)
            sync()
            t = time.perf_counter()
            r = f(self, *args, **kw)
            sync()
            k = f"{name}{variant(self, args)}@{width(args)}"
            acc[k][0] += time.perf_counter() - t
            acc[k][1] += 1
            return r
        setattr(cls, meth, g)

    wrap(T.TriangleMultiplication, "__call__", "trimul", lambda s, a_: ":in" if s.ending else ":out")
    wrap(T.TriangleAttention, "__call__", "triatt", lambda s, a_: ":end" if s.ending else ":start")
    wrap(T.Transition, "__call__", "transition")
    wrap(T.AttentionPairBias, "__call__", "apb")
    wrap(T.OuterProductMean, "__call__", "opm")
    wrap(T.PairWeightedAveraging, "__call__", "pwa")
    wrap(T.Pairformer, "_adapt", "adapter", lambda s, a_: f":{a_[1]}")
    wrap(T.Pairformer, "__call__", "PF_stack")
    wrap(T.PairformerLayer, "__call__", "PF_layer",
         lambda s, a_: ":mixer" if s.triangle_attention_start is None else ":pf")
    wrap(P.Trunk, "_msa", "TRUNK_msa")
    wrap(P.Trunk, "_template", "TRUNK_template")
    wrap(P.Trunk, "__call__", "TRUNK")
    wrap(P.DiffusionModule, "__call__", "DIFFUSION")
    wrap(P.ConfidenceHead, "__call__", "CONFIDENCE")

    cfg, jobs = run_config("mol1", Path(a.input).resolve(), out)
    W._ensure_local_artifacts(cfg)
    st = W._WorkerState("tenstorrent")
    st.load_model(cfg)
    st.bind_run("mol1-bhtrunk", cfg)
    log = open(out / "prof.jsonl", "a")
    for i in range(2 + a.warm):
        kind = "cold" if i == 0 else "bare" if i == 1 else "instrumented"
        on[0] = kind == "instrumented"
        acc.clear()
        sdir = out / f"r{i}"
        sdir.mkdir(exist_ok=True)
        with clocksample.during() as clk:
            t0 = time.time()
            m, _, _ = st.predict_one(jobs[0], dict(cfg, seed=a.seed + i, struct_dir=str(sdir)))
            wall = time.time() - t0
        rec = {"round": i, "kind": kind, "input": a.input, "tokens": m.get("n_tokens"), "wall_s": round(wall, 3),
               "aiclk": clk.summary(),
               "work": {k: cfg.get(k) for k in ("recycling_steps", "sampling_steps", "diffusion_samples")},
               "families": {k: [round(v[0], 4), v[1]] for k, v in sorted(acc.items(), key=lambda kv: -kv[1][0])}}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(kind, rec["wall_s"], clk.line(0), flush=True)


if __name__ == "__main__":
    main()
