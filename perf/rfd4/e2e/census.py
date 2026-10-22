"""Where a 128-token RFD4 design's time goes, by op family, on one Blackhole card.

    TT_VISIBLE_DEVICES=<card> python perf/rfd4/e2e/census.py --spec <spec.yaml> --weights <bundle> \
        --out state/rfd4/census/e2e-128.json

**Instrument, and what it cannot do.** The device profiler is the right tool for per-op device time,
and it is not available here: the ttnn wheel on qb2 is not a Tracy-enabled build, so
`TT_METAL_DEVICE_PROFILER=1` dies at `rtoptions.cpp:708` before the card opens. What this uses
instead is the instrument the profiling skill names for the coarse question: host wall-clock around
named stages with `ttnn.synchronize_device` on both sides of every one. That is honest about *which
family* the time is in and it is not free -- a sync per sublayer serialises a queue the real run
overlaps -- so the run is done twice. The uninstrumented design is the truth about total time; the
instrumented one is the truth about proportions, and the two totals are reported side by side so the
cost of measuring is visible rather than hidden.

Families are the model's, not ttnn's: `pair_write` (the token stream's two projections into the pair
stream), `tri_mult_out`, `tri_mult_in`, `pair_transition` -- together the pair track -- against
`attention`, `pair_bias` and `token_transition`, the token track (`pair_bias` is the attention's bias
from z, which the trunk builds once per z for the two blocks that read it; add it to `attention`
to compare with a census taken before that). What neither timer catches (residual adds,
masking, dtype casts, the recycle projections) falls out as `unattributed` by subtraction rather than
being quietly dropped.

FLOPs per family are ARCH.md section 8's model at n=128, so the measured share can be put beside the
91-98 % of FLOPs that document claims for the pair track.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # perf/ for clocksample

import torch

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()

import ttnn  # noqa: E402

import clocksample  # noqa: E402

TIMES: dict[str, float] = defaultdict(float)
COUNTS: dict[str, int] = defaultdict(int)
SYNCS = [0]


def instrument(engine, dev):
    """Wrap one method per family in a synced timer. Returns a function that undoes it."""
    from tt_bio.rfd4 import trunk as TR

    targets = [(TR.AttnADALN, "__call__", "attention"),
               (TR.TransitionADALN, "__call__", "token_transition"),
               (TR.Trunk, "_group_biases", "pair_bias"),
               (TR.PairUpdate, "_write", "pair_write"),
               (TR.PairUpdate, "_transition", "pair_transition")]
    originals = []

    def wrap(cls, name, family):
        inner = getattr(cls, name)

        def timed(self, *args, **kwargs):
            ttnn.synchronize_device(dev)
            SYNCS[0] += 1
            t0 = time.perf_counter()
            out = inner(self, *args, **kwargs)
            ttnn.synchronize_device(dev)
            SYNCS[0] += 1
            TIMES[family] += time.perf_counter() - t0
            COUNTS[family] += 1
            return out

        setattr(cls, name, timed)
        originals.append((cls, name, inner))

    for cls, name, family in targets:
        wrap(cls, name, family)

    # tri_out / tri_in are instance attributes (TriangleMultiplication objects), not methods on the
    # class, so they are wrapped per PairUpdate instance as the trunk is walked.
    for pu in engine.trunk.pair_updates.values():
        for attr, family in (("tri_out", "tri_mult_out"), ("tri_in", "tri_mult_in")):
            mod = getattr(pu, attr)

            def timed(*args, _m=mod, _f=family, **kwargs):
                ttnn.synchronize_device(dev)
                SYNCS[0] += 1
                t0 = time.perf_counter()
                out = _m(*args, **kwargs)
                ttnn.synchronize_device(dev)
                SYNCS[0] += 1
                TIMES[_f] += time.perf_counter() - t0
                COUNTS[_f] += 1
                return out

            setattr(pu, attr, timed)

    def undo():
        for cls, name, inner in originals:
            setattr(cls, name, inner)

    return undo


#: ARCH.md section 8, per trunk pass at n=128, in GFLOP. The pair block numbers are that document's
#: own arithmetic (tri-mult in-projection 12.9 x2, pair transition 19.3, linear_g 4.8 x2,
#: linear_z 3.2 x2 = 61 GFLOP per pair block, 7 blocks), split here by family.
def arch_gflops(n: int, nlayers: int, n_pair: int) -> dict:
    tok, c_z, c_h, c_pair_hidden = 768, 384, 256, 1536
    f = lambda rows, k, m: 2 * rows * k * m / 1e9  # noqa: E731
    per_block = {
        "attention": f(n, tok, 3 * tok) + f(n, tok, tok) + 2 * 2 * n * n * tok / 1e9,
        "token_transition": f(n, tok, 2 * 2 * tok) + f(n, 2 * tok, tok),
    }
    per_pair = {
        "pair_write": 2 * f(n * n, tok, c_z),                  # linear_g and linear_z
        "tri_mult_out": f(n * n, c_z, 2 * c_h) + f(n * n, c_h, c_z) + 2 * c_h * n ** 3 / 1e9,
        "tri_mult_in": f(n * n, c_z, 2 * c_h) + f(n * n, c_h, c_z) + 2 * c_h * n ** 3 / 1e9,
        "pair_transition": f(n * n, c_z, c_pair_hidden) + f(n * n, c_pair_hidden // 2, c_z),
    }
    out = {k: v * nlayers for k, v in per_block.items()}
    out.update({k: v * n_pair for k, v in per_pair.items()})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default="/home/ttuser/rfd4-e2e-runs/census")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from tt_bio import rfd4
    from tt_bio.rfd4 import engine as E

    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    designs = rfd4.prepare_examples(a.spec, work / "inputs")
    features, _ = rfd4.featurize(designs[0].cif, seed=a.seed)
    n = int(features["padding_mask"].shape[-1])

    engine = E.DesignEngine(a.weights, recipe="thinking_medium")
    dev = engine._device
    nlayers = engine.config["nn"]["nlayers"]
    n_pair = len(engine.trunk.pair_updates)

    # Leg 1: compile, thrown away. Leg 2: the warm, uninstrumented truth about total time.
    engine.design(features, num_designs=1, nsteps=engine.spec.nsteps, seed=a.seed, rng="upstream")
    trunk_s = []
    inner = engine.trunk.run_bucketed

    def timed_trunk(*args, **kwargs):
        t = time.perf_counter()
        r = inner(*args, **kwargs)
        trunk_s.append(time.perf_counter() - t)
        return r

    engine.trunk.run_bucketed = timed_trunk
    with clocksample.during(period=1.0) as clk:
        t0 = time.perf_counter()
        engine.design(features, num_designs=1, nsteps=engine.spec.nsteps, seed=a.seed, rng="upstream")
        warm_s = time.perf_counter() - t0
    warm_trunk_s, warm_aiclk = sum(trunk_s), clk.summary()

    # Leg 3: the same design with every family synced on both sides.
    undo = instrument(engine, dev)
    trunk_s.clear()
    with clocksample.during(period=1.0) as clk2:
        t0 = time.perf_counter()
        engine.design(features, num_designs=1, nsteps=engine.spec.nsteps, seed=a.seed, rng="upstream")
        instr_s = time.perf_counter() - t0
    undo()
    instr_trunk_s = sum(trunk_s)

    gf = arch_gflops(n, nlayers, n_pair)
    passes = engine.spec.nsteps * 3  # thinking_medium: 180 passes over 60 steps
    attributed = sum(TIMES.values())
    pair_fams = ("pair_write", "tri_mult_out", "tri_mult_in", "pair_transition")
    token_fams = ("attention", "pair_bias", "token_transition")
    pass_gf = sum(gf.values())

    families = {}
    for fam, s in sorted(TIMES.items(), key=lambda kv: -kv[1]):
        total_gf = gf.get(fam, 0.0) * passes
        families[fam] = {"device_s": round(s, 3), "calls": COUNTS[fam],
                         "share_of_trunk": round(s / instr_trunk_s, 4),
                         "ms_per_call": round(s / COUNTS[fam] * 1e3, 3),
                         "gflop_per_pass": round(gf.get(fam, 0.0), 2),
                         "tflop_total": round(total_gf / 1e3, 2),
                         "tflops": round(total_gf / 1e3 / s, 2) if s else None}
    families["unattributed"] = {
        "device_s": round(instr_trunk_s - attributed, 3),
        "share_of_trunk": round((instr_trunk_s - attributed) / instr_trunk_s, 4),
        "what": "residual adds, masking, dtype casts, the recycle projections and per-op dispatch"}

    pair_t = sum(TIMES[f] for f in pair_fams)
    token_t = sum(TIMES[f] for f in token_fams)
    pair_gf = sum(gf[f] for f in pair_fams)
    out = {
        "size": n, "recipe": "thinking_medium", "steps": engine.spec.nsteps, "trunk_passes": passes,
        "nlayers": nlayers, "pair_updates_per_pass": n_pair, "batch": 1, "designs": 1,
        "instrument": ("synced host timers per family; the ttnn wheel here is not a Tracy build, so "
                       "the device profiler (TT_METAL_DEVICE_PROFILER=1) cannot run on this host"),
        "warm_uninstrumented": {"design_s": round(warm_s, 3), "trunk_s": round(warm_trunk_s, 3),
                                "host_s": round(warm_s - warm_trunk_s, 3), "aiclk": warm_aiclk},
        "instrumented": {"design_s": round(instr_s, 3), "trunk_s": round(instr_trunk_s, 3),
                         "syncs": SYNCS[0], "aiclk": clk2.summary(),
                         "cost_of_measuring_x": round(instr_s / warm_s, 2)},
        "families": families,
        "tracks": {
            "pair": {"device_s": round(pair_t, 3),
                     "share_of_attributed_time": round(pair_t / attributed, 4),
                     "share_of_flops": round(pair_gf / pass_gf, 4)},
            "token": {"device_s": round(token_t, 3),
                      "share_of_attributed_time": round(token_t / attributed, 4),
                      "share_of_flops": round((pass_gf - pair_gf) / pass_gf, 4)}},
        "flops": {"tflop_per_pass": round(pass_gf / 1e3, 4),
                  "tflop_per_design": round(pass_gf * passes / 1e3, 1),
                  "achieved_tflops_warm": round(pass_gf * passes / 1e3 / warm_trunk_s, 2),
                  "arch_claim": "ARCH.md section 8: 0.496 TFLOP/pass and 89.3 TFLOP/design at n=128"},
        "bytes": {"weights_per_pass_mb": 350.1, "weights_per_design_gb": round(350.1 * passes / 1e3, 1),
                  "dram_roof_gb_s": 435.2, "sustained_gb_s": 391,
                  "weight_read_floor_s_per_design": round(350.1e6 * passes / 391e9, 2),
                  "note": "ARCH section 8: 175,074,048 params = 350 MB bf16 per pass, batch 1"},
        "arch_pair_track_claim": "91-98 % of FLOPs and 93-99 % of bytes",
    }
    share = out["tracks"]["pair"]["share_of_flops"]
    out["agrees_with_arch"] = bool(0.91 <= share <= 0.98)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
