#!/usr/bin/env python3
"""Score saved OF3T training checkpoints on the held-out corpus, many times, to measure the eval.

`perf/of3t_p10trainout/trainarm.py::_evaluate` is the instrument every OF3T training A/B has been
graded on. It draws ONE diffusion noise level and ONE per-atom noise from `--eval-seed`, so each
target's `mse` and `smooth_lddt` are a single draw. This script calls that same `_evaluate` on
fixed weights, repeatedly and across eval seeds, so its noise is measured instead of assumed.

One process builds the model once, discovers the trainable set the way `train_loop` does, and then
for each `--adapter` loads its fp32 masters (`checkpoint.load_adapter` + `rebind`) and evaluates.
`--adapter base` puts the start checkpoint's weights back.

`--context shipped` scores outside `install()`, where `eval_before` and `eval_after` ran.
`--context install` scores inside it, where trainarm's inline `--eval-every` scores ran.

    evalnoise.py --train-corpus <dir> --eval-corpus <dir> --checkpoint <of3.pt> \
        --adapter base --adapter runs/X12b/adapter-00000011.safetensors \
        --seeds 20260926,0-31 --repeat 2 --out out/noise.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from perf.clocksample import during                                   # noqa: E402
from perf.of3t_p10trainout.trainarm import _evaluate, _git            # noqa: E402


def _seeds(spec):
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def _by_file_names(path, params):
    """`params` keyed the way the adapter at `path` names them.

    Four diffusion weights (`sampler.dm._wc.<n>`) are named by the `id()` of a Python object,
    so every process names them differently and `load_adapter` refuses a checkpoint written by
    another process. Their sizes are distinct, so each is matched by element count. Every other
    name must agree exactly, which `load_adapter` still checks.
    """
    import math
    import re
    with open(path, "rb") as fh:
        n = int.from_bytes(fh.read(8), "little")
        header = json.loads(fh.read(n))
    idn = re.compile(r"^(.*\._wc)\.\d+$")
    have = {k.split("|", 1)[1]: v["shape"] for k, v in header.items() if k.startswith("master|")}
    out = {k: t for k, t in params.items() if k in have}
    loose = {}
    for name in set(have) - set(params):
        m = idn.match(name)
        if m:
            loose.setdefault((m.group(1), math.prod(have[name])), []).append(name)
    for name in set(params) - set(have):
        m = idn.match(name)
        size = math.prod(tuple(params[name].value.shape)) if m else None
        # A tile-padded device shape would break this match; it then stays unmatched and
        # load_adapter names it.
        cand = loose.get((m.group(1), size), []) if m else []
        if len(cand) == 1:
            out[cand[0]] = params[name]
        else:
            out[name] = params[name]
    return out


def _trimul_state(model):
    """Every TriangleMultiplication in `model`: how many read device leaves, how many cache.

    `train_in_projections` gives each trimul device leaves for its in-projection, and then
    `_gp_in_chunks` cuts from them per call and caches nothing. A trimul without them would cut
    from the START checkpoint's host copy and cache it per width, so a loaded checkpoint would
    run start-weight in-projections at every width it had not seen at discovery.
    """
    import ttnn
    from tt_bio.tenstorrent import _WALK_OPAQUE, TriangleMultiplication
    seen, stack, n, host, cached = set(), [model], 0, 0, 0
    while stack:
        o = stack.pop()
        if id(o) in seen or isinstance(o, _WALK_OPAQUE + (ttnn.Tensor,)):
            continue
        seen.add(id(o))
        if isinstance(o, TriangleMultiplication):
            n += 1
            host += o.g_in_weight is None
            cached += len(o._gp_cache) + len(o._gp_gout_cache)
        if isinstance(o, dict):
            stack += o.values()
        elif isinstance(o, (list, tuple)):
            stack += o
        elif hasattr(o, "__dict__"):
            stack += vars(o).values()
    return {"trimuls": n, "host_in_proj": host, "cache_entries": cached}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-corpus", required=True, type=Path,
                    help="discovery batch, the same corpus the checkpoints trained on")
    ap.add_argument("--eval-corpus", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--adapter", action="append", required=True,
                    help="safetensors path, or `base` for the start checkpoint; repeatable")
    ap.add_argument("--label", action="append", default=[],
                    help="one per --adapter, defaults to the run directory name")
    ap.add_argument("--seeds", default="20260926")
    ap.add_argument("--warmup", type=int, default=1,
                    help="discarded calls after each load, on the loaded weights")
    ap.add_argument("--repeat", type=int, default=1, help="calls per seed, back to back")
    ap.add_argument("--context", action="append", choices=("shipped", "install"), default=[])
    ap.add_argument("--stage", default="initial_training")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    contexts = a.context or ["shipped"]
    seeds = _seeds(a.seeds)
    labels = a.label or [("base" if p == "base" else Path(p).parent.name) for p in a.adapter]
    if len(labels) != len(a.adapter):
        raise SystemExit("--label count must match --adapter count")

    rec = {"argv": sys.argv[1:], "head": _git("rev-parse", "HEAD"),
           "dirty": bool(_git("status", "--porcelain")),
           "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "seeds": seeds, "repeat": a.repeat, "contexts": contexts, "calls": []}
    a.out.parent.mkdir(parents=True, exist_ok=True)

    def dump():
        a.out.write_text(json.dumps(rec, indent=2, default=str) + "\n")

    t0 = time.perf_counter()
    with during() as clk:
        try:
            from tt_bio import autograd as ag
            from tt_bio.train import openfold3 as of3
            from tt_bio.train.checkpoint import load_adapter
            from tt_bio.train.lora import trainable

            fwd, ds = of3.adapter(a.train_corpus, checkpoint=a.checkpoint, rollout=20,
                                  num_cycles=1, seed=0)
            # Discovery as train_loop runs it: under install(), on a training batch. The walk
            # names every weight the model reaches, which is the set load_adapter insists on.
            # `exact_training` defaults ON, and `install()` arms whatever it reports, so without
            # this the discovery forward runs the host float64 softmax: ~30 min instead of ~1.
            # trainarm's device arms discover under `exact_training(False)`; so does this.
            with ag.exact_training(False):
                ag.install()
                try:
                    _, params = trainable(fwd, None, ds.device, ds.batch([0]),
                                          model=getattr(fwd, "model", None), rng=0)
                finally:
                    ag.uninstall()
            rec["params"] = len(params)
            # The start weights' device handles. `load_adapter` assigns new tensors and never
            # writes into these, so `base` after an adapter puts the checkpoint back exactly.
            start = {name: t.value for name, t in params.items()}
            dump()
            for path, label in zip(a.adapter, labels):
                t1 = time.perf_counter()
                if path == "base":
                    for name, t in params.items():
                        t.value = start[name]
                else:
                    load_adapter(path, _by_file_names(path, params), ds.device)
                moved = params.rebind()
                print(f"[load] {label} {path} rebind {moved} "
                      f"{time.perf_counter() - t1:.1f}s", flush=True)
                # The FIRST forward at a shape is not the forward every later call at that
                # shape runs (the start weights read 7ohe 10.762624 cold and 18.871191 on every
                # call after, out/N1_repro.json). Discarded calls on THESE weights put every
                # scored call in the warm state; a warm-up on other weights is not the same
                # thing (out/N3a.json vs out/N3b.json).
                for _ in range(a.warmup):
                    with ag.exact_training(False):
                        ev = _evaluate(fwd, a.eval_corpus, a.stage, seeds[0])
                    rec.setdefault("warmup", []).append(
                        {"label": label, **{t["pdb_id"]: t["loss"] for t in ev["targets"]}})
                for ctx in contexts:
                    for seed in seeds:
                        for rep in range(a.repeat):
                            with ag.exact_training(False):
                                if ctx == "install":
                                    ag.install()
                                try:
                                    ev = _evaluate(fwd, a.eval_corpus, a.stage, seed)
                                finally:
                                    if ctx == "install":
                                        ag.uninstall()
                            row = {"label": label, "adapter": path, "context": ctx,
                                   "seed": seed, "rep": rep, "mean_loss": ev["mean_loss"],
                                   "targets": {t["pdb_id"]: {"loss": t["loss"],
                                                             **t["breakdown"]}
                                               for t in ev["targets"]}}
                            rec["calls"].append(row)
                            print(f"[noise] {label} {ctx} seed {seed} rep {rep} "
                                  + " ".join(f"{k} {v['loss']:.6f}"
                                             for k, v in row["targets"].items()), flush=True)
                            dump()
            rec["trimul"] = _trimul_state(getattr(fwd, "model", fwd))
            print(f"[trimul] {rec['trimul']}", flush=True)
            rec["ok"] = True
        except BaseException as exc:                                   # noqa: BLE001
            import traceback
            rec["ok"] = False
            rec["error"] = f"{type(exc).__name__}: {exc}"
            rec["traceback"] = traceback.format_exc()
            raise
        finally:
            rec["wall_s"] = round(time.perf_counter() - t0, 3)
            rec["aiclk"] = clk.summary()
            rec["aiclk_line"] = clk.line()
            dump()
    print(rec["aiclk_line"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
