"""Lever A: the trimul chunk width at BoltzGen's own production shape, both arms in one process.

`_trimul_chunk_size` doubles the hidden-channel chunk while `batch * (c*2) * seq_len^2` fits a
budget that scales with core count. At BoltzGen's padded 256 that budget is 5,545,354 elements on a
110-core Blackhole grid and 3,629,993 on a 72-core Wormhole grid, so Blackhole runs chunk 64 and
Wormhole runs chunk 32 -- twice the chunk-loop iterations, each carrying one fused input matmul, a
4-way split, three channel moves and a concat, on a trunk whose binding limit is per-op fixed cost.

The measurement problem this script solves is the box, not the model. The Wormhole Galaxy is
production and its loadavg moves on prod's schedule, so arm-A-then-arm-B is not a slower measurement
than interleaving, it is a wrong one. Both arms therefore run inside ONE process and one model load,
alternating per design: the chunk width is recomputed on every trimul call, so flipping a module
global at each design boundary switches arms with nothing else changing. The first design of each
arm is dropped (it carries that arm's kernel compile) and the median is taken over the rest.

Correctness is checked on the module rather than on the output structure: designs differ by their
noise draw, so two designs are not comparable, but one trimul module called twice on ONE captured
input is. The first live call at the production shape is stashed and replayed under both arms with
`torch.equal`.

    WTC_CHUNKS=32,64 WTC_RUNG=R0 WTC_DESIGNS=10 TT_VISIBLE_DEVICES=<umd> \
      PYTHONPATH=$PWD python3 perf/whdesign/wh_trimul_chunk_ab.py
"""
import json
import os
import pathlib
import statistics
import sys
import time

sys.path.insert(0, os.getcwd())

import torch
import ttnn

import tt_bio.tenstorrent as T
import tt_bio.boltzgen.progress as P

CHUNKS = [int(c) for c in os.environ.get("WTC_CHUNKS", "32,64").split(",")]
RUNG = os.environ.get("WTC_RUNG", "R0")
DESIGNS = int(os.environ.get("WTC_DESIGNS", "10"))
STEPS = int(os.environ.get("WTC_STEPS", "60"))
OUT = pathlib.Path(os.environ.get("WTC_OUT", "perf/whdesign/results/wh_trimul_chunk_ab.json"))
HOST = os.environ.get("WTC_HOST", "galaxy")

state = {"arm": CHUNKS[0], "design": 1, "stamps": {}, "widths": {}, "capture": None}
_orig_chunk = T._trimul_chunk_size


def forced(seq_len, hidden, batch=1):
    """The shipped rule, except that the L1 path's width is pinned to the current arm.

    Only the L1 path is pinned: above `_trimul_l1_max_seq` the shipped code does not widen at all,
    and forcing a width there would be measuring a different change from the one that would ship.
    """
    shipped = _orig_chunk(seq_len, hidden, batch)
    if seq_len > T._trimul_l1_max_seq():
        return shipped
    c = state["arm"]
    if hidden % c or (hidden // c) * c != hidden:
        return shipped
    state["widths"].setdefault((seq_len, hidden, batch), {})[c] = shipped
    return c


T._trimul_chunk_size = forced

_orig_call = T.TriangleMultiplication.__call__


def tapped_call(self, x, mask=None):
    if state["capture"] is None and int(x.shape[1]) > 128:
        state["capture"] = (self, ttnn.to_torch(x).clone(), None if mask is None else ttnn.to_torch(mask).clone())
    return _orig_call(self, x, mask)


T.TriangleMultiplication.__call__ = tapped_call

_orig_set = P.set_reporter


class Tap(P.Reporter):
    """Wraps whatever reporter the CLI installs, so no model code has to be patched to see events."""

    def __init__(self, inner):
        self.inner = inner

    def stage_start(self, name, idx, total):
        self.inner.stage_start(name, idx, total)

    def stage_done(self, ok=True):
        self.inner.stage_done(ok)

    def step(self, kind, n, total):
        if kind == "batch":
            state["design"] = n
            state["arm"] = CHUNKS[(n - 1) % len(CHUNKS)]
        elif kind == "diffusion":
            state["stamps"].setdefault((state["design"], state["arm"]), []).append(time.perf_counter())
        self.inner.step(kind, n, total)


P.set_reporter = lambda r: _orig_set(Tap(r))

sys.argv = ["tt_bio", "design", "perf/dsfix/fixtures/bg_%s.yaml" % RUNG,
            "--model", "boltzgen", "--steps", "design",
            "--num_designs", str(DESIGNS), "--out_dir", "/tmp/wtc_%s" % RUNG,
            "--config", "design", "sampling_steps=%d" % STEPS,
            "--debug", "--log"]

from tt_bio.main import cli as app  # noqa: E402

t0 = time.time()
try:
    app(standalone_mode=False)
except SystemExit:
    pass
wall = time.time() - t0

# ---- per-arm step medians, first design of each arm dropped ------------------------------------
per_design = {}
for (d, arm), ts in sorted(state["stamps"].items()):
    if len(ts) < 8:
        continue
    steps = [ts[i + 1] - ts[i] for i in range(3, len(ts) - 1)]
    per_design[(d, arm)] = statistics.median(steps) * 1000.0

arms = {}
for c in CHUNKS:
    ds = sorted(d for (d, a) in per_design if a == c)
    kept = ds[1:]                       # drop this arm's first design: it carries kernel compile
    vals = [per_design[(d, c)] for d in kept]
    arms[c] = {"chunk": c, "designs_kept": kept, "designs_all": ds,
               "step_ms_per_design": [round(per_design[(d, c)], 3) for d in ds],
               "step_ms_median": round(statistics.median(vals), 3) if vals else None,
               "n_designs": len(vals)}

# ---- module-level bit-exactness on one captured input ------------------------------------------
# The timing is the deliverable and the replay needs a live device the CLI may already have closed,
# so a failed replay records itself and does not take the run down with it.
exact = {"checked": False}
try:
    tm, x_h, m_h = state["capture"]
    dev = T.get_device()
    xt = ttnn.from_torch(x_h, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    mt = None if m_h is None else ttnn.from_torch(m_h, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    outs = {}
    for c in CHUNKS:
        state["arm"] = c
        o = _orig_call(tm, xt, mt)
        ttnn.synchronize_device(dev)
        outs[c] = ttnn.to_torch(o).clone()
        ttnn.deallocate(o)
    ref = outs[CHUNKS[0]]
    exact = {"checked": True, "shape": [int(v) for v in x_h.shape],
             "ref_chunk": CHUNKS[0],
             "torch_equal": {str(c): bool(torch.equal(outs[c], ref)) for c in CHUNKS},
             "max_abs": {str(c): float((outs[c] - ref).abs().max()) for c in CHUNKS}}
except Exception as e:                                    # noqa: BLE001
    exact = {"checked": False, "error": "%s: %s" % (type(e).__name__, e)}

rec = {"host": HOST, "rung": RUNG, "designs": DESIGNS, "sampling_steps": STEPS,
       "chunks": CHUNKS, "grid": list(T.COMPUTE_GRID_MAIN),
       "l1_chunk_budget": T.TRIANGLE_MULT_L1_CHUNK_BUDGET,
       "widths_seen": {str(k): v for k, v in state["widths"].items()},
       "arms": {str(c): arms[c] for c in CHUNKS},
       "loadavg": float(open("/proc/loadavg").read().split()[0]),
       "proc_wall_s": round(wall, 1), "bit_exact": exact}
if all(arms[c]["step_ms_median"] for c in CHUNKS):
    rec["speedup_vs_%d" % CHUNKS[0]] = {
        str(c): round(arms[CHUNKS[0]]["step_ms_median"] / arms[c]["step_ms_median"], 4) for c in CHUNKS}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(rec, indent=1) + "\n")
print("[wtc] " + json.dumps({k: rec[k] for k in ("grid", "chunks", "arms", "bit_exact") if k in rec}), flush=True)
print("[wtc] wrote %s" % OUT, flush=True)
