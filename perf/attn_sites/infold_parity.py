#!/usr/bin/env python3
"""Decide the in0_block_w bit-exactness rule on LIVE in-fold operands, and count calls per unit.

`rfd3_esm_replay.py` established the rule on synthetic operands of each site's padded shape:

    an arm is bit-exact against the shipped path iff its in0_block_w equals the naive path's.

That is not enough to ship. With one block per core the reuse factory's batch-stride increment
never fires, so an op-level pass clears a config for the one shape it tested and nothing else.
This harness runs the arms on the operands a real fold actually produces, inside the fold, and
compares every arm to the value the model would have got.

What it does per matmul call, for the target sites only:

  * returns the UNMODIFIED `auto` result to the model, so the fold it is riding on stays correct
    and its output is the shipped output;
  * for the first `--probes` calls of each (site, shape) class, re-runs the same operands through
    every legal `MatmulMultiCoreReuseProgramConfig` arm and through `core_grid=`, and records
    `torch.equal` against that same `auto` result;
  * counts every call of every class, so per-unit call counts fall out of the same run.

The rule's prediction is falsifiable here and stated before the run: within a class, exactness
partitions STRICTLY by in0_block_w -- every arm sharing the naive path's value is exact, every arm
differing is not -- and `core_grid=` (which hard-codes k_tiles_per_core=1) is exact iff the naive
value is 1. A single class where two arms with the same in0_block_w disagree kills it.

    TT_VISIBLE_DEVICES=0,1 TT_BIO_LOGICAL_DEVICE_ID=0 TT_BIO_LEASE_HOLDER=... \
        python3 -u perf/attn_sites/infold_parity.py --out p.json -- predict examples/prot300.yaml \
            --model esmfold2 --accelerator tenstorrent
"""
from __future__ import annotations

import argparse
import atexit
import json
import math
import sys
from pathlib import Path

import torch
import ttnn

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rfd3_esm_replay import cb_bytes, legal_reuse, naive_config, subblock  # noqa: E402
from tt_bio.tenstorrent import CORE_GRID_MAIN  # noqa: E402

TILE = 32
TSIZE = {"BFLOAT16": 2048, "FLOAT32": 4096, "BFLOAT8_B": 1088}
L1_BUDGET = 1464 * 1024

# The sites this leg owns. Matched as a suffix of the innermost tt_bio frame, so a line drift of
# a few lines in a future tree shows up as "site not seen" rather than as a silent miss.
TARGET_TAILS = (
    "rfd3/model.py:491", "rfd3/model.py:501",
    "rfd3/model.py:1331", "rfd3/model.py:1346", "rfd3/model.py:1358",
    "esmfold2.py:71", "esmfold2.py:75",
)

CLASSES: dict[tuple, dict] = {}
STATE = {"on": False, "depth": 0, "probes": 2, "cores": 110}


def _site() -> str:
    f = sys._getframe(2)
    while f is not None:
        fn = f.f_code.co_filename
        if "/tt_bio/" in fn and "attn_sites" not in fn:
            return f"{fn.split('/tt_bio/')[-1]}:{f.f_lineno}"
        f = f.f_back
    return "?"


def _is_target(site: str) -> bool:
    return any(site.endswith(t) for t in TARGET_TAILS)


def _batch(p) -> int:
    return math.prod(p[:-2]) if len(p) > 2 else 1


def _dram_il(t) -> bool:
    mc = t.memory_config()
    return ("DRAM" in str(mc.buffer_type)) and ("INTERLEAVED" in str(mc.memory_layout))


def _probe(a, b, ref, kw, rec):
    """Run every legal arm on the live operands and record torch.equal against `ref`."""
    mt = a.padded_shape[-2] // TILE
    kt = a.padded_shape[-1] // TILE
    nt = b.padded_shape[-1] // TILE
    batch = max(_batch(list(a.padded_shape)), _batch(list(b.padded_shape)))
    ta = TSIZE.get(str(a.dtype).split(".")[-1].upper(), 2048)
    tb = TSIZE.get(str(b.dtype).split(".")[-1].upper(), 2048)
    tinterm = TSIZE["FLOAT32"]
    cores = STATE["cores"]

    pred = naive_config(mt, kt, nt, mt * TILE, nt * TILE, 13, 10, ta, tb, tinterm, L1_BUDGET)
    rec["naive_predicted"] = pred

    base = {k: v for k, v in kw.items() if k not in ("core_grid", "program_config")}
    arms = []

    def add(name, bw, **extra):
        try:
            out = ttnn.matmul(a, b, **base, **extra)
            eq = bool(torch.equal(ttnn.to_torch(out), ref))
            ttnn.deallocate(out)
        except Exception as exc:
            arms.append({"arm": name, "in0_block_w": bw, "error": str(exc)[:160]})
            return
        arms.append({"arm": name, "in0_block_w": bw, "exact": eq})

    # core_grid= hard-codes k_tiles_per_core = 1 (matmul_program_config.cpp:457).
    grid_fits = cb_bytes(mt, nt, 1, ta, tb, tinterm) < L1_BUDGET
    add("grid" if grid_fits else "grid(cbs-do-not-fit:silent-fallback)", 1, core_grid=CORE_GRID_MAIN)

    for pcm, bw, blocks in legal_reuse(mt, kt, nt, batch, cores, L1_BUDGET, ta, tb, tinterm):
        h, w = subblock(pcm, nt)
        cfg = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=(CORE_GRID_MAIN.x, CORE_GRID_MAIN.y),
            in0_block_w=bw, out_subblock_h=h, out_subblock_w=w,
            per_core_M=pcm, per_core_N=nt)
        add(f"reuse/pcm={pcm}/bw={bw}", bw, program_config=cfg)

    rec["arms"] = arms
    # The rule's verdict, computed here so a reader does not have to re-derive it from the table.
    ok = [x for x in arms if "exact" in x]
    by_bw = {}
    for x in ok:
        by_bw.setdefault(x["in0_block_w"], set()).add(x["exact"])
    rec["partitions_by_in0_block_w"] = all(len(v) == 1 for v in by_bw.values())
    rec["exact_in0_block_w"] = sorted(k for k, v in by_bw.items() if v == {True})
    rec["inexact_in0_block_w"] = sorted(k for k, v in by_bw.items() if v == {False})


def _wrap(fn):
    def inner(*a, **kw):
        if not STATE["on"] or STATE["depth"] or len(a) < 2:
            return fn(*a, **kw)
        ta, tb = a[0], a[1]
        if not (isinstance(ta, ttnn.Tensor) and isinstance(tb, ttnn.Tensor)):
            return fn(*a, **kw)
        site = _site()
        if not _is_target(site):
            return fn(*a, **kw)

        STATE["depth"] = 1
        try:
            out = fn(*a, **kw)
            key = (site, tuple(ta.padded_shape), tuple(tb.padded_shape),
                   str(ta.dtype).split(".")[-1])
            rec = CLASSES.get(key)
            if rec is None:
                CLASSES[key] = rec = {
                    "site": site,
                    "a_padded": list(ta.padded_shape), "b_padded": list(tb.padded_shape),
                    "dtype": str(ta.dtype).split(".")[-1],
                    "batch": max(_batch(list(ta.padded_shape)), _batch(list(tb.padded_shape))),
                    "mt": ta.padded_shape[-2] // TILE, "kt": ta.padded_shape[-1] // TILE,
                    "nt": tb.padded_shape[-1] // TILE,
                    "dram_interleaved": _dram_il(ta) and _dram_il(tb),
                    "hinted": bool(kw.get("core_grid") or kw.get("program_config")),
                    "n": 0, "probed": 0, "arms": None,
                }
            rec["n"] += 1
            if (rec["probed"] < STATE["probes"] and rec["dram_interleaved"]
                    and not rec["hinted"] and rec["batch"] > 1):
                rec["probed"] += 1
                ref = ttnn.to_torch(out)
                _probe(ta, tb, ref, kw, rec)
                print(f"[parity] {rec['site']} B={rec['batch']} "
                      f"Mt/Kt/Nt={rec['mt']}/{rec['kt']}/{rec['nt']} {rec['dtype']} "
                      f"naive={rec['naive_predicted']['factory']}"
                      f"(bw={rec['naive_predicted']['in0_block_w']}) "
                      f"partitions={rec['partitions_by_in0_block_w']} "
                      f"exact_bw={rec['exact_in0_block_w']} "
                      f"inexact_bw={rec['inexact_in0_block_w']}", flush=True)
                for x in rec["arms"]:
                    print("    " + (f"{x['arm']:28s} exact={x['exact']}" if "exact" in x
                                    else f"{x['arm']:28s} REJECTED {x['error'][:80]}"), flush=True)
        finally:
            STATE["depth"] = 0
        return out
    return inner


def dump(out_path: str, unit: str):
    rows = sorted(CLASSES.values(), key=lambda r: -r["n"])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps({"unit": unit, "cores": STATE["cores"], "rows": rows}, indent=1))
    print(f"\n[parity] {len(rows)} target classes, {sum(r['n'] for r in rows)} calls per {unit}",
          flush=True)
    for r in rows:
        v = ("not probed" if r["arms"] is None else
             f"partitions={r['partitions_by_in0_block_w']} exact_bw={r['exact_in0_block_w']}")
        print(f"  {r['site']:<26} n={r['n']:<6} B={r['batch']:<4} "
              f"Mt/Kt/Nt={r['mt']}/{r['kt']}/{r['nt']:<4} {r['dtype']:<9} {v}", flush=True)
    print(f"[parity] wrote {out_path}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--probes", type=int, default=2,
                    help="calls per class to run the full arm set on; the rest are only counted")
    ap.add_argument("--unit", default="run", help="what the call counts are PER (fold, recycle, ...)")
    ap.add_argument("--fold", help="model name for the sprint's in-process fold harness "
                                   "(scripts/gpu_vs_tt/tt_baseline.build_fold). REQUIRED for any "
                                   "model tt-bio predict drives through its scheduler: that path "
                                   "spawns a worker process, the patch below lives in the parent "
                                   "only, and the census comes back empty.")
    ap.add_argument("argv", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    STATE["probes"] = args.probes

    from tt_bio.tenstorrent import get_device
    dev = get_device()
    g = dev.compute_with_storage_grid_size()
    STATE["cores"] = int(g.x) * int(g.y)
    print(f"[parity] grid {g.x}x{g.y} = {STATE['cores']} cores", flush=True)

    ttnn.matmul = _wrap(ttnn.matmul)
    atexit.register(lambda: dump(args.out, args.unit))

    if args.fold:
        sys.path.insert(0, str(REPO / "scripts" / "gpu_vs_tt"))
        import tt_baseline as TB
        one_fold, meta, _state = TB.build_fold(
            args.fold, REPO / "perf" / "attn_sites" / "_msa",
            REPO / "examples" / "prot300.yaml", Path(TB.FIXTURES) / "prot300.a3m")
        print(f"[parity] in-process fold ({meta})", flush=True)
        STATE["on"] = True
        one_fold()
        STATE["on"] = False
        return 0

    import tt_bio.main as tb_main
    STATE["on"] = True
    sys.argv = ["tt-bio"] + [a for a in args.argv if a != "--"]
    # standalone_mode=False so click returns instead of calling sys.exit.
    tb_main.cli(standalone_mode=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
