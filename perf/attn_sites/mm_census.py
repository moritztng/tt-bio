#!/usr/bin/env python3
"""Census every ``ttnn.matmul`` call a real run makes, and decide per call whether ttnn's
automatic program-config selection can distribute the batch dimension across cores.

Why this exists
---------------
For a DRAM-interleaved matmul with a batched second operand, ttnn only reaches the one factory
that splits B across cores (``MatmulMultiCoreReuseProgramConfig``) when the caller passes
``core_grid=`` or ``program_config=``. With neither, ``generate_matmul_program_config`` takes the
``!input_tensor_a.is_sharded()`` branch straight to ``create_simple_matmul_program_config``, which
computes its block counts from a SINGLE batch element's Mt and Nt -- the batch never contributes a
core, and the whole batch loop runs inside each engaged core's kernel.

So the screen is not a shape heuristic. It is: batched-B, DRAM-interleaved, and no
``core_grid=``/``program_config=`` on the call. Everything else the row records (Mt/Kt/Nt, the
engaged-core count, arithmetic intensity) says how much that costs, not whether it fires.

Recorded per call: the innermost tt_bio call site (file:line), padded shapes / dtypes / buffer
types / memory layouts of both operands and the output, and which of the two config keywords was
passed. Rows are aggregated by (site, class) with a count, so one JSON is a per-fold call census.

Nothing is timed here: timing perturbs, and the ranking pass (mm_replay.py) does it separately on
the classes this pass finds.

    TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=... \
        python3 perf/attn_sites/mm_census.py --out c.json -- predict examples/prot300.yaml ...
"""
from __future__ import annotations

import argparse
import atexit
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import ttnn  # noqa: E402

TILE = 32
COUNTS: dict[tuple, dict] = {}
STATE = {"on": False, "depth": 0}


def _site() -> tuple[str, list[str]]:
    """Innermost tt_bio frame and the tt_bio chain above it.

    ``sys._getframe`` rather than ``traceback.extract_stack``: this runs on every matmul of a
    200-step diffusion loop and extract_stack costs ~80 us a call, which is minutes over a fold.
    """
    chain = []
    f = sys._getframe(2)
    while f is not None and len(chain) < 6:
        fn = f.f_code.co_filename
        if "/tt_bio/" in fn and "attn_sites" not in fn:
            chain.append(f"{fn.split('/tt_bio/')[-1]}:{f.f_lineno}")
        f = f.f_back
    return (chain[0] if chain else "?"), chain[:4]


def _desc(t):
    if not isinstance(t, ttnn.Tensor):
        return None
    mc = t.memory_config()
    return {
        "padded": list(t.padded_shape),
        "logical": list(t.shape),
        "dtype": str(t.dtype).split(".")[-1],
        "buf": str(mc.buffer_type).split(".")[-1],
        "layout": str(mc.memory_layout).split(".")[-1],
    }


def _batch(padded: list[int]) -> int:
    """ttnn's ``get_batch_size``: the product of every dim above the last two."""
    return math.prod(padded[:-2]) if len(padded) > 2 else 1


def _wrap(fn):
    def inner(*a, **kw):
        if not STATE["on"] or STATE["depth"]:
            return fn(*a, **kw)
        STATE["depth"] = 1
        try:
            site, chain = _site()
            ta = _desc(a[0]) if a else None
            tb = _desc(a[1]) if len(a) > 1 else _desc(kw.get("input_tensor_b"))
            out = fn(*a, **kw)
        finally:
            STATE["depth"] = 0
        if ta is None or tb is None:
            return out
        to = _desc(out if isinstance(out, ttnn.Tensor) else None)
        # transpose_a/transpose_b swap which padded dim is K; record them so the replay is faithful.
        key = (site, json.dumps(ta, sort_keys=True), json.dumps(tb, sort_keys=True),
               bool(kw.get("core_grid")), bool(kw.get("program_config")),
               bool(kw.get("transpose_a")), bool(kw.get("transpose_b")))
        rec = COUNTS.get(key)
        if rec is None:
            COUNTS[key] = rec = {
                "site": site, "chain": chain, "a": ta, "b": tb, "out": to,
                "core_grid": bool(kw.get("core_grid")),
                "program_config": bool(kw.get("program_config")),
                "transpose_a": bool(kw.get("transpose_a")),
                "transpose_b": bool(kw.get("transpose_b")),
                "n": 0,
            }
        rec["n"] += 1
        return out
    return inner


def _classify(rec: dict, grid_cores: int) -> dict:
    a, b = rec["a"], rec["b"]
    ka = -2 if rec["transpose_a"] else -1
    nb = -2 if rec["transpose_b"] else -1
    mt = a["padded"][-1 if rec["transpose_a"] else -2] // TILE
    kt = a["padded"][ka] // TILE
    nt = b["padded"][nb] // TILE
    batch = max(_batch(a["padded"]), _batch(b["padded"]))
    b_batched = _batch(b["padded"]) > 1
    dram_il = all(t["buf"] == "DRAM" and t["layout"] == "INTERLEAVED" for t in (a, b))
    hinted = rec["core_grid"] or rec["program_config"]
    # Engaged cores on the automatic path: block counts come from ONE batch element, and
    # per_core_M == per_core_N == an L1-fit factor >= 1. The optimistic bound (factor 1) is
    # min(Mt*Nt, grid); the batch contributes nothing either way.
    auto_cores = min(mt * nt, grid_cores)
    reuse_cores = min(batch * mt * nt, grid_cores)
    rec.update({
        "batch": batch, "mt": mt, "kt": kt, "nt": nt,
        "b_batched": b_batched, "dram_interleaved": dram_il, "hinted": hinted,
        "trips": bool(b_batched and dram_il and not hinted),
        "auto_cores_max": auto_cores, "reuse_cores": reuse_cores,
    })
    return rec


def install():
    ttnn.matmul = _wrap(ttnn.matmul)
    STATE["on"] = True


def dump(out_path: str, grid_cores: int):
    rows = [_classify(r, grid_cores) for r in COUNTS.values()]
    rows.sort(key=lambda r: (-r["trips"], -r["n"]))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(
        {"grid_cores": grid_cores, "rows": rows}, indent=1))
    trip = [r for r in rows if r["trips"]]
    print(f"\n[census] {len(rows)} classes, {sum(r['n'] for r in rows)} calls; "
          f"{len(trip)} classes trip the batched signature "
          f"({sum(r['n'] for r in trip)} calls)", flush=True)
    for r in trip[:25]:
        print(f"  {r['site']:<44} n={r['n']:<7} B={r['batch']:<5} "
              f"Mt/Kt/Nt={r['mt']}/{r['kt']}/{r['nt']:<4} "
              f"cores {r['auto_cores_max']}->{r['reuse_cores']} of {grid_cores}", flush=True)
    print(f"[census] wrote {out_path}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--fold", choices=["protenix-v2", "opendde"],
                    help="run the sprint's own 298 aa fold harness (scripts/gpu_vs_tt/tt_baseline.py) "
                         "so the census counts are per PRODUCTION fold and comparable with every "
                         "other leg's numbers")
    ap.add_argument("argv", nargs=argparse.REMAINDER,
                    help="tt_bio.main argv after `--`, for the models tt_baseline does not drive")
    args = ap.parse_args()

    from tt_bio.tenstorrent import get_device

    dev = get_device()
    g = dev.compute_with_storage_grid_size()
    grid_cores = int(g.x) * int(g.y)
    print(f"[census] compute_with_storage_grid_size = {g.x}x{g.y} = {grid_cores} cores", flush=True)

    if args.fold:
        sys.path.insert(0, str(REPO / "scripts" / "gpu_vs_tt"))
        import tt_baseline as TB
        msa_dir = REPO / "perf" / "attn_sites" / "_msa"
        one_fold, meta = TB.build_fold(
            args.fold, msa_dir, REPO / "examples" / "prot300.yaml",
            Path(TB.FIXTURES) / "prot300.a3m")
        print(f"[census] cold fold ({meta})", flush=True)
        one_fold()                       # cold: kernel compile, cache validation
        install()
        COUNTS.clear()                   # census the WARM fold only -> per-fold counts
        one_fold()
        STATE["on"] = False
        dump(args.out, grid_cores)
        return 0

    import tt_bio.main as tb_main
    install()
    atexit.register(lambda: dump(args.out, grid_cores))
    sys.argv = ["tt-bio"] + [a for a in args.argv if a != "--"]
    return tb_main.main() or 0


if __name__ == "__main__":
    raise SystemExit(main())
