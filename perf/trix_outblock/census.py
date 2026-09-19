#!/usr/bin/env python3
"""Census every matmul/linear in a live fold that lets ttnn DERIVE its program config.

The `out_block_h` lever only exists where nobody passed a `program_config`: `ttnn.linear(
core_grid=...)` then derives `out_block_h = per_core_M`, i.e. one block, which is the worst
legal drain schedule. tt-bio already fixes this on the 1D-mcast pair-track path
(`_pair_proj_program_config`) with a hardcoded `out_block_h = 5`. Everywhere else it does not.

This counts the derived sites, in a real fold, by operand shape and callsite, so the ladder is
run at shapes the model actually executes rather than at a shape somebody liked. No timing is
taken here: a sync around every call would change the thing being counted, and the marginal
per-call cost comes from the n-ladder instrument instead (`outblock_ladder.py`). Counts x
measured marginal is the priced census, with the campaign's in-fold factor stated beside it.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))
sys.path.insert(0, str(ROOT / "perf" / "c12_orchestrator" / "relayed" / "c12_kblock"))

ROWS: dict[tuple, dict] = {}
ON = {"v": False}


def _sig(t):
    try:
        return "x".join(str(int(d)) for d in t.shape) + "|" + str(t.dtype).split(".")[-1] + "|" + (
            "L1" if "L1" in str(t.memory_config().buffer_type) else "DRAM")
    except Exception:  # noqa: BLE001
        return "?"


def install(ttnn):
    """Wrap ttnn.linear and ttnn.matmul; record shape, grid and whether a config was passed."""
    for name in ("linear", "matmul"):
        orig = getattr(ttnn, name)

        def make(orig=orig, name=name):
            def w(a, b, *args, **kw):
                if ON["v"]:
                    pc = kw.get("program_config")
                    cg = kw.get("core_grid")
                    f = sys._getframe(1)
                    site = f"{Path(f.f_code.co_filename).stem}.{f.f_code.co_name}:{f.f_lineno}"
                    f2 = f.f_back
                    if f2 is not None:
                        site += f" <- {Path(f2.f_code.co_filename).stem}.{f2.f_code.co_name}:{f2.f_lineno}"
                    key = (name, _sig(a), _sig(b),
                           "derived" if pc is None else type(pc).__name__.replace(
                               "MatmulMultiCoreReuseMultiCast", "mc"),
                           str(cg) if cg is not None else "-",
                           str(kw.get("memory_config", "")).split(",")[0][-6:])
                    r = ROWS.get(key)
                    if r is None:
                        r = ROWS[key] = {"n": 0, "sites": defaultdict(int)}
                    r["n"] += 1
                    r["sites"][site] += 1
                return orig(a, b, *args, **kw)
            return w
        setattr(ttnn, name, make())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="protenix-v2")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--card", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import ttnn
    import tt_bio
    import tt_bio.tenstorrent as TT
    import tt_baseline as B
    assert Path(tt_bio.__file__).resolve().is_relative_to(ROOT), (
        f"imported tt_bio from {tt_bio.__file__}, not this worktree")
    from tt_bio.main import (_detect_p300_devices, _find_ttnn_mesh_graph_descriptor,
                             _resolve_recycling_steps, _resolve_sampling_steps)
    if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
        if mgd:
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd
    B.RECYCLING_STEPS = _resolve_recycling_steps(None, a.model)
    B.SAMPLING_STEPS = _resolve_sampling_steps(None, a.model)

    fix = ROOT / "perf" / "size512" / "fixtures"
    one_fold, meta, state = B.build_fold(a.model, ROOT / f".msa_trixob_{a.model}_{a.size}",
                                         fix / f"cdk2x2_{a.size}.yaml",
                                         fix / f"cdk2x2_{a.size}.a3m")
    dev = TT.get_device()
    g = dev.compute_with_storage_grid_size()
    install(ttnn)

    import clk
    nodes = clk.nodes_open_by_this_process()
    held = clk.force(1350, nodes)
    clocks = []

    ON["v"] = True
    t0 = time.perf_counter()
    fold_s, m = one_fold()
    ON["v"] = False
    wall = time.perf_counter() - t0
    for n in held:
        clocks.append((n, clk.aiclk(n)))

    rows = sorted(({"op": k[0], "in0": k[1], "in1": k[2], "cfg": k[3], "grid": k[4],
                    "out_mc": k[5], "n": v["n"],
                    "sites": dict(sorted(v["sites"].items(), key=lambda kv: -kv[1]))}
                   for k, v in ROWS.items()), key=lambda r: -r["n"])
    res = {"host": socket.gethostname(), "model": a.model, "size": a.size, "card": a.card,
           "grid": [g.x, g.y], "fold_s": round(fold_s, 3), "wall_s": round(wall, 3),
           "plddt": m.get("plddt"), "n_tokens": m.get("n_tokens"),
           "aiclk_after": clocks, "nodes_forced": held,
           "loadavg1": round(os.getloadavg()[0], 2),
           "git_head": os.popen(f"git -C {ROOT} rev-parse HEAD").read().strip(),
           "rows": rows}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    nd = sum(r["n"] for r in rows if r["cfg"] == "derived")
    print(f"fold {fold_s:.2f}s wall {wall:.2f}s  {len(rows)} keys, "
          f"{sum(r['n'] for r in rows)} calls, {nd} on a DERIVED config", flush=True)
    for r in rows[:25]:
        print(f"  {r['n']:6d} {r['cfg']:10s} {r['op']:7s} {r['in0']:34s} {r['in1']:26s} "
              f"{r['grid']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
