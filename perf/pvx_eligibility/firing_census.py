#!/usr/bin/env python3
"""Count, at run time, which shipped lever fires inside one real 512 aa fold, per model.

The campaign hypothesis is that Protenix is slow because levers present in the shared code never
fire on its call mix, not because they are absent. Reading a gate cannot settle that: E6 was
recorded as "losing on boltz2" when it was ineligible on 100 % of boltz2 calls. So this counts.

Every lever in this tree already keeps its own counter (`*_STATS`, `*_REJECTS`, `*_SITES`), because
each was landed with one. Nothing here adds arithmetic to the fold: it snapshots those counters
around a fold and reports the delta, plus the reject reason where the counter carries one.

  python3 perf/pvx_eligibility/firing_census.py --model protenix-v2 --out out/ptx.json

One process, one device open, one fold per model. AICLK is sampled DURING via perf/clocksample.py.
"""
from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
import re
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))
sys.path.insert(0, str(ROOT / "perf"))

# Every module that ships a lever with a counter. A module absent here contributes nothing, so the
# list is the census scope and is stated rather than discovered -- a lever whose module is missing
# reads as "never fired", which is the one error this instrument must not make silently.
MODULES = [
    "tt_bio.tenstorrent", "tt_bio.reblock_permute", "tt_bio.triatt_qkv", "tt_bio.triatt_sdpa",
    "tt_bio.trimul_tail", "tt_bio.protenix", "tt_bio.boltz2", "tt_bio.opendde",
    "tt_bio.esmfold2", "tt_bio.openfold3_msa_embedder", "tt_bio.openfold3_template",
]
COUNTER = re.compile(r"(STATS|REJECTS|SITES|COUNTS|_ROWS)$")


def _plain(v):
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    return v


def snapshot() -> dict:
    out = {}
    for name in MODULES:
        mod = sys.modules.get(name)
        if mod is None:
            try:
                mod = importlib.import_module(name)
            except Exception:
                continue
        short = name.split(".")[-1]
        for attr, val in vars(mod).items():
            if COUNTER.search(attr) and isinstance(val, (list, dict)):
                out[f"{short}.{attr}"] = _plain(copy.deepcopy(val))
    return out


def _sub(a, b):
    """b - a, elementwise, for the int containers the counters use."""
    if isinstance(a, dict) or isinstance(b, dict):
        a, b = a or {}, b or {}
        d = {}
        for k in set(a) | set(b):
            x, y = a.get(k, 0), b.get(k, 0)
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                if y - x:
                    d[k] = y - x
            elif y != x:
                d[k] = y
        return d
    if isinstance(a, list) or isinstance(b, list):
        a, b = a or [], b or []
        n = max(len(a), len(b))
        a = list(a) + [0] * (n - len(a))
        b = list(b) + [0] * (n - len(b))
        return [(y - x) if isinstance(x, (int, float)) and isinstance(y, (int, float)) else y
                for x, y in zip(a, b)]
    return b


def delta(before: dict, after: dict) -> dict:
    keys = set(before) | set(after)
    out = {}
    for k in sorted(keys):
        d = _sub(before.get(k), after.get(k))
        if d in ({}, [], None):
            continue
        if isinstance(d, list) and not any(d):
            continue
        out[k] = d
    return out


def flags() -> dict:
    """Every module-level bool/int lever, so the census records the defaults it was taken under."""
    out = {}
    for name in MODULES:
        mod = sys.modules.get(name)
        if mod is None:
            continue
        short = name.split(".")[-1]
        for attr, val in vars(mod).items():
            if attr.startswith("__") or not attr.lstrip("_").isupper():
                continue
            if isinstance(val, bool) or (isinstance(val, int) and not isinstance(val, bool)):
                out[f"{short}.{attr}"] = val
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="protenix-v2")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--folds", type=int, default=1, help="timed folds after the cold one")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fixdir", type=Path, default=ROOT / "perf" / "size512" / "fixtures")
    a = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)
    import tt_bio as _TB
    assert Path(_TB.__file__).resolve().is_relative_to(ROOT), (
        f"imported tt_bio from {_TB.__file__}, not this worktree")
    import tt_bio.tenstorrent as T
    import tt_baseline as B
    import clocksample
    from tt_bio.main import (_detect_p300_devices, _find_ttnn_mesh_graph_descriptor,
                             _resolve_recycling_steps, _resolve_sampling_steps)

    if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
        if mgd:
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd
    B.RECYCLING_STEPS = _resolve_recycling_steps(None, a.model)
    B.SAMPLING_STEPS = _resolve_sampling_steps(None, a.model)
    if a.model == "boltz2":
        sys.path.insert(0, str(ROOT / "perf" / "other512"))
        import fold_ab_multi as _FAM
        _FAM.patch_boltz2_cfg()

    tgt = a.fixdir / f"cdk2x2_{a.size}.yaml"
    a3m = a.fixdir / f"cdk2x2_{a.size}.a3m"
    res = {"model": a.model, "size": a.size, "host": socket.gethostname(),
           "chip": os.environ.get("TT_VISIBLE_DEVICES"),
           "recycling_steps": B.RECYCLING_STEPS, "sampling_steps": B.SAMPLING_STEPS,
           "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "loadavg": os.getloadavg(), "folds": []}
    try:
        import importlib.metadata as _md
        res["ttnn"] = _md.version("ttnn")
    except Exception:
        pass
    a.out.parent.mkdir(parents=True, exist_ok=True)

    one_fold, meta, state = B.build_fold(a.model, ROOT / f".msa_pvxel_{a.model}_{a.size}", tgt, a3m)
    res["flags"] = flags()
    res["grid"] = list(T.COMPUTE_GRID_MAIN)
    a.out.write_text(json.dumps(res, indent=1))

    print(f"=== {a.model} {a.size}: cold fold ===", flush=True)
    cold_s, cold_m = one_fold()
    res["cold_s"] = round(cold_s, 3)
    res["n_tokens"] = cold_m.get("n_tokens")
    print(f"  cold {cold_s:.2f}s n_tokens={res['n_tokens']}", flush=True)

    for i in range(a.folds):
        before = snapshot()
        with clocksample.during(period=1.0) as clk:
            fold_s, m = one_fold()
        after = snapshot()
        rec = {"i": i, "fold_s": round(fold_s, 3), "plddt": m.get("plddt"),
               "n_tokens": m.get("n_tokens"), "aiclk": clk.summary(),
               "clock_line": clk.line(0), "counters": delta(before, after)}
        res["folds"].append(rec)
        a.out.write_text(json.dumps(res, indent=1))
        print(f"  fold {i}: {fold_s:.2f}s  {clk.line(0)}", flush=True)
        for k, v in sorted(rec["counters"].items()):
            print(f"    {k} = {v}", flush=True)
    a.out.write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
