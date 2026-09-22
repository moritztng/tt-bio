#!/usr/bin/env python3
"""Count which shipped lever fires inside one real fold, per model, and which `_MM_BLOCK` key
each declining matmul wanted.

`pvx-inventory` proved on Protenix that a lever which never fires looks exactly like a lever that
does not work, and `pvx-eligibility` proved the blocker was a lookup table populated at one model
width. Both instruments existed but only ever ran on boltz2 and protenix-v2. This one is the union
of the two, so one device pass per model answers both questions:

  * every `*_STATS` / `*_REJECTS` / `*_SITES` counter, delta around a warm fold;
  * every `(kt, nt)` key presented to `_qkv_mm_config` and to `_mm_block_for`, with the calling
    site, so a table miss is a counted fact with an address rather than a guess.

Two warm folds by default: the counters must read identically on both (a count does not move with
host load) and the pair gives this row a same-arm A/A pair for free.

  python3 perf/allm_gates/gate_census.py --model openfold3 --out perf/allm_gates/of3_512.json
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))
sys.path.insert(0, str(ROOT / "perf"))
sys.path.insert(0, str(ROOT / "perf" / "pvx_eligibility"))

import re  # noqa: E402

import firing_census as FC  # noqa: E402  the delta arithmetic, reused unchanged

# `firing_census.py` matches counter names with `(STATS|REJECTS|SITES|COUNTS|_ROWS)$`, anchored at
# the end, over a hardcoded module list. Both are coverage holes and both hide a lever as "never
# fired": `reblock_permute.STATS_BACK` and `STATS_GATED` -- E6 itself, 2416 calls a fold on
# protenix-v2 -- end in BACK and GATED and match neither, and a module missing from the list
# contributes nothing silently. So match anywhere in the name, and take the module list from what
# the fold actually imported, which is also what tells "never reached" apart from "never loaded".
COUNTER = re.compile(r"STATS|REJECTS|SITES|COUNTS|DECLINES|_ROWS")


def modules():
    return sorted(n for n in list(sys.modules)
                  if n.startswith("tt_bio.") and sys.modules[n] is not None)


def snapshot() -> dict:
    import copy
    out = {}
    for name in modules():
        mod = sys.modules[name]
        short = name.split(".", 1)[1]
        for attr, val in vars(mod).items():
            if COUNTER.search(attr) and isinstance(val, (list, dict)):
                out[f"{short}.{attr}"] = FC._plain(copy.deepcopy(val))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--folds", type=int, default=2)
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

    # ---- the two table readers, censused by call site ---------------------------------------
    MM = collections.Counter()
    ORIG_QKV, ORIG_BLK = T._qkv_mm_config, T._mm_block_for

    def qkv(x, w, *args, **kw):
        cfg = ORIG_QKV(x, w, *args, **kw)
        kt = (int(w.shape[-2]) + 31) // 32
        nt = (int(w.shape[-1]) + 31) // 32
        mt = 1
        for d in [int(d) for d in x.shape][:-1]:
            mt *= d
        mt = (mt + 31) // 32
        if ORIG_BLK(w) is None:
            why = "key_absent"
        elif cfg is None:
            why = "guard_refused"
        else:
            why = "served"
        MM[f"qkv_mm_config kt={kt} nt={nt} mt={mt} {why}"] += 1
        return cfg

    def blk(w):
        b = ORIG_BLK(w)
        f = sys._getframe(1)
        where = f.f_code.co_filename.rsplit("/", 1)[-1] + ":" + str(f.f_lineno)
        kt = (int(w.shape[-2]) + 31) // 32
        nt = (int(w.shape[-1]) + 31) // 32
        hit = "hit" if b else "MISS"
        MM[f"mm_block_for kt={kt} nt={nt} {where} {hit}"] += 1
        return b

    T._qkv_mm_config, T._mm_block_for = qkv, blk

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

    one_fold, meta, state = B.build_fold(a.model, ROOT / f".msa_allmgates_{a.model}_{a.size}",
                                         tgt, a3m)
    res["flags"] = FC.flags()
    res["grid"] = list(T.COMPUTE_GRID_MAIN)
    res["counter_modules"] = modules()
    res["load_s"] = meta.get("load_s")
    a.out.write_text(json.dumps(res, indent=1))

    print(f"=== {a.model} {a.size}: cold fold ===", flush=True)
    cold_s, cold_m = one_fold()
    res["cold_s"] = round(cold_s, 3)
    res["n_tokens"] = cold_m.get("n_tokens")
    print(f"  cold {cold_s:.2f}s n_tokens={res['n_tokens']}", flush=True)

    for i in range(a.folds):
        MM.clear()
        before = snapshot()
        with clocksample.during(period=1.0) as clk:
            fold_s, m = one_fold()
        after = snapshot()
        rec = {"i": i, "fold_s": round(fold_s, 3), "plddt": m.get("plddt"),
               "n_tokens": m.get("n_tokens"), "aiclk": clk.summary(),
               "clock_line": clk.line(0), "counters": FC.delta(before, after),
               "mm_keys": dict(sorted(MM.items()))}
        res["folds"].append(rec)
        a.out.write_text(json.dumps(res, indent=1))
        print(f"  fold {i}: {fold_s:.2f}s  {clk.line(0)}", flush=True)
        for k, v in sorted(rec["counters"].items()):
            print(f"    {k} = {v}", flush=True)
        for k, v in sorted(rec["mm_keys"].items()):
            print(f"    MM {v:6d}  {k}", flush=True)
    a.out.write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
