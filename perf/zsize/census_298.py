#!/usr/bin/env python3
"""S4 + S9: one 298 aa fold, counting BOTH layer_norm populations and reading real per-bank L1.

The two censuses in this org disagree in scope, so this counts them separately in one fold rather
than reconciling by arithmetic:

  * `_l1_layer_norm` calls  -- the h=1.5 capacity-gated class, the one `cc39a867d` added.
  * `ttnn.layer_norm` calls -- every layer_norm the fold issues, whatever the caller.

Also reports `_L1_OUT_REFUSED`, `_BMM_CFG_REFUSED`, and per-bank free L1 as actual byte counts.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=298)
    ap.add_argument("--fixdir", type=Path, default=ROOT / "perf" / "size512" / "fixtures")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import ttnn
    import tt_bio.tenstorrent as T
    import tt_bio.protenix as P
    import tt_baseline as B
    import importlib.metadata as im

    DEC = defaultdict(Counter)
    LN_ALL = Counter()
    STATE = {}

    ORIG_OPEN = T._open_device_locked

    def open_dev(device_id, kwargs):
        d = ORIG_OPEN(device_id, kwargs)
        STATE["dev"] = d
        return d

    T._open_device_locked = open_dev

    # population A: the h=1.5 capacity-gated class
    ORIG_L1LN = T._l1_layer_norm

    def l1ln(x, headroom, **kw):
        out, in_l1 = ORIG_L1LN(x, headroom, **kw)
        shp = "x".join(str(int(d)) for d in x.shape)
        DEC["_l1_layer_norm|h=%s|%s" % (headroom, shp)]["L1" if in_l1 else "DRAM"] += 1
        return out, in_l1

    T._l1_layer_norm = l1ln
    P._l1_layer_norm = l1ln

    # population B: every ttnn.layer_norm the fold issues, whoever calls it
    ORIG_TTNN_LN = ttnn.layer_norm

    def ttnn_ln(*args, **kw):
        mc = kw.get("memory_config")
        buf = "default"
        if mc is not None:
            buf = "L1" if mc.buffer_type == ttnn.BufferType.L1 else "DRAM"
        LN_ALL[buf] += 1
        LN_ALL["total"] += 1
        return ORIG_TTNN_LN(*args, **kw)

    ttnn.layer_norm = ttnn_ln
    T.ttnn.layer_norm = ttnn_ln

    def banks(tag):
        d = STATE.get("dev")
        if d is None:
            return None
        try:
            mv = ttnn.get_memory_view(d, ttnn.BufferType.L1)
            return {
                "tag": tag,
                "num_banks": int(mv.num_banks()),
                "total_bytes_per_bank": int(mv.total_bytes_per_bank()),
                "total_bytes_allocated_per_bank": int(mv.total_bytes_allocated_per_bank()),
                "total_bytes_free_per_bank": int(mv.total_bytes_free_per_bank()),
                "largest_contiguous_bytes_free_per_bank":
                    int(mv.largest_contiguous_bytes_free_per_bank()),
            }
        except Exception as e:                                                   # noqa: BLE001
            return {"tag": tag, "error": "%s: %s" % (type(e).__name__, e)}

    tgt = a.fixdir / ("cdk2x2_%d.yaml" % a.size)
    a3m = a.fixdir / ("cdk2x2_%d.a3m" % a.size)
    one_fold, meta, _st = B.build_fold("protenix-v2", ROOT / (".msa_z%d" % a.size), tgt, a3m)
    struct_dir = Path(meta["struct_dir"])
    res = {"host": "qb1", "card": 0, "ttnn": im.version("ttnn"), "size": a.size,
           "grid": list(T.COMPUTE_GRID_MAIN), "l1_idle": banks("idle_after_device_open")}

    t0 = time.perf_counter()
    fold_s, m = one_fold()
    res["fold_s"] = round(fold_s, 3)
    res["wall_s"] = round(time.perf_counter() - t0, 1)
    res["plddt"] = m.get("plddt")
    res["n_tokens"] = m.get("n_tokens")
    res["cif_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
                         for p in sorted(struct_dir.glob("*")) if p.is_file()}
    res["l1_after_fold"] = banks("after_fold")
    res["l1_layer_norm_population"] = {k: dict(v) for k, v in sorted(DEC.items())}
    res["l1_layer_norm_total"] = sum(sum(v.values()) for v in DEC.values())
    res["l1_layer_norm_l1_branch"] = sum(v.get("L1", 0) for v in DEC.values())
    res["l1_layer_norm_dram_branch"] = sum(v.get("DRAM", 0) for v in DEC.values())
    res["ttnn_layer_norm_population"] = dict(LN_ALL)
    res["l1_out_refused"] = sorted(map(str, T._L1_OUT_REFUSED))
    res["bmm_cfg_refused"] = sorted(map(str, getattr(T, "_BMM_CFG_REFUSED", ())))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items()
                      if k != "l1_layer_norm_population"}, indent=1), flush=True)
    print("--- _l1_layer_norm population ---", flush=True)
    for k, v in sorted(DEC.items()):
        print("  %-44s %s" % (k, dict(v)), flush=True)
    print("wrote", a.out, flush=True)


main()
