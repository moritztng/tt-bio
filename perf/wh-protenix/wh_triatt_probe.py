#!/usr/bin/env python3
"""Why protenix-v2 stops folding above 608 aa on the Wormhole Galaxy.

The size sweep (`pxfix_640/768/1024.log`) fails at 640, 768 and 1024 in the same frame:
`_template` -> PairformerLayer -> `triangle_attention_start` -> the chunked path's qkv
projection, either as `_triatt_qkv.qkv_heads` growing its static CBs past max L1, or as the
`ttnn.experimental.minimal_matmul` fallback clashing with a live L1 buffer.

This runs ONE cold fold with the qkv projection instrumented and reports, per call:
the activation and weight shapes, the block config `_qkv_mm_config` chose, whether the
fused path refused, and how much L1 the allocator says is free at that moment. It is a
diagnostic, not a measurement: it is not timed and does not need the bench lock.
"""
import argparse, json, os, sys, traceback
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", type=Path, required=True)
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-mm-cfg", action="store_true",
                    help="force `_qkv_mm_config` to return None (the unconfigured op) and see "
                         "whether the fold gets past the clash")
    ap.add_argument("--tail", type=int, default=14, help="qkv calls to keep in the trace")
    ap.add_argument("--fast", action="store_true",
                    help="fold on the shipped --fast block-fp8 path, which is what JapanFold "
                         "production runs. Halves the resident bytes, so it is a different "
                         "question from the default arm and has to be measured separately.")
    a = ap.parse_args()

    tree = a.tree.resolve()
    sys.path.insert(0, str(tree))
    sys.path.insert(0, str(tree / "scripts" / "gpu_vs_tt"))

    import ttnn
    import tt_bio.tenstorrent as T
    import tt_baseline as B
    from tt_bio.main import _resolve_recycling_steps, _resolve_sampling_steps
    assert Path(T.__file__).resolve().is_relative_to(tree), f"tt_bio came from {T.__file__}"

    B.RECYCLING_STEPS = _resolve_recycling_steps(None, "protenix-v2")
    B.SAMPLING_STEPS = _resolve_sampling_steps(None, "protenix-v2")

    calls = []

    def shape_of(t):
        try:
            return [int(d) for d in t.shape]
        except Exception:
            return None

    def l1_free():
        try:
            return int(ttnn.get_max_worker_l1_unreserved_size())
        except Exception:
            return None

    _cfg = T._qkv_mm_config

    def traced_cfg(inp, w):
        c = None if a.no_mm_cfg else _cfg(inp, w)
        calls.append({"kind": "cfg", "inp": shape_of(inp), "w": shape_of(w),
                      "blk": None if c is None else [int(c.M_block_size), int(c.K_block_size),
                                                     int(c.N_block_size), int(c.subblock_h),
                                                     int(c.subblock_w)],
                      "grid": list(T.COMPUTE_GRID_MAIN)})
        return c
    T._qkv_mm_config = traced_cfg

    _mm = ttnn.experimental.minimal_matmul

    def traced_mm(*args, **kw):
        rec = {"kind": "minimal_matmul",
               "inp": shape_of(kw.get("input_tensor")), "w": shape_of(kw.get("weight_tensor")),
               "cfg": kw.get("config") is not None, "l1_free": l1_free()}
        calls.append(rec)
        try:
            o = _mm(*args, **kw)
        except Exception as e:
            rec["threw"] = str(e).splitlines()[-1][:300]
            raise
        rec["ok"] = True
        return o
    ttnn.experimental.minimal_matmul = traced_mm

    import tt_bio.triatt_qkv as Q
    _qh = Q.qkv_heads

    def traced_qh(*args, **kw):
        rec = {"kind": "qkv_heads", "inp": shape_of(args[0]) if args else None,
               "l1_free": l1_free()}
        calls.append(rec)
        try:
            o = _qh(*args, **kw)
        except Exception as e:                       # it swallows its own; belt and braces
            rec["threw"] = str(e).splitlines()[-1][:300]
            raise
        rec["refused"] = o is None
        return o
    Q.qkv_heads = traced_qh
    T._triatt_qkv.qkv_heads = traced_qh

    fixdir = tree / "perf" / "size512" / "fixtures"
    tgt, a3m = fixdir / f"cdk2x2_{a.size}.yaml", fixdir / f"cdk2x2_{a.size}.a3m"
    msa_dir = tree / f".msa_xmodel_protenix-v2_{a.size}"
    one_fold, meta = B.build_fold("protenix-v2", msa_dir, tgt, a3m, fast=a.fast)[:2]

    res = {"size": a.size, "tree": str(tree), "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "no_mm_cfg": a.no_mm_cfg, "grid": list(T.COMPUTE_GRID_MAIN),
           "fast": a.fast, "_FAST_MODE": T._FAST_MODE, "dtype": str(T._dtype()),
           "l1_unreserved": l1_free(),
           "SEQ_LEN_MORE_CHUNKING": T.SEQ_LEN_MORE_CHUNKING,
           "TRIANGLE_ATT_CHUNK_SIZE": T.TRIANGLE_ATT_CHUNK_SIZE,
           "TRIANGLE_ATT_CHUNK_SIZE_FAST": T.TRIANGLE_ATT_CHUNK_SIZE_FAST}
    try:
        fold_s, m = one_fold()
        res["ok"] = True
        res["fold_s"] = round(fold_s, 3)
        res["plddt"] = m.get("plddt")
    except Exception as e:
        res["ok"] = False
        res["error"] = str(e).splitlines()[-1][:400]
        res["frames"] = [l.strip() for l in traceback.format_exc().splitlines()
                         if l.strip().startswith("File ")][-8:]
    res["n_qkv_events"] = len(calls)
    res["trace_tail"] = calls[-a.tail:]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1), flush=True)
    T.cleanup()


if __name__ == "__main__":
    main()
