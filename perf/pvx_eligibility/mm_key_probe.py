#!/usr/bin/env python3
"""Which `_MM_BLOCK` key does each declining tri-attention call want?

`no_mm_config` and `dtype_or_memory_or_config` both name a reject without naming the key that
would have served it, so the census can say a call declines but not what to add. This wraps
`_qkv_mm_config` for one fold and records the (kt, nt) key, the activation shape and whether the
key was absent or a divisibility guard refused it. Diagnostic only: it returns exactly what the
production function returns.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))
sys.path.insert(0, str(ROOT / "perf"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="protenix-v2")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fixdir", type=Path, default=ROOT / "perf" / "size512" / "fixtures")
    a = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)
    import tt_bio.tenstorrent as T
    import tt_baseline as B
    from tt_bio.main import (_detect_p300_devices, _find_ttnn_mesh_graph_descriptor,
                             _resolve_recycling_steps, _resolve_sampling_steps)
    if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
        if mgd:
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd
    B.RECYCLING_STEPS = _resolve_recycling_steps(None, a.model)
    B.SAMPLING_STEPS = _resolve_sampling_steps(None, a.model)

    seen: dict = {}
    ORIG = T._qkv_mm_config

    def probe(inp, w):
        out = ORIG(inp, w)
        kt = (int(w.shape[-2]) + 31) // 32
        nt = (int(w.shape[-1]) + 31) // 32
        shape = [int(d) for d in inp.shape]
        mt = 1
        for d in shape[:-1]:
            mt *= d
        mt = (mt + 31) // 32
        blk = T._mm_block_for(w)
        if blk is None:
            why = "key_absent"
        elif out is None:
            M, K, N = blk[0], blk[1], blk[2]
            why = f"guard kt%K={kt % K} mt%M={mt % M} nt%N={nt % N}"
        else:
            why = "served"
        k = f"kt={kt} nt={nt} x={'x'.join(map(str, shape))} mt={mt} -> {why}"
        seen[k] = seen.get(k, 0) + 1
        return out

    T._qkv_mm_config = probe
    tgt = a.fixdir / f"cdk2x2_{a.size}.yaml"
    a3m = a.fixdir / f"cdk2x2_{a.size}.a3m"
    one_fold, meta, state = B.build_fold(a.model, ROOT / f".msa_pvxel_{a.model}_{a.size}", tgt, a3m)
    one_fold()
    seen.clear()
    fold_s, _m = one_fold()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"model": a.model, "size": a.size, "fold_s": round(fold_s, 3),
                                 "keys": seen}, indent=1))
    for k, v in sorted(seen.items()):
        print(f"  {v:6d}  {k}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
