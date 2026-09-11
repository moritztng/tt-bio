#!/usr/bin/env python3
"""What a 512 aa Boltz-2 fold actually moves across DRAM on this card, per phase.

The roofline write-up could bound our realised traffic (below half the unfused count, because
the unfused count would need 859 GB/s on a 438.5 GB/s card) but not measure it. This does.

Method: run one real fold, and for the two phases that carry 92 % of the fold's FLOPs --
`PairformerLayer` and the diffusion `DiffusionTransformerLayer` -- time every invocation with a
device sync on both sides, and `ttnn.graph`-capture one settled invocation of each distinct input
shape. From that capture, DRAM traffic for the call is every DRAM-resident input tensor read once
plus every DRAM buffer allocated inside the call written once. That is a floor on the real
traffic (a multi-pass op re-reads its input), so an achieved GB/s computed from it is also a
floor, which is the safe direction: it cannot manufacture a "we are at the roof" conclusion.

The fold runs with a short diffusion rollout because the per-call numbers are per call; the step
count only changes how many times the same shapes recur, and it is applied as a multiplier at the
end from the production 200.
"""
import argparse
import json
import statistics as st
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

import ttnn                                                                   # noqa: E402
import tt_bio.tenstorrent as T                                                # noqa: E402

WALL = defaultdict(list)          # sig -> [seconds]
CAP = {}                          # sig -> byte breakdown
BUSY = {"on": False}
DEV = {"d": None}


def sig(prefix, args):
    parts = []
    for x in args:
        sh = getattr(x, "shape", None)
        if sh is not None:
            parts.append("x".join(str(int(d)) for d in sh))
    return prefix + "|" + ",".join(parts[:2])


def graph_bytes(g):
    """DRAM bytes a captured region reads and writes, plus the same for L1."""
    out = {"dram_read": 0, "dram_write": 0, "l1_write": 0, "n_ops": 0, "n_tensors": 0}
    seen_tensor = set()
    for n in g:
        t = n.get("node_type")
        p = n.get("params") or {}
        if t == "function_start" and str(p.get("name", "")).startswith("ttnn."):
            out["n_ops"] += 1
        elif t == "tensor":
            tid = p.get("tensor_id")
            if tid in seen_tensor:
                continue
            seen_tensor.add(tid)
            out["n_tensors"] += 1
            if "DRAM" in str(p.get("buffer_type", "")):
                out["dram_read"] += int(p.get("size", 0) or 0)
        elif t == "buffer_allocate":
            if str(p.get("type")) == "DRAM":
                out["dram_write"] += int(p.get("size", 0) or 0)
            else:
                out["l1_write"] += int(p.get("size", 0) or 0)
    out["dram_total"] = out["dram_read"] + out["dram_write"]
    return out


def wrap(cls, prefix, capture_on_call):
    orig = cls.__call__

    def call(self, *a, **k):
        s = sig(prefix, a)
        n = len(WALL[s])
        want = (n == capture_on_call) and s not in CAP and not BUSY["on"]
        if want:
            BUSY["on"] = True
            ttnn.synchronize_device(DEV["d"])
            ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        ttnn.synchronize_device(DEV["d"])
        t0 = time.perf_counter()
        r = orig(self, *a, **k)
        ttnn.synchronize_device(DEV["d"])
        dt = time.perf_counter() - t0
        if want:
            CAP[s] = graph_bytes(ttnn.graph.end_graph_capture())
            BUSY["on"] = False
        WALL[s].append(dt)
        return r

    cls.__call__ = call


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=8, help="diffusion steps for the capture fold")
    ap.add_argument("--prod-steps", type=int, default=200)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import tt_baseline as B
    B.SAMPLING_STEPS = a.steps
    sys.path.insert(0, str(ROOT / 'perf' / 'other512'))
    import fold_ab_multi as FAM          # build_fold's cfg carries no Boltz-2 hyperparameters
    from tt_bio.main import _resolve_recycling_steps
    # fold_ab_multi predates the move of RECYCLING_STEPS into tt_bio.main and still reads it off
    # tt_baseline, so supply it from the table the CLI actually reads.
    B.RECYCLING_STEPS = _resolve_recycling_steps(None, 'boltz2')
    FAM.patch_boltz2_cfg()               # reads B.SAMPLING_STEPS, so set it first

    fix = ROOT / "perf" / "size512" / "fixtures"
    tgt, a3m = fix / f"cdk2x2_{a.size}.yaml", fix / f"cdk2x2_{a.size}.a3m"
    one_fold, meta, state = B.build_fold(
        "boltz2", ROOT / f".msa_bytes_{a.size}", tgt, a3m)
    DEV["d"] = T.get_device()

    wrap(T.PairformerLayer, "pairformer", 3)
    wrap(T.DiffusionTransformerLayer, "difftx", 3)

    fold_s, m = one_fold()
    print("FOLD %.2f s n_tokens=%s plddt=%s steps=%d"
          % (fold_s, m.get("n_tokens"), m.get("plddt"), a.steps), flush=True)

    rows = []
    for s in sorted(WALL):
        ts = WALL[s]
        row = {"sig": s, "calls_in_capture_fold": len(ts),
               "median_ms": round(1e3 * st.median(ts), 4),
               "total_ms": round(1e3 * sum(ts), 2)}
        row.update(CAP.get(s, {}))
        rows.append(row)
        print(json.dumps(row), flush=True)

    Path(a.out).write_text(json.dumps(
        {"size": a.size, "capture_steps": a.steps, "prod_steps": a.prod_steps,
         "fold_s": round(fold_s, 3), "n_tokens": m.get("n_tokens"), "plddt": m.get("plddt"),
         "rows": rows}, indent=1))
    print("WROTE " + a.out)
    T.cleanup()


if __name__ == "__main__":
    main()
