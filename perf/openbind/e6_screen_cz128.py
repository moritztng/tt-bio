#!/usr/bin/env python3
"""E6, the fused chunk+gate TriMul forward move, screened at c_z = 128 -- the OF3 trunk's width.

E6 is default-ON and reads 0 served / 0 declined on every OF3-family census, because no OF3 call
site passes `gated_move=True`; esmfold2, opendde and protenix all do. It serves 2416/2416 calls on
protenix-v2 at c_z 256 and 1048/1216 on opendde at c_z 384, and loses on boltz-2's call mix, so
c_z is the number that decides it and this trunk has never been measured.

The screen is the kernel, not a call site: arm A is the exact sequence the trimul runs today
(`ttnn.chunk(4)` + two `multiply_(p, g, SIGMOID)` + two forward moves), arm B is two
`reblock_permute_gated` calls. Both arms produce a_chunk and b_chunk in the starting-node
orientation, and the ending-node orientation adds the same `ttnn.transpose(-2,-1)` to both, so the
delta screened here is the whole delta. `torch.equal` on both outputs, because E6 does arithmetic
and a near-miss is the failure mode that ships silently.

Priced per fold at the end: 440 TriMul calls per OpenBind fold (measured by the lever census, both
trimuls of 192 pairformer block executions plus the template and MSA pair stacks).

    python3 perf/openbind/e6_screen_cz128.py --tokens 512 544 768 1024 --out <json>
"""
from __future__ import annotations

import argparse, json, os, statistics as st, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch                                                                  # noqa: E402
import ttnn                                                                   # noqa: E402
import tt_bio.tenstorrent as T                                                # noqa: E402
from tt_bio import reblock_permute as RP                                       # noqa: E402

C_Z = 128                       # the OF3 trunk's pair width, and the trimul hidden width
TRIMUL_CALLS_PER_FOLD = 440     # lever census, every rung: 440 TriangleMultiplication calls
DRAM = ttnn.DRAM_MEMORY_CONFIG


def timed(fn, n=5, warm=2):
    """Median of n synced calls. Batched, not one-shot: an isolated synced op over-syncs and
    inflates ~2x (`tt-bio-isolated-op-timing-oversync-inflates-cost`), so both arms are timed the
    same way and only the delta is quoted."""
    for _ in range(warm):
        for o in fn():
            ttnn.deallocate(o)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        outs = fn()
        ttnn.synchronize_device(outs[0].device())
        ts.append(time.perf_counter() - t0)
        for o in outs:
            ttnn.deallocate(o)
    return st.median(ts) * 1e3


def stock(fused, slice_c):
    """What the trimul runs today, starting-node orientation, for BOTH roles."""
    g_a, g_b, p_a, p_b = ttnn.chunk(fused, chunks=4, dim=-1)
    a = ttnn.multiply_(p_a, g_a, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
    b = ttnn.multiply_(p_b, g_b, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
    ttnn.deallocate(g_a)
    ttnn.deallocate(g_b)
    am = RP.reblock_permute(a, memory_config=DRAM, device=a.device())
    bm = RP.reblock_permute(b, memory_config=DRAM, device=b.device())
    ttnn.deallocate(a)
    ttnn.deallocate(b)
    return [am, bm]


def gated(fused, slice_c):
    """E6: the gate rides inside the channel move, no chunk and no multiply."""
    am = RP.reblock_permute_gated(fused, 2 * slice_c, 0, slice_c, memory_config=DRAM)
    bm = RP.reblock_permute_gated(fused, 3 * slice_c, slice_c, slice_c, memory_config=DRAM)
    return [am, bm]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, nargs="+", default=[512, 544, 768, 1024])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    from tt_bio.main import _detect_p300_devices, _find_ttnn_mesh_graph_descriptor
    if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
        if mgd:
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd
    dev = T.get_device()
    out = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "grid": list(T.COMPUTE_GRID_MAIN), "c_z": C_Z,
           "trimul_calls_per_fold": TRIMUL_CALLS_PER_FOLD, "rows": []}

    for n in a.tokens:
        fused = ttnn.from_torch(torch.randn(1, n, n, 4 * C_Z).bfloat16(), layout=ttnn.TILE_LAYOUT,
                                device=dev, dtype=ttnn.bfloat16, memory_config=DRAM)
        elig = RP.eligible_gated(fused, C_Z, DRAM)
        row = {"tokens": n, "eligible_gated": bool(elig),
               "MB_per_role": round(n * n * C_Z * 2 / 1e6, 1)}
        if not elig:
            row["reject"] = "eligible_gated said no"
        else:
            ms_a = timed(lambda f=fused: stock(f, C_Z))
            ms_b = timed(lambda f=fused: gated(f, C_Z))
            sa, sb = stock(fused, C_Z), gated(fused, C_Z)
            eq = [bool(torch.equal(ttnn.to_torch(x), ttnn.to_torch(y))) for x, y in zip(sa, sb)]
            for x in sa + sb:
                ttnn.deallocate(x)
            row.update({"stock_ms": round(ms_a, 4), "gated_ms": round(ms_b, 4),
                        "speedup": round(ms_a / ms_b, 4),
                        "saved_ms_per_trimul": round(ms_a - ms_b, 4),
                        "saved_s_per_fold": round((ms_a - ms_b) * TRIMUL_CALLS_PER_FOLD / 1e3, 3),
                        "bit_exact": eq})
            print(f"{n:5d} tokens: stock {ms_a:8.3f} ms  gated {ms_b:8.3f} ms  "
                  f"{ms_a / ms_b:5.3f}x  saves {(ms_a - ms_b) * TRIMUL_CALLS_PER_FOLD / 1e3:6.2f} "
                  f"s/fold  bit_exact={eq}", flush=True)
        out["rows"].append(row)
        ttnn.deallocate(fused)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
