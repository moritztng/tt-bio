#!/usr/bin/env python3
"""One k chunk, priced against the config the fold actually issues.

The first pass of this measurement used `(padded, min(256, padded))` as the baseline and it
DECLINED at 320 and 768, because that is not what ships: `_sdpa_chunks_shipped` returns (64,64) at
320 and (256,256) at 512/768/1024. So the baseline here is that function's own answer, and the
candidate changes exactly ONE thing -- k_chunk to the full padded length, q_chunk left alone. q
chunking is pure parallelism and cannot change a row's arithmetic, so holding it fixed makes the
A/B single-variable on both axes at once.

Geometry is the pairformer's: batch = S, 4 heads, head_dim 32.
"""
import sys, json, time, argparse
from pathlib import Path
ROOT = Path("/home/ttuser/.coworker/wt/sdpa-rowsum-normalisation-kernel-fix")
sys.path.insert(0, str(ROOT))

ap = argparse.ArgumentParser()
ap.add_argument("--sizes", default="320,512,768")
ap.add_argument("--iters", type=int, default=5)
ap.add_argument("--warmup", type=int, default=2)
ap.add_argument("--out", default="/tmp/kchunk_perf2.json")
a = ap.parse_args()

import torch, ttnn
import tt_bio.tenstorrent as T
import tt_bio.triatt_sdpa as TS
assert Path(T.__file__).resolve().is_relative_to(ROOT), T.__file__

dev = T.get_device()
HEADS, HEAD_DIM = 4, 32
res = {}

for S in [int(x) for x in a.sizes.split(",")]:
    B, padded = S, T._padded_sdpa_len(S)
    qs, ks = T._sdpa_chunks_shipped(S, S)
    torch.manual_seed(0)
    mk = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                   device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    q = mk(torch.randn(B, HEADS, S, HEAD_DIM).to(torch.bfloat16))
    k = mk(torch.randn(B, HEADS, S, HEAD_DIM).to(torch.bfloat16))
    v = mk(torch.ones(B, HEADS, S, HEAD_DIM, dtype=torch.bfloat16))
    bias = mk(torch.zeros(1, HEADS, S, S, dtype=torch.bfloat16))
    scale_inv = HEAD_DIM ** -0.5

    arms = [("AA_1", (qs, ks)), ("AA_2", (qs, ks)), ("k_full", (qs, padded))]
    # the intermediate k values, to see whether the curve is monotone in perf too
    for kc in (128, 256, 512):
        if kc < padded and padded % kc == 0 and kc != ks:
            arms.append((f"k{kc}", (qs, kc)))
    out = {"shipped_q": qs, "shipped_k": ks, "padded": padded}
    for name, (qc, kc) in arms:
        o = None
        try:
            for _ in range(a.warmup):
                o = TS.sdpa(q, k, v, bias, scale_inv, qc, kc,
                            ckc_default=T._TRIATT_FUSED_HIFI_CKC)
                if o is None:
                    break
                ttnn.deallocate(o)
        except Exception as exc:                                       # noqa: BLE001
            out[name] = {"error": str(exc)[:200], "q_chunk": qc, "k_chunk": kc}
            print(f"S{S} {name:8s} q={qc} k={kc} RAISED {str(exc)[:80]}", flush=True)
            continue
        if o is None:
            out[name] = {"declined": True, "q_chunk": qc, "k_chunk": kc}
            print(f"S{S} {name:8s} q={qc} k={kc} DECLINED", flush=True)
            continue
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(a.iters):
            ttnn.deallocate(TS.sdpa(q, k, v, bias, scale_inv, qc, kc,
                                    ckc_default=T._TRIATT_FUSED_HIFI_CKC))
        ttnn.synchronize_device(dev)
        ms = (time.perf_counter() - t0) * 1000.0 / a.iters
        out[name] = {"ms_per_call": ms, "q_chunk": qc, "k_chunk": kc,
                     "k_num_chunks": padded // kc}
        print(f"S{S} {name:8s} q={qc} k={kc} n={padded//kc} {ms:8.2f} ms/call", flush=True)

    for t in (q, k, v, bias):
        ttnn.deallocate(t)
    if all("ms_per_call" in out.get(x, {}) for x in ("AA_1", "AA_2")):
        base = (out["AA_1"]["ms_per_call"] + out["AA_2"]["ms_per_call"]) / 2
        out["_aa_spread_pct"] = 100 * abs(out["AA_2"]["ms_per_call"]
                                          - out["AA_1"]["ms_per_call"]) / base
        print(f"S{S} A/A spread {out['_aa_spread_pct']:.2f} % base {base:.2f} ms", flush=True)
        for name, d in out.items():
            if isinstance(d, dict) and "ms_per_call" in d and not name.startswith("AA"):
                d["vs_shipped"] = d["ms_per_call"] / base
                print(f"S{S}   {name:8s} {d['vs_shipped']:.3f}x shipped", flush=True)
    res[f"S{S}"] = out
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")

print("wrote", a.out, flush=True)
T.cleanup()
