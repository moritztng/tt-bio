"""Does the traced SaProt forward (Saprot._dispatch_traced) return exactly what eager does, across layout switches?

Eager references first, one per batch, before any trace exists. Then the schedule runs with tracing on: a layout seen
twice in a row is captured, a new one releases the trace and runs eager, so the schedule exercises capture, replay,
release and recapture. Every call's per-residue embeddings are compared to its eager reference (max abs difference;
0.0 is bit-identical) and timed.

    TT_VISIBLE_DEVICES=N python perf/spd_m_saprot/trace_check.py --model saprot-35m --fast --out trace.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="saprot-35m")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--seq-file", default="scripts/gpu_vs_tt/fixtures/prot512.seq")
    ap.add_argument("--schedule", default="AAABBAABBBCAAC")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    from tt_bio import saprot as S
    from tt_bio.tenstorrent import trace_region_size

    model = S.load_saprot(a.model, fast=a.fast)
    page = Path(a.seq_file).read_text().strip()
    batches = {"A": {f"s{i}": page for i in range(7)},          # the page cell
               "B": {f"s{i}": page[:300] for i in range(3)},     # another length and batch
               "C": {"s0": page, "s1": page[:200]}}              # mixed lengths: a padded, masked batch
    emb = lambda r: {e.id: e.per_residue.astype(np.float32) for e in r}
    S._SAPROT_TRACE = False
    ref = {n: emb(S.embed_sequences(model, b, batch_size=8)) for n, b in batches.items()}
    S._SAPROT_TRACE = True
    rows = []
    for n in a.schedule:
        t0 = time.perf_counter()
        got = emb(S.embed_sequences(model, batches[n], batch_size=8))
        ms = (time.perf_counter() - t0) * 1e3
        diff = max(float(np.abs(got[k] - ref[n][k]).max()) for k in ref[n])
        rows.append({"batch": n, "ms": round(ms, 2), "max_abs": diff,
                     "traced": getattr(model, "_trace", None) is not None})
        print(rows[-1], flush=True)
    res = {"model": a.model, "fast": a.fast, "trace_region": trace_region_size(), "rows": rows,
           "identical": all(r["max_abs"] == 0.0 for r in rows), "refused": sorted(map(str, getattr(model, "_trace_refused", ())))}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("identical" if res["identical"] else "MISMATCH", flush=True)
    return 0 if res["identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
