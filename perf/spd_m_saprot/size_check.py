"""Does SaProt survive shapes larger than the page cell with mm_rope's q / k in L1, and match the DRAM arm exactly?

The page cell is 7 x 544 rows; `--batch-size` lets a user ask for far more rows a forward, and a single long sequence
is one forward of its own length. Each case runs with q / k in L1 (the arch default, or forced on) and in DRAM, and
the per-residue embeddings are compared (max abs difference; 0.0 is bit-identical). The run also records which head
shapes mm_rope fell back to DRAM for.

    TT_VISIBLE_DEVICES=N python perf/spd_m_saprot/size_check.py --model saprot-650m --fast --out size.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="saprot-650m")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--seq-file", default="scripts/gpu_vs_tt/fixtures/prot512.seq")
    ap.add_argument("--on", type=int, default=1, help="q / k L1 arm: 1 forced on, -1 the arch default")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    from tt_bio import mm_rope
    from tt_bio import saprot as S

    model = S.load_saprot(a.model, fast=a.fast)
    page = Path(a.seq_file).read_text().strip()
    cases = {"page": ({f"s{i}": page for i in range(7)}, 8),          # 3808 rows, the reference shape
             "b16": ({f"s{i}": page for i in range(15)}, 16),         # 15 x 544 = 8160 rows
             "b32": ({f"s{i}": page for i in range(30)}, 32),         # 30 x 544 = 16320 rows
             "long": ({"s0": (page * 4)[:2000]}, 8)}                  # one 2002-row forward
    emb = lambda r: {e.id: e.per_residue.astype(np.float32) for e in r}
    on = None if a.on < 0 else bool(a.on)
    rows = []
    for name, (seqs, bs) in cases.items():
        got = {}
        for arm, val in (("dram", False), ("l1", on)):
            S._SAPROT_MM_ROPE_L1 = val
            model._release_trace()   # a trace captured under the other arm would replay its program
            model._trace_last = None
            row = {"case": name, "arm": arm}
            try:
                t0 = time.perf_counter()
                got[arm] = emb(S.embed_sequences(model, seqs, batch_size=bs))
                row["ms"] = round((time.perf_counter() - t0) * 1e3, 1)
            except Exception as exc:   # the failure this tool exists to find: record it, keep going
                row["error"] = str(exc)[:400]
            row["refused"] = sorted(map(str, mm_rope._L1_REFUSED))
            rows.append(row)
            print(row, flush=True)
        if len(got) == 2:
            rows[-1]["max_abs"] = max(float(np.abs(got["l1"][k] - got["dram"][k]).max()) for k in got["dram"])
            print(name, "max_abs", rows[-1]["max_abs"], flush=True)
    ok = all("error" not in r for r in rows) and all(r.get("max_abs", 0.0) == 0.0 for r in rows)
    Path(a.out).write_text(json.dumps({"model": a.model, "fast": a.fast, "on": a.on, "rows": rows, "ok": ok}, indent=1))
    print("OK" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
