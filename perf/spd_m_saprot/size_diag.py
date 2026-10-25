"""Which L1 placement makes a SaProt forward larger than the page cell clash? One forward, full Python traceback.

ls145w: on Wormhole --fast, 15 x 544 and 30 x 544 rows threw "static circular buffers ... clash with L1 buffers" with
mm_rope's q / k in DRAM, i.e. on main. Each --set turns one module flag of tt_bio.saprot to the given value before the
forward, so a run per flag says which resident L1 tensor the clashing program meets.

    TT_VISIBLE_DEVICES=N python perf/spd_m_saprot/size_diag.py --model saprot-650m --fast --n 15 --set _SAPROT_X_L1=0
"""
import argparse
import json
import time
import traceback
from pathlib import Path

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="saprot-650m")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--n", type=int, default=15, help="sequences of the page sequence in one batch")
    ap.add_argument("--seq-file", default="scripts/gpu_vs_tt/fixtures/prot512.seq")
    ap.add_argument("--set", action="append", default=[], help="saprot module flag=0|1|none")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    from tt_bio import saprot as S

    for kv in a.set:
        k, v = kv.split("=")
        setattr(S, k, None if v == "none" else bool(int(v)))
    model = S.load_saprot(a.model, fast=a.fast)
    page = Path(a.seq_file).read_text().strip()
    row = {"model": a.model, "fast": a.fast, "n": a.n, "set": a.set}
    try:
        t0 = time.perf_counter()
        S.embed_sequences(model, {f"s{i}": page for i in range(a.n)}, batch_size=a.n + 1)
        row["ms"] = round((time.perf_counter() - t0) * 1e3, 1)
    except Exception as exc:
        row["error"] = str(exc).splitlines()[3][:300] if len(str(exc).splitlines()) > 3 else str(exc)[:300]
        row["stack"] = traceback.format_exc().split("Traceback")[-1][-3000:]
        traceback.print_exc()
    row["stats"] = {k: list(getattr(S, k)) for k in ("X_L1_STATS", "LN_L1_STATS", "OUT_L1_STATS") if hasattr(S, k)}
    print(json.dumps(row, indent=1), flush=True)
    Path(a.out).write_text(json.dumps(row, indent=1))
    return 0 if "error" not in row else 1


if __name__ == "__main__":
    raise SystemExit(main())
