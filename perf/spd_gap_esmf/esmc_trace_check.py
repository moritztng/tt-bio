"""Is a replayed ESMC trace still equal to eager after other shapes were captured and run?

A B=1 replay after a B=7 trace and eager B=7 forwards read rel_l2 1.08 vs fp32 (tb-esmc-600m). This replays B=1
traces of two lengths in an interleaved order (the serving pattern), with eager B=7 forwards between, and prints each
output's rel_l2 against the same input's eager output.
Equal to eager is the bar (the traced forward claims bit-identity).

    python3 perf/spd_gap_esmf/esmc_trace_check.py --model esmc-600m
"""
import argparse
import json
from pathlib import Path

import torch

from tt_bio import esmc as E


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="esmc-600m")
    ap.add_argument("--seq-file", default="scripts/gpu_vs_tt/fixtures/prot512.seq")
    ap.add_argument("--batch", action="store_true", help="also capture and replay the B=7 trace (TT_BIO_ESMC_TRACE_BATCH)")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    page = Path(a.seq_file).read_text().strip()
    seqs = {"A": page, "B": page[:300]}
    model = E.load_esmc(a.model, trace=True)
    emb = lambda s: E.embed_sequences(model, {"q": s})[0].per_residue
    seven = {f"s{i}": page for i in range(7)}
    emb7 = lambda: torch.stack([torch.as_tensor(e.per_residue) for e in E.embed_sequences(model, seven, batch_size=8)])
    model.trace = False
    ref = {k: emb(v) for k, v in seqs.items()}
    ref7 = emb7()
    model.trace = True
    rel = lambda x, r: float((torch.as_tensor(x).float() - torch.as_tensor(r).float()).norm()
                             / torch.as_tensor(r).float().norm())
    order = list("AAABBBABAB")
    if a.batch:
        E._TRACE_BATCH = True
    order += ["7", "7", "7", "A", "B", "7", "A"]
    bad = 0
    for i, k in enumerate(order):
        out, r = (emb7(), ref7) if k == "7" else (emb(seqs[k]), ref[k])
        e = rel(out, r)
        bad += e > 1e-3
        print(json.dumps({"i": i, "input": k, "rel_l2_vs_eager": e, "traces": len(model._trace_cache),
                          "broken": model._trace_broken}), flush=True)
    print(json.dumps({"model": a.model, "batch": a.batch, "bad": int(bad)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
