#!/usr/bin/env python3
"""Census every `batched_matmul` call one real fold issues, and why the chooser declines.

The value of relaxing `_batched_matmul_search`'s `batch * m_tiles < cores` gate is
(declined calls per fold) x (per-call win), and only the first factor is unmeasured. This runs one
fold through the same in-process `scripts/gpu_vs_tt/tt_baseline.py` harness E7's fold A/B used and
counts calls per shape class, splitting the declined ones by whether the gate is what declined them.

Why in-process and not the `tt-bio predict` CLI: the CLI runs the fold in a spawned child whose
`__main__` is `tt_bio.main`, so a wrapper installed by a launcher script is never imported in the
process that does the work. `build_fold` folds in this process, so a wrapper here is the one the
model calls.

`applies` is read off the real `_batched_matmul_config` return value, not off a replica of the
guard chain -- a replica drifts, and the guard is the thing under test.

    TT_VISIBLE_DEVICES=0 python3 perf/atomwindow_reconcile/census.py \
        --model protenix-v2 --target examples/prot.yaml --samples 1 --out census_p117_s1.json
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

A3M = {"examples/prot.yaml": "prot117.a3m", "examples/prot300.yaml": "prot300.a3m"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--target", default="examples/prot.yaml")
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--sampling-steps", type=int, default=200)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import ttnn
    import tt_bio.tenstorrent as T

    counts: collections.Counter = collections.Counter()
    seen: dict = {}
    cfg_calls: list = []

    orig_cfg = T._batched_matmul_config

    def cfg_spy(batch, m_tiles, k_tiles, n_tiles, elem_bytes):
        r = orig_cfg(batch, m_tiles, k_tiles, n_tiles, elem_bytes)
        cfg_calls.append(((batch, m_tiles, k_tiles, n_tiles, elem_bytes), r))
        return r

    orig_bmm = T.batched_matmul

    def bmm_spy(x, y, compute_kernel_config=None, dtype=None):
        del cfg_calls[:]
        out = orig_bmm(x, y, compute_kernel_config=compute_kernel_config, dtype=dtype)
        sx = tuple(int(d) for d in x.shape)
        sy = tuple(int(d) for d in y.shape)
        # The guard chain short-circuits before _batched_matmul_config on a rank/dtype/layout
        # mismatch, so "not called" and "called and returned None" are different declines.
        if cfg_calls:
            (batch, mt, kt, nt, eb), r = cfg_calls[-1]
            key = (sx, sy, str(x.dtype), batch, mt, kt, nt, eb, r is not None, "guard_ok")
        else:
            key = (sx, sy, str(x.dtype), 0, 0, 0, 0, 0, False, "guard_declined")
        counts[key] += 1
        seen[key] = str(r) if cfg_calls and cfg_calls[-1][1] is not None else None
        del cfg_calls[:]
        return out

    T._batched_matmul_config = cfg_spy
    T.batched_matmul = bmm_spy

    import tt_baseline as B
    B.SAMPLING_STEPS = a.sampling_steps

    msa_dir = Path(tempfile.mkdtemp(prefix="census-msa-"))
    one_fold, meta, _state = B.build_fold(
        a.model, msa_dir, ROOT / a.target, Path(B.FIXTURES) / A3M[a.target],
        samples=a.samples)

    # The model modules do `from .tenstorrent import batched_matmul`, so anything imported before
    # the rebind above holds the original. Rebinding tenstorrent alone scored a silent no-op as a
    # win once already (FINDINGS, G1/E1 pass), so re-sweep after the model is loaded.
    late = [n for n, m in list(sys.modules.items())
            if getattr(m, "batched_matmul", None) is orig_bmm]
    for n in late:
        setattr(sys.modules[n], "batched_matmul", bmm_spy)
    print("census: rebound late in", late, flush=True)

    grid = tuple(int(v) for v in T.COMPUTE_GRID_MAIN)
    cores = grid[0] * grid[1]
    t_s, fold_meta = one_fold()

    rows = []
    for k, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        sx, sy, dt, batch, mt, kt, nt, eb, applies, why = k
        blocks = batch * mt
        rows.append(dict(
            in0=list(sx), in1=list(sy), dtype=dt, calls=n, batch=batch,
            m_tiles=mt, k_tiles=kt, n_tiles=nt, elem_bytes=eb, blocks=blocks,
            applies=applies, guard=why,
            # the one class the proposed change would newly admit
            gate_declined=bool(why == "guard_ok" and not applies and 2 <= blocks < cores),
            config=seen[k]))

    out = dict(
        model=a.model, target=a.target, samples=a.samples,
        sampling_steps=a.sampling_steps, recycling_steps=B.RECYCLING_STEPS,
        grid=list(grid), cores=cores,
        l1_unreserved=int(ttnn.get_max_worker_l1_unreserved_size()),
        fold_s=round(t_s, 3), n_tokens=fold_meta.get("n_tokens"),
        plddt=fold_meta.get("plddt"), late_rebound=late,
        calls_total=sum(r["calls"] for r in rows),
        calls_applied=sum(r["calls"] for r in rows if r["applies"]),
        calls_gate_declined=sum(r["calls"] for r in rows if r["gate_declined"]),
        calls_other_declined=sum(r["calls"] for r in rows
                                 if not r["applies"] and not r["gate_declined"]),
        rows=rows)
    a.out.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=2), flush=True)
    for r in rows:
        tag = "APPLIES" if r["applies"] else ("GATE" if r["gate_declined"] else "other")
        print("  %-8s %6d x %-20s @ %-20s %-16s blocks=%-5d %s"
              % (tag, r["calls"], r["in0"], r["in1"], r["dtype"], r["blocks"],
                 r["config"] or ""), flush=True)


if __name__ == "__main__":
    main()
