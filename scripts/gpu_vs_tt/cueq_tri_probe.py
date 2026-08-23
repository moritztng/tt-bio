#!/usr/bin/env python3
"""Time cuEquivariance triangle attention at a model's own call shape, in bf16 and fp32.

The call counter proves cuEquivariance ran. It cannot prove WHICH kernel ran, and on Blackwell
that is the whole question: the sm100f triangle-attention kernel is bf16/fp16 only, forward
hidden_dim <= 256, sequence length a multiple of 8, and it ships only in the cu13 ops wheels.
So the counter has to be paired with a per-call timing at the shape the model actually calls,
which is what settled the same question for RF3 (fp32 2.827 ms vs bf16 0.243 ms per call, an
11.65x gap that no counter would have shown).

Shapes come from a gpu5_bench result's `cueq_call_shapes` -- recorded by the run itself, never
guessed:

    python cueq_tri_probe.py --from-json /root/results/gpu_boltz-2_prot512_b200.json \
        --out /root/results/cueq_probe_b200.json
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path

SHAPE = re.compile(r"\((\d+(?:,\s*\d+)*)\)\s*:\s*torch\.(\w+)")


def shapes_from(path: Path):
    """Distinct (q-shape, dtype) pairs of triangle-attention calls, largest first."""
    rec = json.loads(path.read_text())
    sigs = rec.get("cueq_call_shapes") or rec.get("result", {}).get("cueq_call_shapes") or []
    out = []
    for s in sigs:
        if "attention" not in s.split("|")[0].lower():
            continue
        found = SHAPE.findall(s)
        if not found:
            continue
        dims, dtype = found[0]
        out.append((tuple(int(x) for x in dims.split(",")), dtype, s.split("|")[0].strip()))
    seen, uniq = set(), []
    for t in out:
        if t[:2] not in seen:
            seen.add(t[:2])
            uniq.append(t)
    return uniq


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-json", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--reps", type=int, default=20)
    args = ap.parse_args()

    found = shapes_from(args.from_json)
    print("triangle-attention call shapes read off the run:", found)
    if not found:
        print("no triangle-attention shape recorded -- nothing to probe")
        return 1

    import torch
    import cuequivariance_torch as cuet
    from importlib.metadata import version

    rec = {"source": str(args.from_json), "gpu": torch.cuda.get_device_name(0),
           "capability": list(torch.cuda.get_device_capability()),
           "torch": torch.__version__, "probes": []}
    for p in ("cuequivariance-torch", "cuequivariance-ops-torch-cu12",
              "cuequivariance-ops-torch-cu13"):
        try:
            rec[p.replace("-", "_")] = version(p)
        except Exception:
            pass

    for shape, dtype_seen, op in found:
        for dt in (torch.bfloat16, torch.float32):
            try:
                q = torch.randn(*shape, device="cuda", dtype=dt)
                k = torch.randn_like(q)
                v = torch.randn_like(q)
                bias = torch.zeros(shape[0], 1, shape[-3], shape[-2], shape[-2],
                                   device="cuda", dtype=dt)
                fn = cuet.triangle_attention
                for _ in range(3):
                    fn(q, k, v, bias=bias, scale=shape[-1] ** -0.5)
                torch.cuda.synchronize()
                ts = []
                for _ in range(args.reps):
                    t0 = time.perf_counter()
                    fn(q, k, v, bias=bias, scale=shape[-1] ** -0.5)
                    torch.cuda.synchronize()
                    ts.append((time.perf_counter() - t0) * 1e3)
                rec["probes"].append(dict(op=op, shape=list(shape), dtype=str(dt),
                                          ms_median=round(statistics.median(ts), 4),
                                          ms_min=round(min(ts), 4), n=len(ts)))
            except Exception as e:
                rec["probes"].append(dict(op=op, shape=list(shape), dtype=str(dt),
                                          error=repr(e)[:400]))
            print(rec["probes"][-1])
    if args.out:
        args.out.write_text(json.dumps(rec, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
