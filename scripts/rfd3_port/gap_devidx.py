#!/usr/bin/env python3
"""Screen an on-device build of RFD3's atom neighbour index against the host build.

    gap_devidx.py OUT.json [--fixture=perf/dsfix/fixtures/rfd3_R4.json] [--steps=6]

The host build (_neighbours_row_blocked: cdist -> mask -> top-k smallest, k=128 over ~6080 atoms)
runs every diffusion step on the CPU. ttnn.topk is multicore only for k <= 64 and a width >= 8192,
so the device arm takes the 128 nearest as two passes of 64: pad the masked distance rows to 8192,
take the 64 smallest, push everything at or below the 64th value out of reach with one `where`, and
take the next 64. No scatter. Distances are fp32 on the card and bf16 into topk (the op takes no
fp32), so rows can disagree with the reference at the boundary; the agreement columns say how often.

Inputs are the real ones: a spy on _create_attention_indices captures (mask, seq_idx, x, k) at every
call of a short page-fixture design, so early (noisy) and later steps are both screened. Per capture:
  * host_ms      the shipped host build at the process's thread count
  * dev_ms       device distances + both top-64 passes + the read-back of the indices, synced
  * agreement    per-row overlap of the final index sets (after the shipped seq_idx merge) with a
                 float64 reference, for the host fp32 build and for the device build
"""
import json
import os
import pathlib
import statistics
import sys
import time

import torch

sys.path.insert(0, os.getcwd())
import ttnn                                                               # noqa: E402
from tt_bio import tenstorrent as T                                       # noqa: E402
from tt_bio.rfd3 import design as rfd3_design                             # noqa: E402
from tt_bio.rfd3 import model as M                                        # noqa: E402

args = [a for a in sys.argv[1:] if not a.startswith("--")]
opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
OUT = pathlib.Path(args[0])
FIXTURE = pathlib.Path(opts.get("fixture", "perf/dsfix/fixtures/rfd3_R4.json"))
STEPS = int(opts.get("steps", 6))
WEIGHTS = pathlib.Path(os.environ.get("BOLTZ_CACHE", pathlib.Path.home() / ".boltz")) / "rfd3" / "weights"
W = 8192
FAR = 1e30
REPS = 5

caps = []
orig = M._create_attention_indices


def spy(f, X_L, tok_idx, n_keys, n_seq_neighbours):
    parts = M._attention_index_prefix(f, tok_idx, n_keys, n_seq_neighbours)
    if "single" in parts:
        mask, seq_idx = parts["single"]
        x = X_L if X_L.ndim == 3 else X_L.unsqueeze(0)
        caps.append((mask, seq_idx, x[:1].detach().float().clone(), int(parts["k"])))
    return orig(f, X_L, tok_idx, n_keys, n_seq_neighbours)


def merge(fill, seq_idx, L):
    """The shipped tail of _extend_with_neighbours: fill is ascending, flipped as the host does."""
    fill = fill.flip(dims=[-1])
    return torch.where((seq_idx == L).expand_as(fill), fill, seq_idx.expand_as(fill))


def ref64(mask, seq_idx, x, k):
    d = torch.cdist(x[0].double(), x[0].double()).masked_fill_(mask, float("inf"))
    return merge(torch.topk(d, k, dim=-1, largest=False).indices, seq_idx, d.shape[-1])


def overlap(a, b):
    a, b = a.sort(-1).values, b.sort(-1).values
    hits = [len(set(r.tolist()) & set(s.tolist())) / a.shape[-1] for r, s in zip(a, b)]
    return {"mean": statistics.fmean(hits), "min": min(hits),
            "rows_exact": sum(h == 1.0 for h in hits) / len(hits)}


def device_operands(dev, mask, x):
    """Explicitly padded operands, so no tile padding (zeros) can pose as a near atom: rows to a
    multiple of 32, columns to W, FAR in the mask for every masked or padded column. Each
    coordinate is its own [Lr, 1] column and [1, W] row, so d^2 is three broadcast differences in
    fp32 rather than |x|^2 + |y|^2 - 2 x.y, which cancels in the FPU's reduced-mantissa matmul."""
    L = x.shape[1]
    Lr = -(-L // 32) * 32
    xc = x[0] - x[0].mean(0)
    up = lambda t: ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
    cols, rows = [], []
    for c in range(3):
        r = torch.zeros(1, 1, Lr, 1)
        r[0, 0, :L, 0] = xc[:, c]
        k = torch.zeros(1, 1, 1, W)
        k[0, 0, 0, :L] = xc[:, c]
        rows.append(up(r))
        cols.append(up(k))
    m = torch.full((1, 1, Lr, W), FAR)
    m[0, 0, :L, :L] = torch.where(mask, FAR, 0.0)
    return rows, cols, up(m)


PHASES = {"dist": [], "topk1": [], "topk2": []}


def device_fill(dev, ops, L, ckc, timed=False):
    """Ascending indices of the 128 nearest unmasked atoms per row, built on the card."""
    rows, cols, m = ops
    t0 = time.perf_counter()
    d2 = m
    for r, k in zip(rows, cols):
        dd = ttnn.subtract(r, k)                                            # [1,1,Lr,W]
        d2 = ttnn.add(d2, ttnn.multiply(dd, dd))
    d2 = ttnn.typecast(d2, ttnn.bfloat16)
    if timed:
        ttnn.synchronize_device(dev); t1 = time.perf_counter()
    v1, i1 = ttnn.topk(d2, 64, dim=-1, largest=False, sorted=True)
    if timed:
        ttnn.synchronize_device(dev); t2 = time.perf_counter()
    t = ttnn.slice(v1, [0, 0, 0, 63], [1, 1, int(v1.shape[2]), 64])
    d2b = ttnn.where(ttnn.le(d2, t), FAR, d2)
    _v2, i2 = ttnn.topk(d2b, 64, dim=-1, largest=False, sorted=True)
    idx = ttnn.concat([i1, i2], dim=-1)
    out = ttnn.to_torch(idx).reshape(-1, 128)[:L].long()
    if timed:
        t3 = time.perf_counter()
        PHASES["dist"].append((t1 - t0) * 1e3)
        PHASES["topk1"].append((t2 - t1) * 1e3)
        PHASES["topk2"].append((t3 - t2) * 1e3)
    return out


def main():
    M._create_attention_indices = spy
    specs = json.loads(FIXTURE.read_text())
    out_dir = "/tmp/rfd3_devidx_%d" % os.getpid()
    rfd3_design.run_design(specs, out_dir, checkpoint_dir=str(WEIGHTS), from_pdb=True,
                           num_timesteps=STEPS, seed=42, num_designs=1, batch_size=1, verbose=False)
    M._create_attention_indices = orig
    dev = T.get_device()
    ckc = M._default_compute_kernel_config()
    rows = []
    for n, (mask, seq_idx, x, k) in enumerate(caps[:: max(1, len(caps) // 4)]):
        L = x.shape[1]
        assert k == 128, k
        ref = ref64(mask, seq_idx, x, k)
        host = M._neighbours_row_blocked(mask, seq_idx, x, k)[0]
        ts = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            M._neighbours_row_blocked(mask, seq_idx, x, k)
            ts.append((time.perf_counter() - t0) * 1e3)
        ops = device_operands(dev, mask, x)
        fill = device_fill(dev, ops, L, ckc)
        td = []
        for _ in range(REPS):
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            device_fill(dev, ops, L, ckc)
            td.append((time.perf_counter() - t0) * 1e3)
        for _ in range(3):
            device_fill(dev, ops, L, ckc, timed=True)
        devidx = merge(fill, seq_idx, L)
        fill_ref = torch.topk(torch.cdist(x[0].double(), x[0].double()).masked_fill_(mask, float("inf")),
                              k, dim=-1, largest=False).indices
        row = {"fill_vs_f64": overlap(fill, fill_ref),
               "phase_ms": {p: statistics.median(v[-3:]) for p, v in PHASES.items()},
               "capture": n, "L": L, "x_absmax": float(x.abs().max()),
               "host_ms": statistics.median(ts), "dev_ms": statistics.median(td),
               "host_vs_f64": overlap(host, ref), "dev_vs_f64": overlap(devidx, ref),
               "threads": torch.get_num_threads()}
        print(json.dumps(row), flush=True)
        rows.append(row)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"fixture": str(FIXTURE), "steps": STEPS, "rows": rows}, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
