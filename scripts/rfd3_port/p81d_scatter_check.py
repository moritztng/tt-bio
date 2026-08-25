#!/usr/bin/env python3
"""p81d -- does ttnn.scatter share ttnn.gather's silent-wrong-answer threshold?

This is not a perf question. The SHIPPED RFD3 atom attention scatters the compact pair bias into a
dense [B,H,L,6080] tensor on dim 3 with a [B,H,L,128] index (the `dense_bias` route in
tt_bio/rfd3/model.py::_sparse_qk_inputs). p81c showed ttnn.GATHER on dim 3 silently corrupts every
tile-row after the first once the indexed dim reaches 2048 elements, dtype-independent. NK=6080 is
three times past that. If ttnn.scatter has the same threshold, the shipped model is computing a
wrong attention bias and this stops being a perf pass.

Also pins the exact gather boundary in (1920, 2048].
"""
import sys, os, json, pathlib, statistics, time
import torch, ttnn
sys.path.insert(0, os.getcwd())
from tt_bio.tenstorrent import get_device

DEV = get_device()
OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "perf/p81/scatter_check.json")
torch.manual_seed(0)


def scatter_trial(L, NK, K, H, dtype, tdt, fill=-1e4):
    """Mirror of the shipped call: scatter a compact [B,H,L,K] source into a dense [B,H,L,NK]."""
    idx = torch.stack([torch.randperm(NK)[:K].sort().values for _ in range(L)])
    idx = idx.unsqueeze(0).unsqueeze(0).expand(1, H, L, K).contiguous()
    src = (torch.randn(1, H, L, K) * 3.0).to(tdt)
    base = torch.full((1, H, L, NK), fill, dtype=tdt)
    ref = base.clone().scatter_(3, idx, src)

    b_dev = ttnn.from_torch(base, layout=ttnn.TILE_LAYOUT, device=DEV, dtype=dtype)
    i_dev = ttnn.from_torch(idx.to(torch.int32), layout=ttnn.TILE_LAYOUT, device=DEV,
                            dtype=ttnn.uint32)
    s_dev = ttnn.from_torch(src, layout=ttnn.TILE_LAYOUT, device=DEV, dtype=dtype)
    got = ttnn.to_torch(ttnn.scatter(b_dev, 3, i_dev, s_dev)).to(tdt)
    bad = (got != ref)
    rows_bad = bad.any(3)[0, 0]
    first_bad = int(rows_bad.nonzero()[0]) if int(rows_bad.sum()) else -1
    ttnn.deallocate(b_dev); ttnn.deallocate(i_dev); ttnn.deallocate(s_dev)
    return int(bad.sum()), int(ref.numel()), first_bad


def gather_trial(L, NK, K):
    idx = torch.stack([torch.randperm(NK)[:K].sort().values for _ in range(L)])
    idx = idx.unsqueeze(0).unsqueeze(0).contiguous()
    src = torch.randn(1, 1, L, NK, dtype=torch.float32)
    ref = src.gather(3, idx)
    s_dev = ttnn.from_torch(src, layout=ttnn.TILE_LAYOUT, device=DEV, dtype=ttnn.float32)
    i_dev = ttnn.from_torch(idx.to(torch.int32), layout=ttnn.TILE_LAYOUT, device=DEV,
                            dtype=ttnn.uint32)
    got = ttnn.to_torch(ttnn.gather(s_dev, 3, i_dev))
    bad = int((got != ref).sum())
    ttnn.deallocate(s_dev); ttnn.deallocate(i_dev)
    return bad, idx.numel()


rows = []
print("=== ttnn.scatter on dim 3, the shipped direction ===", flush=True)
print("%-40s %12s %8s %12s" % ("case", "wrong", "%", "1st bad row"), flush=True)
CASES = [
    ("L=512  NK=1024 K=128 H=1 fp32", 512, 1024, 128, 1, ttnn.float32, torch.float32),
    ("L=512  NK=2048 K=128 H=1 fp32", 512, 2048, 128, 1, ttnn.float32, torch.float32),
    ("L=512  NK=6080 K=128 H=1 fp32", 512, 6080, 128, 1, ttnn.float32, torch.float32),
    ("L=2048 NK=6080 K=128 H=4 fp32", 2048, 6080, 128, 4, ttnn.float32, torch.float32),
    ("L=6051 NK=6080 K=128 H=4 fp32  <-- PRODUCTION", 6051, 6080, 128, 4,
     ttnn.float32, torch.float32),
    ("L=6051 NK=6080 K=128 H=4 bf16", 6051, 6080, 128, 4, ttnn.bfloat16, torch.bfloat16),
]
for name, L, NK, K, H, dt, tdt in CASES:
    try:
        bad, n, first = scatter_trial(L, NK, K, H, dt, tdt)
        print("%-40s %12d %7.3f%% %12d" % (name, bad, 100.0 * bad / n, first), flush=True)
        rows.append(dict(op="scatter", case=name, L=L, n_key=NK, K=K, H=H, wrong=bad,
                         total_elems=n, first_bad_row=first))
    except Exception as e:
        print("%-40s  EXC %s" % (name, str(e)[:80]), flush=True)
        rows.append(dict(op="scatter", case=name, exc=str(e)[:300]))

print("\n=== exact ttnn.gather boundary in (1920, 2048], L=512 K=128 fp32 ===", flush=True)
for NK in (1952, 1984, 2016, 2032, 2040, 2048):
    try:
        bad, n = gather_trial(512, NK, 128)
        print("NK=%-6d  wrong %8d  %7.2f%%" % (NK, bad, 100.0 * bad / n), flush=True)
        rows.append(dict(op="gather_boundary", n_key=NK, wrong=bad, out_elems=n))
    except Exception as e:
        print("NK=%-6d  EXC %s" % (NK, str(e)[:70]), flush=True)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({"rows": rows, "host": "qb2",
                           "card": os.environ.get("TT_VISIBLE_DEVICES")}, indent=2) + "\n")
print("\nwrote", OUT)
