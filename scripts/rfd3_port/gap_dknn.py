#!/usr/bin/env python3
"""gap_dknn: RFD3's atom neighbour index on the card (tt_bio.dknn) against the model's _neighbours_kdtree.

6051 atoms in 685 tokens, mask = sequence window of 2 tokens either side (N_ATTN_SEQ), k 128, noise 0-160 A on a
protein-like chain. Per sigma: rows equal to the k-d tree (must be all), device ms/call (median of N synced calls,
upload of the coordinates included), readback + host join ms, and the k-d tree's own ms at this process's threads.
    python scripts/rfd3_port/gap_dknn.py OUT.json [reps]
"""
import json, os, pathlib, statistics, sys, time

sys.path.insert(0, os.getcwd())
import numpy as np                                                          # noqa: E402
import torch                                                                # noqa: E402
import ttnn                                                                 # noqa: E402

from tt_bio import dknn                                                     # noqa: E402
from tt_bio.tenstorrent import get_device                                  # noqa: E402
from tt_bio.rfd3 import model as M                                         # noqa: E402

OUT = pathlib.Path(sys.argv[1])
N_REP = int(sys.argv[2]) if len(sys.argv) > 2 else 10
L, K, NT = 6051, 128, 685
rng = np.random.default_rng(0)
tok = np.repeat(np.arange(NT), np.diff(np.linspace(0, L, NT + 1).astype(int)))
ca = np.cumsum(rng.normal(size=(NT, 3)) * 2.2, 0)
ca = (ca - ca.mean(0)) / ca.std(0) * 18.0
x0 = ca[tok] + rng.normal(size=(L, 3)) * 1.2
mask = torch.from_numpy(np.abs(tok[:, None] - tok[None, :]) <= 2)
rows = torch.arange(L).unsqueeze(0).expand(L, L)
seq = torch.where(mask, rows, L).topk(K, dim=1, largest=False, sorted=True).values
n_fill = (K - (seq < L).sum(1)).numpy()
mask_np = mask.numpy()


def med(f, n):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); out = f(); ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts), out


dev = get_device()
rec = {"L": L, "k": K, "threads": torch.get_num_threads(), "card": os.environ.get("TT_VISIBLE_DEVICES"),
       "host": os.uname().nodename, "sigmas": []}
for sig in (0.0, 0.3, 1.0, 5.0, 30.0, 160.0):
    x = torch.from_numpy((x0 + rng.normal(size=x0.shape) * sig).astype(np.float32))
    t_ref, ref = med(lambda: M._neighbours_kdtree(mask, seq, x.unsqueeze(0), K), 5)

    def run():
        o = dknn.outside_neighbours(dev, x.numpy(), mask_np, n_fill, K)
        ttnn.synchronize_device(dev)
        return o
    ttnn.deallocate(run())
    t_dev, o = med(run, N_REP)
    t_join, got = med(lambda: dknn.join(seq.numpy(), ttnn.to_torch(o).numpy(), L, K), 5)
    neq = (got != ref).any(-1).squeeze(0)
    r = {"sigma": sig, "rows_differ": int(neq.sum()), "elements_differ": int((got != ref).sum()),
         "device_ms": t_dev, "readback_join_ms": t_join, "kdtree_ms": t_ref}
    rec["sigmas"].append(r)
    print("[dknn] sigma %6.1f rows_differ %d device %.2f ms readback+join %.2f ms kdtree %.2f ms"
          % (sig, r["rows_differ"], t_dev, t_join, t_ref), flush=True)
    ttnn.deallocate(o)
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(rec, indent=1))
