#!/usr/bin/env python3
"""gap_maskhg: the atom-SDPA mask writer with HG heads per group (rfd3_bias.HG, RFD3_MASKW_HG).

One process per HG (the env var picks it), so no program is shared between settings. The page fixture's shape:
pair bias [1, 4, 6051, 128] bf16, a protein-shaped sorted index [1, 1, 6051, 128] (p109's: 32 sequence + 96
spatial neighbours of a random chain), mask [1, 4, 6080, 6080] bf16. Two calls, as the model makes them:
  full  every tile written; checked against a torch scatter
  css   chunk-64 lists (sdpa_generic.css_lists), only the listed tiles written, over a NaN-dirtied buffer
Each gets ms/call (median of N synced calls) and a sha256 of what the SDPA reads (css: the listed tiles only).
The digests must match across HG; HG=1 is the shipped kernel.
    RFD3_MASKW_HG=4 python scripts/rfd3_port/gap_maskhg.py OUT.json [reps]
"""
import hashlib, json, os, pathlib, statistics, sys, time

sys.path.insert(0, os.getcwd())
import torch                                                                # noqa: E402
import ttnn                                                                 # noqa: E402

from tt_bio import rfd3_bias as RB, sdpa_generic as SG                     # noqa: E402
from tt_bio.tenstorrent import get_device                                  # noqa: E402
from tt_bio.rfd3 import model as M                                         # noqa: E402

OUT = pathlib.Path(sys.argv[1])
N_REP = int(sys.argv[2]) if len(sys.argv) > 2 else 20
H, L, K, C = 4, 6051, 128, 64
N = (L + 31) // 32 * 32
g = torch.Generator().manual_seed(0)


def protein_index():
    step = torch.randn(L, 3, generator=g)
    x = (step / step.norm(dim=1, keepdim=True) * 1.5).cumsum(0)
    i = torch.arange(L)
    seq_nb = (i[:, None] - i[None, :]).abs().float().topk(32, largest=False).indices
    d = torch.cdist(x, x)
    d.scatter_(1, seq_nb, float("inf"))
    sp_nb = d.topk(K - 32, largest=False).indices
    return torch.cat([seq_nb, sp_nb], 1).sort(1).values                    # [L, K]


def timed(fn, dev):
    ttnn.deallocate(fn())
    ttnn.synchronize_device(dev)
    ts = []
    for _ in range(N_REP):
        t0 = time.perf_counter()
        o = fn()
        ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t0) * 1e3)
        ttnn.deallocate(o)
    return {"ms_median": round(statistics.median(ts), 4), "ms_min": round(min(ts), 4)}


def sha(t):
    return hashlib.sha256(t.contiguous().view(torch.int16).numpy().tobytes()).hexdigest()[:16]


def main():
    dev = get_device()
    idx = protein_index()
    pb_t = torch.randn(1, H, L, K, generator=g).bfloat16()
    pb = ttnn.from_torch(pb_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    ix = M._sparse_attn_index_rm(idx.unsqueeze(0), dev)
    occ = torch.zeros(N // C, N // C, dtype=torch.bool)
    occ[(torch.arange(L) // C).unsqueeze(1).expand(L, K), idx // C] = True
    occ[occ.sum(1) == 0, 0] = True                                         # padded q chunks, as p109
    rows = SG.css_lists(occ)
    css = (ttnn.from_torch(rows, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev), C)
    # The tiles the css SDPA reads: (q tile, k tile) whose chunks are listed.
    need = occ.repeat_interleave(C // 32, 0).repeat_interleave(C // 32, 1)
    need_el = need.repeat_interleave(32, 0).repeat_interleave(32, 1)       # [N, N]

    rec = {"hg": RB._heads_per_group(H), "reps": N_REP, "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "host": os.uname().nodename, "arch": str(dev.arch()), "css_tile_density": float(need.float().mean())}
    full = lambda: RB.sparse_mask_bf16(pb, ix)                             # noqa: E731
    sparse = lambda: RB.sparse_mask_bf16(pb, ix, css=css)                  # noqa: E731
    rec["full"] = timed(full, dev)
    o = full()
    d = ttnn.to_torch(o)
    ttnn.deallocate(o)
    ref = torch.full((1, H, L, L), -1e4).bfloat16()
    ref.scatter_(3, idx.reshape(1, 1, L, K).expand(1, H, L, K), pb_t)
    rec["full"]["equal_torch_scatter"] = bool(torch.equal(d[:, :, :L, :L], ref))
    rec["full"]["sha"] = sha(d)
    del d, ref
    rec["css"] = timed(sparse, dev)
    junk = ttnn.from_torch(torch.full((1, H, N, N), float("nan")).bfloat16(), dtype=ttnn.bfloat16,
                           layout=ttnn.TILE_LAYOUT, device=dev)
    ttnn.deallocate(junk)
    o = sparse()
    d = ttnn.to_torch(o)
    ttnn.deallocate(o)
    rec["css"]["sha_listed"] = sha(torch.where(need_el, d, torch.zeros((), dtype=d.dtype)))
    rec["css"]["listed_finite"] = bool(torch.isfinite(d[:, :, need_el]).all())
    print(json.dumps(rec), flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
