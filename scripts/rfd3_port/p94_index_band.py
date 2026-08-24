#!/usr/bin/env python3
"""p94 -- is the atom attention index BANDED, i.e. does a block of query rows share its keys?

Every route this task has tried at the atom site treats the 128-neighbour index as an
arbitrary per-row gather: p77's arm shared one key block across all 6051 rows (not gathered
attention at all), p82's honest arm gathered per row and came out 7.2x SLOWER than the dense
chain it replaces, and `ttnn.gather` is silently wrong above 1920 on the indexed axis anyway.

The question none of them asked: how much do NEIGHBOURING query rows share keys? If rows
i..i+31 (four residues of one chain) draw their 128 keys from a common pool of U keys with
U << 6080, the site is block-sparse, and block-sparse is a batched DENSE matmul over
[n_blocks, Q, U] -- stock ops, tile-aligned, no per-row gather and no custom kernel.

This harvests the REAL index from a real fold (structure, not timing, so box load is
irrelevant) and reports U for Q in {32, 64, 128} at several diffusion steps. The index moves
with the coordinates, so an early step (near-noise) and a late step (folded) are different
questions and both are reported.
"""
import json, os, pathlib, sys
import torch
sys.path.insert(0, os.getcwd())
from tt_bio.rfd3 import design as rfd3_design                            # noqa: E402
from tt_bio.rfd3 import model as M                                       # noqa: E402

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "perf/p94/index_band.json")
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 25
FIXTURE = pathlib.Path("perf/dsfix/fixtures/rfd3_R4.json")
CKPT = "/home/ttuser/.boltz/rfd3/weights"
SEED = 42
QBLOCKS = (32, 64, 128)

_orig = M._create_attention_indices
CALLS = {"n": 0}
GRABBED = []


def _spy(f, X_L, tok_idx, n_keys, n_seq_neighbours):
    idx = _orig(f, X_L, tok_idx, n_keys, n_seq_neighbours)
    CALLS["n"] += 1
    # the atom index is the L=6051 one; the DiT's is I=685 with n_keys=32. Keep only the
    # atom site, and only a handful of calls spread over the schedule.
    if idx.shape[-1] == 128 and idx.shape[-2] > 2000:
        GRABBED.append((CALLS["n"], idx[0].detach().cpu().clone()))
    return idx


M._create_attention_indices = _spy


def analyse(idx):
    """idx: [L, K] sorted int64 neighbour index."""
    L, K = idx.shape
    rep = {"L": L, "K": K}
    # per-row span: is a single row's own 128 keys contiguous?
    span = (idx[:, -1] - idx[:, 0] + 1)
    rep["row_span_mean"] = float(span.float().mean())
    rep["row_span_median"] = float(span.float().median())
    rep["row_contiguous_frac"] = float((span == K).float().mean())
    for Q in QBLOCKS:
        nb = (L + Q - 1) // Q
        us, spans = [], []
        for b in range(nb):
            blk = idx[b * Q:(b + 1) * Q]
            u = torch.unique(blk)
            us.append(int(u.numel()))
            spans.append(int(u[-1] - u[0] + 1))
        us_t = torch.tensor(us, dtype=torch.float64)
        sp_t = torch.tensor(spans, dtype=torch.float64)
        # tile-padded work for the block-sparse form vs the dense chain
        pad = lambda n: ((n + 31) // 32) * 32                            # noqa: E731
        work_bs = sum(pad(Q) * pad(u) for u in us)
        work_dense = ((L + 31) // 32 * 32) * 6080
        rep["Q%d" % Q] = dict(
            n_blocks=nb,
            union_mean=round(float(us_t.mean()), 1),
            union_median=round(float(us_t.median()), 1),
            union_max=int(us_t.max()),
            union_p90=round(float(us_t.quantile(0.9)), 1),
            span_mean=round(float(sp_t.mean()), 1),
            span_max=int(sp_t.max()),
            frac_union_le_1920=round(float((us_t <= 1920).float().mean()), 4),
            work_ratio_vs_dense=round(work_bs / work_dense, 4),
            speedup_if_work_bound=round(work_dense / work_bs, 2),
        )
    return rep


def main():
    specs = json.loads(FIXTURE.read_text())
    out_dir = "/tmp/rfd3_p94"
    os.system("rm -rf %s" % out_dir)
    rfd3_design.run_design(specs, out_dir, checkpoint_dir=CKPT, from_pdb=True,
                           num_timesteps=STEPS, seed=SEED, num_designs=1,
                           batch_size=1, verbose=False)
    print("[p94] atom-index calls captured: %d of %d total index builds"
          % (len(GRABBED), CALLS["n"]), flush=True)
    if not GRABBED:
        print("[p94] NO ATOM INDEX CAPTURED -- the spy did not match", flush=True)
        return
    # first, middle, last captured call: early/mid/late in the denoising schedule
    picks = [0, len(GRABBED) // 2, len(GRABBED) - 1]
    reports = []
    for tag, i in zip(("early", "mid", "late"), picks):
        call_no, idx = GRABBED[i]
        r = analyse(idx)
        r["tag"] = tag
        r["call_no"] = call_no
        r["call_frac"] = round(i / max(1, len(GRABBED) - 1), 3)
        reports.append(r)
        print("\n=== %s (index build %d/%d) L=%d K=%d" % (tag, call_no, CALLS["n"], r["L"], r["K"]), flush=True)
        print("    per-row span mean %.0f median %.0f, fully contiguous rows %.1f %%"
              % (r["row_span_mean"], r["row_span_median"], 100 * r["row_contiguous_frac"]), flush=True)
        for Q in QBLOCKS:
            d = r["Q%d" % Q]
            print("    Q=%3d  %3d blocks  union mean %7.1f  median %7.1f  p90 %7.1f  max %5d"
                  "   work %.4f of dense  (%.2fx)"
                  % (Q, d["n_blocks"], d["union_mean"], d["union_median"], d["union_p90"],
                     d["union_max"], d["work_ratio_vs_dense"], d["speedup_if_work_bound"]),
                  flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    # persist the raw indices: p95 times the block-sparse arm against the REAL index,
    # and a random permutation has none of the block structure this probe measured.
    torch.save({tag: GRABBED[i][1] for tag, i in zip(("early", "mid", "late"), picks)},
               OUT.parent / "indices.pt")
    OUT.write_text(json.dumps(dict(steps=STEPS, seed=SEED, fixture=str(FIXTURE),
                                   n_index_builds=CALLS["n"], n_atom_indices=len(GRABBED),
                                   card=os.environ.get("TT_VISIBLE_DEVICES"),
                                   reports=reports), indent=2) + "\n")
    print("\nwrote", OUT, flush=True)


if __name__ == "__main__":
    main()
