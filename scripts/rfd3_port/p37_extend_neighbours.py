"""L8: where the 16.941 ms of `attn_indices(k=128)` actually goes, and which exact rewrites move it.

The plan (§7 step 3) aimed L8 at `torch.cdist` -- the Gram form without the sqrt -- but the pass-1
ledger says the atom call is 16.941 ms of which **13.713 is `_extend_with_neighbours`**, and cdist is
charged to the caller because it is evaluated as an argument. So the named target is the smaller
half. This itemises the whole call and screens the rewrites that are exact by construction.

Every variant is gated on EXACT INDEX EQUALITY against the shipped path. The output is discrete, so
there is no tolerance to fall back on: one differing index is a hard fail.

Host only. No device, no lease. Timing still needs a quiet box, so run it under benchlock.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--tree", default=".")
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--chunks", type=int, nargs="+", default=[256, 512, 1024])
ap.add_argument("--out", default="perf/p37/extend_neighbours.json")
args = ap.parse_args()

ROOT = Path(args.tree).resolve()
sys.path.insert(0, str(ROOT))
PDB = ROOT / "scripts/rfd3_port/parity_artifacts/iai_protein/IAI_protein.pdb"

import tt_bio.rfd3.model as R  # noqa: E402
from tt_bio.rfd3.featurize import featurize  # noqa: E402
from tt_bio.rfd3.input import InputSpecification  # noqa: E402

INF = float("inf")


def timed(fn, reps):
    fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


spec = InputSpecification.from_dict(
    {"input": str(PDB), "contig": "A1-10,230,A31-40"})
spec.validate()
f = featurize(str(PDB), spec)
f = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in f.items()}
tok_idx = f["atom_to_token_map"].long()
L = len(tok_idx)
NK, NS = R.RFD3DiffusionModule.N_ATTN_KEYS, R.RFD3DiffusionModule.N_ATTN_SEQ
parts = R._attention_index_prefix(f, tok_idx, NK, NS)
assert "single" in parts, "fixture took the multi-chain branch; this screen is the single one"
mask, seq_idx = parts["single"]
K = parts["k"]

# The sampler's coordinates are noisy, not the crystal pose, so the neighbour graph is measured on
# a noised structure. Seeded, so the screen is reproducible.
torch.manual_seed(0)
X = (torch.randn(1, L, 3) * 16.0)

res: dict = {"L": L, "k": K, "n_seq_neighbours": NS, "reps": args.reps}

# --- how much of k a row actually fills from the distance topk -------------------------------
pad = (seq_idx == L).sum(-1)
p_max = int(pad.max())
res["pad_slots"] = {"max": p_max, "min": int(pad.min()), "mean": float(pad.float().mean())}
print(f"L={L} k={K}  padding slots per row: max={p_max} min={int(pad.min())} "
      f"mean={pad.float().mean():.1f}", flush=True)


def ref_full(x):
    """The shipped path, including the outer sort _create_attention_indices applies."""
    d = torch.cdist(x, x, p=2)
    idx = R._extend_with_neighbours(mask, seq_idx, d, K, inplace=True)
    return torch.sort(idx, dim=-1)[0]


REF = ref_full(X)


# --- itemise -----------------------------------------------------------------------------------
def _cdist():
    return torch.cdist(X, X, p=2)


D0 = _cdist()


def _fill():
    return D0.clone().masked_fill_(mask, INF)


D1 = _fill()
items = {
    "cdist": timed(_cdist, args.reps),
    "clone+masked_fill_": timed(_fill, args.reps),
    "topk(k=128)": timed(lambda: torch.topk(D1, K, dim=-1, largest=False).indices, args.reps),
    "topk(sorted=False)": timed(
        lambda: torch.topk(D1, K, dim=-1, largest=False, sorted=False).indices, args.reps),
    f"topk(k=p_max={p_max})": timed(
        lambda: torch.topk(D1, p_max, dim=-1, largest=False).indices, args.reps),
    "final sort(dim=-1)": timed(lambda: torch.sort(REF, dim=-1)[0], args.reps),
    "WHOLE CALL (cdist+extend+sort)": timed(lambda: ref_full(X), args.reps),
}
res["items_ms"] = items
print("\nitemised, median of %d:" % args.reps, flush=True)
for k, v in items.items():
    print(f"  {k:34s} {v:8.3f} ms", flush=True)


# --- variants, each gated on exact index equality -----------------------------------------------
def v_kmax(x):
    """topk only p_max deep, left-padded back to k.

    A row fills at most p_max of its k slots from the distance topk (the rest are sequence
    neighbours), and `flip` puts the NEAREST last, so the slots that get used are the last
    p_row of `fill`. Anything in the first k - p_max columns is therefore never selected.
    p_max is a property of the mask, which is coordinate-independent and already cached.
    """
    d = torch.cdist(x, x, p=2).masked_fill_(mask, INF)
    small = torch.topk(d, p_max, dim=-1, largest=False).indices.flip(dims=[-1])
    fill = torch.cat([small[..., :1].expand(*small.shape[:-1], K - p_max), small], dim=-1)
    idx = torch.where((seq_idx == L).expand_as(fill), fill, seq_idx.expand_as(fill))
    return torch.sort(idx.long(), dim=-1)[0]


def v_chunked(chunk):
    """Row-chunked, so the (L,L) distance block never leaves cache between its four passes.

    cdist writes 45 MB, masked_fill_ reads and rewrites it, topk reads it again. Rows are
    independent, so chunking changes no arithmetic per row -- unless cdist's own matmul path
    rounds differently at a different row count, which is exactly what the equality gate is for.
    """
    def go(x):
        outs = []
        for s in range(0, L, chunk):
            e = min(s + chunk, L)
            d = torch.cdist(x[:, s:e], x, p=2).masked_fill_(mask[s:e], INF)
            small = torch.topk(d, K, dim=-1, largest=False).indices.flip(dims=[-1])
            si = seq_idx[s:e]
            outs.append(torch.where((si == L).expand_as(small), small, si.expand_as(small)))
        return torch.sort(torch.cat(outs, dim=-2).long(), dim=-1)[0]
    return go


def v_chunked_kmax(chunk):
    """Both: chunked rows and a p_max-deep topk."""
    def go(x):
        outs = []
        for s in range(0, L, chunk):
            e = min(s + chunk, L)
            d = torch.cdist(x[:, s:e], x, p=2).masked_fill_(mask[s:e], INF)
            small = torch.topk(d, p_max, dim=-1, largest=False).indices.flip(dims=[-1])
            fill = torch.cat([small[..., :1].expand(*small.shape[:-1], K - p_max), small], dim=-1)
            si = seq_idx[s:e]
            outs.append(torch.where((si == L).expand_as(fill), fill, si.expand_as(fill)))
        return torch.sort(torch.cat(outs, dim=-2).long(), dim=-1)[0]
    return go


def v_gram(chunk=None):
    """Squared distance via the Gram form, no sqrt.

    `topk` over d^2 selects the same set as over d because sqrt is monotone, so the gate is
    index equality and not distance equality -- this variant is deliberately NOT bit-exact as a
    distance. What can still break it is a near-tie whose order flips under the different
    rounding, and at 3359 rows x 128 keys x 199 steps that has to be measured rather than
    argued: torch.cdist already takes the mm path at this size, so the only arithmetic
    difference is the sqrt and the explicit norm terms.
    """
    def block(xs, xa):
        n_s = (xs * xs).sum(-1)
        n_a = (xa * xa).sum(-1)
        return n_s.unsqueeze(-1) + n_a.unsqueeze(-2) - 2.0 * (xs @ xa.transpose(-1, -2))

    def go(x):
        step = chunk or L
        outs = []
        for s in range(0, L, step):
            e = min(s + step, L)
            d = block(x[:, s:e], x).masked_fill_(mask[s:e], INF)
            small = torch.topk(d, K, dim=-1, largest=False).indices.flip(dims=[-1])
            si = seq_idx[s:e]
            outs.append(torch.where((si == L).expand_as(small), small, si.expand_as(small)))
        return torch.sort(torch.cat(outs, dim=-2).long(), dim=-1)[0]
    return go


variants = [("shipped", ref_full), (f"k=p_max({p_max})", v_kmax), ("gram(no sqrt)", v_gram())]
for c in args.chunks:
    variants.append((f"chunked({c})", v_chunked(c)))
    variants.append((f"chunked({c})+kmax", v_chunked_kmax(c)))
    variants.append((f"chunked({c})+gram", v_gram(c)))

print("\nvariants (gate: exact index equality against the shipped path):", flush=True)
res["variants"] = {}
fails = 0
for label, fn in variants:
    got = fn(X)
    ok = bool(torch.equal(got, REF))
    ms = timed(lambda fn=fn: fn(X), args.reps)
    nd = 0 if ok else int((got != REF).sum())
    if not ok:
        fails += 1
    res["variants"][label] = {"exact": ok, "ms": ms, "n_differing": nd,
                              "speedup_vs_shipped": items["WHOLE CALL (cdist+extend+sort)"] / ms}
    print(f"  {label:22s} {ms:8.3f} ms  "
          + ("indices identical" if ok else f"MISMATCH {nd} of {REF.numel()}"), flush=True)

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(res, indent=2))
print(f"\nwrote {args.out}")
print("FAIL: a variant changed the indices" if fails else "all exact variants verified")
