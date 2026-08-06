"""Candidate rewrites of the per-step attn_indices build, measured and checked for equality.

p24's breakdown showed the cost is spread, not concentrated: at 3359 atoms D=8 it is
cdist 85.1 ms, where(mask,inf) 42.5 ms, topk 68.2 ms of 211.5 ms total. There is no single
op to replace, so the lever has to be memory traffic -- three separate (D,L,L) fp32 tensors,
361 MB each at D=8, written and re-read in full.

Two candidates, both intended to be bit-exact (they change allocation, not arithmetic):

  A  masked_fill_  -- `torch.where(mask, inf, D_LL)` allocates and writes a whole second
     (D,L,L); filling D_LL in place writes only the masked positions. Safe only where D_LL
     is not read again afterwards, which is the single-chain branch.
  B  per-design loop -- process one design at a time so a (1,L,L) slice (45 MB at 3359)
     stays resident instead of streaming three 361 MB tensors. Bit-exactness is the open
     question: batched cdist and topk must agree with their per-slice equivalents.

The indices are integers, so "bit-exact" here is exact index equality, and any mismatch is a
hard fail rather than a tolerance.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--tree", required=True)
ap.add_argument("--contigs", nargs="*", default=[])
ap.add_argument("--specs", type=Path, nargs="*", default=[])
ap.add_argument("--batches", type=int, nargs="+", default=[1, 8])
ap.add_argument("--reps", type=int, default=5)
args = ap.parse_args()

ROOT = Path(args.tree).resolve()
sys.path.insert(0, str(ROOT))
PDB = ROOT / "scripts/rfd3_port/parity_artifacts/iai_protein/IAI_protein.pdb"

import tt_bio.rfd3.model as R  # noqa: E402
from tt_bio.rfd3.featurize import featurize  # noqa: E402
from tt_bio.rfd3.input import InputSpecification  # noqa: E402

INF = float("inf")


def baseline(f, X, tok_idx, n_keys, n_seq):
    return R._create_attention_indices(f, X, tok_idx, n_keys, n_seq)


def variant_a(f, X, tok_idx, n_keys, n_seq):
    """masked_fill_ instead of where(), single-chain branch only."""
    parts = R._attention_index_prefix(f, tok_idx, n_keys, n_seq)
    if "single" not in parts:
        return baseline(f, X, tok_idx, n_keys, n_seq)
    D_LL = torch.cdist(X, X, p=2)
    mask, seq_idx = parts["single"]
    k = min(parts["k"], D_LL.shape[-1])
    D_LL.masked_fill_(mask, INF)
    fill = torch.topk(D_LL, k, dim=-1, largest=False).indices.flip(dims=[-1])
    idx = torch.where((seq_idx == D_LL.shape[-1]).expand_as(fill), fill,
                      seq_idx.expand_as(fill)).long()
    return torch.sort(idx, dim=-1)[0].detach()


def variant_b(f, X, tok_idx, n_keys, n_seq):
    """One design at a time, so a (1,L,L) slice stays resident."""
    if X.shape[0] == 1:
        return variant_a(f, X, tok_idx, n_keys, n_seq)
    return torch.cat([variant_a(f, X[b: b + 1], tok_idx, n_keys, n_seq)
                      for b in range(X.shape[0])], dim=0)


VARIANTS = [("baseline", baseline), ("A masked_fill_", variant_a),
            ("B per-design", variant_b)]


def timed(fn, reps):
    fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


cases = [(c, str(PDB), {"input": str(PDB), "contig": c}) for c in args.contigs]
for spec_path in args.specs:
    data = json.loads(Path(spec_path).read_text())
    src = Path(data["input"])
    if not src.is_absolute():
        src = Path(spec_path).parent / src
    cases.append((Path(spec_path).parent.name, str(src), dict(data, input=str(src))))

fails = 0
for name, pdb_path, spec_data in cases:
    s = InputSpecification.from_dict(spec_data)
    s.validate()
    f = featurize(pdb_path, s)
    f = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
         for k, v in f.items()}
    tok_idx = f["atom_to_token_map"].long()
    L = len(tok_idx)
    NK, NS = R.RFD3DiffusionModule.N_ATTN_KEYS, R.RFD3DiffusionModule.N_ATTN_SEQ
    parts = R._attention_index_prefix(f, tok_idx, NK, NS)
    print("\n%s  L=%d  branch=%s" % (
        name, L, "single" if "single" in parts else "multi-chain"), flush=True)
    for D in args.batches:
        torch.manual_seed(0)
        # distinct coordinates per design -- the sampler's batch elements are not copies,
        # and identical rows would hide any per-slice difference in topk tie-breaking
        X = torch.randn(D, L, 3) * 16.0
        ref = baseline(f, X, tok_idx, NK, NS)
        for label, fn in VARIANTS:
            got = fn(f, X, tok_idx, NK, NS)
            ok = torch.equal(got, ref)
            ms = timed(lambda fn=fn: fn(f, X, tok_idx, NK, NS), args.reps)
            if not ok:
                fails += 1
                nd = int((got != ref).sum())
                print("  D=%-2d %-16s %8.1f ms   MISMATCH: %d of %d indices differ" % (
                    D, label, ms, nd, ref.numel()), flush=True)
            else:
                print("  D=%-2d %-16s %8.1f ms   indices identical" % (D, label, ms),
                      flush=True)
print("\n%s" % ("FAIL: a variant changed the indices" if fails else
                "all variants produced identical indices"), flush=True)
