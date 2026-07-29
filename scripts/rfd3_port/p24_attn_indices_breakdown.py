"""Where does the per-step `attn_indices` host cost actually go?

p22 measured `host.attn_indices` at 26.0 ms/step (D=1) and 165.1 ms/step (D=8) at 3359 atoms
and p21/p22/p23 all carried "move it on device" forward untouched. An on-device topk port
carries a real tie-breaking parity risk, so measure the breakdown before paying for it: if
the cost is the two full (D,L,L) materializations rather than the topk itself, the lever is
host-side and free of any parity argument.

Pure host measurement -- no device needed.
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
ap.add_argument("--specs", type=Path, nargs="*", default=[])
ap.add_argument("--contigs", nargs="*", default=[])
ap.add_argument("--batches", type=int, nargs="+", default=[1, 8])
ap.add_argument("--reps", type=int, default=5)
args = ap.parse_args()

ROOT = Path(args.tree).resolve()
sys.path.insert(0, str(ROOT))
PDB = ROOT / "scripts/rfd3_port/parity_artifacts/iai_protein/IAI_protein.pdb"

import tt_bio.rfd3 as R  # noqa: E402
from tt_bio.rfd3_featurize import featurize  # noqa: E402
from tt_bio.rfd3_input import InputSpecification  # noqa: E402


def timed(fn, reps):
    fn()                                    # warm
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

for name, pdb_path, spec_data in cases:
    s = InputSpecification.from_dict(spec_data)
    s.validate()
    f = featurize(pdb_path, s)
    f = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
         for k, v in f.items()}
    tok_idx = f["atom_to_token_map"].long()      # atoms -> tokens, as the sampler passes it
    L = len(tok_idx)
    N_KEYS, N_SEQ = R.RFD3DiffusionModule.N_ATTN_KEYS, R.RFD3DiffusionModule.N_ATTN_SEQ
    parts = R._attention_index_prefix(f, tok_idx, N_KEYS, N_SEQ)
    torch.manual_seed(0)
    X1 = torch.randn(1, L, 3) * 16.0
    print("\n%s  L=%d  n_keys=%d  branch=%s" % (
        name, L, N_KEYS, "single" if "single" in parts else "multi-chain"), flush=True)
    for D in args.batches:
        X = X1.expand(D, -1, -1).contiguous()
        whole = timed(lambda: R._create_attention_indices(f, X, tok_idx, N_KEYS, N_SEQ),
                      args.reps)
        cdist = timed(lambda: torch.cdist(X, X, p=2), args.reps)
        D_LL = torch.cdist(X, X, p=2)
        sq = timed(lambda: torch.cdist(X, X, p=2).pow_(2), args.reps)
        if "single" in parts:
            mask, seq_idx = parts["single"]
            k = parts["k"]
        else:
            mask, seq_idx = parts["intra"]
            k = parts["kc"]
        inf = torch.tensor(float("inf"))
        where = timed(lambda: torch.where(mask, inf, D_LL), args.reps)
        md = torch.where(mask, inf, D_LL)
        topk = timed(lambda: torch.topk(md, k, dim=-1, largest=False).indices, args.reps)
        idx = R._create_attention_indices(f, X, tok_idx, N_KEYS, N_SEQ)
        srt = timed(lambda: torch.sort(idx, dim=-1)[0], args.reps)
        rest = whole - cdist - where - topk - srt
        print("  D=%-2d total=%7.1f ms | cdist=%6.1f  where(mask,inf)=%6.1f  "
              "topk(k=%d)=%7.1f  sort=%5.1f  other=%6.1f   [cdist^2 no-sqrt=%.1f]" % (
                  D, whole, cdist, where, k, topk, srt, rest, sq), flush=True)
