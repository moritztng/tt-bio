"""What binds the 5.79 s/block host-fp32 affinity pairformer block?

The fold trace priced the block but never named its limit, so this times the real module
(reference.PairformerNoSeqModule, 8 blocks, the affinity checkpoint's own args) at the traced
shape across thread counts. Thread scaling separates the candidates: near-linear = host compute
bound, flat = serial/dispatch bound, saturating early = bandwidth bound.

No device needed. Host-only, so it is immune to card co-tenancy but not to host co-tenancy —
loadavg is recorded per point.
"""
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.getcwd())
from tt_bio.reference import PairformerNoSeqModule  # noqa: E402

N = int(os.environ.get("N", "220"))
TOKEN_Z = 128
BLOCKS = int(os.environ.get("BLOCKS", "8"))
REPS = int(os.environ.get("REPS", "2"))

# The affinity checkpoint's affinity_model_args1["pairformer_args"], verbatim.
ARGS = dict(num_blocks=BLOCKS, dropout=0.25, activation_checkpointing=True)

torch.manual_seed(0)
mod = PairformerNoSeqModule(TOKEN_Z, **ARGS).eval()
z = torch.randn(1, N, N, TOKEN_Z, dtype=torch.float32)
pair_mask = torch.ones(1, N, N, dtype=torch.float32)

# Analytic FLOP count for one block, so "% of host peak" has a denominator.
# Per block: 2 triangle multiplications + 2 triangle attentions + 1 transition.
#   trimul  : a/b projections 2 * N^2*cz*cz MAC, einsum N^3*cz MAC, out proj N^2*cz*cz MAC
#   tri-att : qkv 3 * N^2*cz*cz MAC, scores N*N^2*cz MAC, values N*N^2*cz MAC, out N^2*cz*cz MAC
#   transition (factor 4 expand): 2 * N^2*cz*(4*cz) MAC
cz, n = TOKEN_Z, N
trimul = 2 * (3 * n * n * cz * cz + n ** 3 * cz)
triatt = 2 * (4 * n * n * cz * cz + 2 * n * n * n * cz)
trans = 2 * n * n * cz * (4 * cz)
FLOP_BLOCK = 2.0 * (trimul + triatt + trans)   # 2 FLOP per MAC

out = {"n_tok": N, "token_z": TOKEN_Z, "blocks": BLOCKS,
       "flop_per_block": FLOP_BLOCK, "nproc": os.cpu_count(),
       "torch": torch.__version__, "host": os.uname().nodename, "points": []}

for nt in [int(x) for x in os.environ.get("THREADS", "1,2,4,8,16").split(",")]:
    if nt > (os.cpu_count() or 1):
        continue
    torch.set_num_threads(nt)
    with torch.no_grad():
        mod(z, pair_mask)                      # warm: allocator + any lazy init
        best = None
        for _ in range(REPS):
            t = time.monotonic()
            mod(z, pair_mask)
            dt = time.monotonic() - t
            best = dt if best is None else min(best, dt)
    per_block = best / BLOCKS
    out["points"].append({
        "threads": nt, "wall_s": round(best, 4), "s_per_block": round(per_block, 4),
        "gflops": round(FLOP_BLOCK / per_block / 1e9, 2),
        "loadavg": [round(x, 2) for x in os.getloadavg()],
    })
    print("threads=%2d  %7.3f s  %6.3f s/block  %7.2f GFLOP/s  load %.1f"
          % (nt, best, per_block, FLOP_BLOCK / per_block / 1e9, os.getloadavg()[0]), flush=True)

p1 = next((p for p in out["points"] if p["threads"] == 1), None)
pmax = out["points"][-1] if out["points"] else None
if p1 and pmax and pmax["threads"] > 1:
    out["scaling"] = {
        "threads": pmax["threads"],
        "speedup": round(p1["s_per_block"] / pmax["s_per_block"], 3),
        "parallel_efficiency": round(
            (p1["s_per_block"] / pmax["s_per_block"]) / pmax["threads"], 3),
    }
    print(json.dumps(out["scaling"], indent=1))

dest = os.environ.get("OUT", "")
if dest:
    open(dest, "w").write(json.dumps(out, indent=2) + "\n")
