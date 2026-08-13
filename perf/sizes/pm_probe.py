"""Which of the persistent-mask preconditions fails, per size. Pure host arithmetic.

`sdpa_generic.plan` reads only `.padded_shape` / `.shape` off its tensors, so the whole gate can be
evaluated without a device: no fold, no benchlock, no chance of a co-tenant moving the answer.
"""
import json, sys
sys.path.insert(0, ".")
from tt_bio import sdpa_generic as SG


class T:
    def __init__(self, shape):
        self.shape = self.padded_shape = list(shape)


def chunks(padded, prod, tile=32):
    wider = [padded // n for n in range(1, padded // tile + 1)
             if padded % n == 0 and (padded // n) % tile == 0 and padded // n > prod]
    return sorted(wider, reverse=True) + [prod]


# (size, H, k_chunk, production q_chunk, the q_chunk the fold was MEASURED to take)
CASES = [(128, 12, 128, 128, 128), (256, 12, 256, 256, 256), (512, 12, 512, 256, 512),
         (768, 12, 256, 256, 384), (1024, 12, 256, 256, 512)]
GRID = (13, 10)
CKC = (None, True, False, False)
out = []
for S, H, k_chunk, prod, measured in CASES:
    q = T([S, H, S, 32]); mask = T([1, H, S, S])
    for qc in sorted(set(chunks(S, prod) + [measured]), reverse=True):
        p = SG.plan(q, q, q, mask, q, qc, k_chunk, GRID, CKC, 1.0,
                    split=(GRID[0] * GRID[1] // H, H, 1))
        conds = {"nh_per_core==1": p["nh_per_core"] == 1, "q_per_core==1": p["q_per_core"] == 1,
                 "bcast_batch": bool(p["bcast_batch"]),
                 "not use_padded_mask": not p["use_padded_mask"],
                 "NKH==H": p["NKH"] == H, "NVH==H": p["NVH"] == H}
        out.append({"S": S, "q_chunk": qc, "is_measured_pick": qc == measured,
                    "q_num_chunks": p["q_num_chunks"], "q_per_core": p["q_per_core"],
                    "pass": all(conds.values()),
                    "fails": [k for k, v in conds.items() if not v]})
print(json.dumps(out, indent=1))
