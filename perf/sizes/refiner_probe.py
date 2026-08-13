"""Which shapes take the q-split, and with which chunks. Pure host arithmetic, no device.

The 512 aa fold carries a 995-token refiner and the 768 aa fold a 1494-token one. Those are the
only shapes at 512 whose q_chunk ladder can drop below the sequence, so they are the only place the
q-split can engage at a size the change was predicted to leave alone.
"""
import json, sys
sys.path.insert(0, ".")
from tt_bio import tenstorrent as TT
from tt_bio import sdpa_generic as SG


class T:
    def __init__(self, shape, padded):
        self.shape = list(shape); self.padded_shape = list(padded)


GRID, CKC, H = (13, 10), (None, True, False, False), 12
out = []
for S in (512, 768, 995, 1024, 1494):
    pad = -(-S // 32) * 32
    prod_q, prod_k = TT._sdpa_chunks_shipped(S, S)
    ladder = TT._tri_att_q_chunks(S, S)
    row = {"S": S, "padded": pad, "shipped_q_chunk": prod_q, "k_chunk": prod_k,
           "ladder": list(ladder)}
    row["cands"] = []
    for qc in ladder:
        q = T([S, H, S, 32], [pad, H, pad, 32]); mask = T([1, H, S, S], [1, H, pad, pad])
        for label, qpf in (("stock", 1), ("qsplit", max(1, -(-pad // qc)))):
            if qpf > 1 and (GRID[0] * GRID[1]) // (H * qpf) < 1:
                row["cands"].append({"q_chunk": qc, "arm": label, "illegal_split": True}); continue
            p = SG.plan(q, q, q, mask, q, qc, prod_k, GRID, CKC, 1.0,
                        split=((GRID[0] * GRID[1]) // (H * qpf), H, qpf))
            conds = {"nh_per_core==1": p["nh_per_core"] == 1, "q_per_core==1": p["q_per_core"] == 1,
                     "bcast_batch": bool(p["bcast_batch"]),
                     "not use_padded_mask": not p["use_padded_mask"],
                     "NKH==H": p["NKH"] == H, "NVH==H": p["NVH"] == H}
            row["cands"].append({
                "q_chunk": qc, "arm": label, "q_pf": qpf,
                "q_num_chunks": p["q_num_chunks"], "k_num_chunks": p.get("k_num_chunks"),
                "q_per_core": p["q_per_core"], "divides": pad % qc == 0,
                "mask_cb_tiles": p.get("k_num_chunks") and
                    p["k_num_chunks"] * (qc // 32) * (prod_k // 32),
                "gate_pass": all(conds.values()),
                "fails": [k for k, v in conds.items() if not v]})
    out.append(row)
print(json.dumps(out, indent=1))
