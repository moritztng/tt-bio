#!/usr/bin/env python3
"""The complete per-lever count table for both models, with the bucket each lever lands in.

The bucket is NOT derived from the counts alone. `served=0, declined=0` is ambiguous on its own --
it means the code path never executed, and whether that is bucket 2 (a lever Protenix should get
and does not) or bucket 3 (Protenix has its own structure there) is a judgement about the tree.
Those judgements are written down here, one line each, so the table can be audited instead of
trusted. Everything else falls out of the counts.
"""
import json, sys

# flag -> (bucket, why). Only the calls the counts cannot make on their own.
OVERRIDE = {
    # Protenix has its own diffusion module in protenix.py and does not route through
    # tenstorrent.py's atom / token-DiT path at all. Porting these is a second port, not an
    # eligibility fix, so they are bucket 3 and belong to pvx-protenix-specific.
    "B2_BIAS_SLICE_HOIST": ("3", "boltz2.py-exclusive; protenix has its own diffusion module"),
    "B2_ADALN_S_MEMO":     ("3", "same"),
    "B2_TOKEN_DIT_SDPA":   ("3", "same"),
    "ATOM_AXIS_BUCKET":    ("3", "same"),
    "ATOM_SHIFT_GATHER_OFF": ("3", "same"),
    # Fires on a strict majority of Boltz-2's calls and a minority of Protenix's. The counts make
    # this look like bucket 1; the blocking clause makes it bucket 2 with a partial rate, which is
    # exactly the "3 of 4000 calls" case the brief asks to be separated out.
    "TRANSITION_H_CHUNK":  ("2", "110/636 on protenix, base-unraised x526 at c=256 (LEDGER N1)"),
    "RESIDUAL_L1":         ("2", "80/1128 on protenix, capacity refusal at c_z=256"),
    "QKV_MM_CONFIG":       ("2", "2256/6040 on protenix, 3784 missing _MM_BLOCK keys"),
    # Declines correctly: the fused F1 tail already deletes the read this would delete.
    "TRIMUL_FUSED_GOUT":   ("1", "160/1208, the 1048 declines read f1_tail_serves and are correct"),
    "TRIATT_HEAD_MAJOR_QKV": ("1", "1048/1208; the 160 declines are the _MM_BLOCK miss, not this gate"),
    # Offered and refused on protenix, never offered on boltz2. Real, small, and the one row where
    # boltz2 is the model that misses out.
    "TRIMUL_TAIL_F1_L1_OUT": ("2-rev", "0/1048 on protenix, never offered on boltz2"),
    # Correct declines on both models, for a reason the guard records.
    "PAIR_TRANSPOSE_VIA_ROW_MAJOR": ("3", "l1_dest_is_faster on both; a correct decline"),
    "PAIR_PROJ_MINIMAL_MATMUL": ("2", "0/1208 on protenix, no_mm_block:(8,1) -- same table as QKV_MM_CONFIG"),
}

b = json.load(open(sys.argv[1])); p = json.load(open(sys.argv[2]))
B = {r["flag"]: r for r in b["rows"]}; P = {r["flag"]: r for r in p["rows"]}

def cell(r):
    if r is None or r.get("state") == "not-imported":
        return "not-imported"
    s, d = r.get("served"), r.get("declined")
    return "no-counter" if s is None and d is None else "%s / %s" % (s, d)

def rate(r):
    if r is None or r.get("state") == "not-imported":
        return "-"
    s, d = r.get("served"), r.get("declined")
    if not isinstance(s, int) or not isinstance(d, int):
        return "-"
    return "never reached" if s + d == 0 else "%.1f %%" % (100.0 * s / (s + d))

def auto(bs, bd, ps, pd):
    if bs > 0 and ps > 0:
        return "1" if pd == 0 else "2"
    if bs > 0 and ps == 0:
        return "2"
    if bs == 0 and ps > 0:
        return "2-rev"
    return "3"

print("| lever | resolved | Boltz-2 s/d | Protenix s/d | px rate | bucket | why, where the count alone cannot say |")
print("|---|---|---|---|---|---|---|")
for r in b["rows"]:
    flag = r["flag"]
    rb, rp = B.get(flag), P.get(flag)
    res = next((x["resolved"] for x in (rb, rp)
                if x and x.get("resolved") not in (None, "not-imported")), "-")
    def n(r_, k):
        v = (r_ or {}).get(k)
        return v if isinstance(v, int) else 0
    bs, bd, ps, pd = n(rb, "served"), n(rb, "declined"), n(rp, "served"), n(rp, "declined")
    bucket, why = OVERRIDE.get(flag, (auto(bs, bd, ps, pd), ""))
    mark = "**%s**" % bucket if bucket != "3" else "3"
    print("| `%s` | `%s` | %s | %s | %s | %s | %s |"
          % (flag, str(res)[:14], cell(rb), cell(rp), rate(rp), mark, why))
