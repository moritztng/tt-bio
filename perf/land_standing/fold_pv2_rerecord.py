"""Fold the protenix-v2 size-ladder re-record (gate rec-8fb57d73a, tree main 28e026ba4 + 8fb57d73a) into
the baseline, with reasons for the cells that went dark since the 4ba29781f record.

    python perf/land_standing/fold_pv2_rerecord.py <recorded dir> CARD [CARD ...]

Two changes landed after that record and account for every new dark cell:

T2 tail (8a0d1d074, Protenix-v2 normal, both archs). `_t2_serves` hands the trimul output tail to T2 when
g_out arrives fused, which at 512/640 is T1's route. T2 does the layer norm, p projection, gate and
residual in one kernel, so the 1048 p projections that reached `_pair_proj_minimal_matmul` there are gone
(PAIR_PROJ_MINIMAL_MATMUL 1048 -> 0, QKV_MM_CONFIG 2256 -> 1208), and t2cm reads the einsum's channel-major
output itself (REBLOCK_PERMUTE_BACK 1208 -> 160). From 768 g_out is not fused and F1 keeps the tail.

triatt_ln (7533354b6, Protenix-v2 on Wormhole only). Triangle attention's layer norm and q/k/v/gate/bias
projection run as one kernel that writes all five head-major, so `qkvgb_heads` (TRIATT_HEAD_MAJOR_QKV,
QKV_MM_CONFIG) and the head-major gate projection (TRIATT_HEAD_MAJOR_TAIL, TRIATT_TAIL_OVER_L1) are never
called. Its TR mode norms x^T straight from x, so the ending variant's pair transpose (TRANSPOSE_L1_RESIDENT,
PAIR_TRANSPOSE_VIA_ROW_MAJOR, 604 calls = half of the 1208 triangle attentions) is gone too. Above 1024
triangle attention takes the chunked path, triatt_ln is not reached and all of these serve again.
WH 512 runtime 64.9 -> 55.6 s at 1000 MHz across the two records.

Every overridden cell must be dark with the counts named here, or the script refuses.
"""
import json
import pathlib
import sys

sys.path.insert(0, "scripts")
import release_gate as rg

D = pathlib.Path("docs/size_ladder_baseline.d/protenix-v2.json")
MODEL = "protenix-v2"
WH = "tt-galaxy-wh-l"
# rec-8fb57d73a (p300c, WH) and rec150-5a1667fbb (p150a, main 1f083ba6e + the triatt_k1 pricing fix)
RECORDS = ("8fb57d73a", "5a1667fbb")
TO_1024 = ("256", "512", "640", "768", "896", "1024")

T2_PROJ = ("T2 tail (trimul_t2, 8a0d1d074) takes the output tail wherever g_out arrives fused, which at "
           "512/640 is T1's route, and does the p projection inside its own kernel, so the 1048 calls that "
           "reached `_pair_proj_minimal_matmul` here before T2 are gone. From 768 F1 owns the tail with the "
           "projection fused. Falsifier: calls here at 512/640 mean T2 stopped taking T1's tails")
LN_QKV = ("triatt_ln (Wormhole Protenix-v2, 7533354b6) norms the pair and projects q/k/v/gate/bias in one "
          "kernel that writes all five head-major, so `qkvgb_heads` and the head-major gate projection are "
          "never called at 256-1024. Above 1024 triangle attention is chunked, triatt_ln is not reached and "
          "these serve. Falsifier: calls here on Wormhole mean triatt_ln declined")
LN_TR = ("triatt_ln's TR mode (Wormhole Protenix-v2, 7533354b6) norms x^T straight from x, so the ending "
         "triangle attention's pair transpose (604 calls, half of 1208) no longer runs at 256-1024. "
         "Falsifier: calls here on Wormhole mean triatt_ln declined the ending variant")

# (cards, rungs, flags, why, max declined)
JOBS = [(("p150a", "p300c", WH), ("512", "640"), ("PAIR_PROJ_MINIMAL_MATMUL",), T2_PROJ, 1),
        ((WH,), TO_1024, ("QKV_MM_CONFIG", "TRIATT_HEAD_MAJOR_QKV", "TRIATT_HEAD_MAJOR_TAIL",
                          "TRIATT_TAIL_OVER_L1"), LN_QKV, 0),
        ((WH,), TO_1024, ("TRANSPOSE_L1_RESIDENT", "PAIR_TRANSPOSE_VIA_ROW_MAJOR"), LN_TR, 0)]

rec_dir, cards = pathlib.Path(sys.argv[1]), sys.argv[2:]
data = json.loads(D.read_text())
for card in cards:
    frag = json.loads((rec_dir / card / f"{MODEL}.json").read_text())
    entry = frag["cards"][card]
    assert entry["models"][MODEL]["commit"] in RECORDS, (card, entry["models"][MODEL]["commit"])
    data["cards"][card] = entry

n = 0
for jcards, rungs, flags, why, max_declined in JOBS:
    for card in jcards:
        if card not in cards:
            continue
        levers = data["cards"][card]["models"][MODEL]["levers"]
        for rung in rungs:
            for flag in flags:
                e = levers[rung][flag]
                assert rg._size_ladder_dark(e), (card, rung, flag, "not dark")
                assert (e.get("declined") or 0) <= max_declined, (card, rung, flag, e)
                e["reason"] = rg._size_ladder_reason_evidence(e) + ": " + why
                n += 1

todo = [(c, r, f) for c in cards for r, lv in data["cards"][c]["models"][MODEL]["levers"].items()
        for f, e in lv.items() if str(e.get("reason") or "").startswith("TODO")
        or (rg._size_ladder_dark(e) and not e.get("reason"))]
assert not todo, todo
D.write_text(json.dumps(data, indent=2) + "\n")
print(f"{MODEL}: {', '.join(cards)} folded in, {n} reasons written")
