#!/usr/bin/env python3
"""Group `screen.json`'s executed signatures into named leads, and price each lead.

A lead is a set of signatures that one mechanism would move together. The point of grouping is
that the fourteen device op codes are NOT the unit a lever acts on: the diffusion transformer's
head plumbing is spread across NlpCreateHeads, Slice, ReshapeView and Transpose, and the pair
Transition's chunking is spread across Slice and Concat. Pricing per op code hides both.

For a DELETION mechanism the op's percentage of roof is irrelevant -- deleting the work recovers
the measured seconds whether the op ran at 37 % or at 99 % of its roof. For a RATE mechanism only
the gap to the mix-matched roof is available. Each lead states which kind it is, and the Transpose
seconds every lead touches are listed separately and claimed by none of them, because the
Transpose class is one of the five already worked.
"""
import json
import sys
from pathlib import Path

CLOCK_MHZ = 1350.0
J = Path(__file__).resolve().parent / "screen.json"

# lead -> (kind, [(op, first-operand-shape-as-printed, expected programs)], transpose_s_alongside,
#          note)
LEADS = [
    ("L1 diffusion+atom head-major qkv/out", "delete", [
        ("NlpCreateHeadsDeviceOperation", [1, 1, 512, 3072], 4800),
        ("NlpCreateHeadsDeviceOperation", [140, 1, 128, 128], 1200),
        ("NlpCreateHeadsDeviceOperation", [1, 1, 512, 1536], 264),
        ("NLPConcatHeadsDeviceOperation", [140, 4, 32, 32], 1200),
    ], 0.0,
     "head_dim is a whole number of tiles in PADDED form at every one of these signatures "
     "(64 = 2 tiles on the token transformer, 32 = 1 tile on the atom blocks and the trunk "
     "remainder), which is the precondition tt_bio/triatt_qkv.py's transcription needs: output "
     "tile (i, n) of the qkv matmul already IS tile (batch, head, row) of q, k or v, so only the "
     "destination address changes and no element moves inside a tile. That lever is built, "
     "shipped default-on for tri-attention (TRIATT_HEAD_MAJOR_QKV/TAIL = True), bit-exact by "
     "torch.equal at six sizes, and measured in-fold at 512 aa: TriAtt body 19719.8 -> 16716.5 ms, "
     "1.1797x. The diffusion side never got it."),
    ("L2 diffusion head-48 merge", "delete-with-cost", [
        ("ReshapeViewDeviceOperation", [1, 16, 64, 512], 4800),
        ("SliceDeviceOperation", [1, 16, 512, 64], 4800),
    ], 0.0547,
     "The token transformer's head_dim is 48 (768/16), so merging heads back needs 16 pad "
     "channels per head dropped and elements DO move inside tiles -- nlp_concat_heads cannot be "
     "used at all here and the tile-id re-point of L1 does not transcribe. The out projection "
     "could absorb it as a 16-way batched matmul accumulating over heads with zeroed weight rows "
     "on the pad lanes, which moves no element but grows that matmul's K from 768 to 1024, "
     "+33 % FLOP on a 0.0716 s matmul = +0.0236 s back. Net is the lead minus that."),
    ("L3 pair Transition 11-way chunk and concat", "stop", [
        ("SliceDeviceOperation", [1, 512, 512, 128], 2800),
        ("SliceDeviceOperation", [1, 512, 512, 128], 280),
        ("ConcatDeviceOperation", [1, 47, 512, 128], 280),
    ], 0.0,
     "tenstorrent.py:8483 `ttnn.chunk(x, -(-H // transition_h_chunk_size), dim=1)` with H=512 and "
     "an executed chunk height of 47 gives 11 chunks, 10 of 47 rows and one of 42 -- which is "
     "what the executed graph shows, settling the 47-vs-16 dispute on the fold's own dispatch. "
     "The traffic is BYTE-NEUTRAL in the chunk count: every element is read once and written "
     "once whatever the height, so chunk-height tuning cannot move one byte of this and is not a "
     "lever. Both ops already run at 92.8 % and 99.3 % of their mix-matched roof. Deleting the "
     "traffic needs a zero-copy dim-1 slice and a write-into-preallocated concat; ttnn offers "
     "neither."),
    ("L4 OuterProductMean layout round-trip", "named-not-opened", [
        ("UntilizeDeviceOperation", [1, 1, 16384, 16384], 16),
        ("ReshapeViewDeviceOperation", [1, 1, 16384, 16384], 16),
        ("TilizeDeviceOperation", [1, 512, 1024, 512], 16),
    ], 0.0455,
     "MSALayer's outer product writes [16384, 16384] and the consumer wants [524288, 512]. That "
     "reshape is not expressible as a tile relabelling, so ttnn falls back to "
     "untilize -> row-major reshape -> tilize and a transpose, four passes over the same "
     "268.4 M bf16 elements (536.9 MB). All three ops in this screen's classes run at 85-95 % of "
     "their roof, so there is no rate to win, only deletion -- and deletion means a custom "
     "outer-product kernel that writes the consumer's axis order. 16 calls per fold."),
    ("L6 diffusion SDPA", "blocked", [
        ("SDPAOperation", [1, 16, 512, 64], 4800),
        ("SDPAOperation", [1, 560, 32, 32], 1200),
    ], 0.0,
     "The biggest single item in the tail and the one with no mechanism this screen can name. "
     "Traffic-bound with margin: 50.1 FLOP per charged DRAM byte against a 260.9 FLOP/byte "
     "balance, and 5.506 TFLOP over the fold is 0.0484 s at the 113.68 TFLOP/s HiFi2 roof against "
     "0.4335 s measured, so arithmetic is 11.2 % of it. The q_chunk is NOT mis-set: the executed "
     "program_config reads q_chunk 128 / k_chunk 256 at the token site and 32 / 128 at the atom "
     "site, which is exactly what tenstorrent.py:1167 `_grid_q_chunk` specifies for work=16 on a "
     "110-core grid, and the rule carries its own measured sweep (128 is the best rung at 95.60 us "
     "against 140.40 at 32 and 182.70 at 512). The only mechanism left that CPU evidence can "
     "name is the [1,16,512,512] bias, 44.5 % of the token site's traffic, in bfp8: 0.792x the "
     "bytes, so ~0.077 s (104 Mcycles) if the rate holds. That is a dtype change on an attention "
     "bias and it needs an Angstrom reading against the seed-scatter floor, which needs a card. "
     "BLOCKED, not STOP -- and the prize is 0.077 s."),
    ("L5 Embeddings gather", "stop", [
        ("EmbeddingsDeviceOperation", None, None),
    ], 0.0,
     "21.5 GB/s, 5.5 % of the copy roof -- the worst rate in the tail. It is also a known "
     "hardware floor, not a defect of ours: ttnn gather/scatter on Blackhole are per-element-rate "
     "limited at ~10-14 cycles/element and no dtype, layout, index or sub_core_grids change moves "
     "them (measured 2026-07-28 on p150a, and a bfp8 arm at half the bytes came out 0.1 % apart, "
     "which is the proof it is not bandwidth). 0.0571 s over 18 programs."),
]


def main() -> int:
    d = json.loads(J.read_text())
    sigs = d["signatures"]
    by_op = {}
    for s in sigs:
        by_op.setdefault(s["op"], []).append(s)
    print("LEADS -- signatures grouped by the mechanism that would move them\n")
    print("%-42s %-18s %8s %8s %8s %8s"
          % ("lead", "kind", "s/fold", "Mcycles", "progs", "+Transp"))
    tot = 0.0
    out = []
    for name, kind, want, transp, note in LEADS:
        got, ss, pp = [], 0.0, 0.0
        for op, shape0, progs in want:
            if shape0 is None:              # the whole class
                hit = list(by_op.get(op, []))
            else:
                hit = [s for s in by_op.get(op, [])
                       if s["operands"] and s["operands"][0][1] == shape0
                       and abs(s["programs_per_fold"] - progs) < 1.0
                       and s not in got]
                hit = hit[:1]
            if not hit:
                print("  !! no signature for %s %s x%s" % (op, shape0, progs))
                continue
            got.extend(hit)
            ss += sum(h["s_per_fold"] for h in hit)
            pp += sum(h["programs_per_fold"] for h in hit)
        print("%-42s %-18s %8.4f %8.1f %8.0f %8.4f"
              % (name, kind, ss, ss * CLOCK_MHZ, pp, transp))
        for s in got:
            print("      %-30s %8.4f s %7.0f prog %3d core %6s%% roof  %s"
                  % (s["op"], s["s_per_fold"], s["programs_per_fold"], s["cores"],
                     s["pct_of_roof"], s["operands"][0][1]))
        tot += ss
        out.append({"lead": name, "kind": kind, "s_per_fold": round(ss, 5),
                    "Mcycles": round(ss * CLOCK_MHZ, 1), "programs_per_fold": round(pp, 1),
                    "transpose_s_alongside_not_claimed": transp, "note": note,
                    "signatures": [{"op": s["op"], "s": s["s_per_fold"],
                                    "progs": s["programs_per_fold"], "cores": s["cores"],
                                    "pct_of_roof": s["pct_of_roof"]} for s in got]})
    tail = d["controls"]["KA2_tail_total_s"]
    print("\n%-42s %-18s %8.4f %8.1f" % ("LEADS TOTAL", "", tot, tot * CLOCK_MHZ))
    print("unaccounted by any lead: %.4f s of the %.4f s tail (%.1f %%)"
          % (tail - tot, tail, 100 * (tail - tot) / tail))
    claimed = {(x["op"], x["s"]) for L in out for x in L["signatures"]}
    resid = [s for s in sigs if (s["op"], s["s_per_fold"]) not in claimed]
    print("\nRESIDUAL -- every signature over 0.005 s/fold that no lead groups")
    print("%-30s %8s %8s %6s %8s %s" % ("op", "s/fold", "progs", "core", "%roof", "in0 shape"))
    for s in sorted(resid, key=lambda x: -x["s_per_fold"]):
        print("%-30s %8.4f %8.0f %6d %8s %s"
              % (s["op"], s["s_per_fold"], s["programs_per_fold"], s["cores"],
                 s["pct_of_roof"], s["operands"][0][1] if s["operands"] else "-"))
    print("residual listed above: %.4f s; the rest of the %.4f s tail is signatures under "
          "0.005 s/fold each" % (sum(s["s_per_fold"] for s in resid), tail))

    go = sum(x["s_per_fold"] for x in out if x["kind"] == "delete")
    print("\nDELETE-kind leads only: %.4f s (%.1f Mcycles), %.2f %% of the 14.8810 s fold"
          % (go, go * CLOCK_MHZ, 100 * go / 14.8810))
    (J.parent / "leads.json").write_text(json.dumps(
        {"clock_MHz": CLOCK_MHZ, "tail_s": tail, "leads": out,
         "leads_total_s": round(tot, 5),
         "unaccounted_s": round(tail - tot, 5),
         "delete_kind_s": round(go, 5)}, indent=1))
    print("wrote", J.parent / "leads.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
