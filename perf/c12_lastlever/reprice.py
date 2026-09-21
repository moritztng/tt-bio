#!/usr/bin/env python3
"""What the head-major qkv lever (C12 "T1") is worth, recomputed from the one session that ran.

The campaign booked 0.2340 s for this lever. That figure was inherited from the parent row as a
PREDICTION and was never measured: passes 1-9 of `c12-diffusion-head-major` were dispatched
`card=cpu`, and pass 10 got one signature onto a chip. This recomputes the entry from that
session's RAW reps plus the pinned in-situ pricing, and writes down exactly which part is measured.

Reads, and nothing else:
  perf/c12_diffusion_head/ab_qkv_qb2c3_apb_trunk.json   qb2 card 3, 7 reps, cold rep discarded
  perf/c12_diffusion_head/insitu.json                   the 1350 MHz in-situ reprice

    python3 reprice.py            # report + artifact
    python3 reprice.py --negctrl  # invert the session's arms; every verdict below must flip
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
HEAD = OUT.parent / "c12_diffusion_head"

CALLS_PER_FOLD = {"apb_trunk": 264, "dit_token": 4800}
# The built scope, from the row's own in-situ reprice at a during-sampled 1350 MHz.
DELETED_S = 0.34131          # split + concat + the atom pad and slice the lever removes
MOVED_S = 0.31172            # projection the lever moves off `ttnn.linear`
BOOKED_S = 0.2340            # what the campaign carried for this lever
KILL_S = 0.10                # the row's own pre-registered kill line


def med(v):
    return sorted(v)[len(v) // 2]


def session(path, negctrl=False):
    d = json.loads(Path(path).read_text())
    rows = []
    for s in d["signatures"]:
        raw = dict(s["raw_ms"])
        if negctrl:
            # Swap the shipped arm with the lever's. Every "the lever costs" verdict must flip.
            raw["A0_linear_split"], raw["B_head_major"] = (raw["B_head_major"],
                                                           raw["A0_linear_split"])
        m = {k: med(v) for k, v in raw.items()}
        a0 = m["A0_linear_split"]
        calls = CALLS_PER_FOLD[s["sig"]]
        rec = {
            "sig": s["sig"],
            "reps": len(raw["A0_linear_split"]),
            "ms_per_call": m,
            "aa_floor_pct": abs(m["AA_control"] - a0) / a0 * 100.0,
            # named the way they are read: > 1 means SLOWER than the chain that ships
            "over_A0": {k: m[k] / a0 for k in m if k != "A0_linear_split"},
            "calls_per_fold": calls,
            # what the lever DELIVERS at this signature: shipped chain minus lever chain
            "delivers_ms_per_call": a0 - m["B_head_major"],
            "delivers_s_per_fold": (a0 - m["B_head_major"]) * calls / 1e3,
            # the op-class leg on its own: both chains run the SAME split, so the difference is
            # entirely minimal_matmul vs ttnn.linear
            "op_class_cost_ms_per_call": m["A1_mm_split"] - a0,
            "op_class_cost_s_per_fold": (m["A1_mm_split"] - a0) * calls / 1e3,
            "torch_equal_A1_vs_B": s.get("torch_equal_A1_vs_B"),
            "max_abs_A1_vs_B": s.get("max_abs_A1_vs_B"),
        }
        eff = abs(rec["over_A0"]["B_head_major"] - 1.0) * 100.0
        rec["effect_pct_vs_A0"] = eff
        rec["effect_over_aa_floor"] = eff / rec["aa_floor_pct"] if rec["aa_floor_pct"] else None
        rec["above_aa_floor"] = eff > rec["aa_floor_pct"]
        rows.append(rec)
    return d, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--negctrl", action="store_true")
    ap.add_argument("--out", default="reprice.json")
    a = ap.parse_args()

    src = HEAD / "ab_qkv_qb2c3_apb_trunk.json"
    d, rows = session(src, a.negctrl)

    # break-even on the BUILT scope: net = DELETED - MOVED * (r - 1), r = lever chain / shipped
    r_kill = 1.0 + (DELETED_S - KILL_S) / MOVED_S
    r_zero = 1.0 + DELETED_S / MOVED_S

    res = {
        "source": str(src.relative_to(HEAD.parents[1])),
        "negctrl": a.negctrl,
        "clock_MHz": 1350,
        "clock_in_session_artifact": d.get("clock_MHz_before"),
        "in_situ": {"deleted_s": DELETED_S, "moved_s": MOVED_S,
                    "booked_s": BOOKED_S, "kill_s": KILL_S},
        "break_even": {"ratio_that_reaches_the_kill_line": r_kill,
                       "ratio_that_reaches_zero": r_zero},
        "signatures": rows,
        "measured_class_share": {},
    }

    # what fraction of the lever's class actually carries a device number
    apb = next(r for r in rows if r["sig"] == "apb_trunk")
    res["measured_class_share"] = {
        "signature": "apb_trunk", "in_situ_s": 0.00581, "class_in_situ_s": 0.31376,
        "share_pct": 0.00581 / 0.31376 * 100.0,
        "unmeasured_s": 0.31376 - 0.00581,
    }
    delivered = sum(r["delivers_s_per_fold"] for r in rows)
    res["verdict"] = {
        # an unmeasured term is booked as zero, and a measured negative one is booked as measured
        "measured_delivery_s": delivered,
        "unmeasured_signatures": ["dit_token", "apb_trunk(atom)", "atom_kv"],
        "book_entry_s": 0.0 if delivered < 0 else delivered,
        "booked_was_s": BOOKED_S,
        "correction_s": (0.0 if delivered < 0 else delivered) - BOOKED_S,
        "transcription_correct_where_it_runs": apb["torch_equal_A1_vs_B"],
    }

    path = OUT / a.out
    path.write_text(json.dumps(res, indent=1) + "\n")

    print(f"SOURCE {src.name}   negctrl={a.negctrl}")
    for r in rows:
        print(f"\n{r['sig']}  {r['reps']} reps, A/A floor {r['aa_floor_pct']:.2f} %")
        for k, v in r["ms_per_call"].items():
            print(f"   {k:<18} {v*1e3:8.2f} us"
                  + ("" if k == "A0_linear_split" else f"   x{r['over_A0'][k]:.4f} vs A0"))
        print(f"   effect {r['effect_pct_vs_A0']:.2f} % = {r['effect_over_aa_floor']:.1f}x the "
              f"A/A floor -> {'ADMISSIBLE' if r['above_aa_floor'] else 'INSIDE THE FLOOR'}")
        print(f"   delivers {r['delivers_s_per_fold']*1e3:+.3f} ms/fold over "
              f"{r['calls_per_fold']} calls  ({r['delivers_s_per_fold']:+.5f} s)")
        print(f"   op-class leg alone (minimal_matmul vs linear) "
              f"{r['op_class_cost_s_per_fold']:+.5f} s/fold")
        print(f"   torch.equal(A1,B) = {r['torch_equal_A1_vs_B']}  "
              f"max_abs {r['max_abs_A1_vs_B']}")
    print(f"\nbreak-even on the built scope: the lever chain may cost up to "
          f"x{r_zero:.4f} of the shipped chain before it delivers nothing, "
          f"x{r_kill:.4f} before it misses its own {KILL_S} s kill line")
    v = res["verdict"]
    print(f"\nBOOK  measured delivery {v['measured_delivery_s']:+.5f} s   "
          f"entry {v['book_entry_s']:.4f} s   was {v['booked_was_s']:.4f} s   "
          f"correction {v['correction_s']:+.4f} s")
    print(f"WROTE {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
