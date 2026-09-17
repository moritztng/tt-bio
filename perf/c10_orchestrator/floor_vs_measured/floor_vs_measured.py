#!/usr/bin/env python3
"""The 15.031 s floor and the 2.309 s prize, checked against the two measured terms.

The campaign has quoted a 15.031 s roofline floor and a 2.309 s remaining prize all the way
through, and `c10-roofline-reset` could not revalidate them because the byte counter failed its
known-answer control. Both numbers live in one committed artifact,
`perf/roof_true/true_floor_512_qb2c2.json`, and `c10-fixed-cost` has since measured the two things
that artifact models. This compares them. No device, no new measurement; it reads the committed
JSON rather than restating its numbers.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FLOOR_JSON = HERE.parents[1] / "roof_true" / "true_floor_512_qb2c2.json"

# c10-fixed-cost, commit c4e6725bd, four pinned during-sampled clocks, 28 accepted folds per size.
F_512_S, F_512_SE = 3.9830, 0.1181
W_512_MCYC, W_512_SE = 14665.0, 121.1
W_298_MCYC = 10403.4
PIN_MHZ = 1350.0
TOKEN_RATIO = 512 / 298
WORK_RATIO = W_512_MCYC / W_298_MCYC


def load():
    return json.loads(FLOOR_JSON.read_text())


def analyse():
    d = load()
    w_s = W_512_MCYC / PIN_MHZ
    measured_fold_s = F_512_S + w_s
    out = {
        "scope": "CPU comparison of a committed floor artifact against c10-fixed-cost's two "
                 "measured terms. No device, no new fold, no new roof.",
        "floor_artifact": str(FLOOR_JSON),
        "measured": {"F_s": F_512_S, "W_Mcyc": W_512_MCYC, "W_s_at_pin": w_s,
                     "fold_s_at_pin": measured_fold_s, "pin_mhz": PIN_MHZ},
    }

    # --- the two terms, side by side ----------------------------------------------------------
    # An op whose time is set by DRAM traffic does not speed up when AICLK rises, so it lands in
    # the clock-immune term by construction. An arithmetic-bound op does. So the floor's own
    # traffic-bound / arithmetic-bound split is the same split c10-fixed-cost measured.
    tr, ar = d["s_set_by_traffic"], d["s_set_by_arithmetic"]
    out["two_terms"] = {
        "clock_immune": {"modelled_s": tr, "measured_s": F_512_S,
                         "delta_s": F_512_S - tr, "measured_pct_of_modelled": 100.0 * F_512_S / tr},
        "clock_scaled": {"modelled_s": ar, "measured_s": w_s,
                         "delta_s": w_s - ar, "measured_pct_of_modelled": 100.0 * w_s / ar},
        "reading": "Both halves land close, which is structurally interesting and is NOT a "
                   "validation: see provenance below, every input carries a different unrecorded "
                   "clock and one was measured on a different machine.",
    }

    # --- the prize was the clock --------------------------------------------------------------
    cell, floor, prize = d["roofs"]["cell_of_record_s"], d["floor_s"], d["prize_s"]
    out["prize"] = {
        "as_recorded_s": prize,
        "identity": "prize_s = cell_of_record_s - floor_s",
        "identity_holds": abs((cell - floor) - prize) < 1e-6,
        "cell_of_record_s": cell,
        "floor_s": floor,
        "cell_clock_note": "the 17.34 s cell is what fold_s = 2.901 + 15355/AICLK returns at about "
                           "1063 MHz; it was never a slower code path",
        "measured_fold_at_pinned_1350_s": measured_fold_s,
        "measured_minus_floor_s": measured_fold_s - floor,
        "prize_survives": measured_fold_s > floor,
        "verdict": "RETIRED. The fold at a pinned, recorded 1350 MHz is BELOW the floor the prize "
                   "was measured against, so the 2.309 s was the clock and not headroom. Whatever "
                   "the true floor is, the fold is not 2.309 s above it.",
    }

    # --- why the agreement cannot be promoted to a validation ---------------------------------
    out["provenance"] = [
        {"input": "stream_GBps", "value": d["roofs"]["stream_GBps"],
         "problem": "a DRAM streaming roof with no recorded clock. Achievable bandwidth depends on "
                    "how fast the dataflow RISCs issue NoC transactions, which AICLK does drive, "
                    "so it is not automatically clock-clean either."},
        {"input": "shape rates", "value": d["roofs"]["shape_rate_host"],
         "problem": "measured on pc, NOT on qb2. pc's card runs custom 130-core firmware against "
                    "qb2 p300c's 110 cores, so every per-shape achievable rate in the floor comes "
                    "from a different machine with a different core count."},
        {"input": "cell_scale", "value": d["roofs"]["cell_scale"],
         "problem": "the profiled fold was 24.644 s and was scaled by 0.7011 to land on the "
                    "17.34 s cell of record, which is itself an unrecorded ~1063 MHz number."},
        {"input": "dense_cube_TFLOPs", "value": d["roofs"]["dense_cube_TFLOPs"],
         "problem": "no recorded clock. An arithmetic roof measured on a throttled chip inflates "
                    "the arithmetic floor proportionally."},
    ]

    # --- the tension with size_scaling, and it cuts against that finding ----------------------
    # If W is set by per-shape arithmetic roofs, W should scale like the FLOPs do. It does not.
    # Reconciling the two requires the achievable rate to FALL at 298 aa -- which is exactly the
    # grid under-fill confound size_scaling names as the thing that would make it wrong.
    tension = {}
    for name, p in (("FLOPs ~ N^1", 1.0), ("FLOPs ~ N^2", 2.0), ("FLOPs ~ N^3", 3.0)):
        flop_ratio = TOKEN_RATIO ** p
        tension[name] = {
            "flop_ratio": flop_ratio,
            "required_rate_ratio_512_over_298": flop_ratio / WORK_RATIO,
            "298_achieves_pct_of_512_rate": 100.0 * WORK_RATIO / flop_ratio,
        }
    out["tension_with_size_scaling"] = {
        "work_ratio_measured": WORK_RATIO,
        "by_flop_model": tension,
        "tile_arithmetic": "a 512x512 pair tensor is 16x16 = 256 tiles over a 110-core grid, about "
                           "2.3 tiles per core; at 298 aa it pads to 320 and becomes 10x10 = 100 "
                           "tiles, under one tile per core. Severe under-fill at 298 is not "
                           "speculative, it is what those tile counts mean.",
        "reading": "Under an N^2 FLOP model the 298 aa fold would have to achieve about 48 % of "
                   "512 aa's rate for the arithmetic-roof picture and the measured work ratio to "
                   "agree. That is a plausible amount of under-fill for a 100-tile tensor on 110 "
                   "cores. So this is evidence AGAINST size_scaling's size-independent term being "
                   "real work to delete, and it raises the prior that c10-size-scaling returns "
                   "'artifact'. It does not settle it: the same arithmetic-roof picture rests on "
                   "shape rates measured on the wrong machine.",
    }

    out["hands_to_fold_census"] = [
        "Re-measure both roofs on qb2 at a recorded 1350 MHz. The floor's arithmetic half comes "
        "from pc's 130-core firmware and cannot be carried to a 110-core p300c.",
        "Report the traffic-bound and arithmetic-bound split of measured device time, because "
        "that split is what F and W are, and the model's version of it lands within 6 % and 0.5 % "
        "for reasons that may be structural or may be luck.",
        "Do not quote floor_s = 15.031 s or prize_s = 2.309 s again without re-deriving them; the "
        "fold at a recorded clock is already below the first and the second is retired.",
    ]
    out["limits"] = [
        "This compares a measurement to a model. Where they disagree the model is the suspect, and "
        "where they agree it may still be two errors cancelling.",
        "Only 512 aa is compared: no 298 aa floor artifact exists, so the size axis of the floor "
        "model is untested.",
        "Nothing here is a lever, no cycle is deleted, no accuracy is spent.",
    ]
    return out


if __name__ == "__main__":
    r = analyse()
    (HERE / "floor_vs_measured.json").write_text(json.dumps(r, indent=2) + "\n")
    t = r["two_terms"]
    print(f"clock-immune : modelled {t['clock_immune']['modelled_s']:.4f} s  measured "
          f"{t['clock_immune']['measured_s']:.4f} s  ({t['clock_immune']['measured_pct_of_modelled']:.1f} %)")
    print(f"clock-scaled : modelled {t['clock_scaled']['modelled_s']:.4f} s  measured "
          f"{t['clock_scaled']['measured_s']:.4f} s  ({t['clock_scaled']['measured_pct_of_modelled']:.1f} %)")
    p = r["prize"]
    print(f"prize {p['as_recorded_s']:.3f} s = {p['cell_of_record_s']} - {p['floor_s']:.3f}; "
          f"fold at pinned 1350 is {p['measured_fold_at_pinned_1350_s']:.3f} s, "
          f"{p['measured_minus_floor_s']:+.3f} s vs the floor -> survives: {p['prize_survives']}")
    for k, v in r["tension_with_size_scaling"]["by_flop_model"].items():
        print(f"  {k}: 298 aa would have to achieve {v['298_achieves_pct_of_512_rate']:.0f} % of "
              f"512 aa's rate")
