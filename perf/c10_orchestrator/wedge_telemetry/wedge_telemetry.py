#!/usr/bin/env python3
"""A wedge is visible in device telemetry a minute after it happens, in two counters.

Two device rows wedged this campaign and both were first misread off CPU%: the parent called one
healthy at 100 % CPU while the chip had been stopped for ten minutes. `pcpu` cannot work as a
discriminator on this stack, because tt-metal's completion-wait threads spin rather than block.

qb2 already logs something that does work. `/home/ttuser/qbcard/cardtel.py` writes
`cardtel.tsv` every 2 s with, per card, the AI clock, board power, and the NoC data-word counters.
Two of those counters separate a folding chip from a stopped one without ambiguity, and the history
is already on disk, so a wedge can be dated after the fact rather than guessed at.

Numbers below are extracted by `extract.py` from that TSV on qb2 and quoted here; pc's root is at
94 %, so the 60 MB source is not copied across.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = "qb2:/home/ttuser/qbcard/cardtel.tsv"

# c10-size-scaling's 768 aa ladder on card 0, extracted at absolute epochs (see the trap below).
# rate = words/s between consecutive 2 s samples; power in W.
FOLDING = [   # 00:52:00 .. 00:59:04 UTC, arms alternating 1350/800 MHz
    ("00:52:00", 1350, 40.0, 898620, 809017), ("00:53:00", 1350, 56.0, 1151868, 834304),
    ("00:54:00", 800, 37.0, 919244, 541930),  ("00:56:00", 1350, 105.0, 1206224, 521108),
    ("00:57:00", 800, 33.0, 2793, 4410682),   ("00:58:00", 1350, 70.0, 1203860, 537492),
    ("00:59:02", 800, 27.0, 1030858, 323861),
]
WEDGED = [    # 01:00:00 onward, same card, same pinned 800 MHz arm
    ("01:00:00", 800, 24.0, 1072938, 0), ("01:00:02", 800, 22.0, 1073046, 0),
    ("01:00:04", 800, 22.0, 1072946, 0), ("01:01:00", 800, 22.0, 1072930, 0),
    ("01:01:02", 800, 23.0, 1073066, 0), ("01:01:04", 800, 24.0, 1072412, 0),
    ("01:02:00", 800, 22.0, 1073043, 0), ("01:02:02", 800, 24.0, 1072970, 0),
    ("01:02:04", 800, 24.0, 1073030, 0), ("01:03:00", 800, 22.0, 1073112, 0),
    ("01:03:02", 800, 22.0, 1073112, 0), ("01:03:04", 800, 22.0, 1073086, 0),
]
# The second wedge, c10-fold-census's smoke, cannot be scored the same way.
SMOKE = {"window": "01:29:00 onward", "aiclk": 0, "power_W": 0.0, "mst_rd": 0, "slv_rd": 0,
         "note": "cardtel read ZERO for clock, power and both counters from 01:29, while the "
                 "fleet's own aiclk_watch (reading /sys/class/tenstorrent/tenstorrent!N/tt_aiclk) "
                 "kept reporting 800 MHz. Two readers disagreeing is itself a finding; what it "
                 "means is that this discriminator was NOT available for that wedge."}


def _stats(rows):
    mst = [r[3] for r in rows]
    slv = [r[4] for r in rows]
    pwr = [r[2] for r in rows]
    return {"n": len(rows),
            "mst_rd_min": min(mst), "mst_rd_max": max(mst),
            "mst_rd_spread_pct": 100.0 * (max(mst) - min(mst)) / max(mst) if max(mst) else 0.0,
            "slv_rd_min": min(slv), "slv_rd_max": max(slv),
            "power_min_W": min(pwr), "power_max_W": max(pwr)}


def analyse():
    f, w = _stats(FOLDING), _stats(WEDGED)
    return {
        "scope": "CPU reduction of qb2's existing 2 s telemetry log. No device opened, no fold run.",
        "source": SOURCE,
        "folding": dict(f, rows=FOLDING),
        "wedged": dict(w, rows=WEDGED),
        "discriminator": {
            "rule": "slv_rd_data_word_sent0 == 0 while mst_rd_data_word_received0 stays HIGH and "
                    "nearly CONSTANT, with board power flat and low",
            "why_it_works": "slv_rd is the device answering reads. At zero, it is serving nothing. "
                            "mst_rd pinned to a constant rate is a host poll loop at a fixed "
                            "period -- the two together are 'a host busy-polling a chip that has "
                            "stopped', which is exactly what both wedges were.",
            "folding_slv_rd_range": [f["slv_rd_min"], f["slv_rd_max"]],
            "wedged_slv_rd_range": [w["slv_rd_min"], w["slv_rd_max"]],
            "wedged_mst_rd_spread_pct": w["mst_rd_spread_pct"],
            "folding_mst_rd_spread_pct": f["mst_rd_spread_pct"],
            "folding_power_range_W": [f["power_min_W"], f["power_max_W"]],
            "wedged_power_range_W": [w["power_min_W"], w["power_max_W"]],
            "detection_latency_s": 60,
        },
        "second_wedge_not_scorable": SMOKE,
        "parsing_trap": (
            "cardtel.tsv spans 38.9 hours. Filtering it by wall-clock '%H:%M:%S' silently matches "
            "the SAME time on two different days, and the first version of this reduction did "
            "exactly that -- it mixed an idle stretch from the previous day into the fold window "
            "and produced a confident, wrong picture. Use absolute epochs. The tell was an epoch "
            "of 1789520281 in a window that should have held 1789606xxx."
        ),
        "limits": [
            "Two wedges, one of them unscorable. This is a signature from a single scored "
            "incident, not a validated detector, and it has never been run against a false "
            "positive such as a long CPU-side phase between folds.",
            "The counters are per card and cardtel samples every 2 s, so detection is ~60 s, not "
            "instant.",
            "Nothing here reproduces or explains either wedge. It dates one of them and says what "
            "to watch.",
        ],
    }


if __name__ == "__main__":
    r = analyse()
    (HERE / "wedge_telemetry.json").write_text(json.dumps(r, indent=2) + "\n")
    d = r["discriminator"]
    print(f"folding: slv_rd {d['folding_slv_rd_range'][0]:,}-{d['folding_slv_rd_range'][1]:,}/s, "
          f"power {d['folding_power_range_W'][0]}-{d['folding_power_range_W'][1]} W, "
          f"mst_rd spread {d['folding_mst_rd_spread_pct']:.0f} %")
    print(f"wedged : slv_rd {d['wedged_slv_rd_range'][0]}-{d['wedged_slv_rd_range'][1]}/s, "
          f"power {d['wedged_power_range_W'][0]}-{d['wedged_power_range_W'][1]} W, "
          f"mst_rd spread {d['wedged_mst_rd_spread_pct']:.2f} %")
