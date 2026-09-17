#!/usr/bin/env python3
"""Every number of record is on a chip that is off the bus, and the control for that is now clean.

qb2 node 0 was quarantined off the PCIe bus at 01:28:29Z and needs a reboot. Every number this
campaign quotes was measured on it: the 14.8813 s baseline, F = 3.9830 s, W = 14665.0 Mcycles, the
768 aa ladder. Everything measured from now on is on node 1. Without an anchor on node 1, nothing
new can be compared to anything old.

`c10-fixed-cost` left an OPEN-QUESTION about exactly this comparison and could not settle it,
because the one node-1 reading in the corpus changed three things at once. This checks whether that
is still true. It is not: the firmware variable has since been eliminated, so the control the
campaign wanted is now a single warm fold.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Read from sysfs on qb2 this pass; node 0's entries return ERR while it is quarantined.
NODES = {
    0: {"asic": "D7ADCC7E44A5908A", "board_serial": "0000046131934103", "pci": "0000:01:00.0",
        "fw_recorded": "19.15.0.0", "fw_now": None, "state": "QUARANTINED, PCI COMMAND ffff"},
    1: {"asic": "380F6B89681D36A4", "board_serial": "0000046131934103", "pci": "0000:02:00.0",
        "fw_recorded": None, "fw_now": "19.15.0.0", "state": "up, in use"},
    2: {"asic": "B9017B91B319B0DE", "board_serial": "000004613193410D", "pci": None,
        "fw_recorded": None, "fw_now": "19.15.0.0", "state": "excluded, board 410D down"},
    3: {"asic": "7A0F8390D57AB15C", "board_serial": "000004613193410D", "pci": None,
        "fw_recorded": None, "fw_now": "19.15.0.0", "state": "excluded, board 410D down"},
}
# The one node-1 fold in the corpus, and why it could not settle anything.
HISTORICAL_NODE1 = {"commit": "0df13ad9", "fold_s": 14.554, "fw": "19.11.0.0", "node": 1}
BASELINE_NODE0 = {"fold_s": 14.8813, "fw": "19.15.0.0", "node": 0,
                  "source": "c10-bare-baseline, pinned 1350 MHz, 16 folds"}


def analyse():
    gap_s = BASELINE_NODE0["fold_s"] - HISTORICAL_NODE1["fold_s"]
    out = {
        "scope": "CPU read of sysfs identity and two committed numbers. No device opened, no fold.",
        "nodes": NODES,
        "board_pairs": {"0000046131934103": [0, 1], "000004613193410D": [2, 3]},
        "incident": "node 0 quarantined off the PCIe bus at 01:28:29Z; reboot required",
        "numbers_of_record_are_all_node0": [
            "14.8813 s baseline at 512 aa", "F = 3.9830 s", "W = 14665.0 Mcycles",
            "31.5522 s at 768 aa", "the 298 aa fixture at 9.6801 s",
        ],
    }

    out["the_open_question"] = {
        "raised_by": "c10-fixed-cost",
        "gap_s": gap_s, "gap_pct": 100.0 * gap_s / BASELINE_NODE0["fold_s"],
        "node0": BASELINE_NODE0, "node1_historical": HISTORICAL_NODE1,
        "why_it_could_not_be_settled": "that reading changed chip AND firmware together -- node 1 on "
                                       "19.11.0.0 against node 0 on 19.15.0.0 -- so the +2.2 % could "
                                       "not be attributed to either.",
    }

    fw_now = {n: d["fw_now"] for n, d in NODES.items() if d["fw_now"]}
    matched = NODES[1]["fw_now"] == NODES[0]["fw_recorded"]
    out["the_confound_is_gone"] = {
        "node0_firmware_recorded": NODES[0]["fw_recorded"],
        "node1_firmware_now": NODES[1]["fw_now"],
        "match": matched,
        "readable_nodes_firmware": fw_now,
        "reading": "node 1 now runs %s, the same bundle node 0 was recorded on. The firmware "
                   "variable that blocked the attribution has been eliminated, so a warm fold on "
                   "node 1 today against node 0's %.4f s is a CLEAN ONE-VARIABLE CHIP COMPARISON -- "
                   "the control c10-fixed-cost asked for and could not run."
                   % (NODES[1]["fw_now"], BASELINE_NODE0["fold_s"]) if matched else
                   "firmware still differs; the control is NOT clean.",
    }

    out["why_it_is_now_necessary_not_just_available"] = (
        "Node 0 is off the bus, so every future measurement is on node 1. Two of the campaign's "
        "queued rows will run there, and a per-shape rate converted into fold seconds against the "
        "node-0 baseline is a cross-chip step. One warm fold on node 1 turns that from an "
        "unquantified caveat into a measured offset."
    )
    out["what_it_costs"] = (
        "One warm fold at a pinned, during-sampled 1350 MHz on the committed cdk2x2_512 fixture "
        "with its 35-row A3M, 200 steps, 3 recycles, 1 sample, seed 0 -- the c10-bare-baseline "
        "protocol unchanged, so the comparison is exact. About 15 s of device time plus warm-up, "
        "and a cold fold first since warm entry costs about 0.5 s at 512 aa."
    )
    out["interpretations"] = [
        "Node 1 reads within the 0.049 s cross-session spread of 14.8813 s: the chips are "
        "equivalent, every node-0 number carries over, and the historical +2.2 % was the firmware.",
        "Node 1 reads near the historical 14.554 s: the chips differ by about 2.2 %, node-1 "
        "measurements need that offset applied before comparison, and the campaign's ladder "
        "arithmetic has to be redone against a node-1 baseline.",
        "Either way it is one fold and it removes a caveat that otherwise attaches to every "
        "remaining measurement in the campaign.",
    ]
    out["limits"] = [
        "Nodes 0 and 1 share board serial 0000046131934103, so they are a board pair. That means a "
        "tt-smi reset hits both, and it does NOT mean they are interchangeable -- the board pair is "
        "the reset granularity, not an equivalence claim.",
        "Node 0's firmware is quoted from c10-fixed-cost's record, not read now, because its sysfs "
        "returns ERR while quarantined. If the reboot brings it back on a different bundle this "
        "comparison has to be re-checked.",
        "This artifact measures nothing. It says the control is clean and what it would cost.",
    ]
    return out


if __name__ == "__main__":
    r = analyse()
    (HERE / "node1_anchor.json").write_text(json.dumps(r, indent=2) + "\n")
    q = r["the_open_question"]
    c = r["the_confound_is_gone"]
    print(f"node 0 {NODES[0]['asic']} QUARANTINED; node 1 {NODES[1]['asic']} up")
    print(f"open question: {q['gap_s']:.4f} s ({q['gap_pct']:.1f} %) between node 0 and the one "
          f"historical node-1 fold")
    print(f"firmware: node0 recorded {c['node0_firmware_recorded']}, node1 now "
          f"{c['node1_firmware_now']} -> confound gone: {c['match']}")
