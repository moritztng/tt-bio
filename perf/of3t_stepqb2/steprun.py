#!/usr/bin/env python3
"""The campaign's missing number: a FULL exactness-ON OF3T training step, and the same step
with the knob OFF, on one board in one card visit.

Everything here is glue. `of3t-stepfloor`'s `fullstep.py` is the step, `of3t-restep`'s
`rssprofile.py` is the memory instrument and its floor guard, and `of3t-restep`'s
`steparms.py` is the two-arm capture memoisation and the per-arm stamp -- all three imported,
none forked, none edited beyond extracting `rssprofile`'s wiring into functions so two rows
can share one instrument instead of drifting apart.

WHAT THIS ADDS over `steparms.py`, which is why it exists rather than being a flag on it:

  * A CLOCK THAT SURVIVES THE RUN'S DEATH. `fullstep.py` writes `env.aiclk_during` on the
    last line of `main()`, so all four of `of3t-restep`'s arms died with the field absent and
    no OF3T exactness-ON arm has ever produced a DURING-sampled AICLK. This samples AICLK off
    the sysfs CLASS node into the JSONL at 1 Hz, flushed per sample, from a daemon thread that
    does not fork a subprocess. A `SIGKILL` or the guard's `os._exit` now loses nothing.
  * THE HOST FACTS THE COMPARISON NEEDS. Board class, serial, PCIe current link speed and
    width, and `MemAvailable` at launch, read from sysfs with no device opened. pc negotiates
    Gen4 x8; qb2 negotiates Gen4 x4, and the exactness is a host round trip per softmax and
    per layer norm, so a step time is not portable between them without the link beside it.
  * THE ARM BOUNDARY, TAGGED. Both arms share one process so they share one capture. That
    makes arm 2's floor arm 1's residue unless somebody looks, so RSS is recorded either side
    of an explicit collection and published as `arm_boundary`.

The ON arm gets one rep. At ~65x on the forward half alone a second rep buys less than the
risk of losing the first to the memory floor, and the scope of one stated rep beats the
median of a smaller thing. The OFF arm gets four, because it costs ~25 s each, and its
median is what replaces the stale 466.702 s denominator under the published GPU gap.
"""
from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

OUT = REPO / "perf" / "of3t_stepqb2" / "out"
CLASS = Path("/sys/class/tenstorrent")
GIB = 1024.0 ** 3


def _sysfs(node: Path, name: str):
    try:
        return (node / name).read_text().strip()
    except OSError:
        return None


def class_node(card: str) -> Path:
    """The sysfs class node for a card. The directory is `tenstorrent!<n>`, not `<n>` -- the
    `!` is how the kernel escapes the `/` in the device name `tenstorrent/0`, and a path built
    as `/sys/class/tenstorrent/0` exists as neither a file nor an error: every read returns
    None and the artifact records a board of `null`."""
    return CLASS / f"tenstorrent!{card}"


def board_facts(card: str) -> dict:
    """Board class, serial, firmware and the PCIe link, with no device opened.

    The `tt_*` attributes live on the CLASS node; `<node>/device/tt_heartbeat` is empty for
    every card, so reading them through the device symlink returns nothing.
    """
    node = class_node(card)
    d = {"card": str(card),
         "board": _sysfs(node, "tt_card_type"),
         "serial": _sysfs(node, "tt_serial"),
         "fw_bundle": _sysfs(node, "tt_fw_bundle_ver"),
         "aiclk_idle_mhz": _sysfs(node, "tt_aiclk")}
    pci = (node / "device").resolve()
    d["pci"] = {"addr": pci.name,
                "current_link_speed": _sysfs(pci, "current_link_speed"),
                "current_link_width": _sysfs(pci, "current_link_width"),
                "max_link_speed": _sysfs(pci, "max_link_speed"),
                "max_link_width": _sysfs(pci, "max_link_width")}
    return d


def aiclk_thread(node: Path, stop: threading.Event, jsonl, period=1.0):
    """AICLK into the profile JSONL, off sysfs, on a thread, for the whole run.

    Not `tt-smi -s`: that forks a subprocess every sample, and at 15+ GiB resident with 500
    samples to take that is the instrument competing with its subject. `fullstep.py`'s own
    `during()` still runs inside it and still writes `env.aiclk_during`; this is the copy that
    survives a process that never reaches that line.
    """
    import perf.of3t_restep.rssprofile as R
    while not stop.is_set():
        clk = _sysfs(node, "tt_aiclk")
        if clk is not None:
            jsonl.write(json.dumps({"aiclk_mhz": int(clk), "phase": R._PHASE[-1],
                                    "load1": round(os.getloadavg()[0], 2),
                                    "t_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                           time.gmtime())}) + "\n")
            jsonl.flush()
        time.sleep(period)


def clock_summary(path: Path) -> dict:
    """min/median/max/n over the samples taken while the step was actually running.

    `during_work` excludes the setup phases, because a clock sampled while the featuriser
    resolves an MSA on the host is an idle reading and averaging it in is how an 800 MHz
    artifact gets published as a regression.
    """
    import statistics
    setup = {"import", "import_engine", "setup", "capture_build_fold", "capture_prep",
             "get_device", "declare_weights", "arm_boundary"}
    allv, work = [], []
    for line in path.read_text().splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if "aiclk_mhz" not in r:
            continue
        allv.append(r["aiclk_mhz"])
        if r.get("phase") not in setup:
            work.append(r["aiclk_mhz"])

    def red(v):
        if not v:
            return None
        return {"n": len(v), "min": min(v), "median": statistics.median(v), "max": max(v)}

    return {"source": "sysfs tt_aiclk on the class node, 1 Hz, DURING the run",
            "all_phases": red(allv), "during_work": red(work),
            "during_work_excludes": sorted(setup)}


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=384)
    ap.add_argument("--cycles", type=int, default=4)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--reps-on", type=int, default=1)
    ap.add_argument("--reps-off", type=int, default=4)
    ap.add_argument("--floor-gib", type=float, default=2.0)
    ap.add_argument("--period", type=float, default=0.2)
    ap.add_argument("--card", default=os.environ.get("TT_VISIBLE_DEVICES", "0"))
    ap.add_argument("--arms", default="on,off")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    tag = a.tag or str(a.tokens)
    OUT.mkdir(parents=True, exist_ok=True)

    import perf.of3t_restep.rssprofile as R
    jsonl = R.start(OUT / f"rss_step_{tag}.jsonl", floor_gib=a.floor_gib, period=a.period,
                    header={"tag": tag, "row": "of3t-stepqb2",
                            "host_facts": board_facts(a.card)})
    clk_stop = threading.Event()
    threading.Thread(target=aiclk_thread, args=(class_node(a.card), clk_stop, R._JSONL),
                     daemon=True).start()

    # steparms FIRST: it installs the memoised `S.capture` at import. wire_phases then wraps
    # whatever `S.capture` is, so the phase tag sits outside the memoisation and arm 2's
    # rehydrate reads as ~0 s rather than as a second capture.
    import perf.of3t_restep.steparms as A
    ag, F = R.wire_phases()

    base = ["--tokens", str(a.tokens), "--cycles", str(a.cycles), "--samples", str(a.samples)]
    facts = board_facts(a.card)
    rc = 0
    plan = [w.strip() for w in a.arms.split(",") if w.strip()]
    for name in plan:
        reps = a.reps_on if name == "on" else a.reps_off
        path = OUT / f"step_exact_{name}_{tag}.json"
        rc |= A.arm(f"exact_{name}", name == "on", base + ["--reps", str(reps)], path)
        d = json.loads(path.read_text())
        d["host_facts"] = facts
        d["avail_at_start_gib"] = round(R._mem_available_bytes() / GIB, 3)
        path.write_text(json.dumps(d, indent=1, default=str))
        with R.phase("arm_boundary"):
            before = R._rss_bytes()
            gc.collect()
            after = R._rss_bytes()
            R._JSONL.write(json.dumps({
                "arm_boundary": name, "rss_gib_before_gc": round(before / GIB, 4),
                "rss_gib_after_gc": round(after / GIB, 4),
                "note": "arm 2's floor is whatever this leaves; both arms share one process "
                        "so they can share one capture"}) + "\n")
            R._JSONL.flush()

    # The clock thread writes into the same handle `finish` closes, so it stops first.
    clk_stop.set()
    time.sleep(1.5)
    foot = R.finish(rc, 0.0, ag, period=a.period, host_facts=facts)
    summary = {"row": "of3t-stepqb2", "host_facts": facts, "footer": foot,
               "aiclk": clock_summary(jsonl), "arms": plan,
               "jsonl": str(jsonl.relative_to(REPO))}
    (OUT / f"RUN_{tag}.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary["aiclk"], indent=1), flush=True)
    print(f"[steprun] rc={rc} vmhwm={foot['vmhwm_gib']:.3f} GiB -> {OUT}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
