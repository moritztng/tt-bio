#!/usr/bin/env python3
"""Partition one device timeline without adding overlapping program durations."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def tick(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"expected a nonnegative integer device tick, got {value!r}")
    return value


def reconcile(data):
    """Input spans must already share one synchronized device timebase.

    No core/RISC accumulation or host-clock conversion happens here. Gaps are
    unobserved device time, not proof of CPU work. Overlap is retained separately
    rather than arbitrarily assigned to an operation class.
    """
    device, timebase = data["device"], data["timebase"]
    if not isinstance(device, str) or not device or not isinstance(timebase, str) or not timebase:
        raise ValueError("device and timebase identities are required")
    start, end = tick(data["start_tick"]), tick(data["end_tick"])
    if end <= start:
        raise ValueError("timeline must have positive duration")
    events = defaultdict(list)
    events[start], events[end] = [], []
    seen, counts, durations = set(), Counter(), Counter()
    for row in data["programs"]:
        ident, kind = row["id"], row["op_class"]
        if not isinstance(ident, str) or not ident or ident in seen:
            raise ValueError(f"missing or duplicate program identity: {ident!r}")
        if not isinstance(kind, str) or not kind:
            raise ValueError("every program needs an operation class")
        if row["device"] != device or row["timebase"] != timebase:
            raise ValueError("cannot combine devices or unsynchronized timebases")
        a, b = tick(row["start_tick"]), tick(row["end_tick"])
        if not start <= a < b <= end:
            raise ValueError(f"program {ident} is empty or outside the captured interval")
        seen.add(ident)
        counts[kind] += 1
        durations[kind] += b - a
        events[a].append((kind, 1))
        events[b].append((kind, -1))
    active, exclusive = Counter(), Counter()
    active_n = overlap = gaps = peak = 0
    previous = start
    for current in sorted(events):
        width = current - previous
        if not active_n:
            gaps += width
        elif active_n == 1:
            exclusive[next(iter(active))] += width
        else:
            overlap += width
        for kind, delta in events[current]:
            active[kind] += delta
            active_n += delta
            if not active[kind]:
                del active[kind]
        peak = max(peak, active_n)
        previous = current
    if active_n or active:
        raise ValueError("unclosed program span")
    span = end - start
    closure = span - (sum(exclusive.values()) + overlap + gaps)
    assert closure == 0
    return {
        "device": device,
        "timebase": timebase,
        "span_ticks": span,
        "program_count": len(seen),
        "programs_by_class": dict(sorted(counts.items())),
        "exclusive_ticks_by_class": dict(sorted(exclusive.items())),
        "overlap_ticks": overlap,
        "idle_or_unobserved_ticks": gaps,
        "covered_ticks": span - gaps,
        "sum_program_ticks_by_class": dict(sorted(durations.items())),
        "sum_program_ticks": sum(durations.values()),
        "peak_concurrent_programs": peak,
        "closure_ticks": closure,
        "scope": "Arithmetic partition only; does not certify marker coverage, clock calibration, or profiler perturbation.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    args = parser.parse_args()
    try:
        result = reconcile(json.loads(args.capture.read_text()))
    except (KeyError, TypeError, ValueError, OSError) as error:
        parser.exit(1, f"invalid device timeline: {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
