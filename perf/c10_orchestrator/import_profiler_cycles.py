#!/usr/bin/env python3
"""Join Tracy operation labels to raw device cycles, without using derived ns."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from reconcile_cycles import reconcile, tick

IDENTITY = ("DEVICE ID", "GLOBAL CALL COUNT", "METAL TRACE ID",
            "METAL TRACE REPLAY SESSION ID")
START = "DEVICE KERNEL START CYCLE"
END = "DEVICE KERNEL END CYCLE"


def integer(value):
    # Decimal-only strings avoid float rounding for large hardware timestamps.
    if not isinstance(value, str) or not value.strip().isascii() or not value.strip().isdecimal():
        raise ValueError(f"expected decimal integer, got {value!r}")
    return int(value.strip())


def identity(row):
    values = []
    for name in IDENTITY:
        value = row.get(name, "")
        if name in IDENTITY[2:] and value is not None and not value.strip():
            values.append(None)
        else:
            values.append(integer(value))
    if (values[2] is None) != (values[3] is None):
        raise ValueError("trace ID and replay session must both be present or both absent")
    return tuple(values)


def convert(op_rows, device_rows, *, device, timebase, start_tick, end_tick):
    """Require a one-to-one identity join for programs inside explicit boundaries.

    Device rows wholly outside the requested interval are counted and excluded.
    Any row straddling a boundary is an error; no clipping or median replication.
    Host-only label rows have no device identity and do not enter the join.
    """
    device = tick(device)
    start_tick, end_tick = tick(start_tick), tick(end_tick)
    if end_tick <= start_tick:
        raise ValueError("capture must have positive duration")
    labels, host_rows = {}, 0
    for row in op_rows:
        if not str(row.get("DEVICE ID", "")).strip():
            host_rows += 1
            continue
        key = identity(row)
        if key[0] != device:
            raise ValueError("mixed device IDs in operation labels")
        label = row.get("OP CODE", "")
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"missing operation label for {key}")
        if key in labels:
            raise ValueError(f"duplicate operation identity {key}")
        labels[key] = label.strip()

    programs, seen, selected, excluded = [], set(), set(), 0
    for row in device_rows:
        key = identity(row)
        if key[0] != device:
            raise ValueError("mixed device IDs in raw device report")
        if key in seen:
            raise ValueError(f"duplicate device execution identity {key}")
        seen.add(key)
        a, b = integer(row.get(START)), integer(row.get(END))
        if b <= a:
            raise ValueError(f"invalid device span for {key}")
        if b <= start_tick or a >= end_tick:
            excluded += 1
            continue
        if a < start_tick or b > end_tick:
            raise ValueError(f"device span straddles capture boundary: {key}")
        if key not in labels:
            raise ValueError(f"unlabelled device execution {key}")
        selected.add(key)
        programs.append({
            "id": ":".join("none" if x is None else str(x) for x in key),
            "op_class": labels[key], "device": str(device), "timebase": timebase,
            "start_tick": a, "end_tick": b,
        })
    if not programs:
        raise ValueError("no device execution inside capture boundaries")
    # A missing device row cannot be placed in time. Refuse to call this complete.
    missing = set(labels) - seen
    if missing:
        raise ValueError(f"operation labels without raw device spans: {sorted(missing, key=str)}")
    capture = {"device": str(device), "timebase": timebase,
               "start_tick": start_tick, "end_tick": end_tick, "programs": programs}
    return {
        "capture": capture,
        "partition": reconcile(capture),
        "join": {"host_only_label_rows": host_rows,
                 "device_executions_outside_interval": excluded,
                 "labelled_executions_in_interval": len(selected)},
        "scope": "Raw cycle interval accounting only. Explicit boundaries and a timebase name do not establish synchronization, marker completeness, clock coverage or profiler overhead.",
    }


def read_csv(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ops", type=Path, required=True)
    parser.add_argument("--device-report", type=Path, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--timebase", required=True)
    parser.add_argument("--start-cycle", type=int, required=True)
    parser.add_argument("--end-cycle", type=int, required=True)
    args = parser.parse_args()
    try:
        result = convert(read_csv(args.ops), read_csv(args.device_report),
                         device=args.device, timebase=args.timebase,
                         start_tick=args.start_cycle, end_tick=args.end_cycle)
        result["inputs"] = [{"path": str(path.resolve()),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                           for path in (args.ops, args.device_report)]
    except (ValueError, TypeError, KeyError, OSError) as error:
        parser.exit(1, f"invalid profiler evidence: {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
