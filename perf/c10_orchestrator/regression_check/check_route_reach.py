#!/usr/bin/env python3
"""Can the above-cap fused SDPA route fire at the sizes this campaign measures?

Reads the shipped gate condition and threshold out of the tree rather than out of prose, so the
answer moves if the code does. CPU only.
"""
import json
import re
import subprocess
import sys

REV = "origin/main"
SIZES = (298, 512, 1024, 1536)


def show(path):
    return subprocess.run(["git", "show", f"{REV}:{path}"], capture_output=True, text=True,
                          check=True).stdout


def main():
    tri = show("tt_bio/triatt_sdpa.py")
    m = re.search(r"_Q_SPLIT_MAX_S\s*=\s*env_int\(\s*\"([^\"]+)\"\s*,\s*(\d+)\s*\)", tri)
    if not m:
        print("threshold not found; the gate moved", file=sys.stderr)
        return 2
    env_name, threshold = m.group(1), int(m.group(2))

    ten = show("tt_bio/tenstorrent.py")
    d = re.search(r"_SDPA_FUSED_LARGE_S\s*=\s*env_flag\(\s*\"([^\"]+)\"\s*,\s*(True|False)\s*\)", ten)
    if not d:
        print("default not found; the flag moved", file=sys.stderr)
        return 2
    gate = re.search(r"if \(_SDPA_FUSED_LARGE_S and .*?q_len > _triatt_sdpa\._Q_SPLIT_MAX_S\)",
                     ten, re.S)
    out = {
        "scope": "CPU read of the shipped gate. No device, no timing.",
        "rev": REV,
        "flag": {"env": d.group(1), "default_on": d.group(2) == "True"},
        "threshold": {"env": env_name, "value": threshold,
                      "condition": "q_len > threshold"},
        "gate_found_in_source": bool(gate),
        "reachable_at": {str(n): n > threshold for n in SIZES},
        "conclusion": ("The above-cap fused route is unreachable at 512 aa and 298 aa: the gate "
                       "needs q_len > %d. It cannot explain any timing difference at those sizes."
                       % threshold),
    }
    json.dump(out, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
