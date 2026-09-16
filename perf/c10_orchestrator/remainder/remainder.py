#!/usr/bin/env python3
"""What is likely inside the 0.93 s of clock-immune cost the diffusion trace does not reach.

The clock-immune term is 3.952 s measured; the diffusion trace reaches about 3.02 s of it if the
term is per-call dispatch, leaving 0.93 s unowned. The one artifact that measured host STAGES at
this model's shapes is b2x_host_residual/hostpath_512_pc.json. It ran on pc's CPU, not qb2's, so
its seconds do not transfer -- only its structure and rough scale do.

CPU only; measures nothing.
"""
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "b2x_host_residual" / "hostpath_512_pc.json"
FIXED_S = 3.9516          # ../fixed_cost/two_clock_session.json, measured
TRACE_REACH_S = 3.02      # ../dispatch_hypothesis/trace_reach.json, derived
NOW_ON_DEVICE = {"diffusion_conditioning"}   # TT_BIO_DEVICE_CONDITIONING defaults True on main


def main():
    d = json.loads(SRC.read_text())
    stages = d["stages_s"]
    still_host = {k: v for k, v in stages.items() if k not in NOW_ON_DEVICE}
    moved = {k: v for k, v in stages.items() if k in NOW_ON_DEVICE}
    remainder = FIXED_S - TRACE_REACH_S
    accounted = sum(still_host.values())
    out = {
        "scope": "CPU comparison of a measured host-stage census against a derived remainder. "
                 "No device, no new timing.",
        "source": {"file": str(SRC), "host": d["env"]["host"], "cpus": d["env"]["cpus"],
                   "torch_threads": d["env"]["threads"], "started": d["env"]["started"],
                   "note": d["env"]["note"]},
        "clock_immune_term_s": FIXED_S,
        "diffusion_trace_reach_s": TRACE_REACH_S,
        "unowned_remainder_s": remainder,
        "host_stages_measured_on_pc_s": stages,
        "moved_to_device_since": moved,
        "still_host_total_s": accounted,
        "fraction_of_remainder_accounted": accounted / remainder,
        "reading": ("Removing the stage that has since gone device-resident leaves %.3f s of "
                    "named, measured host work -- feature preparation, the input embedder, "
                    "relative-position encoding and CIF writing -- against a %.2f s remainder. "
                    "That is about %.0f %% of it, and it is real work rather than overhead, which "
                    "is why the trace cannot touch it and why the remainder is hard."
                    % (accounted, remainder, 100 * accounted / remainder)),
        "limits": [
            "Measured on pc's CPU with 12 cpus and 6 torch threads, not on qb2. The seconds do "
            "not transfer; only the structure and rough scale do.",
            "Older tree, 2026-09-11. Relative-position gathering has shipped as a host lever "
            "since, so rel_pos is probably smaller now.",
            "That census ran CPU-only with synthetic tensors at the model's shapes, by its own "
            "note, so it is a cost model of the host path and not an observation of a real fold.",
            "The 0.93 s is itself derived, being the measured 3.952 s minus a derived 3.02 s "
            "reach. Two derived numbers subtracted do not make a measurement.",
            "The agreement could be coincidence. It is a hypothesis for c10-fixed-cost's 298 aa "
            "arm to discriminate, since featurization and CIF writing scale with target size "
            "while per-call dispatch scales with call count.",
        ],
    }
    json.dump(out, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
