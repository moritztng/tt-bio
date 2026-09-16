#!/usr/bin/env python3
"""How much of the fold's per-call host dispatch a diffusion trace can actually reach.

Three committed artifacts, no new measurement:
  - the burst census measured 1,096 device programs in ONE diffusion step (its README states the
    four windows "cover one MSA layer, one Pairformer layer, one diffusion step and the confidence
    head"), plus 137 for one Pairformer layer, 227 for one MSA layer and 18 for the confidence head
  - the protocol runs 200 sampling steps and 3 recycles
  - the launch-floor row counted 465,664 top-level ttnn calls at 512 aa, of which about 287,000
    actually launch a program

CPU only; opens no device and measures nothing new.
"""
import json
import sys

PROGRAMS_PER = {"diffusion_step": 1096, "pairformer_layer": 137, "msa_layer": 227,
                "confidence_head": 18}
STEPS = 200
TOP_LEVEL_CALLS = 465664
LAUNCHING_CALLS = 287000          # roof_launch/LAUNCH_FLOOR.md
FIXED_S = 3.9516                  # ../fixed_cost/two_clock_session.json


def main():
    diff = PROGRAMS_PER["diffusion_step"] * STEPS
    per_call = FIXED_S / TOP_LEVEL_CALLS
    per_launch = FIXED_S / LAUNCHING_CALLS
    out = {
        "scope": "CPU arithmetic over three committed artifacts. No device, no new timing.",
        "diffusion_loop": {
            "programs_per_step": PROGRAMS_PER["diffusion_step"],
            "steps": STEPS,
            "programs": diff,
            "share_of_launching_calls_pct": 100.0 * diff / LAUNCHING_CALLS,
            "share_of_top_level_calls_pct": 100.0 * diff / TOP_LEVEL_CALLS,
        },
        "clock_immune_cost": {
            "total_s": FIXED_S,
            "per_top_level_call_us": 1e6 * per_call,
            "per_launching_call_us": 1e6 * per_launch,
            "attributable_to_the_diffusion_loop_s": diff * per_launch,
            "remainder_s": FIXED_S - diff * per_launch,
        },
        "what_this_predicts": (
            "If the clock-immune term is per-call host dispatch, a trace of the diffusion loop "
            "reaches about %.2f s of the %.3f s. Trace replay does not remove the device's program "
            "executions -- the chip still runs those %d programs -- it removes the host work per "
            "call, which is what the clock-immune term measures."
        ) % (diff * per_launch, FIXED_S, diff),
        "limits": [
            "The 1,096 comes from a run carrying graph capture, the identity observer and sum "
            "profiling. A program COUNT is far less instrument-sensitive than a time, but the "
            "observer could add programs of its own, so treat it as an upper-ish estimate.",
            "The program count and the call census come from different instruments on different "
            "trees. A device program is not exactly one launching ttnn call: a fused generic op is "
            "one call and one program, but some calls expand to several.",
            "Per-call cost is assumed uniform. It is not: a large matmul's host cost is the same "
            "as a deallocate's, so a loop made of small ops carries more than its share.",
            "This is a reach, not a win. What the lever actually removes is what c10-trace-lever "
            "measures.",
        ],
    }
    json.dump(out, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
