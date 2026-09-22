#!/usr/bin/env python3
"""The break control for the re-key: does REMOVING it put the reading back?

`traj_retake_shipped.json` reads `tape_resolves_after_step` 26 of 26 at all twenty steps with
`repin` OFF, where `perf/of3t_modeltraj/traj_shipped.json` read 0 at all twenty. A guard that
only ever passes has tested nothing, so this asks the other direction: with the `_PARAMS` re-key
bypassed -- and nothing else changed -- does `parameter_for` go back to returning None?

The bypass is the pre-fix code path exactly. Before `965c24f52`, `AdamW.step` replaced the
handle and the registry kept the old key; writing `t._value` directly is that same program,
because the re-key lives in the `value` SETTER (`tt_bio/autograd.py:216-245`) and assigning the
slot underneath it is how you skip a property. No device, no model: this is the mechanism on its
own, which is why it can be a control rather than another trajectory.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.getcwd())

import numpy as np                                                       # noqa: E402
import torch                                                            # noqa: E402
import ttnn                                                             # noqa: E402

from tt_bio import autograd as ag                                       # noqa: E402
from tt_bio.tenstorrent import get_device                               # noqa: E402


def main() -> int:
    dev = get_device()
    mk = lambda: ttnn.from_torch(torch.randn(32, 32), layout=ttnn.TILE_LAYOUT, device=dev,
                                 dtype=ttnn.float32)

    rows = []

    # ARM 1: through the property, which is where the re-key lives. This is what `AdamW.step`
    # does at `tt_bio/train/optim.py:284`.
    raw0 = mk()
    leaf = ag.parameter(ag.Tensor(raw0))
    before = ag.parameter_for(leaf.value) is leaf
    new1 = mk()
    leaf.value = new1
    rows.append({"arm": "through the value setter (shipped since 965c24f52)",
                 "registered_before_replace": before,
                 "resolves_after_replace": ag.parameter_for(leaf.value) is leaf,
                 "stale_key_still_present": ag.parameter_for(raw0) is not None})

    # ARM 2: around the property, which is the pre-fix program.
    raw0b = mk()
    leaf2 = ag.parameter(ag.Tensor(raw0b))
    before2 = ag.parameter_for(leaf2.value) is leaf2
    new2 = mk()
    leaf2._value = new2                                    # the bypass, and the whole control
    rows.append({"arm": "around the setter (the pre-965c24f52 program)",
                 "registered_before_replace": before2,
                 "resolves_after_replace": ag.parameter_for(leaf2.value) is leaf2,
                 "stale_key_still_present": ag.parameter_for(raw0b) is not None})

    out = {
        "what_this_controls": "whether traj_retake_shipped.json's 26-of-26 is CAUSED by the "
                              "_PARAMS re-key in the value setter, or merely coincides with it",
        "rekey_site": "tt_bio/autograd.py value setter, _PARAMS[id(new)] = self",
        "arms": rows,
        "control_moves_the_reading": (rows[0]["resolves_after_replace"] is True
                                      and rows[1]["resolves_after_replace"] is False),
    }
    def git(*a):
        import subprocess
        return subprocess.run(("git",) + a, capture_output=True, text=True).stdout.strip()

    out["tree"] = {
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "commit": git("rev-parse", "HEAD"),
        "host": os.uname().nodename,
        "card": os.environ.get("TT_VISIBLE_DEVICES"),
        "dirty_paths": sorted(l.split(None, 1)[1]
                              for l in git("status", "--porcelain").splitlines()
                              if len(l.split(None, 1)) > 1),
    }
    p = "perf/of3t_trajretake/rekey_control.json"
    json.dump(out, open(p, "w"), indent=1)
    print(json.dumps(out, indent=1))
    print("wrote %s" % p)
    return 0 if out["control_moves_the_reading"] else 1


if __name__ == "__main__":
    sys.exit(main())
