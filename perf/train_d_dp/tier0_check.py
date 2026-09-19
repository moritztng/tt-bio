#!/usr/bin/env python3
"""Does `tt-bio finetune --chips 2` reach the launcher, or only `train.finetune`?

The README's claim is about a command line, so Tier 0 has to be exercised as one rather than
argued from the fact that Tier 1 works. What blocks that today is not the launcher: the
catalogue ships empty on purpose, because featurisation is per model and no model has registered
one yet, so `--model protenix-v2` refuses with the name of what is missing before a device
opens.

So this registers a STAND-IN featuriser under that name -- the trunk from
``perf/train_d_dp/model.py``, which is real work through the shipped ops and is not the Protenix
featuriser and is not presented as one -- and then runs the Tier-0 command. Everything between
the flag and the chips is the shipped code: `cli.finetune` validates the flags, builds the mesh
from `--chips`, calls Tier 1, which calls the recipe, which hands the wide axis to the launcher.

This file is also the program the launcher re-runs. Each rank re-executes it, registers the same
stand-in, and reaches the same command, which is exactly the contract the launcher asks of any
caller.

    python3 perf/train_d_dp/tier0_check.py --chips 2,0
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from perf.train_d_dp import model as M          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chips", default="2,0", help="passed straight to the CLI --chips")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--channels", type=int, default=1024)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--out", default=str(HERE / "out" / "tier0"))
    a = ap.parse_args()

    nchips = (len([c for c in a.chips.split(",") if c]) if "," in a.chips
              else int(a.chips))

    from tt_bio.train import catalogue, launcher

    catalogue.register("protenix-v2", lambda path, tokens=None: M.build(
        tokens=a.tokens, channels=a.channels, blocks=a.blocks, examples=16, seed=0))

    data = Path(a.out) / "data"
    data.mkdir(parents=True, exist_ok=True)

    from tt_bio.train.cli import finetune

    argv = ["--model", "protenix-v2", "--out", str(Path(a.out) / "run"),
            "--global-batch", str(nchips), "--steps", str(a.steps),
            "--chips", a.chips, "--rank", "8", "--warmup-steps", "10",
            "--checkpoint-every", str(a.steps), str(data)]
    print(f"tt-bio finetune {' '.join(argv)}", flush=True)
    try:
        finetune.main(argv, standalone_mode=False)
    except SystemExit as exc:
        return exc.code or 0
    if launcher.inside():
        return 0
    # The CLI wrote its own run.json; what this file adds is the Tier-0 verdict beside it.
    rj = json.loads((Path(a.out) / "run" / "run.json").read_text())
    dp = (rj.get("provenance") or {}).get("dp") or {}
    verdict = {
        "entry": "tier 0, tt-bio finetune --chips",
        "chips": a.chips, "world": dp.get("world"), "nodes": dp.get("nodes"),
        "distinct_master_sha": dp.get("distinct_master_sha"),
        "median_step_s": dp.get("median_step_s"),
        "reached_launcher": bool(dp),
        "featuriser": "STAND-IN: perf/train_d_dp/model.py registered as protenix-v2, because "
                      "the shipped catalogue is empty by design",
    }
    out = Path(a.out) / "tier0.json"
    out.write_text(json.dumps(verdict, indent=2) + "\n")
    print(json.dumps(verdict, indent=2), flush=True)
    assert verdict["reached_launcher"], "Tier 0 did not reach the launcher"
    assert verdict["world"] == nchips, f"world {verdict['world']} != --chips {a.chips}"
    assert len(set(verdict["nodes"])) == nchips, f"ranks shared a chip: {verdict['nodes']}"
    assert verdict["distinct_master_sha"] == 1, "the ranks diverged"
    print(f"\nTIER 0 OK: --chips {a.chips} ran {verdict['world']} ranks on nodes "
          f"{verdict['nodes']}, {verdict['distinct_master_sha']} distinct master sha", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
