#!/usr/bin/env python3
"""What the shipped auto-trajectory default picks at every rung of the ladder, card-free.

    auto_census.py [--json out.json]

`fix544auto` and `fix704auto` were fitted to Blackhole memory, so the count `auto` picks on a
Wormhole Galaxy chip is an open question at every size, and half of it can be answered without a
chip: before a card is open `duotraj.auto_trajectories` prices the card from `card_total_bytes()`
and the host from `/proc`, both readable on an idle box. The other half -- the count with a card
open, where it reads `free_device_bytes()` -- is stamped per rung by `perf/bgx_size/rung.py`.

It also prints, per rung, the axis the default PRICES AT (`bindcraft2.design_tokens`, which is
BindCraft 2's own `design_residue_count`) beside the arithmetic label `_pad32(residues + binder)`.
Those two differ by one or two buckets on a multi-chain target, and pricing a trajectory a bucket
low is a default that errs high -- the regression `fix704auto` exists to prevent.

Target paths are made ABSOLUTE here. `read_settings` resolves a relative `target_path` against the
directory of the settings file, and `design_tokens` returns 0 for anything it cannot read, so a
relative path silently prices the design at 0 tokens and the census reads as "the axis could not
be read" on exactly the targets that are checked in rather than shipped with BindCraft 2.
"""
import argparse
import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "perf" / "bcx_round"))
sys.path.insert(0, str(ROOT / "perf" / "bcx_predictor"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bc2_state as B                                                   # noqa: E402
from tt_bio import bindcraft2, duotraj                                  # noqa: E402
from bindcraft.settings import parse_setting_overrides, read_settings    # noqa: E402
from bindcraft.preflight import cleaned_campaign_settings               # noqa: E402

TARGETS = json.loads((ROOT / "perf/bgx_size/targets.json").read_text())

#: The ladder, as (target, binder) pairs. The axis each one runs is measured at the Evoformer
#: seam by `perf/bgx_size/rung.py`, never taken from this list.
RUNGS = [("hPDL1", 50), ("hPDL1", 141), ("hIL7RA", 100), ("hCA2", 100), ("hCA2", 150),
         ("hIL2R", 50), ("hIL2R", 90), ("hIL2R", 100), ("hIL2R", 146), ("hTNFa", 100),
         ("hPCSK9", 100), ("hHSA", 100), ("hTF", 100)]


def settings_for(target: str, binder: int, tmp: pathlib.Path):
    spec = TARGETS[target]
    path = spec["path"]
    path = os.path.join(B.BC2, path[4:]) if path.startswith("BC2:") else str(ROOT / path)
    request = {"modality": "binder", "campaign_name": f"{target}_{binder}",
               "number_of_final_designs": 2,
               "targets": [{"name": target, "target_path": path, "chains": spec["chains"],
                            "hotspots": spec["hotspots"]}]}
    req = tmp / "settings_request.json"
    req.write_text(json.dumps(request, indent=1))
    return cleaned_campaign_settings(read_settings(req, parse_setting_overrides(
        ["campaign_seed=100", "max_trajectories=1", f"project_folder={tmp}",
         f"binder_lengths=[{binder}]", "compile_next_length=0"])))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", dest="out", default=None)
    args = ap.parse_args()

    tmp = pathlib.Path(os.environ.get("TMPDIR", "/tmp")) / "b2p_auto_census"
    tmp.mkdir(parents=True, exist_ok=True)
    card = duotraj.card_total_bytes()
    host = duotraj.free_host_bytes()
    print(f"host {os.uname().nodename}  card_total {card} B = {card / 2**30:.3f} GiB "
          f"= {card / 1e9:.3f} GB   host free {host / 2**30:.1f} GiB   "
          f"AUTO_CAP {duotraj.AUTO_CAP}", flush=True)
    rows = []
    for target, binder in RUNGS:
        spec = TARGETS[target]
        settings = settings_for(target, binder, tmp)
        priced = bindcraft2.design_tokens(settings)
        arithmetic = bindcraft2._pad32(spec["residues"] + binder)
        count, why = duotraj.auto_trajectories(priced)
        rows.append({"target": target, "binder": binder, "chains": spec["n_chains"],
                     "priced_axis": priced, "arithmetic_axis": arithmetic,
                     "auto_before_open": count, "why": why,
                     "trajectory_bytes": duotraj.trajectory_bytes(priced) if priced else None,
                     "trajectory_floor_bytes": (duotraj.trajectory_floor_bytes(priced)
                                                if priced else None)})
        print(f"{target:7s} +{binder:<4d} ch{spec['n_chains']}  priced_axis {priced:<4d} "
              f"(arithmetic {arithmetic:<4d})  auto {count}   {why[:96]}", flush=True)
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(
            {"host": os.uname().nodename, "card_total_bytes": card,
             "host_free_bytes": host, "auto_cap": duotraj.AUTO_CAP, "rungs": rows}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
