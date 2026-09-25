#!/usr/bin/env python3
"""Stop a detached BindCraft 2 campaign at its next trajectory boundary, without a signal.

SIGINT does not reach this arm. `/proc/<pid>/status` on the live run reports
`SigIgn: 0000000001001007` -- bits 1, 2 and 3, so SIGHUP, **SIGINT** and SIGQUIT are all
SIG_IGN. That is POSIX background-job disposition inherited from the non-interactive shell
`launch_vhh_device.sh` backgrounded it from, not anything BindCraft 2 or tt-bio installs: the
same status shows `SigCgt` carrying only SIGABRT/SIGBUS/SIGFPE/SIGSEGV and glibc's thread
signal. So two SIGINTs were discarded by the kernel before Python ever saw them, and no number
of further ones would land. SIGTERM is neither caught nor ignored, so it would kill the process
outright -- which is exactly the thing we must not do to the holder of a Tenstorrent device: the
card is left unopenable and the NEXT open hard-hangs the host, and the remedy on a p300c is
`tt-smi -r`, which resets the board PAIR and would take another campaign's card with it.

So the stop goes through the campaign's own budget ledger instead. `campaign.py:152`'s loop
calls `CampaignProgress.claim_trajectory()`, which re-reads `.campaign_state.json` under a
`flock` on the project directory every single iteration (`campaign_output.py:169-176`) and
returns None once `trajectories >= max_trajectories`. Writing the budget as spent therefore
makes the loop `break` at the next boundary and run its normal ending: `write_campaign_summary`,
`rank_accepted_designs`, the campaign-closed banner, then a clean interpreter exit that closes
the device the way every other run closes it.

The honesty cost, stated because it is a real one: afterwards the ledger reads
`trajectories: 10` on a run where 3 trajectories were designed. That counter is a budget claim,
not a record of work -- `recovered_state()` (`campaign_output.py:158`) recomputes it from
`1_Trajectories/!_Trajectories.csv`, which keeps the truth at one row per trajectory actually
run. This script preserves the pre-stop ledger to `.campaign_state.pre_stop.json` and drops a
`STOP.md` beside it, and the row's count is reported from the CSV, never from this file.
"""
import argparse
import fcntl
import json
import os
import pathlib
import time


def stop(project: str, budget: int, reason: str) -> dict:
    state_path = os.path.join(project, ".campaign_state.json")
    folder = os.open(project, os.O_RDONLY)
    try:
        fcntl.flock(folder, fcntl.LOCK_EX)  # the same lock campaign_output.py takes
        before = json.loads(pathlib.Path(state_path).read_text())
        pathlib.Path(project, ".campaign_state.pre_stop.json").write_text(
            json.dumps(before, sort_keys=True, indent=1))
        after = {**before, "trajectories": max(int(before["trajectories"]), budget)}
        partial = f"{state_path}.partial"
        pathlib.Path(partial).write_text(json.dumps(after, sort_keys=True))
        os.replace(partial, state_path)
    finally:
        os.close(folder)
    pathlib.Path(project, "STOP.md").write_text(
        f"# Stopped at a trajectory boundary, {time.strftime('%FT%TZ', time.gmtime())}\n\n"
        f"{reason}\n\n"
        f"`.campaign_state.json` now reads `trajectories: {after['trajectories']}` so that\n"
        f"`CampaignProgress.claim_trajectory()` returns None and `campaign.py:152`'s loop breaks\n"
        f"at the next boundary. **{before['trajectories']} trajectories were actually designed.**\n"
        f"The true ledger at the moment of the stop is `.campaign_state.pre_stop.json`, and the\n"
        f"record of what ran is `1_Trajectories/!_Trajectories.csv`, one row per trajectory.\n"
        f"No count in this campaign is reported from `.campaign_state.json`.\n")
    return {"before": before, "after": after}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("project")
    ap.add_argument("--budget", type=int, required=True,
                    help="max_trajectories as the campaign resolved it (stage_bars.py reports it)")
    ap.add_argument("--reason", required=True)
    args = ap.parse_args()
    print(json.dumps(stop(args.project, args.budget, args.reason), indent=1, sort_keys=True))
