#!/usr/bin/env python3
"""bcx-mutate: the rejection profile, read off arms that may still be running.

Which stage, which filter, how often -- the deliverable of this row. It reads BindCraft 2's own
`.campaign_state.json`, which `campaign_output.py` rewrites under a lock at every trajectory and
candidate outcome, so a partial arm is as readable as a finished one. Two multi-hour arms have
now ended with nothing behind them (a qb2 reboot at 09:06:34Z took the second pair), and this is
the artifact that makes the next one survivable.

THE DENOMINATOR IS NOT `trajectories`. `claim_trajectory` (campaign_output.py:169) increments it
before the trajectory runs and `record_trajectory_outcome` (:198) records the verdict after, so a
trajectory killed in flight is counted as an attempt and contributes nothing to `terminated`.
Both arms relaunched here carry exactly one such phantom from the reboot. `claim_recipe` (:177)
also burns its hash, so that binder is never redrawn and the attempt budget is permanently down
one. The rate this row reports therefore divides by sum(terminated.values()) -- trajectories that
actually reached a verdict -- and reports the phantom gap separately instead of hiding it.
"""
import argparse
import json
import pathlib
import sys

COMPLETED = "completed"


def read_state(project: pathlib.Path) -> dict | None:
    path = project / ".campaign_state.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def read_stamp(project: pathlib.Path) -> dict:
    path = project / "arm_stamp.json"
    return json.loads(path.read_text()) if path.exists() else {}


def profile(project: pathlib.Path) -> dict:
    state = read_state(project)
    if state is None:
        return {"project": str(project), "missing": True}
    rej = state.get("rejections", {})
    terminated = dict(rej.get("terminated", {}))
    verdicts = sum(terminated.values())
    attempted = len(state.get("attempted", []))
    claimed = int(state.get("trajectories", 0))
    stamp = read_stamp(project)
    return {
        "project": str(project),
        "seed": stamp.get("seed"),
        "arm": stamp.get("arm"),
        "started_utc": stamp.get("started_utc"),
        # The arm's own terminal status. A wrapper that echoes an exit code can echo the wrong
        # one; the arm writing its own is what makes "this trajectory completed" readable.
        "status": stamp.get("status"),
        "claimed": claimed,
        "attempted": attempted,
        "verdicts": verdicts,
        # claimed minus verdicts, minus the one trajectory that may be in flight right now.
        "unaccounted": claimed - verdicts,
        "accepted": int(state.get("accepted", 0)),
        "terminated": terminated,
        "failed_filters": dict(rej.get("failed_filters", {})),
        "candidates_scored": int(rej.get("candidates_scored", 0)),
        "candidates_rejected": int(rej.get("candidates_rejected", 0)),
    }


def merge(rows: list[dict]) -> dict:
    live = [r for r in rows if not r.get("missing")]
    terminated: dict[str, int] = {}
    filters: dict[str, int] = {}
    for r in live:
        for stage, n in r["terminated"].items():
            terminated[stage] = terminated.get(stage, 0) + n
        for name, n in r["failed_filters"].items():
            filters[name] = filters.get(name, 0) + n
    return {
        "arms": len(live),
        "claimed": sum(r["claimed"] for r in live),
        "verdicts": sum(r["verdicts"] for r in live),
        "unaccounted": sum(r["unaccounted"] for r in live),
        "accepted": sum(r["accepted"] for r in live),
        "terminated": terminated,
        "failed_filters": filters,
        "candidates_scored": sum(r["candidates_scored"] for r in live),
        "candidates_rejected": sum(r["candidates_rejected"] for r in live),
    }


def render(rows: list[dict], total: dict) -> str:
    out = []
    for r in rows:
        if r.get("missing"):
            out.append(f"{r['project']}: no .campaign_state.json")
            continue
        out.append(f"seed {r['seed']} | {r['project']}")
        out.append(f"  claimed {r['claimed']}  verdicts {r['verdicts']}  "
                   f"unaccounted {r['unaccounted']}  accepted {r['accepted']}")
        out.append(f"  terminated     {r['terminated'] or '{}'}")
        out.append(f"  failed_filters {r['failed_filters'] or '{}'}")
        out.append(f"  candidates     scored {r['candidates_scored']} "
                   f"rejected {r['candidates_rejected']}")
    n = total["verdicts"]
    out.append("")
    out.append(f"POOLED over {total['arms']} arm(s)")
    out.append(f"  trajectories with a verdict (the denominator) : {n}")
    out.append(f"  claimed but unaccounted (phantom + in flight) : {total['unaccounted']}")
    out.append(f"  accepted designs                              : {total['accepted']}")
    if n:
        for stage, c in sorted(total["terminated"].items(), key=lambda kv: -kv[1]):
            label = "reached the end" if stage == COMPLETED else f"died at {stage}"
            out.append(f"    {label:<28} {c:>3}/{n}  {100.0 * c / n:5.1f} %")
    else:
        out.append("    no trajectory has reached a verdict yet")
    if total["candidates_scored"]:
        out.append(f"  candidates scored {total['candidates_scored']}, "
                   f"rejected {total['candidates_rejected']}")
        for name, c in sorted(total["failed_filters"].items(), key=lambda kv: -kv[1]):
            out.append(f"    filter {name:<26} {c:>3}")
    else:
        out.append("  no candidate has been scored yet, so no filter has fired")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("projects", nargs="+", type=pathlib.Path)
    ap.add_argument("--json", action="store_true", help="machine-readable, for banking")
    args = ap.parse_args()
    rows = [profile(p) for p in args.projects]
    total = merge(rows)
    if args.json:
        print(json.dumps({"arms": rows, "pooled": total}, indent=1, sort_keys=True))
    else:
        print(render(rows, total))
    return 0


if __name__ == "__main__":
    sys.exit(main())
