#!/usr/bin/env python3
"""Is the 5-day reproduction alive, and did it actually resume? One read, one verdict.

    PYTHONPATH=$PWD python3 scripts/abb3_port/heartbeat.py --out runs/base

``supervise.py`` restarts the ranks after every watchdog reset, and `train-b3-train` proved the
resume is bit-identical. Neither of those checks that it *happened*. The failure this exists to
catch is a 5-day run that died on day 1 and was noticed at the cap, and the cap being 5 days
rather than 27 makes it worse and not better: a lost day is 20 % of the grant, and there is
less time for anyone to trip over it.

So this is a reader over the artefacts the run already writes, rather than a second write path
the run has to remember to feed. It answers four questions:

* **is it moving** -- the last logged step and the wall clock since it, against a stall bound;
* **did it resume, or did it restart from scratch** -- ``history-rank*.jsonl`` is appended
  across restarts, so a resume shows as the step counter dropping back to the checkpoint and a
  silently un-resumed run shows as it dropping back to 1. That is the one B2/H/J failure nobody
  was checking for, and it is the failure that shows a perfectly healthy loss curve for a model
  nobody asked for;
* **do the ranks agree** -- the master-weight digest at the last step both ranks reached. The
  run's own ``check_equal`` already fails on a mismatch, so a disagreement here means the run
  died before it could; the value is that a *dead* run still shows you where it diverged;
* **will it get anywhere** -- the recent step rate projected onto the remaining grant.

Exit status is the verdict: 0 healthy or finished, 1 an incident that wants a human. That makes
it usable from a relaunch check and from cron without parsing the text.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tt_bio.train import deadline as deadline_mod  # noqa: E402

#: The longest gap between two logged steps that is not yet an incident. A pair step is ~11-12 s,
#: and the longest legitimate gap is a restart: the supervisor's 20 s settle, then a device open
#: and a model build. 15 minutes is roughly an order of magnitude over that, so it does not fire
#: on a healthy restart and still catches a death inside one checkpoint cadence.
STALL_MINUTES = 15.0


def _rows(path: Path) -> list:
    out = []
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            # The last line of a file whose process was SIGKILLed mid-write. Dropping it is
            # right: it is one step, and refusing to read the history because the run died the
            # way it was expected to die would break the tool exactly when it is needed.
            pass
    return out


def resumes(rows: list, checkpoints: dict) -> list:
    """Every point where the step counter went backwards, and whether it resumed or restarted.

    ``checkpoints`` maps step -> path. A resume that lands on ``checkpoint + 1`` restored the
    run. A resume that lands on 1 while checkpoints existed threw the run away and started
    again, which is the silent failure.
    """
    found = []
    for i in range(1, len(rows)):
        before, after = rows[i - 1]["step"], rows[i]["step"]
        if after > before:
            continue
        prior = [s for s in checkpoints if s < before or s == before]
        from_ckpt = (after - 1) in checkpoints or (after == 1 and not prior)
        found.append({"died_at": before, "resumed_at": after, "lost_steps": before - after + 1,
                      "from_checkpoint": bool(from_ckpt),
                      "verdict": "RESUMED" if from_ckpt else "RESTARTED-FROM-SCRATCH"})
    return found


def supervisor_restarts(out: Path) -> int:
    """How many times the supervisor relaunched the world, from its own log.

    The history's backward jumps miss one case: a rank that died on the very step a checkpoint
    was written resumes at that step plus one and leaves a perfectly monotonic history. So the
    log is the count and the jumps are the diagnosis -- the jumps say whether each restart
    RESUMED or threw the run away, which is the thing no counter can tell you.
    """
    log = out / "logs" / "supervisor.log"
    if not log.is_file():
        return 0
    return sum(1 for line in log.read_text().splitlines()
               if "[sup] launching" in line and "(restart 0)" not in line)


def rate(rows: list, window_s: float) -> tuple:
    """Steps per second over the trailing window, from the rows' own epoch stamps.

    Wall clock and not the sum of the rows' ``wall`` field: the gap a restart leaves is exactly
    what the projection has to pay for, and summing per-step device time hides it.
    """
    stamped = [r for r in rows if r.get("t")]
    if len(stamped) < 2:
        return 0.0, 0
    cut = stamped[-1]["t"] - window_s
    win = [r for r in stamped if r["t"] >= cut]
    if len(win) < 2:
        win = stamped[-2:]
    span = win[-1]["t"] - win[0]["t"]
    return ((len(win) - 1) / span if span > 0 else 0.0), len(win)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--world", type=int, default=2)
    ap.add_argument("--stall-minutes", type=float, default=STALL_MINUTES)
    ap.add_argument("--window-minutes", type=float, default=60.0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    now = time.time()

    ckpt_dir = out / "checkpoints"
    checkpoints = {int(p.stem.split("-")[-1]): p
                   for p in sorted(ckpt_dir.glob("step-*.safetensors"))} if ckpt_dir.is_dir() \
        else {}
    per_rank = {r: _rows(out / f"history-rank{r}.jsonl") for r in range(args.world)}
    live = [r for r, rows in per_rank.items() if rows]
    rec = deadline_mod.read(out)
    sup = out / "supervisor.pid"
    sup_pid = sup.read_text().strip() if sup.is_file() else ""
    sup_alive = bool(sup_pid) and Path(f"/proc/{sup_pid}").exists()
    complete = sorted(p.name for p in out.glob("COMPLETE-rank*"))

    state = {
        "out": str(out), "now": now,
        "supervisor": {"pid": sup_pid or None, "alive": sup_alive},
        "deadline": rec,
        "remaining_hours": round(deadline_mod.remaining(out, now=now) / 3600.0, 2) if rec
        else None,
        "complete": complete,
        "checkpoints": {"count": len(checkpoints),
                        "latest_step": max(checkpoints) if checkpoints else None},
    }

    if not live:
        state["verdict"] = "NOT-STARTED"
        state["incident"] = None
        _emit(state, args.json)
        return 0

    rows0 = per_rank[live[0]]
    last_step = max(rows[-1]["step"] for rows in per_rank.values() if rows)
    last_t = max((rows[-1].get("t") or 0.0) for rows in per_rank.values() if rows)
    since = (now - last_t) if last_t else None
    sps, n = rate(rows0, args.window_minutes * 60.0)
    state["last_step"] = last_step
    state["seconds_since_last_step"] = round(since, 1) if since is not None else None
    state["step_seconds"] = round(1.0 / sps, 3) if sps else None
    state["rate_samples"] = n
    state["digest"] = rows0[-1].get("digest")

    # Do the ranks agree at the last step both of them reached? A mismatch means the run died
    # before its own check_equal could fail it.
    agree = None
    if len(live) > 1:
        common = min(rows[-1]["step"] for rows in per_rank.values())
        seen = {}
        for r in live:
            hit = [x for x in per_rank[r] if x["step"] == common]
            if hit:
                seen[r] = hit[-1].get("digest")
        agree = (len(set(seen.values())) == 1) if len(seen) == len(live) else None
        state["digest_agree"] = {"step": common, "per_rank": seen, "agree": agree}

    res = resumes(rows0, checkpoints)
    state["resumes"] = res
    state["restarts_logged"] = supervisor_restarts(out)
    state["resumes_survived"] = max(len(res), state["restarts_logged"])

    projected = None
    if rec and sps:
        projected = int(last_step + sps * deadline_mod.remaining(out, now=now))
    state["projected_step_at_cap"] = projected

    incident = None
    if complete and len(complete) >= len(live):
        verdict = "COMPLETE"
    elif rec and deadline_mod.remaining(out, now=now) <= 0:
        verdict = "CAP-REACHED"
    elif any(r["verdict"] == "RESTARTED-FROM-SCRATCH" for r in res):
        verdict, incident = "INCIDENT", "a restart did not resume: the run began again at step 1"
    elif agree is False:
        verdict, incident = "INCIDENT", "the ranks' master digests disagree"
    elif since is not None and since > args.stall_minutes * 60.0:
        verdict, incident = "INCIDENT", (
            f"no step logged for {since / 60.0:.1f} min "
            f"(supervisor {'alive' if sup_alive else 'GONE'})")
    elif not sup_alive:
        verdict, incident = "INCIDENT", "the supervisor is gone, so nothing will restart the ranks"
    else:
        verdict = "LIVE"
    state["verdict"] = verdict
    state["incident"] = incident
    _emit(state, args.json)
    return 1 if verdict == "INCIDENT" else 0


def _emit(state: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(state, indent=2))
        return
    v = state["verdict"]
    print(f"HEARTBEAT: {v}  {state['out']}")
    if state.get("incident"):
        print(f"  INCIDENT: {state['incident']}")
    if state.get("last_step") is not None:
        print(f"  step {state['last_step']}, {state['seconds_since_last_step']}s ago, "
              f"{state['step_seconds']}s/step over {state['rate_samples']} samples")
        print(f"  master digest {(state.get('digest') or '')[:16]}"
              + (f", ranks agree: {state['digest_agree']['agree']}"
                 if state.get("digest_agree") else ""))
        print(f"  resumes survived {state['resumes_survived']} "
              f"({state['restarts_logged']} in the supervisor log)"
              + "".join(f"\n    {r['verdict']} died@{r['died_at']} -> {r['resumed_at']} "
                        f"(lost {r['lost_steps']})" for r in state["resumes"]))
    print(f"  checkpoints {state['checkpoints']['count']}, "
          f"latest step {state['checkpoints']['latest_step']}")
    if state.get("remaining_hours") is not None:
        print(f"  grant {state['remaining_hours']} h left, "
              f"projected step at cap {state.get('projected_step_at_cap')}")
    print(f"  supervisor pid {state['supervisor']['pid']} "
          f"alive={state['supervisor']['alive']}")


if __name__ == "__main__":
    raise SystemExit(main())
