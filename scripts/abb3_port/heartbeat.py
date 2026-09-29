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
* **will it get anywhere** -- the recent step rate projected onto the remaining grant;
* **is it on the recipe's learning rate** -- every logged lr re-evaluated against the cosine
  the run recorded in ``schedule.json``. Added 2026-09-19 after the run was found holding a
  constant 5e-4 because nothing wrapped the optimizer's lr: the history carried no lr at all,
  so two readers inferred it from two different files and neither read it out of the run.

Exit status is the verdict: 0 healthy or finished, 1 an incident that wants a human, 2 THIS
SCRIPT failed and is saying nothing about the run. The third code exists because it did fail --
on 2026-09-19 it died at import on `ModuleNotFoundError: numpy` under the system interpreter and
exited 1, the code the runbook reserves for *act*. Two causes sharing one code is how a reader
learns to ignore the code. Every repo import is therefore inside a function, so the wrapper at
the bottom of this file can catch it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

#: The longest gap between two logged steps that is not yet an incident. A pair step is ~11-12 s,
#: and the longest legitimate gap is a restart: the supervisor's 20 s settle, then a device open
#: and a model build. 15 minutes is roughly an order of magnitude over that, so it does not fire
#: on a healthy restart and still catches a death inside one checkpoint cadence.
STALL_MINUTES = 15.0

#: How far a logged lr may sit from the cosine before it is an incident. The comparison is
#: between two evaluations of the same class on the same integer, so anything above float noise
#: means the run is not on the schedule it recorded.
LR_TOL = 1e-9

#: Trailing rows the lr is re-checked on. The schedule is a pure function of the step, so a
#: window is as strong as the whole file and stays cheap on a five-day history.
LR_WINDOW = 500


def _rows(path: Path) -> list:
    """The run's own history, with the lines a crash tore recovered rather than dropped.

    One parser, shared with the curve writer, because dropping a NUL-torn line is not a lost
    step: it is a step counter that appears to jump, which is what latched this instrument at
    INCIDENT on a healthy run. See ``tt_bio.train.history``.
    """
    from tt_bio.train.history import read_rows
    return read_rows(path)


def resumes(rows: list, checkpoints: dict, declared: list = ()) -> list:
    """Every point where the step counter went backwards, and what kind of restart it was.

    ``checkpoints`` maps step -> path. The predicate is **did it land past a checkpoint we
    hold**, not *exactly one past one*. A resume replays from ``checkpoint + 1``, but the first
    row a reader sees can be later: the 2026-09-19 host reset resumed from checkpoint 683 and
    the first row this compared was 685, because the torn step-684 line was being dropped. The
    old predicate called that ``RESTARTED-FROM-SCRATCH`` and the record is append-only, so a
    healthy run carried that verdict permanently.

    ``RESTARTED-FROM-SCRATCH`` now means what its name says: the counter back at 1 while
    checkpoints existed. That is the silent failure worth an incident, a perfectly healthy loss
    curve for a model that threw away three days.

    Deliberately *not* "did it land past the LATEST checkpoint". Checkpoints written after the
    death are on disk by the time anyone reads, so the latest is not the one it resumed from,
    and a tighter predicate would invent alarms the way the old one did.

    And the checkpoint set is not the primary evidence at all, because pruning destroys it. The
    three resumes on the base-loss leg came off checkpoints 551 and 683, both deleted once
    ``keep_checkpoints`` had three newer ones, and this called all three
    ``UNEXPLAINED-JUMP-BACK`` on a run whose replayed rows were bit-identical. What a resume
    leaves behind permanently is those rows: same loss, same master digest, because it is the
    same arithmetic on restored state. :func:`~tt_bio.train.history.replay_agreement` reads
    them, and the checkpoints are the fallback for a restart that replayed nothing this reader
    can see.

    ``declared`` is :func:`declared_restarts`: restarts somebody took on purpose and wrote down
    beforehand. A deliberate restart that changes the optimizer -- the lr-schedule repair at
    step 552 -- is a real jump-back with a digest that really does differ, so no evidence in the
    run can explain it and it would latch this instrument at INCIDENT for the rest of the leg.
    A declaration is matched on the exact ``(died_at, resumed_at)`` pair, so it excuses the one
    restart it names and nothing else.
    """
    from tt_bio.train.history import replay_agreement

    found = []
    for i in range(1, len(rows)):
        before, after = rows[i - 1]["step"], rows[i]["step"]
        if after > before:
            continue
        prior = [s for s in checkpoints if s <= before]
        replay = replay_agreement(rows, i)
        if after == 1:
            verdict = "RESTARTED-FROM-SCRATCH" if prior else "RESTARTED-NO-CHECKPOINT"
        elif replay["agrees"] or any(s < after for s in prior):
            verdict = "RESUMED"
        else:
            verdict = "UNEXPLAINED-JUMP-BACK"
        why = next((d.get("why") for d in declared
                     if d.get("died_at") == before and d.get("resumed_at") == after), None)
        if verdict == "UNEXPLAINED-JUMP-BACK" and why:
            verdict = "DECLARED-RESTART"
        found.append({"died_at": before, "resumed_at": after, "lost_steps": before - after + 1,
                      "from_checkpoint": verdict == "RESUMED", "verdict": verdict,
                      "replay": replay, "why": why})
    return found


#: Restart verdicts that want a human. ``RESTARTED-NO-CHECKPOINT`` is not one: the run went back
#: to 1 with nothing on disk to resume from, so nothing recoverable was thrown away. Neither is
#: ``DECLARED-RESTART``, which is one somebody wrote down before taking it.
BAD_RESTARTS = ("RESTARTED-FROM-SCRATCH", "UNEXPLAINED-JUMP-BACK")

#: Deliberate restarts, declared in the run directory rather than argued for in a reply.
DECLARED = "known-restarts.json"


def declared_restarts(out: Path) -> list:
    """Restarts taken on purpose, each naming the exact step pair it excuses and why.

    Written when the restart is taken, not when the alarm goes off, which is the whole point:
    a declaration added after the fact to quiet an instrument is indistinguishable from the
    defect it is quieting. Absent file, empty list, and every jump-back stays unexplained.
    """
    path = Path(out) / DECLARED
    if not path.is_file():
        return []
    try:
        loaded = json.loads(path.read_text())
    except json.JSONDecodeError:
        return []
    return loaded if isinstance(loaded, list) else []


def lr_check(out: Path, rows: list) -> dict:
    """Does the lr the run logged match the cosine at that global step?

    Evaluated with the run's own scheduler class, rebuilt from the ``schedule.json`` the run
    wrote, rather than with a second transcription of the formula here. A second transcription
    is exactly how the schedule went missing: the step file's recipe carried lr and weight decay
    and dropped ``T_0``, ``eta_min`` and ``T_mult``, and every reader believed a different file.
    """
    spec_path = out / "schedule.json"
    last = rows[-1] if rows else {}
    if not spec_path.is_file():
        return {"checked": 0, "ok": False, "lr": last.get("lr"),
                "note": f"{spec_path} is missing: the run did not record its schedule"}
    if last.get("lr") is None:
        return {"checked": 0, "ok": False, "lr": None,
                "note": f"step {last.get('step')} logged no lr, so nothing can check it"}
    from tt_bio.train.abb3_run import CosineRestartsByStep
    sched = CosineRestartsByStep.load(json.loads(spec_path.read_text()))
    worst, worst_step, checked = 0.0, None, 0
    for r in rows[-LR_WINDOW:]:
        if r.get("lr") is None:
            continue
        checked += 1
        d = abs(sched.set_step(int(r["step"])) - float(r["lr"]))
        if d > worst:
            worst, worst_step = d, int(r["step"])
    ok = worst <= LR_TOL
    return {"checked": checked, "ok": ok, "lr": float(last["lr"]), "worst_abs": worst,
            "worst_step": worst_step,
            "note": None if ok else (f"step {worst_step} logged an lr {worst:.3e} away from "
                                     f"the schedule in {spec_path.name}")}


def supervisor_restarts(out: Path) -> int:
    """How many times the world was relaunched, from the supervisor's own log.

    Every ``[sup] launching`` line is one launch of the world and the first one is not a
    restart, so the count is the lines minus one. Counting only the lines the supervisor
    numbered ``(restart N>0)`` missed a whole class: a host reset kills the supervisor too, and
    the process that replaces it logs ``(restart 0)``. qb2 did that twice inside 25 minutes on
    2026-09-19 and this counter read zero.

    The history's backward jumps miss the other case: a rank that died on the very step a
    checkpoint was written resumes at that step plus one and leaves a monotonic history. So the
    log is the count and the jumps are the diagnosis -- the jumps say whether each restart
    RESUMED or threw the run away, which is the thing no counter can tell you.
    """
    log = out / "logs" / "supervisor.log"
    if not log.is_file():
        return 0
    launches = sum(1 for line in log.read_text(errors="replace").splitlines()
                   if "[sup] launching" in line)
    return max(launches - 1, 0)


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
    from tt_bio.train import deadline as deadline_mod
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

    state["lr"] = lr_check(out, rows0)
    state["grad_norm"] = rows0[-1].get("grad_norm")

    res = resumes(rows0, checkpoints, declared_restarts(out))
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
    elif [r for r in res if r["verdict"] in BAD_RESTARTS]:
        bad = [r for r in res if r["verdict"] in BAD_RESTARTS][0]
        verdict, incident = "INCIDENT", (
            f"a restart did not resume: the run died at step {bad['died_at']} and came back at "
            f"step {bad['resumed_at']}, replaying {bad['replay']['compared']} rows of which "
            f"{len(bad['replay']['disagree'])} disagree, and past no checkpoint still on disk")
    elif agree is False:
        verdict, incident = "INCIDENT", "the ranks' master digests disagree"
    elif since is not None and since > args.stall_minutes * 60.0:
        verdict, incident = "INCIDENT", (
            f"no step logged for {since / 60.0:.1f} min "
            f"(supervisor {'alive' if sup_alive else 'GONE'})")
    elif not sup_alive:
        verdict, incident = "INCIDENT", "the supervisor is gone, so nothing will restart the ranks"
    elif state["lr"]["ok"] is False:
        verdict, incident = "INCIDENT", state["lr"]["note"]
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
        lr = state["lr"]
        print(f"  lr {lr['lr']}, grad norm {state['grad_norm']}"
              f" (schedule {'OK' if lr['ok'] else 'MISMATCH'} on {lr['checked']} rows)")
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


def _cli() -> int:
    """``main`` plus the one rule the runbook rests on: the instrument's own death is not a verdict.

    Anything this script fails at -- a missing dependency, a repo import, an unreadable run
    directory -- exits 2, so 1 keeps meaning *the run needs a human*.
    """
    try:
        return main()
    except SystemExit as e:  # argparse usage errors already mean "the instrument was misused"
        return 2 if e.code else 0
    except BaseException:
        traceback.print_exc()
        print("HEARTBEAT: BROKEN  the heartbeat itself failed and says nothing about the run",
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(_cli())
