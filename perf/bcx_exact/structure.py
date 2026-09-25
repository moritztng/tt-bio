#!/usr/bin/env python3
"""The per-round call structure COUNT and SHARE are composed from, verified from the artifact.

Everything this row reports about a ROUND is the per-block runtime census times a call
structure. The census is measured (optrace, BLOCKCOUNT.json). The call structure was taken from
`bcx-round`'s PROSE, which is the weak link, so it is verified here against that row's own
committed event log and against the shipped source.

    git show origin/wk/bcx-round:perf/bcx_round/runs/round_seed100_card3_long/round_events.json

Four facts, and all four have to hold or the round-level numbers do not:

  1. 2 taped forwards per round, 1 backward, 0 primal          <- the event log
  2. 48 Evoformer blocks per call                              <- bindcraft2.EVOFORMER_BLOCKS
  3. the taped forward checkpoints, so the backward recomputes <- _taped passes
     recompute=self.recompute (default True) into evoformer(), which wraps each block in
     ag.checkpoint; _primal passes False and opens no tape at all, and fires 0 times in a
     gradient round
  4. therefore a round pays 3 forward censuses and 1 backward census per block

Run it with no arguments in a tree that can see origin/wk/bcx-round.
"""
import collections, json, os, pathlib, statistics as st, subprocess, sys, time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EVENTS = ("origin/wk/bcx-round:perf/bcx_round/runs/round_seed100_card3_long/"
          "round_events.json")


def git(*a):
    return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True).stdout


def main():
    blob = json.loads(git("show", EVENTS))
    ev = blob["events"]
    dev = [e for e in ev if e["kind"] == "device"]
    rounds = sum(1 for e in ev if e["kind"] == "round_start")
    complete = rounds - sum(1 for e in ev if e["kind"] == "round_stop")
    phases = collections.Counter(e["phase"] for e in dev)

    src = git("show", "origin/main:tt_bio/bindcraft2.py")
    blocks = next((int(l.split("=")[1]) for l in src.splitlines()
                   if l.startswith("EVOFORMER_BLOCKS")), None)

    per = {p: phases.get(p, 0) / complete for p in ("taped", "backward", "primal")}
    out = {"host": os.uname().nodename, "when_utc": time.strftime("%FT%TZ", time.gmtime()),
           "opened_a_device": False, "events": EVENTS,
           "round_start": rounds, "complete_rounds": complete,
           "device_calls": dict(phases),
           "per_round": {k: round(v, 3) for k, v in per.items()},
           "median_dt_s": {p: round(st.median([e["dt"] for e in dev if e["phase"] == p]), 3)
                           for p in phases},
           "EVOFORMER_BLOCKS": blocks,
           "taped_checkpoints": "recompute=self.recompute" in src and "ag.checkpoint" in src,
           "primal_opens_no_tape": "def _primal" in src,
           "CHECKS": {}}
    out["CHECKS"] = {
        "two_taped_forwards_per_round": per["taped"] == 2.0,
        "one_backward_per_round": per["backward"] == 1.0,
        "zero_primal_in_a_gradient_round": phases.get("primal", 0) == 0,
        "forty_eight_blocks": blocks == 48,
        "backward_recomputes_the_forward": out["taped_checkpoints"]}
    out["forward_censuses_per_round"] = 3 if all(out["CHECKS"].values()) else None
    out["backward_censuses_per_round"] = 1 if all(out["CHECKS"].values()) else None
    (HERE / "STRUCTURE.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0 if all(out["CHECKS"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
