#!/usr/bin/env python3
"""Sample a running BindCraft 2 campaign's own loop counter, without touching the campaign.

BindCraft 2 prints one line per design STAGE, so a trajectory that is three hours into a
fifty-step screen looks identical to one that is wedged. `py-spy dump --locals` reads the
live frame instead: `run_gradient_design_stage`'s `sequence_updates` is the gradient step
the trajectory is on and `design_loss` is the loss it just measured.

The loss matters as much as the counter. `trajectory.py:136` BREAKS out of the stage on a
non-finite loss instead of raising, so a NaN ends the stage early and the campaign still
prints a clean stage verdict underneath it. Sampling the loss is the only way to tell a
stage that finished from a stage that was ended.

One sample every `--interval` seconds; py-spy stops the process for milliseconds, which is
below the noise on a multi-hour trajectory. Output is JSONL, one object per sample.
"""
import argparse
import json
import re
import subprocess
import time

FRAME = re.compile(r"^\s{4}(\S+) \((\S+:\d+)\)")
LOCAL = re.compile(r"^\s+(\w+): (.*)$")
WANT = ("sequence_updates", "design_loss", "best_design_loss", "first_sequence_update")


def sample(pyspy, pid):
    out = subprocess.run([pyspy, "dump", "--pid", str(pid), "--locals"],
                         capture_output=True, text=True, timeout=180)
    if out.returncode != 0:
        return {"error": out.stderr.strip()[:300]}
    stack, locals_, in_frame = [], {}, False
    for line in out.stdout.splitlines():
        frame = FRAME.match(line)
        if frame:
            stack.append(frame.group(2))
            in_frame = frame.group(1) == "run_gradient_design_stage"
            continue
        if in_frame:
            local = LOCAL.match(line)
            if local and local.group(1) in WANT:
                locals_[local.group(1)] = local.group(2)
    return {"top": stack[0] if stack else None,
            "stack": [f for f in stack if f.startswith("bindcraft/")], **locals_}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=120.0)
    ap.add_argument("--pyspy", default="/home/moritz/bcx_tail/venv/bin/py-spy")
    args = ap.parse_args()
    while True:
        try:
            row = sample(args.pyspy, args.pid)
        except Exception as exc:                                   # the watcher never kills the run
            row = {"error": f"{type(exc).__name__}: {exc}"[:300]}
        row["utc"] = time.strftime("%FT%TZ", time.gmtime())
        with open(args.out, "a") as f:
            f.write(json.dumps(row) + "\n")
        if row.get("error", "").startswith("Error: No such file") or "not found" in row.get("error", ""):
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
