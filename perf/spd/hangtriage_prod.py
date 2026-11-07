#!/usr/bin/env python3
"""Run on a JapanFold box: tt-triage a chip whose worker went quiet in a job, before the agent resets it.

The agent marks a worker dirty QUIET_S (300 s) after its last heartbeat, then resets the chip. A worker
that has gone 60 s quiet in a job is already hung (it beats every 10 s), so we read the chip in between.
Read-only: never stops a worker, never resets a chip. Uses spd-fasthang's triage kit (~/spd/spd-fasthang).
Usage: hangtriage.py OUTDIR MAX_TRIAGES
"""
import json, os, subprocess, sys, time, urllib.request
from pathlib import Path

out, cap = Path(sys.argv[1]), int(sys.argv[2])
kit = Path.home() / "spd/spd-fasthang"
QUIET = 60
done = {}
out.mkdir(parents=True, exist_ok=True)
log = open(out / "watch.log", "a", buffering=1)
n = 0
while n < cap:
    try:
        snap = json.load(urllib.request.urlopen("http://127.0.0.1:8767/cluster", timeout=5))
    except Exception as e:
        log.write(f"{time.time():.0f} cluster unreadable {e}\n"); time.sleep(10); continue
    now = snap.get("now") or time.time()
    for w in snap.get("workers") or []:
        chip, quiet = str(w.get("device_id")), now - float(w.get("last_seen") or now)
        if not w.get("running") or quiet < QUIET or time.time() - done.get(chip, 0) < 1800:
            continue
        done[chip] = time.time()
        n += 1
        d = out / f"{time.strftime('%m%dT%H%M%SZ', time.gmtime())}-chip{chip}"
        d.mkdir()
        (d / "worker.json").write_text(json.dumps(w, indent=1))
        log.write(f"{time.time():.0f} chip {chip} quiet {quiet:.0f}s model {w.get('model')} running {w.get('running')}\n")
        pid = subprocess.run(["pgrep", "-f", f"chipworker serve .*--chip {chip} "], capture_output=True, text=True).stdout.split()
        for p in pid[:1]:
            subprocess.run(f"timeout 30 sudo -n {kit}/xv/bin/py-spy dump --native --pid {p} > {d}/pyspy.txt 2>&1 || timeout 30 {kit}/xv/bin/py-spy dump --pid {p} >> {d}/pyspy.txt 2>&1", shell=True)
        cache = Path.home() / "japanfold/cache/tt-metal"
        # Prod runs TT_METAL_INSPECTOR=0 (triage-no-inspector.patch), so its firmware is built without -g and
        # dispatcher_data cannot decode mailboxes from it. kit/fwdbg is the same tt-metal's firmware built with -g.
        fw = kit / "fwdbg" if (kit / "fwdbg").is_dir() else next(cache.glob("tt-metal-cache*/firmware"), "")
        env = dict(os.environ, TT_VISIBLE_DEVICES=chip, TT_METAL_LOGS_PATH=str(d / "tmlogs"),
                   TRIAGE_NO_INSPECTOR_FW=str(fw), TRIAGE_KERNEL_CACHE=str(cache), TRIAGE_KERNEL_INDEX=str(kit),
                   PYTHONUNBUFFERED="1", TRIAGE_DUMP_DIR=str(d))
        t0 = time.time()
        with open(d / "triage.txt", "w") as f:
            r = subprocess.run(["timeout", "200", str(kit / "xv/bin/python"), "tools/tt-triage.py", "-vv", "--verbosity=5",
                                "--print-script-times", "--disable-colors", "--disable-progress"], cwd=kit / "tm", env=env, stdout=f, stderr=subprocess.STDOUT)
        log.write(f"{time.time():.0f} chip {chip} triage rc={r.returncode} {time.time() - t0:.0f}s -> {d}\n")
    time.sleep(10)
log.write(f"{time.time():.0f} cap {cap} reached\n")
