"""Is the booth demo healthy? One answer, in words anyone at the booth can read.

    ops/sc26ctl health          (or: python3 demo/sc26/ops/health.py)

The first line is HEALTHY, REPAIRING ITSELF or NOT HEALTHY, then one line per thing it checked.
Exit status 0 when healthy or repairing within its usual time, 1 otherwise. Reads only what the
demo already publishes: the engine's /status and the watchdog's log. It changes nothing.
"""
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

LOG = Path(os.path.expanduser("~/sc26-logs/watchdog.jsonl"))
REPAIR_S = 15 * 60   # a board reset plus warm-up is ~9 min measured; past this a chip is not repairing, it is stuck


def tail(path, n=600):
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 400 * n))
            lines = f.read().splitlines()[1:]
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def ago(s):
    return f"{s:.0f} s" if s < 90 else f"{s / 60:.0f} min" if s < 5400 else f"{s / 3600:.1f} h"


def main():
    env = Path(os.path.expanduser("~/.config/sc26/env"))
    cfg = dict(l.split("=", 1) for l in (env.read_text().splitlines() if env.exists() else []) if "=" in l and l[0] != "#")
    port = os.environ.get("SC26_PORT") or cfg.get("SC26_PORT", "8626").strip()
    now = time.time()
    bad, repairing, ok = [], [], []

    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/status", timeout=5) as r:
            st = json.loads(r.read())
    except (OSError, ValueError):
        st = None
        bad.append("The fold service is not answering. It restarts by itself within a minute.")

    rows = tail(LOG)
    ticks = [r for r in rows if r.get("ev") == "tick"]
    last = ticks[-1] if ticks else None
    if st:
        chips = [c for c in st["chips"] if c["state"] != "out_of_service"]
        out = [c["chip"] for c in st["chips"] if c["state"] == "out_of_service"]
        folding = [c for c in chips if c["state"] in ("busy", "ready")]
        if not chips:
            ok.append("No chip is in the demo: the screen plays folds recorded on this box.")
        else:
            ok.append(f"{len(folding)} of {len(chips)} chips folding, {sum(c.get('folds') or 0 for c in chips)} folds since the last start.")
        # how long each chip that is not folding has been so, from the watchdog's record of it
        for c in chips:
            if c in folding:
                continue
            since = now
            for r in reversed(ticks):
                s = {x[0]: x[1] for x in (r.get("engine") or {}).get("chips", [])}
                if s.get(c["chip"]) in ("busy", "ready"):
                    break
                since = r["t"]
            (repairing if now - since < REPAIR_S else bad).append(
                f"Chip {c['chip'] + 1} is {c['state']} for {ago(now - since)}"
                + (" (normal: it repairs itself within about 10 min)." if now - since < REPAIR_S
                   else ". It should have recovered by now."))
        if out:
            ok.append(f"Out of service on purpose: chip {', '.join(str(c + 1) for c in out)}.")
        if st.get("queue"):
            ok.append(f"{st['queue']} visitor folds waiting.")

    if last is None or now - last["t"] > 60:
        bad.append("The screen watchdog has not checked in"
                   + (f" for {ago(now - last['t'])}." if last else " at all.") + " Nothing is watching the screen.")
    else:
        scr, fps = last.get("screen"), last.get("fps")
        if scr in ("moving", "none", "lost") and (fps or 0) >= 30:
            ok.append(f"Screen moving at {fps:.0f} frames a second." if scr == "moving" else
                      f"Page drawing at {fps:.0f} frames a second (no screen attached).")
        elif scr == "stuck" or (fps is not None and fps < 30) or scr in ("frozen", "blank"):
            repairing.append(f"Screen {scr}, {fps or 0:.0f} frames a second: the watchdog is restarting it.")
        else:
            repairing.append(f"Screen: the page did not answer the last check ({scr}).")
        age = last.get("stream_age_s")
        if age is not None and age > 10:
            repairing.append(f"The page has heard nothing from the fold service for {age:.0f} s; it reconnects by itself.")
    hour = [r for r in rows if r.get("ev") == "restart" and now - r["t"] < 3600]
    if hour:
        ok.append(f"Restarted in the last hour: {', '.join(sorted({r['unit'] for r in hour}))} ({len(hour)}x).")
    if len([r for r in hour if r.get("unit") == "sc26-kiosk"]) >= 4:
        bad.append("The browser was restarted 4 or more times in an hour.")

    res = next((r for r in reversed(ticks) if "disk_free_gb" in r), None)
    if res:
        if res["disk_free_gb"] < 5:
            bad.append(f"Disk nearly full: {res['disk_free_gb']} GB free.")
        if res["mem_avail_gb"] < 16:
            bad.append(f"Memory low: {res['mem_avail_gb']} GB free.")
        ok.append(f"{res['disk_free_gb']:.0f} GB disk and {res['mem_avail_gb']:.0f} GB memory free.")

    verdict = "NOT HEALTHY" if bad else "REPAIRING ITSELF" if repairing else "HEALTHY"
    print(verdict)
    for line in bad + repairing + ok:
        print("  " + line)
    if bad:
        print("  What to do: wait 2 minutes and run this again. If it still says NOT HEALTHY, restart the box "
              "(one short press of the power button) and call Moritz.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
