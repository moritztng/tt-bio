#!/usr/bin/env python3
"""Run `exits.py` legs one after another and time how each process ends.

    driver.py --tree <tt-bio tree> --out <dir> --legs success,refusal,sigterm

Every line the campaign prints is stamped on arrival. A leg records its exit status and the
seconds from the campaign's last word (`EVENT returned` / `EVENT raised`, or the SIGTERM the
driver sent) to the process being reaped. A process still alive 300 s after its last word is a
hang: it gets SIGTERM, and if that is ignored for 120 s the driver stops and leaves it for a
human, because this fleet never hard-kills a process holding a chip.
"""
import argparse
import json
import os
import pathlib
import signal
import subprocess
import sys
import threading
import time

HERE = pathlib.Path(__file__).resolve().parent


def leg(tree, out, mode, py, extra):
    d = pathlib.Path(out) / mode
    d.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": f"{tree}:{os.environ.get('BCX_BC2', '')}"}
    env.pop("TT_BIO_DEBUG_STDERR", None)        # the user's default
    cmd = [py, "-u", "-X", "faulthandler", str(HERE / "exits.py"), "--mode", mode,
           "--out", str(d / "campaign"), *extra]
    log = open(d / "log.txt", "w")
    t0 = time.time()
    p = subprocess.Popen(cmd, cwd=tree, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, errors="replace")
    rec = {"mode": mode, "tree": tree, "pid": p.pid, "events": [], "start": t0}
    last_word = [None]
    sent = []

    def pump():
        for line in p.stdout:
            t = time.time()
            log.write(f"{t - t0:9.2f} {line}")
            log.flush()
            if line.startswith("EVENT "):
                _, name, payload = line.rstrip("\n").split(" ", 2)
                rec["events"].append({"t": round(t - t0, 2), "name": name,
                                      **json.loads(payload)})
                if name in ("returned", "raised"):
                    last_word[0] = t
                if mode == "sigterm" and name == "round" and json.loads(payload)["n"] == 3:
                    p.send_signal(signal.SIGTERM)
                    sent.append(("SIGTERM", time.time()))
                    last_word[0] = time.time()
    th = threading.Thread(target=pump, daemon=True)
    th.start()

    while p.poll() is None:
        time.sleep(0.2)
        now = time.time()
        if last_word[0] and now - last_word[0] > 300 and not any(s == "SIGTERM" for s, _ in sent):
            rec["hung_after_last_word_s"] = round(now - last_word[0], 1)
            p.send_signal(signal.SIGTERM)
            sent.append(("SIGTERM", now))
        elif sent and now - sent[-1][1] > 120 and p.poll() is None:
            rec["sigterm_ignored_for_s"] = round(now - sent[-1][1], 1)
            rec["left_running"] = True
            break
        elif not last_word[0] and now - t0 > 2400 and not sent:
            p.send_signal(signal.SIGINT)
            sent.append(("SIGINT", now))
    rc = p.poll()
    end = time.time()
    th.join(timeout=5)
    rec.update({"returncode": rc, "wall_s": round(end - t0, 1),
                "signals_sent": [[s, round(t - t0, 2)] for s, t in sent]})
    if last_word[0] and rc is not None:
        rec["exit_after_last_word_s"] = round(end - last_word[0], 2)
    (d / "leg.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: v for k, v in rec.items() if k != "events"}), flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--legs", default="success,refusal,sigterm")
    ap.add_argument("--py", default=sys.executable)
    ap.add_argument("extra", nargs="*")
    args = ap.parse_args()
    for mode in args.legs.split(","):
        if leg(args.tree, args.out, mode, args.py, args.extra).get("left_running"):
            sys.exit("a leg ignored SIGTERM and is still running; stopping here")


if __name__ == "__main__":
    main()
