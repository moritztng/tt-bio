"""Log every engine event's arrival time; when a busy chip goes quiet, dump tt-triage on the live
hung process. Never kills, never resets.

    hangwatch.py --port 8641 --out RUN --logs RUN/metal --quiet-s 60
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

ENGINE = Path.home() / "sc26/demo/sc26/engine"
sys.path.insert(0, str(ENGINE))
import client  # noqa: E402

HERE = Path(__file__).resolve().parent
SITE = next((Path.home() / "tt-bio-dev/env/lib").glob("python3*/site-packages"))


def sysfs(chip):
    d = Path(f"/sys/class/tenstorrent/tenstorrent!{chip}")
    out = {f: (d / f).read_text().strip() for f in ("tt_aiclk", "tt_heartbeat") if (d / f).exists()}
    for h in (d / "device/hwmon").glob("hwmon*"):
        for f in ("temp1_input", "power1_input", "curr1_input", "in0_input"):
            try:
                out[f] = (h / f).read_text().strip()
            except OSError:
                pass
    return out


def triage(out, chip, logs, tag):
    dst = out / f"triage-{tag}.txt"
    env = dict(os.environ, TT_VISIBLE_DEVICES=str(chip), TT_METAL_LOGS_PATH=str(logs))
    with open(dst, "w") as f:
        f.write(json.dumps({"sysfs": sysfs(chip), "t": time.time()}) + "\n")
        f.flush()
        subprocess.run(["timeout", "600", str(HERE / "triage-venv/bin/python"), str(SITE / "triage/triage.py"),
                        "--skip-version-check", "--disable-progress", "--disable-colors", "-v"],
                       env=env, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, cwd=str(out))
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--chip", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--logs", required=True, help="TT_METAL_LOGS_PATH of the engine under test")
    ap.add_argument("--quiet-s", type=float, default=60)
    a = ap.parse_args()
    out = Path(a.out)
    gaps = open(out / "gaps.jsonl", "a", buffering=1)
    while True:
        try:
            s = client.connect("127.0.0.1", a.port)
            break
        except OSError:
            time.sleep(2)
    s.settimeout(2.0)
    last, state, fired, replays = time.time(), None, False, set()
    while True:
        try:
            op, data = client.recv(s)
        except TimeoutError:
            op = None
        except (OSError, ConnectionError):
            break
        now = time.time()
        if op == 8:
            break
        if op is not None:
            m = json.loads(data)
            t = m.get("type")
            if t == "fold_start" and m.get("source") == "replay":
                replays.add(m.get("id"))
            if m.get("id") in replays or (t != "status" and m.get("chip") not in (None, a.chip)):
                continue  # a replay or another chip says nothing about this chip
            if t == "status":
                c = next((c for c in m["chips"] if c["chip"] == a.chip), {})
                state = c.get("state")
            else:
                keep = ("id", "stage", "step", "total", "state", "n_res", "name", "seconds", "warm")
                gaps.write(json.dumps({"t": round(now, 3), "gap": round(now - last, 3), "type": t,
                                       **{k: m.get(k) for k in keep if m.get(k) is not None}}) + "\n")
                last, fired = now, False
        if state in ("busy", "warming") and now - last > a.quiet_s and not fired:
            fired = True
            gaps.write(json.dumps({"t": round(now, 3), "type": "HANG", "quiet_s": round(now - last, 1)}) + "\n")
            dst = triage(out, a.chip, a.logs, time.strftime("%H%M%S", time.gmtime()))
            gaps.write(json.dumps({"t": round(time.time(), 3), "type": "TRIAGED", "file": str(dst)}) + "\n")


if __name__ == "__main__":
    main()
