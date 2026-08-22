#!/usr/bin/env python3
"""Characterize the protenix-v2 640 aa hang: run N lever-census-wrapped folds at
one rung, and on a hang capture the signature (python stack, host CPU ticks, ARC
heartbeat, last log lines) before cleaning the card up.

Read-only with respect to model code: it spawns exactly the command
release_gate.py's _run_census_fold builds, nothing else.
"""
import argparse, json, os, re, signal, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def sh(cmd, timeout=60):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=timeout).stdout
    except Exception as e:
        return f"<{e}>"


def proc_tree(root):
    """pids of root and all descendants, from /proc."""
    kids = {}
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            stat = (p / "stat").read_text()
        except Exception:
            continue
        rp = stat.rfind(")")
        f = stat[rp + 2:].split()
        kids.setdefault(int(f[1]), []).append(int(p.name))
    out, stack = [], [root]
    while stack:
        pid = stack.pop()
        out.append(pid)
        stack.extend(kids.get(pid, []))
    return out


def proc_sample(pid):
    try:
        stat = (Path("/proc") / str(pid) / "stat").read_text()
    except Exception:
        return None
    rp = stat.rfind(")")
    f = stat[rp + 2:].split()
    return {"pid": pid, "state": f[0], "ppid": int(f[1]),
            "utime": int(f[11]), "stime": int(f[12]),
            "cmd": sh(f"tr '\\0' ' ' < /proc/{pid}/cmdline")[:160]}


def arc_heartbeat(card):
    txt = sh("~/.local/bin/tt-smi -s --snapshot_no_tty --no_reinit 2>/dev/null", timeout=90)
    try:
        d = json.loads(txt)
        return d["device_info"][card]["smbus_telem"].get("TIMER_HEARTBEAT")
    except Exception:
        return None


def device_holders(card):
    out = sh(f"fuser /dev/tenstorrent/{card} 2>/dev/null")
    return [int(x) for x in out.split()]


def capture(pid, log, card, gap):
    """Two samples `gap` seconds apart: CPU ticks + ARC heartbeat, plus stacks."""
    tree = proc_tree(pid)
    s1 = [x for x in (proc_sample(p) for p in tree) if x]
    hb1 = arc_heartbeat(card)
    try:
        t1 = "\n".join(Path(log).read_text(errors="replace").splitlines()[-25:])
    except Exception:
        t1 = ""
    stacks = {}
    for p in tree:
        stacks[p] = sh(f"~/.local/bin/py-spy dump --pid {p} --nonblocking 2>&1", timeout=90)
    time.sleep(gap)
    s2 = [x for x in (proc_sample(p) for p in proc_tree(pid)) if x]
    hb2 = arc_heartbeat(card)
    by = {x["pid"]: x for x in s1}
    cpu = []
    for x in s2:
        o = by.get(x["pid"])
        if o:
            d = (x["utime"] + x["stime"]) - (o["utime"] + o["stime"])
            cpu.append({"pid": x["pid"], "state": x["state"], "ppid": x["ppid"],
                        "ticks_delta": d, "cpu_pct": round(100.0 * d / (100.0 * gap), 1),
                        "cmd": x["cmd"]})
    def tail():
        try:
            return "\n".join(Path(log).read_text(errors="replace").splitlines()[-25:])
        except Exception:
            return ""
    t2 = tail()
    stacks2 = {}
    for p in proc_tree(pid):
        stacks2[p] = sh(f"~/.local/bin/py-spy dump --pid {p} --nonblocking 2>&1", timeout=90)
    return {"cpu": cpu, "arc_heartbeat": [hb1, hb2],
            "arc_advancing": (hb1 is not None and hb2 is not None and hb1 != hb2),
            "log_tail": t2, "log_advanced": (t1 != t2),
            "stacks": stacks, "stacks_after_gap": stacks2,
            "device_holders": device_holders(card)}


def cleanup(proc, card):
    """Both signals unconditionally (the d5ade211 lesson), then sweep holders by
    explicit pid."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except Exception:
            pass
        try:
            proc.wait(timeout=15)
        except Exception:
            pass
    killed = []
    # Sweep repeatedly: a grandchild can still be mid-spawn when the first sweep
    # runs, and one survivor wedges the card for every later trial (the failure
    # mode that made release-gate trials 4-5 non-independent samples).
    for _ in range(6):
        time.sleep(3)
        holders = device_holders(card)
        if not holders:
            break
        for pid in holders:
            try:
                os.kill(pid, signal.SIGKILL)
                killed.append(pid)
            except Exception:
                pass
    time.sleep(5)
    return {"killed_holders": killed, "holders_after": device_holders(card)}


def one_trial(args, i, workdir):
    label = f"{args.model}-{args.rung}-t{i}"
    fixture = REPO / "perf" / "size512" / "fixtures" / f"cdk2x2_{args.rung}.yaml"
    census_json = workdir / f"census_{label}.json"
    out_dir = workdir / f"out_{label}"
    log = workdir / f"{label}.log"
    subprocess.run(f"rm -rf {out_dir}", shell=True)
    cmd = [sys.executable, str(REPO / "scripts" / "lever_census.py"),
           "--tt-bio", sys.executable, "--label", label, "--out", str(census_json),
           "--", "-m", "tt_bio.main", "predict", str(fixture),
           "--model", args.model, "--single_sequence",
           "--sampling_steps", "6", "--diffusion_samples", "1", "--seed", "0",
           "--out_dir", str(out_dir)]
    env = dict(os.environ)
    env["TT_VISIBLE_DEVICES"] = str(args.card)
    env["TT_BIO_LEASE_CARDS"] = str(args.card)
    env["TT_BIO_LEASE_HOLDER"] = "worker:protenix-v2-640aa-hang-characterize"
    if args.force_grid:
        env["TT_BIO_FORCE_GRID"] = args.force_grid
    rec = {"trial": i, "label": label, "force_grid": args.force_grid or None,
           "card": args.card, "host": os.uname().nodename}
    t0 = time.monotonic()
    with open(log, "w") as fp:
        proc = subprocess.Popen(cmd, cwd=str(REPO), stdout=fp,
                                stderr=subprocess.STDOUT, env=env,
                                start_new_session=True)
        while True:
            if proc.poll() is not None:
                break
            if time.monotonic() - t0 > args.timeout:
                rec["outcome"] = "HANG"
                rec["wall"] = round(time.monotonic() - t0, 1)
                rec["signature"] = capture(proc.pid, log, args.card, args.gap)
                rec["cleanup"] = cleanup(proc, args.card)
                return rec
            time.sleep(4)
    rec["wall"] = round(time.monotonic() - t0, 1)
    rec["rc"] = proc.returncode
    if proc.returncode != 0:
        rec["outcome"] = "ERROR"
        rec["log_tail"] = "\n".join(log.read_text(errors="replace").splitlines()[-20:])
        return rec
    rec["outcome"] = "CLEAN"
    try:
        c = json.loads(census_json.read_text())
        rec["grid"] = c.get("grid")
    except Exception as e:
        rec["grid"] = f"<{e}>"
    # Glob rather than reconstruct the results-dir name: one fewer thing to get
    # wrong, and no import of tt_bio in the probe process. Record WHY it failed
    # instead of a silent None -- an unexplained None reads as "the fold produced
    # no timing" when it may just be a bad path.
    hits = sorted(out_dir.glob("*/results.json"))
    if not hits:
        rec["runtime_s"] = None
        rec["runtime_s_why"] = f"no */results.json under {out_dir.name}"
    else:
        try:
            rows = json.loads(hits[0].read_text())
            ts = [r["runtime_s"] for r in rows
                  if r.get("status") == "ok" and r.get("runtime_s") is not None]
            rec["runtime_s"] = max(ts) if ts else None
            if not ts:
                rec["runtime_s_why"] = f"no ok row with runtime_s in {hits[0].name}"
        except Exception as e:
            rec["runtime_s"] = None
            rec["runtime_s_why"] = f"{type(e).__name__}: {e}"
    if not args.keep:
        subprocess.run(f"rm -rf {out_dir}", shell=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="protenix-v2")
    ap.add_argument("--rung", type=int, default=640)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--gap", type=float, default=20.0)
    ap.add_argument("--card", type=int, default=3)
    ap.add_argument("--force-grid", default="")
    ap.add_argument("--keep", action="store_true",
                    help="keep each fold's out_dir as evidence")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    workdir = Path(args.out)
    workdir.mkdir(parents=True, exist_ok=True)
    results_path = workdir / "trials.jsonl"
    for i in range(args.trials):
        r = one_trial(args, i, workdir)
        with open(results_path, "a") as fp:
            fp.write(json.dumps(r) + "\n")
        print(f"[trial {i}] {r['outcome']} wall={r.get('wall')} "
              f"runtime_s={r.get('runtime_s')} grid={r.get('grid')}", flush=True)
        if r["outcome"] == "HANG":
            sig = r["signature"]
            print(f"    ARC advancing: {sig['arc_advancing']} {sig['arc_heartbeat']}", flush=True)
            for c in sig["cpu"]:
                print(f"    pid {c['pid']} {c['state']} {c['cpu_pct']}% {c['cmd'][:70]}", flush=True)
            print(f"    log tail: {sig['log_tail'].splitlines()[-1] if sig['log_tail'] else ''}", flush=True)


if __name__ == "__main__":
    main()
