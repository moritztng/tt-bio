#!/usr/bin/env python3
"""One chain of depth-axis folds on whglx: chain.py OUT.jsonl model:yaml[:extra args] ...

Takes one card from the pool (never 1, 24-27), holds its lease between folds, folds each job at
--host_threads 2 with the size guard off, and appends one row per fold to OUT.jsonl. A fold is
PASS only when it wrote a structure. Otherwise the row names the refusal that ENDED the run (the
last allocator refusal before the fold's own failure line), not the first one in the log: the
engine recovers from earlier refusals by narrowing, so the first one is usually survived.

AICLK is sampled DURING each fold (tt-smi pinned to the card, every 10 s) and stored in the row.
Wall times on this box are capacity evidence; timing verdicts belong to mgx-speed.
"""
import fcntl
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

HOME = Path.home()
LEASES = HOME / "leases"
ME = "worker:mgx-msa-depth"
PY = str(HOME / "env/bin/python")
POOL = [c for c in range(32) if c not in (1, 24, 25, 26, 27)]
ROOT = Path(__file__).resolve().parents[2]
OOM = re.compile(r"Out of Memory: Not enough space to allocate\s+(\d+) B (DRAM|L1) buffer.*?"
                 r"largest free block:\s*(\d+) B\)", re.S)


def lease(c):
    return LEASES / f"j10glx02-card{c}.json"


def free(c, mine_pid):
    try:
        d = json.loads(lease(c).read_text())
    except (OSError, ValueError):
        d = {"released": 1}
    mine = d.get("holder") == ME and d.get("pid") == mine_pid
    if not (mine or d.get("released") or not Path(f"/proc/{d.get('pid')}").exists()):
        return False
    with open(lease(c), "a") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False
    return True


def claim(c):
    lease(c).write_text(json.dumps({"host": "j10glx02", "card": str(c), "holder": ME,
                                    "pid": os.getpid(), "acquired": time.time(), "released": None,
                                    "note": "held between folds by a mgx-msa-depth chain"}) + "\n")


def release(c):
    d = json.loads(lease(c).read_text())
    d["released"] = time.time()
    lease(c).write_text(json.dumps(d) + "\n")


def take(prefer):
    while True:
        for c in ([prefer] if prefer is not None else []) + POOL:
            if free(c, os.getpid()):
                claim(c)
                return c
        time.sleep(5)


def ending(log):
    """(verdict, detail) from a failed fold's log: the last refusal and the final error line."""
    fails = [ln.strip() for ln in log.splitlines() if ln.strip().startswith("✗")]
    ooms = list(OOM.finditer(log))
    d = {"ended_by": fails[-1][:400] if fails else None, "refusals": len(ooms)}
    if ooms:
        last = ooms[-1]
        d.update(last_request_bytes=int(last[1]), last_largest_free_block=int(last[3]))
        origin = re.findall(r"tt_bio origin: ([^\]\n]*)", log)
        if origin:
            d["origin"] = origin[-1][:300]
    tb = [ln for ln in log.splitlines() if re.match(r"^\w*(Error|Exception)\b", ln)]
    if tb:
        d["exception"] = tb[-1][:400]
    text = (d["ended_by"] or "") + " " + d.get("exception", "")
    if "out of device DRAM" in text or "Out of Memory" in text:
        return "OOM_DRAM", d
    if "refus" in text.lower() and "size" in text.lower():
        return "GUARD_REFUSED", d
    return ("TIMEOUT" if "TIMEOUT" in text else "ERROR"), d


def fold(card, model, yaml, extra, out_root, timeout):
    out_dir = out_root / f"{model}_{Path(yaml).stem}_{time.strftime('%H%M%S', time.gmtime())}"
    out_dir.mkdir(parents=True)
    env = dict(os.environ, PYTHONPATH=str(ROOT), TT_VISIBLE_DEVICES=str(card),
               TT_BIO_LEASE_CARDS=str(card), TT_BIO_LEASE_HOLDER=ME, TT_BIO_LEASE_DIR=str(LEASES),
               TT_BIO_SIZE_LIMIT="0", TT_METAL_LOGGER_LEVEL="FATAL", TT_BIO_LEASE_TIMEOUT="60",
               TT_METAL_CACHE=str(HOME / ".cache/tt-metal-cache-msad"))
    cmd = [PY, "-m", "tt_bio.main", "predict", yaml, "--model", model, "--out_dir", str(out_dir),
           "--accelerator", "tenstorrent", "--override", "--debug", "--host_threads", "2", *extra]
    log_path = out_dir.with_suffix(".log")
    clk = []
    t0 = time.time()
    with open(log_path, "w") as lf:
        lf.write(f"START {time.strftime('%FT%TZ', time.gmtime())} card={card} "
                 f"commit={subprocess.getoutput(f'git -C {ROOT} rev-parse --short HEAD')} "
                 f"cmd={shlex.join(cmd)}\n")
        lf.flush()
        p = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT, cwd=ROOT)
        timed_out = False
        while p.poll() is None:
            if time.time() - t0 > timeout:
                p.terminate()
                timed_out = True
                p.wait()
                break
            v = subprocess.run(
                f"TT_VISIBLE_DEVICES={card} timeout 30 /usr/local/bin/tt-smi -s 2>/dev/null",
                shell=True, capture_output=True, text=True).stdout
            try:
                clk.append(int(json.loads(v)["device_info"][0]["telemetry"]["aiclk"]))
            except (ValueError, KeyError, IndexError, TypeError):
                pass
            time.sleep(10)
    wall = round(time.time() - t0, 1)
    log = log_path.read_text(errors="replace")
    structs = list(out_dir.rglob("*.cif")) + list(out_dir.rglob("*.pdb"))
    if structs and p.returncode == 0 and not timed_out:
        verdict, d = "PASS", {"structure": str(structs[0]),
                              "refusals_recovered": len(OOM.findall(log))}
    else:
        verdict, d = ending(log)
        if timed_out:
            verdict = "TIMEOUT"
    rows = re.findall(r"(\d+) (?:MSA )?rows", log)
    ts = sorted(clk)
    return {"model": model, "rung": Path(yaml).stem, "extra": extra, "card": card,
            "verdict": verdict, "wall_s": wall, "rc": p.returncode, "log": str(log_path),
            "aiclk_median": ts[len(ts) // 2] if ts else None, "aiclk_min": ts[0] if ts else None,
            "aiclk_n": len(ts), "load1": os.getloadavg()[0], **d}


def main():
    out = Path(sys.argv[1]).resolve()
    timeout = int(os.environ.get("MSAD_TIMEOUT", "10800"))
    out_root = out.parent / "runs"
    out_root.mkdir(parents=True, exist_ok=True)
    card = None
    for job in sys.argv[2:]:
        model, yaml, *rest = job.split(":", 2)
        extra = shlex.split(rest[0]) if rest else []
        while True:
            card = take(card)
            row = fold(card, model, yaml, extra, out_root, timeout)
            claim(card)
            if "DeviceInUseError" not in Path(row["log"]).read_text(errors="replace"):
                break
            card = None
        with open(out, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
    if card is not None:
        release(card)
    print("CHAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
