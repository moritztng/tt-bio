"""Runs for mgx-trace-region on whglx, one JSON line per run in runs.jsonl.

    python perf/mgx_trace_region/chain.py "<card pool>" <plan file>

A plan line is `<tag> [ENV=VALUE ...] <tt-bio subcommand and args>`; `#` starts a comment and
`{out}` in the args is replaced by the run's output dir. The chain takes one free card from the
pool (lease flock free, lease JSON released or its holder dead, no live process pinned to it),
holds it between runs, and records per run: card, tree REV, rc, wall, AICLK sampled DURING the
run, every trace capture's bytes (hook/sitecustomize.py) and boltz2's per-sample digests. A tag
already recorded is skipped, so a restarted chain resumes. Each run gets a hard wall limit; a
run that exceeds it is left alive (killing a device holder makes the next open hang the host)
and the chain stops, so its card can be fenced by hand.
"""
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "perf"))
from clocksample import during  # noqa: E402

ME = "worker:mgx-trace-region"
LEASES = Path.home() / "leases"
NEVER = {"1", "4", "10", "11", "15", "17", "22", "24", "25", "26", "27"}
RUNS = HERE / "runs.jsonl"
LIMIT_S = int(os.environ.get("CHAIN_LIMIT_S", "1500"))


def _pinned():
    cards = set()
    for p in Path("/proc").iterdir():
        if not p.name.isdigit() or int(p.name) == os.getpid():
            continue
        try:
            env = (p / "environ").read_bytes().split(b"\0")
        except OSError:
            continue
        for kv in env:
            if kv.startswith(b"TT_VISIBLE_DEVICES="):
                cards.update(kv.split(b"=", 1)[1].decode().split(","))
    return cards


def _lease(card):
    return LEASES / f"j10glx02-card{card}.json"


def _free(card, pinned):
    try:
        d = json.loads(_lease(card).read_text())
    except (OSError, ValueError):
        d = {"released": 1}
    if d.get("holder") == ME and d.get("pid") == os.getpid():
        return True
    if not (d.get("released") or not Path(f"/proc/{d.get('pid')}").exists()):
        return False
    if card in pinned:
        return False
    with open(_lease(card), "a") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False
        fcntl.flock(fh, fcntl.LOCK_UN)
    return True


def _mark(card, released):
    _lease(card).write_text(json.dumps({
        "host": "j10glx02", "card": card, "holder": ME, "pid": os.getpid(),
        "acquired": time.time(), "released": time.time() if released else None,
        "note": "held between runs by the mgx-trace-region chain"}) + "\n")


def take(pool, current=None):
    # Check-and-mark under one host lock: two chains started together both saw card 30 free.
    while True:
        with open(LEASES / "mgx-trace-region-pick.lock", "a") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            pinned = _pinned()
            for c in ([current] if current else []) + pool:
                if c not in NEVER and _free(c, pinned):
                    _mark(c, False)
                    return c
        time.sleep(5)


def done():
    if not RUNS.exists():
        return set()
    return {json.loads(l)["tag"] for l in RUNS.read_text().splitlines() if l.strip()}


def run(card, tag, envs, args):
    work = HERE / "out" / tag
    work.mkdir(parents=True, exist_ok=True)
    tlog, digest, log = work / "trace.jsonl", work / "digest.txt", work / "run.log"
    for f in (tlog, digest):
        f.unlink(missing_ok=True)
    env = dict(os.environ, TT_VISIBLE_DEVICES=card, TT_BIO_LEASE_CARDS=card,
               TT_BIO_LEASE_DIR=str(LEASES), TT_BIO_LEASE_HOLDER=ME, TT_BIO_LEASE_TIMEOUT="60",
               TRACE_REGION_LOG=str(tlog), TT_BIO_SAMPLE_DIGEST=str(digest),
               TT_METAL_LOGGER_LEVEL="FATAL",
               PYTHONPATH=f"{HERE / 'hook'}:{ROOT}", **envs)
    args = [a.replace("{out}", str(work / "o")) for a in args]
    cmd = [sys.executable, "-m", "tt_bio.main", *args]
    rev = (ROOT / "REV").read_text().strip() if (ROOT / "REV").exists() else "?"
    os.environ["TT_VISIBLE_DEVICES"] = card        # tt-smi reports the granted chip as index 0
    t0 = time.time()
    with open(log, "w") as fh, during(period=5.0) as clk:
        fh.write(f"START {time.strftime('%FT%TZ', time.gmtime())} card={card} rev={rev} "
                 f"{' '.join(f'{k}={v}' for k, v in envs.items())} {' '.join(cmd)}\n")
        fh.flush()
        p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT,
                             start_new_session=True)
        try:
            rc = p.wait(timeout=LIMIT_S)
        except subprocess.TimeoutExpired:
            rc = None
    wall = time.time() - t0
    caps = [json.loads(l) for l in tlog.read_text().splitlines()] if tlog.exists() else []
    text = log.read_text(errors="replace")
    rec = {"tag": tag, "card": card, "rev": rev, "env": envs, "args": args, "rc": rc,
           "wall_s": round(wall, 1), "aiclk": clk.summary().get(0),
           "captures": len(caps),
           "trace_bytes_per_bank": sorted({c["trace_used_after"] - (c["trace_used_before"] or 0)
                                           for c in caps}),
           "trace_peak_per_bank": max((c["trace_used_after"] for c in caps), default=0),
           "region_per_bank": caps[0]["region_per_bank"] if caps else None,
           "callers": sorted({c["caller"] for c in caps}),
           "digest": digest.read_text().split("\n") if digest.exists() else None,
           "outputs": {str(f.relative_to(work)): hashlib.sha256(f.read_bytes()).hexdigest()[:16]
                       for f in sorted((work / "o").rglob("*"))
                       if f.suffix in (".cif", ".pdb", ".npy", ".npz") and f.is_file()},
           "tail": None if rc == 0 else text[-900:]}
    with open(RUNS, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps({k: rec[k] for k in ("tag", "card", "rc", "wall_s", "captures",
                                          "trace_peak_per_bank", "aiclk")}), flush=True)
    return rc is not None


def main():
    pool = [c for c in re.split(r"[,\s]+", sys.argv[1]) if c]
    card = None
    for line in Path(sys.argv[2]).read_text().splitlines():
        job = line.split("#")[0].split()
        if not job or job[0] in done():
            continue
        tag, rest = job[0], job[1:]
        envs = {}
        while rest and re.match(r"^[A-Z_][A-Z0-9_]*=", rest[0]):
            k, v = rest.pop(0).split("=", 1)
            envs[k] = v
        card = take(pool, card)
        if not run(card, tag, envs, rest):
            print(f"STALLED {tag} on card {card}: left alive, fence it", flush=True)
            return 3
    if card:
        _mark(card, True)
    print("CHAIN_DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
