"""Folds for mgx-sample-width on whglx, one JSON line per fold in runs.jsonl.

    python perf/mgx_sample_width/fold.py "<card pool>" <plan file>

A plan line is `<tag> <model> <input> <samples> [mps=N] [steps=N]`; `#` starts a comment.
The chain takes one card from the pool (lease released or its holder dead, and no live process
pinned to it), holds the lease between folds and folds the plan in order at --host_threads 2.
A tag already recorded as ok is skipped, so a restarted chain resumes. A fold that loses its card
to another row at the device open is retried on another card, not recorded.

Each record carries the card, the commit, wall seconds, AICLK sampled during the fold, the DRAM
census peak (TT_BIO_DRAM_PEAK; it drains the pipeline at every tag, so these walls are capacity
evidence and never a timing), every "N-sample chunk refused" line the sampler printed,
per-sample digests (boltz2's TT_BIO_SAMPLE_DIGEST) and the error on a failure.
"""
import fcntl
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

ME = "worker:mgx-sample-width"
LEASES = Path.home() / "leases"
NEVER = {"1", "4", "24", "25", "26", "27"}   # cardblocked, and chip 4's hung holder
RUNS = HERE / "runs.jsonl"


def _pinned():
    """Every card some other live process has in its TT_VISIBLE_DEVICES."""
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


def _unlocked(card):
    """The flock tt_bio's DeviceLease takes is the real lease; the JSON beside it is advisory."""
    with open(_lease(card), "a") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False
        fcntl.flock(fh, fcntl.LOCK_UN)
        return True


def _free(card, pinned):
    try:
        d = json.loads(_lease(card).read_text())
    except (OSError, ValueError):
        d = {"released": 1}
    mine = d.get("holder") == ME and d.get("pid") == os.getpid()
    if not (mine or d.get("released") or not Path(f"/proc/{d.get('pid')}").exists()):
        return False
    return mine or card not in pinned


def _claim(card):
    _lease(card).write_text(json.dumps({
        "host": "j10glx02", "card": card, "holder": ME, "pid": os.getpid(),
        "acquired": time.time(), "released": None,
        "note": "held between folds by the mgx-sample-width chain"}) + "\n")


def _release(card):
    d = json.loads(_lease(card).read_text())
    if d.get("holder") == ME and d.get("pid") == os.getpid():
        d["released"] = time.time()
        _lease(card).write_text(json.dumps(d) + "\n")


def take(pool, current=None):
    """Poll every 3 s: released cards are re-taken within seconds by the other rows' chains."""
    while True:
        leased = [c for c in ([current] if current else []) + pool
                  if c not in NEVER and _free(c, set())]
        if leased:
            pinned = _pinned()
            with open(LEASES / ".mgx-sample-width.lock", "a") as fh:
                fcntl.flock(fh, fcntl.LOCK_EX)
                for c in leased:
                    if _free(c, pinned) and _unlocked(c):
                        _claim(c)
                        return c
        time.sleep(3)


def done():
    tags = set()
    if RUNS.exists():
        for line in RUNS.read_text().splitlines():
            r = json.loads(line)
            if r.get("ok"):
                tags.add(r["tag"])
    return tags


def fold(card, tag, model, inp, samples, opts):
    work = HERE / "out" / tag
    work.mkdir(parents=True, exist_ok=True)
    census, digest, log = work / "dram.txt", work / "digest.txt", work / "fold.log"
    for f in (census, digest):
        f.unlink(missing_ok=True)
    env = dict(os.environ, TT_VISIBLE_DEVICES=card, TT_BIO_LEASE_CARDS=card,
               TT_BIO_LEASE_DIR=str(LEASES), TT_BIO_LEASE_HOLDER=ME, TT_BIO_LEASE_TIMEOUT="60",
               TT_BIO_DRAM_PEAK=str(census), TT_BIO_SAMPLE_DIGEST=str(digest),
               TT_METAL_CACHE=str(Path.home() / ".cache/tt-metal-cache-mgxsw"),
               TT_METAL_LOGGER_LEVEL="FATAL", PYTHONPATH=str(ROOT))
    cmd = [sys.executable, "-m", "tt_bio.main", "predict", inp, "--model", model,
           "--out_dir", str(work / "pred"), "--diffusion_samples", str(samples),
           "--host_threads", "2", "--accelerator", "tenstorrent", "--override", "--seed", "0"]
    if "mps" in opts:
        cmd += ["--max_parallel_samples", opts["mps"]]
    if "steps" in opts:
        cmd += ["--sampling_steps", opts["steps"]]
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    t0 = time.time()
    os.environ["TT_VISIBLE_DEVICES"] = card        # tt-smi reports the granted chip as index 0
    with open(log, "w") as fh, during(period=5.0) as clk:
        fh.write(f"START {time.strftime('%FT%TZ', time.gmtime())} card={card} commit={commit} "
                 f"{' '.join(cmd)}\n")
        fh.flush()
        rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    wall = time.time() - t0
    text = log.read_text(errors="replace")
    res = list((work / "pred").glob("*/results.json"))
    status = json.loads(res[0].read_text()) if res else []
    ok = rc == 0 and bool(status) and all(s.get("status") == "ok" for s in status)
    peak = None
    if census.exists():
        used = [float(m) for m in re.findall(r"\[DRAM\] [^:]+: ([\d.]+) GiB used", census.read_text())]
        peak = max(used) if used else None
    clock = clk.summary().get(0)
    rec = {"tag": tag, "model": model, "input": inp, "samples": samples, **opts,
           "card": card, "commit": commit, "rc": rc, "ok": ok, "wall_s": round(wall, 1),
           "aiclk": clock, "dram_peak_gib": peak,
           "narrowed": re.findall(r"\d+-sample chunk refused \(.*?\); denoising \d+", text),
           "digest": digest.read_text().split("\n") if digest.exists() else None,
           "error": None if ok else next((s.get("error", "")[:600] for s in status
                                          if s.get("status") != "ok"), text[-600:])}
    if not ok and "Refusing to open it concurrently" in (rec["error"] or ""):
        print(f"lost card {card} to another row at open; retrying elsewhere", flush=True)
        return False
    with open(RUNS, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps({k: rec[k] for k in ("tag", "ok", "wall_s", "dram_peak_gib", "narrowed")}),
          flush=True)
    return True


def main():
    pool = [c for c in sys.argv[1].split(",") if c]
    plan = [l.split("#")[0].split() for l in Path(sys.argv[2]).read_text().splitlines()]
    card = None
    for job in [j for j in plan if j]:
        tag, model, inp, samples, *kv = job
        if tag in done():
            continue
        card = take(pool, card)
        while not fold(card, tag, model, inp, samples, dict(x.split("=", 1) for x in kv)):
            card = take([c for c in pool if c != card])
    if card:
        _release(card)
    print("CHAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
