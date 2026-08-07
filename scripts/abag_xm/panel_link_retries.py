#!/usr/bin/env python3
"""Link p27's completed retry folds into their origin windows (p25/p25b/p26).

Workstream abag-xm-panel-to-100pct, Step 4. The 2026-08-03 disk-full burst killed
58 folds in p25/p25b/p26; p27 was built as their retry sweep at the identical
(model, target, rung=64, seed), and 56 of them completed there (the other 6 are the
documented Class-A WH DRAM exclusions, which have no p27 source by design). Same
model, target, rung, seed, engine tree -> the same scientific measurement, so the
origin window's slot is filled by hardlinking the verified p27 fold, exactly like
p29_fleet.sh's chunk-0 skip-and-link phase.

For each origin-window failure key, before anything is written the SOURCE fold is
re-verified on disk: results.json status=="ok" and len(all_runs)==64, exactly 64
.cif files, 64 md5-distinct CIFs. Only then:
  (a) cp -al the source fold dir into the origin window,
  (b) append a schema-exact rc=0 record to the origin results.jsonl carrying the
      SOURCE fold's real seconds/mps/umd (never invented, never double-counted),
  (c) append a reused_chunks.jsonl provenance line,
  (d) pre-create the origin window's claim dir for that task index.

Idempotent: a key whose last origin record is already rc=0 is skipped. No CIF is
ever copied to make a count -- the link is legitimate only because the fold it
points at is the same (model, target, rung, seed) measurement, verified 64/64.

Usage: panel_link_retries.py [--apply]   (default: dry-run, prints what would link)
       panel_link_retries.py --p28-from-p29 [--apply]
Runs on the Galaxy; H=/home/cust-team/mthuening (PANEL_LINK_HOME overrides, for fixtures).

--p28-from-p29: fill p28's failed rung-64 slots from p29's rung-256 chunk-0 folds.
Seed nesting makes chunk 0 of the N=256 ladder byte-identical to the N=64 slot:
p29's c0 runs --diffusion_samples 64 (256/4) --seed <base>, p28's rung-64 slot runs
--diffusion_samples 64 (64/1) --seed <base> -- same model, target, samples, seed,
engine tree. p28 records carry chunk/chunks fields; p25/p25b/p26 do not.
"""
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

H = pathlib.Path(os.environ.get("PANEL_LINK_HOME", "/home/cust-team/mthuening"))
ORIGINS = ["p25", "p25b", "p26"]
SOURCE = H / "p27"
MD = {"boltz2": "boltz2", "opendde-abag": "opendde",
      "protenix-v2": "protenix", "esmfold2": "esmfold2"}


def last_wins(path):
    """results.jsonl -> {(model, target, rung, seed): record}, last attempt wins."""
    best = {}
    if not path.exists():
        return best
    for line in path.read_text().splitlines():
        if not line.startswith("{"):
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        best[(r.get("model"), r.get("target"), str(r.get("rung")), str(r.get("seed")))] = r
    return best


def verify_fold(d, t):
    """results.json ok + 64 all_runs + 64 CIFs + md5-distinct count (binding verify)."""
    rs = list(d.glob(f"*results_{t}"))
    if len(rs) != 1:
        return 0, f"results dir count={len(rs)}"
    try:
        rec = json.loads((rs[0] / "results.json").read_text())[0]
        if rec.get("status") != "ok" or len(rec.get("all_runs") or []) != 64:
            return 0, "results.json not ok/64"
    except Exception as exc:
        return 0, f"results.json unreadable: {exc}"
    cifs = list((rs[0] / "structures").glob("*.cif"))
    if len(cifs) != 64:
        return 0, f"cifs={len(cifs)}"
    n_distinct = len({hashlib.md5(p.read_bytes()).hexdigest() for p in cifs})
    if n_distinct != 64:
        return 0, f"distinct={n_distinct}"
    return n_distinct, "ok"


def link_p28_from_p29(apply, now):
    """Fill p28 rung-64 failures from p29's rung-256 chunk-0 folds (seed-nested twin)."""
    SRC, W = H / "p29", H / "p28"
    src_best = last_wins(SRC / "results.jsonl")  # rung-256 seed is chunk-unique (base+1000*j)
    tasks = [l.split() for l in (W / "tasks.txt").read_text().splitlines() if l.strip()]
    idx = {(f[0], f[1], f[2], f[3]): i for i, f in enumerate(tasks, 1) if len(f) >= 4}
    best = last_wins(W / "results.jsonl")
    linked, no_source, bad_verify = 0, [], []
    rf = open(W / "results.jsonl", "a") if apply else None
    pf = open(W / "reused_chunks.jsonl", "a") if apply else None
    try:
        for (m, t, rung, seed), i in sorted(idx.items(), key=lambda kv: kv[1]):
            if rung != "64":
                continue  # only rung-64 slots have a rung-256 chunk-0 twin
            last = best.get((m, t, rung, seed))
            if last and last.get("rc") in (0, "0") and last.get("cifs", 0) > 0:
                continue  # already done (idempotent)
            src = src_best.get((m, t, "256", seed))
            if not (src and src.get("rc") in (0, "0") and src.get("chunk") in (0, "0")
                    and src.get("cifs") == 64 and src.get("distinct") == 64):
                no_source.append(f"{m}/{t}")
                continue
            srcdir = SRC / MD[m] / f"{t}_c0"
            n_distinct, why = verify_fold(srcdir, t)
            if n_distinct != 64:
                bad_verify.append(f"{m}/{t}: {why}")
                continue
            dst = W / MD[m] / t
            if apply:
                if not dst.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    subprocess.run(["cp", "-al", str(srcdir), str(dst)], check=True)
                rf.write(json.dumps(
                    {"model": m, "target": t, "rung": 64, "seed": int(seed),
                     "chunk": 0, "chunks": 1,
                     "mps": str(src["mps"]), "umd": src["umd"], "rc": 0,
                     "seconds": src["seconds"], "cifs": 64, "distinct": n_distinct,
                     "oom": 0}, separators=(",", ":")) + "\n")
                pf.write(json.dumps(
                    {"model": m, "target": t, "rung": 64, "seed": int(seed),
                     "chunk": 0, "chunks": 1,
                     "source_window": SRC.name, "source_dir": str(srcdir),
                     "source_rung": 256, "source_chunk": 0,
                     "source_seconds": src["seconds"], "source_mps": str(src["mps"]),
                     "verify": "results.json ok, 64 cifs, 64 md5-distinct",
                     "reused_at": now, "claim_idx": i}) + "\n")
                (W / "claims" / str(i)).mkdir(parents=True, exist_ok=True)
            linked += 1
    finally:
        if rf:
            rf.close()
            pf.close()
    print(f"p28<-p29: {'linked' if apply else 'would link'} {linked}"
          + (f"  no-p29-source: {sorted(no_source)}" if no_source else "")
          + (f"  FAILED-VERIFY: {bad_verify}" if bad_verify else ""))
    return linked


def main():
    apply = "--apply" in sys.argv[1:]
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if "--p28-from-p29" in sys.argv[1:]:
        link_p28_from_p29(apply, now)
        return
    src_best = last_wins(SOURCE / "results.jsonl")
    total_linked = 0
    for win in ORIGINS:
        W = H / win
        tasks = [l.split() for l in (W / "tasks.txt").read_text().splitlines() if l.strip()]
        # task index (1-based) per key, for the claim dir
        idx = {(f[0], f[1], f[2], f[3]): i for i, f in enumerate(tasks, 1) if len(f) >= 4}
        best = last_wins(W / "results.jsonl")
        linked, no_source, bad_verify = 0, [], []
        rf = open(W / "results.jsonl", "a") if apply else None
        pf = open(W / "reused_chunks.jsonl", "a") if apply else None
        try:
            for key, i in sorted(idx.items(), key=lambda kv: kv[1]):
                m, t, rung, seed = key
                last = best.get(key)
                # "done" means rc=0 WITH structures -- a swallowed OOM records
                # rc=0/cifs=0 and is still a failure (same rule as the DONE_CHECK).
                if last and last.get("rc") in (0, "0") and last.get("cifs", 0) > 0:
                    continue  # already done (idempotent)
                src = src_best.get(key)
                if not (src and src.get("rc") in (0, "0")
                        and src.get("cifs") == 64 and src.get("distinct") == 64):
                    no_source.append(f"{m}/{t}")
                    continue
                srcdir = SOURCE / MD[m] / t
                n_distinct, why = verify_fold(srcdir, t)
                if n_distinct != 64:
                    bad_verify.append(f"{m}/{t}: {why}")
                    continue
                dst = W / MD[m] / t
                if apply:
                    if not dst.exists():
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        subprocess.run(["cp", "-al", str(srcdir), str(dst)], check=True)
                    rf.write(json.dumps(
                        {"model": m, "target": t, "rung": int(rung), "seed": int(seed),
                         "mps": str(src["mps"]), "umd": src["umd"], "rc": 0,
                         "seconds": src["seconds"], "cifs": 64, "distinct": n_distinct,
                         "oom": 0}, separators=(",", ":")) + "\n")
                    pf.write(json.dumps(
                        {"model": m, "target": t, "rung": int(rung), "seed": int(seed),
                         "source_window": SOURCE.name, "source_dir": str(srcdir),
                         "source_seconds": src["seconds"], "source_mps": str(src["mps"]),
                         "verify": "results.json ok, 64 cifs, 64 md5-distinct",
                         "reused_at": now, "claim_idx": i}) + "\n")
                    (W / "claims" / str(i)).mkdir(parents=True, exist_ok=True)
                linked += 1
        finally:
            if rf:
                rf.close()
                pf.close()
        print(f"{win}: linked {linked}"
              + (f"  no-p27-source: {sorted(no_source)}" if no_source else "")
              + (f"  FAILED-VERIFY: {bad_verify}" if bad_verify else ""))
        total_linked += linked
    print(f"TOTAL {'linked' if apply else 'would link'}: {total_linked}")


if __name__ == "__main__":
    main()
