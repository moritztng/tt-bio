#!/usr/bin/env python3
"""Merge the peer host's Tier-A output into this host's campaign directory.

The two hosts fold DISJOINT slices and share nothing: `progress.jsonl`, the generator output
directories and the label JSONs are all host-local. Every downstream stage --
abag_xm_build_release_tables, abag_xm_stage_release, abag_xm_publish -- reads only the local
`tier_a`. Run the release on one host without merging first and it silently ships that host's
half, and the publish preflight ("every generator has the same number of folds") does not catch
it, because both halves are lopsided in the same way.

`abag_xm_merge_qb2_opendde.sh` predates this: it merges opendde-abag ONLY, from when qb2 ran just
the opendde campaign. qb2 now runs all three generators for slices 0-3, so that script would strand
qb2's protenix-v2 and boltz2 output -- about a third of the slab. This supersedes it.

DUPLICATES: a pair can legitimately exist on both hosts, because one host takes over the other's
slices while it is down (4 such pairs as of 2026-07-28). **The local copy wins, wholesale.** Not
because it is better -- both are the same seed and code -- but because the CIFs, the PAE arrays,
the label JSON and the provenance record must all come from the SAME host's copy. Merging a label
from one host onto coordinates from another is how a row stops being auditable.

    python3 scripts/abag_xm_merge_hosts.py --peer tt-quietbox2 [--dry-run]

Also brings the peer's ranker_scores.csv rows across, which is what lets DeepRank-Ab run on BOTH
hosts concurrently: at ~82 s/fold it is ~11 h for 492 folds on one host, and scoring each host's own
half first halves that. Without this the peer's scored rows are lost in the merge and the merged host
has to redo them, so the second host's work counts for nothing. ranker_scores.py --all already skips
any (target, gen) present in the CSV, so carrying the rows over is all that is required.

Idempotent: re-running merges only what is still missing.
"""
import argparse
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

TIERA = Path.home() / "abag_xm" / "tier_a"
RANKER_CSV = "ranker_scores.csv"
# generator -> (output subdirectory, the result-dir prefix inside it, label-file prefix)
GENS = {
    "protenix-v2":  ("protenix_v2", "protenix_results", "protenix_v2"),
    "opendde-abag": ("opendde_abag", "opendde_results", "opendde_abag"),
    "boltz2":       ("boltz2", "boltz2_results", "boltz2"),
    "esmfold2":     ("esmfold2", "esmfold2_results", "esmfold2"),
}


def _ssh(peer, cmd, timeout=120):
    return subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                           f"ttuser@{peer}", cmd], capture_output=True, text=True, timeout=timeout)


def _ok_pairs(lines):
    out = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("status") == "ok":
            out[(r["target"], r["model"])] = r   # latest wins, matching build_release_tables
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peer", default="tt-quietbox2")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    prog = TIERA / "progress.jsonl"
    if not prog.exists():
        sys.exit(f"ABORT: {prog} not found -- run this on a host that has been folding")

    local = _ok_pairs(prog.read_text().splitlines())
    r = _ssh(a.peer, "cat ~/abag_xm/tier_a/progress.jsonl")
    if r.returncode != 0:
        sys.exit(f"ABORT: cannot read {a.peer} progress.jsonl: {r.stderr.strip()[:200]}")
    peer = _ok_pairs(r.stdout.splitlines())

    incoming = {k: v for k, v in peer.items() if k not in local}
    dup = sorted(set(peer) & set(local))
    print(f"local ok={len(local)}  peer ok={len(peer)}  to merge={len(incoming)}  "
          f"already local (peer copy discarded)={len(dup)}")
    for t, g in dup[:8]:
        print(f"    dup, keeping local: {t} {g}")

    # ---- coordinates + PAE, per generator, skipping any target the local host already owns
    for gen, (subdir, prefix, _lab) in GENS.items():
        want = sorted(t for (t, g) in incoming if g == gen)
        if not want:
            print(f"[{gen}] nothing to merge")
            continue
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
            for t in want:
                fh.write(f"{prefix}_{t}/***\n")
            inc = fh.name
        dest = TIERA / subdir
        cmd = ["rsync", "-a", "--prune-empty-dirs",
               f"--include-from={inc}", "--include=*/", "--exclude=*",
               f"ttuser@{a.peer}:abag_xm/tier_a/{subdir}/", f"{dest}/"]
        if a.dry_run:
            cmd.insert(2, "-n")
        else:
            dest.mkdir(parents=True, exist_ok=True)
        rr = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        print(f"[{gen}] {'would rsync' if a.dry_run else 'rsync'} {len(want)} result dirs "
              f"-> rc={rr.returncode}"
              + (f"  {rr.stderr.strip()[:200]}" if rr.returncode else ""))

    # ---- label JSONs, same skip rule so a label always matches its coordinates
    labs = [f"{GENS[g][2]}_{t}.json" for (t, g) in sorted(incoming)]
    if labs:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
            fh.write("\n".join(labs) + "\n")
            inc = fh.name
        dest = TIERA / "labels"
        cmd = ["rsync", "-a", f"--include-from={inc}", "--exclude=*",
               f"ttuser@{a.peer}:abag_xm/tier_a/labels/", f"{dest}/"]
        if a.dry_run:
            cmd.insert(2, "-n")
        else:
            dest.mkdir(parents=True, exist_ok=True)
        rr = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        print(f"[labels] {'would rsync' if a.dry_run else 'rsync'} up to {len(labs)} label "
              f"files -> rc={rr.returncode}"
              + (f"  {rr.stderr.strip()[:200]}" if rr.returncode else ""))

    # ---- ranker_scores.csv rows for the merged folds
    # Why this is here at all: DeepRank-Ab is the longest post-generation phase (~82 s/fold, so
    # ~11 h for 492 folds on one host) and the plan is to run it on BOTH hosts concurrently, which
    # halves it. That only works if the peer's scored rows survive the merge -- otherwise the merged
    # host has to rescore everything and the second host's work is thrown away. ranker_scores.py
    # --all already skips any (target, gen) already present in the CSV, so bringing the peer's rows
    # across is all that is needed to make its half count.
    #
    # Same local-wins rule as everything else: a pair scored on both hosts keeps the local row, so
    # the scores stay consistent with the coordinates and labels the merge kept.
    peer_csv = f"abag_xm/tier_a/{RANKER_CSV}"
    local_csv = TIERA / RANKER_CSV
    rr = _ssh(a.peer, f"cat {peer_csv} 2>/dev/null")
    if rr.returncode != 0 or not (rr.stdout or "").strip():
        print(f"[ranker] peer has no {RANKER_CSV} -- nothing to merge")
    else:
        import csv as _csv
        import io as _io
        peer_rows = list(_csv.DictReader(_io.StringIO(rr.stdout)))
        local_rows, header = [], None
        if local_csv.exists() and local_csv.stat().st_size:
            with open(local_csv, newline="") as fh:
                rd = _csv.DictReader(fh)
                header = rd.fieldnames
                local_rows = list(rd)
        header = header or (list(peer_rows[0].keys()) if peer_rows else None)
        have = {(r.get("target"), r.get("gen")) for r in local_rows}
        add = [r for r in peer_rows if (r.get("target"), r.get("gen")) not in have]
        if a.dry_run:
            print(f"[ranker] would append {len(add)} of {len(peer_rows)} peer rows to "
                  f"{local_csv} ({len(peer_rows) - len(add)} already local)")
        elif add and header:
            write_header = not local_csv.exists() or not local_csv.stat().st_size
            with open(local_csv, "a", newline="") as fh:
                w = _csv.DictWriter(fh, fieldnames=header, extrasaction="ignore")
                if write_header:
                    w.writeheader()
                for r in add:
                    w.writerow(r)
            print(f"[ranker] appended {len(add)} peer rows to {local_csv}")
        else:
            print(f"[ranker] nothing to append ({len(peer_rows)} peer rows all present locally)")

    # ---- progress records last: only after the artifacts they describe are present, so an
    # interrupted merge never leaves a record pointing at coordinates that were not copied.
    new = []
    for k, rec in sorted(incoming.items()):
        rec = dict(rec)
        # Keep the host the FOLD recorded for itself; --peer is only a fallback. --peer is an ssh
        # target, so it can legitimately be an alias or an IP, and overwriting with it would make
        # every merged row claim a provenance that is not a hostname. Verified for this campaign:
        # every qb2 record already carries host="tt-quietbox2".
        if not rec.get("host"):
            rec["host"] = a.peer
        new.append(rec)
    if a.dry_run:
        print(f"[progress] would append {len(new)} records to {prog}")
    else:
        with open(prog, "a") as f:
            for rec in new:
                f.write(json.dumps(rec) + "\n")
        print(f"[progress] appended {len(new)} records to {prog}")

    by_gen = Counter(g for (_t, g) in incoming)
    print("merged by generator:", dict(by_gen) or "nothing")
    total = len(local) + len(incoming)
    print(f"union ok pairs after merge: {total} / {164 * 3}")


if __name__ == "__main__":
    main()
