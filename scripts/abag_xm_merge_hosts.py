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
# generator -> (output subdirectory, the result-dir prefix inside it, label-file prefix)
GENS = {
    "protenix-v2":  ("protenix_v2", "protenix_results", "protenix_v2"),
    "opendde-abag": ("opendde_abag", "opendde_results", "opendde_abag"),
    "boltz2":       ("boltz2", "boltz2_results", "boltz2"),
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
