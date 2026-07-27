"""Campaign status across BOTH hosts.

`abag_xm_status.py` reads one host's `progress.jsonl`, and that file is host-local by design --
the two hosts fold disjoint slices and never share it. The consequence is that on a two-host
campaign the single-host view understates progress by whatever the other host has done, with no
hint that it is doing so. On 2026-07-28 qb1 reported 44 of 492 pairs done while the true figure
was 84: qb2 held 40 `ok` pairs that qb1 could not see.

This also surfaces the one thing neither host can detect alone: pairs folded on BOTH hosts. That
happens legitimately -- when one host is down the other may take over its slices -- and it is
waste, not corruption (same seed, same code, same MSA), but the merge must dedupe on
(target, model) and it is worth knowing how much was spent twice.

Usage: python3 scripts/abag_xm_status_xhost.py [other_host]     default tt-quietbox2 (or
       tt-quietbox when run there). Reads the local file directly and the remote one over ssh.
"""
import collections
import json
import pathlib
import socket
import subprocess
import sys

TARGETS = 164
MODELS = ("protenix-v2", "opendde-abag", "boltz2")
PROGRESS = "abag_xm/tier_a/progress.jsonl"
HOSTS = ("tt-quietbox", "tt-quietbox2")


def _ok_pairs(lines, where):
    pairs, n = set(), 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        n += 1
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            print(f"  !! {where}: skipped an unparseable record", file=sys.stderr)
            continue
        if r.get("status") == "ok":
            pairs.add((r["target"], r["model"]))
    return pairs, n


def main():
    me = socket.gethostname()
    other = sys.argv[1] if len(sys.argv) > 1 else next(h for h in HOSTS if h != me)

    local = pathlib.Path.home() / PROGRESS
    mine, n_mine = _ok_pairs(local.read_text().splitlines(), me)

    # A hung or powered-off peer is the normal reason this campaign goes single-host, so an
    # unreachable peer must degrade to a clearly-labelled partial view rather than abort.
    got_peer = True
    try:
        out = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", f"ttuser@{other}",
             f"cat {PROGRESS}"],
            capture_output=True, text=True, timeout=60)
        if out.returncode != 0:
            raise RuntimeError(out.stderr.strip()[:200] or f"rc={out.returncode}")
        theirs, n_theirs = _ok_pairs(out.stdout.splitlines(), other)
    except Exception as e:
        got_peer, theirs, n_theirs = False, set(), 0
        print(f"!! {other} unreachable ({e}) -- numbers below are {me} ONLY, not the campaign\n")

    union, overlap = mine | theirs, mine & theirs
    print(f"records on disk : {me}={n_mine}  {other}={n_theirs}")
    print(f"ok pairs        : {me}={len(mine)}  {other}={len(theirs)}"
          f"  UNION={len(union)} / {TARGETS * len(MODELS)}")
    if overlap:
        print(f"duplicated pairs: {len(overlap)}  (folded on BOTH hosts -- wasted, not wrong;"
              f" dedupe on (target, model) at merge)")
        for t, m in sorted(overlap)[:12]:
            print(f"    dup: {t} {m}")
        if len(overlap) > 12:
            print(f"    ... and {len(overlap) - 12} more")
    print()
    for m in MODELS:
        ok = sum(1 for (_, mm) in union if mm == m)
        print(f"  {m:<14} ok={ok:3d}  outstanding={TARGETS - ok:3d}")
    print()
    label = "TRUE OUTSTANDING (union)" if got_peer else f"OUTSTANDING ({me} only -- INCOMPLETE)"
    print(f"{label}: {TARGETS * len(MODELS) - len(union)} of {TARGETS * len(MODELS)}")


if __name__ == "__main__":
    main()
