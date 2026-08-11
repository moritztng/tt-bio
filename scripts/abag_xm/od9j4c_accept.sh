#!/bin/bash
# Acceptance gate for the opendde-abag 9j4c cell. Run on the Galaxy. Exit 0 only when all eight
# chunks have a record at rc=0, cifs=64, distinct=64. Prints the per-chunk table either way.
#
# This exists because the completion signal in p34d/od9j4c/results.jsonl is NOT trustworthy. Two
# runners write into that file and each writes its own terminal marker when ITS OWN chunk set
# drains: od9j4c_fleet.sh holds c2/c3/c4 and writes OD9J4C_DONE, od9j4c_r2_fleet.sh holds
# c1/c5/c6/c7 and writes OD9J4C_R2_DONE. So OD9J4C_DONE lands with three chunks folded and four
# still running, and the pass-14 runbook said to wait for exactly that marker. A marker means one
# runner stopped, never that the cell is complete.
#
# The other trap it closes: rc=0 does not mean success here. An OOM in diffusion exits clean and
# writes results.json saying "0 ok, 1 failed", so the campaign's silent-failure shape is
# rc=0 cifs=0 oom=1. Only the cif count separates it from a real fold. A chunk may also carry
# several records as the runner narrows mps 5 -> 2 -> 1; the last record for a chunk is the one
# that counts.
set -u
B=${B:-$HOME/mthuening/p34d/od9j4c}
J=$B/results.jsonl
WANT=${WANT:-8}

[ -r "$J" ] || { echo "no ledger at $J" >&2; exit 2; }

python3 - "$J" "$WANT" <<'PY'
import json, sys
path, want = sys.argv[1], int(sys.argv[2])
last = {}
for line in open(path):
    line = line.strip()
    if not line.startswith('{'):
        continue                      # terminal markers and other noise
    r = json.loads(line)
    if r.get('target') != '9j4c' or r.get('model') != 'opendde-abag':
        continue
    last[r['chunk']] = r             # later record wins: the mps ladder rewrites a chunk

ok = 0
print(f"{'chunk':>5} {'mps':>3} {'umd':>3} {'rc':>4} {'sec':>6} {'cifs':>4} {'dist':>4} {'oom':>3}  verdict")
for c in range(want):
    r = last.get(c)
    if r is None:
        print(f"{c:>5} {'-':>3} {'-':>3} {'-':>4} {'-':>6} {'-':>4} {'-':>4} {'-':>3}  NO RECORD")
        continue
    good = r['rc'] == 0 and r.get('cifs') == 64 and r.get('distinct') == 64 and not r.get('oom')
    ok += good
    print(f"{c:>5} {str(r.get('mps')):>3} {r.get('umd'):>3} {r['rc']:>4} {r.get('seconds'):>6} "
          f"{r.get('cifs'):>4} {r.get('distinct'):>4} {r.get('oom'):>3}  {'ACCEPT' if good else 'REJECT'}")

print(f"\naccepted {ok}/{want}")
sys.exit(0 if ok == want else 1)
PY
