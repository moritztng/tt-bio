#!/usr/bin/env bash
# Wait for s6's unit to go inactive, then take card 1 for the pairwise correctness diagnostic.
# Correctness, not timing: it reads digests and plDDT, which a co-tenant cannot corrupt.
set -u
H=/home/ttuser/.coworker/wt/c12-compose-fold/perf/c12_compose
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
for _ in $(seq 1 240); do
  systemctl --user is-active c12s6 >/dev/null 2>&1 || break
  sleep 5
done
sleep 10
echo "s6 done at $(date -u +%FT%TZ), starting diagnostic"
TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 TT_BIO_LEASE_HOLDER=worker:c12-compose-fold \
  /home/ttuser/scratch/i14venv/bin/python3 "$H/fold_compose.py" \
  --out "$H/out/diag.json" --cifs "$H/out/diag_cifs" \
  --arms base,hoist_elt,silu_elt,base --reps 2 --cold-reps 0 --size 512
echo "diagnostic exit $?"
