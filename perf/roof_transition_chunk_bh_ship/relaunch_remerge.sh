#!/usr/bin/env bash
# Resume the two device legs of the re-verify after a qb2 reboot. Gate on cards 0+1, the missing
# 1024 aa ladder rung on card 2. PYTHONPATH pins the scorer subprocesses to THIS worktree, not the
# shared checkout (parity-gate-scores-installed-package-not-checkout).
set -u
WT=/home/ttuser/.coworker/wt/roof-transition-chunk-remerge-verify
PY=/home/ttuser/tt-bio-dev/env/bin/python3
cd "$WT" || exit 1
O=perf/roof_transition_chunk_bh_ship/out
setsid nohup env PYTHONPATH="$WT" TT_BIO_LEASE_CARDS=0,1 \
  TT_BIO_LEASE_HOLDER=worker:roof-transition-chunk-remerge-verify \
  OPENDDE_DOCKQ_PYTHON=/home/ttuser/dockqenv/bin/python3 \
  "$PY" scripts/full_parity_gate.py \
  --workers tt-quietbox2:0,tt-quietbox2:1 \
  --workdir "$WT/perf/roof_transition_chunk_bh_ship/gate-b3c62ce5a" \
  --out "$WT/$O/gate_remerge.json" \
  >> "$WT/$O/gate_remerge.log" 2>&1 < /dev/null &
echo "gate pid $!"
setsid nohup env PYTHONPATH="$WT" TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 \
  TT_BIO_LEASE_HOLDER=worker:roof-transition-chunk-remerge-verify \
  "$PY" perf/b2z2_size_ladder/ladder.py \
  --levers transition_l1 --arms off,on --sizes 1024 \
  --out "$WT/$O/ladder_remerge_1024_c2.json" \
  --cifdir "$WT/$O/cif_ladder_remerge_1024_c2" \
  >> "$WT/$O/ladder_remerge_1024_c2.log" 2>&1 < /dev/null &
echo "ladder pid $!"
