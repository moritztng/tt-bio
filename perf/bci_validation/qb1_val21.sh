#!/bin/bash
# Issue #21 on qb1 card 2 (logical 2 = /dev/tenstorrent/3 = sysfs tenstorrent!3), three arms in
# one process each, one after the other under one flock. Each arm opens the card for its design
# fold, then scores ten ProteinMPNN candidates through the host-JAX validation model.
#
#   prefix   ~/bci_prefix_int  (wk/bci-integration with the #21 fix reverted)
#   fixed    ~/bci_int         (wk/bci-integration)
#   control  ~/bci_prefix_int --no-extra-msa
#
# CHIPS.md rules: own flock, nice 10, timeout on every device command, SIGINT before SIGTERM,
# never tt-smi -r.
set -u
CARD=2
LOCK=/home/ttuser/bci_qb1_card2.lock
OUT=/home/ttuser/bci_val21
BC2=/home/ttuser/bcx_e2e/bc2
PY=/home/ttuser/bcx_e2e_venv/bin/python
mkdir -p "$OUT"

exec 9>"$LOCK"
echo "=== waiting for $LOCK $(date -u +%FT%TZ)"
flock 9 || exit 1
echo "=== holding card $CARD $(date -u +%FT%TZ)"

run() {
  local arm=$1 tree=$2; shift 2
  local script=$tree/perf/bci_validation/card_validation_arm.py
  echo "=== arm $arm tree $tree $(date -u +%FT%TZ)"
  TT_VISIBLE_DEVICES=$CARD PYTHONPATH=$BC2:$tree \
    timeout -s INT -k 300 90m nice -n 10 "$PY" "$script" \
      --arm "$arm" --ttbio "$tree" --out "$OUT/$arm" "$@" \
      > "$OUT/$arm.log" 2>&1
  echo "=== arm $arm rc=$? $(date -u +%FT%TZ)"
}

run prefix  /home/ttuser/bci_prefix_int
run fixed   /home/ttuser/bci_int
run control /home/ttuser/bci_prefix_int --no-extra-msa
echo "=== done $(date -u +%FT%TZ)"
