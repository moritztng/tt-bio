#!/usr/bin/env bash
# One instrumented fold-and-exit on card 0, with the address-space teardown measured.
#
#   TAG=control ARENAS=0 TRIM=0 bash perf/qb_livelock/fold_once.sh
#   TAG=arena    ARENAS=2 TRIM=0 bash perf/qb_livelock/fold_once.sh
#   TAG=trim     ARENAS=0 TRIM=1 bash perf/qb_livelock/fold_once.sh
#   TAG=both     ARENAS=2 TRIM=1 bash perf/qb_livelock/fold_once.sh
#
# Every arm must run in the same session as the control: the kernel's ladder is cumulative per
# boot and depends on what else the box is doing, so a before/after across days proves nothing.
set -uo pipefail

TAG=${TAG:?set TAG}
ARENAS=${ARENAS:-2}
TRIM=${TRIM:-1}
INPUT=${INPUT:-perf/qb_livelock/prot512.yaml}
MODEL=${MODEL:-boltz2}
CARD=${CARD:-0}

WT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$WT" || exit 1
OUT=$WT/perf/qb_livelock/out/$TAG
rm -rf "$OUT"; mkdir -p "$OUT"

# bpftrace runs as root, so signal it by explicit pid: exec replaces the shell, leaving the
# bpftrace pid in the file.
sudo -n sh -c "echo \$\$ > $OUT/bt.pid; exec bpftrace $WT/perf/qb_livelock/mm_teardown.bt" \
     > "$OUT/probe.txt" 2>&1 &
for _ in $(seq 150); do grep -q MMPROBE-START "$OUT/probe.txt" 2>/dev/null && break; sleep 0.2; done
grep -q MMPROBE-START "$OUT/probe.txt" || { echo "probe failed to start"; cat "$OUT/probe.txt"; exit 1; }

ladder() { journalctl -k -b --no-pager 2>/dev/null | grep -c "mmput_async_fn hogged"; }
last_rung() { journalctl -k -b --no-pager 2>/dev/null | grep "mmput_async_fn hogged" \
              | tail -1 | sed -E 's/.*>10000us ([0-9]+) times.*/\1/'; }

# the venv is only there for the installed deps: assert the tt_bio being folded is THIS worktree,
# not the shared checkout's installed package.
PYBIN=${PYBIN:-/home/ttuser/tt-bio-dev/env/bin/python3}
SRC=$("$PYBIN" -c "import tt_bio, os; print(os.path.dirname(tt_bio.__file__))")
case "$SRC" in "$WT"/*) ;; *) echo "wrong tt_bio: $SRC (want under $WT)"; exit 1;; esac

L_BEFORE=$(ladder); R_BEFORE=$(last_rung)
T0=$(date +%s.%N)

TT_VISIBLE_DEVICES=$CARD \
TT_BIO_LEASE_CARDS=$CARD \
TT_BIO_LEASE_HOLDER=worker:qb-livelock-mm-teardown \
TT_BIO_MALLOC_ARENAS=$ARENAS \
TT_BIO_EXIT_TRIM=$TRIM \
TT_BIO_EXIT_FOOTPRINT=$OUT/footprint.txt \
TT_BIO_EXIT_TAG=$TAG \
    "$PYBIN" -m tt_bio.main predict "$INPUT" \
    --model "$MODEL" --out_dir "$OUT/fold" --override > "$OUT/fold.log" 2>&1
RC=$?

T1=$(date +%s.%N)
sleep 3                                  # let a deferred teardown land before we stop looking
L_AFTER=$(ladder); R_AFTER=$(last_rung)
sudo -n kill -INT "$(cat "$OUT/bt.pid")" 2>/dev/null
for _ in $(seq 100); do pgrep -x bpftrace >/dev/null || break; sleep 0.2; done
wait 2>/dev/null

{
  echo "tag=$TAG arenas=$ARENAS trim=$TRIM rc=$RC"
  echo "wall_s=$(echo "$T1 - $T0" | bc)"
  echo "ladder_reports=${L_BEFORE:-0}->${L_AFTER:-0}  last_rung=${R_BEFORE:-none}->${R_AFTER:-none}"
  echo "footprint(python hook): $(cat "$OUT/footprint.txt" 2>/dev/null | tr '\n' ';')"
  echo "cif_md5=$(find "$OUT/fold" -name '*.cif' -print0 2>/dev/null | sort -z | xargs -0 md5sum 2>/dev/null | awk '{print $1}' | tr '\n' ' ')"
} > "$OUT/summary.txt"
cat "$OUT/summary.txt"
echo "--- teardowns of this fold's own pids ---"
grep EXITMMAP "$OUT/probe.txt" | sort -t= -k2 -rn | head -5
