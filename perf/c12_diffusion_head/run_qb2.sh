#!/usr/bin/env bash
# One command for the device half of c12-diffusion-head-major. Runs ON qb2, under the qb2-card3
# lease, and refuses to produce a number unless the session is admissible.
#
# Six passes of this row were dispatched card=cpu, so everything that needed no chip is already
# done and checked. This script exists so the card window is spent measuring rather than on setup,
# and so the three rules that bit other C12 rows are enforced by the runner instead of remembered:
#
#   LEASE   refuse unless we are on qb2 AND state/leases/qb2-card3.json names this row. The fleet's
#           remote_busy_cards() reads only lease files, so running without one lets another row
#           land on the same chip -- this campaign already lost a fold session to exactly that
#           (both device rows pinned card=3).
#   PAIR    board ...410D is dev2+dev3, so card 3's sibling is card 2 (sibling = card XOR 1). Both
#           chips are checked BEFORE and AFTER, twice each: `tt-smi -r <n>` holds an fd on all four
#           chips for ~41 s, so ONE dirty sample can be a reset on the other pair rather than a
#           busy sibling.
#   CLOCK   the AICLK sets the fold time on this part, so clk.py forces 1350 MHz and samples it
#           for the whole measurement. A sample below the floor voids the session; a number without
#           a qualified clock is not a measurement.
#
# The two guards are taken from their canonical refs rather than vendored, and the SHA is recorded.
set -euo pipefail

SLUG=c12-diffusion-head-major
CARD=3
SIBLING=2
TARGET_MHZ=1350
FLOOR_MHZ=1200
CO=/home/ttuser/.coworker
[ -d "$CO" ] || CO=/home/moritz/.coworker
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$WT/perf/c12_diffusion_head"
SCRATCH="$(mktemp -d)"
trap 'rm -rf "$SCRATCH"' EXIT

say() { printf '%s %s\n' "$(date -u +%H:%M:%SZ)" "$*"; }
die() { say "REFUSED: $*"; exit 1; }

# --- LEASE ------------------------------------------------------------------------------------
[ "$(hostname)" = "tt-quietbox2" ] || die "this runs on qb2, not $(hostname)"
LEASE="$CO/state/leases/qb2-card${CARD}.json"
[ -f "$LEASE" ] || die "no $LEASE -- the fleet still reads card $CARD as free, so another row can land on it"
grep -q "$SLUG" "$LEASE" || die "$LEASE is held by someone else: $(cat "$LEASE")"
say "lease ok: $(cat "$LEASE")"

# --- the two canonical guards -----------------------------------------------------------------
git -C "$WT" fetch -q origin 'refs/heads/wk/*:refs/remotes/origin/wk/*' || say "WARN: fetch failed, using what is local"
PAIR_REF=origin/wk/c12-orchestrator:perf/c12_orchestrator/pair_guard/pair_idle.py
CLK_REF=origin/wk/c12-profiled-fold:perf/c12_profiled_fold/clk.py
git -C "$WT" show "$PAIR_REF" > "$SCRATCH/pair_idle.py" || die "cannot read $PAIR_REF"
git -C "$WT" show "$CLK_REF"  > "$SCRATCH/clk.py"       || die "cannot read $CLK_REF"
PAIR_SHA=$(git -C "$WT" rev-parse "${PAIR_REF%%:*}")
CLK_SHA=$(git -C "$WT" rev-parse "${CLK_REF%%:*}")
say "guards: pair_idle @ $PAIR_SHA, clk @ $CLK_SHA"

PY=${PY:-/home/ttuser/tt-bio/env/bin/python3}
[ -x "$PY" ] || PY=$(command -v python3)
say "python: $PY ($("$PY" -c 'import ttnn;print("ttnn", getattr(ttnn,"__version__","?"))' 2>/dev/null || echo 'ttnn NOT importable'))"

# pair_idle exits 0 idle, 1 SIBLING BUSY, 2 usage/environment. Those are different findings and
# must not be collapsed: reporting a missing device node as "busy sibling" sends the next reader
# hunting a phantom fold.
one_pair_probe() {   # $1 = card whose SIBLING to check, $2 = label
    local card="$1" when="$2" rc=0
    "$PY" "$SCRATCH/pair_idle.py" --card "$card" || rc=$?
    case "$rc" in
        0) return 0 ;;
        1) die "$when: the sibling of card $card is BUSY -- do not measure" ;;
        *) die "$when: pair_idle could not check card $card (exit $rc: missing node, or fuser needs sudo -n here). Not a busy sibling, and not a green light either." ;;
    esac
}

pair_check() {   # $1 = label
    local when="$1" i
    for i in 1 2; do
        one_pair_probe "$CARD" "$when sample $i"
        one_pair_probe "$SIBLING" "$when sample $i"
        sleep 3
    done
    say "$when: both chips of board 410D idle on 2 samples each"
}

# --- BEFORE -----------------------------------------------------------------------------------
pair_check "before"

# --- CLOCK, held for the whole measurement ----------------------------------------------------
CLOCK_JSONL="$OUT/clock_qb2c${CARD}.jsonl"
"$PY" "$SCRATCH/clk.py" --nodes "$CARD" --target "$TARGET_MHZ" --period-ms 50 \
    --out "$CLOCK_JSONL" &
CLK_PID=$!
trap 'kill -TERM '"$CLK_PID"' 2>/dev/null || true; rm -rf "$SCRATCH"' EXIT
sleep 5
kill -0 "$CLK_PID" 2>/dev/null || die "clk.py died immediately -- no forced clock, so no measurement"
say "AICLK forced to ${TARGET_MHZ} MHz on node $CARD (clk pid $CLK_PID), sampling to $(basename "$CLOCK_JSONL")"

# --- THE MEASUREMENT --------------------------------------------------------------------------
say "running ab_qkv.py: A0 linear+split / A1 mm+split / B head-major / A/A, interleaved"
set +e
TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD \
    TT_BIO_APB_HEAD_MAJOR_QKV=1 TT_BIO_APB_ATOM_HEAD_MAJOR_QKV=1 \
    "$PY" "$OUT/ab_qkv.py" --reps "${REPS:-7}" --device-id 0 \
    --out "ab_qkv_qb2c${CARD}.json" 2>&1 | tee "$OUT/ab_qkv_qb2c${CARD}.log"
RC=${PIPESTATUS[0]}
set -e
kill -TERM "$CLK_PID" 2>/dev/null || true
wait "$CLK_PID" 2>/dev/null || true
say "ab_qkv rc=$RC"

# --- AFTER ------------------------------------------------------------------------------------
pair_check "after"

# --- the clock has to qualify, or the numbers are void ----------------------------------------
"$PY" - "$CLOCK_JSONL" "$CARD" "$FLOOR_MHZ" <<'PYQ'
import json, sys
path, node, floor = sys.argv[1], sys.argv[2], float(sys.argv[3])
vals = []
for line in open(path):
    line = line.strip()
    if not line:
        continue
    try:
        rec = json.loads(line)
    except ValueError:
        continue
    if node in rec and isinstance(rec[node], (int, float)):
        vals.append(float(rec[node]))
if not vals:
    print(f"CLOCK: no samples for node {node} in {path} -- the session is NOT admissible")
    raise SystemExit(1)
vals.sort()
lo, med, hi = vals[0], vals[len(vals)//2], vals[-1]
print(f"CLOCK: {len(vals)} samples, min {lo:.0f} / median {med:.0f} / max {hi:.0f} MHz")
if lo < floor:
    print(f"CLOCK: a sample read {lo:.0f} MHz, under the {floor:.0f} MHz floor -- this session is an "
          "ARTIFACT, not a regression. Discard it and say so.")
    raise SystemExit(1)
print("CLOCK: qualified for the whole measurement")
PYQ

say "SESSION ADMISSIBLE. artifacts:"
ls -la "$OUT"/ab_qkv_qb2c${CARD}.json "$OUT"/ab_qkv_qb2c${CARD}.log "$CLOCK_JSONL" 2>/dev/null || true
cat <<EOF

qb2 cannot push to origin. Copy the three artifacts back to pc and commit there:

  scp $(hostname):$OUT/{ab_qkv_qb2c${CARD}.json,ab_qkv_qb2c${CARD}.log,$(basename "$CLOCK_JSONL")} \\
      pc:/home/moritz/.coworker/wt/$SLUG/perf/c12_diffusion_head/

The state doc must be written on pc: that is the host DONE_CHECK reads.
EOF
exit $RC
