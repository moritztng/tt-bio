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
PY_PROBE=${PY:-/home/ttuser/tt-bio-dev/env/bin/python3}
[ -x "$PY_PROBE" ] || PY_PROBE=$(command -v python3)
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$WT/perf/c12_diffusion_head"
SCRATCH="$(mktemp -d)"
trap 'rm -rf "$SCRATCH"' EXIT

say() { printf '%s %s\n' "$(date -u +%H:%M:%SZ)" "$*"; }
die() { say "REFUSED: $*"; exit 1; }

# --- LEASE ------------------------------------------------------------------------------------
[ "$(hostname)" = "tt-quietbox2" ] || die "this runs on qb2, not $(hostname)"
# The grant lease is written by the fleet dispatcher, which runs on pc -- `qb2-card3.json` only
# ever exists in PC's state dir, never in qb2's. Checking `$CO/state/leases/` here always failed
# (qb2's own dir uses the `tt-quietbox2-cardN.json` naming for tt_bio's device-open lease, which is
# a different file with a different writer). So read the grant from the host that writes it.
DISPATCHER=${DISPATCHER:-moritz@pc}
LEASE_REMOTE=/home/moritz/.coworker/state/leases/qb2-card${CARD}.json
LEASE_TXT=$(timeout 25 ssh -o BatchMode=yes -o ConnectTimeout=8 "$DISPATCHER" "cat $LEASE_REMOTE" 2>/dev/null) \
    || die "cannot read the grant lease $LEASE_REMOTE on $DISPATCHER -- without it the fleet reads card $CARD as free and another row can land on it"
[ -n "$LEASE_TXT" ] || die "$LEASE_REMOTE on $DISPATCHER is empty -- no grant"
case "$LEASE_TXT" in
    *"$SLUG"*) ;;
    *) die "$LEASE_REMOTE is held by someone else: $LEASE_TXT" ;;
esac
say "grant lease ok ($DISPATCHER): $LEASE_TXT"

# tt_bio's OWN device-open lease on this host must not be live-held by a different row. A released
# entry (non-null "released") or our own holder is fine; anyone else live is a refusal.
OWN_LEASE="$CO/state/leases/$(hostname)-card${CARD}.json"
if [ -f "$OWN_LEASE" ]; then
    "$PY_PROBE" - "$OWN_LEASE" "$SLUG" <<'PYL' || die "$OWN_LEASE is live-held by another row: $(cat "$OWN_LEASE")"
import json, sys
rec = json.load(open(sys.argv[1]))
if rec.get("released") is None and sys.argv[2] not in str(rec.get("holder", "")):
    raise SystemExit(1)
PYL
    say "device-open lease clear: $(cat "$OWN_LEASE")"
fi

# --- the two canonical guards -----------------------------------------------------------------
git -C "$WT" fetch -q origin 'refs/heads/wk/*:refs/remotes/origin/wk/*' || say "WARN: fetch failed, using what is local"
PAIR_REF=origin/wk/c12-orchestrator:perf/c12_orchestrator/pair_guard/pair_idle.py
CLK_REF=origin/wk/c12-profiled-fold:perf/c12_profiled_fold/clk.py
git -C "$WT" show "$PAIR_REF" > "$SCRATCH/pair_idle.py" || die "cannot read $PAIR_REF"
git -C "$WT" show "$CLK_REF"  > "$SCRATCH/clk.py"       || die "cannot read $CLK_REF"
PAIR_SHA=$(git -C "$WT" rev-parse "${PAIR_REF%%:*}")
CLK_SHA=$(git -C "$WT" rev-parse "${CLK_REF%%:*}")
say "guards: pair_idle @ $PAIR_SHA, clk @ $CLK_SHA"

# /home/ttuser/tt-bio/env does not exist on qb2; the venv with ttnn is tt-bio-dev/env. The old
# fallback `command -v python3` resolved /usr/bin/python3, which has NO ttnn -- that spends the
# card window to reach an import error. Refuse instead.
PY=${PY:-/home/ttuser/tt-bio-dev/env/bin/python3}
[ -x "$PY" ] || die "no usable python at $PY -- set PY=<venv python with ttnn>"
"$PY" -c 'import ttnn' >/dev/null 2>&1 || die "$PY cannot import ttnn -- wrong interpreter, do not burn the card window"
say "python: $PY (ttnn importable)"
# tt_bio must resolve to THIS worktree, not the shared /home/ttuser/tt-bio-dev checkout, whose
# qkv_heads still has the old 6-arg signature. ab_qkv.py also inserts the root on sys.path; this
# is the belt-and-braces half and it makes the intent visible in the env of every child.
export PYTHONPATH="$WT${PYTHONPATH:+:$PYTHONPATH}"
say "PYTHONPATH=$WT ($("$PY" -c 'import tt_bio;print(tt_bio.__file__)'))"

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

# PREFLIGHT=1 exercises everything above -- grant lease, device-open lease, interpreter, import
# path, both chips twice -- and stops before the clock is forced. `bash -n` passes a broken guard
# silently (it already hid a two-line message that parsed as its own command here), so the guards
# get executed before the card window is spent, not just parsed.
if [ "${PREFLIGHT:-0}" = "1" ]; then
    say "PREFLIGHT ok: lease, interpreter, import path and both chips of board 410D all check out"
    exit 0
fi

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
# The window the clock has to hold. clk.py's FIRST sample is taken in the same instant it posts
# the force, so it still reads the pre-force 800 MHz (the header records it as "before": 800).
# Qualifying the whole file therefore voided a session whose clock was pinned for every sample
# that mattered. The brief's rule is "sampled DURING the fold", so bound the window explicitly.
T_START=$(date +%s.%N)
set +e
TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:$SLUG \
    TT_BIO_APB_HEAD_MAJOR_QKV=1 TT_BIO_APB_ATOM_HEAD_MAJOR_QKV=1 \
    "$PY" "$OUT/ab_qkv.py" --reps "${REPS:-7}" --device-id 0 ${ONLY:+--only "$ONLY"} \
    --out "ab_qkv_qb2c${CARD}${ONLY:+_$ONLY}.json" 2>&1 \
    | tee "$OUT/ab_qkv_qb2c${CARD}${ONLY:+_$ONLY}.log"
RC=${PIPESTATUS[0]}
set -e
T_END=$(date +%s.%N)
kill -TERM "$CLK_PID" 2>/dev/null || true
wait "$CLK_PID" 2>/dev/null || true
say "ab_qkv rc=$RC"

# --- AFTER ------------------------------------------------------------------------------------
pair_check "after"

# --- the clock has to qualify, or the numbers are void ----------------------------------------
"$PY" - "$CLOCK_JSONL" "$CARD" "$FLOOR_MHZ" "$T_START" "$T_END" <<'PYQ'
import json, sys
path, node, floor = sys.argv[1], sys.argv[2], float(sys.argv[3])
t0, t1 = float(sys.argv[4]), float(sys.argv[5])
allv, win = [], []
for line in open(path):
    line = line.strip()
    if not line:
        continue
    try:
        rec = json.loads(line)
    except ValueError:
        continue
    v = rec.get(node)
    if not isinstance(v, (int, float)):
        continue          # a read error is a string; it is reported below, never silently dropped
    allv.append(float(v))
    t = rec.get("t")
    if isinstance(t, (int, float)) and t0 <= float(t) <= t1:
        win.append(float(v))
bad = [1 for line in open(path) if '"ERR:' in line]
if not allv:
    print(f"CLOCK: no samples for node {node} in {path} -- the session is NOT admissible")
    raise SystemExit(1)
if not win:
    print(f"CLOCK: {len(allv)} samples in the file but NONE inside the measurement window "
          f"[{t0:.3f}, {t1:.3f}] -- cannot qualify this session")
    raise SystemExit(1)
if bad:
    print(f"CLOCK: {len(bad)} sysfs read errors in {path} -- the session is NOT admissible")
    raise SystemExit(1)
def stat(v):
    v = sorted(v)
    return v[0], v[len(v)//2], v[-1]
alo, amed, ahi = stat(allv)
lo, med, hi = stat(win)
print(f"CLOCK: whole file {len(allv)} samples, min {alo:.0f} / median {amed:.0f} / max {ahi:.0f} MHz")
print(f"CLOCK: measurement window {len(win)} samples, min {lo:.0f} / median {med:.0f} / "
      f"max {hi:.0f} MHz")
if lo < floor:
    print(f"CLOCK: a sample INSIDE the measurement read {lo:.0f} MHz, under the {floor:.0f} MHz "
          "floor -- this session is an ARTIFACT, not a regression. Discard it and say so.")
    raise SystemExit(1)
print(f"CLOCK: qualified -- every one of {len(win)} in-window samples held >= {floor:.0f} MHz")
PYQ

# --- STAGE 2: the fold-level arm set, with the STACK checked on its own ------------------------
# C12 closed with the lesson that two individually-correct levers can compose into a silently wrong
# transform that neither owning row can detect, and that the broken combination can be the FASTEST
# arm. This row ships two flags, so the stack gets its own end-to-end reading rather than an
# inference from the two singles. Skipped if stage 1 failed: a broken op-level arm makes the fold
# uninterpretable.
if [ "$RC" -eq 0 ] && [ "${SKIP_FOLD:-0}" != "1" ]; then
    say "stage 2: fold arms off / token / atom / both, interleaved, ${FOLD_REPS:-3} reps"
    "$PY" "$SCRATCH/clk.py" --nodes "$CARD" --target "$TARGET_MHZ" --period-ms 50 \
        --out "$OUT/clock_fold_qb2c${CARD}.jsonl" &
    CLK2=$!
    sleep 5
    set +e
    TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:$SLUG \
        "$PY" "$OUT/fold_stack.py" --reps "${FOLD_REPS:-3}" --py "$PY" \
        --out "fold_stack_qb2c${CARD}.json" 2>&1 | tee "$OUT/fold_stack_qb2c${CARD}.log"
    FOLD_RC=${PIPESTATUS[0]}
    set -e
    kill -TERM "$CLK2" 2>/dev/null || true
    wait "$CLK2" 2>/dev/null || true
    say "fold_stack rc=$FOLD_RC"
    pair_check "after fold"
else
    say "stage 2 SKIPPED (stage 1 rc=$RC, SKIP_FOLD=${SKIP_FOLD:-0}) -- no fold-level reading"
    FOLD_RC=skipped
fi

say "SESSION ADMISSIBLE. artifacts:"
ls -la "$OUT"/ab_qkv_qb2c${CARD}${ONLY:+_$ONLY}.json "$OUT"/ab_qkv_qb2c${CARD}${ONLY:+_$ONLY}.log "$CLOCK_JSONL" \
       "$OUT"/fold_stack_qb2c${CARD}.json "$OUT"/fold_stack_qb2c${CARD}.log \
       "$OUT"/clock_fold_qb2c${CARD}.jsonl 2>/dev/null || true
cat <<EOF

qb2 cannot push to origin. Copy the three artifacts back to pc and commit there:

  scp -r $(hostname):$OUT/{ab_qkv_qb2c${CARD}.*,fold_stack_qb2c${CARD}.*,clock_*qb2c${CARD}.jsonl} \\
      pc:/home/moritz/.coworker/wt/$SLUG/perf/c12_diffusion_head/

The state doc must be written on pc: that is the host DONE_CHECK reads.
EOF
exit $RC
