#!/bin/bash
# round_re and the ceiling ladder, with the three levers the arm actually runs DECLARED.
#
# Why this exists: sitting3's round died in 10 s with `lever went inert mid-arm`, and the ladder
# then read that rc=1 as a MEMORY REFUSAL at its first rung and announced a ceiling. Two separate
# defects, and the second is the dangerous one.
#
#  1. DIFFS 5, recurring. `--exact 0` means fast=True, and `bindcraft2.fast_round` arms
#     MM_LAYOUT, TAPED_CHANNEL_MOVE and WIDEN_ADD for itself. `perf/bcx_round/meter.py:183` then
#     sees the levers it was TOLD to expect (all off) disagree with the levers the engine is
#     running and raises. The three have to be declared in the environment, which the earlier
#     round sittings did and neither ceiling.sh nor its launcher ever did -- the ladder had never
#     been run, so nobody had found it.
#  2. `ceiling.sh` treats any non-zero rc that is not 124 as a memory refusal. A lever error, an
#     import error and a missing param file all land in that branch, and the ladder then reports
#     "the rung below is the ceiling" off a failure that has nothing to do with memory. The rungs
#     here are guarded by a PRE-FLIGHT at a size already known to run, so a configuration fault
#     cannot be mistaken for a wall.
set -uo pipefail
cd ~/bwx
OUT=~/bwx/out/sitting4
CHIP=${BWX_CHIP:-30}
PY=~/bwx/venv/bin/python
mkdir -p "$OUT"
say(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/sitting.log"; }
. ~/japanfold/env.sh
export TT_METAL_CACHE=$HOME/bwx/cache/tt-metal PYTHONPATH=$HOME/bwx/tt-bio BCX_BC2=$HOME/bwx/bc2
export JAX_COMPILATION_CACHE_DIR=$HOME/bwx/cache/xla
export TT_VISIBLE_DEVICES=$CHIP TT_BIO_LEASE_CARDS=$CHIP
export TT_BIO_LEASE_HOLDER=worker:bwx-bringup TT_BIO_LEASE_TIMEOUT=28800
# the set bindcraft2._FAST_ROUND arms for itself; declaring them is what keeps meter.py quiet
export TT_BIO_MM_LAYOUT=1 TT_BIO_TAPED_CHANNEL_MOVE=1 TT_BIO_WIDEN_ADD=1
ARM="--exact 0 --triatt-hifi 1 --rne-kernel 1 --triatt-bw 1 --extra-msa 1 --template 1"

say "sitting4 start chip=$CHIP levers declared: MM_LAYOUT TAPED_CHANNEL_MOVE WIDEN_ADD"

# --- round_re: also the pre-flight. 288 tokens is known to run on this chip. -----------------
d="$OUT/round_re"; mkdir -p "$d"
say "round_re start binder=146 (288 tokens) arm=(1,1,hifi)"
timeout 1800 $PY -u ~/bwx/tt-bio/perf/bcx_round/run_round.py \
    --rounds 4 --binder 146 --params $HOME/bwx/af2_params --out "$d" $ARM > "$d/round.log" 2>&1
rc=$?; say "round_re rc=$rc"
grep -E "^round |median|s a round|lever went inert" "$d/round.log" | tail -6 | tee -a "$OUT/sitting.log"
if [ $rc -ne 0 ]; then
  say "PRE-FLIGHT FAILED at a size known to run, so the configuration is wrong, not the card."
  say "NOT running the ladder: every rung would fail the same way and the first would be"
  say "misreported as the memory ceiling. Fix the configuration first."
  exit 1
fi
touch "$OUT/DONE-round_re"

# --- the ladder, only now that a known-good size has actually run ----------------------------
say "ceiling start (pre-flight passed, so a rung failure is about the rung)"
for binder in 174 206 238 270 302 334 366; do
  b="$OUT/rung_b$binder"; mkdir -p "$b"
  say "rung binder=$binder start"
  timeout 1800 $PY -u ~/bwx/tt-bio/perf/bcx_round/run_round.py \
      --rounds 1 --binder "$binder" --params $HOME/bwx/af2_params --out "$b" $ARM \
      > "$b/round.log" 2>&1
  rc=$?
  axis=$($PY - "$b" <<'PY'
import json,re,sys,pathlib
p=pathlib.Path(sys.argv[1])/"round_events.json"
try: s=json.load(open(p))["stamp"]
except Exception: print("?"); raise SystemExit
ax=set()
for k in (s.get("rne_add_stats") or {}).get("rejects",{}):
    for m in re.finditer(r"\((\d+), (\d+), (\d+), (\d+)\)",k): ax.add(int(m.group(2)))
print(",".join(str(a) for a in sorted(ax)) or "?")
PY
)
  say "rung binder=$binder rc=$rc seam_axis=$axis"
  if [ $rc -eq 124 ]; then
    say "TIMEOUT at binder=$binder after 1800 s: a RUNTIME wall, not a memory ceiling."
    say "recorded separately; this rung does not bound the memory ceiling"; break
  fi
  if [ $rc -ne 0 ]; then
    # Classify before calling it a ceiling: only an allocator refusal is one.
    if grep -qiE "ran out of device memory|out of memory|allocate .* buffer|Out of Memory" "$b/round.log"; then
      say "MEMORY REFUSAL at binder=$binder (seam axis $axis). Refusal verbatim:"
      grep -A24 -iE "ran out of device memory|Out of Memory" "$b/round.log" | head -40 | tee -a "$OUT/sitting.log"
      say "the cap is the rung BELOW this one, not the largest that passed"
    else
      say "NON-MEMORY failure at binder=$binder rc=$rc -- this does NOT bound the ceiling:"
      grep -iE "lever went inert|Traceback|TT_FATAL|Error" "$b/round.log" | head -8 | tee -a "$OUT/sitting.log"
    fi
    break
  fi
  grep -E "^round |median|s a round" "$b/round.log" | tail -3 | tee -a "$OUT/sitting.log"
done
touch "$OUT/DONE-ceiling"
say "sitting4 done"
