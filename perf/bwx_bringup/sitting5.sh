#!/bin/bash
# Find the actual token ceiling. The first ladder ran every rung it had -- 320 through 512 tokens,
# nothing refused -- so 512 is a floor on the ceiling and not the ceiling. This walks on up.
#
# Rungs, target hPDL1 = 115 residues, axis = pad(115 + binder, 32):
#   binder 398 430 462 494 526 558 590 622  ->  544 576 608 640 672 704 736 768 tokens
# 576 is the interesting one: a Blackhole p150a REFUSES above 576 (MEASURED_MAX_TOKENS_P150A), so
# a 12 GiB Wormhole chip clearing it would mean the p150a limit is not about DRAM at all.
#
# Everything the first ladder learned is kept:
#  * the agent stays DOWN for the whole ladder. It reclaims the card BETWEEN rungs and blocked a
#    rung at 0.0% CPU for 3 m 42 s on a lease it would have waited TT_BIO_LEASE_TIMEOUT for.
#  * the three levers fast_round arms for itself are DECLARED, or meter.py raises
#    `lever went inert mid-arm` and the ladder reads that rc=1 as a memory refusal.
#  * a pre-flight at 288 tokens, a size already known to run, so a configuration fault cannot be
#    published as a ceiling.
#  * a rung failure is called a ceiling ONLY when the allocator says so. Anything else is a
#    non-memory failure that does not bound the ceiling, and a timeout is a runtime wall.
#  * the axis on every rung is read off the tensor the seam ran, never computed.
set -uo pipefail
cd ~/bwx
OUT=~/bwx/out/sitting5
CHIP=${BWX_CHIP:-30}
PY=~/bwx/venv/bin/python
mkdir -p "$OUT"
say(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/sitting.log"; }
. ~/japanfold/env.sh
export TT_METAL_CACHE=$HOME/bwx/cache/tt-metal PYTHONPATH=$HOME/bwx/tt-bio BCX_BC2=$HOME/bwx/bc2
export JAX_COMPILATION_CACHE_DIR=$HOME/bwx/cache/xla
export TT_VISIBLE_DEVICES=$CHIP TT_BIO_LEASE_CARDS=$CHIP
export TT_BIO_LEASE_HOLDER=worker:bwx-bringup TT_BIO_LEASE_TIMEOUT=28800
export TT_BIO_MM_LAYOUT=1 TT_BIO_TAPED_CHANNEL_MOVE=1 TT_BIO_WIDEN_ADD=1
ARM="--exact 0 --triatt-hifi 1 --rne-kernel 1 --triatt-bw 1 --extra-msa 1 --template 1"

axis_of(){ $PY - "$1" <<'PY'
import json,re,sys,pathlib
p=pathlib.Path(sys.argv[1])/"round_events.json"
try: s=json.load(open(p))["stamp"]
except Exception: print("?"); raise SystemExit
ax=set()
for k in (s.get("rne_add_stats") or {}).get("rejects",{}):
    for m in re.finditer(r"\((\d+), (\d+), (\d+), (\d+)\)",k): ax.add(int(m.group(2)))
print(",".join(str(a) for a in sorted(ax)) or "?")
PY
}

say "sitting5 start chip=$CHIP -- extending the ladder above 512 tokens"

# --- pre-flight at a size already known to run ------------------------------------------------
d="$OUT/preflight"; mkdir -p "$d"
say "preflight start binder=146 (288 tokens), 1 round"
timeout 1800 $PY -u ~/bwx/tt-bio/perf/bcx_round/run_round.py \
    --rounds 1 --binder 146 --params $HOME/bwx/af2_params --out "$d" $ARM > "$d/round.log" 2>&1
rc=$?; say "preflight rc=$rc seam_axis=$(axis_of "$d")"
if [ $rc -ne 0 ]; then
  say "PRE-FLIGHT FAILED at a size known to run: the configuration is wrong, not the card."
  grep -iE "lever went inert|Traceback|TT_FATAL|Error" "$d/round.log" | head -8 | tee -a "$OUT/sitting.log"
  say "NOT running the ladder; a rung failure now would be misread as the ceiling"
  touch "$OUT/DONE-ceiling5"; exit 1
fi

# --- the rungs --------------------------------------------------------------------------------
say "ladder start: 544 576 608 640 672 704 736 768 tokens"
for binder in 398 430 462 494 526 558 590 622; do
  b="$OUT/rung_b$binder"; mkdir -p "$b"
  say "rung binder=$binder start"
  timeout 2400 $PY -u ~/bwx/tt-bio/perf/bcx_round/run_round.py \
      --rounds 1 --binder "$binder" --params $HOME/bwx/af2_params --out "$b" $ARM \
      > "$b/round.log" 2>&1
  rc=$?
  say "rung binder=$binder rc=$rc seam_axis=$(axis_of "$b")"
  if [ $rc -eq 124 ]; then
    say "TIMEOUT at binder=$binder after 2400 s: a RUNTIME wall, not a memory ceiling."
    say "recorded separately; this rung does NOT bound the memory ceiling"; break
  fi
  if [ $rc -ne 0 ]; then
    if grep -qiE "ran out of device memory|out of memory|allocate .* buffer|Out of Memory" "$b/round.log"; then
      say "MEMORY REFUSAL at binder=$binder. THIS IS THE CEILING: the cap is the rung BELOW it."
      grep -A24 -iE "ran out of device memory|Out of Memory" "$b/round.log" | head -40 | tee -a "$OUT/sitting.log"
    else
      say "NON-MEMORY failure at binder=$binder rc=$rc -- this does NOT bound the ceiling:"
      grep -iE "lever went inert|Traceback|TT_FATAL|Error|Signal" "$b/round.log" | head -8 | tee -a "$OUT/sitting.log"
    fi
    break
  fi
  grep -E "^round |median" "$b/round.log" | tail -2 | tee -a "$OUT/sitting.log"
done
touch "$OUT/DONE-ceiling5"
say "sitting5 done"
