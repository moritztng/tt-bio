#!/bin/bash
# Is 544 the WALL, or a NOTCH?
#
# sitting5's 544-token rung refused in the Evoformer backward with the card genuinely full
# (12.756 GB of 12.885 GB held by the fold). But its own log says why it was that full:
#
#   "the fused triangle attention declined all 96 calls at 544 tokens: its circular buffers do
#    not fit L1 at this token axis, so the composed path is running and holding the
#    [544,4,544,544] fp32 scores -- 2.576 GB of device memory this fold would not otherwise
#    need. Whether the arm fits is a property of the token axis and is NOT monotone in it."
#
# That line appears at 544 and at NO other rung: 320/352/384/416/448/480/512 all had the fused
# arm. So 544 may be a hole in a non-monotone clash class rather than the top of the card, and
# the tool's own advice is to try one bucket UP. The ladder broke on first failure and never did.
#
# So this does NOT break on a refusal -- it runs every rung and classifies each one:
#   576 608 640 tokens (binder 430 462 494), then 544 AGAIN, alone.
# The 544 repeat is the point: it is the rung that sets the published cap, and a red arm is
# re-scored by itself before it is believed.
#
# Reading it: if 576 PASSES, 544 is a notch and the cap is not 512 -- the honest statement
# becomes "512 is the largest axis that completes at every size below it, and 544 is a hole".
# If 576 and 608 both refuse WITH the fused arm in place, 512 is the wall.
set -uo pipefail
cd ~/bwx
OUT=~/bwx/out/sitting6
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

say "sitting6 start chip=$CHIP -- notch-or-wall probe above 544, no break on refusal"
for binder in 430 462 494 398; do
  b="$OUT/rung_b$binder"; mkdir -p "$b"
  say "rung binder=$binder start"
  timeout 2400 $PY -u ~/bwx/tt-bio/perf/bcx_round/run_round.py \
      --rounds 1 --binder "$binder" --params $HOME/bwx/af2_params --out "$b" $ARM \
      > "$b/round.log" 2>&1
  rc=$?
  axis=$(axis_of "$b")
  # Did the fused triangle attention arm survive at this axis? That is the whole question.
  if grep -q "declined all" "$b/round.log"; then fused="fused arm DECLINED (composed path, +2.576 GB)"
  else fused="fused arm held"; fi
  say "rung binder=$binder rc=$rc axis=$axis -- $fused"
  if [ $rc -eq 124 ]; then
    say "  TIMEOUT after 2400 s: a runtime wall, not a memory one"
  elif [ $rc -ne 0 ]; then
    if grep -qiE "ran out of device memory|Out of Memory" "$b/round.log"; then
      say "  MEMORY REFUSAL at axis $axis:"
      grep -A4 "ran out of device memory" "$b/round.log" | head -6 | tee -a "$OUT/sitting.log"
    else
      say "  NON-MEMORY failure rc=$rc -- does NOT bound the ceiling:"
      grep -iE "lever went inert|TT_FATAL|Traceback" "$b/round.log" | head -4 | tee -a "$OUT/sitting.log"
    fi
  else
    say "  COMPLETED at axis $axis"
    grep -E "^round |median" "$b/round.log" | tail -2 | tee -a "$OUT/sitting.log"
  fi
done
touch "$OUT/DONE-ceiling6"
say "sitting6 done"
