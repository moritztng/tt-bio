#!/bin/bash
# The rest of the redo sitting, re-planned around what the first attempt measured: the float64
# VJP reference does not fit on a .107 that is serving.
#
# `afgrad.py vjp` builds the whole 52-block float64 graph with every activation retained
# (`ref_stack(keep=True)`) and backprops it in one go, so its ~350 GB RSS peak is structural.
# `--blocks` only scopes what is scored after that peak, and shrinking `--evo` would change the
# cotangents (each boundary's cotangent is backpropagated from a readout at the STACK output), so
# it would stop being the same test as the Blackhole artifact. The only lever is the host: with
# the agent up, MemAvailable bottomed at 52 GB and the first grade was killed by its own guard.
#
# So the agent stays down for the grade and goes back up before the round and the ladder, which
# are device-bound. The agent is restarted by a trap, so it comes back even if the grade dies.
#
#   1 wait out grade_45 (already running, orphaned deliberately so round_re could not start)
#   2 grade_re   blocks 4,5,6,7 + both controls, agent DOWN        -- charter item 3
#   3 agent UP
#   4 round_re   the (1,1,hifi) round on the good chip
#   5 ceiling    the token ladder                                   -- charter item 4
set -uo pipefail
cd ~/bwx
OUT=~/bwx/out/sitting2
CHIP=${BWX_CHIP:-30}
PARAMS=$HOME/bwx/af2_params/params_model_1_ptm.npz
PY=~/bwx/venv/bin/python
mkdir -p "$OUT"
say(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/sitting.log"; }

. ~/japanfold/env.sh
export TT_METAL_CACHE=$HOME/bwx/cache/tt-metal PYTHONPATH=$HOME/bwx/tt-bio BCX_BC2=$HOME/bwx/bc2
export JAX_COMPILATION_CACHE_DIR=$HOME/bwx/cache/xla
export TT_VISIBLE_DEVICES=$CHIP TT_BIO_LEASE_CARDS=$CHIP
export TT_BIO_LEASE_HOLDER=worker:bwx-bringup TT_BIO_LEASE_TIMEOUT=28800

agent_up(){
  systemctl is-active --quiet japanfold-agent@ubuntu && return 0
  sudo -n systemctl start japanfold-agent@ubuntu
  say "agent start rc=$? (dev .107 back in /v1/cluster)"
}
trap agent_up EXIT

say "sitting3 start: agent is down for the grade window only"

# --- 1. let grade_45 finish; it was orphaned on purpose so round_re could not take the card ----
for i in $(seq 900); do
  pgrep -f "afgrad.py vjp --n 288 --blocks 4,5 " >/dev/null || break
  sleep 2
done
say "grade_45 finished (or absent); artifact: $(ls -la ~/bwx/tt-bio/perf/bcx_afgrad/vjp_n288_wh_hifi2.json 2>/dev/null || echo MISSING)"
touch "$OUT/DONE-grade_45"

# --- 2. the grade that matters, with the host to itself --------------------------------------
d="$OUT/grade_re"; mkdir -p "$d"
say "grade_re start blocks=4,5,6,7 controls-all, agent down, floor 40 GB"
$PY -u ~/bwx/tt-bio/perf/bcx_afgrad/afgrad.py vjp --n 288 --blocks 4,5,6,7 --controls-all \
    --tag wh_chars2 --params "$PARAMS" > "$d/grade.log" 2>&1 &
gp=$!
bash ~/bwx/memguard.sh "$gp" 40 >> "$d/memguard.log" 2>&1 &
wait $gp; say "grade_re rc=$?"
grep -iE "\b(inf|-inf|nan)\b|Bus error|Non-existent physical|Signal:" "$d/grade.log" | head -8 \
    | sed "s/^/  grade_re: /" | tee -a "$OUT/sitting.log"
tail -12 "$d/grade.log" | tee -a "$OUT/sitting.log"
touch "$OUT/DONE-grade_re"

# --- 3. dev gets its box back before anything device-bound -----------------------------------
agent_up

# --- 4. the round, same arm as the Blackhole peer --------------------------------------------
d="$OUT/round_re"; mkdir -p "$d"
say "round_re start binder=146 arm=(1,1,hifi)"
timeout 1800 $PY -u ~/bwx/tt-bio/perf/bcx_round/run_round.py \
    --rounds 4 --binder 146 --params $HOME/bwx/af2_params --out "$d" \
    --exact 0 --triatt-hifi 1 --rne-kernel 1 --triatt-bw 1 --extra-msa 1 --template 1 \
    > "$d/round.log" 2>&1
say "round_re rc=$?"
grep -E "^round |median|s a round" "$d/round.log" | tail -6 | tee -a "$OUT/sitting.log"
touch "$OUT/DONE-round_re"

# --- 5. the token ceiling --------------------------------------------------------------------
say "ceiling start"
BWX_OUT="$OUT/ceiling" bash ~/bwx/ceiling.sh
touch "$OUT/DONE-ceiling"
say "sitting3 done"
