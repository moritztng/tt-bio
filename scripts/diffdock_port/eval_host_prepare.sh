#!/usr/bin/env bash
# Put a dock host in a state where pose_eval.py can actually run, then start its shard.
#
# Written after Galaxy .112 was handed the 1/2 shard with the tree and the venv copied but no
# weights: every worker raised CheckpointMissing, the run ended in 10 s, and the done-marker
# still printed. The weights are NOT in the tt-bio cache. Two things have to be on the host:
#
#   ~/.boltz/diffdock/models/{score_model,confidence_model}   the released v1.1 pair, ~135 MB
#   ~/.cache/huggingface/hub/models--facebook--esm2_t33_650M_UR50D (+ xet)   the receptor LM
#
# Both are verified here BEFORE the dock is launched, because a host that cannot load a
# checkpoint should fail now and loudly, not 20 workers deep behind an rc=0.
#
#   eval_host_prepare.sh <host> <remote-root> <shard> <workers> <threads>
set -uo pipefail
H=${1:?host} ROOT=${2:?remote root} SHARD=${3:?shard i/n} W=${4:-20} T=${5:-2}
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15"
CKPT_SRC=${CKPT_SRC:-anx-glx:.boltz/diffdock}          # small, 135 MB
HF_SRC=${HF_SRC:-/home/moritz/.cache/huggingface}      # 2.5 GB, served from pc so the
                                                       # timed host is not read at length
say() { echo "$(date -u +%FT%TZ) $*"; }

# ---- refuse to start a second pool in the same tree
if timeout 60 $SSH "$H" "kill -0 \$(cat $ROOT/dock.pid 2>/dev/null) 2>/dev/null"; then
  say "ABORT: $H already has a live dock (dock.pid)"; exit 1
fi
RH=$(timeout 60 $SSH "$H" 'echo $HOME') || { say "ABORT: cannot ssh $H"; exit 1; }
say "host $H home=$RH root=$ROOT shard=$SHARD"

# ---- weights
# rsync cannot go remote-to-remote, so the checkpoints are staged on pc once and reused.
STAGE=${STAGE:-/home/moritz/scratch/dfd/eval/ckpt-stage}
if [ ! -s "$STAGE/models/score_model/best_ema_inference_epoch_model.pt" ]; then
  say "staging diffdock checkpoints on pc from $CKPT_SRC"
  mkdir -p "$STAGE"
  timeout 1800 rsync -a -e "$SSH" "$CKPT_SRC/" "$STAGE/" || { say "ABORT: ckpt stage"; exit 2; }
fi
say "shipping diffdock checkpoints to $H"
timeout 1800 rsync -a -e "$SSH" "$STAGE/" "$H:$RH/.boltz/diffdock/" || { say "ABORT: ckpt rsync"; exit 2; }
say "shipping ESM-2 cache from $HF_SRC"
for sub in hub/models--facebook--esm2_t33_650M_UR50D xet modules; do
  [ -e "$HF_SRC/$sub" ] || continue
  timeout 3600 rsync -a --ignore-existing -e "$SSH" \
    "$HF_SRC/$sub/" "$H:$RH/.cache/huggingface/$sub/" || { say "ABORT: hf rsync $sub"; exit 3; }
done

# ---- assert, do not hope
# The port's own resolver is the check: it is what raises CheckpointMissing per worker, so
# asking it here is the same question, asked once and before 20 workers ask it.
timeout 600 $SSH "$H" "cd $ROOT/tt-bio && $ROOT/venv/bin/python -c '
import sys; sys.path.insert(0, \".\")
from tt_bio.diffdock import checkpoints as c
for kind in (\"score_model\", \"confidence_model\"):
    print(kind, [str(p) for p in c.resolve(kind, None)])
print(\"CKPT_OK\")
'" || { say "ABORT: checkpoints do not resolve on $H"; exit 4; }

# ---- a stale DOCK_DONE from the failed launch must not be what the tick reads
timeout 60 $SSH "$H" "[ -f $ROOT/dock.log ] && mv $ROOT/dock.log $ROOT/dock-\$(date -u +%H%M%SZ).log; true"

say "launching shard $SHARD, ${W}x${T}, nice 10"
timeout 120 $SSH "$H" "cd $ROOT/tt-bio && \
  DFD_PB=$ROOT/pb/posebusters_benchmark_set PYTHONHASHSEED=0 PYTHONUNBUFFERED=1 \
  setsid nohup nice -n 10 $ROOT/venv/bin/python scripts/diffdock_port/pose_eval.py dock \
    --ids all --shard $SHARD --out $ROOT/poses --workers $W --threads $T --seed 42 \
    >> $ROOT/dock.log 2>&1 < /dev/null & echo \$! > $ROOT/dock.pid" || { say "ABORT: launch"; exit 5; }

sleep 90
say "pid=$(timeout 60 $SSH "$H" "cat $ROOT/dock.pid")"
timeout 60 $SSH "$H" "tail -6 $ROOT/dock.log; echo workers=\$(pgrep -fc 'pose_eval.py worker')"
say "PREPARE_DONE"
