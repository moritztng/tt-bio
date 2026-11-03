#!/bin/bash
# 9DBP (791 tokens, PopVax's told ~800 aa two-chain shape) timing cells, hooked in after the timing arms and before
# RFD4 by box_pvo.sh's EXTRA list:  EXTRA="box_9dbp_pvo.sh box_rfd4_pvo.sh run_acc.sh"
# Same recipe, venvs and pins as box_pvo.sh's arms: OpenDDE exact + fast, Mol-1 exact + the fast arm ranked on this
# card (mol1/FAST-ARM), K=1, r0 cold + 3 warm. An OOM at 40 GiB is recorded FAILED and is a result, not a retry.
# Inputs: /root/kc/inputs-9dbp from make_inputs.py, hash-checked here. Resumable on each arm's rc file.
set -u
DEADLINE=${1:?deadline epoch}
B=/root/kc/pvo; I=/root/kc/inputs-9dbp
eval "$(grep -E '^(ODDE_SHA|MOL1_SHA|ODDE_FAST)=' /root/kc/box_pvo.sh)"
say(){ echo "$(date -u +%FT%TZ) box_9dbp_pvo: $*"; }

(cd $I && python3 -c 'import json,hashlib,sys
bad=[p for p,h in json.load(open("INDEX.json"))["sha256"].items() if hashlib.sha256(open(p,"rb").read()).hexdigest()!=h]
sys.exit(f"hash mismatch {bad}" if bad else 0)') > $B/inputs-9dbp.log 2>&1 || { say "inputs failed"; exit 2; }
cp -r $I $B/

arm(){ local m=$1 n=$2 e=$3; shift 3
  local d=$B/$m/9dbp-$n-k1; [ "$(cat $d/rc 2>/dev/null)" = 0 ] && return 0
  [ $(date +%s) -ge $DEADLINE ] && { say "deadline: skip $m $n"; return 1; }
  rm -rf $d; mkdir -p $d; say "arm $m 9dbp-$n k1"
  local ck=$OPENDDE_ROOT_DIR/checkpoint/opendde.pt sha=$ODDE_SHA; [ $m = mol1 ] && ck=$MOL_ROOT_DIR/checkpoint/mol-1-base.pt sha=$MOL1_SHA
  (cd /tmp && env $e PVO_SHA=$sha PATH=/opt/$m-venv/bin:$PATH timeout 5400 /opt/$m-venv/bin/python -I \
     /root/kc/pvo_fold.py --model $m --inputs $I/9DBP.json --recipe /root/kc/recipe-$m.json --arm $n --k 1 \
     --warm 3 --out $d --ckpt $ck "$@") > $d/stdout 2> $d/stderr
  local rc=$?; echo $rc > $d/rc; say "arm $m 9dbp-$n rc=$rc"; }

arm opendde exact LAYERNORM_TYPE=torch
arm opendde fast LAYERNORM_TYPE=fast_layernorm $ODDE_FAST
arm mol1 exact X=1
MBEST=$(cat $B/mol1/FAST-ARM)
case $MBEST in
  cueq-triatt-c5-tf32) MF="--trimul cuequivariance --triatt triattention --chunk 5 --tf32";;
  cueq-cueq-c5-tf32) MF="--trimul cuequivariance --triatt cuequivariance --chunk 5 --tf32";;
  *) MF="--trimul flashpairformer --triatt triattention --chunk 5 --tf32";;
esac
arm mol1 $MBEST X=1 $MF
echo done > $B/9DBP-DONE; say "9DBP arms done"
