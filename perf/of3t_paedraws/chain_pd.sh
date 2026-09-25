#!/usr/bin/env bash
# of3t-paedraws on qb2: draws 1-5 of PROTOCOL A44. Per draw k: float64 and upstream bf16 reference
# (confpfe's run_ref.sh arguments plus --seed k, all ten CPU runs at once, 2 threads each), the
# device step on card 1 (confpfe's devarm_cf.sh through perf/of3t_paedraws/devstep.py --seed k,
# draws in sequence), then confpfe's chain_cf.sh scoring. Every step skips when its output exists,
# so a relaunch resumes.
#
#   chain_pd.sh [K...]              default 1 2 3 4 5
#   CARD=0 REFS=0 chain_pd.sh 1 3   device draws only, on card 0
#
# A failed device step leaves its card un-reinitialisable, and on qb2 the next open of that card
# hangs the host, so every failed step is followed by `tt-smi -r $CARD` before the next open
# (per-chip on tt-kmd 2.11: card 0 kept its heartbeat through a card 1 reset, 2026-09-24). A step
# still in the bring-up probe after 600 s is wedged (draw 1 sat there 2h08m on card 1 on
# 2026-09-24, the C call SIGALRM cannot interrupt) and is killed. Each draw gets two attempts.
W=/home/ttuser/.coworker/wt/of3t-paedraws; P=$W/perf/of3t_paedraws; S=/home/ttuser/of3t_paedraws
R=/home/ttuser/of3t-campaign-refs; CK=/home/ttuser/of3-weights/of3-p2-155k.pt
DRAWS=/home/ttuser/of3t_fullstep64/draws.pt; CARD=${CARD:-1}; SMI=/home/ttuser/.local/bin/tt-smi
SPY=/home/ttuser/.local/bin/py-spy
KS=("${@:-1 2 3 4 5}"); KS=(${KS[*]})
L=$S/chain.log
cd $W
source /home/ttuser/tt-bio-dev/env/bin/activate
log() { echo "=== $* $(date -u +%FT%TZ)" >> $L; }

# --- references: every missing (k, mode) at once
export MALLOC_MMAP_THRESHOLD_=1048576 MALLOC_TRIM_THRESHOLD_=1048576 MALLOC_ARENA_MAX=2
[ "${REFS:-1}" = 1 ] && for k in "${KS[@]}"; do
  for mode in f64 bf16; do
    O=$S/ref_s$k/$mode; G=$O/grads_$mode.pt
    [ -s "$G" ] && continue
    mkdir -p $O
    ( log "ref s$k $mode start"
      PYTHONPATH=$R/of3pkg043:/home/ttuser/of3t_refprec/deps OMP_NUM_THREADS=2 nice -n 5 \
        python3 perf/of3t_fullstep64/ref_step.py --mode $mode --seed $k --denoise \
        --batch $R/bundle_min_043/batch_step003.pt \
        --batch-sha256 3c32597a20f09bf50769defa561f7df86da4de721da325eb76431a9d80b6285f \
        --checkpoint $CK --replay-draws $DRAWS --threads 2 --rss-cap-gb 40 --chunk-size 32 \
        --out-dir $O --disk-checkpoint $S/ckpt/s$k/$mode > $S/ref_s${k}_$mode.log 2>&1
      log "ref s$k $mode exit $?"
      rm -rf $S/ckpt/s$k/$mode ) &
  done
done

# --- device: one draw at a time on card $CARD, each scored once its references land
BOARD=$("$SMI" -s 2>/dev/null | python3 -c "
import sys,json;d=json.load(sys.stdin);print(d['device_info'][$CARD]['board_info']['board_type'])")
[ "$BOARD" = p300c ] || { log "card $CARD is a $BOARD, not p300c, refusing"; wait; exit 3; }
unset TT_MESH_GRAPH_DESC_PATH TT_BIO_SOFTMAX_BW_RENORM
for k in "${KS[@]}"; do
  T=PD384_s$k; OUT=$P/DEV_$T.json; G=$S/grad_$T.pt
  if [ ! -s "$G" ] || [ ! -s "$OUT" ]; then
    (while true; do echo "$(date -u +%T) $("$SMI" -s 2>/dev/null | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['device_info'][$CARD]['telemetry']['aiclk'].strip())")"; sleep 5; done) > $P/AICLK_$T.txt 2>&1 &
    M=$!
    for attempt in 1 2; do
      log "dev $T start $(hostname) card $CARD $BOARD $(git rev-parse --short HEAD) attempt $attempt"
      env TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:of3t-paedraws \
        OMP_NUM_THREADS=8 PYTHONPATH=$W timeout 18000 python3 $P/devstep.py --seed $k \
        --repr-out $S/repr_$T.pt --denoise --exact on --draws $DRAWS --grad-out $G \
        --weights-out $S/weights_walked_$T.pt --out $OUT > $S/dev_$T.log 2>&1 &
      D=$!   # timeout's pid, which leads the step's process group
      ( sleep 600; py=$(pgrep -P $D | head -1)
        [ -n "$py" ] && timeout 60 "$SPY" dump --pid $py 2>/dev/null | grep -q _assert_local_dispatch \
          && { log "dev $T still in the bring-up probe after 600 s, killing"; kill -KILL -- -$D; } ) &
      WD=$!
      wait $D; rc=$?
      kill $WD 2>/dev/null
      log "dev $T exit $rc"
      [ "$rc" = 0 ] && break
      kill -KILL -- -$D 2>/dev/null; sleep 5
      # A lease refusal means another worker holds the card and this step never opened it:
      # resetting then hits their run (2026-09-24: this line reset card 0 at 21:40Z while
      # bcx-round held it; bcx-round's card 0 run died with a bus error and it restarted on
      # card 3 at 21:41Z). Wait for the holder instead.
      if grep -q DeviceInUseError $OUT 2>/dev/null; then
        log "dev $T refused by the lease, card $CARD not opened, no reset; retrying in 1800 s"
        sleep 1800; continue
      fi
      log "dev $T reset card $CARD: $(timeout 180 "$SMI" -r $CARD 2>&1 | tail -1)"
    done
    kill $M
    [ "$rc" = 0 ] || continue
    python3 - "$OUT" "$CARD" "$BOARD" "$rc" "$G" <<'PY'
import hashlib, json, socket, subprocess, sys
out, card, board, rc, g = sys.argv[1:]
d = json.load(open(out))
h = hashlib.sha256()
with open(g, "rb") as f:
    for b in iter(lambda: f.read(1 << 22), b""):
        h.update(b)
git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
d["provenance"] = {"host": socket.gethostname(), "row": "of3t-paedraws", "card": int(card),
                   "board_class": board, "git_commit": git("rev-parse", "HEAD"),
                   "git_dirty": git("status", "--porcelain", "tt_bio"), "exit": int(rc),
                   "grad_dump_sha256": h.hexdigest()}
json.dump(d, open(out, "w"), indent=1)
PY
    python3 perf/of3t_fullstep64/bijmap.py --weights $S/weights_walked_$T.pt --checkpoint $CK \
      --out $P/BIJECTION_$T.json >> $L 2>&1
    python3 - "$S/weights_walked_$T.pt" "$P/DEVICE_SHAPES_$T.json" <<'PY'
import hashlib, json, sys, torch
w, out = sys.argv[1:]
d = torch.load(w, map_location="cpu")
json.dump({"weights_walked_sha256": hashlib.sha256(open(w, "rb").read()).hexdigest(),
           "shapes": {k: list(v.shape) for k, v in d.items()}, "source": w}, open(out, "w"))
PY
  fi
done

# --- score each draw whose three sides exist
wait
for k in "${KS[@]}"; do
  T=PD384_s$k; F=$S/ref_s$k/f64/grads_f64.pt; B=$S/ref_s$k/bf16/grads_bf16.pt
  [ -s $P/SCORE_$T.json ] && continue
  [ -s $F ] && [ -s $B ] && [ -s $S/grad_$T.pt ] && [ -s $P/BIJECTION_$T.json ] || { log "s$k incomplete, not scored"; continue; }
  log "score $T start"
  python3 perf/of3t_fullstep64/score.py --f64 $F --bf16 $B --bijection $P/BIJECTION_$T.json \
    --shapes $P/DEVICE_SHAPES_$T.json --arm $T=$S/grad_$T.pt --out $P/SCORE_$T.json >> $L 2>&1
  log "score $T exit $?"
done
log "chain done"
