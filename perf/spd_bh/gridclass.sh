#!/bin/bash
# gridclass: is transition_b8's BH digest split a grid effect or the small-grid code path? args: CHIP OUT TREE
# Arms: native 11x10, forced 8x8 and 10x10 with the big-grid path held. s8_b8 (small path, 8c7ec5be) and the base-set control were read in gridclass r1.
set -u; CHIP=$1 OUT=$2 TREE=$3; ROW=spd-m-protenix
mkdir -p "$OUT" ~/spd/locks ~/spd/$ROW/leases; [ -f ~/japanfold/env.sh ] && . ~/japanfold/env.sh >/dev/null 2>&1
cd "$TREE" || exit 1
export PYTHONPATH=$TREE TT_VISIBLE_DEVICES=$CHIP TT_BIO_LEASE_HOLDER=$ROW TT_BIO_LEASE_DIR=$HOME/spd/$ROW/leases
exec 9> "$HOME/spd/locks/chip$CHIP.lock"; flock -w "${WAIT:-600}" 9 || { echo flock busy >> $OUT/run.log; exit 3; }
rm -f generated/watcher/kernel_names.txt generated/watcher/kernel_elf_paths.txt
L=$(timeout 300 $PY -c "import torch, tt_bio.tenstorrent as T; T._FAST_MODE=False; T.is_wormhole=lambda: False; from tt_bio.protenix import mode_levers; print('+'.join(sorted(mode_levers({'layernorm_z_cycle.weight': torch.zeros(256)}))))" | tail -1)
[ -n "$L" ] || exit 1; echo "$L" > $OUT/levers.txt
# The fold runs in a spawned worker, so the big-grid hold must be installed at interpreter start (sitecustomize on
# PYTHONPATH), not in this parent. Every call logs the grid it applied and the path it took.
SC=$OUT/sc; mkdir -p $SC; cat > $SC/sitecustomize.py <<'PY'
import os, sys
if os.environ.get("GRIDCLASS_LOG") == "1":
    import tt_bio.tenstorrent as T
    f = T._apply_grid_thresholds
    def g(grid, device=None):
        r = f((11, 10) if os.environ.get("HOLD_BIG") == "1" else grid, device)
        print("GRIDCLASS", tuple(grid), "small", T._IS_SMALL_GRID, file=sys.stderr, flush=True)
        return r
    T._apply_grid_thresholds = g
PY
run() { # name force hold levers
  local n=$1 g=$2 h=$3 lv=$4 o=$OUT/$1; for try in 1 2; do rm -rf $o; mkdir -p $o
  ( [ -n "$g" ] && export TT_BIO_FORCE_GRID=$g; export HOLD_BIG=$h TT_BIO_LEVERS=$lv GRIDCLASS_LOG=1 PYTHONPATH=$SC:$TREE
    timeout 3000 $PY -m tt_bio.main predict examples/affinity_fkg.yaml --model protenix-v2 --single_sequence --sampling_steps 200 \
      --diffusion_samples 1 --seed 0 --out_dir $o/out --debug > $o/log 2>&1 ); rc=$?
  [ $rc = 0 ] && break; sleep 60; done
  md=$(find $o/out -path '*structures/*.cif' | sort | xargs cat 2>/dev/null | md5sum | cut -c1-8)
  echo "$(date -u +%FT%TZ) $n force=$g hold=$h rc=$rc md5=$md try=$try $(grep GRIDCLASS $o/log | sort | uniq -c | tr '\n' ' ')" >> $OUT/run.log
}
B8="$L+transition_b8+ws_b8_plain"
run n_b8 "" 0 "$B8"
run h8_b8 8,8 1 "$B8"
run h10_b8 10,10 1 "$B8"
touch $OUT/DONE
