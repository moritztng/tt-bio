#!/bin/sh
# Run the two remaining RFD3 in-fold passes, waiting for a board-007 window.
#
# Board 007 is a dual-chip p300c and chip 0 cannot be opened alone (TT_FATAL:
# "Custom fabric mesh graph descriptor path must be specified"), so this leg has to expose 0,1 —
# but card 1 is leased by perfwar-shared-matmul-sites and its runs come in bursts. Opening while
# they hold chip 1 fails hard in SiliconSysmemManager::pin_or_map_iommu, so each attempt is gated
# on chip 1 being free and retried rather than queued.
cd /home/ttuser/.coworker/wt/perfwar-rfd3-esmfold2-sites || exit 1
E_VIS="TT_VISIBLE_DEVICES=0,1"
E_LOG="TT_BIO_LOGICAL_DEVICE_ID=0"
E_LEASE="TT_BIO_LEASE_HOLDER=worker:perfwar-rfd3-esmfold2-sites"
P=/home/ttuser/tt-bio/env/bin/python3

run_one() {
  out=$1; unit=$2; spec=$3; steps=$4; log=$5; probes=$6
  i=0
  while [ $i -lt 60 ]; do
    if ! fuser /dev/tenstorrent/1 >/dev/null 2>&1; then
      echo "[driver] window open, launching $out" >&2
      env $E_VIS $E_LOG $E_LEASE PYTHONPATH="$PWD" timeout 1100 "$P" -u \
        perf/attn_sites/infold_parity.py --out "$out" --probes "$probes" --unit "$unit" -- \
        design "$spec" --model rfd3 --from_pdb --num_timesteps "$steps" --device_ids 0 \
        --out_dir perf/attn_sites/_out_rfd3 > "$log" 2>&1
      if grep -q "\[parity\] [0-9]* target classes" "$log"; then
        echo "[driver] $out done" >&2
        return 0
      fi
      echo "[driver] $out attempt failed (device open lost the race), retrying" >&2
    fi
    i=$((i + 1))
    sleep 15
  done
  echo "[driver] $out GAVE UP waiting for chip 1" >&2
  return 1
}

run_one perf/attn_sites/infold_rfd3_286_t4.json "286-token RFD3 design, 4 timesteps" \
        perf/attn_sites/rfd3_iai_286.yaml 4 perf/attn_sites/infold_rfd3_286.log 1
run_one perf/attn_sites/infold_rfd3_298_t8.json "298-token RFD3 design, 8 timesteps" \
        perf/attn_sites/rfd3_iai_298.yaml 8 perf/attn_sites/infold_rfd3_t8.log 0
echo "[driver] all runs attempted" >&2
