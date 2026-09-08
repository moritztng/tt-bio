#!/bin/bash
# Claim a free Galaxy chip, prove it dispatches, then run one PXDesign rung on it.
# usage: px_claim.sh <yaml> <label> <out_root> <deadline_epoch>
yaml=$1; label=$2; root=$3; deadline=$4
NDESIGNS=${NDESIGNS:---num_designs 4}
NSTEP=${NSTEP:---n_step 200}
WT=/home/cust-team/mthuening/ceilpxd/tree
cd "$WT" || exit 2
export TT_METAL_LOGGER_LEVEL=FATAL HF_HUB_CACHE=/home/cust-team/models
export TT_BIO_LEASE_HOLDER=worker:ceiling-pxdesign OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
out=$root/$label; rm -rf "$out"; mkdir -p "$out"

free_chips() {
  sudo lsof /dev/tenstorrent/* 2>/dev/null \
    | awk 'NR>1 && $9 ~ /tenstorrent\/[0-9]/ {split($9,a,"/"); print a[4]}' | sort -un > /tmp/held.$$
  seq 0 31 | grep -vxF -f /tmp/held.$$; rm -f /tmp/held.$$
}

while [ "$(date +%s)" -lt "$deadline" ]; do
  for c in $(free_chips); do
    # claim marker so two of my own arms never pick the same chip
    if ! mkdir "$root/.claim.$c" 2>/dev/null; then continue; fi
    # re-check the chip is still free at the instant of the claim
    if sudo lsof /dev/tenstorrent/$c >/dev/null 2>&1; then rmdir "$root/.claim.$c"; continue; fi
    export TT_VISIBLE_DEVICES=$c TT_BIO_LEASE_CARDS=$c
    # pre-flight: a raced or dirty bring-up shows up here in ~40 s, not 4 min into a rung
    if ! timeout 300 ./env/bin/python -c "
import tt_bio.tenstorrent as t, torch, ttnn
d = t.get_device(); x = ttnn.from_torch(torch.zeros((32,32), dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=d)
ttnn.synchronize_device(d); print('PROBE_OK')" > "$out/probe.log" 2>&1; then
      echo "PROBE_FAIL chip=$c $(grep -oE 'run_mailbox|remote-only|Timeout|Error' "$out/probe.log" | head -1)"
      rmdir "$root/.claim.$c"; continue
    fi
    t0=$(date +%s.%N)
    timeout 2400 ./env/bin/python -u -m tt_bio.main design "$yaml" --model pxdesign \
      --cache /home/cust-team/.boltz $NDESIGNS $NSTEP --seed 42 \
      --out_dir "$out/designs" > "$out/run.log" 2>&1
    rc=$?; t1=$(date +%s.%N)
    n=$(find "$out/designs" -name "*.cif" 2>/dev/null | wc -l)
    err=$(grep -oE "(Out of Memory|out of memory|Statically allocated circular buffers|allocate [0-9]+ B|run_mailbox|RuntimeError:|Error:).*" "$out/run.log" | head -1 | cut -c1-150)
    printf 'RESULT label=%s chip=%s rc=%s wall_s=%.1f cifs=%s err=%s\n' \
      "$label" "$c" "$rc" "$(echo "$t1 - $t0" | bc)" "$n" "${err:-none}"
    rmdir "$root/.claim.$c"; exit 0
  done
  sleep 4
done
echo "RESULT label=$label chip=none rc=timeout wall_s=0 cifs=0 err=no_free_chip_before_deadline"
