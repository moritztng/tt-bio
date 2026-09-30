#!/bin/bash
# The seed floor the arm swap has to be read against: the same arm, the same input, a different
# seed. Without it an arm-vs-arm Angstrom number has no scale. OpenFold3's own floor on this
# target, not protenix-v2's (`~/pxens/gate1/protenix-hsa-msa.json` reads 0.695 A over 10 pairs,
# but that is a different model and cannot stand in for this one).
set -u
cd ~/b2pship
CARD=${CARD:-1}
NODE=${NODE:-2}
export TT_VISIBLE_DEVICES=$CARD PYTHONPATH=$HOME/b2pship
export INERT_DUMP_DIR=$HOME/b2pship/out/inertdump_hsa
export TT_BIO_TRIATT_HIFI_PAD_UP=2
MSA=$HOME/b2pship/out/msa_hsa
mkdir -p "$INERT_DUMP_DIR" out/clk

for SEED in 2 3; do
  out=out/of3_hsa_on_seed$SEED
  clk=out/clk/of3_hsa_on_seed$SEED.txt; : > "$clk"
  ( while :; do
      printf '%s %s\n' "$(date -u +%FT%TZ)" \
        "$(cat /sys/class/tenstorrent/tenstorrent\!$NODE/tt_aiclk 2>/dev/null || echo NA)" >> "$clk"
      sleep 5
    done ) & CLKPID=$!
  echo "=== SEEDFLOOR seed=$SEED PAD_UP=2 $(date -u +%FT%TZ)"
  INERT_TAG=of3_hsa_on_seed$SEED timeout 3000 ~/bcx_e2e_venv/bin/python out/inert_drv.py examples/hsa.yaml \
    --out_dir "$out" --model openfold3 --seed $SEED --msa_dir "$MSA" --msa_cache_only --override \
    > out/of3_hsa_on_seed$SEED.log 2>&1
  echo "=== SEEDFLOOR seed=$SEED rc=$? $(date -u +%FT%TZ)"
  kill $CLKPID 2>/dev/null
  grep -E "Error|FATAL|Traceback|refus|Out of Memory" out/of3_hsa_on_seed$SEED.log | tail -4
  sort -k2 -n "$clk" | awk 'NF==2 && $2!="NA"{v[n++]=$2} END{if(n)printf "CLK %s samples min %s median %s max %s MHz\n",n,v[0],v[int(n/2)],v[n-1]}'
done
echo "=== OF3 HSA SEEDFLOOR DONE $(date -u +%FT%TZ)"
