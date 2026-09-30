#!/bin/bash
# Does the pad-up move OpenFold3's delivered structure?
#
# The first A/B ran on a random 544-mer with `msa: empty` and both arms folded to pLDDT 28, so the
# 7.2 A between them measured the disorder and not the arm. This one uses the repo's own HSA
# fixture: 585 residues, tile-padded axis 608, which is one of the lengths with no legal fused
# config, so the pad-up fires. Its MSA is already searched (1000 sequences, hash a35bb68136a5125a,
# from the protenix-v2 parity runs) and is passed with --msa_cache_only so a cache miss is an error
# rather than a silent single-sequence fold.
#
# Same seed on both arms. The clock is sampled DURING each fold from sysfs, not before.
set -u
cd ~/b2pship
CARD=${CARD:-1}                 # UMD id
NODE=${NODE:-2}                 # /dev/tenstorrent node the UMD id maps to, for the clock
SEED=${SEED:-1}
export TT_VISIBLE_DEVICES=$CARD PYTHONPATH=$HOME/b2pship
export INERT_DUMP_DIR=$HOME/b2pship/out/inertdump_hsa
MSA=$HOME/b2pship/out/msa_hsa
mkdir -p "$INERT_DUMP_DIR" "$MSA" out/clk
cp -n ~/pxens/gate1/msa/protenix-v2__hsa__msa-server_200step_5sample_10cycle_bf16/a35bb68136a5125a.a3m "$MSA"/ 2>/dev/null

for arm in on off; do
  [ "$arm" = off ] && export TT_BIO_TRIATT_HIFI_PAD_UP=0 || export TT_BIO_TRIATT_HIFI_PAD_UP=2
  out=out/of3_hsa_$arm
  clk=out/clk/of3_hsa_$arm.txt; : > "$clk"
  ( while :; do
      printf '%s %s\n' "$(date -u +%FT%TZ)" \
        "$(cat /sys/class/tenstorrent/tenstorrent\!$NODE/tt_aiclk 2>/dev/null || echo NA)" >> "$clk"
      sleep 5
    done ) & CLKPID=$!
  echo "=== ARM $arm PAD_UP=$TT_BIO_TRIATT_HIFI_PAD_UP seed=$SEED $(date -u +%FT%TZ)"
  INERT_TAG=of3_hsa_$arm timeout 3000 ~/bcx_e2e_venv/bin/python out/inert_drv.py examples/hsa.yaml \
    --out_dir "$out" --model openfold3 --seed $SEED --msa_dir "$MSA" --msa_cache_only --override \
    > out/of3_hsa_$arm.log 2>&1
  echo "=== ARM $arm rc=$? $(date -u +%FT%TZ)"
  kill $CLKPID 2>/dev/null
  grep -E "INERT_DUMP|Error|FATAL|Traceback|refus|Out of Memory" out/of3_hsa_$arm.log | tail -6
  sort -k2 -n "$clk" | awk 'NF==2 && $2!="NA"{v[n++]=$2} END{if(n)printf "CLK %s samples min %s median %s max %s MHz\n",n,v[0],v[int(n/2)],v[n-1]}'
done
echo "=== OF3 HSA AB DONE $(date -u +%FT%TZ)"
