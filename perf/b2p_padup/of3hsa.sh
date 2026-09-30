#!/bin/bash
# OF3 on HSA 585 aa (axis 608) with the committed 1000-seq MSA: pad-up on/off at seed 0, and seed 1 for the floor.
cd /home/ttuser/.coworker/wt/b2p-padup
while kill -0 "$1" 2>/dev/null; do sleep 10; done
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:b2p-padup
A=$(ls docs/implementation-parity-data/ref-fixtures/protenix-v2/hsa/*/msa.a3m)
P=~/tt-bio-dev/env/bin/python3
timeout 1500 $P perf/b2p_padup/fold_ab.py --model openfold3 --size 608 --yaml examples/hsa.yaml --a3m $A \
  --arms off,on,off,on --seed 0 --out perf/b2p_padup/of3_hsa608_s0_p300c.json --savecifs perf/b2p_padup/out/of3hsa > perf/b2p_padup/out/of3_hsa608_s0.log 2>&1
echo "s0 rc=$?" >> perf/b2p_padup/out/of3hsa.done
timeout 1200 $P perf/b2p_padup/fold_ab.py --model openfold3 --size 608 --yaml examples/hsa.yaml --a3m $A \
  --arms off,on --seed 1 --out perf/b2p_padup/of3_hsa608_s1_p300c.json --savecifs perf/b2p_padup/out/of3hsa_s1 > perf/b2p_padup/out/of3_hsa608_s1.log 2>&1
echo "s1 rc=$?" >> perf/b2p_padup/out/of3hsa.done
