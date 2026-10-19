[ -f ~/japanfold/env.sh ] && . ~/japanfold/env.sh > /dev/null 2>&1
# mem: 32
# cap: 1800
# threads: 4
# hf2: page cell, normal (bf16 weights, fp32 acc) at the model's HiFi3 vs HiFi2, every op. Normal bar: pcc vs fp32 within 0.0001.
O=/home/ubuntu/spd/spd-gap-whlm/hf2/esmc-300m-normal
[ -s $O/ab.jsonl ] && exit 0
cd /home/ubuntu/spd/spd-gap-whlm/trees/t3ff7185f5 || exit 3
env  bash perf/spd_gap_whlm/run.sh $CHIP $O -- timeout -s INT 1600 python perf/spd_gap_esmf/esmc_ab.py --fidelity HiFi2 --model esmc-300m --pdb perf/bgx_size/targets/hHSA.pdb --ref-cache /home/ubuntu/spd/spd-gap-whlm/refcache --rounds 3 --out $O/ab.part.jsonl && mv $O/ab.part.jsonl $O/ab.jsonl
