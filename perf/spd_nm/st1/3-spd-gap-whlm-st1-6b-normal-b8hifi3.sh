[ -f ~/japanfold/env.sh ] && . ~/japanfold/env.sh > /dev/null 2>&1
# mem: 96
# cap: 2400
# threads: 8
# st1: 6B normal with the default b8 weights, HiFi4 vs HiFi3 in one process (grade on arm vs st1/6b-normal/base)
O=/home/ubuntu/spd/spd-gap-whlm/st1/6b-normal-b8hifi3
[ -s $O/ab.jsonl ] && exit 0
cd /home/ubuntu/spd/spd-gap-whlm/trees/tacd518ff1 || exit 3
bash perf/spd_gap_whlm/run.sh $CHIP $O -- timeout -s INT 1600 python perf/spd_gap_esmf/esmc_ab.py --fidelity HiFi3 --model esmc-6b --pdb perf/bgx_size/targets/hHSA.pdb --ref-cache /home/ubuntu/spd/spd-gap-whlm/refcache --rounds 2 --out $O/ab.part.jsonl && mv $O/ab.part.jsonl $O/ab.jsonl
