[ -f ~/japanfold/env.sh ] && . ~/japanfold/env.sh > /dev/null 2>&1
# mem: 64
# cap: 2400
# threads: 8
[ -s /home/ubuntu/spd/spd-gap-whlm/fd2/esmc-6b-normal-HiFi2/ab.jsonl ] && exit 0
cd /home/ubuntu/spd/spd-gap-whlm/trees/t6ee8ed8d5 && bash perf/spd_gap_whlm/run.sh $CHIP /home/ubuntu/spd/spd-gap-whlm/fd2/esmc-6b-normal-HiFi2 -- timeout -s INT 2000 python perf/spd_gap_esmf/esmc_ab.py --fidelity HiFi2 --model esmc-6b  --pdb perf/bgx_size/targets/hHSA.pdb --ref-cache /home/ubuntu/spd/spd-gap-whlm/refcache --rounds 3 --out /home/ubuntu/spd/spd-gap-whlm/fd2/esmc-6b-normal-HiFi2/ab.part.jsonl && mv /home/ubuntu/spd/spd-gap-whlm/fd2/esmc-6b-normal-HiFi2/ab.part.jsonl /home/ubuntu/spd/spd-gap-whlm/fd2/esmc-6b-normal-HiFi2/ab.jsonl
