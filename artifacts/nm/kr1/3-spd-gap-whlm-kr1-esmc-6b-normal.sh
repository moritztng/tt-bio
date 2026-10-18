[ -f ~/japanfold/env.sh ] && . ~/japanfold/env.sh > /dev/null 2>&1
# mem: 64
# cap: 2400
# threads: 8
[ -s /home/ubuntu/spd/spd-gap-whlm/kr1/esmc-6b-normal/ab.jsonl ] && exit 0
cd /home/ubuntu/spd/spd-gap-whlm/trees/t8582cedbf && bash perf/spd_gap_whlm/run.sh $CHIP /home/ubuntu/spd/spd-gap-whlm/kr1/esmc-6b-normal -- timeout -s INT 2000 python perf/spd_gap_esmf/esmc_ab.py --alt tt_bio.shsdpa._KEYPAD_ROW --model esmc-6b  --pdb perf/bgx_size/targets/hHSA.pdb --ref-cache /home/ubuntu/spd/spd-gap-whlm/refcache --rounds 3 --out /home/ubuntu/spd/spd-gap-whlm/kr1/esmc-6b-normal/ab.part.jsonl && mv /home/ubuntu/spd/spd-gap-whlm/kr1/esmc-6b-normal/ab.part.jsonl /home/ubuntu/spd/spd-gap-whlm/kr1/esmc-6b-normal/ab.jsonl
