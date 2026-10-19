[ -f ~/japanfold/env.sh ] && . ~/japanfold/env.sh > /dev/null 2>&1
# mem: 32
# cap: 1500
# threads: 4
[ -s /home/ubuntu/spd/spd-gap-whlm/nm5/sp650m/l514.json ] && exit 0
mkdir -p /home/ubuntu/spd/spd-gap-whlm/nm5/sp650m
cd /home/ubuntu/spd/spd-gap-whlm/trees/te87740ee5 && bash perf/spd_gap_whlm/run.sh $CHIP /home/ubuntu/spd/spd-gap-whlm/nm5/sp650m -- timeout -s INT 1200 python perf/shsdpa/bench.py /home/ubuntu/spd/spd-gap-whlm/nm5/sp650m/l514.part.json --shape esmc --bh 7,20 --lens 514 --proto "0:192:-:EXPA" --arms e_auto,e_p0_192_-_EXPA --reps 9 && mv /home/ubuntu/spd/spd-gap-whlm/nm5/sp650m/l514.part.json /home/ubuntu/spd/spd-gap-whlm/nm5/sp650m/l514.json
