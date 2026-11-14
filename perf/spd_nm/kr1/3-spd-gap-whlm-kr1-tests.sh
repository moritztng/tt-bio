[ -f ~/japanfold/env.sh ] && . ~/japanfold/env.sh > /dev/null 2>&1
# mem: 64
# cap: 2400
# threads: 8
[ -s /home/ubuntu/spd/spd-gap-whlm/kr1/tests/rc.tests ] && exit 0
cd /home/ubuntu/spd/spd-gap-whlm/trees/t8582cedbf && bash perf/spd_gap_whlm/run.sh $CHIP /home/ubuntu/spd/spd-gap-whlm/kr1/tests -- timeout -s INT 2000 python -m pytest -q tests/test_esmc_sdpa_chunks.py tests/test_esmc_mm_default.py; echo $? > /home/ubuntu/spd/spd-gap-whlm/kr1/tests/rc.tests
