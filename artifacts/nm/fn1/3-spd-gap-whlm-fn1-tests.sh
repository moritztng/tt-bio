[ -f ~/japanfold/env.sh ] && . ~/japanfold/env.sh > /dev/null 2>&1
# mem: 32
# cap: 2400
# threads: 8
[ -s /home/ubuntu/spd/spd-gap-whlm/fn1/tests/rc.tests ] && exit 0
mkdir -p /home/ubuntu/spd/spd-gap-whlm/fn1/tests
cd /home/ubuntu/spd/spd-gap-whlm/trees/t7a5683255 && bash perf/spd_gap_whlm/run.sh $CHIP /home/ubuntu/spd/spd-gap-whlm/fn1/tests -- timeout -s INT 2000 sh -c 'PYTHONPATH=$PYTHONPATH:/home/ubuntu/spd/spd-gap-whlm/pytestpath python -m pytest -q -p no:cacheprovider tests/test_esmc_sdpa_chunks.py tests/test_esmc_mm_default.py tests/test_esmfold2_wormhole_fast.py'; echo $? > /home/ubuntu/spd/spd-gap-whlm/fn1/tests/rc.tests
