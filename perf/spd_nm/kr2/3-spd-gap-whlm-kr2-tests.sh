[ -f ~/japanfold/env.sh ] && . ~/japanfold/env.sh > /dev/null 2>&1
# mem: 64
# cap: 2400
# threads: 8
[ -s /home/ubuntu/spd/spd-gap-whlm/kr2/tests/rc.tests ] && exit 0
cd /home/ubuntu/spd/spd-gap-whlm/trees/t525ee0eaf && bash perf/spd_gap_whlm/run.sh $CHIP /home/ubuntu/spd/spd-gap-whlm/kr2/tests -- timeout -s INT 2000 sh -c 'PYTHONPATH=$PYTHONPATH:/home/ubuntu/spd/spd-gap-whlm/pytestpath python -m pytest -q -p no:cacheprovider tests/test_esmc_sdpa_chunks.py tests/test_esmc_mm_default.py'; echo $? > /home/ubuntu/spd/spd-gap-whlm/kr2/tests/rc.tests
