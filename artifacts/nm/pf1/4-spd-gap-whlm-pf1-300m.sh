[ -f ~/japanfold/env.sh ] && . ~/japanfold/env.sh > /dev/null 2>&1
# mem: 32
# cap: 1200
# threads: 4
[ -s /home/ubuntu/spd/spd-gap-whlm/pf1/300m/rc ] && exit 0
mkdir -p /home/ubuntu/spd/spd-gap-whlm/pf1/300m/prof
cd /home/ubuntu/spd/spd-gap-whlm/trees/t45afeda34 && TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_DIR=/home/ubuntu/spd/spd-gap-whlm/pf1/300m/prof bash perf/spd_gap_whlm/run.sh $CHIP /home/ubuntu/spd/spd-gap-whlm/pf1/300m -- timeout -s INT 900 python perf/shsdpa/bench.py /home/ubuntu/spd/spd-gap-whlm/pf1/300m/bench.json --shape esmc --arms e_auto --host --reps 2 --ncall 3 --bh 7,15 --lens 514,514,514,514,514,514,514
find /home/ubuntu /tmp -name 'profile_log_device.csv' -newer /home/ubuntu/spd/spd-gap-whlm/pf1/300m/prof 2>/dev/null > /home/ubuntu/spd/spd-gap-whlm/pf1/300m/csvs.txt
python3 -c "import ttnn,os;print(os.path.dirname(ttnn.__file__))" >> /home/ubuntu/spd/spd-gap-whlm/pf1/300m/csvs.txt 2>/dev/null
