#!/bin/bash
# gpu_b relaunch at d4253429c (seed offset reaches the step seeds); gpu_a (pid 3808, leg subshell 3806) runs on untouched.
export DATA_PATH=/root/data_path HF_HOME=/root/hf RFD4_REPO=/root/rfd4
B=/root/kc/rfd4; C=$B/curve; cd /root/kc; PY=/opt/rfd4-venv/bin/python
say(){ echo "$(date -u +%FT%TZ) curve_b2: $*"; }
mkdir -p src2 && tar -xzf r4t_curve_src2.tgz -C src2
mv $C/gpu_b.jsonl $C/gpu_b.offset-ignored.jsonl; mv $C/gpu_b.log $C/gpu_b.offset-ignored.log; rm -f $C/gpu_b.rc
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8 $PY src2/perf/r4t/curve_upstream.py --plan $C/plan.json   --out $C/gpu_b.jsonl --side gpu_b --seed-offset 1 --device cuda --sha "src-df1caadc3c064276" > $C/gpu_b.log 2>&1
echo $? > $C/gpu_b.rc; say "gpu_b rc=$(cat $C/gpu_b.rc)"
while [ ! -f $C/gpu_a.rc ]; do sleep 30; done
[ "$(cat $C/gpu_a.rc)$(cat $C/gpu_b.rc)" = 00 ] || { echo "leg rc a=$(cat $C/gpu_a.rc) b=$(cat $C/gpu_b.rc)" > $B/RFD4-FAILED; exit 4; }
touch $B/RFD4-DONE; say done
