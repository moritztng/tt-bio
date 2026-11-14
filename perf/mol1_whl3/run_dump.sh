#!/bin/bash
# upstream Mol-1 CPU fp32 trunk dump on pc (moved off qb1, D-orch-78)
cd /home/moritz/mol1-whl3/scr
export MOL_ROOT_DIR=/home/moritz/mol1-upstream/io-data
PY=/home/moritz/mol1-upstream/io-venv/bin/python
for c in 8fg6 8k83; do
  echo "== $c start $(date -u +%FT%TZ)"
  nice -n 19 $PY -I trunk_dump.py up_${c}_trunk.pt -- --input_json_path in_q_${c}-assembly1.json \
    --dump_dir out_$c --load_checkpoint_path /home/moritz/mol1-whl3/ckpt/mol-1-base.pt --seeds 42 --use_seeds_in_json false \
    --device cpu --triangle_attention torch --triangle_multiplicative torch --sample_diffusion.N_step 2 --sample_diffusion.N_sample 1 > log_dump_$c.txt 2>&1
  echo "== $c rc=$? $(date -u +%FT%TZ)"
done
echo DUMP-EXIT
