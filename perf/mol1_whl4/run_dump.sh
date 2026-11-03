#!/bin/bash
# upstream Mol-1 CPU fp32, 7fxn s42: per-block trunk single track + trunk outputs (mol1-whl4)
cd /home/moritz/mol1-whl4/scr
export MOL_ROOT_DIR=/home/moritz/mol1-upstream/io-data
echo "start $(date -u +%FT%TZ)"
nice -n 19 /home/moritz/mol1-upstream/io-venv/bin/python -I dump.py up_7fxn -- --input_json_path in_7fxn.json \
  --dump_dir out_7fxn --load_checkpoint_path /home/moritz/mol1-whl3/ckpt/mol-1-base.pt --seeds 42 --use_seeds_in_json false \
  --device cpu --triangle_attention torch --triangle_multiplicative torch --sample_diffusion.N_step 2 --sample_diffusion.N_sample 1 > log_dump_7fxn.txt 2>&1
echo "rc=$? $(date -u +%FT%TZ)"; echo DUMP-EXIT
