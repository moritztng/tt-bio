#!/bin/bash
# upstream Mol-1 CPU fp32 per-PairformerBlock single track, 8k83 s42 (mol1-whl3 bisect)
cd /home/moritz/mol1-whl3/scr
export MOL_ROOT_DIR=/home/moritz/mol1-upstream/io-data
echo "start $(date -u +%FT%TZ)"
nice -n 19 /home/moritz/mol1-upstream/io-venv/bin/python -I layer_dump.py up_8k83_layers.pt -- --input_json_path in_q_8k83-assembly1.json \
  --dump_dir out_l8k83 --load_checkpoint_path /home/moritz/mol1-whl3/ckpt/mol-1-base.pt --seeds 42 --use_seeds_in_json false \
  --device cpu --triangle_attention torch --triangle_multiplicative torch --sample_diffusion.N_step 2 --sample_diffusion.N_sample 1 > log_layers_8k83.txt 2>&1
echo "rc=$? $(date -u +%FT%TZ)"; echo LAYERS-EXIT
