#!/bin/bash
# W9: block-level A/B of the SDPA bias dtype, at both SDPA chunk configs.
#   main ships q=k=64 in the 256<N<=384 band; W6's branch ships the full-length q=k=320.
# Runs op_split.py --mode mods, monkeypatching the chunk config where asked.
cd /home/ttuser/.coworker/wt/perfwar-sdpa-kernel

run () {  # $1 label, $2 bias dtype ("" = bf16), $3 sdpa full-length (1/0)
  echo "=================== $1 ==================="
  TT_BIO_SDPA_BIAS_DTYPE="$2" W9_SDPA_FULL="$3" \
  perf/sdpa_kernel/rundev.sh -c "
import os, sys, runpy
import tt_bio.tenstorrent as T
if os.environ.get('W9_SDPA_FULL') == '1':
    T._tri_att_sdpa_program_config = lambda q, k: T._sdpa_program_config(
        q_chunk_size=q, k_chunk_size=k)
sys.argv = ['op_split.py', '--mode', 'mods', '--n', '320', '--iters', '5',
            '--out', 'perf/sdpa_kernel/block_$1.json']
runpy.run_path('perf/attn_block/op_split.py', run_name='__main__')
" 2>&1 | grep -E "^(tri_att|transition|resid|s_track|FULL_BLOCK|sum\(parts\)|layer built)"
}

run bf16_chunk64   ""          0
run bfp8_chunk64   "bfloat8_b" 0
run bf16_chunk320  ""          1
run bfp8_chunk320  "bfloat8_b" 1
run bfp4_chunk320  "bfloat4_b" 1
