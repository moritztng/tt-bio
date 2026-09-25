#!/bin/bash
# Blackhole capture census on pc card 0: bytes per capture only (short steps, no MSA), one
# trace.jsonl per run under out_bh/<tag>. Same hook as chain.py.
cd "$(dirname "$0")/../.." || exit 1
H=perf/mgx_trace_region; F=perf/size512/fixtures; PY=${PY:-$HOME/tt-bio/env/bin/python}
run() {
  tag=$1; shift; d=$H/out_bh/$tag; mkdir -p $d; rm -f $d/trace.jsonl
  echo "START $(date -u +%FT%TZ) $tag $*" > $d/run.log
  env TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:mgx-trace-region \
      TT_METAL_LOGGER_LEVEL=FATAL TRACE_REGION_LOG=$PWD/$d/trace.jsonl PYTHONPATH=$PWD/$H/hook:$PWD \
      timeout 2400 $PY -m tt_bio.main "$@" >> $d/run.log 2>&1 < /dev/null
  echo "$tag rc=$? $(cat $d/trace.jsonl 2>/dev/null | wc -l) captures" | tee -a $H/out_bh/summary.txt
}
P="--diffusion_samples 1 --seed 0 --override --single_sequence --sampling_steps 5"
run b2_512 predict $F/cdk2x2_512.yaml --model boltz2 --out_dir $H/out_bh/b2_512/o $P --diffusion_trace
run px2_512 predict $F/cdk2x2_512.yaml --model protenix-v2 --out_dir $H/out_bh/px2_512/o $P --trace
run esmc_fill8 embed $H/inputs/esmc_fill8.fasta --model esmc-300m --batch_size 1 --out_dir $H/out_bh/esmc_fill8/o
run b2_1536 predict $F/cdk2x2_1536.yaml --model boltz2 --out_dir $H/out_bh/b2_1536/o $P --diffusion_trace
run px2_1536 predict $F/cdk2x2_1536.yaml --model protenix-v2 --out_dir $H/out_bh/px2_1536/o $P --trace
RFD3_TRACE_DECODER=1 run rfd3_dec design examples/rfd3_binder.json --model rfd3 --from_pdb --num_designs 1 --seed 0 --num_timesteps 20 --out_dir $H/out_bh/rfd3_dec/o
echo BH_DONE >> $H/out_bh/summary.txt
