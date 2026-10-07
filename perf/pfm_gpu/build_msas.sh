#!/bin/bash
# Build the PopVax-shaped inputs' MSAs once, on pc, network only (ColabFold API via tt-bio's own calls).
# usage: build_msas.sh   (run from the worktree root; writes perf/pfm_gpu/msa/<pdb>/{A,B}.{paired,unpaired}.a3m)
cd "$(dirname "$0")/../.."
for y in perf/pfm_gpu/inputs/*.yaml; do
  n=$(basename "$y" .yaml)
  echo "$(date -u +%FT%TZ) start $n"
  PYTHONPATH=. timeout 3600 /home/moritz/tt-bio/env/bin/python perf/pfm_gpu/make_msa.py "$y" "perf/pfm_gpu/msa/$n"
  echo "$(date -u +%FT%TZ) $n rc=$?"
done
echo MSAS-DONE
