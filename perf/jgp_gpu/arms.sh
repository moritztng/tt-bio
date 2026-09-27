#!/usr/bin/env bash
# After off1: the kit's exact and fast modes (compile + self-check land in each process's r0 fold, which is not timed) and vanilla at fp32,
# back to back on the same card, never two at once.
cd /root/kit/protenix_v2 && . venv/bin/activate && export CUDA_HOME=/usr/local/cuda PROTENIX_ROOT_DIR=/weights/protenix
cd /root/jgp
MODE=exact bash run_off.sh exact1
MODE=fast bash run_off.sh fast1
DTYPE=fp32 bash run_off.sh fp32
echo ARMS-DONE
