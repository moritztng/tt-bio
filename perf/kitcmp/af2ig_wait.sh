#!/bin/bash
# Box side: wait for the BindCraft chain to finish, hold its done mark, run stock AF2-IG alone on the GPU, release the mark.
cd /root/kc
mkdir -p /root/af2ptm/params && ln -sf /root/.cache/bindcraft/alphafold/params_model_1_ptm.npz /root/af2ptm/params/
until [ -e CHAIN-DONE ]; do sleep 2; done
mv CHAIN-DONE CHAIN-DONE.hold
mkdir -p results/af2ig
nvidia-smi -q -d POWER,CLOCK > results/af2ig/power-clock.txt
timeout 1500 /opt/mamba/envs/cdkit/bin/python af2ig_gpu.py --params /root/af2ptm --warm 5 \
  results/bindcraft/kit-off/s0/design.pdb results/bindcraft/kit-off/s1/design.pdb results/bindcraft/kit-off/s2/design.pdb \
  > results/af2ig/runs.jsonl 2> results/af2ig/stderr.log
echo $? > results/af2ig/rc
mv CHAIN-DONE.hold CHAIN-DONE
