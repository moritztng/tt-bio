#!/bin/bash
# On the rented box (/root/kc): upstream OpenDDE 1.0.3 (fp32, upstream default TF32) and Boltz-2 2.2.1 (default bf16-mixed)
# fold 9DBP at seeds 101 and 202, one sample, 200 steps, recycles as tt-bio ships them (perf/mgx/ref/plan.json: OpenDDE 10, Boltz-2 3), the reference for the bond-error-by-|x| check (spd-q-bondradial).
# Same sequences and a3m (first 2048 rows per chain) as the TT and CPU cells. Writes /root/kc/pvo/<model>/s<seed>.cif.gz,
# then PVO-DONE (or PVO-FAILED when no structure came out), the markers pvo_session-style watchers read.
set -u
cd /root/kc; O=/root/kc/pvo; mkdir -p $O
log(){ echo "$(date -u +%FT%TZ) $*"; }
nvidia-smi --query-gpu=name,memory.total,power.limit --format=csv | tee $O/gpu.txt
apt-get update -qq && apt-get install -y -qq --no-install-recommends curl ca-certificates python3 build-essential > $O/apt.log 2>&1
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH=$HOME/.local/bin:$PATH
uv venv -q -p 3.11 /root/v-ode && uv pip install -q -p /root/v-ode/bin/python --torch-backend cu126 "opendde==1.0.3" huggingface_hub > $O/ode_install.log 2>&1
uv venv -q -p 3.11 /root/v-boltz && uv pip install -q -p /root/v-boltz/bin/python --torch-backend cu126 "boltz==2.2.1" > $O/boltz_install.log 2>&1
/root/v-ode/bin/python -c "from huggingface_hub import hf_hub_download as h; print(h('aurekaresearch/OpenDDE','opendde.pt',local_dir='/root/ckpt'))"
sed -i 's#/home/moritz/.coworker/artifacts/spd-q-bondradial/ode_cpu/msa/#/root/kc/#' 9DBP.json
/root/v-ode/bin/python - <<'PY'
import json
js = json.load(open('/root/kc/9DBP.json'))[0]["sequences"]
y = ["version: 1", "sequences:"]
for cid, s in zip("AB", js):
    p = s["proteinChain"]
    y += ["- protein:", f"    id: {cid}", f"    sequence: {p['sequence']}", f"    msa: {p['unpairedMsaPath']}"]
open('/root/kc/9DBP.yaml', 'w').write("\n".join(y) + "\n")
PY
for s in 101 202; do
  log "opendde s$s"
  /root/v-ode/bin/python -I ode_pred.py pred -i 9DBP.json -o /root/out/ode_s$s -s $s -c 10 -p 200 -e 1 -d fp32 \
    -n opendde_v1 --load_checkpoint_path /root/ckpt/opendde.pt --use_msa true --trimul_kernel torch --triatt_kernel torch \
    > $O/ode_s$s.log 2>&1
  f=$(find /root/out/ode_s$s -name "*.cif" | head -1); [ -n "$f" ] && mkdir -p $O/opendde && gzip -c "$f" > $O/opendde/s$s.cif.gz
  log "boltz2 s$s"
  /root/v-boltz/bin/boltz predict /root/kc/9DBP.yaml --out_dir /root/out/b2_s$s --cache /root/.boltz --devices 1 \
    --accelerator gpu --seed $s --recycling_steps 3 --sampling_steps 200 --diffusion_samples 1 --output_format mmcif \
    --no_kernels --override > $O/b2_s$s.log 2>&1
  f=$(find /root/out/b2_s$s -name "*_model_0.cif" | head -1); [ -n "$f" ] && mkdir -p $O/boltz2 && gzip -c "$f" > $O/boltz2/s$s.cif.gz
done
ls -R $O > $O/listing.txt
if ls $O/*/s*.cif.gz >/dev/null 2>&1; then touch $O/PVO-DONE; else echo "no structure" > $O/PVO-FAILED; fi
