#!/bin/bash
# Runs ON the rented B200 box for BC4 phase 0, rules 3 and 4: BindCraft 2 at NVIDIA's best, then the B200's own
# iteration census, then bc4-quality's reference distribution with whatever time is left.
#   bash box_bc4.sh <cap_h>       markers /root/kc/bc4/{ARMS-DONE,CENSUS-DONE,QUALITY-DONE,BC4-DONE}
# Everything lands under /root/kc/bc4 and session.sh pulls that directory every few minutes, so a box that dies
# mid-way still leaves every finished arm behind.
set -u
CAP_H=${1:-1.8}; T0=$(date +%s); DEADLINE=$(python3 -c "print(int($T0 + $CAP_H*3600 - 900))")
B=/root/kc/bc4; mkdir -p $B; cd /root/kc
say(){ echo "$(date -u +%FT%TZ) box_bc4: $*"; }

# --- the stock BindCraft 2 stack, at the frozen commit (box_bc_setup.sh bc2 skips the colabdesign kit) ---------------
bash box_bc_setup.sh bc2 > $B/setup.log 2>&1 || { say "setup failed"; tail -20 $B/setup.log; exit 2; }
say "setup done after $(( $(date +%s) - T0 ))s"

nvidia-smi -q -d POWER,CLOCK,PERFORMANCE > $B/power-clock.txt 2>&1
nohup nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,utilization.gpu,utilization.memory,clocks_event_reasons.active,memory.used \
  --format=csv,noheader -l 1 > $B/smi.csv 2>&1 &
SMI=$!
trap 'kill $SMI 2>/dev/null' EXIT

# --- rule 4: every arm, then the winner at --warm 9 ------------------------------------------------------------------
/opt/bc2-venv/bin/python /root/kc/bc4_arms.py --bc2 /root/bc2 --out $B --smi $B/smi.csv \
  --screen-warm ${SCREEN_WARM:-2} --final-warm ${FINAL_WARM:-9} --deadline $DEADLINE > $B/arms.log 2>&1
echo $? > $B/ARMS-DONE; say "arms done after $(( $(date +%s) - T0 ))s"

# --- deliverable B: the census, on one warm trajectory ---------------------------------------------------------------
if [ $(date +%s) -lt $(( DEADLINE - 600 )) ]; then
  /opt/bc2-venv/bin/python /root/kc/bc4_census.py --bc2 /root/bc2 --out $B/census > $B/census.log 2>&1
  echo $? > $B/CENSUS-DONE; say "census done after $(( $(date +%s) - T0 ))s"
fi

# --- bc4-quality's stock-GPU reference distribution, with the time that is left ---------------------------------------
# QUALITY.json (pushed by session.sh if bc4-quality has written its request): {"targets": [{"name":..., "settings":
# "<path on the box>", "overrides": [...]}, ...], "trajectories": N}. Stock settings, so no arm switches here.
if [ -f /root/kc/quality-req.json ] && [ $(date +%s) -lt $(( DEADLINE - 600 )) ]; then
  /opt/bc2-venv/bin/python - "$DEADLINE" <<'PY' > $B/quality.log 2>&1
import json, subprocess, sys, time
from pathlib import Path
deadline = float(sys.argv[1])
req = json.loads(Path("/root/kc/quality-req.json").read_text())
out = Path("/root/kc/bc4/quality"); out.mkdir(parents=True, exist_ok=True)
for t in req.get("targets", []):
    seeds = t.get("seeds") or req.get("seeds") or [101, 102, 103, 104]
    for seed in seeds:
        if time.time() > deadline - 600:
            print("out of time before seed", seed, "of", t.get("name")); break
        d = out / f"{t.get('name','target')}-s{seed}"
        cmd = ["/opt/bc2-venv/bin/python", "/root/kc/bc2_gpu.py", "--bc2", "/root/bc2", "--out", str(d),
               "--seed", str(seed), "--warm", str(req.get("warm", 9)),
               "--binder-len", str(t.get("binder_len", req.get("binder_len", 80)))]
        for s in t.get("overrides", []):
            cmd += ["--set", s]
        print(time.strftime("%FT%TZ", time.gmtime()), "running", d, flush=True)
        print("rc", subprocess.call(cmd), flush=True)
PY
  echo $? > $B/QUALITY-DONE; say "quality reference done after $(( $(date +%s) - T0 ))s"
fi

echo done > $B/BC4-DONE; say "BC4-DONE after $(( $(date +%s) - T0 ))s"
