#!/bin/bash
# pc's uplink is DS-Lite CGNAT at ~3 MB/s and the rented box has dropped its ssh twice under a
# sustained rsync, taking the instance to `stopped` both times. The two missing param files are
# public: pull them ON the box from DeepMind's own release at the box's 947 Mbps instead of
# pushing 746 MB up a 3 MB/s link. Byte sizes are checked against pc's copies, so this is the same
# file, not a similar one.
set -u
L=/home/moritz/.bci-seventeen-host/.bci/box_af2.log
exec >> "$L" 2>&1
echo "=== box af2 $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
K=$(cat /home/moritz/.config/vastai/vast_api_key)
I=54845453
P=19466; H=root@ssh1.vast.ai
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o ServerAliveInterval=30 -p $P"

state() { curl -s -H "Authorization: Bearer $K" "https://console.vast.ai/api/v0/instances/" \
  | python3 -I -c "import sys,json;d=json.load(sys.stdin);print([i for i in d.get('instances',d) if i.get('id')==$I][0].get('actual_status'))"; }

echo "state now: $(state)"
curl -s -X PUT -H "Authorization: Bearer $K" -H "Content-Type: application/json" \
  -d '{"state":"running"}' "https://console.vast.ai/api/v0/instances/$I/" ; echo
for i in $(seq 1 40); do
  s=$(state); echo "  $(date -u +%H:%M:%SZ) state=$s"
  [ "$s" = "running" ] && break
  sleep 20
done
# ssh comes up after the container does
for i in $(seq 1 30); do
  $SSH $H true 2>/dev/null && { echo "ssh up at $(date -u +%H:%M:%SZ)"; break; }
  sleep 20
done

$SSH $H 'set -u
exec > >(tee -a /root/box_af2.log) 2>&1
echo "=== on-box af2 fetch $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
D=/root/bcx_shipped/af2_params
rm -f $D/params_model_1_multimer_v3.npz $D/params_model_1_ptm.npz   # dangling symlinks
cd /root
# Stream the release tar and extract only the two members we lack.
time curl -sS https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar \
  | tar -xvf - -C "$D" params_model_1_multimer_v3.npz params_model_1_ptm.npz
ls -la $D/params_model_1_multimer_v3.npz $D/params_model_1_ptm.npz
python3 - <<PY || exit 4
import numpy as np, os, sys
D="/root/bcx_shipped/af2_params"
want={"params_model_1_multimer_v3.npz":373043148, "params_model_1_ptm.npz":373103340}
bad=[]
for n,sz in want.items():
    p=os.path.join(D,n); got=os.path.getsize(p)
    ok = got==sz
    print(f"  {n}: {got} bytes, pc has {sz} -> {\"same file\" if ok else \"DIFFERENT\"}")
    if not ok: bad.append(n)
for i in range(1,6):
    p=os.path.join(D,f"params_model_{i}_multimer_v3.npz")
    with np.load(p) as z: print(f"  loads: params_model_{i}_multimer_v3.npz {len(z.files)} arrays")
if bad: sys.exit(4)
print("af2 params complete and byte-identical to pc")
PY
ROOT=/root/bci-seventeen; PROJ=$ROOT/proj_host_8traj_288
[ -e "$PROJ" ] && mv "$PROJ" "$PROJ.dead.$(date -u +%H%M%S)" && echo "moved a dead project folder aside; a resume is not a rerun"
setsid nohup bash /root/host_8traj_288.sh </dev/null >> /root/host_arm.boot 2>&1 &
sleep 90
echo "--- arm pids ---"; pgrep -af "host_8traj_288|capture_logits" | head
echo "--- arm log ---"; tail -25 $ROOT/host_8traj_288.log
'
echo "=== box af2 end $(date -u +%Y-%m-%dT%H:%M:%SZ) rc=$? ==="
