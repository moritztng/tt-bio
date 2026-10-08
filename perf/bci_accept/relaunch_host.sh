set -u
L=/home/moritz/.bci-seventeen-host/.bci/box_relaunch.log
exec >> "$L" 2>&1
echo "=== relaunch $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
K=$(cat /home/moritz/.config/vastai/vast_api_key)
I=54845453
P=19466; H=root@ssh1.vast.ai
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o ServerAliveInterval=30 -p $P"
state() { curl -s -H "Authorization: Bearer $K" "https://console.vast.ai/api/v0/instances/" \
  | python3 -I -c "import sys,json;d=json.load(sys.stdin);print([i for i in d.get('instances',d) if i.get('id')==$I][0].get('actual_status'))"; }
echo "state now: $(state)"
curl -s -X PUT -H "Authorization: Bearer $K" -H "Content-Type: application/json" \
  -d '{"state":"running"}' "https://console.vast.ai/api/v0/instances/$I/"; echo
for i in $(seq 1 40); do s=$(state); echo "  $(date -u +%H:%M:%SZ) state=$s"; [ "$s" = "running" ] && break; sleep 20; done
for i in $(seq 1 30); do $SSH $H true 2>/dev/null && { echo "ssh up at $(date -u +%H:%M:%SZ)"; break; }; sleep 20; done

# Ship the recovery script every firing. The box keeps /root across stops, so a stale copy there
# silently outlives a fix: the orphan-folder sweep landed at 21:1xZ and the box was still running
# the version that stalled the arm six times. Pushing it here means the fix applies on the next
# relaunch without touching the watchdog.
scp -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=20 -P $P \
  /home/moritz/.bci-seventeen-host/host_recover.py $H:/root/host_recover.py \
  && echo "shipped host_recover.py ($(md5sum /home/moritz/.bci-seventeen-host/host_recover.py | cut -c1-8))" \
  || echo "WARN: could not ship host_recover.py; the box keeps its previous copy"

$SSH $H 'set -u
exec > >(tee -a /root/box_relaunch.log) 2>&1
echo "=== on-box relaunch $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
python3 - <<PY || exit 4
import numpy as np, os, sys
D="/root/bcx_shipped/af2_params"
want={"params_model_1_multimer_v3.npz":373043148, "params_model_1_ptm.npz":373103340}
bad=[]
for n,sz in want.items():
    p=os.path.join(D,n); got=os.path.getsize(p)
    ok = got==sz
    verdict = "same file" if ok else "DIFFERENT"
    print("  %s: %d bytes, pc has %d -> %s" % (n, got, sz, verdict))
    if not ok: bad.append(n)
for i in range(1,6):
    p=os.path.join(D,"params_model_%d_multimer_v3.npz" % i)
    with np.load(p) as z: print("  loads: params_model_%d_multimer_v3.npz %d arrays" % (i, len(z.files)))
if bad: sys.exit(4)
print("af2 params complete and byte-identical to pc")
PY
python3 -c "import jax,sys; d=jax.devices(); print(\"jax devices:\",d); sys.exit(0 if d[0].platform==\"gpu\" else 5)" || exit 5
ROOT=/root/bci-seventeen; PROJ=$ROOT/proj_host_8traj_288
python3 -I /root/host_recover.py "$PROJ" || exit 6
setsid nohup bash /root/host_8traj_288.sh </dev/null >> /root/host_arm.boot 2>&1 &
sleep 120
echo "--- arm pids ---"; pgrep -af "host_8traj_288|capture_logits" | head
echo "--- arm log ---"; tail -30 $ROOT/host_8traj_288.log
'
echo "=== relaunch end $(date -u +%Y-%m-%dT%H:%M:%SZ) rc=$? ==="
