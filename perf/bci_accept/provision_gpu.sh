#!/bin/bash
# bci-seventeen: provision the rented GPU box for the HOST-JAX arm of #17.
#
# Runs on pc, which already holds exactly what the arm needs: bcx_shipped/bc2 (BindCraft 2 v1.0.1
# with its ProteinMPNN weights) and a 1.8 GB af2_params, the same files bci-accept's host arms
# used. Shipping pc's copy rather than qb1's 5.3 GB superset keeps weight parity with the prior
# host arm and moves a third of the bytes.
set -u
ROOT=/home/moritz/.bci-seventeen-host
read -r HOST P < "$ROOT/endpoint.txt"
H=root@$HOST
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -p $P"
LOG=$ROOT/.bci/provision.log
exec >> "$LOG" 2>&1
date -u +"=== provision start %Y-%m-%dT%H:%M:%SZ ==="

echo "--- bc2 + af2 params ---"
rsync -a --info=progress2 -e "$SSH" "$ROOT/../bcx_shipped/" $H:/root/bcx_shipped/
echo "rsync bcx_shipped rc=$?"

echo "--- tt-bio host path, this branch ---"
rsync -a -e "$SSH" --exclude '__pycache__' "$ROOT/tt-bio/" $H:/root/tt-bio/
echo "rsync tt-bio rc=$?"

echo "--- scripts ---"
scp -P "$P" -o StrictHostKeyChecking=no \
  "$ROOT/requirements_host.txt" "$ROOT/remote_setup.sh" "$ROOT/host_8traj_288.sh" $H:/root/
echo "scp rc=$?"

echo "--- remote setup ---"
if ! $SSH $H 'bash /root/remote_setup.sh'; then
  echo "remote_setup FAILED; not launching the arm on an unproved box"
  date -u +"=== provision aborted %Y-%m-%dT%H:%M:%SZ ==="
  exit 1
fi
echo "remote_setup OK"

date -u +"=== provision end %Y-%m-%dT%H:%M:%SZ ==="

# Launch the arm from here rather than waiting for a pass to notice the box is ready. remote_setup
# asserts the GPU, the imports and the weights, and `set -e` aborts before this line if any failed,
# so reaching it means the box is proved. setsid on the far side so the arm outlives this ssh.
echo "--- launching the host arm ---"
$SSH $H 'setsid nohup bash /root/host_8traj_288.sh > /root/host_arm.boot 2>&1 < /dev/null & sleep 3; pgrep -af host_8traj_288 | head -2'
echo "launch rc=$?"
date -u +"=== arm launched %Y-%m-%dT%H:%M:%SZ ==="
