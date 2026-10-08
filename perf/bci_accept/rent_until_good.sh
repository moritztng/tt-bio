#!/bin/bash
# Rent a GPU box, PROVE it actually works, and move on to the next offer if it does not.
#
# Two boxes in a row failed in ways no amount of waiting fixes: one sat 15 minutes on a single
# image layer with its container never starting, the next could not inject its own GPU into the
# container ("unresolvable CDI devices ... gpu=2: unknown"). Both bill while broken. So the test
# is not "did vast accept the rental" but "does ssh answer and does nvidia-smi see the GPU",
# within a bounded window, and anything that fails is destroyed rather than nursed.
set -u
ROOT=/home/moritz/.bci-seventeen-host
LOG=$ROOT/.bci/rent.log
K=$(cat /home/moritz/.config/vastai/vast_api_key)
API=https://console.vast.ai/api/v0
IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
WINDOW=420          # seconds to give one box before writing it off
exec >> "$LOG" 2>&1
date -u +"=== rent_until_good start %Y-%m-%dT%H:%M:%SZ ==="

Q='{"gpu_ram":{"gte":24000},"num_gpus":{"eq":1},"disk_space":{"gte":100},"rentable":{"eq":true},"inet_down":{"gte":400},"cpu_cores_effective":{"gte":8},"cpu_ram":{"gte":28000},"reliability2":{"gte":0.95},"type":"on-demand","order":[["dph_total","asc"]],"limit":20}'
OFFERS=$(curl -s -G -H "Authorization: Bearer $K" --data-urlencode "q=$Q" "$API/bundles/" \
  | python3 -c "import sys,json;[print(o['id'],o['gpu_name'].replace(' ','_'),round(o['dph_total'],3)) for o in json.load(sys.stdin).get('offers',[])]")
echo "$OFFERS" | wc -l | xargs echo "candidate offers:"

while read -r ID NAME DPH; do
  [ -z "${ID:-}" ] && continue
  R=$(curl -s -H "Authorization: Bearer $K" -H "Content-Type: application/json" -X PUT "$API/asks/$ID/" \
      -d "{\"client_id\":\"me\",\"image\":\"$IMAGE\",\"disk\":100,\"label\":\"bci-seventeen-host-jax-arm\",\"runtype\":\"ssh\"}")
  C=$(echo "$R" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('new_contract') or '')" 2>/dev/null)
  if [ -z "$C" ]; then echo "$(date -u +%H:%M:%SZ) offer $ID ($NAME) unavailable"; continue; fi
  echo "$(date -u +%H:%M:%SZ) rented $C from offer $ID ($NAME, \$$DPH/h); proving it"

  OK=0; DEADLINE=$(( $(date +%s) + WINDOW ))
  while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    EP=$(bash "$ROOT/endpoint.sh" "$C" 2>/dev/null) || { sleep 20; continue; }
    set -- $EP
    if timeout 25 ssh -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=15 \
         -p "$2" "root@$1" 'nvidia-smi --query-gpu=name --format=csv,noheader' 2>/dev/null | grep -q .; then
      echo "$(date -u +%H:%M:%SZ) $C PROVED: ssh answers at $1:$2 and nvidia-smi sees a GPU"
      echo "$C" > "$ROOT/contract.id"; echo "$1 $2" > "$ROOT/endpoint.txt"; OK=1; break
    fi
    sleep 20
  done

  if [ "$OK" = 1 ]; then
    date -u +"=== handing off to provision %Y-%m-%dT%H:%M:%SZ ==="
    exec bash "$ROOT/provision_gpu.sh"
  fi
  echo "$(date -u +%H:%M:%SZ) $C did not prove out in ${WINDOW}s, destroying"
  curl -s -X DELETE -H "Authorization: Bearer $K" "$API/instances/$C/" > /dev/null
done <<< "$OFFERS"
date -u +"=== no offer proved out %Y-%m-%dT%H:%M:%SZ ==="
