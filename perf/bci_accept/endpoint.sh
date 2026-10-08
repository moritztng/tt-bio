#!/bin/bash
# Resolve the rented box's ssh endpoint from the vast API by contract id, and print "host port".
# Hardcoding the endpoint cost a re-rent: a destroyed instance's host:port is not the new one's.
set -u
C=${1:?contract id}
K=$(cat /home/moritz/.config/vastai/vast_api_key)
curl -s -H "Authorization: Bearer $K" "https://console.vast.ai/api/v0/instances/$C/" | python3 -c "
import sys, json
d = json.load(sys.stdin).get('instances', {})
if d.get('actual_status') != 'running':
    sys.exit(1)
h, p = d.get('ssh_host'), d.get('ssh_port')
if not h or not p:
    sys.exit(1)
print(h, p)
"
