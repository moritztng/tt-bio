#!/bin/bash
# Run the campaign sitting as soon as the round sitting has let go of the card. Both want card
# 0 and the campaign is the long pole, so the handover is automated rather than left to the
# next pass: an hour of idle card between two measured legs is an hour nobody gets back.
set -uo pipefail
cd "$(dirname "$0")/../.."
log=perf/bcx_default/out/sit1.log
for _ in $(seq 1 360); do
    grep -q "sitting done" "$log" && break
    sleep 20
done
grep -q "sitting done" "$log" || { echo "round sitting never finished; not starting the campaign"; exit 1; }
echo "=== round sitting done, starting the campaign sitting $(date -u +%FT%TZ)"
exec perf/bcx_default/campaign_sit.sh 6 "old:1 auto:auto"
