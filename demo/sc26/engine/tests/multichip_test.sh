#!/bin/bash
# CHIPS: one server, one warm worker per chip, attract folds on every chip, then two visitors
# arrive while all chips are busy (preemption), and the stream is logged.
#   bash demo/sc26/engine/tests/multichip_test.sh 0,3 <seconds> <outdir>
set -u
CHIPS=$1; SECS=$2; OUT=$3
cd "$(dirname "$0")/.."
PY=~/tt-bio-dev/env/bin/python3
mkdir -p "$OUT"
TT_BIO_LEASE_HOLDER=worker:sc26-engine timeout $((SECS + 300)) $PY -u server.py --chips "$CHIPS" \
    --record "$OUT/recorded" --logdir "$OUT" > "$OUT/server.log" 2>&1 < /dev/null &
SRV=$!
n=$(echo "$CHIPS" | tr ',' '\n' | wc -l)
for i in $(seq 120); do
  r=$(curl -s 127.0.0.1:8626/status | python3 -c "import json,sys; print(sum(c['state'] in ('ready','busy') for c in json.load(sys.stdin)['chips']))" 2>/dev/null)
  [ "${r:-0}" -ge "$n" ] && break; sleep 2
done
echo "$(date -u +%T) all $n chips up" > "$OUT/actions.txt"
timeout $((SECS + 30)) python3 client.py --seconds "$SECS" --log "$OUT/client.jsonl" > "$OUT/client.txt" 2>&1 &
CL=$!
sleep 20
echo "$(date -u +%T) two visitors submitted" >> "$OUT/actions.txt"
curl -s -XPOST 127.0.0.1:8626/fold -d '{"sequence":"MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWUAQ"}' >> "$OUT/actions.txt"; echo >> "$OUT/actions.txt"
curl -s -XPOST 127.0.0.1:8626/fold -d '{"sequence":"MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"}' >> "$OUT/actions.txt"; echo >> "$OUT/actions.txt"
curl -s -XPOST 127.0.0.1:8626/fold -d '{"sequence":"NLYIQWLKDGGPSSGRPPPS"}' >> "$OUT/actions.txt"; echo >> "$OUT/actions.txt"
wait $CL
curl -s 127.0.0.1:8626/status > "$OUT/status-end.json"
kill -INT $SRV; wait $SRV
echo "$(date -u +%T) server stopped rc=$?" >> "$OUT/actions.txt"
for c in $(echo "$CHIPS" | tr ',' ' '); do echo "chip $c holders after stop: $(fuser /dev/tenstorrent/$c 2>&1 | cut -d: -f2)" >> "$OUT/actions.txt"; done
echo DONE >> "$OUT/actions.txt"
