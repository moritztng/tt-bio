#!/bin/bash
# Chunk 0 was launched before the fleet runner existed, so it writes p34d/odcamp/result.txt
# instead of a ledger line. harvest_par.py takes its work list from <run>/results.jsonl, so
# give it the record. Idempotent: refuses to append a second c0 line.
set -u
H=$HOME/mthuening
R=$H/p34d/od9j4c/results.jsonl
S=$H/p34d/odcamp
grep -q '"chunk":0' "$R" 2>/dev/null && { echo "c0 record already present"; exit 0; }
grep -q ODCAMP_DONE "$S/result.txt" 2>/dev/null || { echo "chunk 0 has not finished; nothing to emit"; exit 1; }
line=$(grep -v ODCAMP_DONE "$S/result.txt" | tail -1)
mps=$(sed -n 's/.*mps=\([0-9]*\).*/\1/p' <<<"$line")
rc=$(sed -n 's/.*rc=\([0-9]*\).*/\1/p' <<<"$line")
secs=$(sed -n 's/.*secs=\([0-9]*\).*/\1/p' <<<"$line")
n=$(ls $S/9j4c_c0/*results_9j4c/structures/*.cif 2>/dev/null | wc -l)
d=$(md5sum $S/9j4c_c0/*results_9j4c/structures/*.cif 2>/dev/null | awk '{print $1}' | sort -u | wc -l)
oom=$(sed -n 's/.*oom=\([0-9]*\).*/\1/p' <<<"$line")
printf '{"model":"opendde-abag","target":"9j4c","rung":512,"seed":20000,"chunk":0,"chunks":8,"mps":%s,"umd":31,"rc":%s,"seconds":%s,"cifs":%s,"distinct":%s,"oom":%s}\n' \
  "${mps:-1}" "${rc:-1}" "${secs:-0}" "${n:-0}" "${d:-0}" "${oom:-0}" | tee -a "$R"
