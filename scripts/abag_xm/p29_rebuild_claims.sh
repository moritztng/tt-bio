#!/bin/bash
# p29 claims rebuild — run ONCE at fleet drain (P29_DONE or full stop), before the
# final harvest. Clears every claim whose task key lacks an rc=0 record in
# results.jsonl, so a relaunch refolds exactly the missing set: guard kills
# (rc=124), killed-while-queued folds (pass-264: 7 zero-log + ~30 TERM-era),
# and any interrupted in-flight work. Validated matcher from pass-248, key on
# the FULL task tuple (pass-97 lesson). Records are append-only and the link
# phase re-emit skips recorded keys, so refolds are idempotent.
# Refuses to run while a p29 fleet driver is alive (would race live claims).
set -u
B=$HOME/mthuening/p29
if pgrep -f "p29_flee[t].sh 32 8" >/dev/null; then
  echo "REFUSE: p29 fleet driver alive — TERM it first (in-flight folds drain or rebuild later)" >&2
  exit 1
fi
before=$(ls $B/claims | wc -l)
cleared=0
for i in $B/claims/*/; do
  idx=$(basename $i)
  read -r m t r s c k <<<"$(sed -n "${idx}p" $B/tasks.txt)"
  [ -z "${m:-}" ] && continue
  grep -q "\"model\":\"$m\",\"target\":\"$t\",\"rung\":$r,\"seed\":$s,\"chunk\":$c" \
    <(awk '$0 ~ /"rc":0/' $B/results.jsonl) || { rmdir $i && cleared=$((cleared+1)); }
done
after=$(ls $B/claims | wc -l)
echo "claims: $before -> $after (cleared $cleared stale; those tasks refold on relaunch)"
