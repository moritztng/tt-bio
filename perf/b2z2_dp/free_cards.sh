#!/bin/bash
# The chips no process currently has open, first N of them, comma-joined. The lease flock is what
# actually prevents a collision; this only keeps the ladder from picking a card it would then wait
# out TT_BIO_LEASE_TIMEOUT behind. Co-tenants come and go on this box, so a width picks its cards
# when it starts, not when the ladder was launched.
set -u
N="${1:-16}"
BUSY=$(lsof -F n /dev/tenstorrent/* 2>/dev/null | grep tenstorrent | sed 's|.*/||' | sort -u)
FREE=()
for c in $(seq 0 31); do
  [ -e "/dev/tenstorrent/$c" ] || continue
  grep -qx "$c" <<< "$BUSY" && continue
  FREE+=("$c")
done
[ "${#FREE[@]}" -ge "$N" ] || { echo "only ${#FREE[@]} free chips, need $N" >&2; exit 1; }
(IFS=,; echo "${FREE[*]:0:$N}")
