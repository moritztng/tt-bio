#!/usr/bin/env bash
# abag_xm_peer_mirror.sh -- keep a local copy of the other host's progress file.
#
# The two hosts fold disjoint slices and share nothing; `progress.jsonl` is host-local. That is
# fine until one host goes down, which is exactly when you want to know what it had finished:
# on 2026-07-28 qb2 hung with 169 records, and failing its four slices over to qb1 then meant
# either leaving half the slab stalled or refolding the ~125 pairs qb2 had already completed.
# The only reason that was a choice is that nothing kept a local copy of the peer's pair list.
#
# So mirror it while the peer is still up. A few hundred KB, and it makes a takeover free.
#
# This mirror is a SCHEDULING input only (abag_xm_generate.py: peer_done_pairs()). It is never
# merged into the local progress.jsonl and it is never evidence: the CIFs, PAEs, labels and
# provenance still live on the peer, and moving them is abag_xm_merge_hosts.py's job at release
# time. abag_xm_acceptance.py reads each host's real progress.jsonl over ssh, so a pair that
# exists only in a mirror still counts as outstanding -- a mirror can make the campaign skip
# work, never make the gate pass.
#
#   Usage:  abag_xm_peer_mirror.sh [peer]     # peer defaults to the other QuietBox
set -u
case "$(hostname)" in
  tt-quietbox)  DEFAULT_PEER=tt-quietbox2 ;;
  tt-quietbox2) DEFAULT_PEER=tt-quietbox ;;
  *)            DEFAULT_PEER="" ;;
esac
PEER="${1:-${ABAG_XM_PEER:-$DEFAULT_PEER}}"
DEST=$HOME/abag_xm/tier_a/peer_progress.jsonl
say(){ echo "[peer-mirror] $*"; }

[ -n "$PEER" ] || { say "no peer known for $(hostname) -- pass one explicitly"; exit 2; }
mkdir -p "$(dirname "$DEST")"
TMP=$(mktemp "$DEST.XXXXXX") || exit 1
trap 'rm -f "$TMP"' EXIT

if ! timeout 60 scp -q -o BatchMode=yes -o ConnectTimeout=10 \
        "ttuser@$PEER:abag_xm/tier_a/progress.jsonl" "$TMP" 2>/dev/null; then
  # A stale mirror is strictly better than none: it still holds everything the peer had finished
  # as of the last successful copy. Say so rather than failing silently, and keep what we have.
  if [ -s "$DEST" ]; then
    say "WARNING: $PEER unreachable -- keeping the existing mirror from $(date -r "$DEST" '+%F %T')"
  else
    say "WARNING: $PEER unreachable and no mirror exists yet. A takeover of its slices would"
    say "  refold everything it has already done."
  fi
  exit 1
fi

# Only replace a good mirror with a good one. An empty or truncated copy would silently un-skip
# every pair the peer had finished.
if [ ! -s "$TMP" ]; then
  say "WARNING: $PEER returned an empty progress file -- keeping the previous mirror"
  exit 1
fi
lines=$(wc -l < "$TMP")
if [ -s "$DEST" ]; then
  had=$(wc -l < "$DEST")
  if [ "$lines" -lt "$had" ]; then
    # progress.jsonl is append-only, so it can only ever grow. Shrinking means we read something
    # that is not the peer's real file (a truncated transfer, a reset campaign dir); do not trust it.
    say "WARNING: $PEER reports $lines records but the mirror already had $had. That file is"
    say "  append-only, so it cannot shrink -- keeping the previous mirror."
    exit 1
  fi
fi
mv -f "$TMP" "$DEST"
trap - EXIT
say "mirrored $lines records from $PEER"
