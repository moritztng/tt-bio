#!/usr/bin/env bash
# bc4-orchestrator: headdev's own pc-card-0 waiter (pid 3259390) died silently at launch, so its
# A/B/A never ran. Relaunch it gated on bc4-land-fuse's aba-tape chain finishing, so the two rows
# do not both grab card 0 in the gap between land-fuse's B and A2 legs.
set -u
OUT=/home/moritz/.coworker/artifacts/bc4-bet-headdev/sm-pc
FUSE=/home/moritz/.coworker/artifacts/bc4-land-fuse/aba-tape/done
echo "$(date -u +%FT%TZ) relaunched by bc4-orchestrator, waiting for $FUSE" >> "$OUT/wait.log"
while [ ! -f "$FUSE" ]; do sleep 60; done
echo "$(date -u +%FT%TZ) land-fuse done; handing card 0 to headdev" >> "$OUT/wait.log"
exec bash "$OUT/run.sh"
