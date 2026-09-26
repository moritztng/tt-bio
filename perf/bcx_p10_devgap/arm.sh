#!/bin/bash
# bcx-p10-devgap: the composed `hifi` arm, on this row's card, with the meter that counts the
# template stack as card time. `perf/bcx_p10_stack/arm.sh` with three changes and no others:
# card 3 and this row's lease holder, its own out/ and xla cache, and the `hifi` route pinned
# because this row exists to attribute THAT arm's device column and nothing else.
#   arm.sh <tag> <rounds> [extra args...]
set -euo pipefail
cd "$(dirname "$0")/../.."
tag=$1; rounds=$2; shift 2
out=perf/bcx_p10_devgap/out/$tag
# BindCraft 2 resumes a campaign from its project folder, so a re-run against a tag that
# already holds trajectory 1 returns in 5 s having measured nothing.
rm -rf "$out"
mkdir -p "$out"
export PYTHONPATH=$PWD
# A private compile cache: bindcraft/__init__.py points this at /tmp/bindcraft_xla_cache and
# af2.py:27 takes a host-global flock in it, so a co-tenant BindCraft 2 serialises with us for
# the whole compile. qb1 is running one right now.
export JAX_COMPILATION_CACHE_DIR=$PWD/perf/bcx_p10_devgap/out/xlacache
# This rows grant is card 3 and card 3 is DEAD: /sys/class/tenstorrent/tenstorrent!0
# (0000:c1:00.0, which is what TT_VISIBLE_DEVICES=3 selects in PCI order) has tt_heartbeat
# and tt_aiclk both frozen at 4294967295 while the other three tick, and ttnn.open_device
# on it raises "ARC core (8, 0) failed to start". CARD picks the card; the grant is widened
# on this command only, the way the card-fanout rule asks. Card 0 is the default because
# bcx-p10-stack leg 5 measured the 9.107 s device column this row attributes on qb1 card 0,
# so the anchor and the number it reconciles are on one card.
CARD=${CARD:-0}
export TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=3,$CARD TT_BIO_LEASE_HOLDER=worker:bcx-p10-devgap
# ENTRY swaps run_round.py for a wrapper that arms at the round boundary; round_ab.py calls
# run_round.main() itself, so every flag below still reaches the same place.
exec /home/ttuser/bcx_e2e_venv/bin/python3 -u "${ENTRY:-perf/bcx_round/run_round.py}" \
    --rounds "$rounds" --exact 0 --extra-msa 1 --template 1 \
    --triatt-sdpa 0 --triatt-hifi 1 --shipped --binder 146 --out "$out" "$@"
