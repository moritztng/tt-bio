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
# This rows grant is card 3 = PCI 0000:c1:00.0 = /dev/tenstorrent/0. Node order and card
# number DISAGREE on this box, so the map is resolved with tt_bio.runtime.tt_bdf_to_index()
# and never by node order; it prints
#   {0000:01:00.0: 0, 0000:41:00.0: 1, 0000:42:00.0: 2, 0000:c1:00.0: 3}
# That chip's ARC stopped answering on 2026-09-26 (tt_card_type unknown, tt_serial and
# tt_aiclk both 0xFFFFFFFF, tt_heartbeat frozen) and one bounded `tt-smi -r /dev/tenstorrent/0`
# brought it back: p150a, serial 00000403319140AA, heartbeat advancing. CARD overrides for a
# fan-out onto an idle sibling, and the grant is widened on that command only, the way the
# card-fanout rule asks.
CARD=${CARD:-3}
export TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=3,$CARD TT_BIO_LEASE_HOLDER=worker:bcx-p10-devgap
# ENTRY swaps run_round.py for a wrapper that arms at the round boundary; round_ab.py calls
# run_round.main() itself, so every flag below still reaches the same place.
exec /home/ttuser/bcx_e2e_venv/bin/python3 -u "${ENTRY:-perf/bcx_round/run_round.py}" \
    --rounds "$rounds" --exact 0 --extra-msa 1 --template 1 \
    --triatt-sdpa 0 --triatt-hifi 1 --shipped --binder 146 --out "$out" "$@"
