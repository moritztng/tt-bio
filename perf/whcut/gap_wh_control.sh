#!/bin/bash
# Is boltz2-prot-nomsa's Wormhole GAP the assembly's, or prod's own?
#
# The Wormhole gate returned GAP num=7.3021 env=3.4501 ratio=2.1165 on the assembled tree.
# §46 established the same leg GAPs IDENTICALLY on main and the assembled branch on
# Blackhole (num=7.0200 both, "[reproduces committed]"), and the mechanism says it must:
# protein G without an MSA pads to 64, far below K3's 448-960 band. This measures that
# instead of asserting it, against pxmain (b1a3fe61) -- the tree prod actually runs.
#
# Waits on card 28's occupancy, so it queues behind the parity gate and the clash sweep.
set -u
PRE=/home/cust-team/mthuening/whbase/pxmain
PY=/home/cust-team/mthuening/whbase/tt-bio/env/bin/python3
OUT=/home/cust-team/mthuening/whbase/wt-whcut/perf/whcut/out
CARD=${CARD:-28}
cd "$PRE" || exit 1
while [ "$(sudo -n lsof -t /dev/tenstorrent/4 2>/dev/null | wc -w)" != "0" ]; do sleep 60; done
echo "GAP WH CONTROL START $(date -u +%FT%TZ) tree $(git rev-parse --short HEAD)"
ESM_ROOT=/home/cust-team/mthuening/esm \
RELEASE_GATE_MSA_DIR=/home/cust-team/mthuening/abag_xm/msa_cache \
PYTHONPATH="$PRE" TT_METAL_LOGGER_LEVEL=FATAL TT_BIO_LEASE_HOLDER=worker:japanfold-wh-cutover \
  "$PY" scripts/full_parity_gate.py --workers UF-EV-A13-GWH02:$CARD --fresh \
    --workdir "$OUT/gap-wh-pxmain" --leg boltz2-prot-nomsa
echo "GAP WH CONTROL EXIT $? $(date -u +%FT%TZ)"
python3 - <<'PY'
import json, os
p = os.path.expanduser("/home/cust-team/mthuening/whbase/wt-whcut/perf/whcut/out/gap-wh-pxmain/boltz2-prot-nomsa.json")
try:
    d = json.load(open(p)); m = d.get("metrics", {}).get("kabsch_rmsd", {})
    print("pxmain  verdict=%s num=%.4f env=%.4f ratio=%.4f" % (
        d.get("verdict"), m.get("numerator", 0), m.get("envelope", 0), m.get("ratio", 0)))
    print("whcut   verdict=GAP    num=7.3021 env=3.4501 ratio=2.1165   (for comparison)")
except Exception as e:
    print("could not read control result:", e)
PY
