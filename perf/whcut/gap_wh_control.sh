#!/bin/bash
# Are the two Wormhole parity oddities the assembly's, or prod's own?
#
# The Wormhole gate returned GAP num=7.3021 env=3.4501 ratio=2.1165 on the assembled tree.
# §46 established the same leg GAPs IDENTICALLY on main and the assembled branch on
# Blackhole (num=7.0200 both, "[reproduces committed]"), and the mechanism says it must:
# protein G without an MSA pads to 64, far below K3's 448-960 band. This measures that
# instead of asserting it, against pxmain (b1a3fe61) -- the tree prod actually runs.
#
# Second leg, protenix-prot-msa. On the assembled Wormhole tree its envelope ratio is
# cross_over_floor=1.0179 (cross 2.813 against a ref floor of 2.763); the same leg on main's
# Blackhole gate reads 0.9870 (cross 2.727, IDENTICAL ref floor 2.763, since the floor is a
# property of the fixtures, not the device). So the device sits 3.2 % further from the
# reference on Wormhole. Architecture or assembly cannot be told apart from those two numbers
# -- they differ in both -- so this runs the leg on pxmain/Wormhole, which holds the
# architecture fixed and varies only the tree.
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
    --workdir "$OUT/gap-wh-pxmain" --leg boltz2-prot-nomsa --leg protenix-prot-msa
echo "GAP WH CONTROL EXIT $? $(date -u +%FT%TZ)"
python3 - <<'PY'
import json, os
p = os.path.expanduser("/home/cust-team/mthuening/whbase/wt-whcut/perf/whcut/out/gap-wh-pxmain/boltz2-prot-nomsa.json")
try:
    d = json.load(open(p)); m = d.get("metrics", {}).get("kabsch_rmsd", {})
    print("pxmain  boltz2-prot-nomsa verdict=%s num=%.4f env=%.4f ratio=%.4f" % (
        d.get("verdict"), m.get("numerator", 0), m.get("envelope", 0), m.get("ratio", 0)))
    print("whcut   boltz2-prot-nomsa verdict=GAP num=7.3021 env=3.4501 ratio=2.1165")
except Exception as e:
    print("could not read boltz2 control result:", e)
try:
    q = "/home/cust-team/mthuening/whbase/wt-whcut/perf/whcut/out/gap-wh-pxmain/protenix-prot-msa.json"
    e = json.load(open(q))["targets"]["prot"]["kabsch_rmsd"]
    print("pxmain  protenix-prot-msa cross_over_floor=%.4f cross=%.3f floor=%.3f" % (
        e["cross_over_floor"], e["cross"]["mean"], e["ref_floor"]["mean"]))
    print("whcut   protenix-prot-msa cross_over_floor=1.0179 cross=2.813 floor=2.763")
except Exception as ex:
    print("could not read protenix control result:", ex)
PY
