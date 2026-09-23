#!/usr/bin/env bash
# of3t-twoside: the two injected trunk arms on the MODEL frame, D241.
#   arms.sh wire            f64, 4 blocks -- shape/wiring check only, not an experiment
#   arms.sh ctrl            f64, 48 blocks -- the gating control
#   arms.sh inj             bf16auto, 48 blocks -- the missing arm
#
# Producer is perf/of3t_trunkg043/ref_grad.py UNCHANGED, copied to $R/ref_grad.py and its sha
# recorded. No fork, no new argument.
#
# WHY QB2. qb1's root filesystem is at 100 %% and cannot hold a 1.3 GB gradient dump; qb2 has
# 2.6 TB free and already carries every scoring input (of3t_refprec, of3t_covadopt,
# of3t_wholemodel) plus the of3t-modelframe boundary and cotangent. CPU only, no card is opened:
# of3t-cotcoh holds qb1's card and land-standing holds qb2's.
#
# CROP. The device arm is padded n384 with 56 real rows, so --crop 384 (a no-op slice at this
# width), NOT ref_grad.py's --crop 64 default. Same as of3t-frame384 n384.sh and of3t-cotcoh
# arm_ref.sh use on this frame.
set -uo pipefail
R=/home/ttuser/of3t_twoside
M=/home/ttuser/of3t_modelframe
T=/home/ttuser/of3t_refprec
O=/tmp/of3t/of3t-twoside
PY=/home/ttuser/tt-bio-dev/env/bin/python
mkdir -p "$R" "$O"
export PYTHONPATH="$T/of3pkg043:$T/deps:$T/pylibs"
cd "$R"

B=$M/boundary_model_n384.pt
C=$M/cot_model_n384.pt
# D235 + A34: the arm is these two files and nothing else. Refuse a different pair.
echo "583bcd7c91ce6ed46aea59844954fb2bf7ddc3d553d7f9d4f238998183db99e2  $B
4e66d1ef45da2eec18fdff9489d68e980df928dc43d10fd4af82d14ec67141ac  $C" | sha256sum -c - || exit 2

case "${1:?usage: arms.sh wire|ctrl|inj}" in
  wire) NM=wire_f64_b4; POL=f64;      BL=4;  TH=14 ;;
  ctrl) NM=ctrl_f64;    POL=f64;      BL=48; TH=14 ;;
  inj)  NM=inj_bf16;    POL=bf16auto; BL=48; TH=14 ;;
  *) echo "unknown arm $1"; exit 2 ;;
esac

echo "=== $NM start $(date -u +%FT%TZ) host $(hostname) policy $POL blocks $BL threads $TH ==="
S=$(date +%s)
OMP_NUM_THREADS=$TH nice -n 10 "$PY" "$R/ref_grad.py" \
  --tree "$T/of3pkg043" --boundary "$B" --cap-last "$C" \
  --policy $POL --blocks $BL --crop 384 --threads $TH --checkpoint \
  --out "$R/$NM.pt" --report "$R/$NM.json" 2>&1 \
  | grep -vE 'UserWarning|warnings.warn|  from openfold3|Consider using tensor.detach|loss\)\}, a.out\)'
rc=${PIPESTATUS[0]}
echo "=== $NM exit $rc elapsed $(($(date +%s)-S))s $(date -u +%FT%TZ) ==="

# D235: the artifact carries the host that produced it and the digests that make it this arm.
[ -s "$R/$NM.json" ] && "$PY" - "$R/$NM.json" "$R/$NM.pt" "$R/ref_grad.py" "$rc" <<'PY'
import hashlib, json, os, socket, sys
rep, out, prod, rc = sys.argv[1:]
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for x in iter(lambda: f.read(1 << 22), b''):
            h.update(x)
    return h.hexdigest()
d = json.load(open(rep))
d['provenance'] = {
    'host': socket.gethostname(), 'row': 'of3t-twoside', 'defect': 'D241',
    'device_involved': False,
    'why_no_aiclk': 'CPU only, no Tenstorrent device is opened',
    'producer': {'path': prod, 'sha256': sha(prod),
                 'is': 'perf/of3t_trunkg043/ref_grad.py, unchanged'},
    'out': {'path': out, 'sha256': sha(out), 'bytes': os.path.getsize(out)}
           if os.path.exists(out) else None,
    'exit': int(rc),
}
json.dump(d, open(rep, 'w'), indent=1)
print('provenance written into ' + rep)
PY
exit "$rc"
