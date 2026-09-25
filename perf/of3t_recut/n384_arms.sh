#!/usr/bin/env bash
# of3t-recut job 2 and job 3 input: the reference trunk arm on the MODEL frame at n384, both
# conventions.
#
#   n384_arms.sh corrected   the DEFAULT injection. Three things come out of it: block 47 against
#                            the reference's own full-model backward (must reach the 1e-12 bar
#                            fixed in 2520681ed), dL/dz_in against the falsifier of3t-frameself
#                            banked before any of this ran (0.000848887340907281 exact against
#                            0.0014907294032500784 overcounting), and `cot_z_correction`, which
#                            is the delta every other arm is rescored with.
#   n384_arms.sh legacy      --legacy-total-cotangent. Must reproduce of3t-twoside's banked
#                            ctrl_f64 arm -- same host, same tree, same 14 threads, same producer
#                            modulo this repair -- BIT for BIT.
#
# This is `perf/of3t_twoside/arms.sh ctrl` with the out paths changed and the convention flag
# added. Same boundary, same cotangent, same digest refusal, same tree, same crop, same threads.
# CPU only, no card is opened.
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-recut
O=/home/ttuser/of3t_recut
M=/home/ttuser/of3t_modelframe
T=/home/ttuser/of3t_refprec
PY=/home/ttuser/tt-bio-dev/env/bin/python
mkdir -p "$O"
cd "$W"
export PYTHONPATH="$T/of3pkg043:$T/deps:$T/pylibs"

B=$M/boundary_model_n384.pt
C=$M/cot_model_n384.pt
echo "583bcd7c91ce6ed46aea59844954fb2bf7ddc3d553d7f9d4f238998183db99e2  $B
4e66d1ef45da2eec18fdff9489d68e980df928dc43d10fd4af82d14ec67141ac  $C" | sha256sum -c - || exit 2

case "${1:?usage: n384_arms.sh corrected|legacy}" in
  corrected) NM=ref_f64_model_n384_corrected; FLAG=() ;;
  legacy)    NM=ref_f64_model_n384_legacy;    FLAG=(--legacy-total-cotangent) ;;
  *) echo "unknown arm $1"; exit 2 ;;
esac
TH=14

echo "=== $NM start $(date -u +%FT%TZ) host $(hostname) threads $TH ==="
S=$(date +%s)
OMP_NUM_THREADS=$TH nice -n 10 "$PY" perf/of3t_trunkg043/ref_grad.py \
  --tree "$T/of3pkg043" --boundary "$B" --cap-last "$C" \
  --policy f64 --blocks 48 --crop 384 --threads $TH --checkpoint \
  --out "$O/$NM.pt" --report "perf/of3t_recut/${NM^^}.json" "${FLAG[@]}" 2>&1 \
  | grep -vE 'UserWarning|warnings.warn|  from openfold3|Consider using tensor.detach|loss\)\}, a.out\)'
rc=${PIPESTATUS[0]}
echo "=== $NM exit $rc elapsed $(($(date +%s)-S))s $(date -u +%FT%TZ) ==="

[ -s "perf/of3t_recut/${NM^^}.json" ] && "$PY" - "perf/of3t_recut/${NM^^}.json" "$O/$NM.pt" \
    perf/of3t_trunkg043/ref_grad.py "$rc" <<'PY'
import hashlib, json, os, socket, subprocess, sys
rep, out, prod, rc = sys.argv[1:]
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for x in iter(lambda: f.read(1 << 22), b''):
            h.update(x)
    return h.hexdigest()
d = json.load(open(rep))
d['provenance'] = {
    'host': socket.gethostname(), 'row': 'of3t-recut', 'defect': 'D242',
    'device_involved': False,
    'why_no_aiclk': 'CPU only, no Tenstorrent device is opened',
    'producer': {'path': prod, 'sha256': sha(prod),
                 'is': 'perf/of3t_trunkg043/ref_grad.py with the D242 repair'},
    'git_commit': subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True,
                                 text=True).stdout.strip(),
    'out': {'path': out, 'sha256': sha(out), 'bytes': os.path.getsize(out)}
           if os.path.exists(out) else None,
    'exit': int(rc),
}
json.dump(d, open(rep, 'w'), indent=1)
print('provenance written into ' + rep)
PY
exit "$rc"
