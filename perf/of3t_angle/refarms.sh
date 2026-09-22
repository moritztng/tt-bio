#!/usr/bin/env bash
# of3t-angle: the two CORRECTED local references on the frame384 frame, on qb1.
#
#   refarms.sh corr    the D242 correction alone (perf/of3t_angle/correction.py). Cheap; it is
#                      what unblocks the device arms.
#   refarms.sh f64     the full corrected float64 reference. Its own `cot_z_correction` is the
#                      bit-identity control on `corr`.
#   refarms.sh bf16    upstream`s own bf16 autocast step, corrected, driven by `corr` through
#                      --cot-correction (A42: one correction, every arm).
#
# This is `perf/of3t_frame384/n384.sh` with the repaired producer and the outputs in this row`s
# dir. Same tree, same boundary, same capture, same crop, same thread count. qb1 is not a
# choice: D189 makes the bf16 denominator host-dependent and the references R149 read against
# were built here. CPU only, no card is opened on either host.
set -uo pipefail
D=/home/ttuser/of3t_frame384
O=/home/ttuser/of3t_angle
PY=/home/ttuser/tt-bio-dev/env/bin/python
export PYTHONPATH=$D/ref:$D/deps
mkdir -p "$O"
cd "$D"

echo "8cb3a58669eaadc9a6782d927ff4dd4dc7b706f244e5575709b738c5ccb59aa3  $D/boundary_n384.pt
a55ef1c4e6c90c87984242f9d1cc8523fbd9470a5f592eac0d0aff7000d29cf5  $D/block47_boundary.pt" \
  | sha256sum -c - || exit 2

case "${1:?usage: refarms.sh corr|f64|bf16}" in
  corr)
    echo "=== corr start $(date -u +%FT%TZ) host $(hostname) ==="
    S=$(date +%s)
    OMP_NUM_THREADS=8 nice -n 10 "$PY" "$O/correction.py" \
      --producer "$O/ref_grad.py" --tree "$D/of3pkg043" \
      --boundary "$D/boundary_n384.pt" --cap-last "$D/block47_boundary.pt" \
      --crop 384 --threads 8 \
      --out "$O/cot_external_n384.pt" --report "$O/COT_EXTERNAL_N384.json" 2>&1 \
      | grep -vE "UserWarning|warnings.warn|  from openfold3|Consider using tensor.detach"
    rc=${PIPESTATUS[0]}
    echo "=== corr exit $rc elapsed $(($(date +%s)-S))s $(date -u +%FT%TZ) ===" ;;
  f64|bf16)
    if [ "$1" = f64 ]; then NM=ref_f64_n384_corrected; POL=f64; TH=16; FLAG=()
    else NM=ref_bf16auto_n384_corrected; POL=bf16auto; TH=16
         FLAG=(--cot-correction "$O/cot_external_n384.pt"); fi
    echo "=== $NM start $(date -u +%FT%TZ) host $(hostname) threads $TH ==="
    S=$(date +%s)
    OMP_NUM_THREADS=$TH nice -n 10 "$PY" "$O/ref_grad.py" --tree "$D/of3pkg043" \
      --boundary "$D/boundary_n384.pt" --cap-last "$D/block47_boundary.pt" \
      --policy $POL --blocks 48 --crop 384 --threads $TH --checkpoint \
      --out "$O/$NM.pt" --report "$O/${NM^^}.json" "${FLAG[@]}" 2>&1 \
      | grep -vE "UserWarning|warnings.warn|  from openfold3|Consider using tensor.detach|loss\)\}, a.out\)"
    rc=${PIPESTATUS[0]}
    echo "=== $NM exit $rc elapsed $(($(date +%s)-S))s $(date -u +%FT%TZ) ===" ;;
  *) echo "unknown arm $1"; exit 2 ;;
esac
exit "${rc:-0}"
