#!/usr/bin/env bash
# of3t-shapekey scoring. CPU only, no card opened, of3t-widthattr's scorer unchanged.
#
# The float64 reference and the bf16 floor were both built on qb1 (tt-quietbox), CPU only, EPYC
# 8124P, torch 2.8.0+cpu / python 3.10.12. D189 makes that host part of every ratio, so it is
# named on every run below even though the scoring itself runs on qb2. rsync'd to
# /home/ttuser/of3t_shapekey/refs/ from qb1:/home/ttuser/of3t_frame384/, digests recorded by the
# scorer itself.
#
#   score.sh A     arm A, both widths re-taken, the reproduction of 2.1795x
#   score.sh BANK  the banked pair of3t-widthattr scored, as a host control on this box
#   score.sh B     arm B's n384 against arm A's c64
#   score.sh C     arm C's n384 against arm A's c64
#   score.sh D     arm D's n384 against arm A's c64
#   score.sh SIB   AMENDMENT 1, the sibling comparison at both widths
set -euo pipefail
W=/home/ttuser/.coworker/wt/of3t-shapekey
cd "$W"
PY=/home/ttuser/tt-bio-dev/env/bin/python
R=/home/ttuser/of3t_shapekey/refs
O=/home/ttuser/of3t_shapekey
FH="qb1 (tt-quietbox), CPU only, no card, EPYC 8124P, torch 2.8.0+cpu / python 3.10.12"
AH="qb2 (tt-quietbox2) card 0, p300c Blackhole, AICLK median 1350 MHz sampled during every arm"

common=(--ref384 $R/ref_f64_n384.pt --ref64 $R/c64_f64_plain.pt
        --floor384 $R/ref_bf16auto_n384.pt --floor64 $R/c64_bf16_plain.pt
        --floor-host "$FH" --arm-host "$AH")

run() {  # tag ours384 ours64
  OMP_NUM_THREADS=8 nice -n 5 "$PY" perf/of3t_widthattr/widthgrowth.py "${common[@]}" \
    --ours384 "$2" --ours64 "$3" \
    --boundary384 $R/boundary_n384.pt --boundary64 $R/boundary_c64.pt \
    --out perf/of3t_shapekey/GROWTH_$1.json
  echo "wrote perf/of3t_shapekey/GROWTH_$1.json"
}

case "$1" in
  A)    run A    $O/dev_A384.pt $O/dev_A64.pt ;;
  BANK) run BANK $R/dev_RENORM_n384_nocaptures.pt $R/dev_scope_RENORM_c64.pt ;;
  B)    run B    $O/dev_B384.pt $O/dev_A64.pt ;;
  C)    run C    $O/dev_C384.pt $O/dev_A64.pt ;;
  D)    run D    $O/dev_D384.pt $O/dev_A64.pt ;;
  SIB)
    OMP_NUM_THREADS=8 nice -n 5 "$PY" perf/of3t_shapekey/siblings.py "${common[@]}" \
      --ours384 $O/dev_A384.pt --ours64 $O/dev_A64.pt --blocks 44,4,0,47 \
      --out perf/of3t_shapekey/SIBLINGS.json ;;
  *) echo "unknown $1"; exit 2 ;;
esac
