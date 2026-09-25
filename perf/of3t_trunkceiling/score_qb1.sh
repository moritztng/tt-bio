#!/usr/bin/env bash
# of3t-trunkceiling: score a trunk arm IN FRAME on qb1, with of3t-frame384's own scorer.
#
# Amendment condition 1: the threshold 0.4361680548 is a trunk-section rel L2 of ours against
# upstream's own bf16, both driven from the same captured boundary at padded N=384 -- the
# quantity that reads 1.0293953378 today. So this scores against ref_bf16auto_n384 /
# ref_f64_n384, the SAME pair frame384 used, on the SAME host that produced them (D189: the
# bf16 denominator is host-dependent at the percent level, the float64 one is not).
#
#   score_qb1.sh <TAG> <arm.pt>
# CPU only. No card is opened on either host.
set -uo pipefail
D=/home/ttuser/of3t_frame384
M=/home/ttuser/of3t_trunkceiling
PY=/home/ttuser/tt-bio-dev/env/bin/python
export PYTHONPATH=$D/pkg

TAG=$1; ARM=$2
OMP_NUM_THREADS=8 nice -n 10 "$PY" $D/pkg/of3t_frame384/frame384.py \
  --ref-model-f64 $D/grads_f64_043.pt \
  --c64-plain $D/c64_f64_plain.pt --c64-ckpt $D/c64_f64_ckpt.pt \
  --c64-bf16-plain $D/c64_bf16_plain.pt --c64-bf16-ckpt $D/c64_bf16_ckpt.pt \
  --c64-banked $D/ref_f64_c64.pt \
  --refs-built-on "qb1 (tt-quietbox), CPU only, no card, EPYC 8124P, torch 2.8.0+cpu / python 3.10.12" \
  --arm-built-on "qb2 (tt-quietbox2) card 0, p300c Blackhole (subsystem_device 0x0046, read from sysfs) -- the same host and board class the 1.0293953378 arm was built on" \
  --model-ref-built-on "qb2 (tt-quietbox2), CPU, upstream 0.4.3 full-model float64 on batch_step003" \
  --crop 384 \
  --ref-f64-n384 $D/ref_f64_n384.pt --ref-bf16-n384 $D/ref_bf16auto_n384.pt \
  --ours-n384 "$ARM" \
  --ref-f64-report $D/REF_F64_N384.json \
  --model-artifact $D/MODEL_withtrunk_n384.json \
  --reproduces 2.159527121735274 \
  --reproduces-from "perf/of3t_ditmodel/TRUNK_D174.json stats.MASKON_vs_FLOAT64" \
  --out $M/FRAME_${TAG}.json
rc=$?
echo "=== $TAG exit $rc ==="
[ -f $M/FRAME_${TAG}.json ] && "$PY" - "$M/FRAME_${TAG}.json" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
m = d["MATCHED"]
BAR = 0.4361680548
SHIPPED = 1.029395337772341
for k in ("ours_vs_REF_LOCAL_bf16_n384", "ours_vs_REF_LOCAL_f64_n384",
          "floor_REF_LOCAL_bf16_vs_REF_LOCAL_f64_n384"):
    if k in m:
        s = m[k]
        print(f"{k:48s} {s['mass_weighted_rel_l2']!r}  cos {s['mass_weighted_cos']:.6f}  "
              f"nr {s['mass_weighted_norm_ratio']:.6f}  n={s['compared']}")
        w = s.get("worst_by_error_mass")
        if w:
            print(f"{'':48s} worst-by-mass {w['tensor']} rel {w['rel_l2']!r}")
        w = s.get("worst_by_rel")
        if w:
            print(f"{'':48s} worst-by-rel  {w['tensor']} rel {w['rel_l2']!r} "
                  f"ref_norm {w['ref_norm']!r}")
b = m["ours_vs_REF_LOCAL_bf16_n384"]["mass_weighted_rel_l2"]
print(f"IN-FRAME vs upstream bf16 = {b!r}")
print(f"  bar {BAR}  -> {b / BAR!r}x the bar, {'PASSES' if b <= BAR else 'MISSES'}")
print(f"  shipped {SHIPPED} -> this arm is {SHIPPED / b!r}x better")
mp = d.get("MODEL_PROJECTION", {})
for k in ("renorm_vs_UPSTREAM_BF16", "renorm_vs_FLOAT64"):
    if k in mp:
        p = mp[k]
        print(f"PROJECTION {k}: model {p['model_projected_in_frame']!r} "
              f"= {p['multiple_of_the_A26_bar_projected']!r}x the A26 bar "
              f"{mp['A26_reachable_bar_vs_their_bf16']}, pass={p['clause_would_pass']}")
PYEOF
exit $rc
