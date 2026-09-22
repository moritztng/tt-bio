#!/usr/bin/env bash
# of3t-verbinstall: score an arm IN FRAME on qb1, with of3t-frame384's own scorer.
#
# The same invocation of3t-trunkceiling and of3t-f64route used, against the SAME reference pair
# on the host that built them (D189: the bf16 denominator is host-dependent at the percent
# level, the float64 one is not). Only `--arm-built-on` differs, because these arms ran on
# qb1's p150a rather than qb2's p300c. That substitution is measured, not assumed: VERB_HF on
# qb1 and CEIL_HF on qb2 both read 0.4179981990834974.
#
#   score.sh <TAG> <arm.pt> <card>
# CPU only. No card is opened.
set -uo pipefail
D=/home/ttuser/of3t_frame384
M=/home/ttuser/.coworker/wt/of3t-verbinstall/perf/of3t_verbinstall
PY=/home/ttuser/tt-bio-dev/env/bin/python
export PYTHONPATH=$D/pkg

TAG=$1; ARM=$2; CARD=${3:-1}
OMP_NUM_THREADS=8 nice -n 10 "$PY" $D/pkg/of3t_frame384/frame384.py \
  --ref-model-f64 $D/grads_f64_043.pt \
  --c64-plain $D/c64_f64_plain.pt --c64-ckpt $D/c64_f64_ckpt.pt \
  --c64-bf16-plain $D/c64_bf16_plain.pt --c64-bf16-ckpt $D/c64_bf16_ckpt.pt \
  --c64-banked $D/ref_f64_c64.pt \
  --refs-built-on "qb1 (tt-quietbox), CPU only, no card, EPYC 8124P, torch 2.8.0+cpu / python 3.10.12" \
  --arm-built-on "qb1 (tt-quietbox) card $CARD, p150a Blackhole -- NOT the p300c CEIL_HF3's 0.5545352626143085 was built on; VERB_HF/CEIL_HF is the measured cross-board A/A that licenses the comparison" \
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
# The arms this row exists to reproduce and to explain, from their own committed artifacts.
CEIL_HF3 = {"f64": 0.4175214198121818, "bf16": 0.5545352626143085}
VERB_HF  = {"f64": 0.4179981990834974, "bf16": 0.5547455957585244}
ROUTE_HF = {"f64": 0.5605347900452246, "bf16": 0.7734340172378431}
for k in ("ours_vs_REF_LOCAL_f64_n384", "ours_vs_REF_LOCAL_bf16_n384",
          "floor_REF_LOCAL_bf16_vs_REF_LOCAL_f64_n384"):
    s = m.get(k)
    if s:
        print(f"{k:48s} {s['mass_weighted_rel_l2']!r}  cos {s['mass_weighted_cos']:.6f}  "
              f"nr {s['mass_weighted_norm_ratio']:.6f}  n={s['compared']}")
f64 = m["ours_vs_REF_LOCAL_f64_n384"]["mass_weighted_rel_l2"]
bf16 = m["ours_vs_REF_LOCAL_bf16_n384"]["mass_weighted_rel_l2"]
bar = m["A26_style_reachable_bar_for_this_scope"]
print(f"\nvs float64 {f64!r}   vs their bf16 {bf16!r}")
print(f"A26 bar {bar!r} -> {bf16 / bar!r}x, inside={m['inside_the_A26_style_bar']}")
for name, ref in (("CEIL_HF3", CEIL_HF3), ("VERB_HF", VERB_HF), ("ROUTE_HF", ROUTE_HF)):
    print(f"  vs {name:9s} f64 {f64 - ref['f64']:+.16f} ({100*(f64/ref['f64']-1):+.4f} %)"
          f"   bf16 {bf16 - ref['bf16']:+.16f} ({100*(bf16/ref['bf16']-1):+.4f} %)")
PYEOF
exit $rc
