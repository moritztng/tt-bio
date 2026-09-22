#!/usr/bin/env bash
# of3t-f64route: one inference fold, with the host-float64 reach census taken IN the process
# that folded.
#   census_fold.sh <model> <card> [fixture]
#
# `TT_BIO_HOST_F64_SOFTMAX_AB=all` selects every construction site the fold builds, so every
# arrival is counted `refused` (no tape open) and `host_f64_softmax_sites` records which tokens
# this model actually constructed. That is the runtime call census: which sites were built, how
# many calls reached the gate, and how many of those came through the fp32-softmax tail -- the
# route that did not exist before this branch.
set -uo pipefail
W=/home/ttuser/.coworker/wt/of3t-f64route
O=/tmp/of3t/of3t-f64route
cd "$W"
M=$1; CARD=$2; FIX=${3:-perf/size512/fixtures/cdk2x2_128.yaml}
C=$O/census_$M
rm -rf "$C"; mkdir -p "$C"
S=$(date +%s)
env TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:of3t-f64route \
    TT_BIO_HOST_F64_SOFTMAX_AB=all TT_BIO_CAPACITY_CENSUS="$C" PYTHONPATH="$W" \
    /home/ttuser/tt-bio-dev/env/bin/python3 -m tt_bio.main predict "$W/$FIX" \
      --model "$M" --single_sequence --sampling_steps 6 --diffusion_samples 1 --seed 0 \
      --out_dir "$O/censusout_$M" > "$O/census_$M.log" 2>&1
rc=$?
echo "=== $M exit $rc elapsed $(( $(date +%s) - S ))s card $CARD ==="
[ $rc -ne 0 ] && tail -20 "$O/census_$M.log"
/home/ttuser/tt-bio-dev/env/bin/python3 - "$C" "$M" "$W/perf/of3t_f64route/REACH_${M}.json" <<'PYEOF'
import glob, json, os, sys
d, model, out = sys.argv[1], sys.argv[2], sys.argv[3]
files = sorted(glob.glob(os.path.join(d, "capacity_*.json")))
rows = []
for f in files:
    j = json.load(open(f))
    rows.append({"pid_file": os.path.basename(f),
                 "host_f64_softmax": j.get("host_f64_softmax"),
                 "sites": j.get("host_f64_softmax_sites"),
                 "reach": j.get("host_f64_softmax_reach"),
                 "fp32_softmax_calls": (j.get("fp32_softmax") or {}).get("calls"),
                 "fp32_softmax_tail_invocations": ((j.get("fp32_softmax") or {}).get("fused", 0)
                                                   + (j.get("fp32_softmax") or {}).get("unfused", 0))})
rep = {"model": model, "env": "TT_BIO_HOST_F64_SOFTMAX_AB=all", "processes": len(rows),
       "per_process": rows}
json.dump(rep, open(out, "w"), indent=1)
print(json.dumps(rep, indent=1))
PYEOF
exit $rc
