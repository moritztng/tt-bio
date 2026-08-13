#!/bin/bash
# The diffusion trace at 512 aa, uninstrumented, qb2 card 0 (BH p150a), ttnn 0.68.0.
#
# Differs from run_trace.sh (queued in pass 3, never acquired the box) in two ways:
#   - the instrumented leg is DROPPED. tt_baseline.py already records cif_sha256 and plddt per
#     fold (tt_baseline.py:310-313), so parity is decided from the same processes that produce
#     the publishable wall. Pass 2 showed the instrumented walls are unusable anyway (A/A 0.894 s
#     against a 0.25 s bar), so the instrumented leg cost 10 min of box time for a digest the
#     uninstrumented leg already gives.
#   - FOUR processes, alternating OFF/ON/OFF/ON, not two. A vs A across processes is the noise
#     floor the -0.202 s glue delta could not clear in pass 2, and doc section 10 lists that
#     cross-process A/A as owed. Alternating also satisfies PLAYBOOKS ACCELERATE rule 3.
#
# One benchlock hold. Each leg writes its JSON before the next starts, so a truncated hold still
# leaves evidence. 30 s between processes: opening the device immediately after the previous fold
# process exits threw silicon_sysmem_manager.cpp:326 in pass 2 leg 2.
set -u
WT=/home/ttuser/.coworker/wt/opendde-beat-b200
cd $WT || exit 1
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:opendde-beat-b200 PYTHONPATH=$WT
export ESM_ROOT=/home/ttuser/esm
PY=/home/ttuser/tt-bio-dev/env/bin/python3
FIX=$WT/perf/size512/fixtures
O=$WT/perf/oddeb200
REGION=1073741824

# benchlock waits for foreign folds at ACQUIRE time and holds nothing afterwards, so a co-tenant
# that launches mid-hold is invisible to it (root-caused pass 2: boltz2-qb2-hang-bisect on card 3
# drifted three identical arms 86.147 -> 87.041 -> 89.358). Sample ps around every leg instead.
foreign () {
  echo "--- foreign fold check ($1) $(date -Is) ---"
  ps -eo pid,pcpu,etime,args | grep -Ei 'tt_baseline|fold_ab|protenix|boltz|opendde|esmfold|openfold' \
    | grep -v grep | grep -v "$$" | head -10
  echo "loadavg: $(cut -d' ' -f1-3 /proc/loadavg)"
}

leg () {  # leg <name> <out.json> <trace 0|1>
  local name="$1" out="$2" tr="$3"
  echo "=== leg $name (trace=$tr) $(date -Is) ==="
  foreign "before $name"
  if [ "$tr" = "1" ]; then
    TT_BIO_BASE_TRACE=1 TT_BIO_TRACE_REGION_SIZE=$REGION \
      $PY -u scripts/gpu_vs_tt/tt_baseline.py --model opendde --repeat 2 \
          --target $FIX/cdk2x2_512.yaml --msa-a3m $FIX/cdk2x2_512.a3m \
          --label "512 aa cdk2x2, glue ON, trace ON ($name)" \
          --msa-dir $WT/.msa_om512_512 --out "$out"
  else
    $PY -u scripts/gpu_vs_tt/tt_baseline.py --model opendde --repeat 2 \
        --target $FIX/cdk2x2_512.yaml --msa-a3m $FIX/cdk2x2_512.a3m \
        --label "512 aa cdk2x2, glue ON, trace OFF ($name)" \
        --msa-dir $WT/.msa_om512_512 --out "$out"
  fi
  echo "leg $name RC=$?"
  foreign "after $name"
  sleep 30
}

echo "### run_trace2 start $(date -Is) on $(hostname), card ${TT_VISIBLE_DEVICES}"
leg offA $O/base_notrace_512_a.json 0
leg onA  $O/base_trace_512_a.json   1
leg offB $O/base_notrace_512_b.json 0
leg onB  $O/base_trace_512_b.json   1
echo "### run_trace2 done $(date -Is)"
