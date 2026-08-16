#!/bin/bash
# Blackhole neutrality for the re-landed ring 2, on the merged tree (77a6cdad + 6e1f9e77).
#
# Two claims to check, and they are different claims:
#   lever C  must be INERT. `_apply_grid_thresholds` returns early on a full-size grid, before
#            any of the DRAM re-fit, so SEQ_LEN_MORE_CHUNKING must read the module default 1536
#            in every probe. That is the "by construction" argument the brief says not to trust
#            unverified after a rebase, so it is read off the fold, not off the source.
#   K3       must not REGRESS Blackhole. It is not bit-exact there (k_chunk sets the
#            online-softmax reduction order), and it was measured to help 1.0662x/1.1280x.
#
# Arms are interleaved and repeated twice per size, because a single-shot boltz2 leg carries
# +-20-30% noise (memory `perf-gate-single-shot-legs-recurring-false-alarm`). Same card, same
# process shape, same MSA cache for every leg.
set -u
WT=/home/ttuser/.coworker/wt/wh-boltz2-640aa-clash-rootcause
RUN=$WT/perf/whb2clash/runs_bhn
MSA=$WT/perf/whb2clash/runs_a3/msa
DEV=1
mkdir -p $RUN
for REP in 1 2; do
  for ACC in P00352 P22303 P54802; do
    for K3 in 0 1; do
      OUT=$RUN/${ACC}_k3${K3}_r${REP}
      if [ -n "$(find "$OUT" -name '*.cif' -print -quit 2>/dev/null)" ]; then
        echo "=== $ACC k3=$K3 rep$REP already done, skipping ==="
        continue
      fi
      echo "=== $ACC k3=$K3 rep$REP start $(date -u +%FT%TZ) ==="
      timeout 1800 $WT/perf/whb2clash/run_arm.sh $OUT \
        $WT/perf/whb2clash/fixtures/partA/$ACC.yaml $K3 - $DEV $MSA 1 > $OUT.log 2>&1
      echo "=== $ACC k3=$K3 rep$REP rc=$? $(date -u +%FT%TZ) ==="
    done
  done
done
echo "BHN DONE $(date -u +%FT%TZ)"
