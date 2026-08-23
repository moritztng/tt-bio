#!/usr/bin/env bash
# The 4x-land campaign on the MERGED tree: re-measure everything the flip depends on.
#
# Phase order is a correctness constraint, not a preference.
#   A  accuracy anchors + the 1024 aa hole, fanned across all four chips. Timing-insensitive, so
#      four legs may share the box. These are the STOP conditions (brief step 3): if an anchor
#      leaves its floor on the merged tree, nothing gets flipped and the perf hours are wasted.
#   B  the two exposed models under the pad, plus two immune controls, through the parity gate
#      itself so the reference convention is the one their published rows use. Also
#      timing-insensitive.
#   C  the perf ladder. Serial, one leg at a time, under benchlock, and only after A and B have
#      released the other chips -- benchlock excludes other benchlock users, not a co-tenant
#      diffusion rollout, so a quiet window has to be made rather than claimed.
set -u
WT=/home/ttuser/.coworker/wt/rf3-4x-land-and-defaults
PY=/home/ttuser/tt-bio-dev/env/bin/python3
PP="$WT:/home/ttuser/rf3_perf_deps"
HOLD=worker:rf3-4x-land-and-defaults
R=$WT/perf/rf3/results
L=$WT/perf/rf3/land4x
CEN=$L/census
S=$L/status
mkdir -p "$L" "$CEN" "$R"
cd "$WT" || exit 1
export BENCHLOCK_LOAD_WAIT_S=3600 BENCHLOCK_WAIT_S=7200
export TT_BIO_PARENT_PID=$$

say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$L/chain.log"; }
mark() { echo "$1" >> "$S"; }

lease() { echo "TT_VISIBLE_DEVICES=$1 TT_BIO_LEASE_CARDS=$1 TT_BIO_LEASE_HOLDER=$HOLD"; }

# ---------------------------------------------------------------- A: accuracy
acc() {  # acc <card> <tag> <fixture> <pad>
  card=$1; tag=$2; fix=$3; pad=$4
  say "A $tag start (card $card, fixture $fix, pad=$pad)"
  env PYTHONPATH="$PP" $(lease "$card") TT_BIO_SDPA_RAGGED_PAD=$pad \
      TT_BIO_SDPA_RAGGED_CENSUS="$CEN/acc_$tag" \
    "$PY" scripts/rf3_port/accuracy_cell.py --fixture "$fix" --arm a1 --seeds 0,1,2,3,4 \
      --work "$R/l4_$tag" --out "$R/l4_$tag.json" > "$L/acc_$tag.log" 2>&1
  rc=$?; say "A $tag exit $rc"; mark "acc_$tag rc=$rc"
}

phaseA() {
  # card 0/1 carry the two anchors under the pad and their own pad-off drift controls: a
  # committed row may only be cited if it reproduces on THIS tree (18 commits of main moved the
  # fp32-softmax route these arms sit either side of).
  ( acc 0 ubq76_pad1  ubq_76   1; acc 0 ubq76_pad0  ubq_76   0 ) &
  ( acc 1 roa117_pad1 7roa_117 1; acc 1 roa117_pad0 7roa_117 0 ) &
  # card 2: the aligned bit-exactness control the change owes itself (128 mod 32 = 0).
  ( acc 2 cdk128_pad1 cdk2_128 1 ) &
  wait
  mark PHASE_A_DONE
  say "PHASE A DONE"
}

# ---------------------------------------------------------------- D: the 1024 aa hole
# LAST, not first. Its CPU reference is uncached and costs >50 min a seed, so leading with it
# would hold the four flip-critical phases behind five hours of host trunk, and its host load
# would make phase C uncontended-in-name-only if it ran alongside. Everything the flip depends
# on is decided before this starts; this closes the one ladder rung that was never measured.
phaseD() {
  acc 0 cdk1024_pad1 cdk2_1024 1
  mark PHASE_D_DONE
  say "PHASE D DONE"
}

# ---------------------------------------------------------------- B: the exposed models
phaseB() {
  say "B parity gate with the pad ON: the two exposed models + two immune controls"
  env PYTHONPATH="$PP" TT_BIO_SDPA_RAGGED_PAD=1 \
      TT_BIO_LEASE_CARDS=0,1,2,3 TT_BIO_LEASE_HOLDER=$HOLD \
      TT_BIO_SDPA_RAGGED_CENSUS="$CEN/gate_pad1" \
    "$PY" scripts/full_parity_gate.py --workers qb2:0,qb2:1,qb2:2,qb2:3 \
      --leg protenix-prot-msa --leg protenix-ubq-msa --leg protenix-hsa-msa \
      --leg opendde-trpcage-nomsa --leg opendde-prot-prod \
      --leg boltz2-trpcage-nomsa --leg esmfold2-trpcage \
      --workdir "$L/gate_pad1" --out "$L/gate_pad1.json" --legacy-rdx \
      > "$L/gate_pad1.log" 2>&1
  rc=$?; say "B gate exit $rc"; mark "gate_pad1 rc=$rc"
  mark PHASE_B_DONE
  say "PHASE B DONE"
}

# ---------------------------------------------------------------- C: the perf ladder
# 512: the perf page own harness, page fixture, page timed region, two processes per arm,
# interleaved. The process-to-process A/A is the number a two-arm reading has to beat.
p512() {  # p512 <tag> <arm> <pad>
  tag=$1; arm=$2; pad=$3
  say "C 512 $tag start"
  /home/ttuser/.coworker/scripts/benchlock.sh rf3-4x-land-and-defaults -- \
    env PYTHONPATH="$PP" $(lease 0) TT_BIO_SDPA_RAGGED_PAD=$pad \
        TT_BIO_SDPA_RAGGED_CENSUS="$CEN/p512_$tag" \
    "$PY" perf/rf3/page512_tt.py --repeat 2 --arm "$arm" --label "$tag" \
      --out "$L/p512_${tag}.json" > "$L/p512_${tag}.log" 2>&1
  rc=$?; say "C 512 $tag exit $rc"; mark "p512_$tag rc=$rc"
}

# 768/1024: one process interleaves both arms. Both lengths divide 32, so the pad provably fires
# on nothing there and the reading is a0 vs a1 on speed; the per-fold padded count is recorded so
# that cannot be misread as the pad having been exercised.
prung() {  # prung <aa>
  aa=$1
  say "C $aa aa start"
  /home/ttuser/.coworker/scripts/benchlock.sh "rf3-4x-land-and-defaults-${aa}aa" -- \
    env PYTHONPATH="$PP" $(lease 0) \
    "$PY" perf/rf3/ladder_arm_ab.py --aa "$aa" --arms ABAB \
      --out "$L/ladder_${aa}.json" > "$L/ladder_${aa}.log" 2>&1
  rc=$?; say "C $aa aa exit $rc"; mark "ladder_$aa rc=$rc"
}

phaseC() {
  p512 a0_p1    a0 0
  p512 a1pad_p1 a1 1
  p512 a0_p2    a0 0
  p512 a1pad_p2 a1 1
  mark P512_DONE
  prung 768
  prung 1024
  mark PHASE_C_DONE
  say "PHASE C DONE"
}

say "=== land4x chain start, HEAD $(git rev-parse --short HEAD) ==="
phaseA
phaseB
phaseC
phaseD
mark LAND4X_CHAIN_DONE
say "=== land4x chain done ==="
