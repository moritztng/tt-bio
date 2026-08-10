#!/bin/sh
# qb1 leg of the reblock_permute landing gate. Card 0, one step at a time, one device context open at
# a time. Every step writes its own log and its own JSON and drops a `.done` marker, so a pass that
# runs out of turn budget resumes rather than repeats.
#
# Run all remaining steps:      sh perf/z_flip_land/run_qb1.sh
# Run named steps only:         sh perf/z_flip_land/run_qb1.sh gate sweep_protenix-v2
# Steps, in the order they are worth doing:
#   gate                 release_gate.py for protenix-v2 / esmfold2 / esmfold2-fast (the verdict)
#   sweep_protenix-v2    census at 117 / 298 / 384 / 512
#   sweep_esmfold2       census at 117 / 298 / 384 / 512
#   control_protenix-v2  flag-OFF arm at 298 and 384, CIF sha compared
#   control_esmfold2     same
#   band                 protenix-v2 at the 385 and 506 band edges, both arms
#   blockwall            the trimul block wall, i.e. the only instrument that resolves ms/fold here
#
# If `gate` FAILs a leg, run the OFF control for that model in the same session before writing
# anything down (see the OFF CONTROL line at the bottom of this file).
WT=/home/ttuser/.coworker/wt/protenix-trunk--z-permute-flip-qb1
PY=/home/ttuser/tt-bio-dev/env/bin/python3
cd "$WT" || exit 1

# Card 0 by the fleet's lease. TT_VISIBLE_DEVICES is not the /dev/tenstorrent identity map on qb1:
# measured 2026-08-10 14:42 UTC, TT_VISIBLE_DEVICES=0 opens /dev/tenstorrent/1. Nodes 0 and 1 were
# both free; nodes 2 and 3 belong to other legs, so preflight aborts if we land on either.
export TT_VISIBLE_DEVICES=0
export TT_BIO_LEASE_HOLDER=worker:protenix-trunk--z-permute-flip-qb1
export PYTHONPATH=$WT
export ESM_ROOT=/home/ttuser/esm
export OF3_CKPT=/home/ttuser/of3-weights/of3-p2-155k.pt
export RELEASE_GATE_MSA_DIR=/home/ttuser/w6_gate_msa

L=perf/z_flip_land/logs_qb1
mkdir -p "$L"
S=perf/z_flip_land
F=perf/size512/fixtures
T4=examples/prot.yaml,$F/cdk2x2_298.yaml,$F/cdk2x2_384.yaml,$F/cdk2x2_512.yaml
T2=$F/cdk2x2_298.yaml,$F/cdk2x2_384.yaml
TB=$F/cdk2x2_385.yaml,$F/cdk2x2_506.yaml

preflight() {
  out=$($PY -c '
import os, subprocess, sys
import tt_bio, tt_bio.reblock_permute as RP
from tt_bio.tenstorrent import get_device, arch_name
wt = os.environ["PYTHONPATH"]
assert tt_bio.__file__.startswith(wt), "tt_bio loaded from %s, not %s" % (tt_bio.__file__, wt)
assert RP.REBLOCK_PERMUTE is True and RP._ENABLED is True, "the flip is not live in this process"
d = get_device()
g = d.compute_with_storage_grid_size()
assert (g.x, g.y) == (13, 10), "grid is %dx%d, not qb1 13x10" % (g.x, g.y)
held = [n for n in (0, 1, 2, 3)
        if str(os.getpid()) in subprocess.run(["fuser", "/dev/tenstorrent/%d" % n],
                                              capture_output=True, text=True).stdout.split()]
assert held and all(n in (0, 1) for n in held), "opened %s -- nodes 2 and 3 are other legs" % held
print("PREFLIGHT OK  grid=%dx%d arch=%s nodes=%s flip=ON" % (g.x, g.y, arch_name(), held))
' 2>&1)
  echo "$out" | grep -E "PREFLIGHT OK|AssertionError|Error" | tail -3
  echo "$out" | grep -q "PREFLIGHT OK"
}

step() {
  name=$1; shift
  [ -f "$L/$name.done" ] && { echo "SKIP $name"; return; }
  echo "=== START $name $(date -u +%FT%TZ)"
  timeout 3000 "$@" > "$L/$name.log" 2>&1
  rc=$?
  echo "=== END $name rc=$rc $(date -u +%FT%TZ)"
  [ $rc -eq 0 ] && touch "$L/$name.done"
}

want() {
  [ -z "$ARGS" ] && return 0
  for a in $ARGS; do [ "$a" = "$1" ] && return 0; done
  return 1
}

ARGS="$*"
preflight || { echo "PREFLIGHT FAILED -- nothing ran"; exit 1; }

want gate && step gate $PY -u scripts/release_gate.py \
    --model protenix-v2 --model esmfold2 --model esmfold2-fast

for M in protenix-v2 esmfold2; do
  want "sweep_$M" && step "sweep_$M" $PY -u $S/census_sweep.py --model "$M" --targets "$T4" \
      --out "$S/sweep_${M}_qb1c0.json"
done

for M in protenix-v2 esmfold2; do
  want "control_$M" && step "control_$M" $PY -u $S/census_sweep.py --model "$M" --targets "$T2" \
      --control --out "$S/control_${M}_qb1c0.json"
done

want band && step band $PY -u $S/census_sweep.py --model protenix-v2 --targets "$TB" --control \
    --out "$S/band_protenix_qb1c0.json"

# The fold wall cannot resolve 209-251 ms/fold: this host's fold-wall A/A floor runs to 1480 ms and
# a single cold fold pair measured 53.83 s ON against 42.76 s OFF on byte-identical output. The
# trimul block wall is the instrument. --aa-rounds 0 --rounds 0 skips the fold arms entirely.
want blockwall && step blockwall $PY -u perf/p3_permute_op/flip_protenix.py \
    --aa-rounds 0 --rounds 0 --block-reps 15 --out "$S/blockwall_qb1c0.json"

echo "QB1_DRIVER_DONE $(date -u +%FT%TZ)"
# OFF CONTROL, only for a model whose gate leg FAILed, same card, same session:
#   TT_BIO_REBLOCK_PERMUTE=0 $PY -u scripts/release_gate.py --model <that one> \
#     > $L/gate_off_<that one>.log 2>&1
# Identical failure signature (program id excluded, it is a per-session counter) means a pre-existing
# main failure. A different signature means the flip, and then the window narrows in eligible().
