#!/bin/bash
# bwx-flipgate: the Blackhole measurement the extra-MSA/template default flip is gated on.
# Runs serially on qb1 card 0 from a scratch tree = origin/main + origin/wk/bwx-perf:
#   1. tests/test_bindcraft2.py whole on the merge tree, card open (the two flip tests are card-gated)
#   2. the same suite on main alone, as the control
#   3. N=1 sitting: 8 arms alternated, JAX (0:0) vs on card (1:1), perf/bwx_perf/sit.py --shipped
#   4. N=3 sitting: the count duotraj.auto_trajectories(288) picks on this box
#   5. structure: JAX twice (the paired A/A floor) then on card, per-round CIFs
#   chain.sh <merge tree> <out dir> [rounds]
set -uo pipefail
mt=$1; out=$2; r=${3:-6}
mkdir -p "$out"
cd "$mt"
py=/home/ttuser/bcx_e2e_venv/bin/python3
export PYTHONPATH=$mt BCX_BC2=/home/ttuser/bcx_e2e/bc2
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:bwx-flipgate
export JAX_COMPILATION_CACHE_DIR=$out/xlacache
say(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$out/chain.log"; }
say "chain start merge=$(git rev-parse --short HEAD) main=$(git rev-parse --short HEAD^1) perf=$(git rev-parse --short HEAD^2)"
for t in merge:$mt main:$(dirname "$mt"); do
    tag=${t%%:*}; tree=${t#*:}
    say "suite $tag at $(git -C "$tree" rev-parse --short HEAD)"
    ( cd "$tree" && PYTHONPATH=$tree:/home/ttuser/bcx_e2e/bc2 timeout 2400 $py -m pytest \
        tests/test_bindcraft2.py -q -p no:cacheprovider -rs -v ) > "$out/suite_$tag.log" 2>&1
    say "suite $tag rc=$? $(tail -1 "$out/suite_$tag.log")"
done
say "sitting n1"
$py -u perf/bwx_perf/sit.py --chip 0 --rounds "$r" --shipped --params /home/ttuser/bcx_e2e/af2_params \
    --out "$out/n1" a1:0:0:1 b1:1:1:1 b2:1:1:1 a2:0:0:1 a3:0:0:1 b3:1:1:1 b4:1:1:1 a4:0:0:1 \
    > "$out/n1.log" 2>&1
say "sitting n1 rc=$?"
say "sitting n3"
$py -u perf/bwx_perf/sit.py --chip 0 --rounds "$r" --params /home/ttuser/bcx_e2e/af2_params \
    --out "$out/n3" a1:0:0:3 b1:1:1:3 b2:1:1:3 a2:0:0:3 a3:0:0:3 b3:1:1:3 b4:1:1:3 a4:0:0:3 \
    > "$out/n3.log" 2>&1
say "sitting n3 rc=$?"
export ARM_OUT_ROOT=$out/acc ARM_XLA_CACHE=$out/xlacache
export TT_BIO_MM_LAYOUT=1 TT_BIO_TAPED_CHANNEL_MOVE=1 TT_BIO_WIDEN_ADD=1
for a in acc_jax_a:0 acc_jax_b:0 acc_card:1; do
    tag=${a%%:*}; on=${a##*:}
    say "structure $tag extra_msa=template=$on"
    perf/bcx_p10_stack/arm.sh "$tag" 4 "$on" "$on" hifi --triatt-bw 1 --rne-kernel 1 \
        --set save_design_frames=1 > "$out/$tag.log" 2>&1
    say "structure $tag rc=$?"
done
$py perf/bcx_p10_stack/score_struct.py "$out/acc" --base acc_jax_a --floor acc_jax_b \
    --lever acc_card --out "$out/acc_struct.json" > "$out/acc_struct.log" 2>&1
say "score rc=$? chain done"
