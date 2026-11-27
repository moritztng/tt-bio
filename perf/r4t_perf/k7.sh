#!/bin/bash
# K7 chain on qb2 card 1 (booking fleet-r4t-perf to 00:56Z). (1) taped-kernel registry block float64 grade
# (harness + floor from r4t-tape abtree, tt_bio from this worktree; PairUpdate n64/r48 seed 7, as gateg.sh; arms differ only by TT_BIO_TAPED_KERNELS: off, all, off2, brk=all+1.10x grad).
# Order: PERF default arm first, then the grade, then the registry PERF arm. (2) PERF protocol run: 9 logged batches cycled (upstream depths), 54 steps, 2 pinned CPUs, default levers; then the
# same with TT_BIO_TAPED_KERNELS=all (informational, not default).   k7.sh
C=1; WT=$HOME/.coworker/wt/r4t-perf; O=$HOME/r4t/perf/k7; mkdir -p $O; cd $WT
G=$O/grade; mkdir -p $G; T=$HOME/r4t/tape/abtree; PYD=/home/ttuser/tt-bio-dev/env/bin/python3; PY=$HOME/scratch/shipflow/venv-c4f99e695d34/bin/python
wait_card() { local h; for w in $(seq 1 360); do h=$(python3 ~/spd/feed/book.py active $C 2>/dev/null | grep -v r4t-perf); [ -z "$h" ] && [ -z "$(sudo -n fuser /dev/tenstorrent/$C 2>/dev/null)" ] && return; [ $w = 1 ] && echo "WAIT $(date -u +%T) $h $(sudo -n fuser /dev/tenstorrent/$C 2>&1)" >> $O/run.log; sleep 20; done; echo "ABORT no window $(date -u +%T)" >> $O/run.log; exit 1; }
echo "START $(date -u +%FT%TZ) sha $(git rev-parse --short HEAD) dirty $(git status --short tt_bio perf/r4t_step | wc -l)" >> $O/run.log
( while :; do echo "$(date -u +%T) AICLK $(cat /sys/class/tenstorrent/tenstorrent!$C/tt_aiclk 2>/dev/null)"; sleep 5; done ) > $O/aiclk.log 2>&1 &
CL=$!
P() {
B=$(ls $HOME/r4t/step/batch0?.npz | paste -sd,)
for arm in $1; do
  case $arm in def) K="";; all) K=all;; esac
  wait_card; taskset -c 6,7 env -u TT_BIO_LEVERS -u TT_BIO_LNBW_WEIGHTS -u TT_BIO_GATE_BW_FUSED TT_BIO_TAPED_KERNELS=$K OMP_NUM_THREADS=2 PYTHONPATH=$WT \
    TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=0,$C TT_BIO_LEASE_HOLDER=worker:r4t-perf TT_BIO_SDPA_OWN_FORWARD=1 timeout 1800 \
    $PY perf/r4t_step/trunk_step.py --n 256 --k 4 --steps 54 --batch $B --exempt perf/r4t_step/g2_exempt.tsv --out $O/ck > $O/p_$arm.txt 2>&1
  echo "RC_p_$arm $? $(date -u +%T) load $(cut -d' ' -f1 /proc/loadavg)" >> $O/run.log; rm -rf $O/ck
done
}
P def
for arm in off all off2 brk; do
  case $arm in off|off2) K="";; *) K=all;; esac; F=""; [ $arm = brk ] && F="--scale-grad pu.tri_mult_out.linear_g.weight"
  TT_BIO_TAPED_KERNELS=$K TT_BIO_GATE_BW_FUSED=1 TT_BIO_LNBW_WEIGHTS=1 TT_BIO_MMVJP_MIN=0 PYTHONPATH=$WT TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:r4t-perf \
    wait_card; timeout 2400 $PYD $T/perf/r4t_tape/block_gradcheck.py --module pair_update --n 64 --real 48 --probe 8 --seed 7 --arm bf16 $F \
    --floor $T/perf/r4t_tape/floor/pu_n64r48_s7.json --out $G/$arm.json > $G/$arm.log 2>&1
  echo "RC_grade_$arm $? $(date -u +%T)" >> $O/run.log
done
P all
kill $CL; echo "END $(date -u +%FT%TZ)" >> $O/run.log
