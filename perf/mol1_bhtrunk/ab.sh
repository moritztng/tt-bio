#!/bin/bash
# Mol-1 BH trunk lever A/B on one card (CARD, default qb1 card 3): separate processes per arm, interleaved A B A B, each 1 cold + 2 warm
# folds of a frozen PROTOCOL input through perf/mol1/fair/run.py (--part dev: serving path, AICLK sampled during
# every fold, full 20 cycles / 200 steps / 5 samples). Arms are the shipped Mol-1 Blackhole normal set (SHIPPED,
# = protenix.NORMAL_LEVERS | WIDE_NORMAL | MOL1_BH on main 06abce830) plus the arm's levers, via TT_BIO_LEVERS.
# Usage: ab.sh OUT SIZE REPS ARM=levers[@TREE] [ARM=levers[@TREE] ...]   (arm "base" = SHIPPED alone; @TREE runs the
# arm from another source tree, e.g. an origin/main export, for a change that is not a lever; levers "-" leaves
# TT_BIO_LEVERS unset, so the arm runs the Mol-1 Blackhole defaults of its own tree)
set -u
W=/home/ttuser/.coworker/wt/mol1-bh-trunk; PY=/home/ttuser/tt-bio-dev/env/bin/python
O=$1; S=$2; N=$3; shift 3; mkdir -p "$O"
C=${CARD:-3}; LOCK=/home/ttuser/spd_$(hostname | sed "s/tt-quietbox2/qb2/;s/tt-quietbox/qb1/")_card$C.lock
SHIPPED=apb_ln,atom_k1,conf_bins,dit_chunk,dit_mm16,dit_qkv16,dit_qsplit,dit_sdpa,dit_sdpa32,msa_sample,opm_jd,pz_device,silu_f32,transition_ws,triatt_k1,triatt_ln,triatt_tail,trimul_bres,trimul_ibw,trimul_t1_2g,trimul_t1_c256,trimul_t1_c320,trimul_t1_c384,trimul_t1_c448,trimul_t1_dram,trimul_t2,trimul_t2w,trimul_tail,trunk_hifi3,ws_chain,ws_l1acc,ws_psilu,ws_resid,ws_resid2
cd /home/ttuser/mol1-fair/inputs
exec 9>$LOCK; flock -w 3600 9 || { echo "no flock" >> "$O/ab.log"; exit 1; }
for r in $(seq 1 "$N"); do
  for spec in "$@"; do
    arm=${spec%%=*}; lv=${spec#*=}; T=$W; [ "$lv" = "${lv#*@}" ] || { T=${lv#*@}; lv=${lv%%@*}; }
    L=$SHIPPED; [ "$lv" = - ] && L=; [ "$arm" = base ] || [ -z "$L" ] || L="$SHIPPED,$lv"
    echo "$(date -u +%FT%TZ) rep $r arm $arm" >> "$O/ab.log"
    env ${L:+TT_BIO_LEVERS=$L} TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:mol1-bh-trunk PYTHONPATH=$T \
      timeout 2400 $PY $T/perf/mol1/fair/run.py --input mol1-$S.json --chips $C --part dev --rounds 2 --max-attempts 2 \
      --seed 101 --arm "$arm" --out "$O/$arm/r$r" >> "$O/ab.log" 2>&1
    echo "$(date -u +%FT%TZ) rc=$? $arm r$r" >> "$O/ab.log"
  done
done
touch "$O/DONE"
