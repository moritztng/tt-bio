#!/bin/bash
# Mol-1 e2e A/B of the pair-transition lever stack: 8cnh's 475-aa protein chain, single sequence, upstream defaults
# (20 cycles, 200 steps, 1 sample), seeds 100 (warm-up) then 101-103, base levers vs + the transition stack.
# Usage: CARD=<n> [A3M=<a3m>] [STACK=1] perf/mol1_pairmix/e2e_ab.sh OUTDIR. With A3M the target is that MSA's query,
# folded with it. STACK=1 grades Mol-1's whole normal set instead (protenix.WIDE_NORMAL + the trimul row's MOL1_BH),
# on a tree that carries the triangle attention and trimul branches.
set -u
W=$(cd "$(dirname "$0")/../.." && pwd); O=$1; mkdir -p "$O"; C=${CARD:-0}
[ -n "${A3M:-}" ] && export MOL1_A3M=$A3M
SEQ=${A3M:+$(sed -n 2p "$A3M" | tr -d '\n')}
[ -n "$SEQ" ] || SEQ=$(python3 -c "import json;print([s['proteinChain']['sequence'] for j in json.load(open('$HOME/mol1-upstream/mol1-port/src/examples/example_without_msa.json')) if '8cnh' in j['name'] for s in j['sequences'] if 'proteinChain' in s][0])")
# protenix.NORMAL_LEVERS ("normal" alone is the engine-wide set, without the Protenix additions)
BASE=normal+transition_ws+ws_l1acc+trimul_t1_c256+trimul_t1_c384
run() { TT_BIO_LEVERS="$2" TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:mol1-perf-pairmix PYTHONPATH=$W \
  timeout 3000 ~/tt-bio-dev/env/bin/python "$W/perf/mol1_smoke/e2e_tt.py" ~/mol1/weights/mol-1-base.pt "$SEQ" "$O/$1.pt" 100,101,102,103 > "$O/$1.log" 2>&1; }
run base "$BASE"
WS=ws_chain+ws_resid+ws_resid2+ws_psilu
if [ -n "${STACK:-}" ]; then run stack "$BASE+$WS+triatt_ln+apb_ln+triatt_k1+dit_qsplit+pz_device+trimul_t1_c320+trimul_t1_c448+trimul_t1_dram+trimul_t1_2g"
else run ws "$BASE+$WS"; fi
echo E2EDONE >> "$O/done"
