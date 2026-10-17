#!/bin/bash
# chain4.sh: split D3 at its 280 to read the solver across the ~295-310 transition all four runs (D3A, r1A, r1B, r2B)
# share. s280 = D3 command for 280 its (healthy, ~1,000 s, writes state); s280b = 30 more its from that state, whose hist
# shows the per-solve Krylov counts and residuals at D3 its 281-310.
W=$HOME/qb2-boarda-hang; cd $W; export UNTIL=2026-10-17T18:00:00Z
CAP=2000 ./hk2.sh B s280 280; echo "s280 exit $? $(date -u +%FT%TZ)" >> $W/chain.log
[ -s $W/hk2_s280_case_1x2_state.npz ] || { echo "CHAIN4-NOSTATE $(date -u +%FT%TZ)" >> $W/chain.log; exit 1; }
SIN=$W/hk2_s280_case_1x2_state.npz CAP=6000 ./hk2.sh B s280b 30; echo "s280b exit $? $(date -u +%FT%TZ)" >> $W/chain.log
echo "CHAIN4-DONE $(date -u +%FT%TZ)" >> $W/chain.log
