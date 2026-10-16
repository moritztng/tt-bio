#!/bin/bash
# chain3.sh: board B rerun r2B after qb2's 15:30Z reboot killed chain.sh/chain2.sh (r1A had stalled; r2B never ran).
W=$HOME/qb2-boarda-hang; cd $W
./hk2.sh B r2B; echo "B2 exit $?" >> $W/chain.log
echo "CHAIN3-DONE $(date -u +%FT%TZ)" >> $W/chain.log
