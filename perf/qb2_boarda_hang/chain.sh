#!/bin/bash
# chain.sh: board B first (healthy chips 2,3, fw 19.15), then board A (chips 0,1, fw 19.11), the same D3A leg each.
W=$HOME/qb2-boarda-hang; cd $W
./hk2.sh B r1B; echo "B exit $?" >> $W/chain.log
./hk2.sh A r1A; echo "A exit $?" >> $W/chain.log
echo "CHAIN-DONE $(date -u +%FT%TZ)" >> $W/chain.log
