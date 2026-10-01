#!/bin/bash
cd "$(dirname "$0")"
./arm.sh c512 5 1 --binder 370 > out_c512.log 2>&1 &
p=$!; sleep 1; python3 cpusamp.py $p out/c512_cpu.json; wait $p; echo CPU_DONE rc=$?
