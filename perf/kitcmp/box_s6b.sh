#!/bin/bash
# session 6b: 256 under max-autotune-no-cudagraphs, the one size the nocg mode never ran. Runs after the
# s5 cells exit, then writes the markers box_rfd4_ma.sh would have written.
B=/root/kc/rfd4; cd /root/kc
say(){ echo "$(date -u +%FT%TZ) box_s6b: $*"; }
while kill -0 2714 2>/dev/null; do sleep 20; done
echo 0 > $B/CELLS-DONE; say "s5 cells exited, starting 256 max-autotune-nocg"
for b in 256 128; do
  O=$B/s6-max-autotune-nocg/256-b$b
  timeout 2700 /opt/rfd4-venv/bin/python /root/kc/rfd4_gpu.py --spec /root/kc/inputs/pdl1-256.yaml --size 256 \
    --arm max-autotune-nocg --batch $b --warm 5 --out $O \
    --set nn.torch_compile_mode=max-autotune-no-cudagraphs --set 'compile_lengths=[256]' >> $B/s6b.log 2>&1
  grep -q '"kind": "warm"' $O/runs.jsonl 2>/dev/null && break
  say "b$b produced no warm batch"
done
echo done > $B/RFD4-DONE; say "RFD4-DONE"
