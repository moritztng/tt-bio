#!/bin/bash
# RECOVERY: SIGINT the chip worker in the middle of a visitor's fold and record the stream.
cd ~/.coworker/wt/booth-engine/demo/booth/engine
PY=~/tt-bio-dev/env/bin/python3
GFP=MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK
TT_BIO_LEASE_HOLDER=worker:booth-engine timeout 900 $PY -u server.py --chips 0 --attract "" > runs/logs/server-recovery.log 2>&1 < /dev/null &
SRV=$!
for i in $(seq 60); do grep -q '"state":"ready"' runs/logs/server-recovery.log 2>/dev/null && break; curl -s 127.0.0.1:8626/status | grep -q '"state": "ready"' && break; sleep 2; done
timeout 110 python3 client.py --seconds 100 --fold $(python3 -c 'import bench_live as b; print(b.HSA[:400])') --log runs/client-recovery.jsonl > runs/client-recovery.txt 2>&1 &
CL=$!
sleep 4
W=$(pgrep -f "[c]hipworker.py --chip 0" | head -1)
echo "$(date -u +%T) SIGINT chipworker $W mid-fold" | tee runs/recovery-actions.txt
kill -INT $W
wait $CL
echo "$(date -u +%T) client done" >> runs/recovery-actions.txt
kill -INT $SRV; wait $SRV
echo "$(date -u +%T) server stopped rc=$?" >> runs/recovery-actions.txt
