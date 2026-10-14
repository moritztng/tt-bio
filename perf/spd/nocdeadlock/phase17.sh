#!/bin/bash
# phase17.sh: kernel-patch legs on qb2 chip 0 at 1350 MHz after phase 16 (tree fixk = fixm + TT_BIO_KPATCH hook).
# k1_push_sync: tensix_sync after the tiles_received publish (where does TRISC2 hang now?);
# k2_push_riscv: RISC-V publishes tiles_received after the pack thread drains; ibw16: one inner block (no partials
# CB, no packer L1 acc) on the same forced grid. Each pcG x1000; stock hangs chip 0 at iteration 1. Index phase17.log.
cd ~/spd/spd-nocdeadlock
until grep -q P16-DONE phase16.log 2>/dev/null; do sleep 30; done
for c in 0 1; do eval "exec $((8+c))>~/spd_qb2_chip$c.lock"; flock $((8+c)); done
echo "locks 0,1 held $(date -u +%FT%TZ)"
py=~/.tenstorrent-venv/bin/python; tree=~/spd/spd-nocdeadlock/fixk
$py tel.py p17_tel.csv 0.2 & tp=$!
leg() { tag=$1; shift
  timeout 300 ~/.local/bin/tt-smi -r 0,1 > /dev/null 2>&1; echo "reset 0,1 rc=$? $(date -u +%FT%TZ)"; sleep 15
  echo "== $tag c0 KPATCH=${TT_BIO_KPATCH:-none} args=$* start $(date -u +%FT%TZ)"
  ./repro.sh 0 $tag $tree --iters 1000 --every 50 "$@"
  echo "$tag c0: last: $(grep "^it " rp_${tag}_c0.log | tail -1), done=$(grep -c REPRO-DONE rp_${tag}_c0.log), timeout=$(grep -c "TIMEOUT: device timeout" rp_${tag}_c0.log) end $(date -u +%FT%TZ)"
}
export TT_BIO_KPATCH=~/spd/spd-nocdeadlock/kpatch/k1_push_sync.py; leg p17_k1a
export TT_BIO_KPATCH=~/spd/spd-nocdeadlock/kpatch/k2_push_riscv.py; leg p17_k2a
export TT_BIO_KPATCH=~/spd/spd-nocdeadlock/kpatch/k2_push_riscv.py; leg p17_k2b
unset TT_BIO_KPATCH; leg p17_ibw16 --ibw 16
leg p17_stock
kill -TERM $tp; timeout 300 ~/.local/bin/tt-smi -r 0,1 > /dev/null 2>&1; echo "P17-DONE reset rc=$? $(date -u +%FT%TZ)"
