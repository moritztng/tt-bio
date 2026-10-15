#!/bin/bash
# phase21.sh: firmware A/B on qb2 board 410D (chips 0+1, PCI 0/1) for the chip-0 FPU clock-gate hang (ASIC B9017B91B319B0DE).
# Source facts (tt-system-firmware): 19.12 3738f7b2 moved p300c from the legacy V/F curve (832.6 mV at 1350) to the
# process-based curve (809 mV for this die); 19.15 49ae3c4b turned Tensix CG on (inverted off since 18.10).
# 19.11 = legacy curve + CG off; 19.16 = process curve + CG on (no change vs 19.15). Only board 410D is flashed
# (fw/flash410d.py filters tt-flash to PCI 0/1). Each fw: VF query, CG read, pcG x1000 at nominal 1350, stop a fw at its first HANG.
# Ends on the fw named in $1 (default 19.15.0, the starting point). Index: "p21 fw=<v> leg=<n> verdict=HANG|CLEAN".
cd ~/spd/spd-nocdeadlock; final=${1:-19.15.0}
until grep -q P20-DONE phase20.log; do sleep 15; done
touch ~/spd/locks/chip1.BAD
for c in 0 1; do eval "exec $((8+c))>~/spd_qb2_chip$c.lock"; flock $((8+c)); done
echo "locks 0,1 held $(date -u +%FT%TZ)"
py=~/.tenstorrent-venv/bin/python; tree=~/spd/spd-nocdeadlock/fixm
$py tel.py p21_tel.csv 0.2 & tp=$!
fwv() { timeout 120 $py -c "
import pyluwen
for n in (0, 1):
    c = pyluwen.PciChip(pci_interface=n); t = c.get_telemetry()
    v = [c.arc_msg(0xA6, wait_for_done=True, arg0=f, arg1=0)[0] for f in (1250, 1350)]
    print(f\"chip{n} fw={t.fw_bundle_version:08x} vf1250={v[0]} vf1350={v[1]}\")" 2>&1 | tr "\n" " "; }
reset() { timeout 300 ~/.local/bin/tt-smi -r 0,1 > /dev/null 2>&1; echo "reset 0,1 rc=$? $(date -u +%FT%TZ)"; sleep 15; }
flash() { echo "== flash $1 $(date -u +%FT%TZ)"
  (cd fw && timeout 1800 /usr/bin/python3 flash410d.py ~/fw_pack-$1.fwbundle $2 > flash_$1.log 2>&1); echo "flash rc=$? $(tail -3 fw/flash_$1.log | tr "\n" " ")"
  reset; echo "fw: $(fwv)"; echo "cg: $(TT_VISIBLE_DEVICES=0 timeout -s TERM 300 ~/triage_venv/bin/python cgctl.py read 2>&1 | tail -1)"; }
legs() { for n in $(seq 1 $2); do tag=p21_${1//./}_$n
  echo "== $tag start $(date -u +%FT%TZ)"; ./repro.sh 0 $tag $tree --iters 1000 --every 50
  d=$(grep -c REPRO-DONE rp_${tag}_c0.log); V=CLEAN; [ "$d" = 1 ] || V=HANG
  echo "p21 fw=$1 leg=$n verdict=$V last: $(grep "^it " rp_${tag}_c0.log | tail -1) end $(date -u +%FT%TZ)"
  [ $V = HANG ] && { reset; break; }; done; }
echo "fw before: $(fwv)"
flash 19.11.0 --force; legs 19.11.0 5
flash 19.16.0;         legs 19.16.0 3
flash $final --force
kill -TERM $tp; echo "P21-DONE final=$final fw: $(fwv) $(date -u +%FT%TZ)"
