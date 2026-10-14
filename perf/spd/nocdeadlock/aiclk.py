"""aiclk.py <node> <MHz|0>: force (or with 0 release) the AICLK of ONE Blackhole chip via ARC message FORCE_AICLK
(0x33), opening only /dev/tenstorrent/<node>. Prints sysfs tt_aiclk before/after. spd-nocdeadlock2 timing-margin test."""
import sys, time, pyluwen
n, f = int(sys.argv[1]), int(sys.argv[2])
rd = lambda: open(f"/sys/class/tenstorrent/tenstorrent!{n}/tt_aiclk").read().strip()
print("before", rd(), flush=True)
chip = pyluwen.PciChip(pci_interface=n)
print("arc_msg ->", chip.arc_msg(0x33, wait_for_done=True, arg0=f, arg1=0), flush=True)
time.sleep(1); print("after", rd(), flush=True)
