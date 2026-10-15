"""vfq.py <node>...: print the voltage each Blackhole chip's firmware V/F curve asks for at 1000-1350 MHz
(ARC GET_VOLTAGE_CURVE_FROM_FREQ, 0xA6). Read-only. spd-nocdeadlock2: fw >= 19.12 uses the process-based curve on p300c."""
import sys, pyluwen
for n in map(int, sys.argv[1:]):
    c = pyluwen.PciChip(pci_interface=n)
    print(n, [(f, c.arc_msg(0xA6, wait_for_done=True, arg0=f, arg1=0)[0]) for f in (1000, 1200, 1250, 1300, 1350)], flush=True)
