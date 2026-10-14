"""vdd.py <node> <mV|0>: force (or with 0 release) VDD of ONE Blackhole chip via ARC message FORCE_VDD (0x39),
opening only /dev/tenstorrent/<node>. The fw rejects values outside the fw table vdd_min..vdd_max (return 1).
Diagnostic only (spd-nocdeadlock2 phase 19): is the FPU clock-gate hang voltage-sensitive?"""
import glob, sys, time, pyluwen
n, v = int(sys.argv[1]), int(sys.argv[2])
hw = glob.glob(f"/sys/class/tenstorrent/tenstorrent!{n}/device/hwmon/hwmon*/in0_input")[0]
rd = lambda: open(hw).read().strip()
print("before mV", rd(), flush=True)
chip = pyluwen.PciChip(pci_interface=n)
print("arc_msg ->", chip.arc_msg(0x39, wait_for_done=True, arg0=v, arg1=0), flush=True)
time.sleep(1); print("after mV", rd(), flush=True)
