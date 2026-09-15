"""qb2 reboot history from the host watchdog trace, back to 00:09 today.

Which way does the arrow point? Either the host dies on its own and a 1024 aa fold is its victim, or
the fold wedges the card and the host dies spinning on PCIe reads to it. Those are opposite
verdicts -- no engine defect versus a real one -- and the correlation alone cannot separate them.

The reboot history can. wdtrace.tsv has sampled every 2 s since 00:09 and carries a boot_id, so a
change of boot_id is a reboot. If qb2 was resetting every ten minutes overnight, with no 1024 aa
fold of mine on it, the host is the cause. If it was stable overnight and only began resetting when
this ladder started, the fold is implicated.
"""
import time
from pathlib import Path

rows = Path("/home/ttuser/qbfix/wdtrace.tsv").read_text().splitlines()
hdr = rows[0].split("\t")
i_epoch, i_boot = hdr.index("epoch"), hdr.index("boot_id")
i_load = hdr.index("load1")

runs = []
for ln in rows[1:]:
    f = ln.split("\t")
    if len(f) <= max(i_epoch, i_boot, i_load):
        continue
    try:
        e = float(f[i_epoch])
    except ValueError:
        continue
    b = f[i_boot]
    if runs and runs[-1][0] == b:
        runs[-1][2] = e
        runs[-1][3] = max(runs[-1][3], float(f[i_load] or 0))
    else:
        runs.append([b, e, e, float(f[i_load] or 0)])

print("%-10s %-9s %-9s %8s %8s" % ("boot_id", "first", "last", "span_min", "max_load1"))
for b, first, last, ml in runs:
    print("%-10s %-9s %-9s %8.1f %8.2f" % (
        b, time.strftime("%H:%M:%S", time.gmtime(first)),
        time.strftime("%H:%M:%S", time.gmtime(last)), (last - first) / 60.0, ml))
print("\n%d boot cycles in the trace" % len(runs))
