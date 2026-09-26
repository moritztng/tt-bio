"""Per-host card occupancy, printed as card|node|holders|live-lease. Runs ON a QuietBox.

CARD index is NOT the device node: `TT_VISIBLE_DEVICES` counts cards in PCI bus order, which on
qb1 puts card 0 at node 1, 1 at 2, 2 at 3 and 3 at 0. Resolved here rather than assumed.
"""
import glob
import json
import os
import subprocess

ROOT = "/sys/class/tenstorrent"
nodes = sorted(os.listdir(ROOT),
               key=lambda n: os.path.basename(os.path.realpath(f"{ROOT}/{n}/device")))
for card, cls in enumerate(nodes):
    node = cls.split("!")[-1]
    held = subprocess.run(["fuser", f"/dev/tenstorrent/{node}"],
                          capture_output=True, text=True).stdout.split()
    live = ""
    for f in glob.glob(os.path.expanduser(f"~/.coworker/state/leases/*card{card}.json")):
        try:
            pid = int(json.load(open(f)).get("pid") or 0)
        except Exception:
            continue
        if pid > 0:
            try:
                os.kill(pid, 0)
                live = f"{os.path.basename(f)}:{pid}"
            except OSError:
                pass
    print(f"{card}|{node}|{' '.join(held)}|{live}")
