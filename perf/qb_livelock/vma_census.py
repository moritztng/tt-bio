#!/usr/bin/env python3
"""What a running fold's address space is actually made of, by mapping class.

The teardown is paid per VMA and per resident page, so before choosing a lever it is worth
knowing which class holds the VMAs. glibc arenas can be capped and trimmed; /dev/tenstorrent
mappings come back only through the driver at device close; file-backed pages are cheap to
unmap and were never the 4.7 GB anyway.

    python3 perf/qb_livelock/vma_census.py <pid|smaps-file>
"""
import re
import sys
from collections import defaultdict

arg = sys.argv[1]
src = arg if "/" in arg else f"/proc/{arg}/smaps"
HDR = re.compile(r"^([0-9a-f]+)-([0-9a-f]+) (\S{4}) \S+ \S+ \S+\s*(.*)$")


def cls(path, perms, size_kB):
    if path.startswith("/dev/tenstorrent"):
        return "/dev/tenstorrent (driver)"
    if path.startswith("/dev/hugepages") or path.startswith("/dev/shm"):
        return "hugepages / shm"
    if path in ("[heap]",):
        return "[heap]"
    if path.startswith("[stack"):
        return "[stack]"
    if path.startswith("["):
        return path
    if path:
        return ".so / file-backed" if ".so" in path else "other file-backed"
    if perms == "---p" and size_kB >= 32 * 1024:
        return "anon ---p guard (glibc arena reserve)"
    return f"anon {perms}"


n = defaultdict(int)
sz = defaultdict(int)
rss = defaultdict(int)
cur = None
total_vmas = 0
with open(src) as f:
    for line in f:
        m = HDR.match(line)
        if m:
            start, end, perms, path = m.groups()
            size_kB = (int(end, 16) - int(start, 16)) // 1024
            cur = cls(path.strip(), perms, size_kB)
            n[cur] += 1
            sz[cur] += size_kB
            total_vmas += 1
        elif cur and line.startswith("Rss:"):
            rss[cur] += int(line.split()[1])

print(f"{src}: {total_vmas} VMAs, {sum(rss.values())/1e6:.2f} GB resident, "
      f"{sum(sz.values())/1e6:.2f} GB mapped")
print(f"{'class':38s} {'vmas':>6s} {'mapped_MB':>10s} {'rss_MB':>8s}")
for k in sorted(n, key=lambda k: -rss[k]):
    print(f"{k:38s} {n[k]:6d} {sz[k]/1024:10.0f} {rss[k]/1024:8.0f}")
