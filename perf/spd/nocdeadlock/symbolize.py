"""Symbolize a tt-triage dump_callstacks.txt by PC when the triage left the callstack column empty.

usage: python3 symbolize.py <dump_callstacks.txt>   -> one line per (risc, kernel, source site) with core count
"""
import collections, os, re, subprocess, sys

A2L = "/opt/tenstorrent/sfpi/compiler/bin/riscv-tt-elf-addr2line"
RE = "/opt/tenstorrent/sfpi/compiler/bin/riscv-tt-elf-readelf"
text_base = {}


def base(elf):
    if elf not in text_base:
        out = subprocess.run([RE, "-S", elf], capture_output=True, text=True).stdout
        text_base[elf] = int(re.search(r"\s\.text\s+PROGBITS\s+([0-9a-f]+)", out).group(1), 16)
    return text_base[elf]


def site(elf, addr):
    out = subprocess.run([A2L, "-fipC", "-e", elf, hex(addr)], capture_output=True, text=True).stdout
    lines = [l.strip() for l in out.splitlines()]
    # innermost frame + the kernel_main line, paths shortened
    short = lambda l: re.sub(r"/\S*/(\w[\w.]*\.(?:cpp|h|cc|hpp)):", r"\1:", l)
    keep = [short(lines[0])] + [short(l) for l in lines[1:] if "kernel_main" in l or "MAIN" in l]
    return " <- ".join(keep)


hdr = None
agg = collections.defaultdict(list)
for row in open(sys.argv[1]):
    if not row.startswith(("┃", "│")):
        continue
    cells = [c.strip() for c in row.strip().strip("┃│").split("┃" if row.startswith("┃") else "│")]
    if row.startswith("┃"):
        hdr = cells
        continue
    r = dict(zip(hdr, cells))
    try:
        pc, off = int(r["PC"], 16), int(r["Kernel Offset"], 16)
    except (KeyError, ValueError):
        continue
    elf = r["Kernel Path"]
    if not os.path.exists(elf):
        continue
    s = site(elf, base(elf) + pc - off) if off <= pc else "pc<offset (firmware) pc=" + r["PC"]
    agg[(r["RiscV"], r["Kernel Name"], r.get("Waypoint", ""), s)].append(r["Loc"].split()[0])

for (risc, k, wp, s), locs in sorted(agg.items(), key=lambda kv: (kv[0][0], -len(kv[1]))):
    print(f"{len(locs):4d} {risc:7s} {k} wp={wp}\n       {s}\n       {' '.join(locs[:12])}{' ...' if len(locs) > 12 else ''}")
