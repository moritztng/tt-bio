#!/usr/bin/env python3
"""Name the kernel and the instruction a hung core waits on, offline, from a hangtriage_prod.py capture.

dump_callstacks (triage-no-inspector.patch) prints `pc=.. base=.. text_offset=..` for every stuck RISC whose kernel has
no DWARF and writes that core's L1 to l1-<loc>.bin. Here the kernel text in L1 is matched against the kernel cache in
full, skipping the words the loader relocates (.rela.text), which is why a fixed-length prefix cannot name it.
Run on the box with the kit's python: name_stuck_kernel.py CAPTURE_DIR [CAPTURE_DIR ...]
"""
import os, pickle, re, subprocess, sys
from pathlib import Path

from elftools.elf.elffile import ELFFile

KIT = Path(__file__).resolve().parent
OBJDUMP = "/opt/tenstorrent/sfpi/compiler/bin/riscv-tt-elf-objdump"
LINE = re.compile(r"\[(\S+) \(\d+,\d+\)\]: (\w+): Failed to dump callstacks.*\| pc=0x([0-9a-f]+) base=0x([0-9a-f]+) text_offset=0x([0-9a-f]+)")


def elf_text(path):
    with open(path, "rb") as f:
        e = ELFFile(f)
        t = e.get_section_by_name(".text")
        rela = e.get_section_by_name(".rela.text")
        skip = {r["r_offset"] for r in rela.iter_relocations()} if rela is not None else set()
        syms = sorted((s["st_value"], s.name) for s in e.get_section_by_name(".symtab").iter_symbols()
                      if s["st_info"]["type"] == "STT_FUNC")
        return t.data(), t["sh_addr"], skip, syms


def same(l1, text, skip, base_addr):
    if len(l1) < len(text):
        return False
    masked = {o - base_addr + k for o in skip for k in range(4)}
    return all(a == b for i, (a, b) in enumerate(zip(l1, text)) if i not in masked)


def name(cap: Path, index):
    for m in LINE.finditer((cap / "triage.txt").read_text()):
        loc, risc, pc, base, off = m.group(1), m.group(2), *(int(g, 16) for g in m.groups()[2:])
        dump = next(cap.glob(f"l1-{loc}[ .]*bin"), None)  # dump_callstacks names it "l1-<noc> (<logical>).bin"
        if dump is None:
            print(f"{cap.name} {loc} {risc}: pc=0x{pc:x}, no L1 dump")
            continue
        start = base + off
        l1 = dump.read_bytes()[start:start + 64 * 1024]
        # Best prefix scores first; parsing every >=48/64 candidate ELF (thousands share a prologue) ran for minutes.
        cands = sorted(((sum(a == b for a, b in zip(t, l1)), x) for t, x in index.get(risc, [])), reverse=True)
        cands = [x for k, x in cands[:300] if k >= 48]
        hits = []
        for x in cands:
            text, addr, skip, syms = elf_text(x)
            if same(l1, text, skip, addr):
                hits.append((x, addr, syms))
        if not hits:
            print(f"{cap.name} {loc} {risc}: pc=0x{pc:x} start=0x{start:x}: no cached kernel matches ({len(cands)} prefix candidates)")
            continue
        x, addr, syms = hits[0]
        # Identical text can come from builds that differ elsewhere (a packer does not see MATH_FIDELITY): list them all.
        fids = sorted({m.group(1) for h in hits for m in [re.search(r"MathFidelity>\((\d)\)",
                       (Path(h[0]).parents[1] / "chlkc_descriptors.h").read_text())] if m})
        rel = (pc - 0xFFC00000 if pc >= 0xFFC00000 else pc - start) + addr
        fn = max((s for s in syms if s[0] <= rel), default=(0, "?"))[1]
        src = Path(x).parents[1]
        inc = next(src.glob("chlkc_math.cpp"), None)
        print(f"{cap.name} {loc} {risc}: {src.parent.name}/{src.name} ({len(hits)} identical, fidelity {'/'.join(fids)}) pc 0x{rel:x} in {fn}"
              + (f" src {inc.read_text().split(chr(34))[-2]}" if inc else ""))
        dis = subprocess.run([OBJDUMP, "-d", x, f"--start-address={rel - 24}", f"--stop-address={rel + 16}"],
                             capture_output=True, text=True).stdout
        print("    " + "\n    ".join(l for l in dis.splitlines() if re.match(r"\s+[0-9a-f]+:", l)))


if __name__ == "__main__":
    idx = pickle.loads((KIT / "kernel_text_index.pkl").read_bytes())
    for c in sys.argv[1:]:
        name(Path(c), idx)
