#!/usr/bin/env python3
"""Index of the first TEXT_PROBE bytes of every cached kernel's .text, per RISC, for naming a hung kernel without
Inspector (triage-no-inspector.patch). A JapanFold box caches ~350k kernel ELFs; parsing them all takes longer than a
triage may run, so the index is built once into a pickle beside this file and topped up with ELFs newer than it.
Usage (on the box, detached): kernel_text_index.py ROOT[:ROOT...]
"""
import os, pickle, subprocess, sys
from pathlib import Path

TEXT_PROBE = 64
INDEX = Path(__file__).with_name("kernel_text_index.pkl")


def _text(path):
    from elftools.elf.elffile import ELFFile

    with open(path, "rb") as f:
        sec = ELFFile(f).get_section_by_name(".text")
        return sec.data()[:TEXT_PROBE] if sec is not None else None


def load(roots: str) -> dict[str, list[tuple[bytes, str]]]:
    """{risc: [(text prefix, xip path)]}, updated with every ELF under ROOTS newer than the saved index."""
    index, newer = {}, []
    if INDEX.exists():
        index = pickle.loads(INDEX.read_bytes())
        newer = ["-newer", str(INDEX)]
    seen = {x for v in index.values() for _, x in v}
    for root in roots.split(":"):
        found = subprocess.run(["find", root, "-name", "*.elf.xip.elf", *newer], capture_output=True, text=True).stdout
        for xip in found.split():
            if xip in seen:
                continue
            try:
                text = _text(xip)
            except Exception:
                continue
            if text:
                index.setdefault(os.path.basename(os.path.dirname(xip)), []).append((text, xip))
    tmp = INDEX.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps(index))
    tmp.replace(INDEX)
    return index


if __name__ == "__main__":
    idx = load(sys.argv[1])
    print({k: len(v) for k, v in idx.items()})
