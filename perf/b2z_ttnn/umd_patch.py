#!/usr/bin/env python3
"""Make UMD's ETH heartbeat gate warn-and-continue, the way tt-metal `main` already can.

Screen instrument only. It rewrites the first three bytes of
`tt::umd::TopologyDiscovery::eth_heartbeat_running` to `xor eax,eax; ret` -- return false -- which
is exactly what upstream's `TopologyDiscoveryOptions::eth_fw_heartbeat_failure != Action::THROW`
path does: log and skip that ETH core instead of refusing the whole cluster. The released 0.71-0.78
wheels do not expose that option, and whglx has one chip whose ERISC heartbeat has stopped, so
without this no ttnn above 0.68.0 can open any device on the box.

Never ship this. It lives in one private venv, the original is kept alongside as .orig, and any
number taken through it must say so.
"""
import shutil, struct, subprocess, sys
from pathlib import Path

SYM = "_ZN2tt3umd17TopologyDiscovery21eth_heartbeat_runningEPNS0_8TTDeviceEmNS0_9CoreCoordE"
PATCH = bytes([0x31, 0xC0, 0xC3])  # xor eax,eax ; ret  -> return false


def vaddr_to_off(path: Path, vaddr: int) -> int:
    out = subprocess.run(["readelf", "-lW", str(path)], capture_output=True, text=True).stdout
    for line in out.splitlines():
        p = line.split()
        if len(p) >= 6 and p[0] == "LOAD":
            off, va, _pa, filesz = int(p[1], 16), int(p[2], 16), int(p[3], 16), int(p[5], 16)
            if va <= vaddr < va + filesz:
                return off + (vaddr - va)
    raise SystemExit(f"vaddr {vaddr:#x} not in any LOAD segment")


def main() -> int:
    target = Path(sys.argv[1])
    nm = subprocess.run(["nm", "--defined-only", str(target)], capture_output=True, text=True).stdout
    vaddr = None
    for line in nm.splitlines():
        if line.endswith(" " + SYM):
            vaddr = int(line.split()[0], 16)
    if vaddr is None:
        raise SystemExit(f"symbol not found in {target}")
    off = vaddr_to_off(target, vaddr)

    orig = target.with_suffix(target.suffix + ".orig")
    if not orig.exists():
        shutil.copy2(target, orig)
    blob = bytearray(target.read_bytes())
    before = bytes(blob[off:off + 3])
    if before == PATCH:
        print(f"already patched at {off:#x}")
        return 0
    blob[off:off + 3] = PATCH
    target.write_bytes(bytes(blob))
    print(f"{target.name}: {SYM[:48]}... vaddr {vaddr:#x} file {off:#x}  "
          f"{before.hex()} -> {PATCH.hex()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
