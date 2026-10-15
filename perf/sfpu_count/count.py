"""Count the SFPU instructions an SFPI header compiles to, per function, without a device.

    python perf/sfpu_count/count.py [--rev REV] [--src perf/sfpu_count/sigmoid_poly.cpp]

Compiles the source (which includes the header named by -DHDR) with the SFPI compiler for WH and BH, against
stub LLK wrappers that call the face function four times, and prints instructions / SFPU instructions / SFPNOPs
per call of each function (loops followed). --rev also builds the header as of that git revision, for a before/after. The loop body is what
costs: read it with `riscv-tt-elf-objdump -d`. Needs -fno-exceptions -fno-rtti (plain -O3 ICEs in sfpi 7.35).
"""
import argparse, os, re, subprocess, tempfile

SFPI = "/opt/tenstorrent/sfpi"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
HDR = "tt_bio/kernels/trimul_gin_moved/sigmoid_poly.hpp"


def count(obj):
    """Executed instructions per call of each function: walks the code, following li/addi/bnez loop counters."""
    out = subprocess.run([f"{SFPI}/compiler/bin/riscv-tt-elf-objdump", "-d", obj], capture_output=True, text=True).stdout
    fns, fn = {}, None
    for line in out.splitlines():
        m = re.match(r"^([0-9a-f]+) <(_Z\w+)>:", line)
        if m:
            fn = m.group(2)
            fns[fn] = []
            continue
        m = re.match(r"^\s+([0-9a-f]+):\s+\S+\s+(\S+)\s*(\S*)", line)
        if m and fn:
            fns[fn].append((int(m.group(1), 16), m.group(2), m.group(3).split(",")))
    res = {}
    for fn, code in fns.items():
        at = {addr: i for i, (addr, _, _) in enumerate(code)}
        regs, n, sfp, nop, i = {}, 0, 0, 0, 0
        while i < len(code) and n < 10**6:
            _, op, args = code[i]
            n += 1
            sfp += op.startswith("sfp")
            nop += op == "sfpnop"
            if op == "ret":
                break
            if op == "li":
                regs[args[0]] = int(args[1], 0)
            elif op == "addi" and args[0] in regs:
                regs[args[0]] = regs[args[1]] + int(args[2], 0)
            elif op == "bnez" and regs.get(args[0], 0) != 0:
                i = at[int(args[1].split()[0], 16)]
                continue
            i += 1
        res[fn] = [n, sfp, nop]
    return res


def build(hdr, src, cpu, obj):
    subprocess.run([f"{SFPI}/compiler/bin/riscv-tt-elf-g++", f"-mcpu={cpu}", "-O3", "-fno-exceptions", "-fno-rtti",
                    "-ffast-math", "-std=c++17", f"-I{SFPI}/include", f"-I{HERE}/stub", f'-DHDR="{hdr}"', "-c", src,
                    "-o", obj], check=True)


ap = argparse.ArgumentParser()
ap.add_argument("--rev", help="also count the header as of this git revision")
ap.add_argument("--src", default=os.path.join(HERE, "sigmoid_poly.cpp"))
ap.add_argument("--hdr", default=HDR)
a = ap.parse_args()
with tempfile.TemporaryDirectory() as tmp:
    hdrs = [("work", os.path.join(REPO, a.hdr))]
    if a.rev:
        old = os.path.join(tmp, "old.hpp")
        with open(old, "w") as f:
            f.write(subprocess.run(["git", "-C", REPO, "show", f"{a.rev}:{a.hdr}"], capture_output=True, text=True,
                                   check=True).stdout)
        hdrs.insert(0, (a.rev, old))
    for cpu in ("tt-wh-tensix", "tt-bh-tensix"):
        for name, h in hdrs:
            obj = os.path.join(tmp, f"{name}_{cpu}.o")
            build(h, a.src, cpu, obj)
            for fn, (n, s, nop) in count(obj).items():
                print(f"{cpu} {name:>10} {fn:24} instr {n:3d}  sfpu {s:3d}  nop {nop}")
