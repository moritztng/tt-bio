#!/usr/bin/env python3
"""Build a patched copy of the ttnn JIT source tree, without forking or rebuilding tt-metal.

tt-metal compiles every device kernel at runtime from source that the ttnn wheel ships as plain
text, and `tt_metal/llrt/rtoptions.cpp:216` lets TT_METAL_RUNTIME_ROOT repoint that tree. So an LLK
change is a text edit plus an env var: no fork, no source build, no host rebuild.

This builds the overlay with hard links (a few KB on disk, not 180 MB) and materialises only the
files it edits.

    python3 make_overlay.py --arm no_stall   -> drop the per-tile STALLWAIT, keep WRCFG
    python3 make_overlay.py --arm reg2flop   -> use Wormhole's REG2FLOP path verbatim
    python3 make_overlay.py --arm none       -> unpatched control, for the A/A floor

Then run the fold with the printed TT_METAL_RUNTIME_ROOT exported. Both arms are address-programming
only, so the fold must stay bit-exact; a wrong or hanging fold is the falsification.
"""
import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

# tt-llk a7a846ae (the LLK submodule pinned by tt-metal v0.68.0). Refuse to patch anything else.
CPACK = "tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/cpack_common.h"
CPACK_SHA = "1e01e001fc6a00eb48b2ea33ebbc033cc4d301a320887af4d70e2c9e8c34a31f"

ORIG = """    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
    TTI_WRCFG(p_gpr_pack::OUTPUT_ADDR, 0, THCON_SEC0_REG1_L1_Dest_addr_ADDR32);
"""

ARMS = {
    # Keep the config write, drop only the stall. Minimal delta, isolates the stall itself.
    "no_stall": """    // b2z-metal-source-recon: per-tile config-pipe stall removed (Wormhole does not take it).
    TTI_WRCFG(p_gpr_pack::OUTPUT_ADDR, 0, THCON_SEC0_REG1_L1_Dest_addr_ADDR32);
""",
    # Wormhole's sequence, verbatim from tt_llk_wormhole_b0/common/inc/cpack_common.h.
    "reg2flop": """    // b2z-metal-source-recon: Wormhole's direct GPR->CFG path. REG2FLOP is available on
    // Blackhole (ckernel_ops.h:413) and THCON_CFGREG_BASE_ADDR32 = 64 (bh cfg_defines.h:3079).
    TTI_REG2FLOP(
        1, 0, 0, 0, THCON_SEC0_REG1_L1_Dest_addr_ADDR32 - THCON_CFGREG_BASE_ADDR32, p_gpr_pack::OUTPUT_ADDR);
    TTI_PACR(ADDR_MOD_2, 0, 0xf, 0, 0, 1, 0);  // pack flush
""",
    "none": ORIG,
}


def ttnn_root() -> Path:
    """The JIT source root: the directory that holds both tt_metal/ and ttnn/."""
    import ttnn  # noqa: PLC0415  -- only needed to locate the installed tree
    here = Path(ttnn.__file__).resolve().parent
    for cand in (here, *here.parents):
        if (cand / CPACK).is_file():
            return cand
    return here


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(ARMS), required=True)
    ap.add_argument("--out", default=None, help="overlay root (default: /tmp/b2z-overlay-<arm>)")
    ap.add_argument("--root", default=None, help="source tree (default: the installed ttnn)")
    a = ap.parse_args()

    src = Path(a.root).resolve() if a.root else ttnn_root()
    if not (src / CPACK).is_file():
        print(f"ERROR: {src} does not look like a ttnn JIT source root ({CPACK} missing)")
        return 2
    got = hashlib.sha256((src / CPACK).read_bytes()).hexdigest()
    if got != CPACK_SHA:
        print(f"ERROR: {CPACK} is {got[:16]}..., expected {CPACK_SHA[:16]}... (tt-llk a7a846ae).\n"
              "The LLK moved; re-read the function before trusting this patch.")
        return 2

    out = Path(a.out) if a.out else Path(f"/tmp/b2z-overlay-{a.arm}")
    if out.exists():
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Hard links: the overlay costs inodes, not bytes. Falls back to a copy across filesystems.
    if subprocess.run(["cp", "-al", str(src), str(out)], capture_output=True).returncode:
        shutil.copytree(src, out, symlinks=True)

    target = out / CPACK
    text = target.read_text()
    if text.count(ORIG) != 1:
        print(f"ERROR: expected exactly 1 occurrence of the stall sequence, found {text.count(ORIG)}")
        return 2
    target.unlink()                       # break the hard link before writing
    target.write_text(text.replace(ORIG, ARMS[a.arm]))

    print(f"arm          {a.arm}")
    print(f"source       {src}")
    print(f"overlay      {out}")
    print(f"patched      {CPACK}")
    print(f"\nexport TT_METAL_RUNTIME_ROOT={out}")
    print("# clear the kernel cache between arms, or the old binaries are reused:")
    print("rm -rf ~/.cache/tt-metal-cache*")
    return 0


if __name__ == "__main__":
    sys.exit(main())
