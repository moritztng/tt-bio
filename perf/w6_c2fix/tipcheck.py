#!/usr/bin/env python3
"""Do C2FIX's anchors still occur exactly once at an arbitrary main tip?

Card-free. Answers the only question that matters for merging a hand-reapplied patch onto a tip it
was not measured on: does it still apply mechanically, or has main's drift collided with it?

    python3 perf/w6_c2fix/tipcheck.py [ref]     # default origin/main
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arm  # noqa: E402

ref = sys.argv[1] if len(sys.argv) > 1 else "origin/main"
src = subprocess.run(["git", "show", f"{ref}:tt_bio/tenstorrent.py"], cwd=arm.REPO,
                     capture_output=True, text=True, check=True).stdout

anchors = [("trimul-cfg decorator", arm.ANCHOR_BASE),
           ("TriangleAttention permute 1", arm.PERM1_BASE),
           ("TriangleAttention permute 2", arm.PERM2_BASE),
           ("TriangleAttention permute 3", arm.PERM3_BASE)]

ok = True
print(f"C2FIX anchors in {ref} ({subprocess.run(['git','rev-parse','--short',ref],cwd=arm.REPO,capture_output=True,text=True).stdout.strip()}):")
for name, a in anchors:
    n = src.count(a)
    ok &= (n == 1)
    print(f"  {n}x  {name}" + ("" if n == 1 else "   <-- NOT UNIQUE, reapplication is no longer mechanical"))
print("  helper already present:", "_transpose_memory_config" in src)
print("VERDICT:", "applies mechanically" if ok else "COLLISION — needs a fresh manual reapplication")
sys.exit(0 if ok else 1)
