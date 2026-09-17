#!/usr/bin/env python3
"""Host-side check of the `_LINEAR_BLOCK` lever. No device: only the routing decision is tested.

What this can prove without a card: that every tabulated key produces the family and `in0_block_w`
it was measured with, that the drain block equals the per-core block on the live grid, and that
every guard actually refuses. What it cannot prove is the rate -- that is the fold A/B, and it is
still owed. The negative controls below are here because a guard that never fires and a guard that
always fires read identically from the passing side.
"""
import sys

import ttnn

sys.path.insert(0, "/home/ttuser/.coworker/wt/c12-kblock-unlock")
import tt_bio.tenstorrent as TB

GRID = ttnn.CoreGrid(y=10, x=11)          # the grid every entry was measured on
NC = 110

# (mt_total, kt, nt) -> the shape a fold actually presents, so the key is exercised through the
# same arithmetic the call site goes through rather than being handed to the dict directly.
SHAPES = {
    (160, 4, 16): ((1, 16, 320, 128), (128, 512)),
    (256, 4, 16): ((1, 16, 512, 128), (128, 512)),
    (384, 4, 16): ((1, 16, 768, 128), (128, 512)),
    (160, 16, 4): ((1, 16, 320, 512), (512, 128)),
    (256, 16, 4): ((1, 16, 512, 512), (512, 128)),
    (384, 16, 4): ((1, 16, 768, 512), (512, 128)),
    (16, 24, 24): ((1, 512, 768), (768, 768)),
    (24, 24, 24): ((1, 768, 768), (768, 768)),
}
# Shapes the table must NOT name, each removed on a measurement rather than an opinion: DiT at
# 298 aa reads 0.8925x through the production path (a regression), CTB reads 0.9935x at 512 aa and
# 1.0176x at 768 aa, both inside their own A/A floors. If an entry for one of these reappears, this
# test fails rather than the fold quietly getting slower at one size.
ABSENT = {
    (10, 24, 24): ((1, 320, 768), (768, 768)),
    (10, 24, 48): ((1, 320, 768), (768, 1536)),
    (16, 24, 48): ((1, 512, 768), (768, 1536)),
    (24, 24, 48): ((1, 768, 768), (768, 1536)),
}
fail = []


def cfg(a, w, act=None, bias=None, grid=GRID):
    return TB._linear_block_cfg(a, w, act, bias, grid)


# --- guard: with the flag off, nothing fires anywhere -----------------------------------------
TB._LINEAR_KBLOCK = False
for key, (a, w) in SHAPES.items():
    if cfg(a, w) is not None:
        fail.append(f"flag OFF still returned a config for {key}")
print("flag off: %d keys, all inert -> %s" % (len(SHAPES), not fail))

TB._LINEAR_KBLOCK = True
assert set(SHAPES) == set(TB._LINEAR_BLOCK), (
    "test shapes and table keys disagree: %s" % (set(SHAPES) ^ set(TB._LINEAR_BLOCK),))

for key, (a, w) in SHAPES.items():
    fam, bw = TB._LINEAR_BLOCK[key]
    c = cfg(a, w)
    if c is None:
        fail.append(f"{key}: table names it but no config was built")
        continue
    mt_total, kt, nt = key
    pcm, pcn = ((-(-mt_total // NC), nt) if fam == "1d"
                else (-(-mt_total // GRID.y), -(-nt // GRID.x)))
    got_fam = "1d" if "1D" in type(c).__name__ else "2d"
    checks = {
        "family": (got_fam, fam),
        "in0_block_w": (int(c.in0_block_w), bw),
        "per_core_M": (int(c.per_core_M), pcm),
        "per_core_N": (int(c.per_core_N), pcn),
        "out_block_h == per_core_M": (int(c.out_block_h), pcm),
        "out_block_w == per_core_N": (int(c.out_block_w), pcn),
    }
    bad = {k: v for k, v in checks.items() if v[0] != v[1]}
    for k, (g, e) in bad.items():
        fail.append(f"{key}: {k} got {g}, expected {e}")
    sub = int(c.out_subblock_h) * int(c.out_subblock_w)
    if sub > 4:
        fail.append(f"{key}: out_subblock {c.out_subblock_h}x{c.out_subblock_w} = {sub} > 4 dest tiles")
    print("  %-14s %s bw=%-3d pcm=%-3d pcn=%-3d drain=%dx%d sub=%dx%d  %s"
          % (key, got_fam, c.in0_block_w, c.per_core_M, c.per_core_N, c.out_block_h,
             c.out_block_w, c.out_subblock_h, c.out_subblock_w, "ok" if not bad else "FAIL"))

for key, (a, w) in ABSENT.items():
    if key in TB._LINEAR_BLOCK:
        fail.append(f"{key} is back in the table -- it was removed on a measured regression or a "
                    f"ratio inside its own A/A floor")
    if cfg(a, w) is not None:
        fail.append(f"{key} must stay inert and did not")
print("  %d removed keys still absent and inert" % len(ABSENT))

# --- negative controls: each must REFUSE, and each is a distinct guard -------------------------
a512, w512 = SHAPES[(256, 4, 16)]
neg = {
    "fused activation": cfg(a512, w512, act="silu"),
    "bias present": cfg(a512, w512, bias=object()),
    "no core_grid": cfg(a512, w512, grid=None),
    "shape not in table": cfg((1, 16, 448, 128), (128, 512)),
    "not tile aligned": cfg((1, 16, 500, 128), (128, 512)),
    "inner dims disagree": cfg((1, 16, 512, 128), (256, 512)),
    "1-D operand": cfg((128,), (128, 512)),
}
for name, got in neg.items():
    if got is not None:
        fail.append(f"negative control did not refuse: {name}")
    print("  refused: %-22s %s" % (name, got is None))

# The controls have to be able to fail, or they prove nothing. Corrupt one entry so the tabulated
# in0_block_w no longer divides Kt and confirm the divisibility guard is what refuses it.
TB._LINEAR_BLOCK[(256, 4, 16)] = ("2d", 3)
if cfg(a512, w512) is not None:
    fail.append("divisibility guard did not fire on in0_block_w=3 with Kt=4")
TB._LINEAR_BLOCK[(256, 4, 16)] = ("2d", 4)
if cfg(a512, w512) is None:
    fail.append("restoring the entry did not restore the config -- the guard is stuck on")
print("  divisibility guard fires on bw=3 / Kt=4, and releases when restored")

print("\n%s" % ("FAIL:\n  " + "\n  ".join(fail) if fail else "all checks passed"))
sys.exit(1 if fail else 0)
