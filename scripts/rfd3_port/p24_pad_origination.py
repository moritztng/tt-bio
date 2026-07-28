"""Which ttnn ops define their output's tile padding, and which leave it to the heap?

p23 proved `ttnn.scatter` leaves its output tile padding as whatever the freshly allocated
DRAM buffer held, and that `ttnn.softmax(dim=-1)` reads the pad columns of its reduction
axis. p23 §8 then listed three more softmax sites whose reduction axis is tile-padded
(GatedCrossAttention upcast/downcast, PairformerAttention) and measured their pad columns
as 0 -- but only for the process histories it happened to run.

This settles it structurally instead of empirically. Every op is run twice with BIT-IDENTICAL
inputs: once on a clean heap, once after DRAM has been primed with +/-inf and +/-3e38 across
a spread of footprints. The two runs differ in nothing but what the allocator hands back.

    pad region identical               -> the op WRITES its output padding
                                          (a function of its inputs, not of the heap)
    pad region tracks the priming      -> the op ORIGINATES undefined padding

An op that writes its padding cannot introduce heap garbage, so a chain built only from such
ops and rooted at host uploads (which zero-fill padding) has defined padding by construction
-- which is a proof of non-exposure, not a failure to find exposure.

Usage: p24_pad_origination.py <tree>
"""
from __future__ import annotations

import sys

import torch

if len(sys.argv) > 1:
    sys.path.insert(0, sys.argv[1])

import ttnn  # noqa: E402
from tt_bio.tenstorrent import get_device  # noqa: E402

dev = get_device()

PATTERNS = (float("inf"), -float("inf"), 3.3e38, -3.3e38)
FOOTPRINTS = ((1, 4, 32, 32), (1, 4, 64, 64), (1, 4, 128, 160), (1, 4, 160, 160),
              (1, 4, 256, 256), (1, 16, 256, 256), (1, 4, 512, 512), (40, 4, 32, 32),
              (1, 40, 32, 256), (1, 1, 64, 128))


def prime():
    """Dirty the WHOLE DRAM heap, bottom included.

    Priming with small footprints is not enough: tt-metal hands the low addresses to
    whatever is allocated first, so a small dirty buffer gets reused by the op's *inputs*
    and the output lands on untouched DRAM above it (p23's own op-level probe missed for
    exactly this reason -- it planted at 0x40 and the scatter output came back at 0x17340).
    Filling almost all of DRAM and freeing it leaves no clean region for anything to land on.
    """
    n = 0
    for pat in PATTERNS:
        held = []
        while True:
            try:
                held.append(ttnn.full((1, 32, 2048, 2048), pat, dtype=ttnn.bfloat16,
                                      layout=ttnn.TILE_LAYOUT, device=dev))   # 256 MB each
            except Exception:  # noqa: BLE001  DRAM full
                break
        n += len(held)
        for t in reversed(held):
            ttnn.deallocate(t)
    # a final pass of small footprints so sub-tile-granular slots are dirty too
    for pat in PATTERNS:
        for shape in FOOTPRINTS:
            for dt in (ttnn.bfloat16, ttnn.float32):
                try:
                    t = ttnn.full(shape, pat, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
                except Exception:  # noqa: BLE001
                    continue
                ttnn.deallocate(t)
                n += 1
    return n


def pad_region(t):
    full = t.cpu().to_torch_with_padded_shape().float()
    box = tuple(slice(0, s) for s in t.shape)
    rest = full.clone()
    rest[box] = 0.0
    return rest.flatten()


def describe(v):
    fin = v[torch.isfinite(v)]
    nz = fin[fin != 0]
    return "absmax=%-11.5g nonfinite=%-4d nonzero=%-7d uniq3=%s" % (
        float(fin.abs().max()) if fin.numel() else 0.0,
        int((~torch.isfinite(v)).sum()), int(nz.numel()),
        sorted(set(nz.flatten().tolist()))[:3])


def up(shape, dtype=ttnn.bfloat16, seed=0, const=None):
    """Host upload -- zero-fills tile padding, so inputs are always clean AND identical."""
    g = torch.Generator().manual_seed(seed)
    h = torch.full(shape, const) if const is not None else torch.randn(*shape, generator=g)
    return ttnn.from_torch(h, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev)


CASES = []


def case(name, build, run):
    CASES.append((name, build, run))


I, NK, NQ, NH, HD = 40, 14, 1, 4, 32
C = NH * HD

case("ttnn.scatter (p23: originator)",
     lambda: (ttnn.full((1, 4, 130, 130), -1e4, dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, device=dev),
              ttnn.from_torch(torch.arange(8).view(1, 1, 1, 8).expand(1, 4, 130, 8)
                              .to(torch.int32), dtype=ttnn.uint32,
                              layout=ttnn.TILE_LAYOUT, device=dev),
              up((1, 4, 130, 8), const=0.5)),
     lambda t: ttnn.scatter(t[0], 3, t[1], t[2]))
case("ttnn.pad (p23: writer)",
     lambda: (up((1, 4, 130, 32), seed=1),),
     lambda t: ttnn.pad(t[0], [(0, 0), (0, 0), (0, 30), (0, 0)], 0.0))
case("ttnn.matmul (N axis padded)",
     lambda: (up((I, NH, NQ, HD), seed=2), up((I, NH, HD, NK), seed=3)),
     lambda t: ttnn.matmul(t[0], t[1]))
case("ttnn.matmul (M axis padded)",
     lambda: (up((I, NH, NK, HD), seed=4), up((I, NH, HD, HD), seed=5)),
     lambda t: ttnn.matmul(t[0], t[1]))
case("ttnn.linear (M axis padded)",
     lambda: (up((1, I, NK, C), seed=6), up((C, C), seed=7)),
     lambda t: ttnn.linear(t[0], t[1]))
case("ttnn.permute (swap last two)",
     lambda: (up((I, NH, NK, HD), seed=8),),
     lambda t: ttnn.permute(t[0], (0, 1, 3, 2)))
case("ttnn.permute (5D, GCA split order)",
     lambda: (up((1, I, NK, NH, HD), seed=9),),
     lambda t: ttnn.permute(t[0], (0, 1, 3, 2, 4)))
case("ttnn.reshape (4D->5D, re-tile)",
     lambda: (up((1, I, NK, C), seed=10),),
     lambda t: ttnn.reshape(t[0], (1, I, NK, NH, HD)))
case("ttnn.reshape (5D->4D, merge)",
     lambda: (up((1, I, NH, NK, HD), seed=11),),
     lambda t: ttnn.reshape(t[0], (I, NH, NK, HD)))
case("ttnn.rms_norm (last axis)",
     lambda: (up((1, I, NK, C), seed=12), up((C,), seed=13)),
     lambda t: ttnn.rms_norm(t[0], weight=t[1], epsilon=1e-6))
case("ttnn.multiply (scalar)",
     lambda: (up((I, NH, NQ, NK), seed=14),),
     lambda t: ttnn.multiply(t[0], 0.176))
case("ttnn.add (tensor)",
     lambda: (up((I, NH, NQ, NK), seed=15), up((I, NH, NQ, NK), seed=16)),
     lambda t: ttnn.add(t[0], t[1]))
case("ttnn.sigmoid",
     lambda: (up((I, NH, NQ, NK), seed=17),),
     lambda t: ttnn.sigmoid(t[0]))
case("ttnn.typecast (bf16->fp32)",
     lambda: (up((1, 16, 250, 250), seed=18),),
     lambda t: ttnn.typecast(t[0], ttnn.float32, memory_config=t[0].memory_config()))
case("ttnn.softmax (dim=-1)",
     lambda: (up((I, NH, NQ, NK), seed=19),),
     lambda t: ttnn.softmax(t[0], dim=-1))
case("ttnn.to_layout (row-major->tile)",
     lambda: (ttnn.from_torch(torch.randn(I, NH, NQ, NK,
                                          generator=torch.Generator().manual_seed(20)),
                              dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                              device=dev),),
     lambda t: ttnn.to_layout(t[0], ttnn.TILE_LAYOUT))
case("ttnn.from_torch (host upload)",
     lambda: (torch.randn(I, NH, NQ, NK, generator=torch.Generator().manual_seed(21)),),
     lambda t: ttnn.from_torch(t[0], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                               device=dev))
case("ttnn.concat (last axis)",
     lambda: (up((I, NH, NQ, 8), seed=22), up((I, NH, NQ, 6), seed=23)),
     lambda t: ttnn.concat([t[0], t[1]], dim=-1))
case("ttnn.concat (feature axis, padded M)",
     lambda: (up((1, NK, NK, 128), seed=24), up((1, NK, NK, 128), seed=25)),
     lambda t: ttnn.concat([t[0], t[1]], dim=-1))
case("ttnn.slice (last axis)",
     lambda: (up((I, NH, NQ, 64), seed=26),),
     lambda t: ttnn.slice(t[0], [0, 0, 0, 0], [I, NH, NQ, NK]))
case("ttnn.embedding (row-major gather)",
     lambda: (ttnn.from_torch(torch.arange(NK).view(1, NK).to(torch.int32),
                              dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev),
              ttnn.from_torch(torch.randn(64, C,
                                          generator=torch.Generator().manual_seed(27)),
                              dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                              device=dev)),
     lambda t: ttnn.embedding(t[0], t[1], layout=ttnn.TILE_LAYOUT))
case("ttnn.squeeze",
     lambda: (up((I, NH, 1, NK), seed=28),),
     lambda t: ttnn.squeeze(t[0], 2))
case("ttnn.transpose (-2,-1)",
     lambda: (up((I, NH, NK, HD), seed=29),),
     lambda t: ttnn.transpose(t[0], -2, -1))
case("ttnn.full",
     lambda: (),
     lambda t: ttnn.full((I, NH, NQ, NK), -1e4, dtype=ttnn.bfloat16,
                         layout=ttnn.TILE_LAYOUT, device=dev))


def pass_over():
    out = {}
    for name, build, run in CASES:
        ins = build()
        try:
            o = run(ins)
        except Exception as e:  # noqa: BLE001
            out[name] = ("ERR", "%s: %s" % (type(e).__name__, e))
            continue
        if tuple(o.padded_shape) == tuple(o.shape):
            out[name] = ("NOPAD", tuple(o.padded_shape))
        else:
            out[name] = ("PAD", pad_region(o))
        ttnn.deallocate(o)
        del o, ins
    return out


print("=== pass 1: clean heap ===", flush=True)
clean = pass_over()
print("primed %d garbage buffers" % prime(), flush=True)
print("=== pass 2: heap primed with +/-inf, +/-3.3e38 ===", flush=True)
prime()
dirty = pass_over()

print("\n%-38s %-20s %s" % ("op", "verdict", "pad region (clean pass)"), flush=True)
originators = []
for name, _, _ in CASES:
    kc, vc = clean[name]
    kd, vd = dirty[name]
    if kc == "ERR":
        print("%-38s %-20s %s" % (name, "ERROR", vc), flush=True)
        continue
    if kc == "NOPAD":
        print("%-38s %-20s output %s" % (name, "NO-TILE-PADDING", vc), flush=True)
        continue
    same = torch.equal(vc, vd)
    verdict = "WRITES-PAD" if same else "ORIGINATES-GARBAGE"
    if not same:
        originators.append(name)
    print("%-38s %-20s %s" % (name, verdict, describe(vc)), flush=True)
    if not same:
        print("%-38s %-20s %s" % ("", "  primed pass ->", describe(vd)), flush=True)
print("\nORIGINATORS: %s" % (", ".join(originators) or "(none)"), flush=True)
