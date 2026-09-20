#!/usr/bin/env python3
"""The campaign's prize, re-priced on `tmk-assumptions`' measurements. CPU only, no device.

Supersedes the roof half of ceiling.py: that priced route A against a 123.65 TFLOP/s square-cube
roof, and tmk-assumptions refuted the roof (wrong chip, and the wrong KIND of roof for shapes that
sit below machine balance). It does not supersede the fold-share arithmetic, which stands.

One correction is applied to tmk-assumptions itself and it is flagged, not silently folded in.
"""

FOLD_S, MODULE_IN_FOLD_S, MODULE_MS = 48.62, 12.6881, 11.519


def fold(module_ms_after):
    after = MODULE_IN_FOLD_S * (module_ms_after / MODULE_MS)
    return FOLD_S / (FOLD_S - MODULE_IN_FOLD_S + after)


def line(label, after):
    return (f"{label:<52} module {MODULE_MS:6.3f} -> {after:6.3f} ms "
            f"({MODULE_MS / after:5.4f}x)   fold {fold(after):6.4f}x")


print("ROUTE A -- RATE. Re-priced by tmk-assumptions, qb2 p300c node 3, pinned 1350 MHz.")
print("  The three sites sit at 170.6 / 170.7 / 127.9 FLOP/byte against a machine balance of")
print("  263 FLOP/byte, so all three are BELOW balance: none can reach the compute roof at any")
print("  kernel quality, and each one's binding roof is traffic at 73.53 / 73.58 / 50.10-55.15.")
A_BEST, A_PLAUSIBLE = 1.774, 0.906     # ms recoverable, best-BW and same-mix
print(line("  A at its best-BW traffic roof (49 % of the deficit)", MODULE_MS - A_BEST))
print(line("  A at its same-mix traffic roof (25 %)", MODULE_MS - A_PLAUSIBLE))
print("  The campaign booked 3.491 ms here. At most 1.774 ms of it exists.\n")

print("ROUTE B -- THE AXIS EXCHANGE. Resurrected, but its headroom depends on the baseline.")
TILE_GRANULAR_GBS = 364.7   # measured, tmk-assumptions
BASELINES = {
    "ttnn.permute, what tmk-assumptions timed and labelled 'shipped'": (63.3, 2.121),
    "trix-radical's figure for the shipped path": (158.0, None),
    "reblock_permute/_back, OUR kernel, perfwar-trimul-kernel at N=1024": (221.0, None),
}
print("  tile-granular control: 364.7 GB/s, within 3.4 % of a straight clone of the same bytes.")
print("  Three different numbers are in circulation for 'the shipped move', and they disagree:")
for label, (gbs, ms) in BASELINES.items():
    print(f"    {gbs:6.1f} GB/s  headroom {TILE_GRANULAR_GBS / gbs:5.2f}x   {label}")
print()
print("  Production at 512 aa calls reblock_permute_gated (tenstorrent.py:6415, 6492-6495,")
print("  behind eligible_gated), NOT ttnn.permute. perfwar measured our own kernel at 221/210")
print("  GB/s against ttnn.permute's 55 -- and 63.3 is in ttnn.permute territory, not ours.")
print("  So the 5.76x is against an op production does not call. Pricing the term both ways,")
print("  over the 2.551 ms trix-radical booked and the 2.121 ms tmk-assumptions measured:")
for base_ms in (2.551, 2.121):
    for tag, factor in (("vs ttnn.permute  (5.76x)", 364.7 / 63.3),
                        ("vs our kernel    (1.65x)", 364.7 / 221.0)):
        rec = base_ms * (1 - 1 / factor)
        print(line(f"  B from {base_ms:.3f} ms {tag}", MODULE_MS - rec))
print()

print("STACKED, and this is the campaign's answer")
for atag, a in (("A best", A_BEST), ("A plausible", A_PLAUSIBLE)):
    for btag, b in (("B vs ttnn.permute", 2.121 * (1 - 63.3 / 364.7)),
                    ("B vs our kernel", 2.121 * (1 - 221.0 / 364.7))):
        print(line(f"  {atag} + {btag}", MODULE_MS - a - b))
print()
print("  The 1.05x worth-building bar for a NEW build, and the 1.3531x hard ceiling from")
print("  trimul being 26.10 % of the fold, both still stand.")
