"""Float32 emulation of the SFPU silu candidates against a float64 reference.

Each SFPU MAD is emulated as one float32 rounding of a*b+c computed in float64 (the SFPU MAD rounds once);
the integer steps are done on the float32 bit patterns exactly as the kernel does them.
"""
import argparse
import numpy as np

F = np.float32


def mad(a, b, c):
    return (a.astype(np.float64) * np.float64(b) + np.float64(c)).astype(F) if np.ndim(b) == 0 else \
        (a.astype(np.float64) * b.astype(np.float64) + np.asarray(c, np.float64)).astype(F)


def silu_f32(x, newton=2):
    """The candidate in silu_f32_cand.h, step for step."""
    x = x.astype(F)
    z = mad(np.abs(x), F(-1.4426950408889634), 0.0)
    z = np.maximum(z, F(-126.0))
    c231 = F(12582912.0)
    t = mad(z, 1.0, c231)
    r = (z.astype(np.float64) - (t - c231).astype(np.float64)).astype(F)
    k = ((t.view(np.int32) - c231.view(np.int32)) << 23).astype(np.int32)
    p = mad(r, F(0.00958251953), F(0.0559082031))
    p = mad(p * 1, r, F(0.240241066)) if False else (p.astype(np.float64) * r + np.float64(F(0.240241066))).astype(F)
    p = (p.astype(np.float64) * r + np.float64(F(0.693123937))).astype(F)
    p = (p.astype(np.float64) * r + 1.0).astype(F)
    e = (p.view(np.int32) + k).view(F)
    y = (np.float64(F(0.322265625)) * e + np.float64(F(-0.80859375))).astype(F)
    y = (y.astype(np.float64) * e + np.float64(F(0.98828125))).astype(F)
    n = (-e.astype(np.float64) - 1.0).astype(F)
    for _ in range(newton):
        u = (n.astype(np.float64) * y + 1.0).astype(F)
        y = (y.astype(np.float64) * u + y).astype(F)
    y = np.where(x < 0, (y.astype(np.float64) * e).astype(F), y)
    return (x.astype(np.float64) * y).astype(F)


def silu64(x):
    x = x.astype(np.float64)
    return x / (1.0 + np.exp(-x))


def bf16(a):
    b = a.astype(F).view(np.uint32)
    b = (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000
    return b.view(F)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--newton", type=int, default=2)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    xs = {
        "grid[-90,90]": np.linspace(-90, 90, 2_000_001).astype(F),
        "normal(0,3)": rng.normal(0, 3, 2_000_000).astype(F),
        "tiny": (rng.normal(0, 1e-3, 200_000)).astype(F),
    }
    for name, x in xs.items():
        ref = silu64(x)
        got = silu_f32(x, a.newton).astype(np.float64)
        ok = np.abs(ref) > 1e-30
        rel = np.abs(got[ok] / ref[ok] - 1)
        flips = np.mean(bf16(got.astype(F)) != bf16(ref.astype(F)))
        print(f"{name:14s} max rel {rel.max():.2e}  p99.9 {np.quantile(rel, 0.999):.2e}  "
              f"bf16 outputs differing from bf16(exact) {flips:.2e}  finite {np.isfinite(got).all()}")


if __name__ == "__main__":
    main()
