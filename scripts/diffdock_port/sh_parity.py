"""Grade tt-bio's spherical harmonics against e3nn, in the reference environment.

    ~/scratch/dfd/venv-ref/bin/python scripts/diffdock_port/sh_parity.py

Measured 2026-11-26 over 4096 random vectors, e3nn 0.5.1: max absolute difference 2.4e-07 on
values up to 2.24, for both the score network's `1x0e+1x1o+1x2e` and the torsion head's `2e`,
and the zero vector gives e3nn's `[1, 0, ...]` on both sides rather than a NaN.
"""
import sys
from pathlib import Path

import torch
from e3nn import o3

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tt_bio.diffdock.harmonics import spherical_harmonics  # noqa: E402


def main():
    torch.manual_seed(0)
    vec = torch.randn(4096, 3) * 7
    worst = 0.0
    for ls, irreps in (((0, 1, 2), "1x0e+1x1o+1x2e"), ((2,), "2e")):
        ref = o3.spherical_harmonics(irreps, vec, normalize=True, normalization="component")
        diff = (ref - spherical_harmonics(vec, ls)).abs().max().item()
        worst = max(worst, diff)
        print(f"{irreps}: max abs diff {diff:.3e}, max |ref| {ref.abs().max():.4f}")
    zero = torch.zeros(1, 3)
    ref0 = o3.spherical_harmonics("1x0e+1x1o+1x2e", zero, normalize=True,
                                  normalization="component")
    assert torch.equal(ref0, spherical_harmonics(zero, (0, 1, 2))), "zero vector disagrees"
    assert worst < 1e-6, worst
    print("ok")


if __name__ == "__main__":
    main()
