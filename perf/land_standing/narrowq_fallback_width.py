#!/usr/bin/env python3
"""What width does the narrow-q fallback actually pick, at every length users can submit?

WHY THIS IS THE QUESTION THAT DECIDES THE DEFAULT. The lever adds the dividing q_chunks BELOW the
production pick to the candidate list, widest-first, and the caller takes the widest that fits L1.
Every narrower candidate uses LESS L1 than the production pick, which already fits, so the one
that gets picked is simply the largest divisor of the padded length below `prod`. The kernel
re-reads all of K and V once per q chunk, so the cost of the fallback is set by how far below
`prod` that divisor sits -- and that is a property of the ARITHMETIC of the padded length, not of
the hardware.

At 896 the pick is 224 against a prod of 256, a 1.14x re-read increase, and the fused path it
buys back is worth +9.50 s. But a padded length of the form 32*p for a large prime p has no
divisor between 32 and itself, so the fallback would pick 32 and re-read K and V p times. That is
the case that could turn this lever into a regression at a size nobody measured, and it is the
`one-size-tuning-is-a-standing-defect-class` exposure in concrete form.

This enumerates it instead of arguing about it. No device needed: the policy is pure arithmetic.
"""
import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("tt", ROOT / "tt_bio" / "tenstorrent.py")

TILE = 32


def divisors_mult_of_tile(padded):
    return [padded // n for n in range(1, padded // TILE + 1)
            if padded % n == 0 and (padded // n) % TILE == 0]


def main():
    # the shipped production pick for tri-attention is a fixed 256 (see _sdpa_chunks_shipped and
    # the 896 case recorded in _tri_att_q_chunks: the ladder there is (896, 448, 256)).
    PROD = 256
    print("padded length -> widest dividing q_chunk BELOW the production pick of %d\n" % PROD)
    print("  %6s %8s %10s %10s  %s" % ("padded", "divides?", "fallback", "re-read", "note"))
    risky = []
    for padded in range(256, 1537, TILE):
        if padded % PROD == 0:
            print("  %6d %8s %10s %10s  policy inert: prod divides it, identical tuple either way"
                  % (padded, "yes", "-", "-"))
            continue
        d = divisors_mult_of_tile(padded)
        narrower = sorted((q for q in d if q < PROD), reverse=True)
        pick = narrower[0] if narrower else None
        if pick is None:
            print("  %6d %8s %10s %10s  no dividing chunk below prod: lever offers nothing"
                  % (padded, "no", "none", "-"))
            continue
        reread = padded / pick
        base = padded / PROD
        ratio = reread / base
        note = ""
        if ratio > 2.0:
            note = "RISK: re-reads K/V %.1fx more than the shipped pick would" % ratio
            risky.append((padded, pick, ratio))
        print("  %6d %8s %10d %9.1fx  %s" % (padded, "no", pick, ratio, note))

    print("\n%d length(s) where the fallback re-reads K and V more than 2x the shipped pick:"
          % len(risky))
    for padded, pick, ratio in risky:
        print("  padded %d -> q_chunk %d, %.1fx the re-reads (%d = %s)"
              % (padded, pick, ratio, padded, factor(padded)))
    if not risky:
        print("  none in 256..1536.")
    return 0


def factor(n):
    out, d = [], 2
    while d * d <= n:
        while n % d == 0:
            out.append(d); n //= d
        d += 1
    if n > 1:
        out.append(n)
    return "*".join(map(str, out))


if __name__ == "__main__":
    sys.exit(main())
