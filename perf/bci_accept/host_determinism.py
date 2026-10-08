"""Compare two BindCraft 2 runs of the SAME design hash, round by round.

The question this answers is whether host JAX reproduces itself. A design hash pins the design
recipe, so two folders carrying the same hash started from the same input; if their per-round
losses diverge, the backend is not reproducible and a per-trajectory card-vs-host comparison
cannot mean anything.

Usage:
    python3 -I host_determinism.py <a_losses.csv> <b_losses.csv>

Prints the first diverging round and the growth of the largest relative difference across all
loss columns. Round 1 identical with round 2 differing is the signature of a run-to-run numeric
difference entering at the first gradient update rather than at initialisation.
"""
import csv
import sys


def read(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def main(path_a, path_b):
    a, b = read(path_a), read(path_b)
    if not a or not b:
        sys.exit("one of the loss tables is empty")

    cols = [c for c in a[0] if c not in ("phase", "round")]
    n = min(len(a), len(b))
    if len(a) != len(b):
        print("note: row counts differ (%d vs %d), comparing the first %d" % (len(a), len(b), n))

    first = None
    rows = []
    for i in range(n):
        worst = 0.0
        for c in cols:
            try:
                x, y = float(a[i][c]), float(b[i][c])
            except (ValueError, KeyError):
                continue
            worst = max(worst, abs(x - y) / max(abs(x), abs(y), 1e-9))
        if worst > 0 and first is None:
            first = i
        rows.append((i, a[i].get("phase"), a[i].get("round"), a[i].get("hPDL1.iptm"),
                     b[i].get("hPDL1.iptm"), worst))

    print("%-5s %-8s %-6s %-9s %-9s %s" % ("row", "phase", "round", "iptm_A", "iptm_B", "max rel diff"))
    for i, phase, rnd, ia, ib, worst in rows:
        if i < 4 or i % 10 == 0 or i == n - 1:
            print("%-5d %-8s %-6s %-9s %-9s %.4f" % (i, phase, rnd, ia, ib, worst))

    print()
    if first is None:
        print("VERDICT: the two runs are identical in every loss column. Reproducible.")
    else:
        print("VERDICT: identical through row %d, first difference at row %d (phase %s round %s)."
              % (first - 1, first, rows[first][1], rows[first][2]))
        print("         final row: i_pTM %s vs %s, max rel diff %.4f"
              % (rows[-1][3], rows[-1][4], rows[-1][5]))
        print("         Same hash, same settings: host JAX does not reproduce itself.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
