#!/usr/bin/env python3
"""The device column of the triangle multiplication, per arm, out of an arm'd census run."""
import collections
import json
import sys

ROOF = 442.3e9
FAM = ("tri_mul_out", "tri_mul_in")


def rows_for(blob, arm, block="1"):
    reps, lam = blob["reps"], blob["sync_floor_s"]["median"]
    S, F = blob["by_arm"][arm]["sync"], blob["by_arm"][arm]["free"]
    rows = collections.defaultdict(lambda: [0.0, 0.0, 0, 0.0, 0.0])
    for src, idx in ((S, 0), (F, 1)):
        for k, v in src["wall"].items():
            tag, verb, sig = k.split("||")
            st, blk, dr, fam = tag.split("|")
            if fam not in FAM or st != "evo" or blk != block:
                continue
            r = rows[(dr, verb, sig)]
            r[idx] += v
            if idx == 0:
                r[2] += src["calls"][k]
                r[3] += src["read"][k]
                r[4] += src["written"][k]
    dev = {k: max(v[0] - v[1] - lam * v[2], 0.0) / reps for k, v in rows.items()}
    return rows, dev, reps, lam


def main():
    blob = json.load(open(sys.argv[1]))
    arms = blob["arms"]
    print(f'n={blob["n"]} reps={blob["reps"]} arms={arms} '
          f'aiclk={sorted({a["median"] for a in blob["aiclk"]})} '
          f'served={blob["served"]}')
    tot = {}
    for arm in arms:
        rows, dev, reps, lam = rows_for(blob, arm)
        tot[arm] = {d: sum(v for k, v in dev.items() if k[0] == d) for d in ("fwd", "bwd")}
        tot[arm]["rows"] = rows
        tot[arm]["dev"] = dev
    print()
    print(f'{"per block, device ms":<34}' + "".join(f'{a:>12}' for a in arms) + f'{"x":>10}')
    for d in ("fwd", "bwd"):
        a0, a1 = tot[arms[0]][d] * 1e3, tot[arms[-1]][d] * 1e3
        print(f'  {d:<32}{a0:12.3f}{a1:12.3f}{(a0 / a1 if a1 else 0):10.3f}')
    r0 = sum(tot[arms[0]][d] * (2 if d == "fwd" else 1) for d in ("fwd", "bwd")) * 52
    r1 = sum(tot[arms[-1]][d] * (2 if d == "fwd" else 1) for d in ("fwd", "bwd")) * 52
    print(f'  {"round, 52 blocks (2 fwd + 1 bwd), s":<32}{r0:12.3f}{r1:12.3f}'
          f'{(r0 / r1 if r1 else 0):10.3f}   delta {r0 - r1:+.3f} s')

    print("\nthe verbs that moved (backward, device ms/block):")
    keys = set(tot[arms[0]]["dev"]) | set(tot[arms[-1]]["dev"])
    out = []
    for k in keys:
        if k[0] != "bwd":
            continue
        a = tot[arms[0]]["dev"].get(k, 0.0) * 1e3
        b = tot[arms[-1]]["dev"].get(k, 0.0) * 1e3
        if abs(a - b) > 0.15 or (a == 0) != (b == 0):
            out.append((abs(a - b), a, b, k))
    for _, a, b, k in sorted(out, reverse=True)[:14]:
        sig = k[2].split(" @ ")[0]
        print(f'  {a:8.3f} -> {b:8.3f}  {k[1]:28s} {sig}')


if __name__ == "__main__":
    main()
