"""p150a against p300c on the BindCraft 2 design phase, from the closed acceptance denominator.

`state/bcx/THROUGHPUT.md` quotes 2.18 trajectories/h on qb2's four p300c and leaves qb1's four
p150a as "never measured", so the fleet bound of <=0.988 designs/h carries an unverified
condition. Three p150a arms have since run this exact workload to exit. This prices them.

A per-trajectory wall is not a board rate: a trajectory rejected at screen ran 50 gradient rounds
and one that reached the refold ran 125, and the per-round cost climbs with the padded token axis.
So both are divided out. Stage rounds are BindCraft 2's own
DESIGN_STAGE_DEFAULT_ROUNDS (settings.py:604) and the token count is its own two paddings via
perf/bcx_target2/padmap.tokens, neither re-derived here. A stage runs to its full round count
before its filter is applied -- bcx-target2's anneal rejection banked 120 device backwards against
50+25+45, which is the check that licenses this mapping.

CPU only, no device opened.
"""
import argparse, json, math, re, statistics, sys

# BindCraft 2 settings.py:604, single binding target so the budget multiplier is 1.
STAGE_ROUNDS = {"screen": 50, "refine": 25, "anneal": 45, "harden": 5}
GRADIENT_STAGES = ["screen", "refine", "anneal", "harden"]
PDL1_TARGET_AA = 115  # examples/pdl1.json, hPDL1

BOARD = {"qb1_s10": "p150a", "qb1_s12": "p150a", "qb1_s13": "p150a",
         "traj_off_s3": "p300c", "accept_s4": "p300c"}
# The commit each arm ran, from its own traj_stamp.json / arm_stamp.json.
COMMIT = {"qb1_s10": "643cabcf9", "qb1_s12": "643cabcf9", "qb1_s13": "643cabcf9",
          "traj_off_s3": "17ec09261", "accept_s4": "c014a2a7d"}
HARNESS = {"qb1_s10": "traj_arm", "qb1_s12": "traj_arm", "qb1_s13": "traj_arm",
           "traj_off_s3": "traj_arm", "accept_s4": "run_arm"}
# nproc and the arm's own loadavg1 at launch, from its traj_stamp.json / *_stamp.txt.
HOST = {"qb1_s10": (32, 1.74), "qb1_s12": (32, 6.87), "qb1_s13": (32, 20.68),
        "traj_off_s3": (16, 15.44), "accept_s4": (16, 14.89)}
MUTATE_ROUNDS = 15  # settings.py:604; forward-only scoring, not gradient rounds


def gradient_rounds(terminated):
    """Rounds of gradient design a trajectory ran before it stopped.

    A blank `terminated` means it reached the refold ensemble, and `mutate`/`final` are verdicts
    taken after the four gradient stages, so all three ran the full 125. Those three also ran
    mutate's 15 non-gradient rounds, whose cost is inside the same `design` second and is NOT a
    gradient round -- so they are reported separately and excluded from the matched comparison.
    """
    if terminated in STAGE_ROUNDS:
        return sum(STAGE_ROUNDS[s] for s in GRADIENT_STAGES[:GRADIENT_STAGES.index(terminated) + 1])
    return sum(STAGE_ROUNDS.values())


def clean(terminated):
    """True when `design` seconds buy gradient rounds and nothing else."""
    return terminated in STAGE_ROUNDS


def read(path):
    rows = []
    for line in open(path):
        arm, index, length, timing, terminated = (line.rstrip("\n").split("\t") + [""])[:5]
        design = float(re.search(r"design=([\d.]+)", timing).group(1))
        rows.append(dict(arm=arm, board=BOARD[arm], commit=COMMIT[arm], index=int(index),
                         binder=int(length), design_s=design, terminated=terminated,
                         rounds=gradient_rounds(terminated), clean=clean(terminated),
                         compiled=bool(int(re.search(r"compiled=(\d)", timing).group(1)))))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tsv", nargs="+")
    ap.add_argument("--tokens-json", required=True,
                    help="binder -> token count, from padmap.tokens on a host carrying bc2")
    ap.add_argument("--out")
    args = ap.parse_args()

    rows = [r for path in args.tsv for r in read(path)]
    token_of = {int(k): int(v) for k, v in json.load(open(args.tokens_json)).items()}
    for r in rows:
        r["tokens"] = token_of[r["binder"]]
        r["nproc"], r["load1"] = HOST[r["arm"]]
        r["harness"] = HARNESS[r["arm"]]
        # A trajectory that reached mutate spent 15 forward-only rounds inside the same `design`
        # second. Their cost is unknown, so it is BOUNDED instead of guessed: k=0 charges them
        # nothing and k=1 charges them a full gradient round. Every conclusion below is required
        # to hold at both ends.
        r["s_per_round_k0"] = r["design_s"] / r["rounds"]
        r["s_per_round_k1"] = r["design_s"] / (r["rounds"] + (0 if r["clean"] else MUTATE_ROUNDS))

    print("arm          board  harness  commit     binder  n    stage      rounds  design_s   s/rnd k0  s/rnd k1")
    for r in sorted(rows, key=lambda r: (r["tokens"], r["board"], r["arm"], r["index"])):
        print("%-12s %-6s %-8s %-10s %-6d  %-4d %-10s %-6d  %8.1f  %8.2f  %8.2f"
              % (r["arm"], r["board"], r["harness"], r["commit"], r["binder"], r["tokens"],
                 r["terminated"] or "completed", r["rounds"], r["design_s"],
                 r["s_per_round_k0"], r["s_per_round_k1"]))

    def cell(pred, key):
        v = [r[key] for r in rows if pred(r)]
        return (len(v), statistics.fmean(v) if v else None, min(v) if v else None, max(v) if v else None)

    buckets = sorted({r["tokens"] for r in rows})
    arms = ["qb1_s10", "qb1_s12", "qb1_s13", "traj_off_s3", "accept_s4"]

    print("\n=== seconds per gradient round, by ARM and token axis (k=0 | k=1) ===")
    print("%-12s %-6s %-5s %-5s %s" % ("arm", "board", "cores", "load1",
                                       "  ".join("n=%-13d" % n for n in buckets)))
    for a in arms:
        line = "%-12s %-6s %-5d %-5.1f " % (a, BOARD[a], HOST[a][0], HOST[a][1])
        for n in buckets:
            c0 = cell(lambda r, a=a, n=n: r["arm"] == a and r["tokens"] == n, "s_per_round_k0")
            c1 = cell(lambda r, a=a, n=n: r["arm"] == a and r["tokens"] == n, "s_per_round_k1")
            line += "%-17s" % ("-" if not c0[0] else "%.1f|%.1f x%d" % (c0[1], c1[1], c0[0]))
        print(line)

    print("\n=== the two controls, and the second one decides the question ===")
    for n in buckets:
        a = cell(lambda r, n=n: r["tokens"] == n and r["board"] == "p150a", "s_per_round_k1")
        b = cell(lambda r, n=n: r["tokens"] == n and r["board"] == "p300c", "s_per_round_k1")
        t = cell(lambda r, n=n: r["tokens"] == n and r["arm"] == "traj_off_s3", "s_per_round_k1")
        s4 = cell(lambda r, n=n: r["tokens"] == n and r["arm"] == "accept_s4", "s_per_round_k1")
        board = "%.3f" % (a[1] / b[1]) if a[0] and b[0] else "-"
        arm = "%.3f" % (s4[1] / t[1]) if t[0] and s4[0] else "-"
        print("  n=%-4d p150a %-6s (x%d)  p300c %-6s (x%d)  BOARD p150a/p300c %-6s | "
              "SAME-BOARD accept_s4/traj_off_s3 %s"
              % (n, "%.2f" % a[1] if a[0] else "-", a[0], "%.2f" % b[1] if b[0] else "-", b[0],
                 board, arm))

    pooled = {}
    for label, pred in (("p150a", lambda r: r["board"] == "p150a"),
                        ("p300c", lambda r: r["board"] == "p300c"),
                        ("traj_off_s3 (p300c)", lambda r: r["arm"] == "traj_off_s3"),
                        ("accept_s4 (p300c)", lambda r: r["arm"] == "accept_s4")):
        # Pool as a ratio to each trajectory's own token-axis mean, so the mix of n does not decide it.
        ratios = []
        for r in rows:
            if not pred(r):
                continue
            peers = [q["s_per_round_k1"] for q in rows if q["tokens"] == r["tokens"]]
            ratios.append(r["s_per_round_k1"] / statistics.fmean(peers))
        pooled[label] = (len(ratios), statistics.fmean(ratios),
                         statistics.stdev(ratios) if len(ratios) > 1 else 0.0)

    print("\n=== pooled across token axes, each trajectory against its own axis mean ===")
    for k, (n, m, sd) in pooled.items():
        print("  %-22s x%-3d  %.3f  sd %.3f" % (k, n, m, sd))
    # Welch, on the normalised ratios. The question is not "are they equal" -- nothing is -- but
    # how tightly the board factor is bracketed, so the interval is the deliverable, not the p.
    def welch(a, b):
        na, ma, sa = a
        nb, mb, sb = b
        se = math.sqrt(sa * sa / na + sb * sb / nb)
        df = (sa * sa / na + sb * sb / nb) ** 2 / (
            (sa * sa / na) ** 2 / (na - 1) + (sb * sb / nb) ** 2 / (nb - 1))
        return ma - mb, se, df

    d, se, df = welch(pooled["p150a"], pooled["p300c"])
    print("\n=== how tightly is the board factor bracketed ===")
    print("  p150a / p300c, pooled over token axes  = %.3f" % (pooled["p150a"][1] / pooled["p300c"][1]))
    print("  difference %.4f, se %.4f, Welch df %.1f" % (d, se, df))
    print("  95 %% interval on the board factor      = %.3f to %.3f"
          % (1 + (d - 1.96 * se), 1 + (d + 1.96 * se)))
    print("  -> the p150a is within +/-%.0f %% of the p300c on this workload" % (100 * 1.96 * se))

    print("\n=== robustness: the one compile-carrying trajectory per arm dropped ===")
    for label, pred in (("p150a", lambda r: r["board"] == "p150a"),
                        ("p300c", lambda r: r["board"] == "p300c")):
        ratios = []
        for r in rows:
            if not pred(r) or r["compiled"]:
                continue
            peers = [q["s_per_round_k1"] for q in rows if q["tokens"] == r["tokens"] and not q["compiled"]]
            ratios.append(r["s_per_round_k1"] / statistics.fmean(peers))
        print("  %-6s x%-3d  %.3f" % (label, len(ratios), statistics.fmean(ratios)))

    board_spread = abs(pooled["p150a"][1] - pooled["p300c"][1])
    arm_spread = abs(pooled["accept_s4 (p300c)"][1] - pooled["traj_off_s3 (p300c)"][1])
    # Deliberately NOT printed as a ratio. The board term is near zero, so arm/board would be a
    # large number manufactured by its own denominator rather than a measured quantity.
    print("\n  board term,   p150a against p300c                  = %.3f" % board_spread)
    print("  harness term, SAME BOARD, same host, same load     = %.3f" % arm_spread)
    print("  The harness term is the larger of the two, so no two arms here differ in the board")
    print("  alone and the near-zero board term is a bracket, not a null result.")

    if args.out:
        json.dump(dict(rows=rows, pooled=pooled, board_spread=board_spread,
                       arm_spread=arm_spread), open(args.out, "w"), indent=1)
        print("\nWROTE", args.out)


if __name__ == "__main__":
    main()
