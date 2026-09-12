"""What the interleaved fold A/B says, with the floor of each statistic it quotes.

Two estimators, and only one of them survives its own floor.

  fold_s   the whole fold wall. Its A/A pairs span 0.793 to 1.001 on a box that ran at loadavg 23
           to 53 during the session, so a 3 % arm difference in it is noise. REPORTED AND
           DISCARDED, because discarding it after looking is the only honest order.
  trunk_s  the trunk stage, bracketed by `synchronize_device` at the model's own first `diffusion`
           callback. Its A/A pairs span 0.927 to 1.013 with a median of 0.998, and the arms separate
           inside every single rep. This is the number.

The A/A floor is computed twice, because the ratio being quoted is a ratio of MEDIANS and a floor
built from single-fold pairs answers a different question (memory: quote the floor of the statistic
you are reporting). The matched floor is median-of-first-position-mesh over median-of-second-
position-mesh: same estimator, same session, same pairing, no lever.

The sign test is what makes the trunk result robust to the box. The two sharded arms are folded
adjacent in time inside one rep, with their order reversed on alternate reps, so a load excursion
hits both. If one arm is slower in all seven reps that is p = 2^-7 one-sided, and no amount of
drift produces it.

    python3 perf/b2z2_bshardtime/analyze_fold_ab.py
"""

import json
import pathlib
import statistics as st

HERE = pathlib.Path(__file__).resolve().parent
d = json.loads((HERE / "fold_ab_n2.json").read_text())
warm = d["folds"][3:]                       # the three cold folds are discarded, digests kept
REPS = d["reps"]
assert len(warm) == 4 * REPS, f"{len(warm)} warm folds, expected {4 * REPS}"

# Each rep is mesh, X, Y, mesh with X/Y the two sharded arms in alternating order.
reps = [warm[i * 4:(i + 1) * 4] for i in range(REPS)]
for r in reps:
    assert r[0]["arm"] == "mesh" and r[3]["arm"] == "mesh", "rep layout is not mesh,X,Y,mesh"


def by(arm, key, fs=warm):
    return [f[key] for f in fs if f["arm"] == arm]


print(f"{REPS} reps, 1x2 mesh on whglx cards {d['cards']}, loadavg "
      f"{d['summary']['loadavg_range'][0]:.1f}-{d['summary']['loadavg_range'][1]:.1f}, "
      f"benchlocked={d['benchlocked']}\n")

out = {"reps": REPS, "cards": d["cards"], "loadavg_range": d["summary"]["loadavg_range"],
       "benchlocked": d["benchlocked"], "by_statistic": {}}

for key in ("trunk_s", "fold_s"):
    med = {a: st.median(by(a, key)) for a in ("mesh", "shard", "bshard")}
    # matched floor: the same ratio-of-medians estimator, mesh against mesh
    aa_matched = st.median([r[3][key] for r in reps]) / st.median([r[0][key] for r in reps])
    aa_pairs = [r[3][key] / r[0][key] for r in reps]
    # per-rep paired sign test, bshard against shard
    per_rep = [(next(f[key] for f in r if f["arm"] == "shard"),
                next(f[key] for f in r if f["arm"] == "bshard")) for r in reps]
    wins = sum(1 for s, b in per_rep if b < s)
    print(f"=== {key} ===")
    print(f"  mesh {med['mesh']:7.3f}   shard {med['shard']:7.3f}   bshard {med['bshard']:7.3f}  "
          "(median of 7)")
    print(f"  shard  vs mesh   {med['mesh']/med['shard']:.4f}x")
    print(f"  bshard vs mesh   {med['mesh']/med['bshard']:.4f}x")
    print(f"  bshard vs shard  {med['shard']/med['bshard']:.4f}x   "
          f"({'bshard FASTER' if med['bshard'] < med['shard'] else 'bshard SLOWER'} by "
          f"{abs(med['bshard']-med['shard']):.3f} s)")
    print(f"  A/A floor, matched estimator (median vs median) {aa_matched:.5f}x")
    print(f"  A/A floor, single-fold pairs  {min(aa_pairs):.5f}x - {max(aa_pairs):.5f}x  "
          f"median {st.median(aa_pairs):.5f}x")
    print(f"  paired sign test, bshard faster than shard in {wins} of {REPS} reps  "
          f"(one-sided p = {2**-REPS if wins in (0, REPS) else float('nan'):.4f} when it is 0 or {REPS})")
    for i, (s, b) in enumerate(per_rep, 1):
        print(f"    rep {i}  shard {s:7.3f}  bshard {b:7.3f}  {'bshard' if b < s else 'shard '} "
              f"faster by {abs(b-s):6.3f}")
    print()
    out["by_statistic"][key] = {
        "median": med, "shard_vs_mesh": med["mesh"] / med["shard"],
        "bshard_vs_mesh": med["mesh"] / med["bshard"],
        "bshard_vs_shard": med["shard"] / med["bshard"],
        "aa_floor_matched": aa_matched, "aa_pairs": aa_pairs,
        "aa_span": [min(aa_pairs), max(aa_pairs)],
        "per_rep_shard_bshard": per_rep, "bshard_wins": wins,
    }

t = out["by_statistic"]["trunk_s"]
blocks = d["row_shard_calls_total"]["chain"] // (REPS * 2 + 3 - 1) if REPS else 0
per_block_ms = (t["median"]["bshard"] - t["median"]["shard"]) / 264 * 1e3
print(f"the trunk runs 264 sharded blocks per fold, so the {t['median']['bshard']-t['median']['shard']:.3f} s "
      f"penalty is {per_block_ms:.3f} ms per block")
print(f"the ISOLATED block measured b_shard SAVING 1.711 ms per block at the same width "
      f"(63.908 -> 62.197). The fold disagrees with the block by "
      f"{per_block_ms + 1.711:.3f} ms per block.")
print(f"b_shard adds 2 collectives per block; 528 of them per fold. At the measured 1.234 ms each "
      f"they are 0.651 s of the 1.121 s if NONE of them overlaps.")
out["per_block_ms_penalty"] = per_block_ms
out["isolated_block_saving_ms"] = 1.711
out["fold_minus_block_ms_per_block"] = per_block_ms + 1.711
out["digest"] = d["summary"]["cif_sha256_16"]
out["bit_identical"] = d["summary"]["bit_identical_across_all_arms_and_reps"]
out["counters"] = d["row_shard_calls_total"]
(HERE / "fold_ab_n2_analysis.json").write_text(json.dumps(out, indent=1))
print(f"\nwrote {HERE / 'fold_ab_n2_analysis.json'}")
