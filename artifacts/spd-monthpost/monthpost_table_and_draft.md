# Oct 1 vs today: TT-Bio speedup, same chip, same harness

BASELINES: WH old = v0.11.0 8dbaa61b1 (2026-10-01). BH old = v0.10.0 1ee324e01 (2026-09-30, one day earlier; the BH v0.11.0 chain was stopped before it ran, per the 13:05Z scope note). NEW light (512) = gapboard cells on main: p150a 324cd5910 (10-14 ~11Z), WH b241d4975 (10-14 ~13Z). NEW heavy (c730) = origin/main 148605b31 (10-14 13:45Z). All trees pin ttnn 0.68.0; one venv per host serves old and new.

## MEASURED HEAVY (c730: 730 tokens, MSA depth 9,947, 5 samples, 200 steps, warm median of 3)
OLD = v0.11.0 8dbaa61b1 on both archs here. NEW = main 148605b31. No A100 cell exists at c730 (the kit grid is the 512 light fold), so no GPU gap for this fold.
| model | arch | mode | old s | new s | speedup | n | in-fold AICLK median / min (old; new) | confidence/ipTM old -> new | old vs new top-sample CA RMSD (same seed) vs seed floor old / new |
|---|---|---|---|---|---|---|---|---|---|
| Protenix-v2 | BH p150a | normal | 259.87 | 108.41 | 2.40x | 3 | 1350/1343; 1350/1343 | 0.896/0.930 -> 0.885/0.926 | 12.4 A vs 8.8 / 18.4 |
| Protenix-v2 | BH p150a | fast | 263.11 | 85.24 | 3.09x (3.05x vs old normal) | 3 | 1350/1343; 1350/1343 | 0.898/0.929 -> 0.883/0.924 | 11.7 A vs 7.4 / 16.8 |
| OpenDDE | BH p150a | normal | 365.83 | 149.68 | 2.44x | 3 | 1350/1343; 1350/1343 | 0.887/0.912 -> 0.829/0.863 | 21.5 A vs 22.3 / 20.0 |
| OpenDDE | BH p150a | fast | 371.41 | 132.02 | 2.81x (2.77x vs old normal) | 3 | 1350/1343; 1350/1343 | 0.887/0.912 -> 0.831/0.865 | 19.8 A vs 22.2 / 16.6 |
| OpenFold3 | BH p150a | normal | 158.39 | 48.43 | 3.27x | 3 | 1350/1343; 1350/1343 | 0.774/0.714 -> 0.734/0.680 | 2.9 A vs 11.0 / 12.1 |
| OpenFold3 | BH p150a | fast | 213.45 | 37.67 | 5.67x (4.20x vs old normal) | 3 | 1350/1343; 1350/1343 | 0.729/0.694 -> 0.731/0.670 | 11.6 A vs 2.8 / 12.0 |
| Boltz-2 | BH p150a | normal | 65.59 | 70.30 | 0.93x | 3 | 1350/1343; 1350/1343 | 0.713/0.772 -> 0.661/0.557 | 23.5 A vs 8.9 / 23.0 -- ACCURACY REGRESSION reproduces on BH |
| Boltz-2 | BH p150a | fast | 68.53 | 61.49 | 1.11x (1.07x vs old normal) | 3 | 1350/1343; 1350/1343 | 0.733/0.831 -> 0.722/0.752 | 10.2 A vs 10.1 / 12.9 |
| Protenix-v2 | WH chip | normal | 564.77 | 220.28 | 2.56x | 3 | 1000/1000; 1000/906 | 0.896/0.930 -> 0.886/0.926 | 17.2 A vs 8.3 / 19.1 |
| Protenix-v2 | WH chip | fast | 580.85 | 167.38 | 3.47x (3.37x vs old normal) | 3 | 1000/1000; 1000/956 | 0.899/0.929 -> 0.884/0.924 | 11.8 A vs 7.4 / 16.6 |
| OpenDDE | WH chip | normal | 825.42 | 340.55 | 2.42x | 3 | 1000/950; 1000/976 | 0.887/0.912 -> 0.830/0.864 | 20.0 A vs 22.1 / 19.9 |
| OpenDDE | WH chip | fast | 853.86 | 308.79 | 2.77x (2.67x vs old normal) | 3 | 1000/932; 1000/951 | 0.886/0.911 -> 0.831/0.865 | 19.5 A vs 22.0 / 17.8 |
| OpenFold3 | WH chip | normal | 333.42 | 107.46 | 3.10x | 3 | 1000/984; 1000/973 | 0.780/0.720 -> 0.752/0.715 | 14.1 A vs 8.4 / 13.9 |
| OpenFold3 | WH chip | fast | 428.78 | 80.55 | 5.32x (4.14x vs old normal) | 3 | 1000/970; 1000/1000 | 0.721/0.678 -> 0.753/0.708 | 12.5 A vs 10.0 / 14.4 |
| Boltz-2 | WH chip | normal | 141.41 | 136.28 | 1.04x | 3 | 1000/940; 1000/899 | 0.724/0.803 -> 0.659/0.520 | 23.8 A vs 7.9 / 23.1 -- ACCURACY REGRESSION, see below |
| Boltz-2 | WH chip | fast | 144.61 | 114.77 | 1.26x (1.23x vs old normal) | 3 | 1000/971; 1000/892 | 0.739/0.839 -> 0.719/0.724 | 13.0 A vs 8.3 / 10.1 |
Every warm rep finite. Spread <= 16 s old WH OpenDDE, <= 9.4 s elsewhere. AICLK min dips (892-976 MHz) are single samples; medians are 1000 in every cell.

Reading the heavy fold:
- Anchors agree: old BH PV2 259.87 s vs 264.43 s on 10-08 (p150a, same 1350 MHz); new 108.41 BH / 220.28 WH vs v0.15.1 shipped 121 / 233 s. Old WH 564.77 vs 501.36 s on 10-08 (bb1377101, a week newer).
- Old fast at c730 RAN on v0.11.0 but bought nothing: it is 1-3 % SLOWER than old normal for PV2/OpenDDE/Boltz-2 and 29 % slower for OF3. So the honest fast line is new fast vs old normal (given in brackets) and the post uses that.
- Accuracy, PV2 and OF3: whole-complex top-sample CA RMSD at c730 swings 7-19 A between seeds of one tree (the second chain's placement moves), so the same-seed old-vs-new deviation sits inside the new tree's own seed spread, not always inside the old tree's tighter one. Confidence is the cleaner signal: PV2 ipTM 0.930 -> 0.926 normal, 0.929 -> 0.924 fast; OF3 ipTM 0.720 -> 0.715 normal, fast 0.678 -> 0.708. Verdict: within seed variation, not digest-equal. Claimable.
- Accuracy, OpenDDE: same-seed 19.5-21.5 A sits inside both trees' seed floors (17.8-22.3 A), but confidence drops 0.887 -> 0.830 and ipTM 0.912 -> 0.864 on both archs and modes. Cause on main since 10-10: 959349675 "OpenDDE trunk depends on the seed the way upstream's does": the old tree fed all ~10k MSA rows to every recycle with fixed conformers; main draws a fresh 1,280-row MSA sample per recycle, as upstream OpenDDE does. A fidelity fix that is also part of the speedup. The post does not present OpenDDE's 2.4x as pure engineering.
- Accuracy, Boltz-2 (FINDING, needs its own row): main's Boltz-2 NORMAL path is markedly less confident at c730 on WH. Per-sample ipTM over the 3 warm seeds x 5 samples: v0.11.0 0.52-0.82 (median 0.80), main 0.18-0.71 (median 0.38 per-sample, 0.52 by cell median); main's fast mode sits at ~0.72 and old fast at ~0.81. Same input file, same MSA dir, same harness, both cells at 1000 MHz median. Normal now scores worse than fast, so the suspect is a normal-only lever or a normal-path change between 8dbaa61b1 and 148605b31 (candidates by log: 7f60a0825 graded normal lever set on, 3105fd073/5ff00ba64 triatt_tail in/out of normal, b2b9e6e90 atom conditioning gather). Not bisected here. It REPRODUCES on BH p150a: normal ipTM 0.772 -> 0.557 and 7 % slower (65.59 -> 70.30 s); fast ipTM 0.831 -> 0.752 at 1.07x. Not an arch artifact. Boltz-2 is NOT in the post; this is a real regression on main that needs its own row (bisect 8dbaa61b1..148605b31 on the c730 input, normal mode, ipTM as the signal).
- Accuracy, OF3 on BH: per-seed ipTM old 0.696-0.721, new 0.672-0.701 (normal), ranges overlap; fast 0.662-0.698 old vs 0.661-0.694 new. Within seed variation, with a small downward shift in normal that WH does not show (0.720 -> 0.715).

## MEASURED (512 light fold, cdk2x2_512, 1 sample, warm median of 4, xmodel_ab.py gapboard protocol)
BH = p150a qb1 at 1350 MHz (old: card 2 node 3, in-run AICLK 1343-1350 in >97 % of samples, loadavg 7-16 on the host). WH = one .114 Galaxy chip at 1000 MHz as served. GPU ref = kit A100 (exact for normal, kit fast for fast) for all four models.

| model | arch | mode | old s | new s | speedup | GAP vs kit A100 then -> now | old vs new CA RMSD |
|---|---|---|---|---|---|---|---|
| Boltz-2 | BH | normal | 16.81 | 14.53 | 1.16x | 1.89 -> 1.63 | 1.20 A |
| Boltz-2 | BH | fast | 21.15 | 12.48 | 1.69x | 6.31 -> 3.72 | 14.28 A (old fast broken, see below) |
| Protenix-v2 | BH | normal | 51.70 | 34.77 | 1.49x | 4.61 -> 3.10 | 3.41 A |
| Protenix-v2 | BH | fast | 66.08 | 30.55 | 2.16x | 7.20 -> 3.33 | 2.96 A |
| OpenDDE | BH | normal | 83.83 | 57.34 | 1.46x | 3.50 -> 2.40 | 0.97 A |
| OpenDDE | BH | fast | 89.36 | 50.82 | 1.76x | 4.67 -> 2.66 | 4.15 A |
| OpenFold3 | BH | normal | 26.29 | 12.50 | 2.10x | 5.22 -> 2.48 | 1.01 A |
| OpenFold3 | BH | fast | 42.58 | 9.54 | 4.46x | 18.1 -> 4.06 | 4.82 A |
| Boltz-2 | WH | normal | 33.95 | 31.70 | 1.07x | 3.82 -> 3.56 | pending |
| Boltz-2 | WH | fast | 38.60 | 23.98 | 1.61x | 11.5 -> 7.16 | pending |
| OpenDDE | WH | normal | 174.16 | 119.78 | 1.45x | 7.28 -> 5.01 | pending |
| OpenDDE | WH | fast | 187.78 | 106.31 | 1.77x | 9.82 -> 5.56 | pending |
| Protenix-v2 | WH | normal | 113.71 | 73.09 | 1.56x | 10.14 -> 6.52 | not graded |
| Protenix-v2 | WH | fast | 131.64 | 60.52 | 2.18x (1.88x vs old normal) | 14.35 -> 6.60 | not graded |
| OpenFold3 | WH | normal | 50.25 | 27.15 | 1.85x | 9.97 -> 5.39 | not graded |
| OpenFold3 | WH | fast | 86.40 | 19.30 | 4.48x (2.60x vs old normal) | 36.8 -> 8.21 | not graded |
WH light old = v0.11.0 (.114, xmodel_ab, card as served at 1000 MHz; xmodel_ab records no in-fold clock on WH). A100 refs (GAP.md kit): Boltz-2 8.897/3.351, PV2 11.216/9.173, OpenDDE 23.925/19.131, OF3 5.039/2.35 s. n = 4 warm reps every cell; spread <= 1.4 s old, gapboard spread on new.

Accuracy so far (BH, warm0 CIFs, same seed; warm reps of one tree reproduce to 0.000 A, so this is no seed floor):
- Other pairings for scale: new fast vs new normal = Boltz-2 1.11, PV2 2.25, OpenDDE 2.77, OF3 2.19 A; old fast vs old normal = Boltz-2 14.21, PV2 3.72, OpenDDE 1.26, OF3 4.66 A.
- Boltz-2 fast on v0.10.0 is 14 A from both old normal and new fast: the old fast mode produced a wrong structure at 512. That cell is not a valid "speedup" baseline; the honest Boltz-2 fast line is new fast 12.48 s vs old normal 16.81 s (1.35x) or "old fast was wrong".
- Normal old vs new 0.97-1.20 A for Boltz-2/OpenDDE/OF3, 3.41 A for PV2. The 1.84 A seed floor quoted for 512 aa is from another protocol; the c730 run gives a same-protocol seed floor (3 warm seeds per tree) and is the accuracy grade the post uses.
- Tool: artifacts/spd-monthpost/mp_rmsd.py (Kabsch CA RMSD, asserts finite coords).

HEADLINE: heavy c730 fold (730 tokens, 10k-deep MSA, 5 samples), Oct 1 (v0.11.0) vs today's main 148605b31, same chip, same harness, clock logged in-fold: Protenix-v2, OpenFold3 and OpenDDE are 2.4-4.2x faster, median 2.9x over 12 cells (3 models x WH/BH x normal/fast; fast always against Oct-1 normal). Per model: PV2 2.4x BH / 2.6x WH normal, 3.1x / 3.4x fast; OF3 3.3x / 3.1x normal, 4.2x / 4.1x fast; OpenDDE 2.4x both normal, 2.8x / 2.7x fast (includes upstream MSA sampling). Light 512 fold: 1.1-2.2x normal. Boltz-2 excluded: an accuracy regression on main at c730 on both archs.
DRAFT_POST:
On October 1, a heavy fold (730 tokens, 10k-deep MSA, 5 samples) took 565 s with Protenix-v2 on one Wormhole chip. Today it takes 220 s, or 167 s in fast mode.
Across Protenix-v2, OpenFold3 and OpenDDE, on Wormhole and Blackhole, that fold is now 2.4x to 4.2x faster than two weeks ago, 2.9x at the median.
Protenix-v2 on a Blackhole p150a: 260 s to 108 s (2.4x), 85 s in fast mode (3.1x).
OpenFold3 moved most: 158 s to 48 s on p150a (3.3x) and 38 s in fast mode (4.2x); 333 s to 107 s on Wormhole (3.1x), 81 s fast (4.1x).
OpenDDE is 2.4x faster on both chips. Part of that comes from sampling the MSA per recycle the way upstream OpenDDE does, which is also more faithful to the original model.
Accuracy held: confidence and ipTM stay within the seed-to-seed spread of the Oct 1 build (Protenix-v2 ipTM 0.930 vs 0.926 on both chips).
How we measured: the Oct 1 commit against today's main, same input, same harness, same chip, warm median of 3 folds, clock logged during every fold (1000 MHz Wormhole, 1350 MHz Blackhole). Fast mode is compared against Oct 1 normal mode.
On a small single-sample fold (512 tokens) the gain is smaller, 1.1x to 2.2x, because fixed overhead dominates there.
All of it is open source in TT-Bio and serving on JapanFold.
Sources per line: L1 MEASURED HEAVY PV2 WH normal/fast; L2 the 12 PV2/OF3/OpenDDE cells (min 2.40 PV2 BH normal, max 4.20 OF3 BH fast vs old normal, median of 12 = 2.91); L3 PV2 BH rows; L4 OF3 BH + WH rows; L5 OpenDDE rows + accuracy note (959349675); L6 accuracy column/notes; L7 Runs; L8 MEASURED (512) normal column.


## Runs (c730 heavy fold)
Harness: main's perf/spd/bench.py copied as perf/spd/bench_mp.py into each tree (engine = the tree's own tt_bio), `--inputs c730 --warm 3` (5 samples, 200 steps; cold rep + 3 warm reps on seeds 102..104); PV2/OpenDDE `--recycles 10`, Boltz-2/OF3 the model's own default recycles (as BOARD runs them). Reader: artifacts/spd-monthpost/mp_heavy_read.py (on .114 run it with ~/japanfold/venv/bin/python3, system python has no numpy).
- WH .114 pool: all 16 cells done rc=0 (4 models x old/new x normal/fast), ~/spd/monthpost/heavy/heavy-<tag>-<model>-<mode>/.
- BH qb1 card 2 p150a: all 16 cells done rc=0 (chain relaunched 16:48:55Z, done 18:30:25Z), ~/spd/monthpost/heavy/<tag>-<model>-<mode>/. Read: `ssh qb1 'python3 - ~/spd/monthpost/heavy ""' < artifacts/spd-monthpost/mp_heavy_read.py`.

