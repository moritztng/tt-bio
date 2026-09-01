# protenix-v1 perf cell — the draws behind the p300c/tt-quietbox2 baseline

`docs/perf_baselines.json` gained a `protenix-v1` cell on 2026-09-01 at 3.327857 structures/s.
v0.7.2 had left it unseeded: two draws read 3.20726 and 3.292402, 2.65% apart, over that release's
own 2% agreement bar.

`draws.json` holds the 11 fresh-process draws that characterised the protocol before the cell was
written, plus the seeding run and its two verification runs. Every number came from
`scripts/perf_regression.py` unchanged (`--measure` for the draws, `--update-baseline` for the
cell), one process per draw, the whole series under `benchlock.sh` on an idle box.

What the draws say: two noise components, and only one of them survives the median. Inside a
process, one or two of the five timed folds sometimes land up to 3% high (draw 8 is
0.2978/0.2980/0.2983/0.2986/0.3069 s); the median of five absorbs that. What it cannot absorb is a
whole-process offset, and the draws show one plainly: draw 7's fastest fold, 0.3047 s, is slower
than draw 4's slowest, 0.2994 s, two processes forty seconds apart on the same idle box with
non-overlapping fold distributions. That is the component that makes two draws agreeing a coin
flip, and it is why v0.7.2's test failed. More draws fix it, more warmup does not. Ten benchlocked
draws span 2.97% end to end, nine of them inside 1.66%, so a 15% gate keeps about 5x headroom over
the worst excursion.

Same box, same session, as a control on the hardware: the replacement qb2 measured protenix-v2 at
3.118 and opendde at 2.843 against 3.024 and 2.895 on the previous physical box at the same tag.
The two boxes agree to within 3%, so this cell sits on the same scale as the numbers v0.7.2 quoted.
