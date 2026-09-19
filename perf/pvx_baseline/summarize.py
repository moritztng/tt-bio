#!/usr/bin/env python3
"""Read the cell sessions in a directory and print one line each, plus the ratios they support.

Ratios are only printed for sessions that are clean (no foreign device holder on any fold, cold
included) and that held their clock: a session whose sampled AICLK min fell below its target was
not measured at the clock it claims, and is marked rather than quietly averaged in.
"""
import glob
import json
import statistics as st
import sys
from pathlib import Path

d = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
rows = {}
for f in sorted(glob.glob(str(d / "*.json"))):
    j = json.load(open(f))
    s = j.get("summary")
    tag = j.get("tag") or Path(f).stem
    if not s:
        print(f"{tag:14s} in flight, {len(j.get('runs', []))} folds so far")
        continue
    held = (j["clock_target"] == 0
            or (s["aiclk_min_over_timed"] or 0) >= j["clock_target"] - 50)
    rows[tag] = dict(s, clock_target=j["clock_target"], grid=j.get("grid"),
                     host=j["host"], card=j["card_env"], held=held,
                     model=j["model"], rec=j["recycling_steps"], steps=j["sampling_steps"])
    print(f"{tag:14s} {j['model']:11s} n={s['n']} med={s['median_fold_s']:8.3f}s  "
          f"A/A {s['aa_floor_s']:.3f}s ({s['aa_floor_pct']:.2f}%)  "
          f"clk target {j['clock_target']} mean {s['aiclk_mean_over_timed']} "
          f"min {s['aiclk_min_over_timed']} held={held}  "
          f"clean={s['clean_session']}  digests={len(s['digests'])}  "
          f"grid={j.get('grid')}  {s['fold_s_sorted']}")


def pool(prefix):
    f, ok = [], True
    for tag, r in rows.items():
        if tag.startswith(prefix):
            f += r["fold_s_sorted"]
            ok = ok and r["clean_session"] and r["held"]
    return (sorted(f), ok)


print()
for a, b, what in (("b2_old_s", "b2_new_s", "Boltz-2 pinned 1350: old tree / new tree"),
                   ("b2_old_gov", "b2_old_s", "Boltz-2 old tree: governor / pinned (clock term)"),
                   ("b2_new_gov", "b2_new_s", "Boltz-2 new tree: governor / pinned (clock term)"),
                   ("ptx_new_gov", "ptx_new_s", "Protenix: governor / pinned (clock term)")):
    fa, oka = pool(a)
    fb, okb = pool(b)
    if fa and fb:
        r = st.median(fa) / st.median(fb)
        flag = "" if (oka and okb) else "   [NOT CLEAN OR CLOCK NOT HELD -- do not quote]"
        print(f"{what:52s} {st.median(fa):8.3f} / {st.median(fb):8.3f} = {r:.4f}x"
              f"   (n={len(fa)},{len(fb)}){flag}")

for tag in ("ptx_new_s1", "ptx_new_s2"):
    if tag in rows:
        pass
fp, okp = pool("ptx_new_s")
if fp:
    print(f"\nProtenix-v2 pinned 1350, pooled: median {st.median(fp):.3f}s over n={len(fp)} "
          f"(clean and clock held: {okp}); against the 54.760 s cell that is "
          f"{54.760 / st.median(fp):.4f}x")
fb2, okb2 = pool("b2_new_s")
if fb2 and fp:
    print(f"Protenix / Boltz-2 at 512 aa, both pinned: {st.median(fp):.3f} / "
          f"{st.median(fb2):.3f} = {st.median(fp) / st.median(fb2):.4f}x")
