"""Rank census sites by row deficit AND by softmax element volume.

Deficit says how much a site loses; volume says what fixing it costs. A site can have the worst
deficit in a model and still be free to fix (tiny volume), or a mild deficit and be unaffordable.
"""
import glob, json, math, os, collections

ROOT = "/home/ttuser/.coworker/wt/shared-softmax-crossmodel"
LEVER = "tt_bio/tenstorrent.py:4058"   # the AttentionPairBias site accurate_softmax reaches

for f in sorted(glob.glob(os.path.join(ROOT, "perf/xmsoftmax/results/census_*_256.json"))):
    model = os.path.basename(f)[len("census_"):-len("_256.json")]
    d = json.load(open(f))
    rows = [s for s in d["sites"] if s["deficit"] is not None]
    for s in rows:
        s["vol"] = math.prod(s["shape"]) * s["n_calls"]
    tot = sum(s["vol"] for s in rows) or 1
    print(f"\n=== {model} @256aa ===  total softmax volume {tot/1e9:.3f} G elem")
    print(f"{'site':<48} {'dtype':<9} {'shape':<22} {'n':>5} {'deficit':>9} {'min':>8} {'vol G':>8} {'%vol':>6}")
    for s in sorted(rows, key=lambda r: -r["vol"]):
        mark = "  <-- lever" if s["site"] == LEVER else ""
        print(f"{s['site']:<48} {s['dtype'].replace('DataType.',''):<9} {str(s['shape']):<22} "
              f"{s['n_calls']:>5} {s['deficit']:>9.5f} {s['rowsum_min']:>8.5f} "
              f"{s['vol']/1e9:>8.4f} {100*s['vol']/tot:>5.1f}%{mark}")
    lv = sum(s["vol"] for s in rows if s["site"] == LEVER)
    # volume-weighted mean deficit: what the model loses overall, and what share the lever fixes
    wd = sum(s["deficit"] * s["vol"] for s in rows) / tot
    print(f"  volume-weighted mean deficit: {wd:.5f}")
    print(f"  lever ({LEVER}) covers {100*lv/tot:.1f}% of softmax volume "
          f"({lv/1e9:.4f} G of {tot/1e9:.4f} G)")
