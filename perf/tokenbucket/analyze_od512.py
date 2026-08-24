"""Read a paired A/B run and judge it two ways: the plan's strict whole-run criteria, and a
per-pair load filter (a single co-tenant spike should cost that pair, not the whole run)."""
import json, sys, statistics as st

path = sys.argv[1]
LOAD_MAX = 7.0
d = json.load(open(path))
ab, w = d["ab"], d["warm_folds"]
on, off = ab["values"]

folds = {}
for f in w:
    folds.setdefault(f["arm"], []).append(f)
for a in folds:
    folds[a].sort(key=lambda f: f["arm_fold"])

print(f"{path}  model={d['model']}  n_tokens={d['n_tokens']}  ttnn={d['ttnn_version']}  "
      f"card={d['visible_devices']}")
for a in (on, off):
    ts = ab["arms"][a]["warm_times_s"]
    print(f"  arm {a}: median {st.median(ts):8.3f}  span {max(ts)-min(ts):.3f}  n={len(ts)}")
    shas = {v for f in folds[a] for v in f["cif_sha256"].values()}
    plddts = {f["plddt"] for f in folds[a]}
    print(f"           cif {sorted(shas)}  plddt {sorted(plddts)}")

print("\npair   ON        OFF       delta     loadON  loadOFF  kept")
kept = []
for i, (a, b) in enumerate(zip(folds[on], folds[off])):
    la, lb = float(a["loadavg"][0]), float(b["loadavg"][0])
    delta = a["s"] - b["s"]
    ok = la <= LOAD_MAX and lb <= LOAD_MAX
    if ok:
        kept.append(delta)
    print(f"{i:<6} {a['s']:9.3f} {b['s']:9.3f} {delta:+9.3f}  {la:6.2f}  {lb:6.2f}   {'yes' if ok else 'NO'}")

alld = ab["paired_delta_s"]
print(f"\nall {len(alld)} pairs: median {st.median(alld):+.3f} s  mean {st.mean(alld):+.4f} s")
loads = [float(f["loadavg"][0]) for f in w]
spans = {k: round(max(v['warm_times_s']) - min(v['warm_times_s']), 3) for k, v in ab["arms"].items()}
strict = max(loads) <= LOAD_MAX and (st.median(alld) >= 0) == (st.mean(alld) >= 0) and max(spans.values()) < 0.6
print(f"loadavg {min(loads):.2f}-{max(loads):.2f}  per-arm span {spans}")
print(f"STRICT (plan): {'ACCEPTED' if strict else 'REJECTED'}")

if len(kept) >= 4:
    m, mn = st.median(kept), st.mean(kept)
    agree = (m >= 0) == (mn >= 0)
    print(f"LOAD-FILTERED: {len(kept)}/{len(alld)} pairs at load <= {LOAD_MAX}, "
          f"median {m:+.3f} s  mean {mn:+.4f} s  sign-agree {agree}")
    if agree:
        print(f"  -> margin 0.167 s: OpenDDE {'KEEPS' if m <= 0.167 else 'LOSES'} the bar "
              f"(d = {m:+.3f} s)")
else:
    print(f"LOAD-FILTERED: only {len(kept)}/{len(alld)} pairs clean, need >= 4. No decision.")
