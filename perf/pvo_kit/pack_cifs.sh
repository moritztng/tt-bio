#!/bin/bash
# Last step on the box (box_kit.sh EXTRA): pack the structures the DockQ guard needs into one small tar, so they
# leave the box BEFORE it is destroyed. pvo-gpu lost a whole RFD4 design set exactly this way, and the session's
# ordinary pull excludes *.cif on purpose (a 279 MB CIF pull once froze the watch loop for 35 minutes while the box
# kept billing). One top-ranked pose per arm / input / round is the whole guard: ~56 files, tens of MB, not 500.
set -u
B=/root/kc/pvo
python3 - "$B" <<'PY'
import json, pathlib, sys, tarfile

root = pathlib.Path(sys.argv[1])
picked, skipped = [], []
# pvo_fold.py writes <arm>/<input>/r<rnd>/p<rank>/... ; upstream's dumper puts the samples and their
# *_summary_confidence_sample_<k>.json side by side. Top-ranked = the sample with the highest ranking score, the
# same definition pvo-accuracy fixed before it had any data.
for rd in sorted(root.glob("*/*/*/r*/p*")):
    cifs = sorted(rd.rglob("*.cif"))
    if not cifs:
        continue
    best, score = None, None
    for c in cifs:
        j = next(iter(c.parent.glob(f"*summary_confidence*{c.stem.split('_')[-1]}.json")), None)
        s = None
        if j:
            try:
                d = json.loads(j.read_text())
                s = d.get("ranking_score", d.get("ranking_confidence"))
            except Exception:
                s = None
        if best is None or (s is not None and (score is None or s > score)):
            best, score = c, s if s is not None else score
    if score is None:
        skipped.append(str(rd))          # no ranking score found: keep the pose, say so, never pretend it ranked
    picked.append((best, score))

out = root / "guard-cifs.tar"
with tarfile.open(out, "w") as t:
    for c, _ in picked:
        t.add(c, arcname=str(c.relative_to(root)))
        for j in c.parent.glob("*summary_confidence*.json"):
            t.add(j, arcname=str(j.relative_to(root)))
(root / "guard-cifs.index.json").write_text(json.dumps(
    {"picked": [{"cif": str(c.relative_to(root)), "ranking_score": s} for c, s in picked],
     "no_ranking_score": skipped}, indent=1) + "\n")
print(f"packed {len(picked)} top-ranked poses -> {out} "
      f"({out.stat().st_size/2**20:.1f} MiB), {len(skipped)} without a ranking score")
PY
echo "$(date -u +%FT%TZ) pack_cifs: done"
