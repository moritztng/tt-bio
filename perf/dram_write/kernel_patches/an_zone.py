import csv, collections, statistics as st
rows = list(csv.reader(open("/tmp/prof_zone2/.logs/profile_log_device.csv")))[2:]
FREQ = 1350.0
TILE = 2048.0
byrun = collections.defaultdict(list)
for r in rows:
    if len(r) < 12:
        continue
    byrun[r[7]].append(r)

for name, rid in [("K=256 obh=4 DRAM", "4096"), ("K=256 obh=4 L1", "11264"),
                  ("K=4096 obh=1 DRAM", "18432"), ("K=4096 obh=1 L1", "25600")]:
    rs = byrun[rid]
    t0 = min(int(r[5]) for r in rs if r[10].endswith("-KERNEL") and r[11] == "ZONE_START")
    cores = set((r[1], r[2]) for r in rs if r[10] == "SB_DRAIN")
    tr = [(int(r[5]) - t0) / FREQ for r in rs if r[10] == "TRISC-KERNEL" and r[11] == "ZONE_END"]
    br = [(int(r[5]) - t0) / FREQ for r in rs if r[10] == "BRISC-KERNEL" and r[11] == "ZONE_END"]
    done = sorted((int(r[5]) - t0) / FREQ for r in rs if r[10] == "SB_DRAIN" and r[11] == "ZONE_END")
    if not done:
        print("\n=== %s === no zone data" % name); continue
    per = collections.defaultdict(lambda: {"w": 0.0, "d": 0.0, "os": None, "oe": None})
    for r in rs:
        k = (r[1], r[2]); t = int(r[5])
        if r[10] in ("SB_WAIT", "SB_DRAIN"):
            key = "w" if r[10] == "SB_WAIT" else "d"
            if r[11] == "ZONE_START": per[k].setdefault("_" + key, []); per[k]["_" + key] = per[k].get("_" + key, []) + [t]
            else: per[k][key] = per[k].get(key, 0.0)
        if r[10] == "OUT_SECTION":
            if r[11] == "ZONE_START": per[k]["os"] = (t - t0) / FREQ
            else: per[k]["oe"] = (t - t0) / FREQ
    # recompute wait/drain totals properly
    tot = collections.defaultdict(lambda: {"SB_WAIT": [], "SB_DRAIN": []})
    ev = collections.defaultdict(list)
    for r in rs:
        if r[10] in ("SB_WAIT", "SB_DRAIN"):
            ev[((r[1], r[2]), r[10])].append((int(r[5]), r[11]))
    wt, dt = [], []
    for (k, zn), e in ev.items():
        e.sort()
        s = [t for t, ty in e if ty == "ZONE_START"]; en = [t for t, ty in e if ty == "ZONE_END"]
        n = min(len(s), len(en))
        v = sum(b - a for a, b in zip(s[:n], en[:n])) / FREQ
        (wt if zn == "SB_WAIT" else dt).append(v)
    mb = len(done) * 4 * TILE / 1e6
    ostart = min(v["os"] for v in per.values() if v["os"] is not None)
    oend = max(v["oe"] for v in per.values() if v["oe"] is not None)
    print("\n=== %s ===" % name)
    print("  cores writing: %d   subblocks retired: %d   bytes: %.2f MB" % (len(cores), len(done), mb))
    print("  last TRISC end %.1f us | last BRISC end %.1f us" % (max(tr), max(br)))
    print("  OUT_SECTION spans %.1f -> %.1f us (%.1f us wide)" % (ostart, oend, oend - ostart))
    print("  per-core totals: SB_WAIT median %.1f us | SB_DRAIN median %.1f us" % (st.median(wt), st.median(dt)))
    tend = max(tr)
    during = sum(1 for d in done if d <= tend)
    after = len(done) - during
    bd = during * 4 * TILE / 1e6; ba = after * 4 * TILE / 1e6
    print("  bytes retired DURING compute (t<=%.1f): %.2f MB  -> %.1f GB/s over that window" % (tend, bd, bd * 1e6 / tend / 1e3))
    if oend > tend:
        print("  bytes retired AFTER  compute (%.1f us tail): %.2f MB -> %.1f GB/s" % (oend - tend, ba, ba * 1e6 / (oend - tend) / 1e3))
    print("  whole writeback: %.2f MB / %.1f us = %.1f GB/s" % (mb, oend - ostart, mb * 1e6 / (oend - ostart) / 1e3))
    print("  retirement curve (cum MB by t):")
    for e in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170]:
        if e > oend + 10: break
        c = sum(1 for d in done if d <= e)
        print("     t=%3d us  %6.2f MB (%4.1f%%)" % (e, c * 4 * TILE / 1e6, 100.0 * c / len(done)))
