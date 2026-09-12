import json, sys
new = json.load(open("perf/b2z_stack/out/roofs_card3.json"))
old = json.load(open("perf/bioir_roofline/roofs_p300c_qb2_card2.json"))
print("grid card3", new["grid"], " card2", old["grid"])
for sec in ("copy", "read", "matmul"):
    print("===", sec)
    for a, b in zip(new.get(sec, []), old.get(sec, [])):
        label = str(a.get("shape") or a.get("fidelity"))
        for kk in a:
            if "GBps" not in kk and "TFLOP" not in kk:
                continue
            r = a[kk] / b[kk] if b.get(kk) else float("nan")
            print("  %-26s %-16s card3=%9.2f card2=%9.2f ratio=%.4f" % (label, kk, a[kk], b[kk], r))
