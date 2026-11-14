import os, sys
dump = bytes.fromhex(sys.argv[2])
pat = dump[0x40:0x68]   # 0x13240..0x13268, spans the polled PCs
root = sys.argv[1]; hits = []
for dp, dn, fn in os.walk(root):
    for f in fn:
        if f.endswith(".xip.elf") or (f.endswith(".elf") and not os.path.exists(os.path.join(dp, f + ".xip.elf"))):
            p = os.path.join(dp, f); d = open(p, "rb").read(); i = d.find(pat)
            if i >= 0: hits.append((os.path.getmtime(p), p, i))
for h in sorted(hits)[-15:]: print(h)
print(len(hits), "hits")
