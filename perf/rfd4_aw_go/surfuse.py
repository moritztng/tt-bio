import json, re, subprocess, sys
S = sys.argv[1]
a = json.load(open(f"{S}/b120.json")); b = json.load(open(f"{S}/b160.json"))
newmeth = set(); newfn = set()
for k, v in b.items():
    if k not in a: newfn.add(k.rsplit(".", 1)[1])
    elif v is not None and a[k] is not None:
        for m in set(v) - set(a[k]): newmeth.add((k, m))
names = sorted({m for _, m in newmeth} | newfn)
dirs = [f"{S}/t304/tt_bio/_vendor/atomworks", f"{S}/e2e304/tt_bio/_vendor/rfd4_prep", f"{S}/e2e304/tt_bio/rfd4", f"{S}/t304/tt_bio/rf3"]
src = subprocess.run(["bash", "-c", "cat $(find " + " ".join(dirs) + " -name '*.py')"], capture_output=True, text=True).stdout
for n in names:
    if len(n) < 4: continue
    c = len(re.findall(r"\.\s*" + re.escape(n) + r"\b", src))
    if c:
        owners = sorted({k for k, m in newmeth if m == n})[:3] or ["module-level"]
        print(f"{n:40s} {c:4d}  {owners}")
