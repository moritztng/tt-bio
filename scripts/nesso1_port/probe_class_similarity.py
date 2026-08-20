import ast, difflib, pathlib
NES = pathlib.Path("/home/ttuser/scratch/nesso1/nesso")
TT  = pathlib.Path("/home/ttuser/.coworker/wt/nesso1-port-p1-parity")

def defs(path):
    src = path.read_text()
    tree = ast.parse(src)
    lines = src.splitlines()
    out = {}
    for n in ast.walk(tree):
        if isinstance(n, (ast.ClassDef, ast.FunctionDef)):
            out.setdefault(n.name, []).append("\n".join(lines[n.lineno-1:n.end_lineno]))
    return out

nes = {}
for p in sorted(NES.glob("nesso/**/*.py")):
    for k, v in defs(p).items():
        nes.setdefault(k, []).extend((str(p.relative_to(NES)), b) for b in v)
tt = {}
for rel in ["tt_bio/boltz2.py", "tt_bio/reference.py", "tt_bio/data/featurizer.py",
            "tt_bio/data/tokenize.py", "tt_bio/data/const.py", "tt_bio/data/types.py",
            "tt_bio/data/mol.py", "tt_bio/data/parse.py", "tt_bio/data/pad.py"]:
    p = TT / rel
    for k, v in defs(p).items():
        tt.setdefault(k, []).extend((p.name, b) for b in v)

def norm(s):
    return [l.rstrip() for l in s.splitlines() if l.strip() and not l.strip().startswith("#")]

rows = []
for name, items in sorted(nes.items()):
    if name.startswith("_") or name in ("forward", "__init__", "process"):
        continue
    src_file, body = items[0]
    best = (0.0, "-")
    for tf, tb in tt.get(name, []):
        r = difflib.SequenceMatcher(None, norm(body), norm(tb)).ratio()
        if r > best[0]:
            best = (r, tf)
    rows.append((name, src_file, len(norm(body)), best[1], round(best[0], 3)))
rows.sort(key=lambda r: (-r[4], -r[2]))
print("%-40s %-32s %4s %-18s %s" % ("name", "nesso file", "loc", "tt_bio match", "sim"))
for r in rows:
    print("%-40s %-32s %4d %-18s %s" % r)
