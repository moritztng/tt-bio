import sys, pathlib, importlib, difflib, pkgutil
root = pathlib.Path(sys.argv[1]); other = pathlib.Path(sys.argv[2]); sys.path.insert(0, str(root))
import tt_bio.rfd4
for m in pkgutil.walk_packages([str(root/"tt_bio/_vendor/rfd4_prep")], "tt_bio._vendor.rfd4_prep."):
    try: importlib.import_module(m.name)
    except Exception as e: print("IMPORTFAIL", m.name, e)
for m in list(pkgutil.walk_packages([str(root/"tt_bio/rfd4")], "tt_bio.rfd4.")):
    try: importlib.import_module(m.name)
    except Exception as e: print("IMPORTFAIL", m.name, e)
tot = dict(same=0, diff=0, new=0); lines = dict(same=0, diff=0, new=0); chg = 0
rows = []
for name, mod in sorted(sys.modules.items()):
    if not name.startswith("tt_bio._vendor.atomworks") or not getattr(mod, "__file__", None): continue
    f = pathlib.Path(mod.__file__); rel = f.relative_to(root); o = other/rel
    a = f.read_text().splitlines()
    if not o.exists(): k = "new"; d = len(a)
    else:
        b = o.read_text().splitlines()
        d = sum(1 for l in difflib.unified_diff(b, a, lineterm="", n=0) if l[:1] in "+-" and not l.startswith(("+++","---")))
        k = "same" if d == 0 else "diff"
    tot[k] += 1; lines[k] += len(a); chg += d if k != "same" else 0
    if k != "same": rows.append((k, d, str(rel)))
print("modules", tot, "lines", lines, "changed_lines", chg)
for r in sorted(rows, key=lambda r: -r[1])[:45]: print(*r)
