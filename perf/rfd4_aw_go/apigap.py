import ast, importlib, sys, pathlib, collections
root = pathlib.Path(sys.argv[1]); sys.path.insert(0, str(root))
pairs = collections.defaultdict(set)
files = list((root/"tt_bio/rfd4").rglob("*.py")) + list((root/"tt_bio/_vendor/rfd4_prep").rglob("*.py"))
for f in files:
    for n in ast.walk(ast.parse(f.read_text())):
        if isinstance(n, ast.ImportFrom) and n.module and "atomworks" in n.module:
            for a in n.names: pairs[(n.module, a.name)].add(f"{f.relative_to(root)}:{n.lineno}")
        elif isinstance(n, ast.Import):
            for a in n.names:
                if "atomworks" in a.name: pairs[(a.name, None)].add(f"{f.relative_to(root)}:{n.lineno}")
miss = collections.defaultdict(list)
for (m, name), where in sorted(pairs.items(), key=lambda kv: (kv[0][0], kv[0][1] or "")):
    try:
        mod = importlib.import_module(m)
        if name and not hasattr(mod, name):
            try: importlib.import_module(m + "." + name)
            except Exception: miss["name"].append((m, name, sorted(where)[0], len(where)))
    except Exception as e:
        miss["module"].append((m, name, sorted(where)[0], len(where), type(e).__name__ + ":" + str(e)[:80]))
print("pairs", len(pairs))
for k, v in miss.items():
    print("==", k, len(v))
    for x in v: print("  ", *x)
