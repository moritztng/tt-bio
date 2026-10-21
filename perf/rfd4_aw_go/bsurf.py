import biotite, importlib, inspect, json, pkgutil, sys
out = {}
for m in pkgutil.walk_packages(biotite.__path__, "biotite."):
    if any(x in m.name for x in ("application", "database", "interface", "sequence.align", "sequence.graphics")): continue
    try: mod = importlib.import_module(m.name)
    except Exception: continue
    for n, o in vars(mod).items():
        if n.startswith("__"): continue
        key = m.name + "." + n
        out[key] = sorted(a for a in dir(o) if not a.startswith("__")) if inspect.isclass(o) else None
json.dump(out, open(sys.argv[1], "w"))
print(biotite.__version__, len(out))
