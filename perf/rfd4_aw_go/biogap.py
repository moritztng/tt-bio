import ast, sys, pathlib, importlib, collections
roots = [pathlib.Path(p) for p in sys.argv[1:]]
uses = collections.defaultdict(set)
for root in roots:
  for f in root.rglob("*.py"):
    try: tree = ast.parse(f.read_text())
    except Exception: continue
    alias = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name.startswith("biotite"): alias[a.asname or a.name] = a.name
        elif isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("biotite") and n.level == 0:
            for a in n.names:
                try:
                    importlib.import_module(n.module + "." + a.name); alias[a.asname or a.name] = n.module + "." + a.name
                except Exception:
                    uses[(n.module, a.name)].add(f"{f}:{n.lineno}")
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute):
            chain = []; x = n
            while isinstance(x, ast.Attribute): chain.append(x.attr); x = x.value
            if isinstance(x, ast.Name) and x.id in alias:
                chain.reverse(); mod = alias[x.id]
                # walk modules as deep as they go, then check the first attribute
                for i, part in enumerate(chain):
                    try: importlib.import_module(mod + "." + part); mod = mod + "." + part
                    except Exception: uses[(mod, part)].add(f"{f}:{n.lineno}"); break
import biotite; print("biotite", biotite.__version__)
miss = []
for (m, a), w in sorted(uses.items()):
    try: ok = hasattr(importlib.import_module(m), a)
    except Exception: ok = False
    if not ok: miss.append((m, a, len(w), sorted(w)[0]))
print("checked", len(uses), "missing", len(miss))
for x in miss: print(*x)
