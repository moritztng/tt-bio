"""net6: record every taped op of the l1c1 trunk forward (output digest + parent digests at read time), warm vs graded.
A parent whose bytes changed between its production and a later read is the stale-memory site."""
import os, sys, runpy, hashlib, json
sys.path.insert(0, os.getcwd())
import ttnn
from tt_bio import autograd as AG
import tt_bio
LOG = []; PHASE = ["init"]; SEEN = {}; PH0 = {"init": 0}; SAVE = {int(x) for x in os.environ.get("SAVE", "").split(",") if x}
def dg(v):
    try:
        if v.storage_type() != ttnn.StorageType.DEVICE or not v.is_allocated():
            return None
        t = ttnn.to_torch(v).float()
        return hashlib.sha1(t.numpy().tobytes()).hexdigest()[:12]
    except Exception as e:  # noqa
        return "ERR"
orig = AG._tape
def tape(out_value, parents, make_fn, reads=None):
    i = len(LOG)
    name = getattr(make_fn, "__qualname__", str(make_fn))
    pd = []
    for p in parents:
        d = dg(p.value)
        prev = SEEN.get(id(p))
        if prev is not None and prev[1] != d:
            pd.append(("CHANGED", prev[0], prev[1], d))
        else:
            pd.append(("ok", None if prev is None else prev[0], d))
    out = orig(out_value, parents, make_fn, reads)
    od = dg(out_value)
    SEEN[id(out)] = (i, od)
    for p in parents:
        SEEN.setdefault(id(p), (-1, dg(p.value)))
    k = i - PH0[PHASE[0]]
    if k in SAVE:
        import torch
        torch.save({"out": ttnn.to_torch(out_value).float(), "parents": [ttnn.to_torch(p.value).float() if p.value.is_allocated() else None for p in parents],
                    "pdt": [str(p.value.dtype) for p in parents], "odt": str(out_value.dtype),
                    "mem": [str(p.value.memory_config()) for p in parents] + [str(out_value.memory_config())]},
                   os.environ["OPLOG"] + f".{PHASE[0]}.op{k}.pt")
    LOG.append({"i": i, "phase": PHASE[0], "op": name, "shape": list(out_value.shape), "out": od, "parents": pd})
    return out
AG._tape = tape
import importlib, pkgutil
for m in list(sys.modules.values()):
    if m and getattr(m, "__name__", "").startswith("tt_bio") and getattr(m, "_tape", None) is orig:
        m._tape = tape
import tt_bio.taped_ttnn as TT
TT._tape = tape
otape = TT.tape
import contextlib
n = [0]
@contextlib.contextmanager
def tape_ctx(*a, **k):
    PHASE[0] = f"tape{n[0]}"; n[0] += 1; SEEN.clear(); PH0[PHASE[0]] = len(LOG)
    with otape(*a, **k) as t:
        yield t
TT.tape = tape_ctx
import atexit
def dump():
    json.dump(LOG, open(os.environ["OPLOG"], "w"))
    by = {}
    for e in LOG: by.setdefault(e["phase"], []).append(e)
    ph = sorted(by)
    print("phases", {p: len(by[p]) for p in ph})
    for e in LOG:
        for p in e["parents"]:
            if p[0] == "CHANGED":
                print("CHANGED parent", e["phase"], e["i"], e["op"], p)
    if len(ph) >= 2:
        a, b = by[ph[0]], by[ph[1]]
        for x, y in zip(a, b):
            if x["op"] != y["op"] or x["out"] != y["out"]:
                print("FIRST DIFF warm vs graded at", x["i"] - a[0]["i"], x["op"], y["op"], x["shape"], x["out"], y["out"], "parents", x["parents"], y["parents"])
                break
atexit.register(dump)
sys.argv = ["perf/r4t_tape/block_gradcheck.py"] + sys.argv[1:]
runpy.run_path(os.environ.get("BGC", "perf/r4t_tape/block_gradcheck.py"), run_name="__main__")
