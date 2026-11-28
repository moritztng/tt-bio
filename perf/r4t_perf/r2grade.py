"""r4t-perf R2 bar (a)/(d): every bfloat8_b device upload of a real trunk_step run, packed both ways.

Wraps ttnn.from_torch before trunk_step imports, runs it with device packing OFF (so the run itself is the
host-packed baseline), and for each bf8 upload also builds the device-packed tensor (float32 upload +
typecast; --broken: from a bfloat16 source, double rounding) and compares both against the float32 source.
(a) PASS iff no element of the device form is further from the source than the host form, and the differing
fraction is <= 1e-4. Usage: r2grade.py [--broken] OUT.json trunk_step.py ARGS..."""
import json, sys, runpy
from tt_bio.main import ensure_p300_mesh_descriptor; ensure_p300_mesh_descriptor()
import torch, ttnn
broken = sys.argv[1] == "--broken"
if broken:
    sys.argv.pop(1)
out = sys.argv[1]
orig = ttnn.from_torch
S = dict(uploads=0, elems=0, differ=0, dev_worse=0, dev_closer=0, max_worse=0.0)


def graded(x, *a, **k):
    t = orig(x, *a, **k)
    if torch.is_tensor(x) and not a and k.get("dtype") == ttnn.bfloat8_b and k.get("device") is not None:
        src = x.detach().to(torch.float32)
        s2 = src.to(torch.bfloat16) if broken else src
        d32 = orig(s2, **{**k, "dtype": ttnn.bfloat16 if broken else ttnn.float32})
        d = ttnn.typecast(d32, ttnn.bfloat8_b)
        h_ = ttnn.to_torch(t).float().reshape(src.shape)
        d_ = ttnn.to_torch(d).float().reshape(src.shape)
        ttnn.deallocate(d32); ttnn.deallocate(d)
        eh, ed = (h_ - src).abs(), (d_ - src).abs()
        S["uploads"] += 1; S["elems"] += src.numel(); S["differ"] += int((h_ != d_).sum())
        w = ed > eh
        S["dev_worse"] += int(w.sum()); S["dev_closer"] += int((ed < eh).sum())
        if w.any():
            S["max_worse"] = max(S["max_worse"], float((ed - eh)[w].max()))
    return t


ttnn.from_torch = graded
sys.argv = sys.argv[2:]
try:
    runpy.run_path(sys.argv[0], run_name="__main__")
finally:
    S["frac"] = S["differ"] / max(S["elems"], 1)
    S["broken"] = broken
    S["a"] = "PASS" if S["uploads"] and S["dev_worse"] == 0 and S["frac"] <= 1e-4 else "FAIL"
    json.dump(S, open(out, "w"), indent=1)
    print("R2GRADE", json.dumps(S), flush=True)
