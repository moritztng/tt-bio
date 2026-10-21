"""Replay the vendored RFD4 prep over every shipped benchmark input and compare, bit for bit, with a
reference dump (rfd4-io's vendored outputs, themselves 385/385 exact vs upstream prepare_cif, plus 6
identical refusals). Usage: PYTHONPATH=<tree> python io_replay.py <bench> <ref> <out> [seed]"""
import sys
from pathlib import Path

import numpy as np
import torch

from tt_bio._vendor.rfd4_prep.processing import prepare_cif

bench, ref, out = map(Path, sys.argv[1:4]); seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
refs = {p.relative_to(ref).with_suffix(""): p for p in list(ref.rglob("*.npz")) + list(ref.rglob("*.err"))}
srcs = {}
for p in bench.rglob("*"):
    if ".cif" in p.name or p.suffix in (".json", ".yaml"):
        srcs[p.relative_to(bench).parent / p.name.split(".cif")[0].split(".json")[0].split(".yaml")[0]] = p
ok = bad = 0
for key, rp in sorted(refs.items()):
    src = srcs.get(key)
    if src is None:
        print("NOSRC", key); bad += 1; continue
    try:
        feats, _ = prepare_cif(str(src), seed=seed, device=torch.device("cpu"))
        got = ("ok", {k: (v[0].detach().cpu().view(torch.int16) if v.dtype == torch.bfloat16 else v[0].detach().cpu()).numpy()
                      for k, v in feats.items()})
    except Exception as e:
        got = ("error", f"{type(e).__name__}: {e}")
    if rp.suffix == ".err":
        want = ("error", rp.read_text())
        same = got == want
    else:
        with np.load(rp) as z:
            want = {k: z[k] for k in z.files}
        same = got[0] == "ok" and set(got[1]) == set(want) and all(
            got[1][k].shape == want[k].shape and got[1][k].dtype == want[k].dtype
            and np.array_equal(got[1][k], want[k], equal_nan=True) for k in want)
    if same:
        ok += 1
    else:
        bad += 1
        detail = got[1][:200] if got[0] == "error" else [k for k in want if k not in got[1] or not np.array_equal(got[1][k], want[k], equal_nan=True)][:8]
        print("DIFF", key, detail, flush=True)
print(f"exact {ok}/{ok + bad}")
