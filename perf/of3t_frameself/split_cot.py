#!/usr/bin/env python3
"""of3t-frameself step 0, the s-path / z-path half: two cotangent files, each with one side
zeroed.

The trunk gradient is LINEAR in the cotangent, so g(cot_s, cot_z) = g(cot_s, 0) + g(0, cot_z)
exactly. Zeroing one side and re-running the unchanged `perf/of3t_trunkg043/ref_grad.py` splits
the injected gradient into the part that arrives through the single output and the part that
arrives through the pair output, with no fork of the producer and no new argument: only the
--cap-last file changes. Two different best scalars localise the defect to one output; one
scalar localises it to the loss or to a shared normalisation.
"""
import hashlib, json, sys
from pathlib import Path
import torch

src = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)
cap = torch.load(src, map_location="cpu", weights_only=False)
cot_s, cot_z = cap["cot"]
rep = {"source": str(src), "source_keys": sorted(cap.keys()),
       "cot_s_norm": float(torch.linalg.vector_norm(cot_s.double())),
       "cot_z_norm": float(torch.linalg.vector_norm(cot_z.double())), "written": {}}
for name, pair in (("cot_sonly_n384.pt", (cot_s, torch.zeros_like(cot_z))),
                   ("cot_zonly_n384.pt", (torch.zeros_like(cot_s), cot_z))):
    p = out_dir / name
    torch.save({"cot": pair}, p)
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    rep["written"][name] = {"path": str(p), "sha256": h.hexdigest(), "bytes": p.stat().st_size,
                            "cot_s_norm": float(torch.linalg.vector_norm(pair[0].double())),
                            "cot_z_norm": float(torch.linalg.vector_norm(pair[1].double()))}
    print(name, rep["written"][name]["sha256"], rep["written"][name]["bytes"])
(out_dir / "SPLIT_COT.json").write_text(json.dumps(rep, indent=1))
print("wrote " + str(out_dir / "SPLIT_COT.json"))
