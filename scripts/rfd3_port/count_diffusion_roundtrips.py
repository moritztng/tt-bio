"""p23 continuation: precisely count host<->device round-trips in ONE
`RFD3DiffusionModule.__call__` (n_recycle=2, the reference default), by
monkey-patching `ttnn.from_torch`/`ttnn.to_torch` for the duration of a
single warm call. Confirms the host-dispatch-bound diagnosis in
`bench_designs_per_sec.py` quantitatively (93 round-trips/step measured on
the p12 IAI_protein fixture, ~1.66ms average dispatch latency each) rather
than by reading the code and guessing a count.

Usage:
  TT_VISIBLE_DEVICES=0 python3 scripts/rfd3_port/count_diffusion_roundtrips.py
"""
import os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import ttnn
from tt_bio.rfd3 import build_diffusion_module, build_token_initializer
from tt_bio.rfd3_featurize import featurize
from tt_bio.rfd3_input import InputSpecification

PDB = os.path.join(os.path.dirname(__file__), "parity_artifacts", "iai_protein", "IAI_protein.pdb")
CONTIG = "A1-10,20,A31-40"
GOLDEN_DIR = os.path.expanduser("~/.coworker/artifacts/rfd3-goldens/capture")


def main():
    spec = InputSpecification.from_dict({"input": PDB, "contig": CONTIG})
    spec.validate()
    f = featurize(PDB, spec)
    f = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in f.items()}

    ti_weights = torch.load(os.path.join(GOLDEN_DIR, "token_initializer.real_weights.pt"),
                             map_location="cpu", weights_only=True)
    dm_weights = torch.load(os.path.join(GOLDEN_DIR, "diffusion_module.real_weights.pt"),
                             map_location="cpu", weights_only=True)
    dev_ti = build_token_initializer(ti_weights)
    dev_dm = build_diffusion_module(dm_weights)
    coord0 = f["motif_pos"].float().unsqueeze(0)
    with torch.no_grad():
        init = dev_ti({k: (v.clone() if torch.is_tensor(v) else v) for k, v in f.items()})

    counts = {"from_torch": 0, "to_torch": 0}
    orig_from, orig_to = ttnn.from_torch, ttnn.to_torch

    def counted_from_torch(*a, **kw):
        counts["from_torch"] += 1
        return orig_from(*a, **kw)

    def counted_to_torch(*a, **kw):
        counts["to_torch"] += 1
        return orig_to(*a, **kw)

    ttnn.from_torch, ttnn.to_torch = counted_from_torch, counted_to_torch

    X_noisy = coord0.clone()
    t = torch.tensor([100.0])
    with torch.no_grad():
        dev_dm(X_noisy_L=X_noisy, t=t, f=f, **init)  # warm up (compiles kernels)
    counts["from_torch"] = counts["to_torch"] = 0
    with torch.no_grad():
        dev_dm(X_noisy_L=X_noisy, t=t, f=f, **init)
    total = counts["from_torch"] + counts["to_torch"]
    print(f"[roundtrips] ONE __call__ (n_recycle=2 default): "
          f"from_torch(host->device)={counts['from_torch']} "
          f"to_torch(device->host)={counts['to_torch']} total={total}")


if __name__ == "__main__":
    main()
